"""PhysMamba Trainer (FFT + CWT spectral curriculum only)."""

import os
import math
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from scipy.signal import welch

from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysMamba import PhysMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def spectral_log_magnitude(x, n_fft=256):
    X = torch.fft.rfft(x, n=n_fft)
    mag = torch.abs(X)
    return torch.log1p(mag)


def morlet_wavelet(freq, sr, width=6.0, duration=None):

    if duration is None:
        cycles = 4.0
        duration = max(0.5, cycles / float(freq))

    total_samples = int(max(3, round(duration * sr)))

    if total_samples % 2 == 0:
        total_samples += 1

    t = (torch.arange(total_samples) - (total_samples // 2)) / float(sr)

    sigma = width / (2 * math.pi * freq)

    gauss = torch.exp(-t**2 / (2 * sigma**2))

    real = gauss * torch.cos(2 * math.pi * freq * t)
    imag = gauss * torch.sin(2 * math.pi * freq * t)

    real = real / (real.abs().sum() + 1e-12)
    imag = imag / (imag.abs().sum() + 1e-12)

    return real, imag


def make_morlet_bank(freqs_hz, sr, device):

    tmp = []
    max_len = 0

    for f in freqs_hz:
        r, im = morlet_wavelet(f, sr)
        tmp.append((r, im))
        max_len = max(max_len, r.numel())

    if max_len % 2 == 0:
        max_len += 1

    kernels_r = []
    kernels_i = []

    for r, im in tmp:

        pad = max_len - r.numel()
        left = pad // 2
        right = pad - left

        rpad = F.pad(r, (left, right))
        impad = F.pad(im, (left, right))

        kernels_r.append(rpad.view(1, 1, -1))
        kernels_i.append(impad.view(1, 1, -1))

    kernels_real = torch.cat(kernels_r, dim=0).to(device)
    kernels_imag = torch.cat(kernels_i, dim=0).to(device)

    return kernels_real, kernels_imag


def cwt_magnitude_conv1d(x, kernels_real, kernels_imag):

    x = x.unsqueeze(1)

    kernel_size = kernels_real.shape[-1]
    pad = kernel_size // 2

    x_padded = F.pad(x, (pad, pad), mode='reflect')

    real_out = F.conv1d(x_padded, kernels_real)
    imag_out = F.conv1d(x_padded, kernels_imag)

    mag = torch.sqrt(real_out**2 + imag_out**2 + 1e-12)

    return mag


def cosine_interp(a0, a1, alpha):

    mu = 0.5 * (1 - math.cos(math.pi * alpha))
    return (1 - mu) * a0 + mu * a1


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

class PhysMambaTrainer(BaseTrainer):

    def __init__(self, config, data_loader):

        super().__init__()

        self.device = torch.device(config.DEVICE)
        self.config = config

        self.max_epoch_num = config.TRAIN.EPOCHS
        self.frame_rate = config.TRAIN.DATA.FS

        self.model = PhysMamba(
            frames=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        ).to(self.device)

        if config.NUM_OF_GPU_TRAIN > 0:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=list(range(config.NUM_OF_GPU_TRAIN))
            )

        self.criterion_Pearson = Neg_Pearson()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.TRAIN.LR,
            weight_decay=0.0005
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.TRAIN.LR,
            epochs=config.TRAIN.EPOCHS,
            steps_per_epoch=len(data_loader["train"])
        )

    # ------------------------------------------------------------

    def train(self, data_loader):

        sr = int(self.frame_rate)

        freqs = torch.linspace(0.5, 4.0, 32).tolist()

        kernels_real, kernels_imag = make_morlet_bank(
            freqs, sr, self.device
        )

        n_fft = 256

        L0_fft = None
        L0_cwt = None

        eps = 1e-8

        for epoch in range(self.max_epoch_num):

            print(f"\n==== Training Epoch {epoch} ====")

            self.model.train()

            tbar = tqdm(data_loader["train"], ncols=80)

            for idx, batch in enumerate(tbar):

                data, labels = batch[0].float(), batch[1].float()

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                pred_ppg = self.model(data)

                # -----------------------------------------
                # Ensure temporal lengths match
                # -----------------------------------------

                if pred_ppg.shape[-1] != labels.shape[-1]:

                    pred_ppg = F.interpolate(
                        pred_ppg.unsqueeze(1),
                        size=labels.shape[-1],
                        mode="linear",
                        align_corners=False
                    ).squeeze(1)

                # -----------------------------------------
                # Normalize predictions
                # -----------------------------------------

                pred_mean = torch.mean(pred_ppg, dim=-1).view(-1, 1)
                pred_std = torch.std(pred_ppg, dim=-1).view(-1, 1) + 1e-8

                pred_ppg = (pred_ppg - pred_mean) / pred_std

                labels = (labels - torch.mean(labels)) / (torch.std(labels) + 1e-8)

                # -----------------------------------------
                # FFT loss
                # -----------------------------------------

                fft_pred = spectral_log_magnitude(pred_ppg, n_fft)
                fft_gt = spectral_log_magnitude(labels, n_fft)

                L_fft = F.mse_loss(fft_pred, fft_gt)

                # -----------------------------------------
                # CWT loss
                # -----------------------------------------

                cwt_pred = cwt_magnitude_conv1d(pred_ppg, kernels_real, kernels_imag)
                cwt_gt = cwt_magnitude_conv1d(labels, kernels_real, kernels_imag)

                L_cwt = F.mse_loss(torch.log1p(cwt_pred), torch.log1p(cwt_gt))

                # -----------------------------------------

                if L0_fft is None:
                    L0_fft = max(L_fft.detach().item(), 1e-6)
                    L0_cwt = max(L_cwt.detach().item(), 1e-6)

                L_fft_norm = L_fft / (L0_fft + eps)
                L_cwt_norm = L_cwt / (L0_cwt + eps)

                loss = 0.85 * L_fft_norm + 0.15 * L_cwt_norm

                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                tbar.set_postfix(loss=loss.item())

        print("Training finished.")

    # ------------------------------------------------------------

    def valid(self, data_loader):

        valid_loss = []

        self.model.eval()

        with torch.no_grad():

            for batch in tqdm(data_loader["valid"], ncols=80):

                data = batch[0].to(self.device)
                label = batch[1].to(self.device)

                pred = self.model(data)

                if pred.shape[-1] != label.shape[-1]:

                    pred = F.interpolate(
                        pred.unsqueeze(1),
                        size=label.shape[-1],
                        mode="linear",
                        align_corners=False
                    ).squeeze(1)

                pred = (pred - pred.mean(dim=-1, keepdim=True)) / (pred.std(dim=-1, keepdim=True) + 1e-8)

                label = (label - label.mean()) / (label.std() + 1e-8)

                loss = self.criterion_Pearson(pred, label)

                valid_loss.append(loss.item())

        return np.mean(valid_loss)
