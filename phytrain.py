"""FINAL PhysMamba Trainer (Temporal-Mamba + FFT + CWT + Pearson)"""

import os
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from scipy.signal import welch

from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysMamba import PhysMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer


# ------------------------------------------------------------
# FFT
# ------------------------------------------------------------

def spectral_log_magnitude(x, n_fft=256):
    X = torch.fft.rfft(x, n=n_fft)
    return torch.log1p(torch.abs(X))


# ------------------------------------------------------------
# TEMPORAL LOSS
# ------------------------------------------------------------

def temporal_diff_loss(x, y):
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    return F.l1_loss(dx, dy)


# ------------------------------------------------------------
# CWT (Morlet)
# ------------------------------------------------------------

def morlet_wavelet(freq, sr, width=6.0):

    cycles = 4.0
    duration = max(0.5, cycles / float(freq))

    total_samples = int(max(3, round(duration * sr)))
    if total_samples % 2 == 0:
        total_samples += 1

    t = (torch.arange(total_samples) - total_samples // 2) / sr

    sigma = width / (2 * math.pi * freq)
    gauss = torch.exp(-t**2 / (2 * sigma**2))

    real = gauss * torch.cos(2 * math.pi * freq * t)
    imag = gauss * torch.sin(2 * math.pi * freq * t)

    real /= (real.abs().sum() + 1e-12)
    imag /= (imag.abs().sum() + 1e-12)

    return real, imag


def make_morlet_bank(freqs, sr, device):

    temp = []
    max_len = 0

    for f in freqs:
        r, i = morlet_wavelet(f, sr)
        temp.append((r, i))
        max_len = max(max_len, r.numel())

    if max_len % 2 == 0:
        max_len += 1

    kernels_r, kernels_i = [], []

    for r, i in temp:

        pad = max_len - r.numel()
        left = pad // 2
        right = pad - left

        r = F.pad(r, (left, right))
        i = F.pad(i, (left, right))

        kernels_r.append(r.view(1, 1, -1))
        kernels_i.append(i.view(1, 1, -1))

    kernels_r = torch.cat(kernels_r, dim=0).to(device)
    kernels_i = torch.cat(kernels_i, dim=0).to(device)

    return kernels_r, kernels_i


def cwt_magnitude_conv1d(x, kernels_r, kernels_i):

    x = x.unsqueeze(1)

    k = kernels_r.shape[-1]
    pad = k // 2

    x = F.pad(x, (pad, pad), mode='reflect')

    real = F.conv1d(x, kernels_r)
    imag = F.conv1d(x, kernels_i)

    return torch.sqrt(real**2 + imag**2 + 1e-12)


# ------------------------------------------------------------
# COSINE SCHEDULER
# ------------------------------------------------------------

def cosine_interp(a0, a1, alpha):
    mu = 0.5 * (1 - math.cos(math.pi * alpha))
    return (1 - mu) * a0 + mu * a1


# ------------------------------------------------------------
# TRAINER
# ------------------------------------------------------------

class PhysMambaTrainer(BaseTrainer):

    def __init__(self, config, data_loader):

        super().__init__()

        self.device = torch.device(config.DEVICE)
        self.config = config

        self.max_epoch_num = config.TRAIN.EPOCHS
        self.frame_rate = config.TRAIN.DATA.FS

        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME

        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN

        self.min_valid_loss = None
        self.best_epoch = 0

        self.model = PhysMamba(
            frames=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        ).to(self.device)

        if self.num_of_gpu > 0:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=list(range(self.num_of_gpu))
            )

        self.criterion_Pearson = Neg_Pearson()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.TRAIN.LR,
            weight_decay=5e-4
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.TRAIN.LR,
            epochs=config.TRAIN.EPOCHS,
            steps_per_epoch=len(data_loader["train"])
        )

        self.frac_fft_major = 0.4
        self.transition_frac = 0.1


    def _compute_weights(self, epoch):

        E = self.max_epoch_num
        E_major = int(self.frac_fft_major * E)
        E_transition = max(1, int(self.transition_frac * E))

        if epoch < E_major:
            w_fft = 0.85
        elif epoch >= E_major + E_transition:
            w_fft = 0.05
        else:
            alpha = (epoch - E_major) / E_transition
            w_fft = cosine_interp(0.85, 0.05, alpha)

        w_cwt = 1 - w_fft
        return w_fft, w_cwt


    def train(self, data_loader):

        sr = int(self.frame_rate)
        freqs = torch.linspace(0.5, 4.0, 32).tolist()

        kernels_real, kernels_imag = make_morlet_bank(freqs, sr, self.device)

        for epoch in range(self.max_epoch_num):

            print(f"\n==== Training Epoch {epoch} ====")

            w_fft, w_cwt = self._compute_weights(epoch)
            print(f"Spectral weights → FFT:{w_fft:.3f} CWT:{w_cwt:.3f}")

            self.model.train()

            for batch in tqdm(data_loader["train"], ncols=80):

                data, labels = batch[0].float(), batch[1].float()

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                pred = self.model(data)

                if pred.shape[-1] != labels.shape[-1]:
                    pred = F.interpolate(
                        pred.unsqueeze(1),
                        size=labels.shape[-1],
                        mode='linear',
                        align_corners=False
                    ).squeeze(1)

                # Normalize
                pred = (pred - pred.mean(dim=-1, keepdim=True)) / (pred.std(dim=-1, keepdim=True) + 1e-8)
                labels = (labels - labels.mean(dim=-1, keepdim=True)) / (labels.std(dim=-1, keepdim=True) + 1e-8)

                # LOSSES
                Lp = self.criterion_Pearson(pred, labels)
                Lt = temporal_diff_loss(pred, labels)

                Lf = F.mse_loss(
                    spectral_log_magnitude(pred),
                    spectral_log_magnitude(labels)
                )

                cwt_pred = cwt_magnitude_conv1d(pred, kernels_real, kernels_imag)
                cwt_gt = cwt_magnitude_conv1d(labels, kernels_real, kernels_imag)

                Lc = F.mse_loss(
                    torch.log1p(cwt_pred),
                    torch.log1p(cwt_gt)
                )

                loss = (
                    0.5 * Lp +
                    0.2 * Lt +
                    0.2 * w_fft * Lf +
                    0.1 * w_cwt * Lc
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

            self.save_model(epoch)

            valid_loss = self.valid(data_loader)

            print("validation loss:", valid_loss)

            if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                self.min_valid_loss = valid_loss
                self.best_epoch = epoch
                print("Update best model! Best epoch:", epoch)

            torch.cuda.empty_cache()

        print("Training finished.")


    def valid(self, data_loader):

        self.model.eval()
        losses = []

        with torch.no_grad():

            for batch in tqdm(data_loader["valid"], ncols=80):

                data = batch[0].to(self.device)
                label = batch[1].to(self.device)

                pred = self.model(data)

                if pred.shape[-1] != label.shape[-1]:
                    pred = F.interpolate(
                        pred.unsqueeze(1),
                        size=label.shape[-1],
                        mode='linear'
                    ).squeeze(1)

                pred = (pred - pred.mean(dim=-1, keepdim=True)) / (pred.std(dim=-1, keepdim=True)+1e-8)
                label = (label - label.mean(dim=-1, keepdim=True)) / (label.std(dim=-1, keepdim=True)+1e-8)

                loss = self.criterion_Pearson(pred, label)
                losses.append(loss.item())

        return np.mean(losses)


    def test(self, data_loader):

        print("\n===Testing===")

        predictions, labels = {}, {}

        model_path = os.path.join(
            self.model_dir,
            self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth'
        )

        print("Loading model:", model_path)

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        with torch.no_grad():

            for batch in tqdm(data_loader["test"], ncols=80):

                data = batch[0].to(self.device)
                label = batch[1].to(self.device)

                pred = self.model(data)

                for i in range(data.shape[0]):

                    subj = batch[2][i]
                    idx = int(batch[3][i])

                    if subj not in predictions:
                        predictions[subj] = {}
                        labels[subj] = {}

                    predictions[subj][idx] = pred[i].cpu()
                    labels[subj][idx] = label[i].cpu()

        calculate_metrics(predictions, labels, self.config)


    def save_model(self, epoch):

        os.makedirs(self.model_dir, exist_ok=True)

        path = os.path.join(
            self.model_dir,
            self.model_file_name + '_Epoch' + str(epoch) + '.pth'
        )

        torch.save(self.model.state_dict(), path)
        print("Saved Model Path:", path)


    def get_hr(self, y, sr=30, min=30, max=180):

        p, q = welch(y, sr, nfft=1e5/sr, nperseg=min(len(y)-1, 256))

        return p[(p>min/60)&(p<max/60)][
            np.argmax(q[(p>min/60)&(p<max/60)])
        ] * 60