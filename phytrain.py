"""PhysMamba Trainer (FFT + CWT + Pearson curriculum)."""

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
            weight_decay=0.0005
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.TRAIN.LR,
            epochs=config.TRAIN.EPOCHS,
            steps_per_epoch=len(data_loader["train"])
        )

        # curriculum
        self.frac_fft_major = 0.4
        self.transition_frac = 0.10
        self.w_fft_high = 0.85
        self.w_fft_low = 0.05


    def _compute_weights(self, epoch):

        E = self.max_epoch_num

        E_major = int(math.floor(self.frac_fft_major * E))
        E_transition = max(1, int(round(self.transition_frac * E)))

        t_start = E_major
        t_end = E_major + E_transition

        if epoch < t_start:
            w_fft = self.w_fft_high

        elif epoch >= t_end:
            w_fft = self.w_fft_low

        else:
            alpha = float(epoch - t_start) / float(max(1, t_end - t_start))
            w_fft = cosine_interp(self.w_fft_high, self.w_fft_low, alpha)

        w_cwt = 1.0 - w_fft

        return float(w_fft), float(w_cwt)


    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    def train(self, data_loader):

        sr = int(self.frame_rate)

        freqs = torch.linspace(0.5, 4.0, 32).tolist()

        kernels_real, kernels_imag = make_morlet_bank(freqs, sr, self.device)

        n_fft = 256

        L0_fft = None
        L0_cwt = None
        eps = 1e-8

        for epoch in range(self.max_epoch_num):

            print(f"\n==== Training Epoch {epoch} ====")

            w_fft, w_cwt = self._compute_weights(epoch)

            print(f"Spectral weights → FFT:{w_fft:.3f} CWT:{w_cwt:.3f}")

            self.model.train()

            tbar = tqdm(data_loader["train"], ncols=80)

            for idx, batch in enumerate(tbar):

                data, labels = batch[0].float(), batch[1].float()

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                pred_ppg = self.model(data)

                if pred_ppg.shape[-1] != labels.shape[-1]:

                    pred_ppg = F.interpolate(
                        pred_ppg.unsqueeze(1),
                        size=labels.shape[-1],
                        mode="linear",
                        align_corners=False
                    ).squeeze(1)

                # normalize prediction
                pred_ppg = (
                    pred_ppg - pred_ppg.mean(dim=-1, keepdim=True)
                ) / (pred_ppg.std(dim=-1, keepdim=True) + 1e-8)

                # normalize labels (FIXED)
                labels = (
                    labels - labels.mean(dim=-1, keepdim=True)
                ) / (labels.std(dim=-1, keepdim=True) + 1e-8)

                # remove DC bias
                pred_ppg = pred_ppg - pred_ppg.mean(dim=-1, keepdim=True)
                labels = labels - labels.mean(dim=-1, keepdim=True)

                # Pearson loss
                L_pearson = self.criterion_Pearson(pred_ppg, labels)

                # FFT loss
                fft_pred = spectral_log_magnitude(pred_ppg, n_fft)
                fft_gt = spectral_log_magnitude(labels, n_fft)
                L_fft = F.mse_loss(fft_pred, fft_gt)

                # CWT loss
                cwt_pred = cwt_magnitude_conv1d(pred_ppg, kernels_real, kernels_imag)
                cwt_gt = cwt_magnitude_conv1d(labels, kernels_real, kernels_imag)

                L_cwt = F.mse_loss(
                    torch.log1p(cwt_pred),
                    torch.log1p(cwt_gt)
                )

                if L0_fft is None:

                    L0_fft = max(L_fft.detach().item(), 1e-6)
                    L0_cwt = max(L_cwt.detach().item(), 1e-6)

                L_fft_norm = L_fft / (L0_fft + eps)
                L_cwt_norm = L_cwt / (L0_cwt + eps)

                loss = (
                    0.5 * L_pearson
                    + 0.3 * w_fft * L_fft_norm
                    + 0.2 * w_cwt * L_cwt_norm
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                tbar.set_postfix(
                    loss=loss.item(),
                    pearson=L_pearson.item(),
                    w_fft=w_fft,
                    w_cwt=w_cwt
                )

            self.save_model(epoch)

            valid_loss = self.valid(data_loader)

            print("validation loss:", valid_loss)

            if self.min_valid_loss is None or valid_loss < self.min_valid_loss:

                self.min_valid_loss = valid_loss
                self.best_epoch = epoch

                print("Update best model! Best epoch:", self.best_epoch)

            torch.cuda.empty_cache()

        print("Training finished.")


    # ------------------------------------------------------------
    # Validation
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
                label = (label - label.mean(dim=-1, keepdim=True)) / (label.std(dim=-1, keepdim=True) + 1e-8)

                loss = self.criterion_Pearson(pred, label)

                valid_loss.append(loss.item())

        return np.mean(valid_loss)


    # ------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------

    def test(self, data_loader):

        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("\n===Testing===")

        predictions = dict()
        labels = dict()

        if self.config.TEST.USE_LAST_EPOCH:

            model_path = os.path.join(
                self.model_dir,
                self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth'
            )

        else:

            model_path = os.path.join(
                self.model_dir,
                self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth'
            )

        print("Loading model:", model_path)

        self.model.load_state_dict(torch.load(model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()

        with torch.no_grad():

            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):

                batch_size = test_batch[0].shape[0]

                data = test_batch[0].to(self.config.DEVICE)
                label = test_batch[1].to(self.config.DEVICE)

                pred_ppg_test = self.model(data)

                for idx in range(batch_size):

                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])

                    if subj_index not in predictions:
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()

                    predictions[subj_index][sort_index] = pred_ppg_test[idx].detach().cpu()
                    labels[subj_index][sort_index] = label[idx].detach().cpu()

        calculate_metrics(predictions, labels, self.config)


    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------

    def save_model(self, index):

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        model_path = os.path.join(
            self.model_dir,
            self.model_file_name + '_Epoch' + str(index) + '.pth'
        )

        torch.save(self.model.state_dict(), model_path)

        print("Saved Model Path:", model_path)


    def get_hr(self, y, sr=30, min=30, max=180):

        p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))

        return p[(p > min/60) & (p < max/60)][
            np.argmax(q[(p > min/60) & (p < max/60)])
        ] * 60