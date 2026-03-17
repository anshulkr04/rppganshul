"""PhysMamba Trainer (Stable + Complete + Improved Loss)"""

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
    return torch.log1p(torch.abs(X))


def temporal_diff_loss(x, y):
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    return F.l1_loss(dx, dy)


def bandpass_loss(x, sr=30):
    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1/sr).to(x.device)

    mask = (freqs >= 0.7) & (freqs <= 4.0)
    power = torch.abs(X)**2

    return -power[:, mask].mean()


def cosine_interp(a0, a1, alpha):
    mu = 0.5 * (1 - math.cos(math.pi * alpha))
    return (1 - mu) * a0 + mu * a1


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

class PhysMambaTrainer(BaseTrainer):

    def __init__(self, config, data_loader):

        super().__init__()

        # 🔥 IMPORTANT: restore ALL attributes
        self.device = torch.device(config.DEVICE)
        self.config = config

        self.max_epoch_num = config.TRAIN.EPOCHS
        self.frame_rate = config.TRAIN.DATA.FS

        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME

        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN

        self.min_valid_loss = None   # ✅ FIXED
        self.best_epoch = 0          # ✅ FIXED

        # Model
        self.model = PhysMamba(
            frames=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        ).to(self.device)

        if self.num_of_gpu > 0:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=list(range(self.num_of_gpu))
            )

        # Loss
        self.criterion_Pearson = Neg_Pearson()

        # Optimizer
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

        # curriculum params
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

        w_cwt = min(1 - w_fft, 0.6)

        return w_fft, w_cwt


    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    def train(self, data_loader):

        n_fft = 256
        sr = int(self.frame_rate)

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

                # match length
                if pred.shape[-1] != labels.shape[-1]:
                    pred = F.interpolate(
                        pred.unsqueeze(1),
                        size=labels.shape[-1],
                        mode='linear',
                        align_corners=False
                    ).squeeze(1)

                # normalize (per sample)
                pred = (pred - pred.mean(dim=-1, keepdim=True)) / (pred.std(dim=-1, keepdim=True) + 1e-8)
                labels = (labels - labels.mean(dim=-1, keepdim=True)) / (labels.std(dim=-1, keepdim=True) + 1e-8)

                # remove DC
                pred = pred - pred.mean(dim=-1, keepdim=True)
                labels = labels - labels.mean(dim=-1, keepdim=True)

                # LOSSES
                Lp = self.criterion_Pearson(pred, labels)
                Lt = temporal_diff_loss(pred, labels)
                Lb = bandpass_loss(pred, sr)

                Lf = F.mse_loss(
                    spectral_log_magnitude(pred, n_fft),
                    spectral_log_magnitude(labels, n_fft)
                )

                # FINAL LOSS (stable)
                loss = (
                    0.6 * Lp +
                    0.2 * w_fft * Lf +
                    0.15 * Lt +
                    0.05 * Lb
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

            # SAVE
            self.save_model(epoch)

            # VALIDATE
            valid_loss = self.valid(data_loader)

            print("validation loss:", valid_loss)

            if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                self.min_valid_loss = valid_loss
                self.best_epoch = epoch
                print("Update best model! Best epoch:", epoch)

            torch.cuda.empty_cache()

        print("Training finished.")


    # ------------------------------------------------------------
    # VALID
    # ------------------------------------------------------------

    def valid(self, data_loader):

        self.model.eval()
        valid_loss = []

        with torch.no_grad():

            for batch in tqdm(data_loader["valid"], ncols=80):

                data = batch[0].to(self.device)
                label = batch[1].to(self.device)

                pred = self.model(data)

                if pred.shape[-1] != label.shape[-1]:
                    pred = F.interpolate(
                        pred.unsqueeze(1),
                        size=label.shape[-1],
                        mode='linear',
                        align_corners=False
                    ).squeeze(1)

                pred = (pred - pred.mean(dim=-1, keepdim=True)) / (pred.std(dim=-1, keepdim=True)+1e-8)
                label = (label - label.mean(dim=-1, keepdim=True)) / (label.std(dim=-1, keepdim=True)+1e-8)

                loss = self.criterion_Pearson(pred, label)

                valid_loss.append(loss.item())

        return np.mean(valid_loss)


    # ------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    def save_model(self, epoch):

        os.makedirs(self.model_dir, exist_ok=True)

        path = os.path.join(
            self.model_dir,
            self.model_file_name + '_Epoch' + str(epoch) + '.pth'
        )

        torch.save(self.model.state_dict(), path)

        print("Saved Model Path:", path)


    # ------------------------------------------------------------
    # HR
    # ------------------------------------------------------------

    def get_hr(self, y, sr=30, min=30, max=180):

        p, q = welch(y, sr, nfft=1e5/sr, nperseg=min(len(y)-1, 256))

        return p[(p>min/60)&(p<max/60)][
            np.argmax(q[(p>min/60)&(p<max/60)])
        ] * 60