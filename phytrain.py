"""PhysMamba Trainer (FFT + CWT only training with FFT-major 20% schedule)."""

import os
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.optim as optim
import random
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysMamba import PhysMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
from scipy.signal import welch
import torch.nn.functional as F

# ---------------- Helper functions ----------------

def spectral_log_magnitude(x, n_fft=256):
    """
    x: [B, T]
    returns: log1p(|rfft(x)|) shape [B, n_fft//2 + 1]
    """
    X = torch.fft.rfft(x, n=n_fft)
    mag = torch.abs(X)
    return torch.log1p(mag)

def morlet_wavelet(freq, sr, width=6.0, duration=None):
    """
    Build a Morlet wavelet for center frequency `freq` (Hz).
    Returns real, imag 1D tensors (cpu float32).
    """
    if duration is None:
        cycles = 4.0
        duration = max(0.5, cycles / float(freq))
    total_samples = int(max(3, round(duration * sr)))
    if total_samples % 2 == 0:
        total_samples += 1
    t = (torch.arange(total_samples, dtype=torch.float32) - (total_samples // 2)) / float(sr)
    sigma = width / (2 * math.pi * freq)
    gauss = torch.exp(-t**2 / (2 * sigma**2))
    real = gauss * torch.cos(2 * math.pi * freq * t)
    imag = gauss * torch.sin(2 * math.pi * freq * t)
    # L1 normalize for stability in conv outputs
    real = real / (real.abs().sum() + 1e-12)
    imag = imag / (imag.abs().sum() + 1e-12)
    return real, imag

def make_morlet_bank(freqs_hz, sr, device):
    """
    Build Morlet conv kernels for each frequency in freqs_hz.
    Returns kernels_real, kernels_imag on `device` with shape [n_scales, 1, K].
    """
    tmp = []
    max_len = 0
    for f in freqs_hz:
        r, im = morlet_wavelet(f, sr)
        tmp.append((r, im))
        max_len = max(max_len, r.numel())
    # ensure odd common length
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
    """
    Compute CWT magnitude by convolving with Morlet kernels.
    x: [B, T] -> returns [B, n_scales, T]
    """
    x = x.unsqueeze(1)  # [B, 1, T]
    kernel_size = kernels_real.shape[-1]
    pad = kernel_size // 2
    # reflect padding to avoid step artifacts
    x_padded = F.pad(x, (pad, pad), mode='reflect')
    real_out = F.conv1d(x_padded, kernels_real, padding=0)  # [B, n_scales, T]
    imag_out = F.conv1d(x_padded, kernels_imag, padding=0)
    mag = torch.sqrt(real_out**2 + imag_out**2 + 1e-12)
    return mag

def cosine_interp(a0, a1, alpha):
    """
    Cosine interpolation between a0 (alpha=0) and a1 (alpha=1).
    alpha in [0,1]
    """
    mu = 0.5 * (1 - math.cos(math.pi * alpha))
    return (1 - mu) * a0 + mu * a1

# ---------------- PhysMambaTrainer ----------------

class PhysMambaTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Initialize trainer — preserves original fields and scheduler usage."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        self.frame_rate = config.TRAIN.DATA.FS

        # model (keep same shape/placement as original)
        self.model = PhysMamba().to(self.device)
        if self.num_of_gpu > 0:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

        # optimizer & scheduler (same as original)
        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            # keep Pearson criterion instance for validation (original used Neg_Pearson)
            self.criterion_Pearson = Neg_Pearson()
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0.0005)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.criterion_Pearson_test = Neg_Pearson()
            pass
        else:
            raise ValueError("PhysMamba trainer initialized in incorrect toolbox mode!")

    def _schedule_weights_fft_cwt(self, epoch, total_epochs,
                                  frac_fft_major=0.2,
                                  transition_frac=0.10,
                                  w_fft_high=0.85,
                                  w_fft_low=0.05):
        """
        Compute (w_fft, w_cwt) for given epoch.

        Math:
        - Let E = total_epochs.
        - Let E_major = floor(frac_fft_major * E)  (epochs where FFT must be majority).
          This guarantees FFT is majority for exactly E_major epochs (0-indexed: epochs 0 .. E_major-1).
        - Let E_transition = max(1, round(transition_frac * E)) (smooth transition window length).
        - Transition starts at t_start = E_major and ends at t_end = E_major + E_transition.
        - For epoch < t_start: w_fft = w_fft_high (FFT-major).
        - For epoch >= t_end: w_fft = w_fft_low (CWT-major).
        - For t_start <= epoch < t_end: interpolate smoothly with cosine between w_fft_high -> w_fft_low.
        - w_cwt = 1 - w_fft (we use only spectral losses).
        """
        E = int(total_epochs)
        E_major = int(math.floor(frac_fft_major * E))
        E_transition = max(1, int(round(transition_frac * E)))
        t_start = E_major
        t_end = E_major + E_transition

        if epoch < t_start:
            w_fft = float(w_fft_high)
        elif epoch >= t_end:
            w_fft = float(w_fft_low)
        else:
            alpha = float(epoch - t_start) / float(max(1, t_end - t_start))
            w_fft = cosine_interp(w_fft_high, w_fft_low, alpha)
        w_cwt = 1.0 - w_fft
        return w_fft, w_cwt

    def train(self, data_loader):
        """Training routine using only FFT + CWT losses with the 20% FFT-major schedule."""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        # CWT bank: choose scales that map to approx 0.5-4.0 Hz (HR band)
        sr = int(self.frame_rate)
        n_scales = getattr(self.config, 'CWT', {}).get('N_SCALES', 32) if getattr(self.config, 'CWT', None) else 32
        freqs = torch.linspace(0.5, 4.0, n_scales).tolist()
        kernels_real, kernels_imag = make_morlet_bank(freqs, sr, self.device)

        # FFT param
        n_fft = getattr(self.config, 'FFT', {}).get('N_FFT', 256) if getattr(self.config, 'FFT', None) else 256

        # normalization baselines (computed from first batch)
        L0_fft = None
        L0_cwt = None
        eps = 1e-8

        total_epochs = self.max_epoch_num
        pre_transition_saved = False

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            self.model.train()
            running_loss = 0.0
            tbar = tqdm(data_loader["train"], ncols=80)

            # compute schedule weights; defaults:
            # - frac_fft_major = 0.2 (FFT majority for first 20% epochs)
            # - transition_frac = 0.10 (10% of epochs used for smooth transition)
            # - w_fft_high = 0.85 (FFT weight during majority)
            # - w_fft_low = 0.05  (FFT weight after transition)
            w_fft, w_cwt = self._schedule_weights_fft_cwt(
                epoch, total_epochs,
                frac_fft_major=0.2,
                transition_frac=0.10,
                w_fft_high=0.85,
                w_fft_low=0.05
            )

            # compute t_start to reduce LR and checkpoint at transition start (optional stabilizer)
            E = total_epochs
            E_major = int(math.floor(0.2 * E))
            E_transition = max(1, int(round(0.10 * E)))
            t_start = E_major
            if epoch == t_start and epoch > 0:
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.33
                if not pre_transition_saved:
                    self.save_model(epoch)
                    pre_transition_saved = True

            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].float(), batch[1].float()
                N, D, C, H, W = data.shape  # preserve original shape handling

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)  # model expected to return waveform in same shape as before

                # preserve original normalization style exactly for compatibility:
                # original: pred_ppg = (pred_ppg - torch.mean(pred_ppg, axis=-1).view(-1,1)) / torch.std(pred_ppg, axis=-1).view(-1,1)
                # That treated each sample across time (axis=-1) per-row; replicate with keepdims
                # handle different tensor shapes just in case
                if pred_ppg.dim() == 3 and pred_ppg.shape[1] == 1:
                    pred_ppg = pred_ppg.squeeze(1)
                if pred_ppg.dim() == 3 and pred_ppg.shape[-1] == 1:
                    pred_ppg = pred_ppg.squeeze(-1)

                # compute per-sample mean/std along last axis (time)
                pred_mean = torch.mean(pred_ppg, dim=-1).view(-1, 1)
                pred_std = torch.std(pred_ppg, dim=-1).view(-1, 1) + 1e-8
                pred_ppg = (pred_ppg - pred_mean) / pred_std

                # Original labels normalization in your code was global (no axis). To preserve behavior:
                labels = (labels - torch.mean(labels)) / (torch.std(labels) + 1e-8)

                # --- FFT loss (log-magnitude MSE) ---
                fft_pred_log = spectral_log_magnitude(pred_ppg, n_fft=n_fft)
                fft_target_log = spectral_log_magnitude(labels, n_fft=n_fft)
                L_fft = F.mse_loss(fft_pred_log, fft_target_log, reduction='mean')

                # --- CWT loss (log1p magnitude MSE) ---
                cwt_pred = cwt_magnitude_conv1d(pred_ppg, kernels_real, kernels_imag)
                cwt_target = cwt_magnitude_conv1d(labels, kernels_real, kernels_imag)
                L_cwt = F.mse_loss(torch.log1p(cwt_pred), torch.log1p(cwt_target), reduction='mean')

                # init normalization baselines from first seen batch
                if L0_fft is None:
                    L0_fft = max(L_fft.detach().item(), 1e-6)
                    L0_cwt = max(L_cwt.detach().item(), 1e-6)
                    print(f"Init baselines: L0_fft={L0_fft:.6e}, L0_cwt={L0_cwt:.6e}")

                L_fft_norm = L_fft / (L0_fft + eps)
                L_cwt_norm = L_cwt / (L0_cwt + eps)

                # combined loss — only FFT + CWT (weights computed above)
                loss = float(w_fft) * L_fft_norm + float(w_cwt) * L_cwt_norm

                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

                self.optimizer.step()
                # step scheduler (OneCycleLR expects step per batch)
                try:
                    self.scheduler.step()
                except Exception:
                    pass

                tbar.set_postfix(loss=loss.item(), w_fft=w_fft, w_cwt=w_cwt)

            # end of epoch
            self.save_model(epoch)

            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
            torch.cuda.empty_cache()

        # training finished
        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))

    def valid(self, data_loader):
        """ Runs the model on valid sets using the same Pearson criterion as original."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(
                    torch.float32).to(self.device)
                rPPG = self.model(
                    valid_batch[0].to(torch.float32).to(self.device))
                # reshape if model output has extra dims
                if rPPG.dim() == 3 and rPPG.shape[1] == 1:
                    rPPG = rPPG.squeeze(1)
                if rPPG.dim() == 3 and rPPG.shape[-1] == 1:
                    rPPG = rPPG.squeeze(-1)

                # preserve original normalization style for validation as well (global label norm in original)
                rPPG = (rPPG - torch.mean(rPPG, dim=-1).view(-1, 1)) / (torch.std(rPPG, dim=-1).view(-1, 1) + 1e-8)
                BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)

                loss_ecg = self.criterion_Pearson(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets. (kept same shape/behavior as original)"""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test = self.model(data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs 
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    # HR calculation based on ground truth label
    def get_hr(self, y, sr=30, min=30, max=180):
        p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
        return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60
