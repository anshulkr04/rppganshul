"""PhysMamba Trainer with FFT-majority-for-20%-then-CWT-dominant schedule.
Only FFT + CWT losses are used (no time-domain loss).
"""

import os
import math
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from scipy.signal import welch

from collections import OrderedDict
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson  # kept for compatibility if needed
from neural_methods.model.PhysMamba import PhysMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer

# ---------------- Helper functions ----------------

def spectral_log_magnitude(x, n_fft=256):
    """
    x: [B, T]
    returns log1p(|rfft(x, n=n_fft)|)
    """
    X = torch.fft.rfft(x, n=n_fft)
    mag = torch.abs(X)
    return torch.log1p(mag)

def morlet_wavelet(freq, sr, width=6.0, duration=None):
    """
    Build Morlet wavelet (real, imag) sampled at sr for center frequency `freq` (Hz).
    Returns CPU torch tensors (float32).
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
    # L1 normalize kernel to keep conv magnitudes stable
    real = real / (real.abs().sum() + 1e-12)
    imag = imag / (imag.abs().sum() + 1e-12)
    return real, imag

def make_morlet_bank(freqs_hz, sr, device):
    """
    Returns kernels_real, kernels_imag: [n_scales, 1, K] on `device`.
    All kernels padded to same odd length.
    """
    tmp = []
    max_len = 0
    for f in freqs_hz:
        r, im = morlet_wavelet(f, sr)
        tmp.append((r, im))
        max_len = max(max_len, r.numel())
    if max_len % 2 == 0:
        max_len += 1
    kr = []
    ki = []
    for r, im in tmp:
        pad = max_len - r.numel()
        left = pad // 2
        right = pad - left
        rpad = F.pad(r, (left, right))
        impad = F.pad(im, (left, right))
        kr.append(rpad.view(1,1,-1))
        ki.append(impad.view(1,1,-1))
    kernels_real = torch.cat(kr, dim=0).to(device)
    kernels_imag = torch.cat(ki, dim=0).to(device)
    return kernels_real, kernels_imag

def cwt_magnitude_conv1d(x, kernels_real, kernels_imag):
    """
    x: [B, T] -> returns magnitude [B, n_scales, T]
    """
    x = x.unsqueeze(1)  # [B,1,T]
    kernel_size = kernels_real.shape[-1]
    pad = kernel_size // 2
    x_padded = F.pad(x, (pad, pad), mode='reflect')
    real_out = F.conv1d(x_padded, kernels_real, padding=0)  # [B, n_scales, T]
    imag_out = F.conv1d(x_padded, kernels_imag, padding=0)
    mag = torch.sqrt(real_out**2 + imag_out**2 + 1e-12)
    return mag

def cosine_interp(a0, a1, alpha):
    """
    Cosine interpolation between a0 (alpha=0) and a1 (alpha=1).
    """
    mu = 0.5 * (1 - math.cos(math.pi * alpha))
    return (1 - mu) * a0 + mu * a1

# ---------------- PhysMambaTrainer ----------------

class PhysMambaTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.frame_rate = config.TRAIN.DATA.FS

        # model
        self.model = PhysMamba().to(self.device)
        if self.num_of_gpu > 0:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

        # optimizer & scheduler
        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0.0005)
            # OneCycleLR used previously; safe to keep (step each batch)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=max(1, self.num_train_batches))
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysMamba trainer initialized in incorrect toolbox mode!")

    def _compute_schedule_weights(self, epoch, total_epochs,
                                  frac_fft_major=0.2,
                                  transition_frac=0.10,
                                  w_fft_high=0.85,
                                  w_fft_low=0.05):
        """
        Compute weights w_fft, w_cwt for given epoch.
        - frac_fft_major: fraction of epochs where FFT must be majority (e.g., 0.2)
        - transition_frac: fraction of epochs used for smooth transition after majority period (e.g., 0.10)
        - w_fft_high: FFT weight during majority period
        - w_fft_low: FFT weight after transition (final)
        w_cwt = 1 - w_fft (no time loss).
        """
        E = total_epochs
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
        """Training with only FFT + CWT losses and the specified schedule."""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        # CWT / Morlet bank settings
        sr = int(self.frame_rate)
        n_scales = getattr(self.config, 'CWT', {}).get('N_SCALES', 32) if getattr(self.config, 'CWT', None) else 32
        freqs = torch.linspace(0.5, 4.0, n_scales).tolist()  # HR band
        kernels_real, kernels_imag = make_morlet_bank(freqs, sr, self.device)

        # FFT param
        n_fft = getattr(self.config, 'FFT', {}).get('N_FFT', 256) if getattr(self.config, 'FFT', None) else 256

        # baseline normalization (computed on first batch)
        L0_fft = None
        L0_cwt = None
        eps = 1e-8

        total_epochs = self.max_epoch_num
        # save a pre-transition checkpoint flag
        pre_transition_saved = False

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"==== Training Epoch: {epoch} / {self.max_epoch_num - 1} ====")
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)

            # compute schedule weights: FFT-major for first 20% epochs (parameters inside)
            w_fft, w_cwt = self._compute_schedule_weights(epoch, total_epochs,
                                                          frac_fft_major=0.2,
                                                          transition_frac=0.10,
                                                          w_fft_high=0.85,
                                                          w_fft_low=0.05)

            # optional: reduce LR at start of transition to stabilize (when epoch == t_start)
            # compute t_start to know if we're at transition start
            E = total_epochs
            E_major = int(math.floor(0.2 * E))
            E_transition = max(1, int(round(0.10 * E)))
            t_start = E_major
            if epoch == t_start and epoch > 0:
                # reduce learning rate by factor 0.33 to stabilize (recommended)
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.33
                if not pre_transition_saved:
                    self.save_model(epoch)
                    pre_transition_saved = True

            running_loss = 0.0
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].float(), batch[1].float()
                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)  # expected [B, T] (reshape if needed)

                # reshape if model outputs [B,1,T] etc.
                if pred_ppg.dim() == 3 and pred_ppg.shape[1] == 1:
                    pred_ppg = pred_ppg.squeeze(1)
                if pred_ppg.dim() == 3 and pred_ppg.shape[-1] == 1:
                    pred_ppg = pred_ppg.squeeze(-1)

                # per-sample normalization along time axis
                pred_ppg = (pred_ppg - torch.mean(pred_ppg, dim=-1, keepdim=True)) / \
                           (torch.std(pred_ppg, dim=-1, keepdim=True) + 1e-8)
                labels = (labels - torch.mean(labels, dim=-1, keepdim=True)) / \
                         (torch.std(labels, dim=-1, keepdim=True) + 1e-8)

                # FFT loss (log-magnitude MSE)
                fft_pred_log = spectral_log_magnitude(pred_ppg, n_fft=n_fft)
                fft_target_log = spectral_log_magnitude(labels, n_fft=n_fft)
                L_fft = F.mse_loss(fft_pred_log, fft_target_log, reduction='mean')

                # CWT loss (log1p magnitude MSE using in-graph Morlet conv)
                cwt_pred = cwt_magnitude_conv1d(pred_ppg, kernels_real, kernels_imag)
                cwt_target = cwt_magnitude_conv1d(labels, kernels_real, kernels_imag)
                L_cwt = F.mse_loss(torch.log1p(cwt_pred), torch.log1p(cwt_target), reduction='mean')

                # initialize normalization baselines on first batch
                if L0_fft is None:
                    L0_fft = max(L_fft.detach().item(), 1e-6)
                    L0_cwt = max(L_cwt.detach().item(), 1e-6)
                    print(f"Init baselines: L0_fft={L0_fft:.6e}, L0_cwt={L0_cwt:.6e}")

                # normalize losses
                L_fft_norm = L_fft / (L0_fft + eps)
                L_cwt_norm = L_cwt / (L0_cwt + eps)

                # combined loss: only FFT + CWT (weights sum to ~1)
                loss = float(w_fft) * L_fft_norm + float(w_cwt) * L_cwt_norm

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                # step scheduler if present (OneCycleLR expects step per batch)
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass

                running_loss += loss.item()
                if idx % 100 == 99:
                    print(f'[{epoch}, {idx + 1:5d}] avg loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                tbar.set_postfix(loss=loss.item(), w_fft=w_fft, w_cwt=w_cwt)

            # end epoch: save and validate
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print('validation loss (pearson time-domain): ', valid_loss)
                if self.min_valid_loss is None or (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
            torch.cuda.empty_cache()

        # training finished
        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))

    def valid(self, data_loader):
        """Runs model on validation set and returns negative Pearson loss (time-domain) as metric."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validating===")
        valid_loss = []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(torch.float32).to(self.device)
                rPPG = self.model(valid_batch[0].to(torch.float32).to(self.device))
                if rPPG.dim() == 3 and rPPG.shape[1] == 1:
                    rPPG = rPPG.squeeze(1)
                if rPPG.dim() == 3 and rPPG.shape[-1] == 1:
                    rPPG = rPPG.squeeze(-1)

                rPPG = (rPPG - torch.mean(rPPG, dim=-1, keepdim=True)) / (torch.std(rPPG, dim=-1, keepdim=True) + 1e-8)
                BVP_label = (BVP_label - torch.mean(BVP_label, dim=-1, keepdim=True)) / (torch.std(BVP_label, dim=-1, keepdim=True) + 1e-8)
                # Use negative Pearson as validation metric (lower is better)
                num = (rPPG * BVP_label).sum(dim=1)
                den = torch.sqrt((rPPG**2).sum(dim=1) * (BVP_label**2).sum(dim=1) + 1e-12)
                corr = (num / den).cpu().numpy()
                loss_ecg = 1.0 - np.mean(corr)
                valid_loss.append(float(loss_ecg))
                vbar.set_postfix(loss=loss_ecg)
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """Runs the model on test sets (same as before)."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL.PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL.PATH))
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
                data, label = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test = self.model(data)

                if pred_ppg_test.dim() == 3 and pred_ppg_test.shape[1] == 1:
                    pred_ppg_test = pred_ppg_test.squeeze(1)
                if pred_ppg_test.dim() == 3 and pred_ppg_test.shape[-1] == 1:
                    pred_ppg_test = pred_ppg_test.squeeze(-1)

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
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    def get_hr(self, y, sr=30, min=30, max=180):
        p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
        return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60
