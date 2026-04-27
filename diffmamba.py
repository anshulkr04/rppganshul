import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.mamba.mamba_ssm import Mamba2 as Mamba


# ------------------------------------------------------------
# Conv Block
# ------------------------------------------------------------
def conv_block(in_channels, out_channels, kernel_size, stride, padding,
               bn=True, activation='relu'):

    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]

    if bn:
        layers.append(nn.BatchNorm3d(out_channels))

    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(inplace=True))

    return nn.Sequential(*layers)


# ------------------------------------------------------------
# Frequency Attention (band-limited to the physiological HR range)
# ------------------------------------------------------------
class FrequencyAttention(nn.Module):
    """
    Channel-wise attention computed *only* over the cardiac band
    (0.5–4.0 Hz ≈ 30–240 BPM). Discarding DC drift and high-frequency
    motion bins before pooling stops noise from bleeding into the
    attention weights. The two mixing scalars (originally hard-coded
    0.1 / 0.05) are made learnable.
    """

    def __init__(self, fs=30.0, low_hz=0.5, high_hz=4.0):
        super().__init__()
        self.fs = fs
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        B, C, T, H, W = x.shape

        pooled = x.mean(dim=[3, 4])                       # (B, C, T)
        mag = torch.abs(torch.fft.rfft(pooled, dim=2))    # (B, C, F)

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(x.device)
        band = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        mag_band = mag[..., band] if band.any() else mag

        mag_band = mag_band / (mag_band.mean(dim=2, keepdim=True) + 1e-6)

        weight = mag_band.mean(dim=2, keepdim=True)       # (B, C, 1)
        weight = weight.unsqueeze(-1).unsqueeze(-1)

        return x * (1 + self.alpha * weight) + self.beta * weight


# ------------------------------------------------------------
# Temporal Multi-Scale Block
# ------------------------------------------------------------
class TemporalMultiScale(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv3 = nn.Conv3d(channels, channels, (3,1,1),
                               padding=(1,0,0), groups=channels)

        self.conv7 = nn.Conv3d(channels, channels, (7,1,1),
                               padding=(3,0,0), groups=channels)

        self.conv15 = nn.Conv3d(channels, channels, (15,1,1),
                                padding=(7,0,0), groups=channels)

        self.pointwise = nn.Conv3d(channels*3, channels, 1)

        self.bn = nn.BatchNorm3d(channels)
        self.act = nn.GELU()

    def forward(self, x):

        f1 = self.conv3(x)
        f2 = self.conv7(x)
        f3 = self.conv15(x)

        out = torch.cat([f1,f2,f3], dim=1)
        out = self.pointwise(out)
        out = self.bn(out)

        return x + self.act(out)


# ------------------------------------------------------------
# Bidirectional Multi-Scale Mamba Layer
# ------------------------------------------------------------
class MambaLayer(nn.Module):
    """
    Two changes vs. the original block:

    1. Bidirectional scan. Mamba2 is a causal SSM, but the BVP target is
       non-causal — future frames carry as much info about pulse(t) as
       past frames. We run a forward Mamba and a second Mamba on the
       time-flipped sequence, then average.

    2. Temporal-major token order. Tokens are flattened as
       (H, W, T) → length H·W·T, so each contiguous slice of the scan is
       one spatial location's full time sequence. The SSM hidden state
       therefore mixes along time first, which is the dominant axis for
       heart-rate inference. (Original order was (T, H, W), forcing the
       state to traverse all spatial tokens between consecutive frames.)

    The fine→fine residual mixing scalar is also made learnable.
    """

    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mamba_fine_fwd   = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_fine_bwd   = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_coarse_fwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_coarse_bwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)

        self.res_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _bidir(tokens, fwd_block, bwd_block):
        y_fwd = fwd_block(tokens)
        y_bwd = bwd_block(tokens.flip(dims=[1])).flip(dims=[1])
        return 0.5 * (y_fwd + y_bwd)

    def _scan(self, x_pooled, fwd_block, bwd_block):
        B, C, T, H, W = x_pooled.shape

        # (B, H, W, T, C) → temporal sub-sequences contiguous in token dim
        tokens = x_pooled.permute(0, 3, 4, 2, 1).reshape(B, H * W * T, C)

        y = self.norm1(tokens)
        y = self._bidir(y, fwd_block, bwd_block)

        out = self.norm2(tokens + y)
        out = out.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)
        return out

    def forward(self, x):
        _, _, T, H, W = x.shape

        # Clamp pool kernels to the actual spatial size — by Block3 the
        # feature map can be as small as 2×2, where a fixed 4×4 coarse
        # pool would error out. When H/W is too small, the coarse branch
        # degenerates to the fine resolution but its (separate) Mamba
        # weights still contribute, so all params stay live for DDP/DP.
        fine_kh,   fine_kw   = min(2, H), min(2, W)
        coarse_kh, coarse_kw = min(4, H), min(4, W)

        # Pool → Mamba scan at fine and coarse spatial resolutions
        x_fine = F.avg_pool3d(x, kernel_size=(1, fine_kh, fine_kw))
        out_fine = self._scan(x_fine, self.mamba_fine_fwd, self.mamba_fine_bwd)

        x_coarse = F.avg_pool3d(x, kernel_size=(1, coarse_kh, coarse_kw))
        out_coarse = self._scan(x_coarse, self.mamba_coarse_fwd, self.mamba_coarse_bwd)

        # Upsample both branches back to the input spatial size so
        # MambaLayer is shape-preserving. Slow and fast pathways apply
        # MambaLayer a different number of times; if MambaLayer reduced
        # spatial, the two paths' final feature maps would mismatch and
        # the closing torch.cat would fail. Only the explicit MaxpoolSpa
        # layers (symmetric across pathways) should down-sample.
        out_fine   = F.interpolate(out_fine,   size=(T, H, W),
                                   mode='trilinear', align_corners=False)
        out_coarse = F.interpolate(out_coarse, size=(T, H, W),
                                   mode='trilinear', align_corners=False)

        return out_fine + out_coarse + self.res_scale * x


# ------------------------------------------------------------
# Lateral Connection
# ------------------------------------------------------------
class ChannelGate(nn.Module):
    """Squeeze-Excite gate. Used on the projected fast features before
    they are added into the slow stream — lets the model suppress
    fast-path channels that carry motion/noise instead of cardiac signal."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C = x.shape[:2]
        s = x.mean(dim=[2, 3, 4])
        g = self.fc(s).view(B, C, 1, 1, 1)
        return x * g


class LateralConnection(nn.Module):

    def __init__(self, fast_channels=32, slow_channels=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(fast_channels, slow_channels,
                      kernel_size=[3,1,1],
                      stride=[2,1,1],
                      padding=[1,0,0]),
            nn.BatchNorm3d(slow_channels),
            nn.ReLU()
        )

        self.gate = ChannelGate(slow_channels)

    def forward(self, slow, fast):
        fast = self.gate(self.conv(fast))
        return slow + fast


# ------------------------------------------------------------
# Temporal Refiner
# ------------------------------------------------------------
class TemporalRefiner(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv3d(channels, channels, (5,1,1), padding=(2,0,0))
        self.conv2 = nn.Conv3d(channels, channels, (3,1,1), padding=(1,0,0))

        self.bn1 = nn.BatchNorm3d(channels)
        self.bn2 = nn.BatchNorm3d(channels)

        self.act = nn.GELU()

    def forward(self, x):
        r = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + r)


# ------------------------------------------------------------
# MAIN MODEL
# ------------------------------------------------------------
class PhysMamba(nn.Module):

    def __init__(self, frames=128):
        super().__init__()

        # 6 input channels = 3 raw RGB + 3 temporal first-difference.
        # The diff stream cancels static appearance (skin tone, lighting,
        # background) and exposes the inter-frame intensity changes that
        # carry the BVP signal. This is the central trick behind TS-CAN /
        # EfficientPhys / DiffPhys and is the largest single accuracy
        # lever in the rPPG literature.
        self.ConvBlock1 = conv_block(6,16,[1,5,5],1,[0,2,2])
        self.ConvBlock2 = conv_block(16,32,[3,3,3],1,1)
        self.ConvBlock3 = conv_block(32,64,[3,3,3],1,1)

        self.ConvBlock4 = conv_block(64,64,[4,1,1],[4,1,1],0)
        self.ConvBlock5 = conv_block(64,32,[2,1,1],[2,1,1],0)

        self.ConvBlock6 = conv_block(32,32,[3,1,1],1,[1,0,0],activation='elu')

        self.MaxpoolSpa = nn.MaxPool3d((1,2,2),(1,2,2))

        # Slow path is downsampled 4× in time vs. fast path. Use the
        # respective effective sample rates so the band mask matches reality.
        self.freq_attn_slow = FrequencyAttention(fs=30.0/4.0)
        self.freq_attn_fast = FrequencyAttention(fs=30.0/2.0)

        self.temporal_slow = TemporalMultiScale(64)
        self.temporal_fast = TemporalMultiScale(32)

        self.Block1 = MambaLayer(64)
        self.Block2 = MambaLayer(64)
        self.Block3 = MambaLayer(64)

        self.Block4 = MambaLayer(32)
        self.Block5 = MambaLayer(32)

        self.fuse_1 = LateralConnection(32,64)
        self.fuse_2 = LateralConnection(32,64)

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(64,64,[3,1,1],padding=(1,0,0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(96,48,[3,1,1],padding=(1,0,0)),
            nn.BatchNorm3d(48),
            nn.ELU()
        )

        self.refiner = TemporalRefiner(48)

        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))
        self.ConvBlockLast = nn.Conv3d(48,1,[1,1,1])


    def forward(self, x):

        # Temporal first-difference stream, padded so length matches.
        diff = x[:, :, 1:] - x[:, :, :-1]
        diff = F.pad(diff, (0, 0, 0, 0, 1, 0))   # (W,W,H,H,T_left,T_right)
        x = torch.cat([x, diff], dim=1)          # (B, 6, T, H, W)

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpa(x)

        s_x = self.ConvBlock4(x)
        f_x = self.ConvBlock5(x)

        s_x = self.freq_attn_slow(s_x)
        f_x = self.freq_attn_fast(f_x)

        s_x = self.temporal_slow(s_x)
        f_x = self.temporal_fast(f_x)

        # Stage 1
        s_x1 = self.MaxpoolSpa(self.Block1(s_x))
        f_x1 = self.MaxpoolSpa(self.Block4(f_x))
        s_x1 = self.fuse_1(s_x1, f_x1)

        # Stage 2
        s_x2 = self.MaxpoolSpa(self.Block2(s_x1))
        f_x2 = self.MaxpoolSpa(self.Block5(f_x1))
        s_x2 = self.fuse_2(s_x2, f_x2)

        # Stage 3
        s_x3 = self.Block3(s_x2)

        s_x3 = self.upsample1(s_x3)
        f_x3 = self.ConvBlock6(f_x2)

        x = torch.cat((f_x3, s_x3), dim=1)

        x = self.upsample2(x)
        x = self.refiner(x)

        x = self.poolspa(x)
        x = self.ConvBlockLast(x)

        rPPG = x.squeeze(1).squeeze(-1).squeeze(-1)

        return rPPG
