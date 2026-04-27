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
# Bidirectional Multi-Temporal-Rate Mamba Layer
# ------------------------------------------------------------
class MambaLayer(nn.Module):
    """
    A multi-temporal-rate Mamba block, structured after RhythmMamba's
    multi-temporal scan and PhysMamba's slow/fast philosophy — both
    pick the *temporal* axis as the multi-scale axis because the BVP
    signal is essentially a global 1-D temporal sequence; multi-spatial-
    scale Mamba (the previous design here) is a poor fit for rPPG.

    Design choices and the papers that motivate them:

    • **Bidirectional scan** (Vim, RhythmMamba). Mamba2 is causal, but
      the BVP target is non-causal — futures carry as much info as pasts.
      Each branch is forward + time-flipped, averaged.

    • **Temporal-major token order**. Tokens are flattened as (H,W,T)
      so each contiguous slice along the scan dim is one spatial
      location's whole time series — the SSM state mixes along time
      first, which is the axis that carries the cardiac signal.

    • **Two parallel temporal rates**: native T (fine cardiac-cycle
      detail) and T/2 via temporal pooling (longer effective context
      and finer state-space coverage of low-HR / inter-beat structure).
      Spatial dim is shared by both branches (single mild 2x2 pool) so
      each branch's only difference is its temporal sample rate.

    • **d_state = 32** (Mamba2 advantage). Mamba2's structured-state
      design lets the SSM hidden dim grow at near-zero compute cost.
      A bigger state is better at memorising periodic patterns whose
      period (~30–50 frames at 30 fps for adult HR) approaches d_state.

    Layer is shape-preserving (output H,W,T = input H,W,T) so it can be
    stacked an asymmetric number of times across slow/fast pathways
    without producing mismatched feature maps at the final concat.
    """

    def __init__(self, dim, d_state=32):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Two parallel temporal rates × bidirectional → 4 SSMs per layer.
        self.mamba_t1_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=2)
        self.mamba_t1_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=2)
        self.mamba_t2_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=2)
        self.mamba_t2_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=2)

        self.res_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _bidir(tokens, fwd_block, bwd_block):
        y_fwd = fwd_block(tokens)
        y_bwd = bwd_block(tokens.flip(dims=[1])).flip(dims=[1])
        return 0.5 * (y_fwd + y_bwd)

    def _scan(self, x_in, fwd_block, bwd_block):
        B, C, T, H, W = x_in.shape

        # (B, H, W, T, C) — each spatial location's full time series
        # is contiguous in the scan dim.
        tokens = x_in.permute(0, 3, 4, 2, 1).reshape(B, H * W * T, C)

        y = self.norm1(tokens)
        y = self._bidir(y, fwd_block, bwd_block)

        out = self.norm2(tokens + y)
        return out.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)

    def forward(self, x):
        _, _, T, H, W = x.shape

        # Single mild spatial pool, shared across both temporal rates.
        # Clamped so it can't error on already-tiny feature maps.
        spa_kh, spa_kw = min(2, H), min(2, W)
        x_pool = F.avg_pool3d(x, kernel_size=(1, spa_kh, spa_kw))
        _, _, _, Hp, Wp = x_pool.shape

        # Branch 1: native temporal rate
        out_t1 = self._scan(x_pool, self.mamba_t1_fwd, self.mamba_t1_bwd)

        # Branch 2: half temporal rate — averaging two adjacent frames
        # both lengthens the effective receptive field of the SSM and
        # supplies a smoother view of slow rhythm components.
        if T >= 2:
            x_pool_t2 = F.avg_pool3d(x_pool, kernel_size=(2, 1, 1),
                                     stride=(2, 1, 1))
        else:
            x_pool_t2 = x_pool
        out_t2 = self._scan(x_pool_t2, self.mamba_t2_fwd, self.mamba_t2_bwd)
        out_t2 = F.interpolate(out_t2, size=(T, Hp, Wp),
                               mode='trilinear', align_corners=False)

        # Fuse temporal rates and upsample spatial back to input.
        out = 0.5 * (out_t1 + out_t2)
        out = F.interpolate(out, size=(T, H, W),
                            mode='trilinear', align_corners=False)

        return x + self.res_scale * out


# ------------------------------------------------------------
# Lateral Connection
# ------------------------------------------------------------
class ChannelGate(nn.Module):
    """Squeeze-Excite gate on the projected fast features inside the
    lateral connection. Initialised at identity (gate ≈ 0.98) so it
    can't disrupt early training — it only contributes once the
    optimiser sees a reason to suppress particular channels.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        # zero-weight + large positive bias on the projection out → the
        # gate ignores its input at init and outputs ~1.
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 4.0)

    def forward(self, x):
        B, C = x.shape[:2]
        s = x.mean(dim=[2, 3, 4])
        g = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * g.view(B, C, 1, 1, 1)


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

        self.ConvBlock1 = conv_block(3,16,[1,5,5],1,[0,2,2])
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
