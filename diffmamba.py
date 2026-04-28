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
# Frequency Attention
# ------------------------------------------------------------
class FrequencyAttention(nn.Module):
    def __init__(self, fs=30.0, low_hz=0.5, high_hz=4.0):
        super().__init__()
        self.fs = fs
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        B, C, T, H, W = x.shape

        pooled = x.mean(dim=[3, 4])
        mag = torch.abs(torch.fft.rfft(pooled, dim=2))

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(x.device)
        band = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        mag_band = mag[..., band] if band.any() else mag

        mag_band = mag_band / (mag_band.mean(dim=2, keepdim=True) + 1e-6)

        weight = mag_band.mean(dim=2, keepdim=True)
        weight = weight.unsqueeze(-1).unsqueeze(-1)

        return x * (1 + self.alpha * weight) + self.beta * weight


# ------------------------------------------------------------
# Temporal Multi-Scale
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
# DiffMamba Layer (STABLE VERSION)
# ------------------------------------------------------------
class MambaLayer(nn.Module):

    def __init__(self, dim, d_state=32):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=2)

        self.res_scale = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.0))  # start near abs-only

        # pre temporal filter
        self.pre_filter = nn.Conv3d(dim, dim, (3,1,1), padding=(1,0,0), groups=dim)

    def _scan(self, x):
        B, C, T, H, W = x.shape
        tokens = x.permute(0, 3, 4, 2, 1).reshape(B, H * W * T, C)

        y = self.norm1(tokens)
        y = self.mamba(y)

        out = self.norm2(tokens + y)
        return out.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)

    def temporal_diff(self, x):
        x_diff = x[:,:,1:] - x[:,:,:-1]
        x_diff = F.pad(x_diff, (0,0,0,0,1,0))

        mean = x_diff.mean(dim=2, keepdim=True)
        std = x_diff.std(dim=2, keepdim=True) + 1e-5
        x_diff = (x_diff - mean) / std

        return x_diff

    def forward(self, x):

        x = x + self.pre_filter(x)

        x_diff = self.temporal_diff(x)

        out_abs = self._scan(x)
        out_diff = self._scan(x_diff)

        gate = torch.sigmoid(self.gamma)
        out = (1 - gate) * out_abs + gate * out_diff

        return x + self.res_scale * out


# ------------------------------------------------------------
# Remaining components (unchanged)
# ------------------------------------------------------------
class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
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
# FINAL MODEL (reduced depth)
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

        self.freq_attn_slow = FrequencyAttention(fs=30.0/4.0)
        self.freq_attn_fast = FrequencyAttention(fs=30.0/2.0)

        self.temporal_slow = TemporalMultiScale(64)
        self.temporal_fast = TemporalMultiScale(32)

        # REDUCED DEPTH
        self.Block1 = MambaLayer(64)
        self.Block2 = MambaLayer(64)
        self.Block4 = MambaLayer(32)

        self.fuse_1 = LateralConnection(32,64)

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

        s_x1 = self.MaxpoolSpa(self.Block1(s_x))
        f_x1 = self.MaxpoolSpa(self.Block4(f_x))
        s_x1 = self.fuse_1(s_x1, f_x1)

        s_x2 = self.Block2(s_x1)

        s_x2 = self.upsample1(s_x2)
        f_x2 = self.ConvBlock6(f_x1)

        x = torch.cat((f_x2, s_x2), dim=1)

        x = self.upsample2(x)
        x = self.refiner(x)

        x = self.poolspa(x)
        x = self.ConvBlockLast(x)

        rPPG = x.squeeze(1).squeeze(-1).squeeze(-1)

        # SIGNAL NORMALIZATION
        rPPG = (rPPG - rPPG.mean(dim=1, keepdim=True)) / (rPPG.std(dim=1, keepdim=True) + 1e-6)

        return rPPG