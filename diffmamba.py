import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.mamba.mamba_ssm import Mamba2 as Mamba
from timm.models.layers import DropPath, trunc_normal_


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
# Lateral Connection (SlowFast)
# ------------------------------------------------------------

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

    def forward(self, slow_path, fast_path):

        fast_path = self.conv(fast_path)

        return slow_path + fast_path


# ------------------------------------------------------------
# Temporal Multi-Scale Block
# ------------------------------------------------------------

class TemporalMultiScale(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv3 = nn.Conv3d(
            channels, channels,
            (3,1,1),
            padding=(1,0,0),
            groups=channels
        )

        self.conv7 = nn.Conv3d(
            channels, channels,
            (7,1,1),
            padding=(3,0,0),
            groups=channels
        )

        self.conv15 = nn.Conv3d(
            channels, channels,
            (15,1,1),
            padding=(7,0,0),
            groups=channels
        )

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
# Periodic State Kernel
# ------------------------------------------------------------

class PeriodicStateKernel(nn.Module):

    def __init__(self, channels, fps=30):

        super().__init__()

        self.freq = nn.Parameter(torch.tensor(1.2))
        self.amp = nn.Parameter(torch.ones(channels))

        self.fps = fps

    def forward(self, x):

        B,C,T,H,W = x.shape

        t = torch.arange(T, device=x.device).float() / self.fps

        freq = torch.clamp(self.freq,0.7,4.0)

        sinusoid = torch.sin(2*torch.pi*freq*t)

        sinusoid = sinusoid.view(1,1,T,1,1)

        amp = self.amp.view(1,C,1,1,1)

        return x * (1 + 0.1 * amp * sinusoid)


# ------------------------------------------------------------
# Mamba Layer
# ------------------------------------------------------------

class MambaLayer(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):

        B, C, T, H, W = x.shape

        # ---------------------------------
        # REDUCED SPATIAL TOKENS (KEY FIX)
        # ---------------------------------

        # downsample spatially (keep structure!)
        x_ds = F.avg_pool3d(x, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        # shape: (B, C, T, H/4, W/4)

        B, C, T, Hs, Ws = x_ds.shape

        # flatten tokens
        tokens = x_ds.reshape(B, C, T * Hs * Ws).transpose(1, 2)  # (B, N, C)

        # ---------------------------------
        # MAMBA
        # ---------------------------------

        y = self.norm1(tokens)
        y = self.mamba(y)

        out = self.norm2(tokens + y)

        # ---------------------------------
        # MEMORY-SAFE FREQUENCY GATING
        # ---------------------------------

        # 1. spatial pooling (VERY IMPORTANT)
        pooled = out.mean(dim=[3, 4])   # (B, C, T)

        # 2. FFT on temporal only
        fft = torch.fft.rfft(pooled, dim=2)
        mag = torch.abs(fft)

        # 3. HR band selection
        band = mag[:, :, 1:8]   # adjust if needed

        # 4. compute weight
        weight = band.mean(dim=-1, keepdim=True)  # (B, C, 1)
        weight = torch.sigmoid(weight)

        # 5. apply back (lightweight broadcast)
        out = out * weight.unsqueeze(-1).unsqueeze(-1)

        # ---------------------------------
        # RESTORE SHAPE
        # ---------------------------------

        out = out.transpose(1, 2).reshape(B, C, T, Hs, Ws)

        # after upsample
        gate = torch.sigmoid(x)   # original input features

        out = F.interpolate(out, size=(T, H, W), mode='trilinear', align_corners=False)

        out = out * gate

        return out

# ------------------------------------------------------------
# Temporal Refiner
# ------------------------------------------------------------

class TemporalRefiner(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv3d(
            channels,
            channels,
            (5,1,1),
            padding=(2,0,0)
        )

        self.conv2 = nn.Conv3d(
            channels,
            channels,
            (3,1,1),
            padding=(1,0,0)
        )

        self.bn1 = nn.BatchNorm3d(channels)
        self.bn2 = nn.BatchNorm3d(channels)

        self.act = nn.GELU()

    def forward(self,x):

        r = x

        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.act(x+r)


# ------------------------------------------------------------
# PhysMamba
# ------------------------------------------------------------

class PhysMamba(nn.Module):

    def __init__(self,
                 theta=0.5,
                 frames=128):

        super().__init__()

        self.periodic_slow = PeriodicStateKernel(64)
        self.periodic_fast = PeriodicStateKernel(32)

        # Backbone
        self.ConvBlock1 = conv_block(3,16,[1,5,5],1,[0,2,2])
        self.ConvBlock2 = conv_block(16,32,[3,3,3],1,1)
        self.ConvBlock3 = conv_block(32,64,[3,3,3],1,1)

        self.ConvBlock4 = conv_block(64,64,[4,1,1],[4,1,1],0)
        self.ConvBlock5 = conv_block(64,32,[2,1,1],[2,1,1],0)

        self.ConvBlock6 = conv_block(32,32,[3,1,1],1,[1,0,0],activation='elu')

        self.MaxpoolSpa = nn.MaxPool3d((1,2,2),(1,2,2))

        # Temporal modules
        self.temporal_slow = TemporalMultiScale(64)
        self.temporal_fast = TemporalMultiScale(32)

        # Mamba blocks
        self.Block1 = MambaLayer(64)
        self.Block2 = MambaLayer(64)
        self.Block3 = MambaLayer(64)

        self.Block4 = MambaLayer(32)
        self.Block5 = MambaLayer(32)

        # SlowFast fusion
        self.fuse_1 = LateralConnection(32,64)
        self.fuse_2 = LateralConnection(32,64)

        # Upsampling
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

        self.ConvBlockLast = nn.Conv3d(48,1,[1,1,1])

        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))


    def forward(self, x):

        B,C,T,H,W = x.shape

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpa(x)

        s_x = self.ConvBlock4(x)
        f_x = self.ConvBlock5(x)

        s_x = self.periodic_slow(s_x)
        f_x = self.periodic_fast(f_x)

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

        x = torch.cat((f_x3,s_x3), dim=1)

        x = self.upsample2(x)

        x = self.refiner(x)

        x = self.poolspa(x)

        x = self.ConvBlockLast(x)

        rPPG = x.squeeze(1).squeeze(-1).squeeze(-1)

        return rPPG