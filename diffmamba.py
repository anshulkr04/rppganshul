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
# Frequency Attention (Improved)
# ------------------------------------------------------------
def frequency_attention(x):
    B,C,T,H,W = x.shape

    pooled = x.mean(dim=[3,4])  # (B,C,T)

    fft = torch.fft.rfft(pooled, dim=2)
    mag = torch.abs(fft)

    mag = mag / (mag.mean(dim=2, keepdim=True) + 1e-6)

    weight = mag.mean(dim=2, keepdim=True)  # (B,C,1)
    weight = weight.unsqueeze(-1).unsqueeze(-1)

    return x * (1 + 0.1 * weight) + 0.05 * weight


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
# Multi-Scale Mamba Layer (CORE FIX)
# ------------------------------------------------------------
class MambaLayer(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mamba_fine = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_coarse = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)

    def forward(self, x):

        B,C,T,H,W = x.shape

        # -------- Fine branch
        x_fine = F.avg_pool3d(x, kernel_size=(1,2,2))
        B,C,T,Hf,Wf = x_fine.shape

        tokens_fine = x_fine.permute(0,2,3,4,1).reshape(B, T*Hf*Wf, C)

        y1 = self.norm1(tokens_fine)
        y1 = self.mamba_fine(y1)

        out_fine = self.norm2(tokens_fine + y1)
        out_fine = out_fine.reshape(B,T,Hf,Wf,C).permute(0,4,1,2,3)

        # -------- Coarse branch
        x_coarse = F.avg_pool3d(x, kernel_size=(1,4,4))
        B,C,T,Hc,Wc = x_coarse.shape

        tokens_coarse = x_coarse.permute(0,2,3,4,1).reshape(B, T*Hc*Wc, C)

        y2 = self.norm1(tokens_coarse)
        y2 = self.mamba_coarse(y2)

        out_coarse = self.norm2(tokens_coarse + y2)
        out_coarse = out_coarse.reshape(B,T,Hc,Wc,C).permute(0,4,1,2,3)

        # -------- Upsample coarse → fine
        out_coarse = F.interpolate(out_coarse, size=(T,Hf,Wf), mode='trilinear', align_corners=False)

        # -------- Fusion
        out = out_fine + out_coarse

        # -------- Residual
        out = out + 0.1 * x_fine

        return out


# ------------------------------------------------------------
# Lateral Connection
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

    def forward(self, slow, fast):
        fast = self.conv(fast)
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

        # 🔥 frequency awareness
        s_x = frequency_attention(s_x)
        f_x = frequency_attention(f_x)

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