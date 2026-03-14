import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from timm.models.layers import DropPath, trunc_normal_

from neural_methods.model.layers import conv_block
from neural_methods.model.layers import CDC_T
from neural_methods.model.layers import ChannelAttention3D
from neural_methods.model.layers import LateralConnection


# ------------------------------------------------------------
# Periodic State Kernel (physiological bias)
# ------------------------------------------------------------

class PeriodicStateKernel(nn.Module):

    def __init__(self, channels, fps=30):
        super().__init__()

        self.channels = channels
        self.fps = fps

        self.freq = nn.Parameter(torch.tensor(1.2))
        self.amp = nn.Parameter(torch.ones(channels))

    def forward(self, x):

        B, C, T, H, W = x.shape
        device = x.device

        t = torch.arange(T, device=device).float() / self.fps
        freq = torch.clamp(self.freq, 0.7, 4.0)

        sinusoid = torch.sin(2 * torch.pi * freq * t)
        sinusoid = sinusoid.view(1,1,T,1,1)

        amp = self.amp.view(1,C,1,1,1)

        return x * (1 + amp * sinusoid)


# ------------------------------------------------------------
# Mamba Layer
# ------------------------------------------------------------

class MambaLayer(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()

        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        drop_path = 0
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_patch_token(self, x):

        B, C, T, H, W = x.shape

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        x_norm = self.norm1(x_flat)

        x_mamba = self.mamba(x_norm)

        x_out = self.norm2(x_flat + self.drop_path(x_mamba))

        return x_out.transpose(-1, -2).reshape(B, C, *img_dims)

    def forward(self, x):

        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()

        return self.forward_patch_token(x)


# ------------------------------------------------------------
# DiffMamba Layer
# ------------------------------------------------------------

class DiffMambaLayer(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=4, layer_idx=0):

        super().__init__()

        self.dim = dim
        self.layer_idx = layer_idx

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        d_state_small = max(1, d_state // 2)

        self.mixer_1 = Mamba(
            d_model=dim,
            d_state=d_state_small,
            d_conv=d_conv,
            expand=1
        )

        self.mixer_2 = Mamba(
            d_model=dim,
            d_state=d_state_small,
            d_conv=d_conv,
            expand=1
        )

        self.lambda_q = nn.Parameter(torch.randn(dim))

        self.subln = nn.LayerNorm(dim)

    def forward_patch_token(self, x):

        B, C, T, H, W = x.shape

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        x_norm = self.norm1(x_flat)

        y1 = self.mixer_1(x_norm)
        y2 = self.mixer_2(x_norm)

        lam = torch.sigmoid(self.lambda_q.sum())

        attn = y1 - lam * y2
        attn = self.subln(attn)

        x_out = self.norm2(x_flat + attn)

        return x_out.transpose(-1, -2).reshape(B, C, *img_dims)

    def forward(self, x):

        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()

        return self.forward_patch_token(x)


# ------------------------------------------------------------
# PhysMamba Model
# ------------------------------------------------------------

class PhysMamba(nn.Module):

    def __init__(self,
                 theta=0.5,
                 drop_rate1=0.25,
                 drop_rate2=0.5,
                 frames=128,
                 diffmamba=False):

        super().__init__()

        self.periodic_kernel = PeriodicStateKernel(64)

        # Backbone
        self.ConvBlock1 = conv_block(3,16,[1,5,5],stride=1,padding=[0,2,2])
        self.ConvBlock2 = conv_block(16,32,[3,3,3],stride=1,padding=1)
        self.ConvBlock3 = conv_block(32,64,[3,3,3],stride=1,padding=1)

        self.ConvBlock4 = conv_block(64,64,[4,1,1],stride=[4,1,1],padding=0)
        self.ConvBlock5 = conv_block(64,32,[2,1,1],stride=[2,1,1],padding=0)
        self.ConvBlock6 = conv_block(32,32,[3,1,1],stride=1,padding=[1,0,0],activation='elu')

        # Build Mamba blocks
        if diffmamba:
            block_slow = DiffMambaLayer(64)
            block_fast = DiffMambaLayer(32)
        else:
            block_slow = MambaLayer(64)
            block_fast = MambaLayer(32)

        self.Block1 = block_slow
        self.Block2 = block_slow

        self.Block4 = block_fast
        self.Block5 = block_fast

        # Removed redundant layers safely
        self.Block3 = nn.Identity()
        self.Block6 = nn.Identity()

        self.MaxpoolSpa = nn.MaxPool3d((1,2,2),(1,2,2))

        self.fuse_1 = LateralConnection(fast_channels=32, slow_channels=64)
        self.fuse_2 = LateralConnection(fast_channels=32, slow_channels=64)

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

        self.ConvBlockLast = nn.Conv3d(48,1,[1,1,1])

        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

    def forward(self, x):

        batch, channel, length, width, height = x.shape

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpa(x)

        s_x = self.ConvBlock4(x)
        f_x = self.ConvBlock5(x)

        s_x = self.periodic_kernel(s_x)

        # first stage
        s_x1 = self.Block1(s_x)
        s_x1 = self.MaxpoolSpa(s_x1)

        f_x1 = self.Block4(f_x)
        f_x1 = self.MaxpoolSpa(f_x1)

        s_x1 = self.fuse_1(s_x1, f_x1)

        # second stage
        s_x2 = self.Block2(s_x1)
        s_x2 = self.MaxpoolSpa(s_x2)

        f_x2 = self.Block5(f_x1)
        f_x2 = self.MaxpoolSpa(f_x2)

        s_x2 = self.fuse_2(s_x2, f_x2)

        # removed third blocks
        s_x3 = s_x2
        f_x3 = f_x2

        s_x3 = self.upsample1(s_x3)

        f_x3 = self.ConvBlock6(f_x3)

        x_fusion = torch.cat((f_x3, s_x3), dim=1)

        x_final = self.upsample2(x_fusion)

        x_final = self.poolspa(x_final)

        x_final = self.ConvBlockLast(x_final)

        rPPG = x_final.view(-1, length)

        return rPPG
