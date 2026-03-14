import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from timm.models.layers import DropPath, trunc_normal_

# ------------------------------------------------------------
# Periodic State Kernel
# ------------------------------------------------------------

class PeriodicStateKernel(nn.Module):

    def __init__(self, channels, fps=30):
        super().__init__()

        self.channels = channels
        self.fps = fps

        self.freq = nn.Parameter(torch.tensor(1.2))
        self.amp = nn.Parameter(torch.ones(channels))

    def forward(self, x):

        B,C,T,H,W = x.shape
        device = x.device

        t = torch.arange(T, device=device).float() / self.fps
        freq = torch.clamp(self.freq, 0.7, 4.0)

        sinusoid = torch.sin(2 * torch.pi * freq * t)
        sinusoid = sinusoid.view(1,1,T,1,1)

        amp = self.amp.view(1,C,1,1,1)

        return x * (1 + amp * sinusoid)


# ------------------------------------------------------------
# Shared Mamba Wrapper
# ------------------------------------------------------------

class SharedMambaWrapper(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):

        out_f = self.block(x)

        x_rev = torch.flip(x, dims=[2])
        out_b = self.block(x_rev)
        out_b = torch.flip(out_b, dims=[2])

        return 0.5 * (out_f + out_b)


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
            expand=expand,
            bimamba=True
        )

        self.drop_path = nn.Identity()

    def forward_patch_token(self, x):

        B,C,T,H,W = x.shape

        n_tokens = x.shape[2:].numel()

        x_flat = x.reshape(B,C,n_tokens).transpose(-1,-2)

        x_norm = self.norm1(x_flat)

        x_mamba = self.mamba(x_norm)

        x_out = self.norm2(x_flat + self.drop_path(x_mamba))

        return x_out.transpose(-1,-2).reshape(B,C,T,H,W)

    def forward(self,x):

        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()

        return self.forward_patch_token(x)


# ------------------------------------------------------------
# DiffMamba Layer
# ------------------------------------------------------------

class DiffMambaLayer(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=4):

        super().__init__()

        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        d_state_small = max(1, d_state//2)

        self.mixer_1 = Mamba(
            d_model=dim,
            d_state=d_state_small,
            d_conv=d_conv,
            expand=1,
            bimamba=True
        )

        self.mixer_2 = Mamba(
            d_model=dim,
            d_state=d_state_small,
            d_conv=d_conv,
            expand=1,
            bimamba=True
        )

        self.lambda_q = nn.Parameter(torch.randn(dim))

        self.subln = nn.LayerNorm(dim)

    def forward_patch_token(self,x):

        B,C,T,H,W = x.shape

        n_tokens = x.shape[2:].numel()

        x_flat = x.reshape(B,C,n_tokens).transpose(-1,-2)

        x_norm = self.norm1(x_flat)

        y1 = self.mixer_1(x_norm)
        y2 = self.mixer_2(x_norm)

        lam = torch.sigmoid(self.lambda_q.sum())

        attn = y1 - lam * y2

        attn = self.subln(attn)

        out = self.norm2(x_flat + attn)

        return out.transpose(-1,-2).reshape(B,C,T,H,W)

    def forward(self,x):

        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()

        return self.forward_patch_token(x)


# ------------------------------------------------------------
# PhysMamba
# ------------------------------------------------------------

class PhysMamba(nn.Module):

    def __init__(self, frames=128, diffmamba=False):

        super().__init__()

        self.periodic_kernel = PeriodicStateKernel(64)

        # Backbone

        self.ConvBlock1 = nn.Conv3d(3,16,(1,5,5),padding=(0,2,2))
        self.ConvBlock2 = nn.Conv3d(16,32,3,padding=1)
        self.ConvBlock3 = nn.Conv3d(32,64,3,padding=1)

        self.ConvBlock4 = nn.Conv3d(64,64,(4,1,1),stride=(4,1,1))
        self.ConvBlock5 = nn.Conv3d(64,32,(2,1,1),stride=(2,1,1))

        # Blocks

        if diffmamba:

            base_slow = DiffMambaLayer(64)
            base_fast = DiffMambaLayer(32)

        else:

            base_slow = MambaLayer(64)
            base_fast = MambaLayer(32)

        self.Block1 = SharedMambaWrapper(base_slow)
        self.Block2 = SharedMambaWrapper(base_slow)

        self.Block4 = SharedMambaWrapper(base_fast)
        self.Block5 = SharedMambaWrapper(base_fast)

        self.Block3 = nn.Identity()
        self.Block6 = nn.Identity()

        # Upsample

        self.upsample1 = nn.Upsample(scale_factor=(2,1,1))
        self.upsample2 = nn.Upsample(scale_factor=(2,1,1))

        self.final_conv = nn.Conv3d(48,1,1)

        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

    def forward(self,x):

        B,C,T,H,W = x.shape

        x = F.elu(self.ConvBlock1(x))
        x = F.elu(self.ConvBlock2(x))
        x = F.elu(self.ConvBlock3(x))

        s_x = self.ConvBlock4(x)
        f_x = self.ConvBlock5(x)

        s_x = self.periodic_kernel(s_x)

        s_x1 = self.Block1(s_x)
        f_x1 = self.Block4(f_x)

        s_x2 = self.Block2(s_x1)
        f_x2 = self.Block5(f_x1)

        s_x3 = s_x2
        f_x3 = f_x2

        s_x3 = self.upsample1(s_x3)

        x_fusion = torch.cat((f_x3,s_x3),dim=1)

        x_final = self.upsample2(x_fusion)

        x_final = self.poolspa(x_final)

        x_final = self.final_conv(x_final)

        rPPG = x_final.view(B,T)

        return rPPG
