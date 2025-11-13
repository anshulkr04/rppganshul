class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False, layer_idx: int = 0):
        super(MambaLayer, self).__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        drop_path = 0
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_patch_token(self, x):
        B, C, nf, H, W = x.shape
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = self.norm2(x_flat + self.drop_path(x_mamba))
        out = x_out.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out 

    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        out = self.forward_patch_token(x)
        return out

# --- New DiffMambaLayer ---
class DiffMambaLayer(nn.Module):
    """
    Diff-style Mamba layer for image model. Contains two smaller Mamba mixers (mixer_1, mixer_2)
    and combines them with a learned lambda-like parameter similar to DiffMamba2Block.
    """
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, layer_idx: int = 0):
        super(DiffMambaLayer, self).__init__()
        self.dim = dim
        self.layer_idx = layer_idx
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Choose reduced sizes for the internal mixers (half the state) but at least 1
        d_state_small = max(1, d_state // 2)
        # reduce expansion to 1 for the smaller mixers (as in your transformer example)
        expand_small = 1

        # Two smaller Mamba mixers
        self.mixer_1 = Mamba(
                d_model=dim,
                d_state=d_state_small,
                d_conv=d_conv,
                expand=expand_small,
                bimamba=True,
        )
        self.mixer_2 = Mamba(
                d_model=dim,
                d_state=d_state_small,
                d_conv=d_conv,
                expand=expand_small,
                bimamba=True,
        )

        # Drop path (same behaviour as original layer)
        drop_path = 0
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # lambda initialization (mirrors DiffMamba2Block scheme)
        # scalar base initialization depending on layer index
        self.lambda_init = 0.8 - 0.6 * torch.exp(torch.tensor(-0.3 * float(self.layer_idx)))
        # per-d_model parameter that gets aggregated to a scalar (keeps flexibility)
        self.lambda_q1 = nn.Parameter(torch.randn(self.dim))

        # small sub layernorm (used after combining mixers)
        self.subln = nn.LayerNorm(self.dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # reuse same initialization rules as MambaLayer
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_patch_token(self, x):
        """
        x: [B, C, nf, H, W]
        same flatten & norm pattern as MambaLayer
        """
        B, C, nf, H, W = x.shape
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm1(x_flat)

        # run both smaller mixers
        y1 = self.mixer_1(x_norm)
        y2 = self.mixer_2(x_norm)

        # combine like DiffMamba: y1 - lambda * y2
        lambda_q1 = torch.sum(self.lambda_q1, dim=-1).float()
        lambda_full = torch.sigmoid(lambda_q1) + float(self.lambda_init)

        attn = y1 - lambda_full * y2
        attn = self.subln(attn)
        # scale down similarly to original Diff block
        hidden_states = attn * (1.0 - float(self.lambda_init))

        x_out = self.norm2(x_flat + self.drop_path(hidden_states))
        out = x_out.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out

    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        return self.forward_patch_token(x)


# --- PhysMamba modifications: allow diffmamba settings & build blocks accordingly ---

class PhysMamba(nn.Module):
    def __init__(self, theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=128, diffmamba: bool = False, diffmamba_settings: str = "alternate"):
        """
        diffmamba: enable Diff-style Mamba blocks
        diffmamba_settings: 'none'|'all'|'quarter'|'alternate'
        """
        super(PhysMamba, self).__init__()

        self.diffmamba = diffmamba
        self.diffmamba_settings = diffmamba_settings  # "alternate", "all", "quarter", "none"

        self.ConvBlock1 = conv_block(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])  
        self.ConvBlock2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
        self.ConvBlock5 = conv_block(64, 32, [2, 1, 1], stride=[2, 1, 1], padding=0)
        self.ConvBlock6 = conv_block(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0], activation='elu')

        # Temporal Difference Mamba Blocks
        # We'll create blocks using _build_block which decides whether to return a MambaLayer or a DiffMambaLayer
        # Slow Stream -> channels 64
        self._layer_build_counter = 0
        self.Block1 = self._build_block(64, theta)
        self.Block2 = self._build_block(64, theta)
        self.Block3 = self._build_block(64, theta)
        # Fast Stream -> channels 32
        self.Block4 = self._build_block(32, theta)
        self.Block5 = self._build_block(32, theta)
        self.Block6 = self._build_block(32, theta)

        # Upsampling
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(96, 48, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(48),
            nn.ELU(),
        )

        self.ConvBlockLast = nn.Conv3d(48, 1, [1, 1, 1], stride=1, padding=0)
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.fuse_1 = LateralConnection(fast_channels=32, slow_channels=64)
        self.fuse_2 = LateralConnection(fast_channels=32, slow_channels=64)

        self.drop_1 = nn.Dropout(drop_rate1)
        self.drop_2 = nn.Dropout(drop_rate1)
        self.drop_3 = nn.Dropout(drop_rate2)
        self.drop_4 = nn.Dropout(drop_rate2)
        self.drop_5 = nn.Dropout(drop_rate2)
        self.drop_6 = nn.Dropout(drop_rate2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def _should_make_diff(self, idx, total_layers):
        """Decide whether to make block idx a Diff block given settings."""
        mode = self.diffmamba_settings
        if not self.diffmamba or mode == "none":
            return False
        if mode == "all":
            return True
        if mode == "quarter":
            # convert last quarter (approximation)
            # here idx is a build counter; we approximate "last quarter" using total_layers
            return idx >= (total_layers - (total_layers // 4))
        if mode == "alternate":
            return (idx % 2) == 0
        return False

    def _build_block(self, channels, theta):
        """
        Build either a MambaLayer or a DiffMambaLayer depending on self.diffmamba settings.
        We maintain a simple build counter to decide alternation / positions.
        """
        idx = self._layer_build_counter
        # We'll estimate total blocks as 6 for this architecture (3 slow + 3 fast) to support quarter logic
        total_blocks_est = 6
        make_diff = self._should_make_diff(idx, total_blocks_est)
        self._layer_build_counter += 1

        if make_diff:
            # Use slightly smaller state size for the diff mixers; re-use channels as d_model
            # set d_state to a default but ensure Diff uses smaller internal d_state
            d_state = 16
            d_conv = 4
            layer = nn.Sequential(
                CDC_T(channels, channels, theta=theta),
                nn.BatchNorm3d(channels),
                nn.ReLU(),
                DiffMambaLayer(dim=channels, d_state=d_state, d_conv=d_conv, expand=1, layer_idx=idx),
                ChannelAttention3D(in_channels=channels, reduction=2),
            )
        else:
            # regular Mamba block
            d_state = 16
            d_conv = 4
            layer = nn.Sequential(
                CDC_T(channels, channels, theta=theta),
                nn.BatchNorm3d(channels),
                nn.ReLU(),
                MambaLayer(dim=channels, d_state=d_state, d_conv=d_conv, expand=2, layer_idx=idx),
                ChannelAttention3D(in_channels=channels, reduction=2),
            )
        return layer
    
    def forward(self, x): 
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x) 
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)  
        x = self.MaxpoolSpa(x) 
    
        # Process streams
        s_x = self.ConvBlock4(x) # Slow stream 
        f_x = self.ConvBlock5(x) # Fast stream 

        # First set of blocks and fusion
        s_x1 = self.Block1(s_x)
        s_x1 = self.MaxpoolSpa(s_x1)
        s_x1 = self.drop_1(s_x1)

        f_x1 = self.Block4(f_x)
        f_x1 = self.MaxpoolSpa(f_x1)
        f_x1 = self.drop_2(f_x1)

        s_x1 = self.fuse_1(s_x1,f_x1) # LateralConnection

        # Second set of blocks and fusion
        s_x2 = self.Block2(s_x1)
        s_x2 = self.MaxpoolSpa(s_x2)
        s_x2 = self.drop_3(s_x2)
        
        f_x2 = self.Block5(f_x1)
        f_x2 = self.MaxpoolSpa(f_x2)
        f_x2 = self.drop_4(f_x2)

        s_x2 = self.fuse_2(s_x2,f_x2) # LateralConnection
        
        # Third blocks and upsampling
        s_x3 = self.Block3(s_x2) 
        s_x3 = self.upsample1(s_x3) 
        s_x3 = self.drop_5(s_x3)

        f_x3 = self.Block6(f_x2)
        f_x3 = self.ConvBlock6(f_x3) 
        f_x3 = self.drop_6(f_x3)

        # Final fusion and upsampling
        x_fusion = torch.cat((f_x3, s_x3), dim=1) 
        x_final = self.upsample2(x_fusion) 

        x_final = self.poolspa(x_final)
        x_final = self.ConvBlockLast(x_final)

        rPPG = x_final.view(-1, length)

        return rPPG