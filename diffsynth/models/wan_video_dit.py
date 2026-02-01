import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x
    

# ===========================================================================
# fantasy world
# --- DPT Head Utilities ---

def inverse_log_transform(y):
    return torch.sign(y) * (torch.expm1(torch.abs(y)))

def activate_head(out, activation="inv_log", conf_activation="expp1"):
    # out: B, C, H, W (or B S C H W flattened?)
    # DPTHead output conv uses output_dim.
    # We assume out is [B*S, output_dim, H, W] inside DPTHead, 
    # but DPTHead.forward reshapes to [B, S, C, H, W].
    # So out here is [B, S, C, H, W]???
    # No, inside DPTHead.forward:
    # out = self.scratch.output_conv2(out) -> [B*S, C, H, W]
    # preds, conf = activate_head(out, ...)
    
    fmap = out.permute(0, 2, 3, 1) # B*S, H, W, C
    xyz = fmap[..., :-1]
    conf = fmap[..., -1:] # Keep dim? VGGT uses [..., -1] reducing dim.
    # But we want [B, S, C-1, H, W].
    
    # Let's align with VGGT logic:
    # xyz: B*S, H, W, 3
    # conf: B*S, H, W, 1 (if keeping dim)
    
    if activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    else:
        pts3d = xyz
        
    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    else:
        conf_out = conf
        
    # Reshape back to channel first for consistency with PyTorch [B*S, C, H, W]
    pts3d = pts3d.permute(0, 3, 1, 2)
    conf_out = conf_out.permute(0, 3, 1, 2)
    
    return pts3d, conf_out

def create_uv_grid(width, height, aspect_ratio=1.0, dtype=torch.float32, device='cpu'):
    xs = torch.linspace(0, 1, width, dtype=dtype, device=device)
    ys = torch.linspace(0, 1, height, dtype=dtype, device=device)
    # Correcting for aspect ratio if needed, but standard UV is 0-1
    # VGGT logic usually keeps 0-1.
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([grid_x, grid_y], dim=0) # 2, H, W

def position_grid_to_embed(grid, dim_out):
    # Simple projection or sinusoidal. VGGT uses a projection.
    # Assuming simple projection for DPT:
    # We can use a Conv2d or similar. But here let's assume DPTHead handles it.
    # In VGGT `position_grid_to_embed` calls `get_2d_sincos_pos_embed` or similar?
    # Actually DPTHead code called it.
    # To save space, let's implement a learnable or fixed embedding.
    # For now, let's omit complex pos embed if feature_only is sufficient or use a simple Sinusoidal.
    # Placeholder:
    return torch.zeros(1, dim_out, grid.shape[1], grid.shape[2], device=grid.device)

# --- DPT Head ---

class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, has_residual=True):
        super().__init__()
        self.has_residual = has_residual
        self.resConfUnit1 = nn.Sequential(
            activation, nn.Conv2d(features, features, 3, 1, 1),
            activation, nn.Conv2d(features, features, 3, 1, 1)
        ) if has_residual else None
        self.resConfUnit2 = nn.Sequential(
            activation, nn.Conv2d(features, features, 3, 1, 1),
            activation, nn.Conv2d(features, features, 3, 1, 1)
        )
        self.out_conv = nn.Conv2d(features, features, 1, 1, 0)
        
    def forward(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        if size is not None:
             output = F.interpolate(output, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(output)

class DPTHead(nn.Module):
    def __init__(self, dim_in, output_dim=4, features=256, intermediate_layer_idx=[7, 11, 17, 23]):
        super().__init__()
        self.intermediate_layer_idx = intermediate_layer_idx
        self.norm = nn.LayerNorm(dim_in)
        
        out_channels = [256, 512, 1024, 1024]
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, 1) for oc in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, 4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, 2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, 2, 1)
        ])
        
        # Scratch (RefineNet)
        self.refinenet4 = FeatureFusionBlock(features, nn.ReLU(True), has_residual=False)
        self.refinenet3 = FeatureFusionBlock(features, nn.ReLU(True))
        self.refinenet2 = FeatureFusionBlock(features, nn.ReLU(True))
        self.refinenet1 = FeatureFusionBlock(features, nn.ReLU(True))
        
        self.layer1_rn = nn.Conv2d(out_channels[0], features, 3, 1, 1)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, 3, 1, 1)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, 3, 1, 1)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, 3, 1, 1)

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, 1, 1),
            nn.Conv2d(features // 2, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, output_dim, 1)
        )

    def forward(self, features_list, patch_h, patch_w):
        # features_list: List of [B, S, D]
        # We need to reshape to [B*S, D, H, W]
        out = []
        for i, idx in enumerate(self.intermediate_layer_idx):
            x = features_list[i] # Already selected
            b, s, d = x.shape
            x = x.reshape(b*s, d, patch_h, patch_w) # Assuming already patches
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
            
        l1, l2, l3, l4 = out
        l1 = self.layer1_rn(l1)
        l2 = self.layer2_rn(l2)
        l3 = self.layer3_rn(l3)
        l4 = self.layer4_rn(l4)

        o = self.refinenet4(l4, size=l3.shape[2:])
        o = self.refinenet3(o, l3, size=l2.shape[2:])
        o = self.refinenet2(o, l2, size=l1.shape[2:])
        o = self.refinenet1(o, l1)
        
        return self.output_conv(o)
# ===========================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import copy

# --------------------------
# Fantasy World Components
# --------------------------

def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)

class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn, groups=1):
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return self.skip_add.add(out, x)

class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None, has_residual=True, groups=1):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups)
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
            
        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand:
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch

def _make_fusion_block(features, use_bn=False, size=None, has_residual=True):
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
    )

class DPTHead(nn.Module):
    def __init__(self, dim_in: int, output_dim: int = 3, features: int = 256, out_channels: List[int] = [256, 512, 1024, 1024]):
        super().__init__()
        # Simplified DPTHead using only necessary parts
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])
        
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        
        head_features_1 = features
        self.output_head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, output_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, features_list, h, w):
        out = []
        for feat in features_list:
            b, l, c = feat.shape
            f = l // (h*w)
            feat = feat.view(b, f, h, w, c).permute(0, 1, 4, 2, 3).flatten(0, 1) # [B*F, C, h, w]
            out.append(feat)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.output_head(path_1)
        
        # Activate Head
        pts, conf = activate_head(out)
        
        # Reshape to [B, F, C, H, W]
        pts = pts.view(b, f, -1, pts.shape[-2], pts.shape[-1])
        conf = conf.view(b, f, -1, conf.shape[-2], conf.shape[-1])
        
        return pts, conf


class PoseEncoder(nn.Module):
    def __init__(self, in_dim=9, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class MMBiCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # dim=2048, num_heads=16
        # Use existing CrossAttention
        self.attn1 = CrossAttention(dim, num_heads)
        self.attn2 = CrossAttention(dim, num_heads)
        # Gating
        self.gate1 = nn.Parameter(torch.zeros(1))
        self.gate2 = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, x_geo):
        # x: Video, x_geo: Geometry
        # x_new = x + tanh(gate1) * CrossAttn(x, x_geo)
        # x_geo_new = x_geo + tanh(gate2) * CrossAttn(x_geo, x)
        
        out1 = self.attn1(x, x_geo)
        out2 = self.attn2(x_geo, x)
        
        x = x + torch.tanh(self.gate1) * out1
        x_geo = x_geo + torch.tanh(self.gate2) * out2
        
        return x, x_geo


class GeoDiTBlock(DiTBlock):
    def __init__(self, original_block: DiTBlock, dim: int):
        super().__init__(
            has_image_input=original_block.cross_attn.has_image_input,
            dim=original_block.dim,
            num_heads=original_block.num_heads,
            ffn_dim=original_block.ffn_dim,
            eps=original_block.norm1.eps
        )
        self.load_state_dict(original_block.state_dict())
        self.adapter = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        # Freeze original params? controlled by enable_fantasy_world_mode loop.

    def forward(self, x, context, t_mod, freqs, plucker_emb):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        
        # Original Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        
        # Self Attn
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        
        # [Fantasy World] Inject Plucker Embedding
        if plucker_emb is not None:
             # Assume plucker_emb is [B, 1, C] or [B, C] -> broadcast
             # Or if it represents spatial cues? usually standard adapter broadacast.
             x = x + self.adapter(plucker_emb)

        # Cross Attn
        x = x + self.cross_attn(self.norm3(x), context)
        
        # FFN
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class CameraHead(nn.Module):
    def __init__(self, in_dim, out_dim=9):
        super().__init__()
        # Lightweight 1D convolution for temporal upsampling and prediction
        # We assume 4x upsampling to match T = (t-1)*4 + 1 approximately
        self.conv_in = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x: [B, T_latent, C]
        x = x.transpose(1, 2) # [B, C, T_latent]
        
        # Temporal Upsampling (approx 4x)
        # We use interpolate linear
        t_in = x.shape[-1]
        t_out = (t_in - 1) * 4 + 1
        x = torch.nn.functional.interpolate(x, size=t_out, mode='linear', align_corners=True)
        
        x = self.conv_in(x)
        x = self.act(x)
        x = self.conv_out(x)
        
        x = x.transpose(1, 2) # [B, T_out, 9]
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        self.enable_fantasy_world = False

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def enable_fantasy_world_mode(self, split_layer=12):
        self.enable_fantasy_world = True
        
        # 1. Pose Encoder (for Plucker Embedding)
        self.pose_enc = PoseEncoder(in_dim=9, out_dim=self.dim)
        
        # 2. Tokens (Camera + Registers)
        # "Concatenate one learned camera token and four register tokens"
        self.token_camera = nn.Parameter(torch.randn(1, 1, self.dim))
        self.tokens_register = nn.Parameter(torch.randn(1, 4, self.dim))

        # 3. Heads
        # Camera Head
        self.head_camera = CameraHead(self.dim, out_dim=9)
        
        # Depth Head (D = 1 channel)
        # Using 3D DPT Head equivalent (spatial DPT + temporal handling in output)
        self.head_depth = DPTHead(
            dim_in=self.dim, 
            out_channels=[256, 512, 1024, 1024], 
            output_dim=1
        )
        
        # Point Map Head (P = 3 channels + 1 confidence)
        self.head_point = DPTHead(
            dim_in=self.dim, 
            out_channels=[256, 512, 1024, 1024], 
            output_dim=4
        )

        # 4. Camera Adapters (Video Branch Injection)
        self.camera_adapters = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.dim, self.dim)
            ) for _ in range(len(self.blocks))
        ])
        
        # 5. Geometry Projector
        self.geo_projector = nn.Linear(self.dim, self.dim)
        
        # 6. Geometry Blocks
        self.geo_blocks = nn.ModuleList([
            GeoDiTBlock(self.blocks[i], self.dim) 
            for i in range(split_layer, len(self.blocks))
        ])
        
        # 7. IRG Cross Attention
        self.irg_cross_attns = nn.ModuleList([
            MMBiCrossAttention(self.dim, self.blocks[0].num_heads) 
            for _ in range(len(self.geo_blocks))
        ])
        
        for param in self.blocks.parameters():
            param.requires_grad = False


    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
