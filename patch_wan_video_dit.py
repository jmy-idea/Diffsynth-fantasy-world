
import os

target_file = "/ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world/diffsynth/models/wan_video_dit.py"

extra_classes = """
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
        # features_list: [L1, L2, L3, L4] tensors of shape [B, L, C] where L=F*H/8*W/8
        # We need to reshape them to [B*F, C, H/8, W/8] or similar.
        # WanVideo patch size (1,2,2)? or (1,8,8)? 
        # DiT patchify gives grid size (f_g, h_g, w_g).
        # We need to assume some shape.
        # features are [B, T*H*W, C].
        
        out = []
        for feat in features_list:
            # Assuming h, w are the grid dimensions?
            b, l, c = feat.shape
            # Here we assume 'h' and 'w' passed to forward are the latent spatial dimensions.
            # And F is inferred.
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
        
        # Reshape back to [B, F, C_out, H_high, W_high]?
        # DPT output is upsampled by 2 in output_head.
        # Input to DPT is latent resolution (h, w).
        # Output is 2 * h, 2 * w.
        # Depending on DPTHead conf.
        
        # Returns [B, F, 3, H_out, W_out]
        return out.view(b, f, -1, out.shape[-2], out.shape[-1])


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

"""

method_code = """
    def enable_fantasy_world_mode(self):
        self.enable_fantasy_world = True
        
        # 1. Pose Encoder (Input: 9D camera params -> Dim)
        self.pose_enc = PoseEncoder(in_dim=9, out_dim=self.dim)
        
        # 2. Camera Adapters (Layers 0-23)
        self.camera_adapters = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.dim, self.dim)
            ) for _ in range(24)
        ])
        
        # 3. Geometry Projector
        self.geo_projector = nn.Linear(self.dim, self.dim)
        
        # 4. Geometry Blocks (Clone Layers 16-39)
        self.geo_blocks = nn.ModuleList([
            copy.deepcopy(self.blocks[i]) for i in range(16, min(len(self.blocks), 40))
        ]) # Indices 16..39
        
        # 5. IRG Cross Attention (24 layers, 16-39)
        self.irg_cross_attns = nn.ModuleList([
            MMBiCrossAttention(self.dim, self.num_heads) 
            for _ in range(len(self.geo_blocks))
        ])
        
        # 6. DPT Head
        self.dpt_head = DPTHead(
            dim_in=self.dim, 
            out_channels=[256, 512, 1024, 1024], 
            output_dim=3 # For points
        )
"""

with open(target_file, "r") as f:
    lines = f.readlines()

# Insert Method
new_lines = []
inserted_method = False

# We append classes at the end first?
# or before WanModel?
# If we append at end, WanModel can't see them if we use type hints? Python doesn't care for untyped init.
# But cleaner to put before WanModel.

wan_model_idx = -1
for i, line in enumerate(lines):
    if "class WanModel(torch.nn.Module):" in line:
        wan_model_idx = i
        break

if wan_model_idx == -1:
    print("Could not find class WanModel")
    exit(1)

# Split lines
pre_lines = lines[:wan_model_idx]
post_lines = lines[wan_model_idx:]

# Insert classes
pre_lines.append(extra_classes)

# Insert method into post_lines (WanModel)
# Look for __init__ end or forward.
# forward_idx
forward_idx = -1
for i, line in enumerate(post_lines):
    if "def forward(self," in line:
        forward_idx = i
        break

if forward_idx != -1:
    post_lines.insert(forward_idx, method_code + "\n")
else:
    post_lines.append(method_code)

final_content = "".join(pre_lines + post_lines)

with open(target_file, "w") as f:
    f.write(final_content)
