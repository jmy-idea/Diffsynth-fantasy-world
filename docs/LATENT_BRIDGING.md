"""
Fantasy World: Latent Bridging Stage Architecture

Stage 1 - Latent Bridging
========================

Purpose:
--------
Map video branch features (from block 16 of Wan2.1) to a geometry-aligned latent space
that can be effectively processed by the geometry branch.

Architecture:
-------------

INPUT: Hidden features from block 16 (split_layer)
       Shape: [B, L, D] where B=batch, L=sequence_length, D=hidden_dim

    ↓

LIGHTWEIGHT TRANSFORMER ADAPTER (LatentBridgeAdapter)
├── Layer 1:
│   ├── LayerNorm([B, L, D])
│   ├── Multi-Head Self-Attention (8 heads)
│   │   Purpose: Refine and aggregate video features
│   │   Allows temporal communication across latent tokens
│   ├── Residual Connection: x = x + Attn(LN(x))
│   │
│   ├── LayerNorm([B, L, D])
│   ├── Feed-Forward Network (FFN)
│   │   dim -> 4*dim -> dim (GELU activation)
│   │   Purpose: Non-linear feature transformation
│   └── Residual Connection: x = x + FFN(LN(x))
│
└── Layer 2: (same as Layer 1, repeated)

OUTPUT: Geometry-aligned latent
        Shape: [B, L, D]
        These features are more suitable for geometry prediction

Design Rationale:
-----------------

1. Why Transformer instead of just Linear?
   - Video features contain complex temporal relationships
   - Self-attention allows the adapter to selectively focus on relevant parts
   - Better capacity to learn geometry-aligned representations
   - More expressive than a single linear layer

2. Why only 2 layers?
   - "Lightweight" adapter - computational efficiency
   - Too many layers could hurt training dynamics
   - Empirically, 2 layers provides good balance

3. Why Pre-Norm?
   - Improves training stability
   - Prevents gradient explosion in deep networks
   - Standard in modern Transformers (e.g., Vision Transformer)

4. Why Multi-Head Attention (8 heads)?
   - 8 heads is a good balance for "lightweight" design
   - Each head can learn different aspects of geometry
   - Common choice for adapter modules

Connection to Rest of Framework:
--------------------------------

Block 16 Features (from video branch)
         ↓
    LatentBridgeAdapter (Stage 1)
         ↓
   Geometry-aligned latent
         ↓
    Used in Geometry Branch:
    - Depth Head: Predicts depth maps [B, T, 1, H, W]
    - Point Head: Predicts point clouds [B, T, 3, H, W] + confidence
    - Camera Head: Predicts camera parameters [B, T, 9]
    - IRG Blocks: Further refined through attention with video features

Paper Reference:
-----------------
"Fantasy World: Stage 1: Latent Bridging - we select hidden features from 
block 16 of Wan2.1 and feed them to the geometry branch through a lightweight 
transformer adapter that maps to a geometry-aligned latent space."

Implementation in Code:
-----------------------
File: wan_video_dit.py
Class: LatentBridgeAdapter
Integration: WanModel.enable_fantasy_world_mode()
             -> self.latent_bridge = LatentBridgeAdapter(...)

Usage in Pipeline:
-----------------
During forward pass (to be implemented in wan_video.py):
    
    features_block16 = x  # [B, L, D] from block 16
    
    geo_latent = self.dit.latent_bridge(features_block16)
    # geo_latent: [B, L, D] - geometry-aligned features
    
    # Then used for geometry head predictions
    depth = self.dit.head_depth(geo_latent, ...)
    points = self.dit.head_point(geo_latent, ...)
    camera = self.dit.head_camera(geo_latent)

Hyperparameters (can be tuned):
-------------------------------
- num_heads: 8 (default) - multi-head attention heads
- ffn_dim: dim * 4 (default) - feed-forward hidden dimension
- num_layers: 2 (default) - number of transformer blocks
  * Increase for more expressive adapter (but slower training)
  * Decrease for faster inference (but less expressive)

Training Tips:
--------------
1. Initialize latent_bridge weights properly (already done with randn * 0.02)
2. Gradients flow through adapter during geometry loss backprop
3. Consider using gradient checkpointing if memory is tight:
   - torch.utils.checkpoint.checkpoint(latent_bridge, features_block16)
4. Monitor loss curves - adapter should help geometry branch learn faster

Ablation Study Suggestions:
--------------------------
1. Compare with just Linear projection (baseline)
2. Vary num_layers: [1, 2, 3, 4]
3. Vary num_heads: [4, 8, 16]
4. Compare different activations (GELU vs ReLU)
5. Test with/without pre-norm
"""
