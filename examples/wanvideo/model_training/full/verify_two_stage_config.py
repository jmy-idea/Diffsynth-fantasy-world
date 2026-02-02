#!/usr/bin/env python3
"""
Verify Fantasy World two-stage training configuration.

This script checks:
1. Which modules are trainable in Stage 1
2. Which modules are trainable in Stage 2
3. Parameter counts for each stage
"""

import sys
import os

# Add project root to path
project_root = "/ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from diffsynth.models.wan_video_dit import WanModel


def count_parameters(model, requires_grad_only=False):
    """Count parameters in a model or module."""
    if requires_grad_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def check_stage_configuration(stage):
    """Check which modules are trainable in a given stage."""
    print(f"\n{'='*80}")
    print(f"Checking Fantasy World {stage.upper()} Configuration")
    print(f"{'='*80}\n")
    
    # Create a minimal WanModel for testing
    # Use Wan2.1-1.3B configuration parameters
    model = WanModel(
        dim=1536,
        in_dim=16,
        ffn_dim=6144,
        out_dim=16,
        text_dim=4096,
        freq_dim=512,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=24,
        num_layers=30,  # 12 PCB + 18 IRG
        has_image_input=True,
        has_image_pos_emb=False,
    )
    
    # Enable Fantasy World mode with specified stage
    print(f"Enabling Fantasy World mode (training_stage={stage})...")
    model.enable_fantasy_world_mode(split_layer=12, training_stage=stage)
    print()
    
    # Check each module group
    module_groups = {
        "Original Blocks (30 layers)": model.blocks,
        "Latent Bridge": model.latent_bridge,
        "Geometry Blocks (18 layers)": model.geo_blocks,
        "Pose Encoder": model.pose_enc,
        "Camera Token": [model.token_camera],
        "Register Tokens": [model.tokens_register],
        "Camera Head": model.head_camera,
        "Depth Head": model.head_depth,
        "Point Head": model.head_point,
        "Camera Adapters (12 modules)": [a for a in model.camera_adapters if a is not None],
        "IRG Cross-Attentions (18 modules)": model.irg_cross_attns,
    }
    
    results = []
    total_params = 0
    total_trainable = 0
    
    for name, modules in module_groups.items():
        if isinstance(modules, list):
            if not modules:
                continue
            # Special handling for parameter list
            if isinstance(modules[0], torch.nn.Parameter):
                num_params = sum(p.numel() for p in modules)
                num_trainable = sum(p.numel() for p in modules if p.requires_grad)
            else:
                num_params = sum(count_parameters(m) for m in modules)
                num_trainable = sum(count_parameters(m, requires_grad_only=True) for m in modules)
        else:
            num_params = count_parameters(modules)
            num_trainable = count_parameters(modules, requires_grad_only=True)
        
        total_params += num_params
        total_trainable += num_trainable
        
        is_trainable = num_trainable > 0
        status = "✅ TRAINABLE" if is_trainable else "❄️  FROZEN"
        
        results.append({
            "name": name,
            "params": num_params,
            "trainable": num_trainable,
            "status": status,
        })
    
    # Print table
    print(f"{'Module':<40} {'Total Params':<15} {'Trainable':<15} {'Status'}")
    print(f"{'-'*80}")
    
    for r in results:
        params_str = f"{r['params']/1e6:.2f}M"
        trainable_str = f"{r['trainable']/1e6:.2f}M" if r['trainable'] > 0 else "-"
        print(f"{r['name']:<40} {params_str:<15} {trainable_str:<15} {r['status']}")
    
    print(f"{'-'*80}")
    print(f"{'TOTAL':<40} {total_params/1e6:.2f}M       {total_trainable/1e6:.2f}M")
    print()
    
    return {
        "stage": stage,
        "total_params": total_params,
        "total_trainable": total_trainable,
        "results": results,
    }


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("Fantasy World Two-Stage Training Configuration Verification")
    print("="*80)
    
    # Check both stages
    stage1_info = check_stage_configuration("stage1")
    stage2_info = check_stage_configuration("stage2")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"Stage 1 (Latent Bridging):")
    print(f"  - Trainable parameters: {stage1_info['total_trainable']/1e6:.2f}M")
    print(f"  - Frozen parameters: {(stage1_info['total_params'] - stage1_info['total_trainable'])/1e6:.2f}M")
    print()
    
    print(f"Stage 2 (Unified Co-Optimization):")
    print(f"  - Trainable parameters: {stage2_info['total_trainable']/1e6:.2f}M")
    print(f"  - Frozen parameters: {(stage2_info['total_params'] - stage2_info['total_trainable'])/1e6:.2f}M")
    print()
    
    additional_trainable = stage2_info['total_trainable'] - stage1_info['total_trainable']
    print(f"Additional trainable in Stage 2: {additional_trainable/1e6:.2f}M")
    print(f"  (Camera Adapters + IRG Cross-Attentions)")
    print()
    
    print("✅ Configuration verification complete!")
    print()
    print("Next steps:")
    print("  1. Update DATA_DIR in train_fantasy_world_stage1.sh")
    print("  2. Run: bash examples/wanvideo/model_training/full/train_fantasy_world_stage1.sh")
    print("  3. After Stage 1 completes, run train_fantasy_world_stage2.sh")
    print()


if __name__ == "__main__":
    main()
