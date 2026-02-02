#!/usr/bin/env python3
"""
Fantasy World Inference Script

This script demonstrates how to:
1. Load base Wan2.1 model
2. Load trained Fantasy World checkpoint (geometry modules only)
3. Generate camera-controlled videos with geometry-aware generation

Usage:
    python fantasy_world_inference.py \
        --checkpoint outputs/fantasy_world_stage1/step-2.safetensors \
        --prompt "Your prompt here" \
        --camera_trajectory camera_trajectories/orbit_360deg.txt \
        --output output.mp4
"""

import torch
import argparse
from PIL import Image
from pathlib import Path
from safetensors.torch import load_file as load_safetensors

from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


def load_fantasy_world_checkpoint(pipe, checkpoint_path, training_stage="stage1"):
    """
    Load Fantasy World checkpoint into pipeline.
    
    Args:
        pipe: WanVideoPipeline instance
        checkpoint_path: Path to checkpoint (e.g., outputs/fantasy_world_stage1/step-2.safetensors)
        training_stage: "stage1" or "stage2"
    """
    print(f"\n{'='*80}")
    print(f"Loading Fantasy World Checkpoint")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training stage: {training_stage}")
    
    # 1. Enable Fantasy World mode
    print("\nEnabling Fantasy World mode on DiT...")
    if not hasattr(pipe.dit, 'enable_fantasy_world_mode'):
        raise RuntimeError("DiT does not support Fantasy World mode. Check your model version.")
    
    pipe.dit.enable_fantasy_world_mode(split_layer=12, training_stage=training_stage)
    print("✓ Fantasy World mode enabled")
    
    # 2. Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    state_dict = load_safetensors(checkpoint_path)
    
    # 3. Load into DiT (strict=False allows missing keys for frozen blocks)
    missing_keys, unexpected_keys = pipe.dit.load_state_dict(state_dict, strict=False)
    
    print(f"\n{'='*80}")
    print(f"Checkpoint Loading Summary")
    print(f"{'='*80}")
    print(f"Total keys loaded: {len(state_dict)}")
    print(f"Missing keys (frozen blocks): {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Check if critical geometry modules are loaded
    geo_modules = [k for k in state_dict.keys() if 'geo_blocks' in k]
    print(f"\nGeometry modules loaded: {len(geo_modules)} keys")
    
    if len(geo_modules) == 0:
        print("⚠️  WARNING: No geometry modules found in checkpoint!")
    else:
        print("✓ Geometry modules successfully loaded")
    
    # Missing keys should be the frozen blocks (blocks.0 to blocks.29)
    frozen_blocks = [k for k in missing_keys if k.startswith('blocks.')]
    print(f"Frozen blocks (from base model): {len(frozen_blocks)} parameters")
    
    if len(unexpected_keys) > 0:
        print(f"\n⚠️  WARNING: {len(unexpected_keys)} unexpected keys:")
        for key in unexpected_keys[:5]:  # Show first 5
            print(f"  - {key}")
        if len(unexpected_keys) > 5:
            print(f"  ... and {len(unexpected_keys) - 5} more")
    
    print(f"\n{'='*80}")
    print("✓ Fantasy World checkpoint loaded successfully!")
    print(f"{'='*80}\n")
    
    return pipe


def create_camera_trajectory_file(trajectory_type="orbit", output_path="camera_trajectory.txt", 
                                 num_frames=21, width=832, height=480):
    """
    Create a sample camera trajectory file for testing.
    
    Camera format: Each line has 19 values
    - [0]: Frame index/sequence number
    - [1:5]: Camera intrinsics (fx, fy, cx, cy)
    - [5:7]: Distortion parameters (k1, k2, both default to 0)
    - [7:19]: World-to-camera matrix flattened (3x4 = 12 values)
    
    Args:
        trajectory_type: "orbit", "forward", "left_right"
        output_path: Where to save the trajectory file
        num_frames: Number of frames
        width: Image width (for intrinsics)
        height: Image height (for intrinsics)
    """
    import numpy as np
    
    # Default camera intrinsics (typical values for a pinhole camera)
    focal_length = max(width, height)  # Simple estimate
    fx = fy = focal_length
    cx = width / 2.0
    cy = height / 2.0
    
    # Distortion parameters (k1, k2) - set to 0 for no distortion
    k1, k2 = 0.0, 0.0
    
    if trajectory_type == "orbit":
        # Orbit around center (360 degrees)
        angles = np.linspace(0, 2*np.pi, num_frames)
        radius = 2.0
        
        poses = []
        for frame_idx, angle in enumerate(angles):
            # Camera position (circular motion)
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0.0
            
            # Look at center (0, 0, 0)
            # Create rotation matrix
            forward = np.array([0, 0, 0]) - np.array([x, y, z])
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(np.array([0, 1, 0]), forward)
            right = right / np.linalg.norm(right)
            
            up = np.cross(forward, right)
            
            # World-to-camera matrix (3x4)
            w2c = np.eye(4)
            w2c[:3, 0] = right
            w2c[:3, 1] = up
            w2c[:3, 2] = forward
            w2c[:3, 3] = [x, y, z]
            
            # Flatten to 12D
            pose_12d = w2c[:3, :].flatten()
            
            # Format: [frame_idx, fx, fy, cx, cy, k1, k2, w2c_00, w2c_01, ..., w2c_23]
            full_cam = np.concatenate([
                [frame_idx],           # Frame index (1 value)
                [fx, fy, cx, cy],      # Intrinsics (4 values)
                [k1, k2],              # Distortion (2 values)
                pose_12d               # Extrinsics (12 values)
            ])
            poses.append(full_cam)
    
    elif trajectory_type == "forward":
        # Move forward (zoom in)
        z_positions = np.linspace(3.0, 1.0, num_frames)
        
        poses = []
        for frame_idx, z in enumerate(z_positions):
            # Identity rotation, changing z position
            w2c = np.eye(4)
            w2c[2, 3] = z
            pose_12d = w2c[:3, :].flatten()
            
            # Format: [frame_idx, fx, fy, cx, cy, k1, k2, w2c_00, ...]
            full_cam = np.concatenate([
                [frame_idx],
                [fx, fy, cx, cy],
                [k1, k2],
                pose_12d
            ])
            poses.append(full_cam)
    
    elif trajectory_type == "left_right":
        # Pan left to right
        x_positions = np.linspace(-1.0, 1.0, num_frames)
        
        poses = []
        for frame_idx, x in enumerate(x_positions):
            w2c = np.eye(4)
            w2c[0, 3] = x
            w2c[2, 3] = 2.0  # Fixed distance
            pose_12d = w2c[:3, :].flatten()
            
            # Format: [frame_idx, fx, fy, cx, cy, k1, k2, w2c_00, ...]
            full_cam = np.concatenate([
                [frame_idx],
                [fx, fy, cx, cy],
                [k1, k2],
                pose_12d
            ])
            poses.append(full_cam)
    
    # Save to file
    poses_array = np.array(poses)  # [T, 19]
    np.savetxt(output_path, poses_array, fmt='%.6f')
    print(f"✓ Created camera trajectory: {output_path}")
    print(f"  Type: {trajectory_type}")
    print(f"  Frames: {num_frames}")
    print(f"  Format: {poses_array.shape[1]} values per frame")
    print(f"    - [0]: Frame index")
    print(f"    - [1:5]: Intrinsics (fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f})")
    print(f"    - [5:7]: Distortion (k1={k1}, k2={k2})")
    print(f"    - [7:19]: w2c matrix (3x4=12 values)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fantasy World Inference")
    
    # Model and checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Fantasy World checkpoint (e.g., outputs/fantasy_world_stage1/step-2.safetensors)")
    parser.add_argument("--training_stage", type=str, default="stage1", choices=["stage1", "stage2"],
                        help="Which training stage the checkpoint is from")
    
    # Input
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt")
    parser.add_argument("--input_image", type=str, default=None,
                        help="Input image for I2V (optional)")
    
    # Camera control
    parser.add_argument("--camera_trajectory", type=str, default=None,
                        help="Path to camera trajectory file (txt with [T, 12] w2c matrices)")
    parser.add_argument("--create_trajectory", type=str, default=None, 
                        choices=["orbit", "forward", "left_right"],
                        help="Create a sample trajectory (for testing)")
    
    # Generation settings
    parser.add_argument("--num_frames", type=int, default=21,
                        help="Number of frames to generate")
    parser.add_argument("--height", type=int, default=480,
                        help="Video height")
    parser.add_argument("--width", type=int, default=832,
                        help="Video width")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of denoising steps")
    
    # Output
    parser.add_argument("--output", type=str, default="fantasy_world_output.mp4",
                        help="Output video path")
    parser.add_argument("--fps", type=int, default=15,
                        help="Output video FPS")
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*80)
    print("Fantasy World Inference")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames}")
    print("="*80 + "\n")
    
    # 1. Create or load camera trajectory
    if args.create_trajectory:
        print(f"Creating sample trajectory: {args.create_trajectory}")
        trajectory_path = f"camera_trajectory_{args.create_trajectory}.txt"
        create_camera_trajectory_file(
            args.create_trajectory, trajectory_path, 
            num_frames=args.num_frames, width=args.width, height=args.height
        )
        args.camera_trajectory = trajectory_path
    
    if not args.camera_trajectory:
        print("⚠️  No camera trajectory provided. Creating default orbit trajectory...")
        args.camera_trajectory = create_camera_trajectory_file(
            "orbit", "camera_trajectory_default.txt",
            num_frames=args.num_frames, width=args.width, height=args.height
        )
    
    # 2. Initialize base pipeline
    print("\n" + "="*80)
    print("Initializing Base Pipeline")
    print("="*80 + "\n")
    
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", 
                       origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", 
                       origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", 
                       origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", 
                       origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", 
                                     origin_file_pattern="google/umt5-xxl/"),
    )
    
    print("✓ Base pipeline loaded")
    
    # 3. Load Fantasy World checkpoint
    pipe = load_fantasy_world_checkpoint(pipe, args.checkpoint, args.training_stage)
    
    # 4. Load input image (if provided)
    if args.input_image:
        print(f"\nLoading input image: {args.input_image}")
        input_image = Image.open(args.input_image)
        print(f"✓ Input image loaded: {input_image.size}")
    else:
        # Download example image
        print("\nNo input image provided. Downloading example...")
        dataset_snapshot_download(
            dataset_id="DiffSynth-Studio/examples_in_diffsynth",
            local_dir="./",
            allow_file_pattern=f"data/examples/wan/input_image.jpg"
        )
        input_image = Image.open("data/examples/wan/input_image.jpg")
        print(f"✓ Example image loaded: {input_image.size}")
    
    # 5. Generate video
    print("\n" + "="*80)
    print("Generating Video")
    print("="*80 + "\n")
    
    try:
        video = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            tiled=True,
            input_image=input_image,
            pose_file_path=args.camera_trajectory,  # Fantasy World uses pose_file_path
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
        )
        
        # 6. Save video
        print(f"\nSaving video to: {args.output}")
        save_video(video, args.output, fps=args.fps, quality=5)
        
        print("\n" + "="*80)
        print("✓ Video generation complete!")
        print("="*80)
        print(f"Output saved to: {args.output}")
        print(f"Duration: {args.num_frames / args.fps:.2f} seconds")
        print(f"Resolution: {args.width}x{args.height}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during generation:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
