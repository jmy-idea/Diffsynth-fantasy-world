#!/usr/bin/env python3
"""
Fantasy World Training Script

Trains Wan2.1 DiT with geometry-aware heads (depth, point cloud, camera).
Based on the Fantasy World paper: https://arxiv.org/abs/XXX

Usage:
    python train_fantasy_world.py \
        --model_paths /path/to/wan2.1_dit.safetensors \
        --dataset_base_path /path/to/dataset \
        --dataset_metadata_path /path/to/metadata.json \
        --output_path /path/to/checkpoints \
        --task fantasy_world \
        --trainable_models dit
"""

import torch
import os
import argparse
import accelerate
import warnings
from diffsynth.core.data.fantasy_world_dataset import FantasyWorldDataset
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.diffusion.loss import FantasyWorldLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FantasyWorldTrainingModule(DiffusionTrainingModule):
    """
    Training module for Fantasy World with geometry supervision.
    
    Extends WanTrainingModule with:
    - Fantasy World mode activation
    - Geometry data handling (depth, points, camera)
    - Combined loss computation
    """
    
    def __init__(
        self,
        model_paths=None, 
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, 
        lora_target_modules="", 
        lora_rank=32, 
        lora_checkpoint=None,
        preset_lora_path=None, 
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        # Fantasy World specific
        enable_depth_head=True,
        enable_point_head=True,
        enable_camera_head=True,
        geo_loss_weight=1.0,
    ):
        super().__init__()
        
        # Force gradient checkpointing
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is disabled. Enabling to prevent OOM.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths, 
            fp8_models=fp8_models, offload_models=offload_models, device=device
        )
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") 
            if tokenizer_path is None 
            else ModelConfig(tokenizer_path)
        )
        
        # Initialize pipeline
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, 
            device=device, 
            model_configs=model_configs, 
            tokenizer_config=tokenizer_config
        )
        
        # Split for training
        self.pipe = self.split_pipeline_units("fantasy_world", self.pipe, trainable_models, lora_base_model)
        
        # Enable training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task="fantasy_world",
        )
        
        # Enable Fantasy World mode on DiT
        if hasattr(self.pipe, 'dit') and self.pipe.dit is not None:
            if hasattr(self.pipe.dit, 'enable_fantasy_world_mode'):
                print("[FantasyWorld] Enabling Fantasy World mode on DiT...")
                self.pipe.dit.enable_fantasy_world_mode(
                    enable_depth=enable_depth_head,
                    enable_points=enable_point_head,
                    enable_camera=enable_camera_head,
                )
            else:
                warnings.warn("DiT does not have enable_fantasy_world_mode method!")
        
        # Store configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.fp8_models = fp8_models
        self.geo_loss_weight = geo_loss_weight
        self.enable_depth_head = enable_depth_head
        self.enable_point_head = enable_point_head
        self.enable_camera_head = enable_camera_head
        
    def get_pipeline_inputs(self, data):
        """Prepare inputs for the pipeline, including geometry supervision data."""
        
        inputs_posi = {"prompt": data.get("prompt", "")}
        inputs_nega = {}
        
        # Basic video inputs
        video_frames = data.get("video", [])
        if len(video_frames) == 0:
            raise ValueError("No video frames in data!")
        
        inputs_shared = {
            "input_video": video_frames,
            "height": video_frames[0].size[1],
            "width": video_frames[0].size[0],
            "num_frames": len(video_frames),
            # Training config
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
        }
        
        # Add geometry supervision data
        if self.enable_depth_head and "gt_depth" in data:
            inputs_shared["gt_depth"] = data["gt_depth"]
            
        if self.enable_point_head and "gt_points" in data:
            inputs_shared["gt_points"] = data["gt_points"]
            
        if self.enable_camera_head and "pose_params" in data:
            inputs_shared["pose_params"] = data["pose_params"]
        
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        """Forward pass with combined diffusion + geometry loss."""
        
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        
        # Transfer to device
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        
        # Run pipeline units
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        
        # Compute Fantasy World loss
        inputs_shared, inputs_posi, inputs_nega = inputs
        loss = FantasyWorldLoss(self.pipe, **inputs_shared, **inputs_posi)
        
        return loss


def fantasy_world_parser():
    """Argument parser for Fantasy World training."""
    parser = argparse.ArgumentParser(description="Fantasy World Training Script")
    
    # General training args
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    
    # Model paths
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    
    # Fantasy World specific
    parser.add_argument("--enable_depth_head", default=True, action="store_true", 
                        help="Enable depth prediction head")
    parser.add_argument("--disable_depth_head", dest="enable_depth_head", action="store_false")
    parser.add_argument("--enable_point_head", default=True, action="store_true",
                        help="Enable point cloud prediction head")
    parser.add_argument("--disable_point_head", dest="enable_point_head", action="store_false")
    parser.add_argument("--enable_camera_head", default=True, action="store_true",
                        help="Enable camera prediction head")
    parser.add_argument("--disable_camera_head", dest="enable_camera_head", action="store_false")
    parser.add_argument("--geo_loss_weight", type=float, default=1.0,
                        help="Weight for geometry loss relative to diffusion loss")
    
    # Dataset
    parser.add_argument("--load_depth", default=True, action="store_true")
    parser.add_argument("--load_points", default=True, action="store_true")
    parser.add_argument("--load_camera", default=True, action="store_true")
    
    return parser


def main():
    parser = fantasy_world_parser()
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    )
    
    # Create dataset
    dataset = FantasyWorldDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        load_depth=args.load_depth,
        load_points=args.load_points,
        load_camera=args.load_camera,
    )
    
    print(f"[FantasyWorld] Dataset loaded with {len(dataset)} samples")
    
    # Create training module
    model = FantasyWorldTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        enable_depth_head=args.enable_depth_head,
        enable_point_head=args.enable_point_head,
        enable_camera_head=args.enable_camera_head,
        geo_loss_weight=args.geo_loss_weight,
    )
    
    # Model logger
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    
    # Launch training
    print(f"[FantasyWorld] Starting training...")
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
