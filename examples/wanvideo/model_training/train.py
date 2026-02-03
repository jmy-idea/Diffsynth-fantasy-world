import torch, os, argparse, accelerate, warnings, sys

# Add project root to path to enable diffsynth import
# Find the project root by going up from this script location
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 5 levels: examples/wanvideo/model_training/train.py -> Diffsynth-fantasy-world
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.core.data.fantasy_world_operators import (
    default_depth_operator, default_points_operator, default_camera_operator
)
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.diffusion.loss import FantasyWorldLoss
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        fantasy_world_checkpoint=None,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        
        # Parse Fantasy World training stage (if task is fantasy_world:stage1 or fantasy_world:stage2)
        training_stage = "stage2"  # Default
        if self.task.startswith("fantasy_world"):
            if ":" in self.task:
                stage_str = self.task.split(":")[-1]
                if stage_str in ["stage1", "stage2"]:
                    training_stage = stage_str
            
            if hasattr(self.pipe.dit, "enable_fantasy_world_mode"):
                self.pipe.dit.enable_fantasy_world_mode(training_stage=training_stage)
                
                # Load Fantasy World checkpoint (for Stage 2, load Stage 1 weights)
                if fantasy_world_checkpoint is not None:
                    print(f"\n{'='*80}")
                    print(f"Loading Fantasy World Checkpoint")
                    print(f"{'='*80}")
                    print(f"Checkpoint: {fantasy_world_checkpoint}")
                    print(f"Training stage: {training_stage}")
                    
                    from safetensors.torch import load_file as load_safetensors
                    state_dict = load_safetensors(fantasy_world_checkpoint)
                    
                    # Load into DiT (strict=False allows missing keys for frozen blocks)
                    missing_keys, unexpected_keys = self.pipe.dit.load_state_dict(state_dict, strict=False)
                    
                    print(f"Total keys loaded: {len(state_dict)}")
                    print(f"Missing keys (frozen blocks): {len(missing_keys)}")
                    print(f"Unexpected keys: {len(unexpected_keys)}")
                    
                    # Check if critical geometry modules are loaded
                    geo_modules = [k for k in state_dict.keys() if 'geo_blocks' in k]
                    print(f"Geometry modules loaded: {len(geo_modules)} keys")
                    
                    if len(geo_modules) == 0:
                        print("⚠️  WARNING: No geometry modules found in checkpoint!")
                    else:
                        print("✓ Geometry modules successfully loaded")
                    
                    print(f"{'='*80}\n")

        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "fantasy_world": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FantasyWorldLoss(pipe, **inputs_shared, **inputs_posi),
            "fantasy_world:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FantasyWorldLoss(pipe, **inputs_shared, **inputs_posi),
            "fantasy_world:stage1": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FantasyWorldLoss(pipe, **inputs_shared, **inputs_posi),
            "fantasy_world:stage2": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FantasyWorldLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        """
        Override to save complete dit model for Fantasy World, not just trainable params.
        For fantasy_world tasks, we need the full dit (both frozen and trainable parts)
        for inference, not just the trainable parameters.
        """
        if self.task.startswith("fantasy_world"):
            # For Fantasy World: save entire dit + new geometry modules
            trainable_param_names = self.trainable_param_names()
            # Include both frozen dit params and trainable fantasy world params
            state_dict_filtered = {}
            for name, param in state_dict.items():
                # Keep dit.blocks.* (frozen) and all trainable params
                if name.startswith("dit.") or name in trainable_param_names:
                    state_dict_filtered[name] = param
            state_dict = state_dict_filtered
        else:
            # For other tasks: use standard logic (only trainable params)
            trainable_param_names = self.trainable_param_names()
            state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "pose_file_path":
                # For Fantasy World: camera_params is the txt file path
                # This will be processed by WanVideoUnit_FunCameraControl
                if "camera_params" in data:
                    inputs_shared["pose_file_path"] = data["camera_params"]
                else:
                    inputs_shared["pose_file_path"] = None
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Handle Fantasy World geometry data (rename keys for loss computation)
        if self.task.startswith("fantasy_world"):
            # Map depth -> gt_depth
            if "depth" in data and data["depth"] is not None:
                depth = data["depth"]
                if isinstance(depth, torch.Tensor):
                    if depth.ndim == 3:  # [T, H, W] -> [1, T, H, W]
                        depth = depth.unsqueeze(0)
                    inputs_shared["gt_depth"] = depth
            # Map points -> gt_points  
            if "points" in data and data["points"] is not None:
                points = data["points"]
                if isinstance(points, torch.Tensor):
                    if points.ndim == 4:  # [T, H, W, 3] -> [1, T, H, W, 3]
                        points = points.unsqueeze(0)
                        # Permute to [B, T, 3, H, W]
                        points = points.permute(0, 1, 4, 2, 3)
                    inputs_shared["gt_points"] = points
            # Camera params: the txt file path is passed via pose_file_path
            # WanVideoUnit_FunCameraControl will parse it and generate pose_params
            # For loss computation, we'll extract ground truth from the same file
            if "camera_params" in data and data["camera_params"] is not None:
                # Store the path for later extraction during loss computation
                inputs_shared["gt_camera_file"] = data["camera_params"]
                    
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--fantasy_world_checkpoint", type=str, default=None, help="Path to Fantasy World checkpoint (Stage 1 output for Stage 2 training).")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    
    # Build special operator map based on task
    special_operator_map = {
        "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
        "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
    }
    
    # Add Fantasy World specific operators if task is fantasy_world
    if args.task.startswith("fantasy_world"):
        special_operator_map.update({
            "depth": default_depth_operator(args.dataset_base_path, args.num_frames),
            "points": default_points_operator(args.dataset_base_path, args.num_frames),
            "camera_params": default_camera_operator(args.dataset_base_path),
        })
        # Extend data_file_keys for fantasy_world
        data_file_keys = args.data_file_keys.split(",")
        for key in ["depth", "points", "camera_params"]:
            if key not in data_file_keys:
                data_file_keys.append(key)
    else:
        data_file_keys = args.data_file_keys.split(",")
    
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=data_file_keys,
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map=special_operator_map,
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        fantasy_world_checkpoint=args.fantasy_world_checkpoint,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "fantasy_world": launch_training_task,
        "fantasy_world:train": launch_training_task,
        "fantasy_world:stage1": launch_training_task,
        "fantasy_world:stage2": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
