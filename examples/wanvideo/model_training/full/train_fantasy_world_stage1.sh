#!/bin/bash
# =============================================================================
# FantasyWorld Stage 1 Training: Latent Bridging
# 
# This stage:
# - Freezes the Wan2.1 backbone (all 30 blocks)
# - Trains ONLY the geometry branch:
#   * Latent Bridge Adapter (mapping video features to geometry space)
#   * GeoDiT Blocks (18 blocks with VGGT-style attention)
#   * DPT Heads (depth, point, camera)
#   * Pose Encoder
#   * Special Tokens (camera + register)
# - Duration: 20,000 steps
# - Batch size: 64 (global)
# - Resolution: Lower resolution for stability
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world"
DATA_DIR="/ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world/test_data/fantasy_world_fake"  # TODO: Update this
OUTPUT_DIR="${PROJECT_ROOT}/outputs/fantasy_world_stage1"

# Training Configuration
export CUDA_VISIBLE_DEVICES=4,5  # Adjust based on your GPUs
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1  # 1 * 2 = 2 global batch size
GRADIENT_ACCUMULATION=1
NUM_STEPS=20000
LEARNING_RATE=1e-5
NUM_FRAMES=21
HEIGHT=480
WIDTH=832

echo "=============================================="
echo "FantasyWorld Stage 1: Latent Bridging"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Global batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS))"
echo "Training steps: ${NUM_STEPS}"
echo ""

# Go to project root
cd "${PROJECT_ROOT}"

# Run distributed training
accelerate launch \
    --num_processes ${NUM_GPUS} \
    --mixed_precision bf16 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    examples/wanvideo/model_training/train.py \
    --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --dataset_base_path "${DATA_DIR}" \
    --dataset_metadata_path "${DATA_DIR}/metadata.json" \
    --dataset_repeat 1 \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --output_path "${OUTPUT_DIR}" \
    --task fantasy_world:stage1 \
    --trainable_models "dit" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --data_file_keys "video,depth,points,camera_params" \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 1e-2 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --num_epochs 2 \
    --save_steps 1000 \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --extra_inputs "input_image,pose_file_path"

echo ""
echo "=============================================="
echo "Stage 1 Training Complete!"
echo "Checkpoint saved to: ${OUTPUT_DIR}"
echo "=============================================="
echo ""
echo "Next step: Run Stage 2 with:"
echo "  bash examples/wanvideo/model_training/full/train_fantasy_world_stage2.sh"
