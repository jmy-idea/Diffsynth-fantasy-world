#!/bin/bash
# =============================================================================
# FantasyWorld Stage 2 Training: Unified Co-Optimization
# 
# This stage:
# - Loads Stage 1 checkpoint
# - Keeps Wan2.1 backbone FROZEN (all 30 blocks)
# - Keeps geometry branch core FROZEN (GeoDiT blocks, already trained in Stage 1)
# - Trains ONLY the interaction modules:
#   * IRG Bidirectional Cross-Attention (18 modules)
#   * Camera Control Adapters (first 12 blocks)
#   * DPT Heads fine-tuning (optional, usually frozen)
# - Duration: 10,000 steps
# - Batch size: 112 (global)
# - Resolution: Full resolution 592×336 or 336×592
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world"
DATA_DIR="/path/to/your/fantasy_world_data"  # TODO: Update this
STAGE1_CHECKPOINT="${PROJECT_ROOT}/outputs/fantasy_world_stage1/step-20000.safetensors"  # Stage 1 output
OUTPUT_DIR="${PROJECT_ROOT}/outputs/fantasy_world_stage2"

# Training Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on your GPUs
NUM_GPUS=8
BATCH_SIZE_PER_GPU=14  # 14 * 8 = 112 global batch size
GRADIENT_ACCUMULATION=1
NUM_STEPS=10000
LEARNING_RATE=1e-5
NUM_FRAMES=81
HEIGHT=592
WIDTH=336

echo "=============================================="
echo "FantasyWorld Stage 2: Unified Co-Optimization"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Data dir: ${DATA_DIR}"
echo "Stage 1 checkpoint: ${STAGE1_CHECKPOINT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Global batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS))"
echo "Training steps: ${NUM_STEPS}"
echo ""

# Check Stage 1 checkpoint exists
if [ ! -f "${STAGE1_CHECKPOINT}" ]; then
    echo "ERROR: Stage 1 checkpoint not found: ${STAGE1_CHECKPOINT}"
    echo "Please run Stage 1 training first:"
    echo "  bash examples/wanvideo/model_training/full/train_fantasy_world_stage1.sh"
    exit 1
fi

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
    --dataset_num_workers 4 \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --output_path "${OUTPUT_DIR}" \
    --task fantasy_world:stage2 \
    --trainable_models "dit" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --stage1_checkpoint "${STAGE1_CHECKPOINT}" \
    --data_file_keys "video,depth,points,camera_params" \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 1e-2 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --num_epochs 1 \
    --save_steps 1000 \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --extra_inputs "input_image,pose_file_path"

echo ""
echo "=============================================="
echo "Stage 2 Training Complete!"
echo "Final checkpoint saved to: ${OUTPUT_DIR}"
echo "=============================================="
echo ""
echo "Model is ready for inference!"
