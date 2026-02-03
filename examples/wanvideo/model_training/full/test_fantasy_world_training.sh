#!/bin/bash
# =============================================================================
# Fantasy World Training Test Script
# 
# This script:
# 1. Creates fake test data
# 2. Runs a quick training test to verify the pipeline works
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world"
TEST_DATA_DIR="${PROJECT_ROOT}/test_data/fantasy_world_fake"
OUTPUT_DIR="${PROJECT_ROOT}/test_outputs/fantasy_world_test"

export CUDA_VISIBLE_DEVICES=1

echo "=============================================="
echo "Fantasy World Training Pipeline Test"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Test data dir: ${TEST_DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Check if diffsynth is importable, if not install in development mode
# echo "[Setup] Checking diffsynth installation..."
# python -c "import diffsynth" 2>/dev/null || {
#     echo "Installing diffsynth in development mode..."
#     cd "${PROJECT_ROOT}"
#     pip install -e .
#     echo "Installation complete!"
# }

# Step 1: Create fake data
# echo ""
# echo "[Step 1] Creating fake test data..."
# cd "${PROJECT_ROOT}"
# python scripts/create_fake_data.py \
#     --output_dir "${TEST_DATA_DIR}" \
#     --num_samples 1 \
#     --num_frames 21 \
#     --height 480 \
#     --width 832 \
#     --verify

echo ""
echo "[Step 1] Done. Fake data created at: ${TEST_DATA_DIR}"

# Step 2: Run training test (dry run with minimal steps)
echo ""
echo "[Step 2] Running training test..."
echo "Note: This requires the actual model weights to be available."
echo "If you don't have model weights, this step will fail."
echo ""

# Go to project root for proper module resolution
cd "${PROJECT_ROOT}"

# Run training
python examples/wanvideo/model_training/train.py \
    --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --dataset_base_path "${TEST_DATA_DIR}" \
    --dataset_metadata_path "${TEST_DATA_DIR}/metadata.json" \
    --dataset_repeat 1 \
    --height 480 \
    --width 832 \
    --output_path "${OUTPUT_DIR}" \
    --task fantasy_world \
    --trainable_models "dit" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --num_frames 21 \
    --data_file_keys "video,depth,points,camera_params" \
    --learning_rate 1e-5 \
    --extra_inputs "input_image,pose_file_path"

echo ""
echo "[Step 2] Training command executed."
echo ""
echo "=============================================="
echo "Test Complete!"
echo "=============================================="
