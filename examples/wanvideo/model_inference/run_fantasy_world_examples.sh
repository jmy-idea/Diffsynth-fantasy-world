#!/bin/bash
# =============================================================================
# Fantasy World Inference Examples
# 
# This script demonstrates different inference scenarios with trained Fantasy World models.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world"

cd "${PROJECT_ROOT}"

# Configuration
CHECKPOINT="outputs/fantasy_world_stage1/step-2.safetensors"  # Update with your checkpoint
STAGE="stage1"  # or "stage2" for Stage 2 checkpoints

echo "=============================================="
echo "Fantasy World Inference Examples"
echo "=============================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Stage: ${STAGE}"
echo ""

# Example 1: Orbit camera trajectory
echo "Example 1: Orbit Camera (360° rotation)"
echo "----------------------------------------------"
python examples/wanvideo/model_inference/fantasy_world_inference.py \
    --checkpoint "${CHECKPOINT}" \
    --training_stage "${STAGE}" \
    --prompt "一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒。" \
    --negative_prompt "静态，模糊，低质量，畸形" \
    --create_trajectory orbit \
    --num_frames 21 \
    --height 480 \
    --width 832 \
    --seed 0 \
    --output "outputs/fantasy_world_orbit.mp4"

echo ""
echo "✓ Example 1 complete: outputs/fantasy_world_orbit.mp4"
echo ""

# Example 2: Forward motion (zoom in)
echo "Example 2: Forward Motion (Zoom In)"
echo "----------------------------------------------"
python examples/wanvideo/model_inference/fantasy_world_inference.py \
    --checkpoint "${CHECKPOINT}" \
    --training_stage "${STAGE}" \
    --prompt "A majestic mountain landscape with snow-capped peaks. The camera slowly approaches, revealing intricate details of the rocky terrain and alpine vegetation." \
    --negative_prompt "blurry, static, low quality, distorted" \
    --create_trajectory forward \
    --num_frames 21 \
    --height 480 \
    --width 832 \
    --seed 42 \
    --output "outputs/fantasy_world_forward.mp4"

echo ""
echo "✓ Example 2 complete: outputs/fantasy_world_forward.mp4"
echo ""

# Example 3: Left-right pan
echo "Example 3: Horizontal Pan (Left to Right)"
echo "----------------------------------------------"
python examples/wanvideo/model_inference/fantasy_world_inference.py \
    --checkpoint "${CHECKPOINT}" \
    --training_stage "${STAGE}" \
    --prompt "A serene forest scene with tall trees and dappled sunlight. The camera pans horizontally, showcasing the peaceful woodland atmosphere." \
    --negative_prompt "blurry, static, low quality, distorted" \
    --create_trajectory left_right \
    --num_frames 21 \
    --height 480 \
    --width 832 \
    --seed 123 \
    --output "outputs/fantasy_world_pan.mp4"

echo ""
echo "✓ Example 3 complete: outputs/fantasy_world_pan.mp4"
echo ""

# Example 4: Custom trajectory (if you have a custom trajectory file)
# echo "Example 4: Custom Camera Trajectory"
# echo "----------------------------------------------"
# python examples/wanvideo/model_inference/fantasy_world_inference.py \
#     --checkpoint "${CHECKPOINT}" \
#     --training_stage "${STAGE}" \
#     --prompt "Your custom prompt" \
#     --camera_trajectory path/to/your/custom_trajectory.txt \
#     --num_frames 21 \
#     --height 480 \
#     --width 832 \
#     --seed 456 \
#     --output "outputs/fantasy_world_custom.mp4"

echo "=============================================="
echo "All examples complete!"
echo "=============================================="
echo "Generated videos:"
echo "  - outputs/fantasy_world_orbit.mp4"
echo "  - outputs/fantasy_world_forward.mp4"
echo "  - outputs/fantasy_world_pan.mp4"
echo ""
echo "To generate with your own settings:"
echo "  python examples/wanvideo/model_inference/fantasy_world_inference.py --help"
