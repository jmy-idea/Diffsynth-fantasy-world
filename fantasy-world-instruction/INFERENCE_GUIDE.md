# 🎥 推理与应用完整指南

本文档详细说明如何使用 Fantasy World 模型进行推理和生成视频。

---

## 📋 目录

1. [推理基础](#推理基础)
2. [模型加载](#模型加载)
3. [相机轨迹控制](#相机轨迹控制)
4. [视频生成](#视频生成)
5. [批处理与优化](#批处理与优化)
6. [输出处理](#输出处理)
7. [推理示例](#推理示例)

---

## 🚀 推理基础

### 什么是推理？

推理是指使用已训练的模型生成新数据的过程：

```
输入 (文本提示 + 图像/相机轨迹) 
    ↓
[已训练的 Fantasy World 模型]
    ↓
输出 (生成的视频 + 几何预测)
```

### 推理 vs 训练

| 特性 | 训练 | 推理 |
|------|------|------|
| **GPU 需求** | 8 × H20 (40GB) | 1 × 任何 GPU (12GB+) |
| **内存要求** | 高 | 中等 |
| **时间** | 200 小时 | 1-2 分钟 / 视频 |
| **可编辑** | 是 | 否 |
| **用途** | 改进模型 | 生成结果 |

### 推理模式

Fantasy World 支持多种推理模式：

| 模式 | 说明 | 需要的输入 | 输出 |
|------|------|---------|------|
| **文本到视频** | 基于文本提示生成 | 文本 + 相机轨迹 | 视频 + 深度 + 点云 |
| **图像到视频** | 从起始帧扩展视频 | 图像 + 相机轨迹 | 视频 + 几何 |
| **相机控制** | 精确控制摄像机路径 | 文本 + 相机文件 | 相机受控视频 |

---

## 🔌 模型加载

### 前置要求

```bash
# 1. 检查环境
python -c "import diffsynth; print('OK')"

# 2. 检查 GPU 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 3. 检查模型文件
ls outputs/fantasy_world_stage2/step-10000.safetensors
```

### 方法 1: 从本地检查点加载 (推荐)

```python
import torch
from diffsynth import WanVideoPipeline

# 1. 初始化基础模型
pipe = WanVideoPipeline.from_pretrained(
    "PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera",
    torch_dtype=torch.bfloat16,  # 使用 BFloat16 节省显存
    device_map="cuda:0"           # 使用 GPU 0
)

# 2. 启用 Fantasy World 模式
pipe.dit.enable_fantasy_world_mode(training_stage="stage2")

# 3. 加载微调的检查点
checkpoint_path = "outputs/fantasy_world_stage2/step-10000.safetensors"
state_dict = torch.load(checkpoint_path, map_location="cpu")

# 转换为 float32 (如果模型是 float32)
state_dict = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v 
              for k, v in state_dict.items()}

# 加载状态字典
# strict=False: 允许缺失的键 (frozen Wan blocks)
pipe.dit.load_state_dict(state_dict, strict=False)

print("✅ 模型加载成功")
```

### 方法 2: 使用推理脚本 (最简单)

```bash
# 使用预准备的推理脚本
python examples/wanvideo/model_inference/fantasy_world_inference.py \
    --checkpoint outputs/fantasy_world_stage2/step-10000.safetensors \
    --prompt "a serene indoor scene with camera slowly rotating" \
    --output_dir results/ \
    --num_frames 81 \
    --seed 42
```

### 方法 3: 通过命令行包装器

```bash
# 创建便捷脚本
cat > run_inference.py << 'EOF'
#!/usr/bin/env python
import argparse
from diffsynth_fantasy_world import run_inference

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", required=True, help="生成提示")
parser.add_argument("--checkpoint", default="outputs/fantasy_world_stage2/step-10000.safetensors")
parser.add_argument("--output_dir", default="results/")
parser.add_argument("--num_frames", type=int, default=81)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()
run_inference(**vars(args))
EOF

python run_inference.py --prompt "a camera moving through a room"
```

---

## 📷 相机轨迹控制

### 相机轨迹文件格式

相机轨迹是一个 `.txt` 文件，每行 19 个值：

```
frame_idx fx fy cx cy k1 k2 w2c_00 w2c_01 ... w2c_23

其中:
- frame_idx: 帧序号 (0-T)
- fx, fy: 焦距 (内参)
- cx, cy: 主点 (内参)
- k1, k2: 径向畸变系数
- w2c_*: 世界到相机的 3×4 矩阵 (12 个值)
```

**示例**:
```
0 500.0 500.0 320.0 240.0 0.0 0.0 0.9 0.1 0.2 1.0 0.3 -0.5 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
1 500.0 500.0 320.0 240.0 0.0 0.0 0.85 0.12 0.25 1.1 0.32 -0.48 0.42 0.52 0.62 0.72 0.82 0.92 1.02 1.12 1.22
...
```

### 生成预定义轨迹

Fantasy World 提供多种预定义轨迹：

#### 1. 轨道运动 (Orbital Motion)

```python
import numpy as np

def create_orbital_trajectory(num_frames, radius=3.0, height=1.5):
    """
    创建绕场景旋转的轨迹
    
    Args:
        num_frames: 总帧数
        radius: 旋转半径 (米)
        height: 相机高度 (米)
    """
    trajectories = []
    
    for i in range(num_frames):
        t = i / num_frames * 2 * np.pi  # 0 到 2π
        
        # 位置 (绕 Y 轴旋转)
        x = radius * np.cos(t)
        y = height
        z = radius * np.sin(t)
        
        # 注视点 (场景中心)
        look_at = np.array([0, height, 0])
        
        # 构建 w2c 矩阵 (通常由 COLMAP 或手动指定)
        # 这里简化处理...
        
        trajectories.append({
            'frame': i,
            'position': np.array([x, y, z]),
            'look_at': look_at
        })
    
    return trajectories
```

#### 2. 前向运动 (Forward Motion)

```python
def create_forward_trajectory(num_frames, start_z=10.0, end_z=0.5):
    """相机向前移动"""
    trajectories = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        z = start_z * (1 - t) + end_z * t  # 线性插值
        
        trajectories.append({
            'frame': i,
            'position': np.array([0, 0, z]),
            'look_at': np.array([0, 0, 0])
        })
    
    return trajectories
```

#### 3. 自定义轨迹

```python
def create_custom_trajectory(num_frames, keyframes):
    """
    从关键帧插值生成轨迹
    
    Args:
        keyframes: {frame_idx: (x, y, z), ...}
    """
    trajectories = []
    frame_indices = sorted(keyframes.keys())
    
    for i in range(num_frames):
        # 找到相邻的关键帧
        if i <= frame_indices[0]:
            pos = keyframes[frame_indices[0]]
        elif i >= frame_indices[-1]:
            pos = keyframes[frame_indices[-1]]
        else:
            # 线性插值
            for j in range(len(frame_indices) - 1):
                f1, f2 = frame_indices[j], frame_indices[j + 1]
                if f1 <= i <= f2:
                    p1 = np.array(keyframes[f1])
                    p2 = np.array(keyframes[f2])
                    alpha = (i - f1) / (f2 - f1)
                    pos = p1 * (1 - alpha) + p2 * alpha
                    break
        
        trajectories.append({
            'frame': i,
            'position': pos,
            'look_at': np.array([0, 0, 0])
        })
    
    return trajectories
```

### 保存轨迹文件

```python
def save_trajectory_file(trajectories, output_file, intrinsics=None):
    """
    保存轨迹为 .txt 文件
    
    Args:
        trajectories: 轨迹列表
        output_file: 输出文件路径
        intrinsics: 相机内参 {'fx': ..., 'fy': ..., ...}
    """
    if intrinsics is None:
        intrinsics = {
            'fx': 500.0,
            'fy': 500.0,
            'cx': 320.0,
            'cy': 240.0,
            'k1': 0.0,
            'k2': 0.0
        }
    
    with open(output_file, 'w') as f:
        for traj in trajectories:
            frame_idx = traj['frame']
            
            # 构建 w2c 矩阵 (需要从位置和注视点计算)
            # 这里简化为示例矩阵
            w2c = np.eye(3, 4)  # [3, 4]
            
            # 组装 19 值
            line = f"{frame_idx} "
            line += f"{intrinsics['fx']} {intrinsics['fy']} "
            line += f"{intrinsics['cx']} {intrinsics['cy']} "
            line += f"{intrinsics['k1']} {intrinsics['k2']} "
            line += " ".join(map(str, w2c.flatten()))
            
            f.write(line + "\n")
    
    print(f"✅ 轨迹已保存: {output_file}")

# 使用示例
orbital_traj = create_orbital_trajectory(num_frames=81)
save_trajectory_file(orbital_traj, "trajectory_orbital.txt")
```

---

## 🎬 视频生成

### 基础生成

```python
import torch
from diffsynth import WanVideoPipeline

# 加载模型 (见前面的模型加载部分)
pipe = WanVideoPipeline.from_pretrained(...)
pipe.dit.enable_fantasy_world_mode(training_stage="stage2")
# ... 加载检查点

# 生成视频
video = pipe(
    prompt="a beautiful living room with sunlight coming through windows",
    negative_prompt="blurry, low quality",
    num_frames=81,
    height=336,
    width=592,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
)

print(f"生成视频形状: {video.shape}")  # [81, 3, 336, 592]
```

### 带相机控制的生成

```python
# 使用相机轨迹控制生成
video = pipe(
    prompt="a camera moving through an abandoned building",
    pose_file_path="trajectory_orbital.txt",  # 相机轨迹
    num_frames=81,
    height=336,
    width=592,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
)
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | str | - | **必需**: 文本描述 |
| `negative_prompt` | str | "" | 要避免的描述 |
| `num_frames` | int | 81 | 生成帧数 (21-81) |
| `height` | int | 336 | 视频高度 (多 16 倍数) |
| `width` | int | 592 | 视频宽度 (多 16 倍数) |
| `num_inference_steps` | int | 50 | 推理步数 (越多越好，越慢) |
| `guidance_scale` | float | 7.5 | 引导强度 (1-20) |
| `seed` | int | - | 随机种子 (可复现) |
| `pose_file_path` | str | None | 相机轨迹文件 |
| `generator` | Generator | None | PyTorch 随机生成器 |

### 推理速度 vs 质量

```python
# 快速但质量一般 (30 步, 1-2 分钟)
video_fast = pipe(
    prompt="...",
    num_inference_steps=30,
    guidance_scale=5.0
)

# 平衡 (50 步, 2-3 分钟)
video_balanced = pipe(
    prompt="...",
    num_inference_steps=50,
    guidance_scale=7.5
)

# 质量优先 (70 步, 4-5 分钟)
video_quality = pipe(
    prompt="...",
    num_inference_steps=70,
    guidance_scale=9.0
)
```

---

## 🚄 批处理与优化

### 批量生成

```python
import torch
from diffsynth import WanVideoPipeline

pipe = WanVideoPipeline.from_pretrained(...)
pipe.dit.enable_fantasy_world_mode(training_stage="stage2")
# ... 加载检查点

# 提示列表
prompts = [
    "a minimalist interior with soft lighting",
    "a vibrant marketplace with people and activity",
    "a serene nature scene with flowing water",
]

# 生成所有视频
for i, prompt in enumerate(prompts):
    print(f"生成视频 {i+1}/{len(prompts)}...")
    
    video = pipe(
        prompt=prompt,
        num_frames=81,
        seed=42 + i  # 不同种子保证多样性
    )
    
    # 保存 (见后面的输出处理)
    save_video(video, f"results/video_{i:02d}.mp4")
```

### 显存优化

**方法 1: 启用内存高效注意力**

```python
# 加载模型时启用
pipe = WanVideoPipeline.from_pretrained(
    ...,
    enable_attention_slicing=True  # 节省显存
)
```

**方法 2: 减少推理步数**

```python
# 从 50 步降到 30 步 (显存减少 40%, 速度快 40%)
video = pipe(..., num_inference_steps=30)
```

**方法 3: 降低分辨率**

```python
# 从 336×592 降到 224×384 (显存减少 ~50%)
video = pipe(
    ...,
    height=224,
    width=384,
    num_frames=41  # 也可减少帧数
)
```

**方法 4: 使用 CPU offloading**

```python
# 在 GPU 和 CPU 之间转移模块
pipe.enable_model_cpu_offload()  # 稍微慢一些，但显存减少
```

### 推理加速

**方法 1: 使用 BFloat16**

```python
pipe = WanVideoPipeline.from_pretrained(
    ...,
    torch_dtype=torch.bfloat16  # 加速 + 节省显存
)
```

**方法 2: 启用 xFormers 优化**

```bash
pip install xformers

python << 'EOF'
from diffsynth import WanVideoPipeline
pipe = WanVideoPipeline.from_pretrained(...)
pipe.enable_xformers_memory_efficient_attention()
EOF
```

---

## 💾 输出处理

### 保存视频

```python
import cv2
import numpy as np
from pathlib import Path

def save_video(video_tensor, output_path, fps=24):
    """
    保存生成的视频
    
    Args:
        video_tensor: [T, 3, H, W] 或 [T, H, W, 3] 张量
        output_path: 输出文件路径 (.mp4)
        fps: 帧率
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换张量格式
    if isinstance(video_tensor, torch.Tensor):
        video_tensor = video_tensor.cpu().numpy()
    
    # 确保形状为 [T, H, W, 3]
    if video_tensor.shape[1] == 3:
        video_tensor = video_tensor.transpose(0, 2, 3, 1)
    
    # 转换值域 [0, 1] → [0, 255]
    if video_tensor.max() <= 1.0:
        video_tensor = (video_tensor * 255).astype(np.uint8)
    else:
        video_tensor = video_tensor.astype(np.uint8)
    
    # 初始化 VideoWriter
    height, width = video_tensor.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )
    
    # 逐帧写入
    for frame in video_tensor:
        # 转换 RGB → BGR (OpenCV 格式)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()
    print(f"✅ 视频已保存: {output_path}")

# 使用
video = pipe(prompt="...")  # [T, 3, H, W]
save_video(video, "results/output.mp4", fps=24)
```

### 保存 3D 几何

```python
def save_depth_map(depth_tensor, output_dir):
    """保存预测的深度图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # depth_tensor: [T, 1, H, W] 或 [T, H, W]
    if depth_tensor.dim() == 4:
        depth_tensor = depth_tensor.squeeze(1)
    
    for i, depth in enumerate(depth_tensor):
        # 转换为 numpy 并归一化
        depth_np = depth.cpu().numpy()
        depth_np = (depth_np * 255).astype(np.uint8)
        
        # 保存为 PNG
        output_path = output_dir / f"depth_{i:04d}.png"
        cv2.imwrite(str(output_path), depth_np)
    
    print(f"✅ 深度图已保存: {output_dir}")

def save_point_cloud(points_tensor, output_dir):
    """保存预测的点云"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # points_tensor: [T, 3, H, W]
    for i, points in enumerate(points_tensor):
        # 转换为 [H*W, 3] 格式
        points_np = points.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        points_flat = points_np.reshape(-1, 3)
        
        # 保存为 PLY (点云格式)
        output_path = output_dir / f"points_{i:04d}.ply"
        save_ply(points_flat, str(output_path))
    
    print(f"✅ 点云已保存: {output_dir}")
```

### 生成可视化

```python
def visualize_output(video, depth, points, output_path):
    """生成对比可视化"""
    import matplotlib.pyplot as plt
    
    # 选择关键帧
    key_frames = [0, len(video) // 2, -1]
    
    fig, axes = plt.subplots(3, len(key_frames), figsize=(15, 10))
    
    for col, frame_idx in enumerate(key_frames):
        # 视频帧
        frame = video[frame_idx].permute(1, 2, 0).cpu().numpy()
        axes[0, col].imshow(frame)
        axes[0, col].set_title(f"Frame {frame_idx}")
        
        # 深度图
        depth_map = depth[frame_idx, 0].cpu().numpy()
        axes[1, col].imshow(depth_map, cmap='viridis')
        axes[1, col].set_title("Depth")
        
        # 点云 (可视化为 3D)
        points_map = points[frame_idx].cpu().numpy()
        # 这里简化为显示 X 分量
        axes[2, col].imshow(points_map[0], cmap='coolwarm')
        axes[2, col].set_title("Points (X)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ 可视化已保存: {output_path}")
```

---

## 📝 推理示例

### 示例 1: 简单文本到视频

```python
#!/usr/bin/env python
import torch
from diffsynth import WanVideoPipeline

# 加载模型
pipe = WanVideoPipeline.from_pretrained(
    "PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)

# 启用 Fantasy World
pipe.dit.enable_fantasy_world_mode(training_stage="stage2")

# 加载检查点
state_dict = torch.load("outputs/fantasy_world_stage2/step-10000.safetensors", map_location="cpu")
pipe.dit.load_state_dict(state_dict, strict=False)

# 生成视频
video = pipe(
    prompt="a beautiful garden with flowers and butterflies",
    num_frames=81,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42
)

# 保存
save_video(video, "results/garden.mp4", fps=24)
```

### 示例 2: 带相机控制的生成

```python
#!/usr/bin/env python
import torch
from diffsynth import WanVideoPipeline
import numpy as np

# 加载模型 (同上)
pipe = WanVideoPipeline.from_pretrained(...)
# ... 初始化代码

# 创建相机轨迹
def create_camera_trajectory(num_frames):
    cameras = []
    for i in range(num_frames):
        # 简化的相机参数 (实际需要完整的 w2c 矩阵)
        line = f"{i} 500.0 500.0 320.0 240.0 0.0 0.0 "
        line += " ".join([str(float(j)) for j in range(12)])
        cameras.append(line)
    
    with open("trajectory.txt", "w") as f:
        f.write("\n".join(cameras))

create_camera_trajectory(81)

# 生成视频
video = pipe(
    prompt="a camera slowly panning through an abandoned castle",
    pose_file_path="trajectory.txt",
    num_frames=81,
    num_inference_steps=50,
    seed=42
)

save_video(video, "results/castle_pan.mp4")
```

---

## 🎯 最佳实践

### 提示词设计

**好的提示词**:
- ✅ "a serene living room with soft warm lighting"
- ✅ "a camera orbiting a modern sculpture"
- ✅ "people walking through a busy marketplace"

**不好的提示词**:
- ❌ "room" (太简洁)
- ❌ "something moving in a place" (太模糊)
- ❌ 超过 100 字 (太长)

### 优化策略

1. **品质优化**: 增加 inference steps (50-70)
2. **速度优化**: 减少 frames (41 而非 81) 或 steps (30)
3. **稳定性**: 设置种子确保可复现
4. **多样性**: 改变种子生成多个变体

### 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| "显存不足" | 模型太大 | 启用 CPU offloading 或降低分辨率 |
| "生成重复帧" | 模型欠拟合 | 增加 inference steps |
| "视频抖动" | 不稳定 | 增加 guidance scale 或改变 seed |
| "几何无效" | 检查点问题 | 重新加载或验证检查点 |

---

**下一步**: 如遇问题，查看 [故障排查](./TROUBLESHOOTING.md) 或 [技术深入](./TECHNICAL_DEEP_DIVE.md)。
