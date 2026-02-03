# 📊 数据准备与处理完全指南

本文档详细说明如何为 Fantasy World 模型准备训练和推理所需的数据。

---

## 📋 目录

1. [数据需求概览](#数据需求概览)
2. [数据格式规范](#数据格式规范)
3. [数据生成管道](#数据生成管道)
4. [数据集组织结构](#数据集组织结构)
5. [元数据配置](#元数据配置)
6. [数据验证与质量检查](#数据验证与质量检查)
7. [常见数据问题解决](#常见数据问题解决)

---

## 📦 数据需求概览

### 总体需求

| 用途 | 最少数据量 | 推荐数据量 | 存储空间 |
|------|-----------|-----------|---------|
| **推理测试** | 示例图片 (自动下载) | - | 100MB |
| **训练** (完整) | 100 个样本 | 1000+ 个样本 | 500GB+ |
| **微调** | 50 个样本 | 200-500 个样本 | 100-200GB |

### 每个样本包含

```
sample_001/
├── video.mp4                    # 原始视频 (必需)
├── depth/                       # 深度图序列 (必需)
│   ├── frame_0000.npy
│   ├── frame_0001.npy
│   └── ...
├── points/                      # 点云序列 (必需)
│   ├── frame_0000.npy
│   ├── frame_0001.npy
│   └── ...
└── camera_params.txt            # 相机轨迹 (必需)
```

### 数据特性

| 指标 | 要求 | 说明 |
|------|------|------|
| 视频分辨率 | 336×592 - 592×336 | 不同阶段可能不同 |
| 帧数 | 21-81 | Stage 1 可用 21 帧，Stage 2 使用 81 帧 |
| 帧率 | 10-30 fps | 对数据质量影响不大 |
| 编码 | H.264/H.265 | 常见视频编码 |
| 颜色空间 | RGB | 需要转换到 RGB |
| 深度范围 | 0.1-100m | 取决于场景尺度 |

---

## 📐 数据格式规范

### 1. 视频格式

**输入**: MP4, AVI, MOV, WebM 等常见格式

**处理步骤**:
```python
# 使用 OpenCV 或 ffmpeg 读取
import cv2
cap = cv2.VideoCapture("video.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB
    frames.append(frame)  # [H, W, 3]
cap.release()
```

**输出**: 帧列表，每帧 [H, W, 3] (0-255 uint8 或 0-1 float32)

### 2. 深度图格式

**生成工具**: Depth Anything V2

**格式规范**:
- 类型: `.npy` (numpy 格式)
- 值域: 0-1 (归一化) 或 0-255 (原始)
- 形状: [H, W] (单通道)
- 数据类型: float32 或 uint8

**保存方式**:
```python
import numpy as np

# 生成或获取深度图
depth = np.random.rand(H, W).astype(np.float32)

# 保存
np.save("frame_0000.npy", depth)

# 加载
depth_loaded = np.load("frame_0000.npy")  # [H, W]
```

**预处理**:
```python
# 如果值域不是 0-1，进行归一化
def normalize_depth(depth):
    min_val = depth.min()
    max_val = depth.max()
    if max_val > min_val:
        depth = (depth - min_val) / (max_val - min_val)
    return depth
```

### 3. 点云格式

**生成工具**: DUSt3R / CUT3R 或其他 MVS 方法

**格式规范**:
- 类型: `.npy` (numpy 格式)
- 值域: -1 到 1 (归一化) 或实际尺度
- 形状: [H, W, 3] (XYZ 坐标)
- 数据类型: float32

**结构**:
```python
# 点云是一个 3D 坐标网格
points = np.random.randn(H, W, 3).astype(np.float32)
# 对应图像的每个像素 (i, j)，有一个 3D 点 (x, y, z)
```

**保存方式**:
```python
import numpy as np

# 保存
np.save("frame_0000.npy", points)

# 加载
points_loaded = np.load("frame_0000.npy")  # [H, W, 3]
```

**预处理** (归一化):
```python
def normalize_points(points):
    # 计算中心
    center = points.mean(axis=(0, 1))
    points = points - center
    
    # 计算尺度
    scale = np.abs(points).max()
    if scale > 0:
        points = points / scale
    
    return points
```

### 4. 相机参数格式

这里实际上前面7维保持" 0 0.532139961 0.946026558 0.5 0.5 0 0 "即可，只需要获得后面12维的w2c外参。参考Diffsynth-fantasy-world/Move_Left.txt

**格式**: 19 值 per frame

```
frame_idx fx fy cx cy k1 k2 w2c_00 w2c_01 ... w2c_23

其中:
- frame_idx: 帧号 (0-T)
- fx, fy: 焦距 (内参)
- cx, cy: 主点 (内参)
- k1, k2: 径向畸变系数
- w2c_00...w2c_23: 世界到相机的 3×4 变换矩阵 (行优先)
```

**文件格式**: `.txt` 文件，每行一帧

```
0 500.0 500.0 320.0 240.0 0.0 0.0 0.9 0.1 0.2 1.0 0.3 -0.5 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
1 500.0 500.0 320.0 240.0 0.0 0.0 0.85 0.12 0.25 1.1 ...
...
```

**示例代码**:

```python
def save_camera_trajectory(frames, cameras, output_file):
    """
    Args:
        frames: 帧列表 (或帧数)
        cameras: Camera 对象列表
        output_file: 输出 txt 文件
    """
    with open(output_file, 'w') as f:
        for idx, camera in enumerate(cameras):
            # 内参
            fx, fy = camera.intrinsics['fx'], camera.intrinsics['fy']
            cx, cy = camera.intrinsics['cx'], camera.intrinsics['cy']
            
            # 畸变
            k1, k2 = 0.0, 0.0  # 如无畸变信息，设为 0
            
            # 外参 (w2c 3×4 矩阵，行优先)
            w2c = camera.w2c.flatten().tolist()  # [12 values]
            
            # 写入一行
            line = f"{idx} {fx} {fy} {cx} {cy} {k1} {k2} " + " ".join(map(str, w2c))
            f.write(line + "\n")

def load_camera_trajectory(camera_file):
    """读取相机轨迹文件"""
    cameras = []
    with open(camera_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            
            # 解析
            frame_idx = int(values[0])
            fx, fy, cx, cy = values[1:5]
            k1, k2 = values[5:7]
            w2c_flat = values[7:19]
            
            # 重组 w2c 矩阵 (3×4)
            w2c = np.array(w2c_flat).reshape(3, 4)
            
            camera = Camera(
                frame_idx=frame_idx,
                intrinsics={'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy},
                distortion={'k1': k1, 'k2': k2},
                w2c=w2c
            )
            cameras.append(camera)
    
    return cameras
```

**获取相机参数**:

相机参数通常来自：
1. **DUSt3R** : 从多视图重建得到
2. **ViPE** : 现在已有此pipeline
3. **COLMAP**: 结构光重建工具
4. **手动标注**: 如果有外参设备
5. **估计**: 从 MVS 结果反推

---

## 🔄 数据生成管道

### 完整流程

```
原始视频
    ↓
1. 提取帧 (ffmpeg)
    ↓
2. 生成深度图 (Depth Anything V2)
    ↓
3. 估计点云 (DUSt3R)
    ↓
4. 估计相机参数 (COLMAP 或 DUSt3R)
    ↓
5. 数据组织与验证
    ↓
完成！
```

### 详细步骤

#### 步骤 1: 提取视频帧

```bash
# 使用 ffmpeg 提取帧
ffmpeg -i video.mp4 -q:v 2 frame_%04d.jpg

# 或使用 Python
python << 'EOF'
import cv2
import os

video_path = "video.mp4"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{output_dir}/frame_{idx:04d}.jpg", frame)
    idx += 1
cap.release()

print(f"提取了 {idx} 帧")
EOF
```

#### 步骤 2: 生成深度图 (Depth Anything V2)

```bash
# 安装 Depth Anything V2
pip install -e git+https://github.com/DepthAnything/Depth-Anything-V2.git

# 运行推理
python << 'EOF'
from depth_anything_v2.dpt import DepthAnythingV2

# 初始化模型
model = DepthAnythingV2(
    encoder='vitb',  # vitb, vitl, vitg
    features=256,
    out_channels=[48, 96, 192, 384]
)
model.eval()

import cv2
import numpy as np
import os

frames_dir = "frames"
depth_dir = "depth"
os.makedirs(depth_dir, exist_ok=True)

for frame_file in sorted(os.listdir(frames_dir)):
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    
    # 推理
    with torch.no_grad():
        depth = model.infer_image(frame)
    
    # 保存
    frame_idx = frame_file.replace("frame_", "").replace(".jpg", "")
    np.save(f"{depth_dir}/frame_{frame_idx}.npy", depth)

print("深度图生成完成")
EOF
```

#### 步骤 3: 估计点云 (DUSt3R)

```bash
# 安装 DUSt3R
pip install -e git+https://github.com/naver/dust3r.git

# 运行推理
python << 'EOF'
import torch
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
import cv2
import numpy as np
import os

# 初始化模型
model = AsymmetricCroCo3DStereo.from_pretrained(
    "naver/DUSt3R_ViTLarge_BaseDecoder_224_linear"
).eval()

frames_dir = "frames"
points_dir = "points"
os.makedirs(points_dir, exist_ok=True)

frame_files = sorted(os.listdir(frames_dir))

# 处理所有帧对 (或仅邻近帧对以加快速度)
for i in range(0, len(frame_files) - 1):
    frame1_path = os.path.join(frames_dir, frame_files[i])
    frame2_path = os.path.join(frames_dir, frame_files[i + 1])
    
    img1 = cv2.imread(frame1_path)
    img2 = cv2.imread(frame2_path)
    
    # DUSt3R 推理
    with torch.no_grad():
        output = inference([img1, img2], model, device='cuda')
    
    # 提取点云 (三角化或其他方法)
    points1 = extract_points(output['view1'])
    points2 = extract_points(output['view2'])
    
    # 保存
    np.save(f"{points_dir}/frame_{i:04d}.npy", points1)
    np.save(f"{points_dir}/frame_{i+1:04d}.npy", points2)

print("点云估计完成")
EOF
```

#### 步骤 4: 估计相机参数

**选项 A: 使用 COLMAP** (推荐精度)

```bash
# 安装 COLMAP
sudo apt-get install colmap

# 运行 COLMAP
colmap feature_extractor \
    --database_path database.db \
    --image_path frames/

colmap exhaustive_matcher \
    --database_path database.db

colmap mapper \
    --database_path database.db \
    --image_path frames/ \
    --output_path colmap_output

# 导出相机参数
python << 'EOF'
# 从 COLMAP output 提取相机参数
# 详见 COLMAP 文档或 Fantasy World 示例代码
EOF
```

**选项 B: 使用 DUSt3R 输出** (更快)

```python
# DUSt3R 的输出中已包含相机参数
# 直接从 DUSt3R 的 camera matrics 提取
def extract_cameras_from_dust3r(dust3r_output):
    cameras = []
    for view in dust3r_output['views']:
        # 提取内参和外参
        K = view['K']  # 3×3 内参矩阵
        w2c = view['w2c']  # 3×4 外参矩阵
        
        camera = {
            'fx': K[0, 0],
            'fy': K[1, 1],
            'cx': K[0, 2],
            'cy': K[1, 2],
            'k1': 0.0,
            'k2': 0.0,
            'w2c': w2c
        }
        cameras.append(camera)
    return cameras
```

#### 步骤 5: 数据组织

```bash
# 创建最终的数据集结构
mkdir -p dataset/sample_001/{depth,points}

# 复制文件
cp video.mp4 dataset/sample_001/
cp -r depth/* dataset/sample_001/depth/
cp -r points/* dataset/sample_001/points/
cp camera_params.txt dataset/sample_001/

# 验证
find dataset/sample_001/ -type f | sort
```

---

## 📁 数据集组织结构

### 完整结构

```
fantasy_world_dataset/
│
├── metadata.json                 # 数据集元数据 (必需)
│
├── sample_001/
│   ├── video.mp4                # 原始视频
│   ├── depth/
│   │   ├── frame_0000.npy
│   │   ├── frame_0001.npy
│   │   └── ...
│   ├── points/
│   │   ├── frame_0000.npy
│   │   ├── frame_0001.npy
│   │   └── ...
│   └── camera_params.txt        # 19-value 相机轨迹
│
├── sample_002/
│   ├── video.mp4
│   ├── depth/
│   ├── points/
│   └── camera_params.txt
│
└── sample_N/
    ├── ...
```

### 关键要点

1. **metadata.json** 位置: 数据集根目录
2. **视频文件名**: 必须是 `video.mp4` (或指定扩展名)
3. **深度文件命名**: `frame_0000.npy` 格式，从 0 开始编号
4. **点云文件命名**: 同深度文件
5. **相机文件名**: 必须是 `camera_params.txt`

---

## 📝 元数据配置

### metadata.json 格式

```json
{
    "version": "1.0",
    "description": "Fantasy World Training Dataset",
    "samples": [
        {
            "id": "sample_001",
            "video_path": "sample_001/video.mp4",
            "depth_dir": "sample_001/depth",
            "points_dir": "sample_001/points",
            "camera_file": "sample_001/camera_params.txt",
            "num_frames": 81,
            "height": 336,
            "width": 592,
            "fps": 24,
            "scene_type": "indoor",
            "camera_motion": "orbit",
            "note": "Living room with camera rotation"
        },
        {
            "id": "sample_002",
            "video_path": "sample_002/video.mp4",
            "depth_dir": "sample_002/depth",
            "points_dir": "sample_002/points",
            "camera_file": "sample_002/camera_params.txt",
            "num_frames": 81,
            "height": 336,
            "width": 592,
            "fps": 24,
            "scene_type": "outdoor",
            "camera_motion": "forward",
            "note": "Park with camera forward movement"
        }
    ],
    "splits": {
        "train": ["sample_001", "sample_002", ...],
        "val": ["sample_100", "sample_101", ...],
        "test": ["sample_200", "sample_201", ...]
    },
    "statistics": {
        "total_samples": 1000,
        "train_count": 800,
        "val_count": 100,
        "test_count": 100,
        "total_frames": 81000,
        "avg_resolution": "336x592"
    }
}
```

### 生成 metadata.json

```python
import json
import os
from pathlib import Path

def create_metadata(dataset_dir, output_file):
    """自动生成 metadata.json"""
    
    samples = []
    sample_dirs = sorted([d for d in os.listdir(dataset_dir) 
                         if os.path.isdir(os.path.join(dataset_dir, d))])
    
    for sample_id in sample_dirs:
        sample_path = os.path.join(dataset_dir, sample_id)
        
        # 检查必需文件
        video_file = os.path.join(sample_path, "video.mp4")
        depth_dir = os.path.join(sample_path, "depth")
        points_dir = os.path.join(sample_path, "points")
        camera_file = os.path.join(sample_path, "camera_params.txt")
        
        if not all(os.path.exists(p) for p in [video_file, depth_dir, points_dir, camera_file]):
            print(f"⚠️ {sample_id} 缺少必需文件，跳过")
            continue
        
        # 获取帧数
        num_frames = len([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
        
        # 获取分辨率 (从第一帧深度图)
        import cv2
        import numpy as np
        first_depth = np.load(os.path.join(depth_dir, "frame_0000.npy"))
        height, width = first_depth.shape
        
        sample = {
            "id": sample_id,
            "video_path": f"{sample_id}/video.mp4",
            "depth_dir": f"{sample_id}/depth",
            "points_dir": f"{sample_id}/points",
            "camera_file": f"{sample_id}/camera_params.txt",
            "num_frames": num_frames,
            "height": int(height),
            "width": int(width),
            "fps": 24,  # 可根据需要调整
            "scene_type": "unknown",
            "camera_motion": "unknown",
            "note": ""
        }
        samples.append(sample)
    
    # 分割数据集
    total = len(samples)
    train_count = int(total * 0.8)
    val_count = int(total * 0.1)
    
    train_samples = [s["id"] for s in samples[:train_count]]
    val_samples = [s["id"] for s in samples[train_count:train_count+val_count]]
    test_samples = [s["id"] for s in samples[train_count+val_count:]]
    
    metadata = {
        "version": "1.0",
        "description": "Fantasy World Training Dataset",
        "samples": samples,
        "splits": {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples
        },
        "statistics": {
            "total_samples": total,
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "test_count": len(test_samples),
            "total_frames": total * num_frames,
            "avg_resolution": f"{height}x{width}"
        }
    }
    
    # 保存
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ metadata.json 生成完成: {output_file}")
    print(f"   总样本数: {total}")
    print(f"   训练: {len(train_samples)}, 验证: {len(val_samples)}, 测试: {len(test_samples)}")

# 使用示例
create_metadata("fantasy_world_dataset", "fantasy_world_dataset/metadata.json")
```

---

## ✅ 数据验证与质量检查

### 验证脚本

```python
def validate_dataset(dataset_dir, metadata_file):
    """验证数据集完整性和质量"""
    
    import json
    import numpy as np
    from pathlib import Path
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    issues = []
    
    for sample in metadata['samples']:
        sample_id = sample['id']
        sample_dir = os.path.join(dataset_dir, sample_id)
        
        # 1. 检查视频文件
        video_file = os.path.join(sample_dir, sample['video_path'])
        if not os.path.exists(video_file):
            issues.append(f"❌ {sample_id}: 视频文件不存在")
        
        # 2. 检查深度图
        depth_dir = os.path.join(sample_dir, sample['depth_dir'])
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
        if len(depth_files) != sample['num_frames']:
            issues.append(f"⚠️ {sample_id}: 深度图数量不匹配 ({len(depth_files)} vs {sample['num_frames']})")
        
        # 检查深度图值域
        first_depth = np.load(os.path.join(depth_dir, depth_files[0]))
        if first_depth.max() > 1.1:  # 假设应该归一化到 0-1
            issues.append(f"⚠️ {sample_id}: 深度图值域异常 (max={first_depth.max()}, 应该 ≤ 1)")
        
        # 3. 检查点云
        points_dir = os.path.join(sample_dir, sample['points_dir'])
        points_files = sorted([f for f in os.listdir(points_dir) if f.endswith('.npy')])
        if len(points_files) != sample['num_frames']:
            issues.append(f"⚠️ {sample_id}: 点云数量不匹配")
        
        # 检查点云维度
        first_points = np.load(os.path.join(points_dir, points_files[0]))
        if first_points.shape != (sample['height'], sample['width'], 3):
            issues.append(f"⚠️ {sample_id}: 点云维度不匹配 ({first_points.shape})")
        
        # 4. 检查相机文件
        camera_file = os.path.join(sample_dir, sample['camera_file'])
        if not os.path.exists(camera_file):
            issues.append(f"❌ {sample_id}: 相机文件不存在")
        else:
            with open(camera_file, 'r') as f:
                camera_lines = f.readlines()
            if len(camera_lines) != sample['num_frames']:
                issues.append(f"⚠️ {sample_id}: 相机参数行数不匹配 ({len(camera_lines)} vs {sample['num_frames']})")
    
    # 输出结果
    if issues:
        print("🔍 验证结果:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ 所有检查通过！数据集有效")
        return True

# 使用
validate_dataset("fantasy_world_dataset", "fantasy_world_dataset/metadata.json")
```

### 质量检查清单

```
数据集完整性检查:
✅ metadata.json 存在且格式正确
✅ 所有样本目录存在
✅ 每个样本都有 video.mp4
✅ 深度图数量与帧数匹配
✅ 点云数量与帧数匹配
✅ 相机文件行数与帧数匹配

数据质量检查:
✅ 视频无损坏或黑帧
✅ 深度图值在合理范围 (0-1 或 0-255)
✅ 点云不是全 NaN 或无穷大
✅ 相机参数在合理范围
✅ 深度图和点云的分辨率匹配

统计检查:
✅ 训练/验证/测试集数量合理
✅ 数据集大小足够 (≥ 100 个样本)
✅ 无重复样本
```

---

## 🐛 常见数据问题解决

### 问题 1: 深度图值域不对

**症状**: 训练时深度 loss 异常大

**解决方案**:
```python
# 检查并修复
def fix_depth_values(depth_dir):
    import numpy as np
    import os
    
    for file in os.listdir(depth_dir):
        if not file.endswith('.npy'):
            continue
        
        depth = np.load(os.path.join(depth_dir, file))
        
        # 如果最大值 > 1.5，进行归一化
        if depth.max() > 1.5:
            depth = depth / 255.0  # 假设原始值 0-255
            np.save(os.path.join(depth_dir, file), depth)
            print(f"已修复: {file}")
```

### 问题 2: 点云包含 NaN

**症状**: 训练崩溃或梯度为 NaN

**解决方案**:
```python
# 检查和清理
def clean_points(points_dir):
    import numpy as np
    import os
    
    for file in os.listdir(points_dir):
        if not file.endswith('.npy'):
            continue
        
        points = np.load(os.path.join(points_dir, file))
        
        # 替换 NaN 为 0
        if np.isnan(points).any():
            print(f"⚠️ {file} 包含 NaN，正在修复...")
            points = np.nan_to_num(points, nan=0.0)
            np.save(os.path.join(points_dir, file), points)
```

### 问题 3: 相机文件格式错误

**症状**: "无法解析相机文件" 错误

**解决方案**:
```python
# 检查和修复格式
def verify_camera_format(camera_file):
    with open(camera_file, 'r') as f:
        for line_idx, line in enumerate(f):
            values = line.strip().split()
            
            # 检查值数量 (应该是 19)
            if len(values) != 19:
                print(f"❌ 第 {line_idx} 行: 值数量 {len(values)} != 19")
                return False
            
            # 尝试转换为浮点数
            try:
                values = [float(v) for v in values]
            except ValueError as e:
                print(f"❌ 第 {line_idx} 行: 无法转换为数字 - {e}")
                return False
    
    print("✅ 相机文件格式正确")
    return True

# 使用
verify_camera_format("sample_001/camera_params.txt")
```

### 问题 4: 分辨率不一致

**症状**: "维度不匹配" 错误

**解决方案**:
```python
# 重新调整所有数据到统一分辨率
def resize_all_data(sample_dir, target_height=336, target_width=592):
    import cv2
    import numpy as np
    import os
    from PIL import Image
    
    # 调整视频 (使用 ffmpeg)
    import subprocess
    video_file = os.path.join(sample_dir, "video.mp4")
    subprocess.run([
        'ffmpeg', '-i', video_file,
        '-vf', f'scale={target_width}:{target_height}',
        '-y',  # 覆盖输出文件
        os.path.join(sample_dir, "video_resized.mp4")
    ])
    
    # 调整深度图
    depth_dir = os.path.join(sample_dir, "depth")
    for file in os.listdir(depth_dir):
        if not file.endswith('.npy'):
            continue
        
        depth = np.load(os.path.join(depth_dir, file))
        depth_resized = cv2.resize(depth, (target_width, target_height))
        np.save(os.path.join(depth_dir, file), depth_resized)
    
    # 类似地处理点云
    points_dir = os.path.join(sample_dir, "points")
    for file in os.listdir(points_dir):
        if not file.endswith('.npy'):
            continue
        
        points = np.load(os.path.join(points_dir, file))
        # 调整空间维度，保持 3 通道
        points_resized = cv2.resize(
            points,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )
        np.save(os.path.join(points_dir, file), points_resized)
    
    print("✅ 所有数据已调整到统一分辨率")
```

---

**下一步**: 查看 [训练指南](./TRAINING_GUIDE.md) 开始训练！
