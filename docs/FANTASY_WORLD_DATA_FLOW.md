# Fantasy World Data Flow 数据流说明

## 概述

本文档说明 Fantasy World 训练管线中相机位姿数据的流转逻辑，确保符合 DiffSynth 的架构设计。

## 架构原则

DiffSynth 的核心设计：
1. **数据加载**: Dataset 读取原始数据（文件路径、张量等）
2. **inputs_shared 封装**: 训练脚本将数据封装进 `inputs_shared` 字典
3. **Units 流水线**: 各个 `PipelineUnit` 从 `inputs_shared` 获取所需参数并处理
4. **model_fn 执行**: 处理后的数据传入 model 函数进行前向计算

## 相机位姿数据流

### 1. 数据格式

相机位姿存储在 `.txt` 文件中，每行格式：
```
[prefix values...] [w2c_00 w2c_01 ... w2c_23]
```

- **前缀值**: 任意数量，会被忽略
- **后12个值**: 3×4 w2c 矩阵的展平形式（行优先）
  - 前9个值: 3×3 旋转矩阵
  - 后3个值: 3D 平移向量

示例：
```
0 0.5 0.8 0.5 0.5 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
```

### 2. 数据流转路径

#### Step 1: Dataset 加载 (UnifiedDataset)

```python
# train.py
special_operator_map = {
    "camera_params": default_camera_operator(base_path=args.dataset_base_path)
}

dataset = UnifiedDataset(
    data_file_keys=("video", "depth", "points", "camera_params"),
    special_operator_map=special_operator_map,
)
```

**输出**: `data["camera_params"]` = `"/path/to/cameras/sample_0000.txt"` (字符串路径)

#### Step 2: 封装到 inputs_shared (train.py)

```python
# WanTrainingModule.parse_extra_inputs()
if extra_input == "pose_file_path":
    if "camera_params" in data:
        inputs_shared["pose_file_path"] = data["camera_params"]
```

**关键映射**:
- `data["camera_params"]` → `inputs_shared["pose_file_path"]`
- `data["camera_params"]` → `inputs_shared["gt_camera_file"]` (用于loss计算)

#### Step 3: Unit 处理 (WanVideoUnit_FunCameraControl)

```python
# wan_video.py
class WanVideoUnit_FunCameraControl(PipelineUnit):
    def process(self, pipe, pose_file_path, ...):
        if pose_file_path is not None:
            # 解析 txt 文件
            # 提取 w2c 矩阵
            # 生成相机 token
            # 存储到 inputs["control_camera_latents_input"]
```

**功能**: 
- 读取 `pose_file_path` 指向的 txt 文件
- 解析最后12个值作为 w2c 矩阵
- 生成相机控制 latents
- 传递给 DiT 模型

#### Step 4: Loss 计算 (loss.py)

```python
# FantasyWorldLoss()
gt_camera_file = inputs.get("gt_camera_file", None)
if gt_camera_file is not None:
    gt_w2c = parse_camera_txt(gt_camera_file, num_frames=...)
    # 与模型预测的相机参数比较
    loss_cam = huber_loss(pred_cam, gt_w2c)
```

**功能**:
- 从 `gt_camera_file` 读取 ground truth
- 解析 w2c 矩阵
- 计算相机损失

### 3. 关键代码位置

| 文件 | 函数/类 | 作用 |
|------|---------|------|
| `fantasy_world_operators.py` | `LoadCameraParams` | 返回 txt 文件路径（不解析） |
| `train.py` | `parse_extra_inputs()` | 将 `camera_params` 映射到 `pose_file_path` 和 `gt_camera_file` |
| `wan_video.py` | `WanVideoUnit_FunCameraControl` | 解析 txt 文件，生成相机控制信号 |
| `loss.py` | `parse_camera_txt()` | 解析 txt 文件提取 w2c 矩阵 |
| `loss.py` | `FantasyWorldLoss()` | 计算相机损失 |

## 测试验证

### 运行数据流测试

```bash
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
python test_data_flow.py
```

**预期输出**:
```
Loading dataset from: /path/to/test_data/fantasy_world_fake
Dataset length: 4

Loading sample 0...
Data keys: ['video', 'prompt', 'depth', 'points', 'camera_params']
✓ video: 9 frames
✓ depth: torch.Size([9, 128, 128])
✓ points: torch.Size([9, 128, 128, 3])
✓ camera_params: /path/to/cameras/sample_0000.txt
  ✓ File exists and is readable
  ✓ Contains 9 lines
  ✓ First line has 19 values
  ✓ Last 12 values (w2c): [1.0, 0.0, 0.0, 0.0]...

Data loading test PASSED ✓
```

### 运行训练测试

```bash
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
bash examples/wanvideo/model_training/full/test_fantasy_world_training.sh
```

## 常见问题

### Q1: KeyError: 'pose_params'

**原因**: 使用 `UnifiedDataset` 而不是 `FantasyWorldDataset`，key 名称不匹配

**解决**: 
- `train.py` 中使用 `data["camera_params"]` 而不是 `data["pose_params"]`
- 正确映射到 `inputs_shared["pose_file_path"]`

### Q2: 相机损失无法计算

**原因**: `inputs` 中没有 `gt_camera_file`

**解决**: 
- 确保 `parse_extra_inputs()` 设置了 `inputs_shared["gt_camera_file"]`
- 确保 `data_file_keys` 包含 `"camera_params"`

### Q3: txt 文件格式错误

**原因**: 生成的 txt 文件格式不正确

**解决**: 
- 使用 `scripts/create_fake_data.py` 生成标准格式数据
- 每行至少包含12个数值
- 最后12个值是 w2c 矩阵

## 总结

Fantasy World 的相机位姿数据流遵循 DiffSynth 的 **"数据加载 → inputs封装 → Units处理 → model执行"** 范式：

1. ✅ Dataset 返回文件路径（不解析）
2. ✅ train.py 将路径映射到 `pose_file_path` 和 `gt_camera_file`
3. ✅ WanVideoUnit_FunCameraControl 解析并处理相机控制
4. ✅ FantasyWorldLoss 解析并计算相机损失

这样设计使得：
- **解耦**: 数据加载与处理分离
- **灵活**: 可在不同阶段处理相同数据
- **高效**: 避免重复解析
