# 🎓 详细训练指南 - 两阶段完整版

本文档详细说明 Fantasy World 的两阶段训练策略、配置、监控和优化方法。

---

## 📋 目录

1. [两阶段训练策略](#两阶段训练策略)
2. [环境与准备](#环境与准备)
3. [Stage 1: Latent Bridging](#stage-1-latent-bridging)
4. [Stage 2: Unified Co-Optimization](#stage-2-unified-co-optimization)
5. [训练监控与调试](#训练监控与调试)
6. [性能优化](#性能优化)
7. [故障恢复与检查点管理](#故障恢复与检查点管理)

---

## 🔄 两阶段训练策略

### 为什么需要两阶段？

**问题**: 直接联合训练所有模块导致：
- ❌ Geometry branch 无法有效学习 (video features 不稳定)
- ❌ 梯度冲突 (不同分支学习目标不一致)
- ❌ 训练不稳定 (loss 震荡，收敛困难)
- ❌ 模型性能差 (最终输出不理想)

**解决方案**: 两阶段策略

```
Stage 1 (稳定适配)              Stage 2 (双向交互)
─────────────────              ──────────────────
冻结视频分支                    继续冻结视频分支
    ↓                                ↓
让几何分支适配                   添加交互模块
    ↓                                ↓
几何 loss 快速下降              整体 loss 微调优化
    ↓                                ↓
收敛到局部最优                   找到全局更优解
```

### 论文设计 vs 我们的实现

| 方面 | 论文设计 | 我们的实现 |
|------|---------|---------|
| **总 DiT 层数** | 40 | 30 |
| **PCB 层数** | 不明确 | 12 |
| **IRG 层数** | 不明确 | 18 |
| **Stage 1 步数** | 20,000 | 20,000 ✅ |
| **Stage 2 步数** | 10,000 | 10,000 ✅ |
| **Stage 1 batch** | 64 | 64 ✅ |
| **Stage 2 batch** | 112 | 112 ✅ |

---

## 🔧 环境与准备

### 前置要求

**硬件**:
- 8 × NVIDIA H20 或 A100 (40GB 显存)
- 500GB+ 硬盘空间 (用于检查点和临时文件)
- 网络连接 (下载模型权重)

**软件**:
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (仅 NVIDIA)
- DiffSynth-Studio (已安装)

### 环境检查

```bash
# 1. 检查 Python 版本
python --version

# 2. 检查 PyTorch
python -c "import torch; print(torch.__version__)"

# 3. 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 4. 检查 GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 5. 检查显存
python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

**预期输出**:
```
Python 3.9.18
2.0.1+cu118
True
NVIDIA H20 Tensor Core GPU
40.0 GB
```

### 项目安装

```bash
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world

# 开发模式安装 (可编辑)
pip install -e .

# 验证
python -c "import diffsynth; print('✅ 安装成功')"
```

### 数据准备

确保数据集结构正确:

```
fantasy_world_dataset/
├── metadata.json
├── sample_001/
│   ├── video.mp4
│   ├── depth/
│   ├── points/
│   └── camera_params.txt
├── sample_002/
└── ...
```

详见 [数据准备指南](./DATA_PREPARATION.md)

---

## 🟢 Stage 1: Latent Bridging

### 目标与原理

**目标**: 训练几何分支以适配到冻结的视频特征空间

**可训练模块**:
- ✅ Latent Bridge Adapter (~5M 参数)
- ✅ GeoDiT Blocks (18 layers, ~900M 参数)
- ✅ DPT Heads (Depth, Point, Camera, ~50M 参数)
- ✅ Pose Encoder (~1M 参数)
- ✅ Special Tokens (Camera + Register, ~0.01M 参数)

**冻结模块**:
- ❄️ Wan2.1 所有 30 blocks (~1616M 参数)
- ❄️ Camera Adapters (~30M 参数)
- ❄️ IRG Cross-Attention (~200M 参数)

**总可训练参数**: ~956M

### 配置文件

编辑 `examples/wanvideo/model_training/full/train_fantasy_world_stage1.sh`:

```bash
#!/bin/bash

# ====== 数据配置 ======
DATA_DIR="/path/to/fantasy_world_dataset"           # 修改这里！
DATASET_METADATA="$DATA_DIR/metadata.json"

# ====== 输出配置 ======
OUTPUT_DIR="./outputs/fantasy_world_stage1"
mkdir -p "$OUTPUT_DIR"

# ====== GPU 配置 ======
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7              # 使用 8 块 GPU
NUM_GPUS=8

# ====== 训练参数 ======
NUM_STEPS=20000
BATCH_SIZE_PER_GPU=8                               # 8 × 8 = 64 全局 batch
LEARNING_RATE=1e-5
GRADIENT_ACCUMULATION=1                            # 梯度累积步数 (可选)

# ====== 数据参数 ======
HEIGHT=336
WIDTH=592
NUM_FRAMES=21                                       # Stage 1 使用较少帧

# ====== 模型参数 ======
TASK="fantasy_world:stage1"                        # 关键：指定 stage1
TRAINABLE_MODELS="dit"                             # 训练 DiT 模块

# ====== 其他参数 ======
MIXED_PRECISION="bf16"                             # BFloat16 混合精度
FIND_UNUSED_PARAMS="--find_unused_parameters"      # 处理冻结参数 (DDP)

# ====== 运行训练 ======
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    examples/wanvideo/model_training/train.py \
    --task $TASK \
    --dataset_base_path "$DATA_DIR" \
    --dataset_metadata_path "$DATASET_METADATA" \
    --output_path "$OUTPUT_DIR" \
    --num_steps $NUM_STEPS \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation $GRADIENT_ACCUMULATION \
    --height $HEIGHT \
    --width $WIDTH \
    --num_frames $NUM_FRAMES \
    --mixed_precision $MIXED_PRECISION \
    --trainable_models $TRAINABLE_MODELS \
    $FIND_UNUSED_PARAMS
```

### 关键配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `DATA_DIR` | `/path/to/dataset` | **必改**: 你的数据集路径 |
| `TASK` | `fantasy_world:stage1` | **必须**: 指定 stage1 |
| `NUM_STEPS` | 20000 | 论文推荐值 |
| `BATCH_SIZE_PER_GPU` | 8 | 8 GPUs → 全局 batch 64 |
| `HEIGHT` × `WIDTH` | 336 × 592 | Stage 1 分辨率 |
| `NUM_FRAMES` | 21 | Stage 1 帧数 |
| `LEARNING_RATE` | 1e-5 | 标准学习率 |
| `MIXED_PRECISION` | bf16 | BFloat16 节省显存 |
| `--find_unused_parameters` | 启用 | 处理 DDP 冻结参数问题 |

### 运行 Stage 1

```bash
# 1. 编辑配置
vim examples/wanvideo/model_training/full/train_fantasy_world_stage1.sh
# 修改 DATA_DIR 为你的数据集路径

# 2. 赋予执行权限
chmod +x examples/wanvideo/model_training/full/train_fantasy_world_stage1.sh

# 3. 运行训练
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
bash examples/wanvideo/model_training/full/train_fantasy_world_stage1.sh
```

### 性能期望

**训练时间**:
- 硬件: 8 × H20
- 20K steps: ~36 小时

**Loss 期望值**:
```
Step 1000:
  - L_diffusion: 0.5-0.8
  - L_depth: 0.3-0.5
  - L_point: 1.0-1.5
  - L_camera: 0.2-0.3
  
Step 10000 (中期):
  - L_diffusion: 0.2-0.3
  - L_depth: 0.1-0.15
  - L_point: 0.4-0.6
  - L_camera: 0.05-0.1
  
Step 20000 (完成):
  - L_diffusion: 0.15-0.25
  - L_depth: < 0.1
  - L_point: 0.2-0.4
  - L_camera: < 0.05
```

**检查点输出**:

```
outputs/fantasy_world_stage1/
├── step-1000.safetensors
├── step-2000.safetensors
├── ...
└── step-20000.safetensors  # 最终 Stage 1 检查点 ⭐
```

### 质量评估

Stage 1 完成后，验证训练质量:

```python
# 加载检查点并测试推理
import torch
from diffsynth import WanVideoPipeline

checkpoint = "outputs/fantasy_world_stage1/step-20000.safetensors"

pipe = WanVideoPipeline.from_pretrained("PAI/Wan2.1-Fun-V1.1-1.3B")
pipe.dit.enable_fantasy_world_mode(training_stage="stage1")
state = torch.load(checkpoint, map_location="cpu")
pipe.dit.load_state_dict(state, strict=False)

# 测试推理
video = pipe(
    prompt="a camera moving through a room",
    num_frames=21,
    height=336,
    width=592
)

print("✅ Stage 1 推理成功")
```

---

## 🔵 Stage 2: Unified Co-Optimization

### 目标与原理

**目标**: 添加交互模块，实现视频-几何的双向交互

**新增可训练模块**:
- ✅ Camera Adapters (12 个, ~30M 参数)
- ✅ IRG Cross-Attention (18 个, ~200M 参数)

**继续训练**:
- ✅ 保留 Stage 1 的所有可训练模块

**冻结模块**:
- ❄️ Wan2.1 所有 30 blocks (始终冻结)

**总可训练参数**: ~1186M (比 Stage 1 多 230M)

### 配置文件

编辑 `examples/wanvideo/model_training/full/train_fantasy_world_stage2.sh`:

```bash
#!/bin/bash

# ====== 数据配置 ======
DATA_DIR="/path/to/fantasy_world_dataset"
DATASET_METADATA="$DATA_DIR/metadata.json"

# ====== 检查点配置 ======
STAGE1_CHECKPOINT="outputs/fantasy_world_stage1/step-20000.safetensors"  # 必需！

# ====== 输出配置 ======
OUTPUT_DIR="./outputs/fantasy_world_stage2"
mkdir -p "$OUTPUT_DIR"

# ====== GPU 配置 ======
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# ====== 训练参数 ======
NUM_STEPS=10000
BATCH_SIZE_PER_GPU=14                              # 8 × 14 = 112 全局 batch
LEARNING_RATE=1e-5
GRADIENT_ACCUMULATION=1

# ====== 数据参数 ======
HEIGHT=592
WIDTH=336                                           # 注意：分辨率与 Stage 1 互换
NUM_FRAMES=81                                       # Stage 2 使用完整帧数

# ====== 模型参数 ======
TASK="fantasy_world:stage2"                        # 关键：指定 stage2
TRAINABLE_MODELS="dit"

# ====== 其他参数 ======
MIXED_PRECISION="bf16"
FIND_UNUSED_PARAMS="--find_unused_parameters"

# ====== 前置检查 ======
if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "❌ 错误: Stage 1 检查点不存在: $STAGE1_CHECKPOINT"
    echo "请先运行 Stage 1 训练"
    exit 1
fi

# ====== 运行训练 ======
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    examples/wanvideo/model_training/train.py \
    --task $TASK \
    --stage1_checkpoint "$STAGE1_CHECKPOINT" \
    --dataset_base_path "$DATA_DIR" \
    --dataset_metadata_path "$DATASET_METADATA" \
    --output_path "$OUTPUT_DIR" \
    --num_steps $NUM_STEPS \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation $GRADIENT_ACCUMULATION \
    --height $HEIGHT \
    --width $WIDTH \
    --num_frames $NUM_FRAMES \
    --mixed_precision $MIXED_PRECISION \
    --trainable_models $TRAINABLE_MODELS \
    $FIND_UNUSED_PARAMS
```

### 关键配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `STAGE1_CHECKPOINT` | `outputs/fantasy_world_stage1/step-20000.safetensors` | **必需**: Stage 1 输出 |
| `TASK` | `fantasy_world:stage2` | **必须**: 指定 stage2 |
| `NUM_STEPS` | 10000 | 比 Stage 1 少 |
| `BATCH_SIZE_PER_GPU` | 14 | 8 GPUs → 全局 batch 112 |
| `HEIGHT` × `WIDTH` | 592 × 336 | 完整分辨率，与 Stage 1 互换 |
| `NUM_FRAMES` | 81 | 完整视频长度 |
| `LEARNING_RATE` | 1e-5 | 保持不变 |

### 运行 Stage 2

```bash
# 1. 验证 Stage 1 完成
ls -lh outputs/fantasy_world_stage1/step-20000.safetensors

# 2. 编辑 Stage 2 配置
vim examples/wanvideo/model_training/full/train_fantasy_world_stage2.sh

# 3. 赋予执行权限
chmod +x examples/wanvideo/model_training/full/train_fantasy_world_stage2.sh

# 4. 运行 Stage 2
bash examples/wanvideo/model_training/full/train_fantasy_world_stage2.sh
```

### 性能期望

**训练时间**:
- 硬件: 8 × H20
- 10K steps: ~144 小时

**Loss 期望值**:
```
Step 1000 (从 Stage 1 checkpoint 初始化):
  - L_diffusion: 0.2-0.3        # 更低的起点
  - L_depth: 0.08-0.12
  - L_point: 0.3-0.5
  - L_camera: 0.05-0.08
  
Step 5000 (中期):
  - L_diffusion: 0.12-0.18
  - L_depth: 0.05-0.08
  - L_point: 0.15-0.25
  - L_camera: 0.03-0.05
  
Step 10000 (完成):
  - L_diffusion: 0.1-0.15
  - L_depth: < 0.05
  - L_point: 0.1-0.2
  - L_camera: < 0.03
```

### 最终模型

```
outputs/fantasy_world_stage2/
└── step-10000.safetensors  # 最终模型 ⭐⭐⭐
```

这个模型包含完整的 Fantasy World 能力，可用于推理。

---

## 📊 训练监控与调试

### 实时监控

#### 方法 1: TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/fantasy_world_stage1

# 在浏览器中打开
# http://localhost:6006
```

#### 方法 2: 日志文件

```bash
# Stage 1 日志
tail -f outputs/fantasy_world_stage1/training.log

# 查看最后 100 行
tail -100 outputs/fantasy_world_stage1/training.log
```

#### 方法 3: 脚本监控

```python
# monitor_training.py
import json
from pathlib import Path
import time

def monitor_training(log_dir):
    while True:
        log_file = Path(log_dir) / "training.log"
        if log_file.exists():
            lines = log_file.read_text().strip().split('\n')
            if lines:
                last_line = lines[-1]
                print(f"\r{last_line}", end="")
        time.sleep(5)

if __name__ == "__main__":
    monitor_training("outputs/fantasy_world_stage1")
```

### 关键指标

监控这些指标判断训练状态：

| 指标 | 含义 | 正常范围 | 警告阈值 |
|------|------|---------|---------|
| `loss/diffusion` | 扩散损失 | 逐步递减 | 不降或波动大 |
| `loss/depth` | 深度预测损失 | 快速递减 | > 0.2 (Stage 1后期) |
| `loss/point` | 点云预测损失 | 逐步递减 | > 0.5 (Stage 1后期) |
| `loss/camera` | 相机参数损失 | 快速递减 | > 0.1 (Stage 1后期) |
| `learning_rate` | 学习率 | 固定或衰减 | 意外变化 |
| `gpu_memory` | GPU 显存使用 | 稳定 | 持续增长 |

### 异常排查

#### 问题 1: Loss 不下降或反向增长

**症状**:
```
Step 1000: loss = 1.5
Step 2000: loss = 1.7
Step 3000: loss = 2.0
```

**可能原因**:
1. 学习率过高
2. 数据不适配
3. 模型架构错误

**解决方案**:
```bash
# 尝试降低学习率
LEARNING_RATE=5e-6  # 原来是 1e-5

# 或检查数据有效性
python scripts/verify_dataset.py --dataset_path $DATA_DIR
```

#### 问题 2: GPU 显存溢出

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
```bash
# 选项 1: 减小 batch size
BATCH_SIZE_PER_GPU=4  # 从 8 降低

# 选项 2: 增加梯度累积
GRADIENT_ACCUMULATION=2  # 补偿 batch size 下降

# 选项 3: 启用梯度检查点 (需要代码修改)
```

#### 问题 3: Loss 震荡或不收敛

**症状**:
```
Step 5000: loss = 0.15
Step 5100: loss = 0.25
Step 5200: loss = 0.12
...
```

**解决方案**:
```bash
# 使用动态学习率调度
# 在训练脚本中配置 lr_scheduler

# 或手动降低学习率后继续
LEARNING_RATE=5e-6
```

---

## ⚙️ 性能优化

### 显存优化

**措施**:
1. **混合精度训练** (已启用)
   ```bash
   MIXED_PRECISION="bf16"  # BFloat16
   ```

2. **梯度检查点** (可选)
   ```python
   # 在 train.py 中启用
   model.gradient_checkpointing_enable()
   ```

3. **减小 batch size** (最后手段)
   ```bash
   BATCH_SIZE_PER_GPU=4  # 默认 8
   ```

### 训练速度优化

**措施**:
1. **数据预加载** (通常已自动)
2. **多 GPU 同步频率**
   ```bash
   # 每 N 步同步一次梯度
   GRADIENT_ACCUMULATION=2
   ```

3. **关闭不必要的检查**
   ```bash
   # 减少 checkpoint 保存频率
   SAVE_EVERY=500  # 不是 100
   ```

### 最大吞吐量配置

为了最大化训练速度:

```bash
NUM_STEPS=20000
BATCH_SIZE_PER_GPU=8
GRADIENT_ACCUMULATION=1
MIXED_PRECISION="bf16"
SAVE_EVERY=1000  # 不频繁保存

# 预期: 20-30 样本/秒
```

---

## 🔄 故障恢复与检查点管理

### 检查点结构

```
outputs/fantasy_world_stage1/
├── step-1000.safetensors     # 中间检查点
├── step-2000.safetensors
├── ...
├── step-20000.safetensors    # 最终检查点 ⭐
├── latest.safetensors        # 最新检查点 (硬链接)
└── training.log
```

### 恢复训练

如果训练被中断，继续从最新检查点:

```bash
# 1. 编辑脚本添加 resume 参数
# 在 train.py 调用前添加:
--resume_from_checkpoint ./outputs/fantasy_world_stage1/latest.safetensors

# 2. 重新运行脚本
bash examples/wanvideo/model_training/full/train_fantasy_world_stage1.sh
```

### 检查点验证

```python
# verify_checkpoint.py
import torch
import safetensors.torch as sf

checkpoint_path = "outputs/fantasy_world_stage1/step-20000.safetensors"

# 加载检查点
state_dict = sf.load_file(checkpoint_path)

print(f"检查点大小: {len(state_dict)} 个张量")
print(f"文件大小: {sf.safe_open(checkpoint_path, framework='pt').metadata()}")

# 统计参数
total_params = sum(p.numel() for p in state_dict.values())
print(f"总参数: {total_params / 1e6:.0f}M")

# 检查 key 名称
print("\n前 10 个 key:")
for key in list(state_dict.keys())[:10]:
    shape = state_dict[key].shape
    print(f"  {key}: {shape}")

print("\n✅ 检查点有效")
```

### 检查点清理

```bash
# 保留重要检查点，删除中间的
rm outputs/fantasy_world_stage1/step-{1000..19000}.safetensors

# 只保留最后 5 个
ls -t outputs/fantasy_world_stage1/step-*.safetensors | tail -n +6 | xargs rm
```

---

## 📈 训练流程总结

### Timeline

```
Day 1-2: 环境搭建 + 数据准备
Day 3-4: Stage 1 训练 (36 hours)
Day 5-10: Stage 2 训练 (144 hours, 6 days)
Day 11: 验证和测试

总计: 10-11 天 (包括等待 GPU 的时间)
```

### 检查清单

**开始 Stage 1 前:**
- [ ] 环境变量配置正确
- [ ] 数据集验证通过
- [ ] GPU 数量和显存充足
- [ ] Stage 1 脚本配置修改完成
- [ ] 有备份计划

**Stage 1 运行中:**
- [ ] Loss 正常下降
- [ ] GPU 利用率 > 90%
- [ ] 无错误或警告消息
- [ ] 检查点定期保存

**Stage 1 完成后:**
- [ ] Step-20000 检查点存在
- [ ] 验证了推理功能
- [ ] 备份了 Stage 1 检查点

**开始 Stage 2 前:**
- [ ] Stage 1 检查点路径正确
- [ ] Stage 2 脚本配置修改完成
- [ ] 数据集仍然可用
- [ ] GPU 内存重新清空

**Stage 2 运行中:**
- [ ] 初始 loss 从 Stage 1 继承 (较低值)
- [ ] Loss 继续下降
- [ ] 监控几何交互是否改进

**Stage 2 完成后:**
- [ ] 最终检查点生成
- [ ] 完整验证推理
- [ ] 备份最终模型

---

## 🎯 下一步

完成 Stage 2 训练后:

1. ✅ 查看 [推理指南](./INFERENCE_GUIDE.md) 使用模型
2. ✅ 参考 [故障排查](./TROUBLESHOOTING.md) 解决问题
3. ✅ 查阅 [技术深入](./TECHNICAL_DEEP_DIVE.md) 理解设计

**恭喜！** 你现在拥有一个完整的 Fantasy World 模型！ 🎉
