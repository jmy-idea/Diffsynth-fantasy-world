# 🐛 故障排查与常见问题 (FAQ)

包含常见错误、解决方案和最佳实践。

---

## 📋 目录

1. [环境问题](#环境问题)
2. [数据相关问题](#数据相关问题)
3. [训练问题](#训练问题)
4. [推理问题](#推理问题)
5. [性能问题](#性能问题)
6. [常见问题 FAQ](#常见问题-faq)

---

## 🔌 环境问题

### E1: CUDA out of memory

**错误信息**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.40 GiB (GPU 0; 40.00 GiB total capacity; ...)
```

**原因**:
- Batch size 过大
- 模型太大
- GPU 被其他进程占用

**解决**:

```bash
# 1. 清除缓存
python -c "import torch; torch.cuda.empty_cache()"

# 2. 查看 GPU 占用
nvidia-smi

# 3. 杀死其他进程
kill <PID>

# 4. 减小 batch size (在脚本中)
BATCH_SIZE_PER_GPU=4  # 从 8 改为 4
```

**预防**:
- 监控显存使用
- 使用梯度累积补偿
- 启用混合精度 (bf16)

---

### E2: "No module named 'diffsynth'"

**错误信息**:
```
ModuleNotFoundError: No module named 'diffsynth'
```

**原因**:
- diffsynth 未安装
- 虚拟环境未激活
- PYTHONPATH 未设置

**解决**:

```bash
# 1. 检查虚拟环境
which python

# 2. 激活虚拟环境
source ~/envs/fantasy_world/bin/activate

# 3. 重新安装
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
pip install -e .

# 4. 验证
python -c "import diffsynth; print('OK')"
```

---

### E3: "CUDA is not available"

**错误信息**:
```
torch.cuda.is_available() returns False
```

**原因**:
- 驱动不匹配
- CUDA 版本问题
- PyTorch 未正确安装

**解决**:

```bash
# 1. 检查驱动
nvidia-smi  # 应该显示 GPU 信息

# 2. 检查 CUDA
nvcc --version  # 应该显示 CUDA 版本

# 3. 重装 PyTorch (针对正确的 CUDA 版本)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 验证
python << 'EOF'
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
EOF
```

---

## 📊 数据相关问题

### D1: "数据集验证失败"

**错误信息**:
```
Error: Metadata not found or invalid format
```

**原因**:
- metadata.json 缺失或格式错误
- 文件路径不正确
- 数据文件损坏

**解决**:

```bash
# 1. 检查文件存在
ls fantasy_world_dataset/metadata.json

# 2. 验证 JSON 格式
python -c "import json; json.load(open('metadata.json'))"

# 3. 检查数据完整性
python << 'EOF'
import json
import os

with open("fantasy_world_dataset/metadata.json") as f:
    metadata = json.load(f)

for sample in metadata['samples']:
    sample_dir = f"fantasy_world_dataset/{sample['id']}"
    
    # 检查必需文件
    video = os.path.join(sample_dir, sample['video_path'])
    if not os.path.exists(video):
        print(f"❌ {sample['id']}: 视频文件缺失")
    
    # 检查深度图数量
    depth_dir = os.path.join(sample_dir, sample['depth_dir'])
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.npy')]
    if len(depth_files) != sample['num_frames']:
        print(f"⚠️ {sample['id']}: 深度图数量不匹配")

print("✅ 验证完成")
EOF
```

---

### D2: "深度图值域异常"

**症状**:
```
Warning: Depth values outside expected range (max=255.0, expected <= 1.0)
```

**原因**:
- 深度图未归一化
- 生成工具输出格式不同

**解决**:

```python
import numpy as np
import os

depth_dir = "fantasy_world_dataset/sample_001/depth"

for file in os.listdir(depth_dir):
    if not file.endswith('.npy'):
        continue
    
    depth = np.load(os.path.join(depth_dir, file))
    
    # 如果值在 0-255 范围，转换为 0-1
    if depth.max() > 1.5:
        depth = depth / 255.0
        np.save(os.path.join(depth_dir, file), depth)
        print(f"已修复: {file}")
```

---

### D3: "点云包含 NaN"

**症状**:
```
Warning: NaN values detected in point cloud
```

**原因**:
- 点云估计失败
- 文件损坏

**解决**:

```python
import numpy as np
import os

def clean_points(points_dir):
    for file in os.listdir(points_dir):
        if not file.endswith('.npy'):
            continue
        
        points = np.load(os.path.join(points_dir, file))
        
        # 检查并替换 NaN
        if np.isnan(points).any():
            print(f"⚠️ {file} 包含 NaN，正在修复...")
            # 选项 1: 替换为 0
            points = np.nan_to_num(points, nan=0.0)
            # 选项 2: 使用中值
            # points[np.isnan(points)] = np.nanmedian(points)
            np.save(os.path.join(points_dir, file), points)

clean_points("fantasy_world_dataset/sample_001/points")
```

---

## 🎓 训练问题

### T1: "Loss 不下降或反向增长"

**症状**:
```
Step 1000: loss = 1.5
Step 2000: loss = 1.7
Step 3000: loss = 2.0
```

**原因**:
- 学习率过高
- 数据质量问题
- 模型架构错误

**解决**:

```bash
# 1. 降低学习率
LEARNING_RATE=5e-6  # 从 1e-5 改为 5e-6

# 2. 检查数据
python scripts/verify_dataset.py --dataset_path fantasy_world_dataset

# 3. 检查模型初始化
python << 'EOF'
import torch
from diffsynth import WanVideoPipeline

pipe = WanVideoPipeline.from_pretrained(...)
pipe.dit.enable_fantasy_world_mode()

# 输出初始 loss
x = torch.randn(1, 81, 1536)
# ... 计算 loss
EOF
```

---

### T2: "Loss 震荡不收敛"

**症状**:
```
Step 5000: loss = 0.15
Step 5100: loss = 0.25
Step 5200: loss = 0.12
```

**原因**:
- 学习率调度不当
- Batch size 过小
- 梯度不稳定

**解决**:

```bash
# 1. 启用学习率预热
# 在训练脚本中配置 warmup

# 2. 增加 batch size
BATCH_SIZE_PER_GPU=8  # 从 4 改为 8

# 3. 启用梯度裁剪
# 在优化器中设置 max_grad_norm=1.0
```

---

### T3: "DDP 报错: 'unused parameters'"

**错误**:
```
RuntimeError: Expected to have finished reduction in the backward pass before final callback...
```

**原因**:
- Stage 1 中某些模块冻结
- DDP 不知道参数不需要梯度

**解决**:

```bash
# 在训练脚本中添加
--find_unused_parameters

# 在 train.py 中
model = torch.nn.parallel.DistributedDataParallel(
    model,
    find_unused_parameters=True  # 关键
)
```

---

### T4: "检查点加载失败"

**错误**:
```
KeyError: 'expected key not found in checkpoint'
```

**原因**:
- 检查点架构不匹配
- 模块命名不一致

**解决**:

```python
import torch

# 使用 strict=False 允许不匹配
checkpoint = torch.load("checkpoint.pt", map_location="cpu")
model.load_state_dict(checkpoint, strict=False)

# 检查缺失的键
model_keys = set(model.state_dict().keys())
checkpoint_keys = set(checkpoint.keys())

missing = model_keys - checkpoint_keys
extra = checkpoint_keys - model_keys

print(f"缺失键: {len(missing)}")
print(f"额外键: {len(extra)}")
```

---

## 🎬 推理问题

### I1: "推理输出全黑"

**症状**:
```
输出视频完全黑色或无效
```

**原因**:
- 检查点加载错误
- 模型未正确初始化

**解决**:

```python
import torch
from diffsynth import WanVideoPipeline

# 验证检查点
checkpoint = torch.load("checkpoint.pt", map_location="cpu")
print(f"检查点大小: {len(checkpoint)} 键")
print(f"首个键: {list(checkpoint.keys())[0]}")

# 重新加载
pipe = WanVideoPipeline.from_pretrained(...)
pipe.dit.enable_fantasy_world_mode(training_stage="stage2")
state_dict = torch.load("checkpoint.pt", map_location="cpu")
pipe.dit.load_state_dict(state_dict, strict=False)

# 测试简单推理
video = pipe(prompt="test", num_frames=21, num_inference_steps=10)
print(f"视频范围: {video.min():.3f} ~ {video.max():.3f}")
assert video.max() > 0, "视频为零"
```

---

### I2: "内存不足 (推理时)"

**症状**:
```
CUDA out of memory during inference
```

**原因**:
- 分辨率过高
- 帧数过多
- 推理步数过多

**解决**:

```python
# 方案 1: 降低分辨率
video = pipe(
    prompt="...",
    num_frames=41,  # 从 81 改为 41
    height=224,     # 从 336 改为 224
    width=384       # 从 592 改为 384
)

# 方案 2: 启用内存优化
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

# 方案 3: 减少推理步数
video = pipe(
    prompt="...",
    num_inference_steps=30  # 从 50 改为 30
)
```

---

### I3: "推理速度很慢"

**症状**:
```
每帧需要 2-3 秒，81 帧需要 3-4 分钟
```

**原因**:
- 推理步数过多
- 未启用优化

**解决**:

```python
# 1. 减少推理步数
num_steps = 30  # 推荐 30-50

# 2. 启用优化
import torch
from diffsynth import WanVideoPipeline

pipe = WanVideoPipeline.from_pretrained(
    ...,
    torch_dtype=torch.float16  # 使用 float16 加快速度
)
pipe.enable_xformers_memory_efficient_attention()

# 3. 使用更快的 scheduler
from diffusers import EulerDiscreteScheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(...)

# 预期速度: 50-100 ms/step, 总共 1-2 分钟
```

---

## ⚡ 性能问题

### P1: "GPU 利用率低"

**症状**:
```
nvidia-smi 显示 GPU 占用 < 50%
```

**原因**:
- Batch size 过小
- 数据加载速度慢

**解决**:

```bash
# 1. 增加 batch size
BATCH_SIZE_PER_GPU=16  # 从 8 改为 16

# 2. 增加 num_workers
--num_workers 8

# 3. 启用 pin_memory
# 在数据加载器中: pin_memory=True
```

---

### P2: "训练速度慢于预期"

**症状**:
```
实际: 5 样本/秒
预期: 20+ 样本/秒
```

**原因**:
- GPU 利用率低
- I/O 瓶颈

**解决**:

```bash
# 1. 监控 GPU 使用
watch -n 1 nvidia-smi

# 2. 检查数据加载时间
python << 'EOF'
import time
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=64, num_workers=8)

for i, batch in enumerate(loader):
    if i == 0:
        # 第一个 batch 的时间包括初始化
        continue
    
    start = time.time()
    # 模型前向传播
    time_forward = time.time() - start
    
    if i > 10:
        break

print(f"平均时间: {time_forward:.3f}s/batch")
EOF

# 3. 使用 SSD 而不是 HDD
```

---

### P3: "显存占用过多"

**症状**:
```
nvidia-smi 显示使用 35-40GB (接近上限)
```

**原因**:
- Batch size 过大
- 梯度积累

**解决**:

```bash
# 1. 使用混合精度 (推荐)
MIXED_PRECISION="bf16"

# 2. 启用梯度检查点
# 在模型中: model.gradient_checkpointing_enable()

# 3. 启用分布式数据并行的梯度同步优化
# 在 DDP 中: gradient_as_bucket_view=True
```

---

## ❓ 常见问题 FAQ

### Q1: 训练需要多长时间？

**A**: 
- Stage 1: ~36 小时 (8 × H20, 20K steps)
- Stage 2: ~144 小时 (8 × H20, 10K steps)
- 总计: 180 小时 (7.5 天 GPU 时间)

实际墙钟时间取决于队列等待时间。

---

### Q2: 可以用更少的 GPU 训练吗？

**A**: 可以，但需要调整参数：

```bash
# 4 × GPU (而非 8 ×)
BATCH_SIZE_PER_GPU=16  # 4 × 16 = 64 (保持全局 batch)
GRADIENT_ACCUMULATION=1

# 预期时间: 36 × 2 = 72 小时 (翻倍)
```

---

### Q3: 可以在 Google Colab 上训练吗？

**A**: 可以进行推理，但不能训练：
- Colab GPU: 单个 T4 (16GB) 或 A100 (40GB)
- 需要: 8 × GPU

**替代方案**:
- Google Cloud TPU
- AWS 或 Azure
- Lambda Labs
- 本地 GPU 服务器

---

### Q4: 如何从中断的位置继续训练？

**A**:

```bash
# 最新检查点会自动保存到 latest.safetensors
# 添加 resume 参数继续训练

--resume_from_checkpoint ./outputs/fantasy_world_stage1/latest.safetensors
```

---

### Q5: Stage 1 和 Stage 2 都需要吗？

**A**: 是的。原因：
- Stage 1: 让几何分支适配到稳定的视频特征
- Stage 2: 添加交互模块以改进质量

直接用 Stage 2 会不收敛。

---

### Q6: 可以微调预训练的模型吗？

**A**: 是的，有两种方式：

**方式 1: 继续训练**
```bash
--resume_from_checkpoint outputs/fantasy_world_stage2/step-10000.safetensors
--num_steps 5000  # 额外 5K 步
```

**方式 2: LoRA 微调** (需要额外实现)
```
使用 LoRA 在现有权重上进行低秩调整
```

---

### Q7: 推理需要什么硬件？

**A**:
- 最低: 任何 12GB+ VRAM 的 GPU
- 推荐: RTX 3090 或 A6000
- 时间: 1-2 分钟 / 视频 (81 帧)

可在 CPU 上运行，但会非常慢 (10-20 分钟)。

---

### Q8: 如何改进生成质量？

**A**:

| 方法 | 效果 | 成本 |
|------|------|------|
| 增加 inference steps | 高 | 时间翻倍 |
| 改进提示词 | 中等 | 免费 |
| 增加数据训练 | 高 | 200+ 小时 GPU |
| 调整 guidance scale | 低 | 免费 |
| 使用更好的初始化 | 中等 | 需要修改代码 |

---

### Q9: 支持哪些数据格式？

**A**:
- 视频: MP4, AVI, MOV, WebM
- 深度图: NPY 格式
- 点云: NPY 格式
- 相机: TXT 格式 (19-value)

详见 [数据准备](./DATA_PREPARATION.md)

---

### Q10: 如何扩展模型到更多功能？

**A**:

目前 Fantasy World 支持：
- ✅ 文本到视频
- ✅ 相机控制
- ✅ 深度预测
- ✅ 点云估计

可能的扩展：
- 📝 图像驱动
- 📝 更多几何输出 (法线, 实例分割)
- 📝 音频同步生成

需要修改 [架构](./ARCHITECTURE.md)

---

## 🆘 如何获取帮助

如以上信息未能解决问题：

1. **检查日志**: `cat outputs/fantasy_world_stage1/training.log`
2. **查看代码**: 相关文件位置见 [架构](./ARCHITECTURE.md)
3. **运行诊断**: `python scripts/diagnose.py`
4. **搜索相似问题**: 项目文档中其他部分
5. **社区支持**: DiffSynth-Studio 官方仓库

---

**祝你使用顺利！** 🚀
