# Fantasy World Two-Stage Training Guide

本指南详细说明 Fantasy World 的两阶段训练策略。

## 📖 论文中的训练策略 (Section 4.3)

Fantasy World 采用两阶段训练策略来稳定 geometry-aware video generation 的训练过程：

### Stage 1: Latent Bridging (潜在空间桥接)

**目的**: 将 geometry branch 适配到已冻结的 video backbone 特征空间

**训练配置**:
- **训练步数**: 20,000 steps
- **Batch Size**: 64 (global)
- **Resolution**: 可以使用较低分辨率以提高稳定性
- **学习率**: 1e-5 (AdamW)
- **硬件**: 64 × H20 GPUs, ~36 hours

**可训练模块**:
- ✅ Latent Bridge Adapter (映射 video features 到 geometry space)
- ✅ GeoDiT Blocks (18 blocks with VGGT-style attention)
- ✅ DPT Heads (depth, point, camera)
- ✅ Pose Encoder (Plucker embedding)
- ✅ Special Tokens (camera token + register tokens)

**冻结模块**:
- ❄️ Wan2.1 原有的 30 blocks (PCB 12 + IRG 18)
- ❄️ Camera Adapters (Stage 2 才会使用)
- ❄️ IRG Cross-Attention (Stage 2 才会使用)

**关键点**:
- 这一阶段只训练 geometry branch 核心
- 让 geometry branch 学会从 frozen video features 中提取几何信息
- **不使用** video-geometry interaction (cross-attention, camera injection)

---

### Stage 2: Unified Co-Optimization (联合协同优化)

**目的**: 微调 interaction modules，实现 video 和 geometry 特征的双向交互

**训练配置**:
- **训练步数**: 10,000 steps
- **Batch Size**: 112 (global)
- **Resolution**: 592×336 或 336×592 (full resolution)
- **学习率**: 1e-5 (AdamW)
- **硬件**: 112 × H20 GPUs, ~144 hours
- **初始化**: 从 Stage 1 checkpoint 加载

**可训练模块**:
- ✅ **继续训练** Stage 1 的所有模块 (latent_bridge, geo_blocks, heads, pose_enc, tokens)
- ✅ **新增训练** IRG Bidirectional Cross-Attention (18 modules)
- ✅ **新增训练** Camera Control Adapters (first 12 blocks)

**仍然冻结**:
- ❄️ Wan2.1 原有的 30 blocks (始终冻结)

**关键点**:
- 在 Stage 1 基础上添加 interaction modules
- Geometry branch 已经 well-adapted，现在学习双向交互
- 使用完整分辨率的 81-frame clips

---

## 🚀 使用方法

### 准备数据

确保你的数据集包含：
```
dataset/
├── metadata.json         # 数据集元信息
├── videos/              # 视频文件
│   ├── sample1.mp4
│   └── sample2.mp4
├── depth/               # 深度图 (Depth Anything V2)
│   ├── sample1/
│   │   ├── frame_0000.npy  # [H, W]
│   │   └── ...
│   └── sample2/
├── points/              # 点云 (DUSt3R)
│   ├── sample1/
│   │   ├── frame_0000.npy  # [H, W, 3]
│   │   └── ...
│   └── sample2/
└── camera_params/       # 相机参数 (DUSt3R + PnP)
    ├── sample1.txt      # [T, 12] world-to-camera matrices
    └── sample2.txt
```

### Stage 1: Latent Bridging

```bash
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world

# 1. 配置训练脚本
vim examples/wanvideo/model_training/full/train_fantasy_world_stage1.sh
# 修改:
#   DATA_DIR="/path/to/your/fantasy_world_data"
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 根据你的GPU数量调整
#   BATCH_SIZE_PER_GPU=8  # 根据显存调整

# 2. 运行训练
bash examples/wanvideo/model_training/full/train_fantasy_world_stage1.sh
```

**输出**:
- Checkpoints 保存在 `outputs/fantasy_world_stage1/`
- 每 1000 steps 保存一次
- 最终使用 `step-20000.safetensors` 进入 Stage 2

---

### Stage 2: Unified Co-Optimization

```bash
# 1. 确认 Stage 1 checkpoint 存在
ls outputs/fantasy_world_stage1/step-20000.safetensors

# 2. 配置训练脚本
vim examples/wanvideo/model_training/full/train_fantasy_world_stage2.sh
# 修改:
#   DATA_DIR="/path/to/your/fantasy_world_data"
#   STAGE1_CHECKPOINT="outputs/fantasy_world_stage1/step-20000.safetensors"
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   BATCH_SIZE_PER_GPU=14  # 全分辨率需要更多显存

# 3. 运行训练
bash examples/wanvideo/model_training/full/train_fantasy_world_stage2.sh
```

**输出**:
- Checkpoints 保存在 `outputs/fantasy_world_stage2/`
- 最终模型: `step-10000.safetensors`

---

## 📊 可训练参数对比

| Module | Parameters | Stage 1 | Stage 2 |
|--------|-----------|---------|---------|
| **Wan2.1 Blocks** (PCB 12 + IRG 18) | ~1616M | ❄️ Frozen | ❄️ Frozen |
| **Latent Bridge Adapter** | ~5M | ✅ Train | ✅ Train |
| **GeoDiT Blocks** (18 blocks) | ~900M | ✅ Train | ✅ Train |
| **DPT Heads** (depth, point, camera) | ~50M | ✅ Train | ✅ Train |
| **Pose Encoder** | ~1M | ✅ Train | ✅ Train |
| **Special Tokens** (camera + register) | ~0.01M | ✅ Train | ✅ Train |
| **Camera Adapters** (12 modules) | ~30M | ❄️ Frozen | ✅ Train |
| **IRG Cross-Attention** (18 modules) | ~200M | ❄️ Frozen | ✅ Train |
| **Total Trainable** | - | ~956M | ~1186M |

---

## 🔍 技术细节

### Latent Bridge Adapter

```python
self.latent_bridge = LatentBridgeAdapter(
    dim=model_dim,           # 1536 for Wan2.1-1.3B
    num_heads=8,
    ffn_dim=model_dim * 4,
    num_layers=2             # Lightweight: only 2 layers
)
```

- 接收 split_layer (block 12) 的输出
- 映射到 geometry-aligned latent space
- 输入给 GeoDiT blocks

### GeoDiT Blocks

- 基于 VGGT 架构 (Global + Frame attention)
- 18 blocks (对应 IRG 的 18 layers)
- 从 4 个中间层提取特征给 DPT heads

### Camera Adapters

```python
camera_adapters[i] = Sequential(
    SiLU(),
    Linear(dim, dim)
)
```

- 预测 shift βᵢ (not full AdaLN)
- 注入到 video branch: fᵢ = fᵢ₋₁ + βᵢ
- 只应用于前 12 blocks (PCB)

### IRG Cross-Attention

```python
self.irg_cross_attns = ModuleList([
    MMBiCrossAttention(dim, num_heads) 
    for _ in range(18)  # One per IRG block
])
```

- Bidirectional cross-attention
- Video features ↔ Geometry features
- 在每个 IRG block 后应用

---

## ⚠️ 常见问题

### Q1: 为什么需要两阶段训练？

**A**: 直接联合训练所有模块会导致：
- Geometry branch 学不到有效特征 (video features 一直在变)
- Training instability (gradient conflicts)
- Poor convergence

两阶段策略：
1. 先让 geometry branch 适配到 frozen video features
2. 再引入 interaction，微调双向交互

### Q2: Stage 1 可以用更少的 steps 吗？

**A**: 可以，但建议至少 10K steps。论文用 20K steps 是为了充分收敛。
可以监控 geometry loss (depth, point, camera) 来判断是否收敛。

### Q3: 显存不足怎么办？

**调整 batch size**:
```bash
# Stage 1: 64 global batch size
BATCH_SIZE_PER_GPU=8  # 8 GPUs → 64
# 如果显存不足:
BATCH_SIZE_PER_GPU=4  # 8 GPUs → 32
# 增加 gradient accumulation 补偿:
GRADIENT_ACCUMULATION=2  # Effective batch size = 32 * 2 = 64
```

**降低分辨率** (Stage 1 only):
```bash
HEIGHT=288  # 从 336 降低
WIDTH=512   # 从 592 降低
```

**减少 frames**:
```bash
NUM_FRAMES=41  # 从 81 降低到 41
```

### Q4: 如何验证 Stage 1 训练效果？

**监控指标**:
- `loss/depth_loss`: 应该降到 < 0.1
- `loss/point_loss`: 应该降到 < 0.5
- `loss/camera_loss`: 应该降到 < 0.05

**可视化** (推荐):
加载 Stage 1 checkpoint，在验证集上：
1. 可视化预测的 depth maps
2. 可视化预测的 point clouds
3. 对比 GT 和预测的 camera trajectories

### Q5: Stage 2 必须从 Stage 1 checkpoint 加载吗？

**A**: 是的，必须。Stage 2 依赖 Stage 1 训练好的 geometry branch。
脚本会自动检查 `STAGE1_CHECKPOINT` 是否存在。

---

## 📁 文件结构

```
examples/wanvideo/model_training/full/
├── train_fantasy_world_stage1.sh      # Stage 1 训练脚本
├── train_fantasy_world_stage2.sh      # Stage 2 训练脚本
└── TWO_STAGE_TRAINING_GUIDE.md        # 本文档

outputs/
├── fantasy_world_stage1/              # Stage 1 输出
│   ├── step-1000.safetensors
│   ├── step-2000.safetensors
│   ├── ...
│   └── step-20000.safetensors         # → Stage 2 input
└── fantasy_world_stage2/              # Stage 2 输出
    ├── step-1000.safetensors
    ├── ...
    └── step-10000.safetensors         # Final model
```

---

## 🎯 下一步

训练完成后，使用最终模型进行推理：

```python
from diffsynth import WanVideoPipeline

# 加载 Stage 2 final checkpoint
pipe = WanVideoPipeline.from_pretrained(
    model_configs=[
        {"model_path": "outputs/fantasy_world_stage2/step-10000.safetensors"}
    ]
)

# Enable Fantasy World mode
pipe.dit.enable_fantasy_world_mode(training_stage="stage2")

# 生成视频 with camera control
video = pipe(
    prompt="A serene underwater scene with swimming fish",
    pose_file_path="camera_trajectories/orbit_360deg.txt",
    num_frames=81,
    height=592,
    width=336,
)
```

---

## 📚 参考文献

- Fantasy World Paper: [arXiv:2501.XXXXX]
- Wan2.1 Model: [HuggingFace/PAI/Wan2.1-Fun-V1.1-1.3B]
- VGGT Architecture: [arXiv:2407.XXXXX]
- DUSt3R: [arXiv:2312.14132]
- Depth Anything V2: [arXiv:2406.09414]

---

**祝训练顺利！** 🚀

如有问题，请查看:
- [SETUP_GUIDE.md](../../docs/SETUP_GUIDE.md)
- [ROPE_FIX_EXPLANATION.md](../../docs/ROPE_FIX_EXPLANATION.md)
- [DTYPE_FIX_EXPLANATION.md](../../docs/DTYPE_FIX_EXPLANATION.md)
