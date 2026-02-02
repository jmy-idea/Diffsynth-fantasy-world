# Fantasy World 两阶段训练实现总结

## ✅ 已完成的工作

### 1. 核心架构修改

#### `wan_video_dit.py` - `enable_fantasy_world_mode()`
添加 `training_stage` 参数支持两阶段训练：

```python
def enable_fantasy_world_mode(self, split_layer=12, training_stage="stage2"):
    """
    Args:
        training_stage: "stage1" or "stage2"
            - stage1: Only geometry branch trainable
            - stage2: Geometry branch + interaction modules trainable
    """
```

**Stage 1 可训练模块**:
- ✅ Latent Bridge Adapter
- ✅ GeoDiT Blocks (18 layers)
- ✅ DPT Heads (depth, point, camera)
- ✅ Pose Encoder
- ✅ Special Tokens (camera + register)

**Stage 2 新增可训练**:
- ✅ IRG Cross-Attention (18 modules)
- ✅ Camera Adapters (12 modules)

**始终冻结**:
- ❄️ Wan2.1 原有 30 blocks (PCB 12 + IRG 18)

---

### 2. 训练脚本

#### `train.py`
- 解析 task 字符串中的 stage 信息 (`fantasy_world:stage1`, `fantasy_world:stage2`)
- 传递 `training_stage` 给 `enable_fantasy_world_mode()`
- 添加 stage1/stage2 到 `task_to_loss` 和 `launcher_map`

#### 两个训练脚本

**Stage 1**: `train_fantasy_world_stage1.sh`
```bash
--task fantasy_world:stage1
--num_steps 20000
--batch_size 64 (8 GPUs × 8 per GPU)
--height 336 --width 592
```

**Stage 2**: `train_fantasy_world_stage2.sh`
```bash
--task fantasy_world:stage2
--stage1_checkpoint outputs/fantasy_world_stage1/step-20000.safetensors
--num_steps 10000
--batch_size 112 (8 GPUs × 14 per GPU)
--height 592 --width 336  # Full resolution
```

---

### 3. 文档和工具

#### 文档
- `TWO_STAGE_TRAINING_GUIDE.md`: 完整的训练指南
  - 论文策略详解
  - 使用方法
  - 技术细节
  - 常见问题 FAQ

#### 验证脚本
- `verify_two_stage_config.py`: 检查两阶段配置是否正确
  - 列出每个阶段的可训练模块
  - 统计参数量
  - 对比两阶段差异

---

## 📊 参数统计 (预估)

| Module | Parameters | Stage 1 | Stage 2 |
|--------|-----------|---------|---------|
| Wan2.1 Blocks | ~1616M | ❄️ | ❄️ |
| Latent Bridge | ~5M | ✅ | ✅ |
| GeoDiT Blocks | ~900M | ✅ | ✅ |
| DPT Heads | ~50M | ✅ | ✅ |
| Pose Encoder | ~1M | ✅ | ✅ |
| Tokens | ~0.01M | ✅ | ✅ |
| Camera Adapters | ~30M | ❄️ | ✅ |
| IRG Cross-Attn | ~200M | ❄️ | ✅ |
| **Total Trainable** | - | **~956M** | **~1186M** |

---

## 🎯 训练流程

```
Step 1: 准备数据
  ├── videos/ (原始视频)
  ├── depth/ (Depth Anything V2 预测)
  ├── points/ (DUSt3R 预测)
  └── camera_params/ (DUSt3R + PnP 估计)

Step 2: Stage 1 训练 (20K steps)
  ├── 修改 train_fantasy_world_stage1.sh 中的 DATA_DIR
  ├── bash train_fantasy_world_stage1.sh
  └── 输出: outputs/fantasy_world_stage1/step-20000.safetensors

Step 3: Stage 2 训练 (10K steps)
  ├── 确认 Stage 1 checkpoint 存在
  ├── bash train_fantasy_world_stage2.sh
  └── 输出: outputs/fantasy_world_stage2/step-10000.safetensors (final model)

Step 4: 推理
  └── 使用 final model 进行 camera-controlled video generation
```

---

## 🔍 关键设计决策

### 1. 为什么两阶段？

**问题**: 直接联合训练所有模块导致：
- Geometry branch 学不到有效特征 (video features 不稳定)
- Training instability (gradient conflicts)
- Poor convergence

**解决方案**:
1. **Stage 1**: 冻结 video branch，让 geometry branch 适配到稳定的 video features
2. **Stage 2**: 在已收敛的 geometry branch 基础上，微调 interaction modules

### 2. 为什么 Stage 1 不使用 interaction modules?

**理由**:
- Interaction modules (cross-attention, camera adapters) 的目的是双向交互
- 如果 geometry branch 还没学好，interaction 只会引入噪声
- Stage 1 专注让 geometry branch 学会从 video features 提取几何信息

### 3. 为什么 Stage 2 还要继续训练 geometry branch?

**理由**:
- Stage 1 训练的 geometry branch 是基于 **frozen** video features
- Stage 2 引入 interaction 后，video features 会有微小变化 (通过 camera adapters)
- 联合训练让 geometry branch 和 interaction modules 协同优化

---

## ⚙️ 技术实现细节

### Trainable Parameter 控制

在 `enable_fantasy_world_mode()` 中：

```python
# Stage 1: Only geometry branch
if training_stage == "stage1":
    for param in self.latent_bridge.parameters():
        param.requires_grad = True
    for param in self.geo_blocks.parameters():
        param.requires_grad = True
    # ... 其他 geometry modules
    
    # Freeze interaction modules
    for adapter in self.camera_adapters:
        if adapter is not None:
            for param in adapter.parameters():
                param.requires_grad = False
    # ... IRG cross-attns

# Stage 2: Geometry + interaction
elif training_stage == "stage2":
    # Keep geometry branch trainable
    for param in self.latent_bridge.parameters():
        param.requires_grad = True
    # ...
    
    # Unfreeze interaction modules
    for adapter in self.camera_adapters:
        if adapter is not None:
            for param in adapter.parameters():
                param.requires_grad = True
    # ... IRG cross-attns
```

### Task String Parsing

在 `train.py` 中：

```python
training_stage = "stage2"  # Default
if self.task.startswith("fantasy_world"):
    if ":" in self.task:
        stage_str = self.task.split(":")[-1]
        if stage_str in ["stage1", "stage2"]:
            training_stage = stage_str
    
    self.pipe.dit.enable_fantasy_world_mode(training_stage=training_stage)
```

支持的 task 字符串:
- `fantasy_world` → Stage 2 (默认)
- `fantasy_world:stage1` → Stage 1
- `fantasy_world:stage2` → Stage 2
- `fantasy_world:train` → Stage 2 (向后兼容)

---

## ✅ 验证清单

在开始训练前，运行验证脚本：

```bash
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
python examples/wanvideo/model_training/full/verify_two_stage_config.py
```

**期望输出**:
- Stage 1 可训练参数: ~956M
- Stage 2 可训练参数: ~1186M
- 差值 (interaction modules): ~230M

---

## 📚 文件清单

```
examples/wanvideo/model_training/full/
├── train_fantasy_world_stage1.sh          # Stage 1 训练脚本
├── train_fantasy_world_stage2.sh          # Stage 2 训练脚本
├── TWO_STAGE_TRAINING_GUIDE.md           # 用户指南 (详细)
├── TWO_STAGE_IMPLEMENTATION_SUMMARY.md   # 本文档 (技术总结)
└── verify_two_stage_config.py            # 配置验证脚本

diffsynth/models/
└── wan_video_dit.py                      # 修改: enable_fantasy_world_mode()

examples/wanvideo/model_training/
└── train.py                              # 修改: 支持 stage1/stage2
```

---

## 🚀 下一步

1. **准备数据集**
   - 使用 Depth Anything V2 生成 depth
   - 使用 DUSt3R 生成 points 和 camera params
   - 创建 metadata.json

2. **运行 Stage 1**
   ```bash
   bash examples/wanvideo/model_training/full/train_fantasy_world_stage1.sh
   ```

3. **监控训练**
   - 检查 loss curves (depth_loss, point_loss, camera_loss)
   - 建议在验证集上可视化预测结果

4. **运行 Stage 2**
   ```bash
   bash examples/wanvideo/model_training/full/train_fantasy_world_stage2.sh
   ```

5. **推理测试**
   - 使用 final checkpoint 生成视频
   - 测试不同 camera trajectories

---

## 📝 论文对应关系

| 论文章节 | 实现位置 |
|---------|---------|
| Section 3.3 (Architecture) | `wan_video_dit.py::enable_fantasy_world_mode()` |
| Section 4.3 (Training Strategy) | `train_fantasy_world_stage1.sh`, `train_fantasy_world_stage2.sh` |
| Table 2 (Hyperparameters) | Bash scripts 中的配置 |
| Figure 3 (Two-stage training) | `TWO_STAGE_TRAINING_GUIDE.md` 图解 |

---

## 🎉 完成状态

- ✅ 两阶段架构实现
- ✅ 可训练参数控制
- ✅ 训练脚本 (Stage 1 + Stage 2)
- ✅ 完整文档和指南
- ✅ 配置验证工具
- ✅ Checkpoint 加载逻辑
- ✅ 论文策略对齐

**准备开始训练！** 🚀
