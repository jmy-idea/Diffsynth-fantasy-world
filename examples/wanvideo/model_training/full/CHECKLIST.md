# ✅ Fantasy World 两阶段训练 - 快速检查清单

## 开始训练前

### 1. 环境检查
- [ ] PyTorch >= 1.13 安装完成
- [ ] DiffSynth 已安装 (`pip install -e .`)
- [ ] GPU 可用 (`nvidia-smi`)
- [ ] 足够的磁盘空间 (建议 > 500GB)

### 2. 数据准备
- [ ] 视频文件已准备 (`dataset/videos/`)
- [ ] Depth maps 已生成 (`dataset/depth/`)
- [ ] Point clouds 已生成 (`dataset/points/`)
- [ ] Camera params 已生成 (`dataset/camera_params/`)
- [ ] `metadata.json` 已创建

### 3. 模型权重
- [ ] Wan2.1 基础模型已下载
- [ ] 模型路径已配置在训练脚本中

---

## Stage 1: Latent Bridging

### 配置检查
- [ ] 编辑 `train_fantasy_world_stage1.sh`
- [ ] 设置 `DATA_DIR` 为你的数据集路径
- [ ] 设置 `CUDA_VISIBLE_DEVICES` (根据可用 GPU)
- [ ] 调整 `BATCH_SIZE_PER_GPU` (根据显存)
- [ ] 确认 `NUM_STEPS=20000`

### 运行训练
```bash
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
bash examples/wanvideo/model_training/full/train_fantasy_world_stage1.sh
```

### 监控
- [ ] 检查 `loss/total` 是否下降
- [ ] 检查 `loss/depth_loss` < 0.1
- [ ] 检查 `loss/point_loss` < 0.5
- [ ] 检查 `loss/camera_loss` < 0.05
- [ ] 每 1000 steps 保存 checkpoint

### 完成标准
- [ ] 训练到 20,000 steps
- [ ] `outputs/fantasy_world_stage1/step-20000.safetensors` 存在
- [ ] Geometry losses 已收敛

---

## Stage 2: Unified Co-Optimization

### 配置检查
- [ ] 编辑 `train_fantasy_world_stage2.sh`
- [ ] 设置 `DATA_DIR` 为同一数据集路径
- [ ] 设置 `STAGE1_CHECKPOINT` 为 Stage 1 输出
- [ ] 确认 `NUM_STEPS=10000`
- [ ] 调整 `BATCH_SIZE_PER_GPU` (全分辨率需要更多显存)

### 运行训练
```bash
# 先验证 Stage 1 checkpoint 存在
ls outputs/fantasy_world_stage1/step-20000.safetensors

# 运行 Stage 2
bash examples/wanvideo/model_training/full/train_fantasy_world_stage2.sh
```

### 监控
- [ ] 确认从 Stage 1 checkpoint 加载成功
- [ ] 检查 `loss/total` 继续下降
- [ ] 监控 geometry losses 保持稳定
- [ ] 每 1000 steps 保存 checkpoint

### 完成标准
- [ ] 训练到 10,000 steps
- [ ] `outputs/fantasy_world_stage2/step-10000.safetensors` 存在
- [ ] 所有 losses 稳定或下降

---

## 训练完成后

### 验证
- [ ] 加载 final checkpoint 进行推理测试
- [ ] 测试不同 camera trajectories
- [ ] 可视化生成的 depth maps
- [ ] 可视化生成的 point clouds
- [ ] 对比不同训练阶段的效果

### 保存和分享
- [ ] 备份最终 checkpoint
- [ ] 保存训练日志
- [ ] 记录最佳超参数配置
- [ ] (可选) 分享到 HuggingFace Hub

---

## 故障排查

### 显存不足 (OOM)
- [ ] 减小 `BATCH_SIZE_PER_GPU`
- [ ] 增加 `GRADIENT_ACCUMULATION`
- [ ] 降低分辨率 (仅 Stage 1)
- [ ] 减少 `NUM_FRAMES`

### Loss 不收敛
- [ ] 检查数据质量 (depth, points, camera)
- [ ] 降低学习率
- [ ] 延长 Stage 1 训练 (超过 20K steps)
- [ ] 检查 geometry loss weights

### Checkpoint 加载失败
- [ ] 确认 Stage 1 checkpoint 路径正确
- [ ] 检查 checkpoint 文件完整性
- [ ] 确认使用相同的模型配置

---

## 快速验证命令

```bash
# 1. 检查数据集结构
ls -lh dataset/videos dataset/depth dataset/points dataset/camera_params

# 2. 验证两阶段配置
python examples/wanvideo/model_training/full/verify_two_stage_config.py

# 3. 测试单个样本 (dry run)
# [可选] 创建一个只有 1 个样本的小数据集快速测试

# 4. 监控 GPU 使用
watch -n 1 nvidia-smi

# 5. 监控训练日志
tail -f outputs/fantasy_world_stage1/train.log
```

---

## 重要提醒

⚠️ **始终冻结**:
- Wan2.1 原有的 30 blocks (PCB 12 + IRG 18)
- 这是两阶段训练的核心 - 保持 video backbone 稳定

✅ **Stage 1 可训练**:
- Latent Bridge, GeoDiT Blocks, DPT Heads, Pose Encoder, Tokens

✅ **Stage 2 新增可训练**:
- IRG Cross-Attention, Camera Adapters

📊 **参数量**:
- Stage 1: ~956M trainable params
- Stage 2: ~1186M trainable params (+230M interaction modules)

⏱️ **预计时间** (64-112 H20 GPUs):
- Stage 1: ~36 hours
- Stage 2: ~144 hours

---

## 资源

- 📖 详细指南: `TWO_STAGE_TRAINING_GUIDE.md`
- 🔧 技术总结: `TWO_STAGE_IMPLEMENTATION_SUMMARY.md`
- 🐛 Bug 修复文档: `docs/ROPE_FIX_EXPLANATION.md`, `docs/DTYPE_FIX_EXPLANATION.md`

---

**祝训练顺利！** 🚀

记得定期保存 checkpoint 和监控 loss curves。
