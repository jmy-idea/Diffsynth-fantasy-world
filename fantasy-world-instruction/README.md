# Fantasy World复现

基于 Wan2.1 的 Fantasy World 模型实现，集成了深度预测、点云估计和相机控制能力。本文档提供从环境搭建、数据准备到训练推理的完整工作流程。

---

## 📋 文档导航

### 🚀 快速开始 (选择你的起点)

| 情景 | 推荐文档 |
|------|---------|
| **首次使用，不确定从何开始** | [快速入门指南](./QUICKSTART.md) |
| **需要搭建开发环境** | [环境与前期准备](./ENVIRONMENT_SETUP.md) |
| **需要准备训练数据** | [数据准备与处理](./DATA_PREPARATION.md) |
| **准备开始训练模型** | [详细训练指南](./TRAINING_GUIDE.md) |
| **要进行推理测试** | [推理与应用](./INFERENCE_GUIDE.md) |
| **遇到问题需要帮助** | [故障排查与 FAQ](./TROUBLESHOOTING.md) |
| **想深入理解架构** | [架构与模型修改详解](./ARCHITECTURE.md) |

---

## 📚 完整文档目录

### 一、基础入门
1. **[快速入门指南](./QUICKSTART.md)** ⭐
   - 30 秒了解项目
   - 5 分钟完成环境搭建
   - 10 分钟运行第一个示例

2. **[环境与前期准备](./ENVIRONMENT_SETUP.md)**
   - 系统要求与依赖
   - Python 环境配置
   - CUDA/GPU 设置
   - 项目安装步骤
   - 验证安装清单

### 二、数据与模型

3. **[架构与模型修改详解](./ARCHITECTURE.md)**
   - Wan2.1 原始架构概览
   - Fantasy World 的结构修改
   - 与论文 Section 3.3 的对应
   - 所有新增/修改模块详细说明
   - 完整的文件位置映射
   - 代码流程演示

4. **[数据准备与处理](./DATA_PREPARATION.md)**
   - 数据格式规范
   - 获取原始视频
   - 深度图生成 (Depth Anything V2)
   - 点云估计 (DUSt3R)
   - 相机参数估计与处理
   - 数据集组织结构
   - 元数据配置

### 三、训练流程

5. **[详细训练指南](./TRAINING_GUIDE.md)**
   - 两阶段训练策略详解 (论文 Section 4.3)
   - **Stage 1: Latent Bridging (20K steps, batch 64)**
     - 目的与设计原理
     - 可训练模块列表
     - 配置参数详解
     - 训练脚本使用
     - 性能监控指标
   - **Stage 2: Unified Co-Optimization (10K steps, batch 112)**
     - 目的与设计原理
     - 新增可训练模块
     - 检查点加载
     - 完整分辨率训练
     - 收敛判断方法
   - 训练流程图与时间线
   - 性能优化建议

### 四、推理与应用

6. **[推理与应用](./INFERENCE_GUIDE.md)**
   - 推理管道概述
   - 模型加载步骤
   - 相机轨迹控制
   - 视频生成过程
   - 批处理与优化
   - 输出处理与保存
   - 推理示例代码

### 五、问题解决

7. **[故障排查与 FAQ](./TROUBLESHOOTING.md)**
   - 常见错误与解决方案
   - 性能问题诊断
   - 内存优化策略
   - 训练不稳定的原因
   - 推理失败排查
   - 常见问题答疑

### 六、技术深入阅读 (可选)

8. **[技术细节深入阅读](./TECHNICAL_DEEP_DIVE.md)**
   - 两阶段训练的理论基础
   - 损失函数详解
   - 交叉注意力机制
   - 相机适配器设计
   - Latent Bridge 工作原理
   - RoPE 长度扩展
   - DType 处理问题

### 七、支持文档

9. **[快速参考卡](./QUICK_REFERENCE.md)**
   - 常用命令一览
   - 参数配置速查表
   - 文件位置快速定位
   - 关键代码片段

10. **[论文对应检查表](./PAPER_COMPLIANCE.md)**
    - 论文设计 vs 实现对照
    - 每个论文章节的代码位置
    - 超参数对应关系
    - 架构完成度统计

---

## 🎯 工作流程图

```
准备阶段
├─ 环境设置 (ENVIRONMENT_SETUP.md)
│  └─ Python 环境 + 项目安装
│
├─ 数据准备 (DATA_PREPARATION.md)
│  ├─ 收集视频素材
│  ├─ 深度图生成
│  ├─ 点云估计
│  ├─ 相机参数处理
│  └─ 数据验证
│
└─ 前期验证
   └─ 环境检查清单

训练阶段
├─ Stage 1: Latent Bridging (TRAINING_GUIDE.md)
│  ├─ 配置 train_fantasy_world_stage1.sh
│  ├─ 运行 20K steps 训练
│  ├─ 监控 geometry losses
│  └─ 验证 checkpoint
│
└─ Stage 2: Co-Optimization (TRAINING_GUIDE.md)
   ├─ 加载 Stage 1 checkpoint
   ├─ 配置 train_fantasy_world_stage2.sh
   ├─ 运行 10K steps 训练
   └─ 生成最终模型

推理阶段
├─ 加载最终模型 (INFERENCE_GUIDE.md)
├─ 生成相机轨迹
├─ 运行推理
└─ 保存输出视频

应用/部署
└─ 根据需求自定义
```

---

## ⏱️ 预计时间投入

| 阶段 | 时间 | 说明 |
|------|------|------|
| 环境搭建 | 15-30 分钟 | 一次性工作 |
| 数据准备 | 2-8 小时 | 取决于数据量 |
| Stage 1 训练 | 36 小时 | 8 × H20 GPUs, 20K steps |
| Stage 2 训练 | 144 小时 | 8 × H20 GPUs, 10K steps |
| 推理单视频 | 1-2 分钟 | 81 frames, 592×336 |
| **总计 (包括数据)** | **约 200 小时** | 仅 GPU 时间 |

---

## 📊 项目概览

### 核心改动

**Fantasy World** 通过以下方式扩展 Wan2.1 视频扩散模型：

| 组件 | 作用 | 文件位置 |
|------|------|---------|
| **Latent Bridge Adapter** | 将视频特征映射到几何空间 | `wan_video_dit.py` L93-100 |
| **GeoDiT Blocks (18层)** | 可训练的几何感知 transformer | `wan_video_dit.py` L101-115 |
| **DPT Heads (3个)** | 深度、点云、相机预测头 | `wan_video_dit.py` L116-135 |
| **相机适配器 (12个)** | 视频分支的相机控制注入 | `wan_video_dit.py` L136-145 |
| **IRG 交叉注意力 (18个)** | 双向视频-几何交互 | `wan_video_dit.py` L146-155 |
| **Pose 编码器** | 相机参数到嵌入映射 | `wan_video_dit.py` L156-162 |

### 架构对比

```
Wan2.1 (原始)                Fantasy World (扩展)
─────────────                ─────────────────
30 DiT blocks          →     30 DiT blocks (冻结)
     ↓                       +  Latent Bridge (可训)
最终输出 (视频)               +  18 GeoDiT blocks (可训)
                            +  3 DPT heads (可训)
                            +  相机适配器 (Stage 2)
                            +  交叉注意力 (Stage 2)
                                 ↓
                            视频 + 几何预测
```

### 训练资源需求

| 指标 | Stage 1 | Stage 2 |
|------|---------|---------|
| GPU 数量 | 8 × H20 | 8 × H20 |
| 显存 (每 GPU) | 40GB+ | 40GB+ |
| Batch Size | 64 | 112 |
| 训练步数 | 20,000 | 10,000 |
| 预计时间 | 36 小时 | 144 小时 |

---

## 🛠️ 快速命令参考

### 环境设置
```bash
# 进入项目目录
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world

# 以开发模式安装
pip install -e .

# 验证安装
python -c "import diffsynth; print('OK')"
```

### 数据准备 (概览)
```bash
# 详细见 DATA_PREPARATION.md
# 生成深度图、点云、相机参数（待实现）
python scripts/prepare_dataset.py --input_dir data/raw --output_dir data/processed
```

### 训练
```bash
# Stage 1
bash examples/wanvideo/model_training/full/train_fantasy_world_stage1.sh

# Stage 2
bash examples/wanvideo/model_training/full/train_fantasy_world_stage2.sh
```

### 推理
```bash
# 详细见 INFERENCE_GUIDE.md
python examples/wanvideo/model_inference/fantasy_world_inference.py \
    --checkpoint outputs/fantasy_world_stage2/step-10000.safetensors \
    --prompt "a camera moving through a room" \
    --output_path results/
```

---

## 📁 项目结构

```
Diffsynth-fantasy-world/
├── fantasy-world-instrucion/                                    # 📚 完整文档 (你在这里)
│   ├── README.md                           # 本文件
│   ├── QUICKSTART.md                       # 快速入门 ⭐
│   ├── ENVIRONMENT_SETUP.md                # 环境设置
│   ├── ARCHITECTURE.md                     # 架构详解
│   ├── DATA_PREPARATION.md                 # 数据准备
│   ├── TRAINING_GUIDE.md                   # 训练指南
│   ├── INFERENCE_GUIDE.md                  # 推理指南
│   ├── TROUBLESHOOTING.md                  # 故障排查
│   ├── TECHNICAL_DEEP_DIVE.md             # 技术深入
│   └── QUICK_REFERENCE.md                  # 快速参考
│
├── examples/
│   └── wanvideo/
│       ├── model_training/
│       │   ├── train.py                    # 主训练脚本
│       │   └── full/
│       │       ├── train_fantasy_world_stage1.sh
│       │       └── train_fantasy_world_stage2.sh
│       └── model_inference/
│           └── fantasy_world_inference.py
│
├── diffsynth/
│   ├── models/
│   │   ├── wan_video_dit.py               # 核心架构修改
│   │   └── wan_video_camera_controller.py # 相机控制
│   ├── core/data/
│   │   ├── fantasy_world_dataset.py       # 数据加载
│   │   └── fantasy_world_operators.py     # 数据操作
│   ├── diffusion/
│   │   └── loss.py                        # FantasyWorld损失函数
│   └── pipelines/
│       └── wan_video.py                   # 核心管线脚本
│
│
└── scripts/
    ├── prepare_dataset.py                 # 数据准备（待实现）
    └── setup_and_diagnose.sh              # 环境诊断
    
```

---

## 🚀 开始使用

### 第一步：选择你的路径

👉 **完全不熟悉** → 从 [快速入门](./QUICKSTART.md) 开始 (5 分钟)

👉 **已有环境** → 直接去 [数据准备](./DATA_PREPARATION.md)

👉 **深入了解架构** → 先读 [架构详解](./ARCHITECTURE.md)

👉 **遇到问题** → 查阅 [故障排查](./TROUBLESHOOTING.md)

### 第二步：按顺序完成

推荐流程：
1. ✅ [环境与前期准备](./ENVIRONMENT_SETUP.md)
2. ✅ [数据准备与处理](./DATA_PREPARATION.md)
3. ✅ [详细训练指南](./TRAINING_GUIDE.md)
4. ✅ [推理与应用](./INFERENCE_GUIDE.md)

---

## ❓ 需要帮助？

| 问题类型 | 查看 |
|---------|------|
| "我该从哪开始？" | [快速入门](./QUICKSTART.md) |
| "环境配置失败" | [环境设置](./ENVIRONMENT_SETUP.md) \| [故障排查](./TROUBLESHOOTING.md) |
| "不知道数据怎么准备" | [数据准备](./DATA_PREPARATION.md) |
| "训练不理想/中断" | [详细训练指南](./TRAINING_GUIDE.md) \| [故障排查](./TROUBLESHOOTING.md) |
| "推理出错" | [推理指南](./INFERENCE_GUIDE.md) \| [故障排查](./TROUBLESHOOTING.md) |
| "想了解架构原理" | [架构详解](./ARCHITECTURE.md) \| [技术深入](./TECHNICAL_DEEP_DIVE.md) |
| "对应论文哪部分？" | [论文对应检查](./PAPER_COMPLIANCE.md) |

---

## 📖 关键术语

| 术语 | 定义 | 相关文档 |
|------|------|---------|
| **PCB** | Preconditioning Blocks - Wan2.1 的前 12 层 | ARCHITECTURE.md |
| **IRG** | Integrated Reconstruction & Generation - Wan2.1 的后 18 层 | ARCHITECTURE.md |
| **GeoDiT** | 几何感知 DiT 块，处理深度/点云预测 | ARCHITECTURE.md |
| **Latent Bridge** | 轻量级适配器，连接视频和几何特征空间 | ARCHITECTURE.md |
| **PoseEncoder** | 将相机位姿转换为嵌入向量 | ARCHITECTURE.md |
| **Stage 1** | 第一阶段训练，仅训练几何分支 | TRAINING_GUIDE.md |
| **Stage 2** | 第二阶段训练，添加交互模块 | TRAINING_GUIDE.md |
| **DPT Head** | 深度预测 Transformer 头 | ARCHITECTURE.md |

---

## 📝 文档维护

- 最后更新: 2026-02-03
- 文档版本: v1.0
- 对应代码: Fantasy World 完整实现

如发现文档有误或不明之处，请参考代码实现 (`diffsynth/models/wan_video_dit.py`) 或 [故障排查](./TROUBLESHOOTING.md)。

---

## 📚 相关资源

- **论文**: Fantasy World: A Geometry-Aware Video Generation with Multi-Modal Cues
- **基础模型**: [Wan2.1-Fun-Control-Camera](https://huggingface.co/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera)
- **框架**: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)

---

**祝你使用愉快！** 🎉

如有问题，始终可以回到本文档的导航部分找到答案。
