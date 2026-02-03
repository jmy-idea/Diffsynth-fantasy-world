# 🔧 环境与前期准备完整指南

详细的环境配置和项目安装步骤。

---

## 📋 目录

1. [系统需求](#系统需求)
2. [Python 环境配置](#python-环境配置)
3. [CUDA 与 GPU 设置](#cuda-与-gpu-设置)
4. [项目安装](#项目安装)
5. [环境验证](#环境验证)
6. [故障排查](#故障排查)

---

## 💻 系统需求

### 操作系统

- **推荐**: Ubuntu 20.04 LTS 或更高版本
- **支持**: CentOS 8+, Debian 11+
- **也可**: Windows (WSL2) 或 macOS (限推理)

### 硬件

#### 最低要求 (仅推理)

- CPU: Intel Xeon 或同等级 (8+ 核)
- GPU: 任何支持 CUDA 的 NVIDIA GPU (12GB+ 显存)
  - RTX 3080/3090
  - RTX 4090
  - A6000 (48GB)
  - H100 (80GB)
- RAM: 32GB
- 存储: 50GB (模型 + 临时文件)

#### 推荐配置 (训练)

- CPU: 128 核或更多
- GPU: 8 × H20 (40GB) 或 A100 (80GB)
- RAM: 512GB
- 存储: 1TB+ SSD (快速存储)
- 网络: 100Mbps+ (用于下载权重)

### 软件栈

```
操作系统 (Linux)
    ↓
Python 3.9+ ⭐
    ↓
CUDA 11.8+ (仅 NVIDIA GPU)
    ↓
PyTorch 2.0+
    ↓
DiffSynth-Studio + Fantasy World ⭐
```

---

## 🐍 Python 环境配置

### 步骤 1: 检查 Python 版本

```bash
python3 --version

# 期望输出: Python 3.9.x 或更高
```

如果没有 Python 3.9+:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.9 python3.9-venv python3.9-dev

# 设置为默认
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
```

### 步骤 2: 创建虚拟环境 (推荐)

```bash
# 创建虚拟环境
python3.9 -m venv ~/envs/fantasy_world

# 激活虚拟环境
source ~/envs/fantasy_world/bin/activate

# 验证
python --version  # 应该显示 Python 3.9.x
which python      # 应该在 ~/envs/fantasy_world/bin/python
```

**添加到 shell profile** (可选，方便后续使用):

```bash
echo 'alias fw_env="source ~/envs/fantasy_world/bin/activate"' >> ~/.bashrc
source ~/.bashrc

# 现在可以直接使用
fw_env
```

### 步骤 3: 升级 pip

```bash
pip install --upgrade pip setuptools wheel

# 验证
pip --version
```

### 步骤 4: 配置 pip 源 (可选，加速下载)

创建 `~/.pip/pip.conf`:

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
# 或其他镜像:
# https://mirrors.aliyun.com/pypi/simple/
# https://pypi.org/simple/
```

---

## 🎮 CUDA 与 GPU 设置

### 检查 NVIDIA 驱动

```bash
nvidia-smi

# 期望输出:
# +-----------------------------------+
# | NVIDIA-SMI 535.00    Driver Version: 535.00 |
# | GPU  Name         Persistence-M| Bus-Id |
# |   0  NVIDIA H20 ... |  GPU-UUID  |
# +-----------------------------------+
```

如果没有驱动:

```bash
# Ubuntu 22.04
sudo apt-get update
sudo apt-get install nvidia-driver-535

# 重启
sudo reboot
```

### 安装 CUDA Toolkit

```bash
# 检查当前 CUDA 版本
nvcc --version

# 如果没有或版本过低，下载安装
# https://developer.nvidia.com/cuda-11-8-0-download-archive

# Ubuntu 22.04 示例:
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --driver --toolkit

# 添加到 PATH
echo 'export PATH=$PATH:/usr/local/cuda-11.8/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64' >> ~/.bashrc
source ~/.bashrc
```

### 验证 CUDA 设置

```bash
# 检查 nvcc 编译器
nvcc --version

# 检查是否可用
python << 'EOF'
import torch
print("CUDA 可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("GPU 数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))
    print("显存大小:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
EOF
```

**期望输出**:
```
CUDA 可用: True
CUDA 版本: 11.8
GPU 数量: 8
GPU 名称: NVIDIA H20 Tensor Core GPU
显存大小: 40.0 GB
```

---

## 📦 项目安装

### 步骤 1: 进入项目目录

```bash
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
pwd  # 验证位置
```

### 步骤 2: 安装依赖

```bash
# 方法 A: 开发模式 (推荐，支持代码修改)
pip install -e .

# 方法 B: 生产模式 (如果仅使用，不修改代码)
pip install .
```

**安装会自动下载并安装**:
- torch
- torchvision
- diffusers
- transformers
- safetensors
- 等其他依赖

这通常需要 5-10 分钟。

### 步骤 3: 下载模型权重 (首次)

```bash
# 自动在第一次推理时下载，或手动下载
python << 'EOF'
from diffsynth import WanVideoPipeline

# 这会下载 ~5GB 的模型权重
pipe = WanVideoPipeline.from_pretrained(
    "PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera"
)
print("✅ 模型权重已下载")
EOF
```

权重保存位置: `~/.cache/huggingface/hub/`

### 步骤 4: (可选) 安装 cuDNN

```bash
# CUDA 11.8 对应 cuDNN 8.x
# 从 NVIDIA 官网下载 (需要账户): https://developer.nvidia.com/cudnn

# 解压并复制到 CUDA 目录
tar -xzvf cudnn-linux-x86_64-8.*.tar.xz
sudo cp cudnn-linux-x86_64-8.*/include/cudnn*.h /usr/local/cuda/include/
sudo cp cudnn-linux-x86_64-8.*/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

---

## ✅ 环境验证

### 完整验证脚本

```bash
cat > verify_environment.py << 'EOF'
#!/usr/bin/env python
import sys

print("=" * 60)
print("Fantasy World 环境验证")
print("=" * 60)

# 1. Python 版本
import sys
print(f"✓ Python 版本: {sys.version}")
assert sys.version_info >= (3, 8), "Python 版本太低，需要 3.8+"

# 2. PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA 版本: {torch.version.cuda}")
        print(f"✓ GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  - GPU {i}: {props.name} ({props.total_memory / 1e9:.0f}GB)")
except ImportError:
    print("✗ PyTorch 未安装")
    sys.exit(1)

# 3. diffsynth
try:
    import diffsynth
    print(f"✓ DiffSynth: 已安装")
    from diffsynth import WanVideoPipeline
    print(f"✓ WanVideoPipeline: 可用")
except ImportError as e:
    print(f"✗ DiffSynth: {e}")
    sys.exit(1)

# 4. Fantasy World 模块
try:
    from diffsynth.core.data.fantasy_world_dataset import FantasyWorldDataset
    print(f"✓ FantasyWorldDataset: 可用")
except ImportError as e:
    print(f"⚠ FantasyWorldDataset: {e}")

# 5. 其他关键依赖
critical_packages = [
    'numpy', 'PIL', 'cv2', 'safetensors', 'transformers'
]

for pkg_name in critical_packages:
    try:
        __import__(pkg_name)
        print(f"✓ {pkg_name}: 已安装")
    except ImportError:
        print(f"✗ {pkg_name}: 未安装")

print("=" * 60)
print("✅ 环境验证完成！")
print("=" * 60)
EOF

python verify_environment.py
```

### 快速验证

```bash
# 最快的验证方法
python -c "from diffsynth import WanVideoPipeline; print('✅ 环境正常')"
```

### 验证 GPU 可用性

```bash
python << 'EOF'
import torch
from torch.utils.data import DataLoader

# 创建测试张量
x = torch.randn(2, 3, 224, 224).cuda()
print(f"✓ 张量位置: {x.device}")

# 测试基本操作
y = torch.nn.functional.relu(x)
z = y.mean()
print(f"✓ 张量操作成功")

print("✅ GPU 可用且正常工作")
EOF
```

---

## 🐛 故障排查

### 问题 1: "ModuleNotFoundError: No module named 'diffsynth'"

**症状**:
```
Traceback (most recent call last):
  File "train.py", line 1, in <module>
    from diffsynth import ...
ModuleNotFoundError: No module named 'diffsynth'
```

**解决方案**:

```bash
# 1. 检查是否在虚拟环境中
which python

# 2. 重新安装
pip install -e /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world

# 3. 清除缓存后重试
pip cache purge
pip install -e .

# 4. 检查 PYTHONPATH
export PYTHONPATH="/ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world:$PYTHONPATH"
python -c "import diffsynth"
```

### 问题 2: "CUDA out of memory"

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate ...
```

**解决方案**:

```bash
# 1. 清除 GPU 缓存
python << 'EOF'
import torch
torch.cuda.empty_cache()
EOF

# 2. 检查哪些进程占用 GPU
nvidia-smi

# 3. 杀死占用进程
kill <PID>

# 4. 调整训练参数 (见 TRAINING_GUIDE.md)
```

### 问题 3: "CUDA not available"

**症状**:
```
torch.cuda.is_available() returns False
```

**解决方案**:

```bash
# 1. 检查驱动
nvidia-smi

# 2. 检查 CUDA Toolkit
nvcc --version

# 3. 重新安装 PyTorch（针对你的 CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 验证
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题 4: 模型权重下载失败

**症状**:
```
huggingface_hub.utils._errors.RepositoryNotFoundError
```

**解决方案**:

```bash
# 1. 检查网络连接
ping huggingface.co

# 2. 设置 HF 镜像 (中国用户)
export HF_ENDPOINT=https://huggingface.co
export HF_HOME=~/.cache/huggingface

# 3. 手动下载权重 (如果自动下载失败)
# 从 https://huggingface.co/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera 下载

# 4. 指定本地路径
pipe = WanVideoPipeline.from_pretrained(
    "/path/to/local/model"
)
```

### 问题 5: pip 下载很慢

**症状**:
```
Collecting torch...
  Downloading torch-2.0.1+cu118-cp39-cp39-linux_x86_64.whl (2.0GB)
  0% |                           | 50.0MB / 2.0GB [00:00<...]
```

**解决方案**:

```bash
# 1. 使用国内镜像 (中国用户)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 或单次使用
pip install torch -i https://pypi.aliyun.com/simple

# 3. 使用清华源 (一般最快)
pip install torch -i https://mirrors.tsinghua.edu.cn/pypi/web/simple

# 4. 检查镜像配置
pip config show
```

### 问题 6: 虚拟环境激活失败

**症状**:
```
command not found: activate
```

**解决方案**:

```bash
# 确保使用正确的激活脚本
source ~/envs/fantasy_world/bin/activate  # Linux/macOS
# 或
~/envs/fantasy_world/Scripts/activate     # Windows (带.bat)

# 验证激活成功
echo $VIRTUAL_ENV  # 应该显示虚拟环境路径
```

---

## 📋 环境配置清单

在开始项目前，确保完成所有项目：

```
环境检查清单:
☐ Python 版本 >= 3.8
☐ NVIDIA 驱动已安装
☐ CUDA Toolkit 已安装 (11.8+)
☐ PyTorch 已安装并支持 CUDA
☐ diffsynth 已安装 (pip install -e .)
☐ 所有依赖已安装 (pip list 检查)
☐ GPU 可用 (nvidia-smi 或 torch.cuda.is_available())
☐ 模型权重已下载 (~/.cache/huggingface)
☐ 虚拟环境已创建并激活

硬件检查清单:
☐ GPU 显存充足 (>= 12GB)
☐ 硬盘空间充足 (>= 50GB)
☐ 网络连接正常

验证检查清单:
☐ python -c "import diffsynth" 成功
☐ python -c "import torch; print(torch.cuda.is_available())" 返回 True
☐ nvidia-smi 显示 GPU 信息
☐ 可以启动训练脚本 (不报 import 错误)
```

---

## 🎯 下一步

环境准备完成后：

1. 查看 [数据准备](./DATA_PREPARATION.md) 准备训练数据
2. 或查看 [训练指南](./TRAINING_GUIDE.md) 开始训练
3. 或查看 [推理指南](./INFERENCE_GUIDE.md) 进行推理

---

## 📞 获取帮助

如环境问题无法解决：

1. 检查 [故障排查](#%EF%B8%8F-故障排查) 部分
2. 查看项目文档: `/docs/`
3. 检查 PyTorch 官方文档: https://pytorch.org/
4. 检查 CUDA 官方文档: https://docs.nvidia.com/cuda/

祝环境配置顺利！ 🚀
