# Fantasy World 训练环境设置指南

## 问题诊断

当你看到这个错误时：
```
ModuleNotFoundError: No module named 'diffsynth'
```

说明 Python 找不到 `diffsynth` 模块。

## 快速解决方案（推荐）

### 1. 进入项目根目录
```bash
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
```

### 2. 以开发模式安装项目
```bash
pip install -e .
```

这个命令会：
- 安装项目的所有依赖
- 创建一个可编辑链接，使得 `diffsynth` 包在系统范围内可导入
- 后续对代码的修改会立即生效，无需重新安装

### 3. 验证安装
```bash
python -c "import diffsynth; print('Success!')"
```

## 完整设置步骤

### 步骤 1：运行诊断脚本
```bash
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
bash scripts/setup_and_diagnose.sh
```

这会自动：
- 检查 Python 版本
- 检查所有依赖
- 验证项目结构
- 测试模块导入
- 如果需要，自动安装

### 步骤 2：创建假数据
```bash
python scripts/create_fake_data.py \
    --output_dir ./test_data/fantasy_world_fake \
    --num_samples 4 \
    --num_frames 9 \
    --height 128 \
    --width 128 \
    --verify
```

### 步骤 3：运行训练（需要模型权重）
```bash
python examples/wanvideo/model_training/train.py \
    --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:diffusion_pytorch_model*.safetensors,..." \
    --dataset_base_path ./test_data/fantasy_world_fake \
    --dataset_metadata_path ./test_data/fantasy_world_fake/metadata.json \
    --output_path ./test_outputs/fantasy_world_test \
    --task fantasy_world \
    --trainable_models "dit" \
    --data_file_keys "video,depth,points,camera_params"
```

## 不同场景的解决方案

### 场景 A：从脚本目录运行训练
如果你从 `examples/wanvideo/model_training/` 目录运行脚本：

```bash
cd examples/wanvideo/model_training/
python train.py ...  # ✗ 会失败
```

**解决方案**：
```bash
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
python examples/wanvideo/model_training/train.py ...  # ✓ 正确
```

或者在脚本中设置 PYTHONPATH：
```bash
cd examples/wanvideo/model_training/
PYTHONPATH=/ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world:$PYTHONPATH python train.py ...
```

### 场景 B：在 Conda 环境中
如果使用特定的 Conda 环境：

```bash
# 激活环境
conda activate fantasy_world_env  # 或你的环境名

# 然后安装
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
pip install -e .
```

### 场景 C：在容器/远程环境中
如果在 Docker 或远程服务器中：

```bash
# 拉取最新代码
git pull

# 确保在项目目录
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world

# 安装
pip install -e .

# 验证
python -c "import diffsynth; print('OK')"
```

## 环境变量配置（可选）

如果你想在任何地方都能运行，设置这些：

```bash
# 添加到 ~/.bashrc 或 ~/.bash_profile
export PYTHONPATH="/ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world:$PYTHONPATH"
export PATH="/ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world/scripts:$PATH"

# 然后重新加载
source ~/.bashrc
```

## 常见问题

### Q: "ModuleNotFoundError: No module named 'diffsynth'" 一直出现
**A**: 确保运行了 `pip install -e .`，然后重启 Python 终端

### Q: "ModuleNotFoundError: No module named 'diffsynth.core.data.fantasy_world_dataset'"
**A**: 检查文件是否存在：
```bash
ls -la diffsynth/core/data/fantasy_world_*.py
```

### Q: 修改代码后仍然生效不了
**A**: 使用开发模式安装 (`pip install -e .`)，修改会自动生效。
如果还不行，清除 Python 缓存：
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### Q: 可以在虚拟环境之外运行吗？
**A**: 可以，但建议始终在虚拟环境中工作：
```bash
# 创建虚拟环境
python -m venv fantasy_world_env
source fantasy_world_env/bin/activate

# 安装依赖
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
pip install -e .
```

## 验证清单

运行这个命令检查所有内容是否正确设置：

```bash
cat > /tmp/check_setup.py << 'EOF'
import sys
import os

checks = [
    ("Python version >= 3.8", lambda: sys.version_info >= (3, 8)),
    ("PyTorch installed", lambda: __import__('torch') and True),
    ("diffsynth importable", lambda: __import__('diffsynth') and True),
    ("Fantasy World dataset", lambda: __import__('diffsynth.core.data.fantasy_world_dataset', fromlist=['FantasyWorldDataset']) and True),
    ("Fantasy World operators", lambda: __import__('diffsynth.core.data.fantasy_world_operators', fromlist=['default_depth_operator']) and True),
    ("Latent Bridge Adapter", lambda: __import__('diffsynth.models.wan_video_dit', fromlist=['LatentBridgeAdapter']) and True),
]

print("Setup Verification Results:")
print("-" * 50)
passed = 0
for check_name, check_fn in checks:
    try:
        check_fn()
        print(f"✓ {check_name}")
        passed += 1
    except Exception as e:
        print(f"✗ {check_name}: {str(e)[:60]}")

print("-" * 50)
print(f"Passed: {passed}/{len(checks)}")
sys.exit(0 if passed == len(checks) else 1)
EOF
python /tmp/check_setup.py
```

## 获取帮助

如果仍有问题，运行诊断脚本并提供输出：
```bash
bash scripts/setup_and_diagnose.sh 2>&1 | tee setup_diagnosis.log
```

然后检查 `setup_diagnosis.log` 中的错误信息。
