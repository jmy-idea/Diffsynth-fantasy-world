# RoPE 长度不匹配问题修复说明

## 问题描述

### 错误信息
```
RuntimeError: The size of tensor a (197) must match the size of tensor b (192) at non-singleton dimension 1
```

**位置**: `wan_video_dit.py` → `rope_apply()` 函数
```python
x_out = torch.view_as_real(x_out * freqs).flatten(2)
```

## 根本原因分析

### 1. 序列长度变化

| 模式 | 序列组成 | 总长度 |
|------|----------|--------|
| **标准 Wan DiT** | 视频 latent tokens (F×H×W) | 192 |
| **Fantasy World** | 视频 tokens + Camera token(1) + Register tokens(4) | 192 + 5 = **197** |

### 2. RoPE (Rotary Position Embedding) 机制

RoPE 为序列中的每个位置生成旋转矩阵：

```python
# 预计算 freqs
freqs = precompute_freqs_cis_3d(dim, end=max_seq_len)
# freqs.shape: [seq_len, 1, dim_per_head]

# 应用 RoPE
def rope_apply(x, freqs, num_heads):
    # x: query or key tensor
    # x.shape: [B, seq_len, num_heads * dim_per_head]
    # freqs.shape: [seq_len, 1, dim_per_head]
    
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_complex = torch.view_as_complex(x.reshape(...))
    x_out = x_complex * freqs  # <-- 这里要求 seq_len 匹配！
    return x_out
```

**关键点**: `x_complex` 和 `freqs` 的维度1（sequence length）必须匹配。

### 3. 错误触发路径

```
model_fn_wan_video()
  ├─ freqs = precompute_freqs_cis_3d(..., end=192)  # 只为 192 个位置
  │
  ├─ [标准分支] dit.blocks[i](x, freqs)  ✓ x: [B, 192, C]
  │
  └─ [Fantasy World 分支]
       ├─ x_geo = dit.geo_projector(x)             # [B, 192, C]
       ├─ x_geo = cat([x_geo, cam_tok, reg_toks])  # [B, 197, C]  <-- 长度增加!
       │
       └─ dit.geo_blocks[i](x_geo, freqs)          # ✗ x_geo: [B, 197, C]
            └─ GeoDiTBlock.self_attn(x_geo, freqs)      # freqs: [192, 1, D]
                 └─ rope_apply(q, freqs, num_heads)     # ✗ 197 != 192
```

## 解决方案

### 核心思路

为额外的 5 个 tokens（camera + register）扩展 freqs，使用**零频率**（相当于不施加位置编码）。

**理论依据**:
1. **Camera token**: 全局语义，不需要空间位置信息
2. **Register tokens**: 辅助信息存储，位置无关
3. **Video tokens**: 需要完整的 3D RoPE（时间+空间）

### 代码修复

**修改位置**: `wan_video.py` → `model_fn_wan_video()` → Fantasy World geometry branch

#### 修复前（有 Bug）
```python
if use_gradient_checkpointing:
    # TODO: Handle freqs padding in gradient checkpointing if needed
    x_geo = torch.utils.checkpoint.checkpoint(
        create_custom_forward(dit.geo_blocks[idx]), 
        x_geo, context, t_mod, freqs,  # <-- freqs 长度不匹配!
        use_reentrant=False
    )
else:
    # 只有非 checkpointing 分支有 freqs 扩展
    if x_geo.shape[1] > freqs.shape[0]:
        freqs_ext = torch.cat([freqs, torch.zeros(...)], dim=0)
    x_geo = dit.geo_blocks[idx](x_geo, context, t_mod, freqs_ext, pose_emb)
```

#### 修复后（已修复）
```python
# 统一在两个分支前扩展 freqs
if x_geo.shape[1] > freqs.shape[0]:
    extra_len = x_geo.shape[1] - freqs.shape[0]
    # 为额外 token 填充零频率
    freqs_ext = torch.cat([
        freqs,  # 原始 video tokens 的 RoPE
        torch.zeros(extra_len, *freqs.shape[1:], 
                   device=freqs.device, dtype=freqs.dtype)
    ], dim=0)
else:
    freqs_ext = freqs

# 两个分支都使用扩展后的 freqs_ext
if use_gradient_checkpointing:
    x_geo = torch.utils.checkpoint.checkpoint(
        create_custom_forward(dit.geo_blocks[idx]), 
        x_geo, context, t_mod, freqs_ext, pose_emb,  # ✓ 使用 freqs_ext
        use_reentrant=False
    )
else:
    x_geo = dit.geo_blocks[idx](x_geo, context, t_mod, freqs_ext, pose_emb)
```

### 扩展后的 freqs 结构

```python
# 原始 freqs: [192, 1, dim_per_head]
# 扩展后 freqs_ext: [197, 1, dim_per_head]

freqs_ext = [
    freqs[0],      # Video token 0 的 RoPE
    freqs[1],      # Video token 1 的 RoPE
    ...
    freqs[191],    # Video token 191 的 RoPE
    # --- 以下是扩展部分 ---
    zeros,         # Camera token (位置 192) - 零频率
    zeros,         # Register token 0 (位置 193) - 零频率
    zeros,         # Register token 1 (位置 194) - 零频率
    zeros,         # Register token 2 (位置 195) - 零频率
    zeros,         # Register token 3 (位置 196) - 零频率
]
```

**效果**: 
- Video tokens (0-191): 正常的 3D RoPE，保留时空位置信息
- Camera token (192): 零频率 → 旋转矩阵为单位矩阵 → 无位置编码
- Register tokens (193-196): 零频率 → 无位置编码

## 技术细节

### RoPE 的数学原理

RoPE 通过复数乘法实现旋转：

$$
\text{RoPE}(x, \theta, m) = x \cdot e^{im\theta}
$$

其中：
- $x$: query/key 向量
- $\theta$: 频率参数
- $m$: 位置索引

**零频率的效果**:
$$
\theta = 0 \Rightarrow e^{i \cdot 0} = 1 \Rightarrow \text{RoPE}(x, 0, m) = x
$$

即不施加任何旋转变换，等价于无位置编码。

### 为什么不直接跳过额外 tokens？

**方案对比**:

| 方案 | 优点 | 缺点 |
|------|------|------|
| **A: 跳过额外 tokens** | 节省计算 | 需要大量代码改动，在 attention 中手动 mask |
| **B: 零频率扩展 (本方案)** | 简单统一，代码改动最小 | 额外的 5 个零值计算（开销极小） |

### Gradient Checkpointing 的影响

Gradient Checkpointing 时，传入 `checkpoint()` 的参数必须是 tensor，因此必须在**调用前**准备好 `freqs_ext`。

```python
# ✗ 错误：在 custom_forward 内部扩展 freqs
def custom_forward(module):
    def forward(*inputs):
        x, context, t_mod, freqs, pose_emb = inputs
        freqs_ext = expand_freqs(freqs, x.shape[1])  # <-- 无法追踪梯度!
        return module(x, context, t_mod, freqs_ext, pose_emb)
    return forward

# ✓ 正确：在外部扩展 freqs，作为参数传入
freqs_ext = expand_freqs(freqs, x_geo.shape[1])
x_geo = torch.utils.checkpoint.checkpoint(
    custom_forward(module), 
    x_geo, context, t_mod, freqs_ext, pose_emb  # <-- freqs_ext 作为 tensor 参数
)
```

## 验证方法

### 1. 检查序列长度

```python
# 在 model_fn_wan_video 中添加调试信息
print(f"[Debug] x_geo.shape: {x_geo.shape}")       # 应为 [B, 197, C]
print(f"[Debug] freqs.shape: {freqs.shape}")       # 原始 [192, 1, D/H]
print(f"[Debug] freqs_ext.shape: {freqs_ext.shape}") # 扩展后 [197, 1, D/H]
```

### 2. 验证 RoPE 应用

```python
# 在 SelfAttention.forward() 中
q = rope_apply(q, freqs, self.num_heads)  # 应该不再报错
```

### 3. 运行训练测试

```bash
cd /ML-vePFS/research_gen/jmy/jmy_ws/Diffsynth-fantasy-world
bash examples/wanvideo/model_training/full/test_fantasy_world_training.sh
```

**预期**: 不再出现 `RuntimeError: The size of tensor a (197) must match the size of tensor b (192)`

## 相关代码位置

| 文件 | 函数/类 | 作用 |
|------|---------|------|
| `wan_video.py` | `model_fn_wan_video()` | **修复位置**: 扩展 freqs |
| `wan_video_dit.py` | `rope_apply()` | 应用 RoPE，触发错误的地方 |
| `wan_video_dit.py` | `SelfAttention.forward()` | 调用 rope_apply |
| `wan_video_dit.py` | `GeoDiTBlock.forward()` | Geometry branch 的 attention block |
| `wan_video_dit.py` | `WanModel.enable_fantasy_world_mode()` | 初始化 camera/register tokens |

## 总结

### 问题本质
Fantasy World 模式在 geometry branch 中添加了额外的 tokens，但 RoPE freqs 未相应扩展，导致序列长度不匹配。

### 解决方案
在**调用 GeoDiTBlock 前**，统一扩展 freqs 以匹配新的序列长度，为额外 tokens 使用零频率（无位置编码）。

### 关键改进
1. ✅ 修复了 gradient checkpointing 分支的 freqs 扩展缺失
2. ✅ 统一了两个分支的处理逻辑
3. ✅ 保持了代码的简洁性和可维护性

### 设计合理性
- Camera token 和 register tokens 本质上是全局信息，不需要位置编码
- 使用零频率是最优雅的解决方案，符合 RoPE 的数学特性
- 对性能影响极小（仅 5 个额外的零值操作）
