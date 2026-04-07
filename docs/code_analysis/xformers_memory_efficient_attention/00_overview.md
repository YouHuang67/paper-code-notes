---
tags:
  - Triton
  - Flash Attention
  - Online Softmax
  - KV Cache
  - LLM Inference
---
# xformers Memory Efficient Attention 总览

**源码仓库**: [facebookresearch/xformers](https://github.com/facebookresearch/xformers)

## 核心思想

xformers 的 `memory_efficient_attention` 是一个统一的高效注意力 API，底层根据硬件和输入条件自动分发到最优的后端实现（CUTLASS / FlashAttention / Triton / CK）。其中 **Triton Split-K** 后端（`triton_splitk`）是纯 Triton 实现的前向推理内核，核心特点：

- **Split-K 并行**：将 KV 序列沿 N 维分成多个 chunk 并行计算，最后归约合并，解决 decode 阶段 batch × head 并行度不足的问题
- **直接 mask 操作**：原生支持因果掩码、局部窗口、块对角、分页注意力等多种 attention bias，在 kernel 内部直接操作而非依赖外部 mask 矩阵
- **INT4/FP8 量化融合**：kernel 内部融合反量化，直接处理 int32 打包的量化 KV，零额外内存开销
- **GQA/MQA 优化**：通过 head-swapping trick 将多查询头映射到序列维度，避免 K/V 展开复制

## API 架构

```
memory_efficient_attention(query, key, value, attn_bias, scale, op)
        │
        ├── dispatch_fw()  →  选择前向后端
        │     ├── cutlass.FwOp         (CUTLASS)
        │     ├── cutlass_blackwell.FwOp (CUTLASS Blackwell)
        │     ├── flash.FwOp           (FlashAttention 2)
        │     ├── flash3.FwOp          (FlashAttention 3)
        │     ├── ck.FwOp              (Composable Kernel, AMD)
        │     └── triton_splitk.FwOp   (Triton Split-K) ← 本文分析
        │
        └── dispatch_bw()  →  选择反向后端
              └── (Triton Split-K 仅有前向，反向使用其他后端)
```

Triton Split-K 后端仅实现前向（`FwOp`），没有反向（`BwOp`）。推理场景下无需反向传播，这正是其设计定位：**高性能推理前向内核**。

## Triton Split-K 后端

### 适用条件

| 属性 | 值 |
|------|------|
| 设备 | CUDA（SM ≥ 8.0，即 A100/H100/L4+） |
| 数据类型 | float16, bfloat16（Q），int32（量化 K/V） |
| Head dim | 16, 32, 64, 128, 256, 512 |
| 最大 query 长度 | 因果/局部 mask 下 ≤ 16；无 mask 不限 |
| Dropout | 不支持 |
| 反向传播 | 不支持 |

### Split-K 值选择

Split-K 的核心在于将 KV 维度切分成 `split_k` 个 chunk 并行计算，然后归约。`split_k` 值由启发式函数 `get_split_k` 根据 B, G, H, Mk, Mq 动态决定：

- **decode 场景**（Mq=1, B×G×H 较小）：split_k 较大（可达 128），充分利用 SM 并行
- **prefill 场景**（Mq>1, B×G×H>64）：split_k=1，无需额外归约开销

### 支持的 Attention Bias

Triton Split-K 后端支持丰富的 mask 类型，这是其区别于简单 Flash Attention 实现的关键优势：

| Bias 类型 | 用途 | 分页 |
|-----------|------|------|
| `None` | 无 mask | - |
| `torch.Tensor` | 任意 additive bias | - |
| `BlockDiagonalCausalWithOffsetPaddedKeysMask` | 因果 + 变长序列 | ✗ |
| `BlockDiagonalCausalLocalAttentionPaddedKeysMask` | 因果 + 局部窗口 | ✗ |
| `BlockDiagonalLocalAttentionPaddedKeysMask` | 非因果局部窗口 | ✗ |
| `BlockDiagonalGappyKeysMask` | 不连续 KV 段 | ✗ |
| `BlockDiagonalPaddedKeysMask` | 块对角 + padding | ✗ |
| `PagedBlockDiagonal*Mask` | 上述各种的分页版本 | ✓ |

分页注意力（Paged Attention）通过 `block_tables` 实现物理页到逻辑页的映射，直接在 kernel 内完成地址转换，与 vLLM 等推理框架的 KV Cache 管理兼容。

### GQA/MQA Head-Swapping Trick

当 K/V 的 head 维度通过 stride=0 广播（即 GQA/MQA）时，kernel 使用 head-swapping trick 避免显式展开：

$$
Q: (B, M_q, G, H_q, D) \xrightarrow{\text{reshape}} (B, H_q \times M_q, G, 1, D)
$$

将 $H_q$ 个查询头映射到序列维度，K/V 只保留 1 个头，通过 `tl.dot` 自然广播。这样 K/V 无需复制，内存访问量不变。

## 代码结构

```
xformers/ops/fmha/
├── __init__.py           # memory_efficient_attention API 入口
├── common.py             # AttentionFwOpBase 基类、Inputs、Context
├── dispatch.py           # 前后向操作符分发逻辑
├── attn_bias.py          # 所有 Attention Bias 类定义
├── triton_splitk.py      # FwOp 类、apply()、merge_attentions()
└── _triton/
    └── splitk_kernels.py # Triton JIT kernel 实现
```

**核心内核一览**：

| 内核 | 功能 | 源码位置 |
|------|------|----------|
| `_fwd_kernel_splitK` | Split-K 前向注意力计算 | [splitk_kernels.py#L31](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L31) |
| `_splitK_reduce` | Split-K 结果归约合并 | [splitk_kernels.py#L908](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L908) |
| `load_dequantize_k_v_group` | K/V 加载 + 反量化 | [splitk_kernels.py#L699](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L699) |
| `FwOp.apply` | 前向入口（reshape + kernel launch） | [triton_splitk.py#L606](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py#L606) |

## 文档导航

1. [Split-K 前向内核](01_splitk_forward.md) - 前向计算核心逻辑、mask 处理、量化融合
2. [Split-K 归约与合并](02_splitk_reduce.md) - Online Softmax 归约、varargs 合并、反向传播
