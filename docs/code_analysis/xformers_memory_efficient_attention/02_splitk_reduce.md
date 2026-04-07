---
tags:
  - Triton
  - Online Softmax
---
# Split-K 归约与合并

本节分析 Split-K 的归约 kernel，将多个 chunk 的 partial attention 和 LSE 合并为最终输出。这是 Split-K 技术的关键第二步。

**源码位置**: [splitk_kernels.py#L908-L1274](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L908-L1274) / [triton_splitk.py#L1047-L1098](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py#L1047-L1098)

## 1. 归约的数学原理

Split-K 将 KV 序列分成 $S$ 个 chunk，每个 chunk $s$ 独立计算 partial attention $\mathbf{o}_s$ 和 partial LSE $\ell_s$。归约的目标是将这些 partial 结果合并为全局正确的输出。

每个 chunk 的 partial attention 已经做了 softmax 归一化：

$$
\mathbf{o}_s = \frac{\sum_{j \in \text{chunk}_s} e^{q k_j^T - m_s} v_j}{\sum_{j \in \text{chunk}_s} e^{q k_j^T - m_s}} = \frac{N_s}{D_s}
$$

其中 $\ell_s = \log D_s + m_s$ 是 partial LSE。

全局输出的合并公式：

$$
\mathbf{o} = \frac{\sum_s D_s \cdot \mathbf{o}_s}{\sum_s D_s} = \frac{\sum_s e^{\ell_s} \cdot \mathbf{o}_s}{\sum_s e^{\ell_s}}
$$

为了数值稳定性，引入全局最大值 $m^* = \max_s \ell_s$：

$$
\mathbf{o} = \frac{\sum_s e^{\ell_s - m^*} \cdot \mathbf{o}_s}{\sum_s e^{\ell_s - m^*}}
$$

全局 LSE：

$$
\ell = m^* + \log \sum_s e^{\ell_s - m^*}
$$

## 2. _splitK_reduce Kernel

**源码位置**: [splitk_kernels.py#L908-L1020](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L908-L1020)

这是堆叠张量版本的归约 kernel，输入为 `(B, G, H, split_k, M, K)` 的 6D 张量。

### 2.1 Grid 与索引

```python
@triton.jit
def _splitK_reduce(Out_splitK, LSE_splitK, Out, LSE, ...):
    '''
    grid = (M, B*G*H, 1)
    每个 program 处理一个 query position 的所有 split_k 个 partial 结果
    '''
    off_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_h = (off_zhg // G) % H
    off_g = off_zhg % G
```

### 2.2 归约计算

```python
    '''
    加载所有 chunk 的 partial attention 和 LSE
    splitK_pow2 是 split_k 向上取到 2 的幂（Triton 要求向量操作的维度为 2 的幂）
    '''
    head_dim_mask = tl.arange(0, head_dim_pow_2) < head_dim

    Out_splitK_ptr = (
        Out_splitK + stride_osk_z * off_z + stride_osk_g * off_g
        + stride_osk_h * off_h + stride_osk_m * off_m
        + tl.arange(0, head_dim_pow_2)[None, :]        # [1, head_dim_pow_2]
        + stride_osk_s * tl.arange(0, splitK_pow2)[:, None]  # [splitK_pow2, 1]
    )
    # 一次性加载所有 chunk: [splitK_pow2, head_dim_pow_2]
    out_splitk = tl.load(Out_splitK_ptr, mask=mask_2d, other=0)
    lse_splitk = tl.load(LSE_splitK_ptr0, mask=mask_1d, other=float("-inf"))

    '''
    归约核心计算：
    1. lse_max = max(lse_splitk)              全局最大 LSE
    2. weight_s = exp2((lse_s - lse_max) * log2e)   各 chunk 的归一化权重
    3. sum_weights = sum(weight_s)             权重之和
    4. acc = sum(out_s * weight_s) / sum_weights    加权平均
    '''
    lse_max = tl.max(lse_splitk)
    sumexp_normalized_splitk = tl.math.exp2(
        (lse_splitk - lse_max).to(tl.float32) * 1.44269504
    )                                                    # [splitK_pow2]
    sumexp_normalized = tl.sum(sumexp_normalized_splitk, axis=0)
    numerator_normalized = tl.sum(
        out_splitk * sumexp_normalized_splitk[:, None], axis=0
    )                                                    # [head_dim_pow_2]
    acc = numerator_normalized / sumexp_normalized
    acc = tl.where(lse_max == float("-inf"), 0.0, acc)   # 全 -inf 时填零

    tl.store(Out_ptr, acc, mask=head_dim_mask)

    '''
    全局 LSE 写回：
    lse = lse_max + ln(sum_weights) = lse_max + log2(sum_weights) / log2(e)
    '''
    if WRITE_LSE:
        to_store = lse_max + tl.math.log2(sumexp_normalized) / 1.44269504
        to_store = tl.where(lse_max == float("-inf"), lse_max, to_store)
        tl.store(l_ptrs, to_store)
```

关键设计：所有 split_k 个 chunk 的数据通过一次 `tl.load` 批量读入（利用 splitK_pow2 维度的向量化），然后用一次 `tl.sum` 完成归约。这比逐 chunk 循环更高效。

## 3. merge_attentions 入口

**源码位置**: [triton_splitk.py#L1047-L1098](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py#L1047-L1098)

Python 端的 `merge_attentions` 函数负责维度检查和 kernel launch：

```python
def merge_attentions(attn_out, lse_out, attn_split, lse_split):
    B, M, G, H, Kq = attn_out.shape
    B1, G1, H1, split_k, M1, Kq1 = attn_split.shape

    num_warps = 4 if B * G * H < 32 or torch.version.hip else 2
    splitK_pow2 = triton.next_power_of_2(split_k)
    grid = (M, B * G * H, 1)
    _splitK_reduce[grid](
        attn_split, lse_split, attn_out, lse_out,
        split_k=split_k, splitK_pow2=splitK_pow2,
        ...,
    )
```

当 `B*G*H < 32` 时使用 4 个 warp（增加并行度弥补 grid 较小），否则使用 2 个 warp。AMD 上始终使用 4 个 warp。

## 4. Varargs 归约

**源码位置**: [splitk_kernels.py#L1024-L1133](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L1024-L1133) / [triton_splitk.py#L1101-L1154](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py#L1101-L1154)

`_splitK_reduce_varargs` 是另一种归约 kernel，接受 **列表** 形式的 partial attention/LSE（而非堆叠张量）。这用于 `merge_attentions_varargs`——合并来自不同 attention 调用的 partial 结果（如 prefix caching 场景）。

### 4.1 Kernel 结构

```python
@triton.jit
def _splitK_reduce_varargs(
    Out_splitK: "VAR_ARGS_ARRAY",   # N 个 [B, G, H, M, K] tensor
    LSE_splitK: "VAR_ARGS_ARRAY",   # N 个 [B, G, H, M] tensor
    Out, LSE, ...
):
    '''
    与 _splitK_reduce 的区别：
    1. 输入是变长 tensor 列表而非堆叠张量，每个 tensor 可以有不同 stride
    2. 使用两遍循环：第一遍找全局 max，第二遍累加
    3. 通过 unroll_varargs 预处理展开为固定长度
    '''
    lse_max = float("-inf")
    for split_k_idx in range(len(Out_splitK)):
        lse_splitk = tl.load(LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx])
        lse_max = tl.maximum(lse_max, lse_splitk)

    sumexp_normalized = 0.0
    numerator_normalized = tl.zeros([head_dim_pow_2], dtype=tl.float32)
    for split_k_idx in range(len(Out_splitK)):
        out_splitk = tl.load(Out_splitK[split_k_idx] + out_splitk_offset[split_k_idx])
        lse_splitk = tl.load(LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx])
        sumexp_normalized_splitk = tl.math.exp2((lse_splitk - lse_max) * 1.44269504)
        sumexp_normalized += sumexp_normalized_splitk
        numerator_normalized += out_splitk * sumexp_normalized_splitk

    acc = numerator_normalized / sumexp_normalized
```

### 4.2 unroll_varargs 预处理

Triton 不原生支持张量列表参数。`unroll_varargs` 是一个 AST 变换工具，将 `VAR_ARGS_ARRAY` 标注的参数在编译前展开为固定数量的具名参数。例如 N=3 时：

```python
# 展开前
Out_splitK: "VAR_ARGS_ARRAY"
for i in range(len(Out_splitK)):
    tl.load(Out_splitK[i])

# 展开后
Out_splitK_0, Out_splitK_1, Out_splitK_2,
tl.load(Out_splitK_0)
tl.load(Out_splitK_1)
tl.load(Out_splitK_2)
```

前向 kernel `_fwd_kernel_splitK` 也使用了同样的机制来处理多 group 量化——`q`、`k`、`v`、`acc` 都是 VAR_ARGS_ARRAY，展开为 N_GROUPS 个独立变量。

### 4.3 torch.library 注册

`merge_attentions_varargs` 通过 `@torch.library.custom_op` 注册为 PyTorch custom op，这使得它可以：
- 与 `torch.compile` 兼容
- 正确传播梯度（虽然反向抛出 NotImplementedError）
- 被 FakeTensor（`register_fake`）追踪用于编译时 shape 推导

## 5. Varargs 反向传播

**源码位置**: [splitk_kernels.py#L1136-L1274](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L1136-L1274) / [triton_splitk.py#L1199-L1239](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py#L1199-L1239)

`_splitK_reduce_varargs_backward` 计算归约操作本身的梯度（注意这不是注意力的反向传播，而是 merge 操作的反向）。

### 5.1 梯度推导

设合并的输出为：

$$
\mathbf{o} = \frac{\sum_s w_s \cdot \mathbf{o}_s}{\sum_s w_s}, \quad w_s = e^{\ell_s - m^*}
$$

对 $\mathbf{o}_s$ 的梯度：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{o}_s} = \frac{\partial \mathcal{L}}{\partial \mathbf{o}} \cdot \frac{w_s}{\sum_s w_s} = \frac{\partial \mathcal{L}}{\partial \mathbf{o}} \cdot \frac{e^{\ell_s}}{e^{\ell}}
$$

对 $\ell_s$ 的梯度（包含通过 $\mathbf{o}$ 和通过 $\ell$ 两条路径）：

$$
\frac{\partial \mathcal{L}}{\partial \ell_s} = \frac{e^{\ell_s}}{e^{\ell}} \left[ \frac{\partial \mathcal{L}}{\partial \ell} + \sum_d \frac{\partial \mathcal{L}}{\partial \mathbf{o}_d} (\mathbf{o}_{s,d} - \mathbf{o}_d) \right]
$$

### 5.2 Kernel 实现

```python
for split_k_idx in range(len(Out_splitK)):
    out_splitk = tl.load(Out_splitK[split_k_idx] + ...)
    lse_splitk = tl.load(LSE_splitK[split_k_idx] + ...)

    '''
    dattn_dattn_i = exp(lse_s - lse_max) / exp(lse - lse_max) = w_s / sum(w)
    这是 chunk s 在最终输出中的权重
    '''
    dattn_dattn_i = tl.exp(lse_splitk - lse_max) / tl.exp(lse - lse_max)
    dX_dattn_i = dattn_dattn_i * dattn            # d(Loss)/d(o_s)
    tl.store(dout_splitk_ptr, dX_dattn_i)

    '''
    d(Loss)/d(lse_s) 包含两条梯度路径：
    1. 通过 lse: dlse * dlse/dlse_i = dlse * dattn_dattn_i
    2. 通过 attn: sum_d(dattn_d * (o_s_d - o_d) * dattn_dattn_i)
    '''
    dattn_dlse_i = (out_splitk - out) * dattn_dattn_i
    dlse_dlse_i = dattn_dattn_i
    dX_dlse_i = dlse_dlse_i * dlse + tl.sum(dattn_dlse_i * dattn)
    tl.store(dlse_splitk_ptr, dX_dlse_i)
```

## 6. 端到端数据流

```
Input: Q [B, Mq, G, Hq, D], K/V [B, Mk, G, Hkv, D]
           │
           ▼
    ┌─ FwOp.apply ─┐
    │  reshape      │  GQA head-swap, bias 解析, 量化检测
    │  split_k 选择 │  get_split_k() 启发式
    └──────┬────────┘
           │
           ▼
    ┌─ _fwd_kernel_splitK ─┐
    │  grid: (M//BM,        │
    │         B*G*H,        │  3D grid 并行
    │         split_k)      │
    │                       │
    │  for start_n in       │  KV 迭代
    │    range(lo, hi, BN): │
    │    load K/V + 反量化  │
    │    qk = Q @ K^T       │  [BM, BN]
    │    apply mask          │  因果/局部/additive
    │    online softmax      │  m_i, l_i, acc 更新
    │                       │
    │  store partial o, LSE │
    └──────┬────────────────┘
           │
           ▼ (split_k > 1 时)
    ┌─ _splitK_reduce ─┐
    │  grid: (M, B*G*H) │  2D grid
    │                    │
    │  load all chunks   │  [split_k, head_dim]
    │  weight = exp(LSE) │  LSE 加权
    │  out = Σ(w*o) / Σw │  归一化合并
    │  store final o,LSE │
    └──────┬─────────────┘
           │
           ▼
    Output: [B, Mq, G, Hq, D]
```
