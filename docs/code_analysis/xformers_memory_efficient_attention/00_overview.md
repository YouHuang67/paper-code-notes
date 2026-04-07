---
tags:
  - Triton
  - Flash Attention
  - Online Softmax
  - KV Cache
  - LLM Inference
---
# xformers Triton Split-K Attention

**源码仓库**: [facebookresearch/xformers](https://github.com/facebookresearch/xformers)

xformers 的 `memory_efficient_attention` API 底层有多个后端（CUTLASS / FlashAttention / Triton / CK），其中 **Triton Split-K** 是纯 Triton 实现的前向推理内核，仅有前向无反向，定位于高性能推理。核心特点：

- **Split-K 并行**：KV 序列沿 N 维分成多个 chunk 并行计算后归约，解决 decode 阶段并行度不足
- **直接 mask 操作**：因果掩码、局部窗口、块对角、分页注意力等在 kernel 内部通过条件判断实现
- **INT4/FP8 量化融合**：kernel 内部融合反量化，直接处理 int32 打包的量化 KV
- **GQA/MQA 优化**：head-swapping trick 将多查询头映射到序列维度，避免 K/V 展开复制

涉及的核心文件：

| 文件 | 内容 |
|------|------|
| [triton_splitk.py](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py) | FwOp 类、apply() 入口、merge_attentions() |
| [splitk_kernels.py](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py) | Triton JIT kernel：前向、归约、反量化 |

## 1. FwOp.apply 张量预处理

**源码位置**: [triton_splitk.py#L606-L1012](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py#L606-L1012)

kernel launch 前将用户输入统一到 kernel 期望的 `(Bqq, Mqq, G, H, K)` 形状。G 是 GQA group 数，H 是每个 group 内的 head 数。

### 1.1 Attention Bias 解析

apply 的第一步区分 tensor bias 和结构化 bias：

```python
if not isinstance(inp.attn_bias, torch.Tensor):
    attn_bias_tensor = None
    attn_bias = cast(Optional[Union[...]], inp.attn_bias)
else:
    attn_bias_tensor = inp.attn_bias
    attn_bias = None
```

- **tensor bias**（`torch.Tensor`）：任意 additive bias，shape `(B, G, H, Mq, Mkv)` 或 `(B, H, Mq, Mkv)`，kernel 内加到 QK^T 上
- **结构化 bias**（`BlockDiagonal*Mask`）：提供 `seq_len`、`seq_starts_k` 等元数据，kernel 通过条件判断实现 mask

对结构化 bias 提取序列信息：

```python
seq_len = attn_bias.k_seqinfo.seqlen         # [B] 每个 batch 的 KV 长度
seq_starts_k = attn_bias.k_seqinfo.seqstart   # [B+1] KV 起始偏移（gappy bias）
seq_starts_q = attn_bias.q_seqinfo.seqstart    # [B+1] Q 起始偏移（variable q）
IS_CAUSAL = _is_supported_causal_bias(attn_bias)
IS_LOCAL = _is_supported_local_bias(attn_bias)
```

支持的结构化 bias 包括因果 + 变长（`BlockDiagonalCausalWithOffsetPaddedKeysMask`）、因果 + 局部窗口（`BlockDiagonalCausalLocalAttentionPaddedKeysMask`）、非因果局部窗口、不连续 KV 段（Gappy）、块对角 + padding，以及以上所有的分页版本（`PagedBlockDiagonal*Mask`）。分页注意力通过 `block_tables` 实现物理页到逻辑页映射，与 vLLM 等推理框架的 KV Cache 管理兼容。

### 1.2 GQA Head-Swapping Trick

当 K/V 通过 `stride=0` 广播（GQA/MQA），启用 head-swapping：

```python
'''
Head-swapping: 将 Hq 个查询头映射到序列维度
Q: (B, Mq, G, Hq, D) → (B, Hq*Mq, G, 1, D)
K/V: 只取第 0 个 head slice → (B, Mkv, G, 1, D)
kernel 只看到 H=1，序列长度变为 Hq*Mq，K/V 无需复制
'''
if k.stride(3) == 0 and v.stride(3) == 0 and attn_bias_tensor is None:
    mqa_swap_seqlen_head = True
    q = q.permute(0, 3, 1, 2, 4).reshape(B, -1, G, 1, Kq)  # [B, Hq*Mq, G, 1, D]
    k = k[:, :, :, :1]                                        # [B, Mkv, G, 1, D]
    v = v[:, :, :, :1]                                        # [B, Mkv, G, 1, D]
```

variable_q 时 permute 顺序不同——Hq 放在 Mq 之后，保证同一 batch 元素的 query token 在内存中连续。

### 1.3 INT4/FP8 量化 K/V 处理

量化 K/V 以 int32 存储，每个 int32 打包多个量化值：

```python
if k.dtype == torch.int32:
    if k_fp8_scale_shift is not None:
        Lk = k.shape[-1] * 4        # FP8: 每个 int32 = 4 个 fp8
        PACKED_PER_VAL = 4
    else:
        PACKED_PER_VAL = 8           # INT4: 每个 int32 = 8 个 int4
        Lk = (k.shape[-1] - NUM_GROUPS) * 8
else:
    Lk = k.shape[-1]                # 非量化
    PACKED_PER_VAL = 1
```

INT4 的行布局为 `[quant_coef_0, ..., quant_coef_{G-1} | group0_data... | group1_data...]`，量化系数和数据打包在同一行内。FP8 的 scale/shift 存储在独立 tensor `k_fp8_scale_shift` 中。

### 1.4 Split-K 值选择与输出分配

**源码位置**: [triton_splitk.py#L337-L380](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py#L337-L380)

`get_split_k` 根据 B, H, Mk, Mq 启发式选择 split_k：

```python
'''
核心逻辑：
- prefill (Mq>1, B*G*H>64): split_k=1，Q 维度已有足够并行度
- decode (Mq=1 或 B*G*H 较小): split_k = max(Mk, 1024) / (B*H)
  然后 halving 直到 chunk_size >= max_chunk_size (64 或 128)
- 上界 64（CUDA）或 512（AMD）
'''
if Mq > 1 and B * G * H > 64:
    return 1
split_k = max(Mk, 1024) // bh
max_chunk_size = 64 if Mk <= 512 and bh <= 64 else 128
while split_k > Mk / max_chunk_size:
    split_k = split_k // 2
```

Split-K 模式下分配中间缓冲区：

```python
if IS_SPLITK:
    o_splitk = torch.empty(
        [Bqq, G, H, split_k, M_ceil, Kq],  # 每个 chunk 一份 partial attention
        dtype=torch.float32, device=q.device,
    )
    lse_splitk = torch.empty(
        [Bqq, G, H, split_k, Mqq],          # 每个 chunk 一份 partial LSE
        dtype=torch.float32, device=q.device,
    )
```

split_k=1 时直接写入最终输出，跳过归约。

## 2. _fwd_kernel_splitK 前向 Kernel

**源码位置**: [splitk_kernels.py#L31-L587](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L31-L587)

### 2.1 Grid 与并行策略

```python
def grid(META):
    return triton.cdiv(M, META["BLOCK_M"]), B * G * H, split_k
```

三维 grid：dim0 是 Q 序列分块，dim1 是 batch × group × head，dim2 是 Split-K chunk 索引。每个 kernel instance 负责 BLOCK_M 个 query × BLOCK_N_PER_SPLIT 个 key 的注意力子块。

### 2.2 索引解码与 Split-K 边界

```python
@triton.jit
def _fwd_kernel_splitK(Q, K, V, sm_scale, Out_splitK, LSE_splitk, ...):
    '''
    编译前由 unroll_varargs 处理：将 VAR_ARGS_ARRAY 标注的变量
    展开为长度 N_GROUPS 的列表，解决 Triton 不支持张量列表的问题
    '''
    start_m = tl.program_id(0)       # Q 分块索引
    off_zhg = tl.program_id(1)       # batch*group*head 线性索引
    off_z = off_zhg // (H * G)       # batch 索引
    off_h = (off_zhg % (H * G)) // G # head 索引
    off_g = off_zhg % G              # group 索引
    splitk_idx = tl.program_id(2)    # Split-K chunk 索引

    '''
    序列长度获取：
    - USE_SEQ_LEN=True: 从 Seq_len tensor 读取当前 batch 的 KV 长度
    - USE_SEQ_LEN=False: 使用编译期常量 N_CTX_K
    Gappy bias 额外从 Seq_starts_k 读取 KV 起始偏移
    '''
    if USE_SEQ_LEN:
        kv_len = tl.load(Seq_len + off_z)
    else:
        kv_len = N_CTX_K

    '''
    Split-K chunk 边界：
    - 非分页: lo = splitk_idx * BLOCK_N_PER_SPLIT, hi = min(lo + BLOCK_N_PER_SPLIT, kv_len)
    - 分页模式: 对齐到 BLOCK_N 边界，避免跨页访问
    '''
    chunk_hi = (splitk_idx + 1) * BLOCK_N_PER_SPLIT
    chunk_lo = splitk_idx * BLOCK_N_PER_SPLIT
    if PAGE_SIZE > 0:
        BLOCKS_IN_PAGE = PAGE_SIZE // BLOCK_N
        lo = (tl.maximum(chunk_lo, start_kv_idx) // BLOCK_N) * BLOCK_N
        hi = ((chunk_hi + shift) // BLOCK_N) * BLOCK_N
        hi = tl.minimum(hi, kv_len + start_kv_idx)
    else:
        lo = chunk_lo
        hi = tl.minimum(chunk_hi, kv_len)
```

### 2.3 Q 加载与 Online Softmax 初始化

```python
    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_m * stride_qm + off_h * stride_qh
             + off_z * stride_qz * queries_use_batch_dim + off_g * stride_qg,
        shape=(q_len, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_PER_GROUP),
        order=(1, 0),
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # running sum(exp)
    acc = tl.zeros([BLOCK_M, D_PER_GROUP], dtype=internal_dtype) # running weighted sum

    '''
    预乘 log2(e) 使后续用 exp2 替代 exp：e^x = 2^{x * log2(e)}
    tl.math.exp2 在 GPU 上通常比 tl.exp 更快
    '''
    log2e = tl.full((), 1.44269504, tl.float32)
    qk_scale = sm_scale * log2e

    q = tl.load(tl.advance(Q_block_ptr, (0, i * D_PER_GROUP)), boundary_check=(0,))
```

对于多 group 量化，Q 按 group 分段加载，每段 D_PER_GROUP 维。

### 2.4 因果/局部 Mask 实现

因果 mask 和局部窗口不使用额外 mask 矩阵，而是通过对角线偏移的条件判断：

```python
    '''
    因果 mask 推导（假设 num_queries <= BLOCK_M）：
      kv_pos = kv_start + range(0, BLOCK_N)
      q_pos = kv_start + kv_len - num_queries + (q_offset % num_queries)
      mask = q_pos >= kv_pos

    简化为:
      diag_idx = q_offset % NUM_QUERIES_CAUSAL - range(0, BLOCK_N)
      diag_idx_shifted = diag_idx - NUM_QUERIES_CAUSAL + kv_len
      因果条件: diag_idx_shifted >= start_n - start_kv_idx
      左窗口: diag_idx_shifted < start_n - start_kv_idx + WINDOW_LEFT + 1
      右窗口: diag_idx_shifted >= start_n - start_kv_idx - WINDOW_RIGHT
    '''
    if IS_CAUSAL or IS_LOCAL:
        q_offset = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        diag_idx = (q_offset[:, None] % NUM_QUERIES_CAUSAL) - tl.arange(0, BLOCK_N)[None, :]
        diag_idx_shifted = diag_idx - NUM_QUERIES_CAUSAL + kv_len
```

对角线偏移在循环外一次计算，循环内只需与当前 `start_n` 比较——这是推理场景 query 长度短（≤16）的特化优化。

### 2.5 主循环：K/V 迭代与注意力计算

```python
    for start_n in range(lo, hi, BLOCK_N):
        '''
        分页注意力地址转换：
        logical_block_idx → logical_page_idx → 查 block_table → physical_page_idx
        物理偏移 = physical_page_idx * PAGE_SIZE + block_offset_in_page * BLOCK_N
        每次迭代重建 K/V block_ptr，因为物理地址不连续
        '''
        if PAGE_SIZE > 0:
            block_offset_in_page = logical_block_idx % BLOCKS_IN_PAGE
            logical_page_idx = logical_block_idx // BLOCKS_IN_PAGE
            physical_page_idx = tl.load(
                block_table + stride_blocktablesl * logical_page_idx
            ).to(tl.int32)
            offset = physical_page_idx * PAGE_SIZE + block_offset_in_page * BLOCK_N
            K_block_ptr = tl.make_block_ptr(
                base=k_base + stride_kk * INT4_QUANTIZED * N_GROUPS,
                shape=(PACKED_D_PER_GROUP, offset + current_block_size),
                strides=(stride_kk, stride_kn),
                offsets=(0, offset),
                block_shape=(PACKED_D_PER_GROUP, BLOCK_N),
                order=(0, 1),
            )

        '''
        K/V 加载与反量化：
        load_dequantize_k_v_group 根据量化模式加载并反量化一个 group 的 K/V
        '''
        for i in range(N_GROUPS):
            k[i], v[i] = load_dequantize_k_v_group(
                K_block_ptr, V_block_ptr,
                K_scale_shift_block_ptr, V_scale_shift_block_ptr,
                BOUNDS_CHECKS_N, PACKED_PER_VAL, PACKED_D_PER_GROUP,
                FP8_QUANTIZED, Q.dtype.element_ty, i, IS_HIP,
            )

        '''
        QK^T 计算：多 group 的 dot product 累加
        qk = sum_g q[g] @ k[g]，每个 group 的 D_PER_GROUP 维分别 dot 再求和
        '''
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for i in range(N_GROUPS):
            qk += tl.dot(q[i], k[i])                    # [BLOCK_M, BLOCK_N]
        qk *= qk_scale

        if HAS_ADDITIVE_BIAS:
            loaded_bias = tl.load(additive_bias_block_ptr, boundary_check=(0, 1))
            qk += loaded_bias.to(tl.float32) * log2e     # additive bias 也乘 log2e

        '''
        Mask 应用：全部用 tl.where 将 masked 位置设为 -inf
        '''
        if BOUNDS_CHECKS_N:
            qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(diag_idx_shifted >= start_n - start_kv_idx, qk, float("-inf"))
        if IS_LOCAL:
            qk = tl.where(
                diag_idx_shifted < start_n - start_kv_idx + WINDOW_LEFT + 1,
                qk, float("-inf"),
            )
            if not IS_CAUSAL and WINDOW_RIGHT >= 0:
                qk = tl.where(
                    diag_idx_shifted >= start_n - start_kv_idx - WINDOW_RIGHT,
                    qk, float("-inf"),
                )

        '''
        Online Softmax 更新（标准 FlashAttention 的 safe softmax rescale）：
        m_i_new = max(m_i, max(qk))
        alpha = 2^(m_i - m_i_new)           旧 max 到新 max 的缩放因子
        p = 2^(qk - m_i_new)                当前 block 的 unnormalized softmax
        l_i = l_i * alpha + sum(p)           更新 sum(exp)
        acc = acc * alpha + p @ v            更新 weighted sum
        注意整个 block 可能全被 mask 掉（m_i_new = -inf），此时 alpha=0, p=0
        '''
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))            # [BLOCK_M]
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])             # [BLOCK_M, BLOCK_N]
        if HAS_ADDITIVE_BIAS or IS_CAUSAL or IS_LOCAL:
            alpha = tl.where(m_i_new == float("-inf"), 0, alpha)
            p = tl.where(m_i_new[:, None] == float("-inf"), 0, p)

        l_i = l_i * alpha + tl.sum(p, 1)                    # [BLOCK_M]
        m_i = m_i_new
        p = p.to(Q.dtype.element_ty)                         # cast 回 fp16/bf16 用于 dot

        for i in range(N_GROUPS):
            acc[i] *= alpha[:, None]                         # 缩放历史累积
            acc[i] += tl.dot(p, v[i])                        # 累加当前 block 贡献

        if not PAGE_SIZE:
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
```

### 2.6 输出写回与 LSE 存储

```python
    '''
    写回 partial attention：acc / l_i（l_i=0 时填零避免 NaN）
    每个 Split-K chunk 输出已经做了 softmax 归一化
    '''
    for i in range(N_GROUPS):
        attn_out = tl.where(l_i[:, None] == 0, 0.0, acc[i] / l_i[:, None])
        tl.store(
            tl.advance(O_block_ptr, (0, i * D_PER_GROUP)),
            attn_out.to(Out_splitK.dtype.element_ty),
            boundary_check=(0,),
        )

    '''
    写回 partial LSE：
    m_i 是 log2 域的（因为用了 exp2），转回 ln 域：
    LSE = ln(l_i * 2^m_i) = (log2(l_i) + m_i) / log2(e)
    '''
    if WRITE_LSE:
        lse = (tl.math.log2(l_i) + m_i) / log2e
        tl.store(LSE_splitk_ptr, lse, mask=mask)
```

## 3. K/V 加载与反量化

**源码位置**: [splitk_kernels.py#L699-L780](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L699-L780)

`load_dequantize_k_v_group` 统一处理三种 K/V 格式。对于多 group 量化，通过 `tl.advance` 移动 block_ptr 到对应 group 的数据段。

**FP8 反量化**（PACKED_PER_VAL=4）：

```python
'''
1. 加载 scale_shift（int32 = 两个 fp16 打包：scale 低 16 位，shift 高 16 位）
2. 解包：scale = (x & 0xFFFF).to(fp16), shift = (x >> 16).to(fp16)
3. 从 int32 提取 4 个 fp8: x >> [0,8,16,24] 取低 8 位，bitcast 为 fp8
4. 线性反量化: value = fp8_to_float(packed) * scale + shift
'''
v_scale_shift = tl.load(V_scale_shift_block_ptr)
v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)    # 解包 int32 → 2×fp16
v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL, IS_HIP).to(dtype)
```

**INT4 反量化**（PACKED_PER_VAL=8）：

```python
'''
每个 int32 包含 8 个 4-bit 值：
1. offsets = [0, 4, 8, ..., 28]，右移提取各 nibble
2. & 0xF 取低 4 位
3. 技巧：将 4-bit uint 视为 fp16 位模式，乘以 32768×512 = 2^24 得到正确浮点值
   这利用了 IEEE 754 fp16 指数偏移（避免整数→浮点转换指令）
4. dequant = (quant_as_fp16 × 32768) × (scale × 512) + shift
'''
offsets = tl.arange(0, PACKED_PER_VAL) * 4                # [0, 4, 8, ..., 28]
quant_offset = (x_[:, :, None, :] >> offsets)             # 位移提取
quant_offset = (quant_offset & 0xF).to(tl.uint16).to(tl.float16, bitcast=True)
quant_offset = (quant_offset * 32768.0).to(tl.float16)
scale_512 = scale * 512
dequant = quant_offset * scale_512 + shift
```

AMD（HIP）路径与 CUDA 不同：K 的反量化使用 `dequantize_k_hip`，reshape 顺序不同（先 `[D_packed, N]` 展开为 `[D, N]`）；fp8 用 `tl.float8e4b8` 而非 `tl.float8e4nv`；中间计算在 float32（MI300 无 fp8→bf16 直接指令）。

## 4. Split-K 归约

**源码位置**: [splitk_kernels.py#L908-L1020](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L908-L1020)

split_k > 1 时，前向 kernel 写出 split_k 份 partial attention 和 LSE，需要归约合并。

### 4.1 归约数学

每个 chunk $s$ 的 partial attention $\mathbf{o}_s$ 已归一化，对应 LSE 为 $\ell_s$。全局合并：

$$
\mathbf{o} = \frac{\sum_s e^{\ell_s - m^*} \cdot \mathbf{o}_s}{\sum_s e^{\ell_s - m^*}}, \quad m^* = \max_s \ell_s
$$

$$
\ell = m^* + \log \sum_s e^{\ell_s - m^*}
$$

### 4.2 _splitK_reduce Kernel

```python
@triton.jit
def _splitK_reduce(Out_splitK, LSE_splitK, Out, LSE, ...):
    '''
    grid = (M, B*G*H, 1)
    每个 program 处理一个 query position 的所有 split_k 份 partial 结果
    '''
    off_m = tl.program_id(0)
    off_zhg = tl.program_id(1)

    '''
    一次性加载所有 chunk（利用 splitK_pow2 维度向量化）：
    out_splitk: [splitK_pow2, head_dim_pow_2]
    lse_splitk: [splitK_pow2]
    '''
    Out_splitK_ptr = (
        Out_splitK + stride_osk_z * off_z + stride_osk_g * off_g
        + stride_osk_h * off_h + stride_osk_m * off_m
        + tl.arange(0, head_dim_pow_2)[None, :]             # [1, D]
        + stride_osk_s * tl.arange(0, splitK_pow2)[:, None] # [S, 1]
    )
    out_splitk = tl.load(Out_splitK_ptr, mask=mask_2d, other=0)
    lse_splitk = tl.load(LSE_splitK_ptr0, mask=mask_1d, other=float("-inf"))

    '''
    归约：exp2 + 加权平均
    '''
    lse_max = tl.max(lse_splitk)
    sumexp_normalized_splitk = tl.math.exp2(
        (lse_splitk - lse_max).to(tl.float32) * 1.44269504
    )                                                         # [splitK_pow2]
    sumexp_normalized = tl.sum(sumexp_normalized_splitk, axis=0)
    numerator_normalized = tl.sum(
        out_splitk * sumexp_normalized_splitk[:, None], axis=0
    )                                                         # [head_dim_pow_2]
    acc = numerator_normalized / sumexp_normalized
    acc = tl.where(lse_max == float("-inf"), 0.0, acc)

    tl.store(Out_ptr, acc, mask=head_dim_mask)

    if WRITE_LSE:
        to_store = lse_max + tl.math.log2(sumexp_normalized) / 1.44269504
        tl.store(l_ptrs, to_store)
```

所有 chunk 通过一次 batch load 读入，一次 `tl.sum` 完成归约。`splitK_pow2` 是 split_k 向上取 2 的幂（Triton 向量操作要求 2 的幂维度），多余的位置用 mask 填 -inf/0。

### 4.3 Varargs 归约

**源码位置**: [splitk_kernels.py#L1024-L1133](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L1024-L1133)

`_splitK_reduce_varargs` 接受 **列表** 形式的 partial attention/LSE（而非堆叠张量），用于合并来自不同 attention 调用的 partial 结果（如 prefix caching）。因为每个 tensor 可能有不同 stride，使用两遍循环：第一遍找全局 max，第二遍累加。

Triton 不原生支持张量列表参数，通过 `unroll_varargs` AST 变换工具在编译前将 `VAR_ARGS_ARRAY` 标注的参数展开为固定数量的具名参数（如 `Out_splitK_0, Out_splitK_1, ...`）。前向 kernel 的 `q`、`k`、`v`、`acc` 也用了同样机制处理多 group 量化。

`merge_attentions_varargs` 通过 `@torch.library.custom_op` 注册为 PyTorch custom op，兼容 `torch.compile` 和 FakeTensor shape 追踪。

## 5. 启发式参数选择

**源码位置**: [triton_splitk.py#L414-L603](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py#L414-L603)

`get_extra_args` 根据硬件（CUDA/AMD）、batch size、head dim、是否量化等选择 BLOCK_M、BLOCK_N、num_warps、num_stages。AMD 和 CUDA 有完全不同的调参策略：

**CUDA**：
- SM ≥ 8.9（Ada/Hopper）且 head_dim=128：BLOCK_M 根据 M 动态选择（16/32/64/128），Mq>1 时 num_warps=4
- 默认：BLOCK_M=16, BLOCK_N=64, num_warps=2, num_stages=1

**AMD**（HIP）：大量 if-else 分支按 batch size × sequence length × 是否 FP8 × 是否分页组合手工调优，BLOCK_N 在 16-128 间选择，num_warps 在 1-8 间选择。

`BOUNDS_CHECKS_N` 由 heuristics 自动推导：当 split 大小和 block 大小完美对齐且无变长序列时关闭越界检查减少指令开销。

此外支持 Triton autotune 模式（`AUTOTUNE=True`），遍历 BLOCK_M∈{16,32,64,128} × BLOCK_N∈{16,32,64,128} × stages∈{1,2,3} × warps∈{1,2,4,8} 的组合（约束 BLOCK_N≥BLOCK_M）。

## 6. 端到端数据流

```
Input: Q [B, Mq, G, Hq, D], K/V [B, Mk, G, Hkv, D]
           │
           ▼
    ┌─ FwOp.apply ─┐
    │  bias 解析    │  tensor bias / 结构化 bias 分流
    │  GQA 重排     │  head-swapping: (B,Mq,G,Hq,D) → (B,Hq*Mq,G,1,D)
    │  量化检测     │  PACKED_PER_VAL = 1/4/8
    │  split_k 选择 │  get_split_k() 启发式
    └──────┬────────┘
           │
           ▼
    ┌─ _fwd_kernel_splitK ──────────────────────────────┐
    │  grid: (ceil(M/BM), B*G*H, split_k)               │
    │                                                    │
    │  Q 加载 → SRAM（循环外一次加载）                     │
    │                                                    │
    │  for start_n in range(lo, hi, BLOCK_N):            │
    │    ┌ 分页: block_table 查表 → 物理地址              │
    │    ├ K/V 加载 + 反量化（FP8/INT4/原始）             │
    │    ├ qk = Σ_g q[g] @ k[g]    [BM, BN]             │
    │    ├ additive bias / 因果 / 局部窗口 mask           │
    │    └ online softmax: m_i, l_i, acc 更新            │
    │                                                    │
    │  store partial o = acc/l_i, partial LSE            │
    └──────┬─────────────────────────────────────────────┘
           │
           ▼ (split_k > 1)
    ┌─ _splitK_reduce ──────────────────────────────────┐
    │  grid: (M, B*G*H)                                  │
    │  load all S chunks: [S, D]                         │
    │  weight_s = exp(LSE_s - max(LSE))                  │
    │  out = Σ(weight * o) / Σ(weight)                   │
    │  LSE_global = max + ln(Σ weight)                   │
    └──────┬─────────────────────────────────────────────┘
           │
           ▼
    ┌─ 输出重排 ─────────────────────────────────────────┐
    │  mqa_swap 逆变换: (B,G,Hq,Mq,D) → (B,Mq,G,Hq,D)  │
    │  variable_q 合并: (B,Mq,G,Hq,D) → (1,B*Mq,G,Hq,D) │
    └───────────────────────────────────────────────────┘
```
