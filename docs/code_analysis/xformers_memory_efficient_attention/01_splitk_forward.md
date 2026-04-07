---
tags:
  - Triton
  - Flash Attention
  - Online Softmax
  - KV Cache
---
# Split-K 前向内核

本节分析 xformers Triton Split-K 前向 kernel 的核心逻辑，包括 FwOp.apply 的张量预处理、kernel 的并行策略、mask 处理和量化融合。

**源码位置**: [triton_splitk.py](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py) / [splitk_kernels.py](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py)

## 1. FwOp.apply 张量预处理

**源码位置**: [triton_splitk.py#L606-L1012](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py#L606-L1012)

kernel launch 前需要将用户输入统一到 kernel 期望的 `(Bqq, Mqq, G, H, K)` 形状。这里 G 是 GQA group 数，H 是每个 group 内的 head 数。

### 1.1 Attention Bias 解析

apply 的第一步是区分 tensor bias 和结构化 bias：

```python
if not isinstance(inp.attn_bias, torch.Tensor):
    attn_bias_tensor = None
    attn_bias = cast(Optional[Union[...]], inp.attn_bias)
else:
    attn_bias_tensor = inp.attn_bias
    attn_bias = None
```

- **tensor bias**（`torch.Tensor`）：任意 additive bias，shape `(B, G, H, Mq, Mkv)` 或 `(B, H, Mq, Mkv)`，在 kernel 内部加到 QK^T 上
- **结构化 bias**（`BlockDiagonal*Mask`）：提供 `seq_len`、`seq_starts_k` 等元数据，kernel 通过条件判断实现 mask

对于结构化 bias，从中提取序列信息：

```python
seq_len = attn_bias.k_seqinfo.seqlen         # [B] 每个 batch 的 KV 长度
seq_starts_k = attn_bias.k_seqinfo.seqstart   # [B+1] KV 起始偏移（gappy bias）
seq_starts_q = attn_bias.q_seqinfo.seqstart    # [B+1] Q 起始偏移（variable q）
IS_CAUSAL = _is_supported_causal_bias(attn_bias)
IS_LOCAL = _is_supported_local_bias(attn_bias)
```

### 1.2 GQA Head-Swapping Trick

当 K/V 通过 `stride=0` 实现 head 广播时（GQA/MQA），启用 head-swapping：

```python
'''
Head-swapping: 将 Hq 个查询头映射到序列维度
Q: (B, Mq, G, Hq, D) → (B, Hq*Mq, G, 1, D)
K/V: 只取第 0 个 head slice → (B, Mkv, G, 1, D)
这样 kernel 只看到 H=1，序列长度变为 Hq*Mq
'''
if k.stride(3) == 0 and v.stride(3) == 0 and attn_bias_tensor is None:
    mqa_swap_seqlen_head = True
    q = q.permute(0, 3, 1, 2, 4).reshape(B, -1, G, 1, Kq)  # [B, Hq*Mq, G, 1, D]
    k = k[:, :, :, :1]                                        # [B, Mkv, G, 1, D]
    v = v[:, :, :, :1]                                        # [B, Mkv, G, 1, D]
```

对于 variable_q（不同 batch 元素 Q 长度不同）的情况，permute 顺序不同——Hq 放在 Mq 之后，保证同一 batch 元素的 query token 在内存中连续。

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

INT4 量化的行布局为 `[quant_coef_0, ..., quant_coef_{G-1} | group0_data... | group1_data...]`，量化系数和数据打包在同一行内。FP8 量化的 scale/shift 存储在独立的 tensor `k_fp8_scale_shift` 中。

### 1.4 输出与 LSE 分配

Split-K 模式下需要中间缓冲区存储每个 chunk 的 partial 结果：

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

当 split_k=1 时，直接写入最终输出，跳过归约步骤。

## 2. _fwd_kernel_splitK 前向 Kernel

**源码位置**: [splitk_kernels.py#L31-L587](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L31-L587)

### 2.1 Grid 与并行策略

```python
def grid(META):
    return triton.cdiv(M, META["BLOCK_M"]), B * G * H, split_k
```

三维 grid：
- **dim0**：Q 序列分块（每个 program 处理 BLOCK_M 个 query token）
- **dim1**：batch × group × head（每个 program 处理一个 head）
- **dim2**：Split-K chunk 索引

每个 kernel instance 负责 `BLOCK_M` 个 query × `BLOCK_N_PER_SPLIT` 个 key 的注意力子块。

### 2.2 Kernel 主体逻辑

```python
@triton.jit
def _fwd_kernel_splitK(Q, K, V, sm_scale, Out_splitK, LSE_splitk, ...):
    '''
    编译前由 unroll_varargs 处理：将 VAR_ARGS_ARRAY 标注的变量
    展开为长度为 N_GROUPS 的列表，解决 Triton 不支持张量列表的问题
    '''
    internal_dtype = tl.float64 if Out_splitK.dtype.element_ty is tl.float64 else tl.float32

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

    if Seq_starts_k is None:
        start_kv_idx = 0
    else:
        start_kv_idx = tl.load(Seq_starts_k + off_z)

    '''
    Split-K chunk 边界计算：
    - lo = splitk_idx * BLOCK_N_PER_SPLIT
    - hi = min((splitk_idx + 1) * BLOCK_N_PER_SPLIT, kv_len)
    分页模式下对齐到 BLOCK_N 边界，避免跨页访问
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
    '''
    Q 加载：block_ptr 定位到 [start_m*BLOCK_M : start_m*BLOCK_M + BLOCK_M, 0 : D_PER_GROUP]
    对于多 group 量化，Q 按 group 分段加载（每段 D_PER_GROUP 维）
    '''
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

    log2e = tl.full((), 1.44269504, tl.float32)
    qk_scale = sm_scale * log2e   # 预乘 log2(e) 使后续用 exp2 替代 exp（更快）

    q = tl.load(tl.advance(Q_block_ptr, (0, i * D_PER_GROUP)), boundary_check=(0,))
```

使用 `exp2` 替代 `exp` 是一个常见的 Triton 性能优化：$e^x = 2^{x \cdot \log_2 e}$，`tl.math.exp2` 在 GPU 上通常比 `tl.exp` 更快。

### 2.4 因果/局部 Mask 的高效实现

因果 mask 和局部窗口 mask 不使用额外的 mask 矩阵，而是通过对角线偏移的条件判断实现：

```python
    '''
    因果 mask 的数学推导：
    假设 num_queries <= BLOCK_M:
      kv_pos = kv_start + range(0, BLOCK_N)
      q_offset = start_m * BLOCK_M + range(0, BLOCK_M)
      q_pos = kv_start + kv_len - num_queries + q_offset % num_queries
      mask = q_pos - kv_pos >= 0

    最终条件简化为:
      diag_idx = q_offset % NUM_QUERIES_CAUSAL - range(0, BLOCK_N)
      diag_idx_shifted = diag_idx - NUM_QUERIES_CAUSAL + kv_len
      mask = diag_idx_shifted >= start_n - start_kv_idx
    '''
    if IS_CAUSAL or IS_LOCAL:
        q_offset = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        diag_idx = (q_offset[:, None] % NUM_QUERIES_CAUSAL) - tl.arange(0, BLOCK_N)[None, :]
        diag_idx_shifted = diag_idx - NUM_QUERIES_CAUSAL + kv_len
```

这里 `NUM_QUERIES_CAUSAL` 是每个 batch 元素的实际 query 数量（而非 BLOCK_M 对齐后的值）。对角线偏移在循环外一次计算，循环内只需与当前 `start_n` 比较。

### 2.5 主循环：K/V 迭代与注意力计算

```python
    for start_n in range(lo, hi, BLOCK_N):
        '''
        分页注意力地址转换：
        logical_block_idx → logical_page_idx → 查 block_table → physical_page_idx
        物理偏移 = physical_page_idx * PAGE_SIZE + block_offset_in_page * BLOCK_N
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
        对于多 group 量化，循环 N_GROUPS 次
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

        '''
        Additive bias：直接加到 qk 上（乘以 log2e 因为后续用 exp2）
        '''
        if HAS_ADDITIVE_BIAS:
            loaded_bias = tl.load(additive_bias_block_ptr, boundary_check=(0, 1))
            qk += loaded_bias.to(tl.float32) * log2e

        '''
        Mask 应用：
        - 边界检查：BOUNDS_CHECKS_N 在最后一个 block 可能越界时启用
        - 因果 mask：diag_idx_shifted >= start_n - start_kv_idx
        - 局部窗口：额外检查 diag_idx_shifted < start_n - start_kv_idx + WINDOW_LEFT + 1
        - 窗口右边界：非因果模式下 diag_idx_shifted >= start_n - start_kv_idx - WINDOW_RIGHT
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
        Online Softmax 更新：
        m_i_new = max(m_i, max(qk))
        alpha = exp2(m_i - m_i_new)          # 旧 max 到新 max 的缩放因子
        p = exp2(qk - m_i_new)               # 当前 block 的 softmax 概率（未归一化）
        l_i = l_i * alpha + sum(p)           # 更新 sum(exp)
        acc = acc * alpha + p @ v            # 更新 weighted sum
        '''
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        if HAS_ADDITIVE_BIAS or IS_CAUSAL or IS_LOCAL:
            alpha = tl.where(m_i_new == float("-inf"), 0, alpha)
            p = tl.where(m_i_new[:, None] == float("-inf"), 0, p)

        l_i = l_i * alpha + tl.sum(p, 1)    # [BLOCK_M]
        m_i = m_i_new
        p = p.to(Q.dtype.element_ty)         # cast 回 fp16/bf16 用于 dot

        for i in range(N_GROUPS):
            acc[i] *= alpha[:, None]          # 缩放历史累积
            acc[i] += tl.dot(p, v[i])         # 累加当前 block 贡献
```

### 2.6 输出写回与 LSE 存储

```python
    '''
    写回 partial attention output：
    attn_out = acc / l_i（如果 l_i=0 则填零，避免 NaN）
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
    LSE = log(sum_exp) + max = log2(l_i) / log2(e) + m_i
    注意这里 m_i 是 log2 域的（因为之前用了 exp2），需要除以 log2(e) 转回 ln 域
    '''
    if WRITE_LSE:
        lse = (tl.math.log2(l_i) + m_i) / log2e   # ln(sum * 2^m) = ln(l_i * 2^m_i)
        tl.store(LSE_splitk_ptr, lse, mask=mask)
```

每个 Split-K chunk 输出的 partial attention 已经做了 softmax 归一化（除以了 l_i），对应的 LSE 记录了该 chunk 内的 log-sum-exp 值。后续归约 kernel 使用 LSE 将各 chunk 的结果正确加权合并。

## 3. K/V 加载与反量化

**源码位置**: [splitk_kernels.py#L699-L780](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L699-L780)

`load_dequantize_k_v_group` 统一处理三种 K/V 格式：

### 3.1 非量化路径（PACKED_PER_VAL=1）

直接 `tl.load` 返回原始 fp16/bf16 数据。

### 3.2 FP8 量化路径（PACKED_PER_VAL=4）

```python
'''
FP8 反量化：
1. 加载 scale_shift（int32 打包两个 fp16: scale 在低 16 位，shift 在高 16 位）
2. 解包：scale = (x & 0xFFFF).to(fp16), shift = (x >> 16).to(fp16)
3. 反量化：value = fp8_to_float(packed) * scale + shift
'''
if FP8_QUANTIZED:
    v_scale_shift = tl.load(V_scale_shift_block_ptr)
    v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
    v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL, IS_HIP).to(dtype)
```

`dequantize` 函数通过位移和掩码从 int32 中提取 4 个 fp8 值，再用 `bitcast` 转为浮点后线性反量化。

### 3.3 INT4 量化路径（PACKED_PER_VAL=8）

```python
'''
INT4 反量化：
1. 每个 int32 包含 8 个 4-bit 值
2. 通过 offsets = [0, 4, 8, 12, 16, 20, 24, 28] 右移提取
3. & 0xF 取低 4 位
4. 技巧：将 4-bit int 视为 fp16 位模式，乘以 32768*512 = 2^24 得到正确浮点值
5. dequant = (quant_as_fp16 * 32768) * (scale * 512) + shift
'''
offsets = tl.arange(0, PACKED_PER_VAL) * 4           # [0, 4, 8, ..., 28]
quant_offset = (x_[:, :, None, :] >> offsets)        # 位移提取
quant_offset = (quant_offset & 0xF).to(tl.uint16).to(tl.float16, bitcast=True)
quant_offset = (quant_offset * 32768.0).to(tl.float16)
scale_512 = scale * 512
dequant = quant_offset * scale_512 + shift
```

这个 INT4 → fp16 的 bitcast trick 避免了显式的整数-浮点转换指令，直接利用 IEEE 754 半精度浮点的位表示特性。

## 4. 启发式参数选择

**源码位置**: [triton_splitk.py#L337-L603](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py#L337-L603)

### 4.1 Split-K 值

`get_split_k` 根据 B, H, Mk, Mq 选择 split_k：

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

### 4.2 Block 大小与 Warp 数

`get_extra_args` 根据硬件（CUDA/AMD）、batch size、head dim、是否量化等条件选择 BLOCK_M、BLOCK_N、num_warps、num_stages。AMD 和 CUDA 有完全不同的调参策略，代码中包含了大量针对不同 batch size 和 sequence length 范围的手工调优分支。

关键选择逻辑（CUDA）：

- **SM ≥ 8.9**（Ada/Hopper）且 head_dim=128：BLOCK_M 根据 M 动态选择（16/32/64/128），Mq>1 时 num_warps=4
- **默认**：BLOCK_M=16, BLOCK_N=64, num_warps=2, num_stages=1

此外支持 Triton autotune 模式（`cls.AUTOTUNE=True`），遍历 BLOCK_M∈{16,32,64,128} × BLOCK_N∈{16,32,64,128} × stages∈{1,2,3} × warps∈{1,2,4,8} 的组合（约束 BLOCK_N≥BLOCK_M）。

### 4.3 Heuristics: BOUNDS_CHECKS_N

```python
kernel = triton.heuristics({
    "BOUNDS_CHECKS_N": lambda args: bool(
        (args["BLOCK_N_PER_SPLIT"] % args["BLOCK_N"])     # chunk 不对齐 block
        or (args["BLOCK_N_PER_SPLIT"] > 0
            and args["N_CTX_K"] % args["BLOCK_N_PER_SPLIT"])  # 总长不对齐 chunk
        or args["USE_SEQ_LEN"]                             # 变长序列
    )
})(_fwd_kernel_splitK_unrolled)
```

`BOUNDS_CHECKS_N` 控制是否在最后一个 block 做越界检查。当 split 大小和 block 大小完美对齐且无变长序列时，可跳过检查减少指令开销。
