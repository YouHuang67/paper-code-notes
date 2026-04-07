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

xformers 的 Triton Split-K 后端是面向推理的高效注意力前向内核。与 FlashAttention 相同的数学基础（Online Softmax），但针对 decode 场景做了特化：将 KV 序列切分为多个 chunk 并行计算再归约，并在 kernel 内部融合了因果/局部/分页等 mask 操作以及 INT4/FP8 反量化。

## 1. 数学基础

### 1.1 标准 Attention

给定 Query $\mathbf{q} \in \mathbb{R}^{d}$，Key $\mathbf{K} \in \mathbb{R}^{N \times d}$，Value $\mathbf{V} \in \mathbb{R}^{N \times d}$：

$$
\mathbf{o} = \sum_{j=1}^{N} \frac{\exp(s_j)}{\sum_{i=1}^{N} \exp(s_i)} \mathbf{v}_j, \quad s_j = \frac{\mathbf{q} \cdot \mathbf{k}_j}{\sqrt{d}}
$$

直接计算需要物化 $N$ 个 softmax 概率，内存消耗 $O(N)$。

### 1.2 Online Softmax（FlashAttention）

将 KV 序列分成 $C$ 个 block，每次处理一个 block 并在线更新三个状态：running max $m$、running sum $\ell$、running weighted sum $\mathbf{a}$。

第 $j$ 个 block 的更新：

$$
\begin{aligned}
m^{(j)} &= \max(m^{(j-1)},\; \max_i\; s_i^{(j)}) \\
\alpha^{(j)} &= e^{m^{(j-1)} - m^{(j)}} \\
p_i^{(j)} &= e^{s_i^{(j)} - m^{(j)}} \\
\ell^{(j)} &= \alpha^{(j)} \cdot \ell^{(j-1)} + \sum_i p_i^{(j)} \\
\mathbf{a}^{(j)} &= \alpha^{(j)} \cdot \mathbf{a}^{(j-1)} + \sum_i p_i^{(j)} \mathbf{v}_i
\end{aligned}
$$

最终输出 $\mathbf{o} = \mathbf{a}^{(C)} / \ell^{(C)}$，LSE $= m^{(C)} + \log \ell^{(C)}$。

### 1.3 Split-K 并行化

标准 Online Softmax 是串行遍历 KV 的——decode 阶段只有 1 个 query token，`batch × head` 的并行度可能不够打满 GPU。Split-K 将 KV 序列切成 $S$ 个 chunk，每个 chunk 独立跑 Online Softmax 得到 partial 结果 $(\mathbf{o}_s, \text{LSE}_s)$，然后用第二个 kernel 归约合并：

$$
\mathbf{o} = \frac{\sum_s e^{\ell_s - m^*} \cdot \mathbf{o}_s}{\sum_s e^{\ell_s - m^*}}, \quad m^* = \max_s \ell_s
$$

$$
\text{LSE} = m^* + \log \sum_s e^{\ell_s - m^*}
$$

这样 grid 从 `(M // BM, B*G*H)` 变为 `(M // BM, B*G*H, split_k)`，并行度提升 `split_k` 倍。

## 2. 前向 Kernel

**源码位置**: [splitk_kernels.py#L31-L587](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L31-L587)

下面是前向 kernel 的核心逻辑，省略了量化和分页相关的分支（这两个特性在第 3、4 节单独分析），保留主干以对照上面的数学公式。

```python
@triton.jit
def _fwd_kernel_splitK(Q, K, V, sm_scale, Out_splitK, LSE_splitk, ...):
    '''
    grid = (ceil(M / BLOCK_M), B * G * H, split_k)
    - dim0: Q 序列分块
    - dim1: batch × group × head
    - dim2: Split-K chunk 索引

    编译前由 unroll_varargs 处理：将 VAR_ARGS_ARRAY 标注的变量
    展开为长度 N_GROUPS 的列表，解决 Triton 不支持张量列表的问题。
    非量化场景 N_GROUPS=1，下面的 for i in range(N_GROUPS) 实际只执行一次。
    '''
    start_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = (off_zhg // (H * G)).to(tl.int64)               # batch
    off_h = off_zhg % (H * G) // G                           # head
    off_g = off_zhg % G                                       # group
    splitk_idx = tl.program_id(2)

    if USE_SEQ_LEN:
        kv_len = tl.load(Seq_len + off_z)                    # 变长序列
    else:
        kv_len = N_CTX_K                                      # 定长，编译期常量

    '''
    ========== Split-K chunk 边界 ==========
    当前 chunk 负责 KV 序列的 [lo, hi) 范围
    '''
    lo = splitk_idx * BLOCK_N_PER_SPLIT
    hi = tl.minimum((splitk_idx + 1) * BLOCK_N_PER_SPLIT, kv_len)

    '''
    ========== Q 加载到 SRAM（循环外一次性加载）==========
    block_ptr 定位到 [start_m*BLOCK_M, 0] 处，shape (BLOCK_M, D_PER_GROUP)
    Q 在整个 KV 迭代中保持不变，常驻 SRAM
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

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")    # 对应公式 m^(j)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                   # 对应公式 ℓ^(j)
    acc = tl.zeros([BLOCK_M, D_PER_GROUP], dtype=tl.float32)      # 对应公式 a^(j)

    '''
    预乘 log2(e)：后续全部用 exp2 替代 exp，因为 GPU 上 exp2 更快
    e^x = 2^{x·log2(e)}，所以 qk_scale 吸收了 sm_scale 和 log2(e)
    '''
    log2e = tl.full((), 1.44269504, tl.float32)
    qk_scale = sm_scale * log2e

    for i in range(N_GROUPS):
        q[i] = tl.load(tl.advance(Q_block_ptr, (0, i * D_PER_GROUP)),
                        boundary_check=(0,))

    '''
    ========== 因果/局部 Mask 的对角线偏移（循环外预计算）==========
    推理 decode 场景 query 数量极少（≤16），mask 条件可以用简单的对角线比较：
      q_pos = kv_start + kv_len - num_queries + (q_offset % num_queries)
      kv_pos = start_n + range(0, BLOCK_N)
      因果条件: q_pos >= kv_pos
    简化后:
      diag_idx_shifted = (q_offset % NQ - range(0,BN)) - NQ + kv_len
      因果: diag_idx_shifted >= start_n
      左窗口: diag_idx_shifted < start_n + WINDOW_LEFT + 1
      右窗口: diag_idx_shifted >= start_n - WINDOW_RIGHT
    '''
    if IS_CAUSAL or IS_LOCAL:
        q_offset = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        diag_idx = (q_offset[:, None] % NUM_QUERIES_CAUSAL
                    - tl.arange(0, BLOCK_N)[None, :])
        diag_idx_shifted = diag_idx - NUM_QUERIES_CAUSAL + kv_len

    '''
    ========== 主循环：遍历当前 chunk 内的 KV blocks ==========
    每次迭代处理 BLOCK_N 个 key/value token
    对照公式：每次迭代是一个 "block j" 的 online softmax 更新
    '''
    K_block_ptr = tl.make_block_ptr(...)                      # (D, hi) 起始于 (0, lo)
    V_block_ptr = tl.make_block_ptr(...)                      # (hi, D) 起始于 (lo, 0)

    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(1,))         # [D_PER_GROUP, BLOCK_N]
        v = tl.load(V_block_ptr, boundary_check=(0,))         # [BLOCK_N, D_PER_GROUP]

        '''--- QK^T 计算 ---'''
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for i in range(N_GROUPS):
            qk += tl.dot(q[i], k[i])                          # [BLOCK_M, BLOCK_N]
        qk *= qk_scale                                        # 对应 s_j / sqrt(d) * log2(e)

        '''--- Additive bias ---'''
        if HAS_ADDITIVE_BIAS:
            loaded_bias = tl.load(additive_bias_block_ptr, boundary_check=(0, 1))
            qk += loaded_bias.to(tl.float32) * log2e

        '''--- Mask 应用：masked 位置设为 -inf ---'''
        if BOUNDS_CHECKS_N:
            qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n,
                           qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(diag_idx_shifted >= start_n - start_kv_idx,
                           qk, float("-inf"))
        if IS_LOCAL:
            qk = tl.where(diag_idx_shifted < start_n - start_kv_idx + WINDOW_LEFT + 1,
                           qk, float("-inf"))

        '''
        --- Online Softmax 更新（对照 §1.2 公式）---
        m_i_new  = max(m_i, max(qk))        ← m^(j)
        alpha    = 2^(m_i - m_i_new)         ← α^(j) = e^{m^(j-1) - m^(j)}
        p        = 2^(qk - m_i_new)          ← p_i^(j) = e^{s_i - m^(j)}
        l_i      = l_i * alpha + sum(p)      ← ℓ^(j) = α·ℓ^(j-1) + Σp
        acc      = acc * alpha + p @ v       ← a^(j) = α·a^(j-1) + Σp·v
        '''
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))              # [BLOCK_M]
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])               # [BLOCK_M, BLOCK_N]

        if HAS_ADDITIVE_BIAS or IS_CAUSAL or IS_LOCAL:
            alpha = tl.where(m_i_new == float("-inf"), 0, alpha)
            p = tl.where(m_i_new[:, None] == float("-inf"), 0, p)

        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        p = p.to(Q.dtype.element_ty)                           # cast fp16/bf16 for dot

        for i in range(N_GROUPS):
            acc[i] *= alpha[:, None]
            acc[i] += tl.dot(p, v[i])                          # [BLOCK_M, D_PER_GROUP]

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    '''
    ========== 写回 partial 结果 ==========
    o_s = a^(C) / ℓ^(C)    （已归一化的 partial attention）
    LSE_s = (log2(ℓ) + m) / log2(e)    （从 log2 域转回 ln 域）
    '''
    for i in range(N_GROUPS):
        attn_out = tl.where(l_i[:, None] == 0, 0.0, acc[i] / l_i[:, None])
        tl.store(tl.advance(O_block_ptr, (0, i * D_PER_GROUP)),
                 attn_out.to(Out_splitK.dtype.element_ty), boundary_check=(0,))

    if WRITE_LSE:
        lse = (tl.math.log2(l_i) + m_i) / log2e
        tl.store(LSE_splitk_ptr, lse, mask=mask)
```

## 3. 量化 K/V 的加载与反量化

**源码位置**: [splitk_kernels.py#L699-L904](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L699-L904)

上面主循环中的 `tl.load(K_block_ptr)` 在量化场景下替换为 `load_dequantize_k_v_group`，它统一处理三种格式。量化 K/V 以 int32 存储，每个 int32 打包多个量化值：FP8 时 `PACKED_PER_VAL=4`（4 个 fp8），INT4 时 `PACKED_PER_VAL=8`（8 个 int4）。

```python
@triton.jit
def load_dequantize_k_v_group(
    K_block_ptr, V_block_ptr,
    K_scale_shift_block_ptr, V_scale_shift_block_ptr,
    BOUNDS_CHECKS_N, PACKED_PER_VAL, PACKED_D_PER_GROUP,
    FP8_QUANTIZED, dtype, group_id, IS_HIP,
):
    K_block_ptr = tl.advance(K_block_ptr, (PACKED_D_PER_GROUP * group_id, 0))
    V_block_ptr = tl.advance(V_block_ptr, (0, PACKED_D_PER_GROUP * group_id))
    k = tl.load(K_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ())
    v = tl.load(V_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ())

    if FP8_QUANTIZED:
        '''
        FP8 反量化：
        1. scale_shift 是 int32，低 16 位 = scale(fp16)，高 16 位 = shift(fp16)
        2. 从 int32 提取 4 个 fp8 值: >> [0,8,16,24] 取低 8 位，bitcast 为 fp8
        3. dequant = fp8_to_float(packed) * scale + shift
        '''
        v_scale_shift = tl.load(V_scale_shift_block_ptr,
                                boundary_check=(0,) if BOUNDS_CHECKS_N else ())
        v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
        v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL, IS_HIP).to(dtype)

        k_scale_shift = tl.load(K_scale_shift_block_ptr,
                                boundary_check=(1,) if BOUNDS_CHECKS_N else ())
        k_scale, k_shift = cast_uint32_to_half2(k_scale_shift)
        k_t = dequantize(tl.trans(k), tl.trans(k_scale), tl.trans(k_shift),
                         PACKED_PER_VAL, IS_HIP).to(dtype)
        k = tl.trans(k_t)

    elif PACKED_PER_VAL > 1:
        '''
        INT4 反量化：
        每个 int32 = 8 个 4-bit 值。行布局:
        [quant_coef_0, ..., quant_coef_{G-1} | group0_data... | group1_data...]
        量化系数在行首，数据在后面。
        '''
        K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr, (group_id, 0))
        V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr, (0, group_id))
        k_scale_shift = tl.load(K_scale_shift_block_ptr,
                                boundary_check=(1,) if BOUNDS_CHECKS_N else ())
        v_scale_shift = tl.load(V_scale_shift_block_ptr,
                                boundary_check=(0,) if BOUNDS_CHECKS_N else ())
        k_scale, k_shift = cast_uint32_to_half2(k_scale_shift)
        v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
        v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL, IS_HIP).to(dtype)
        k_t = dequantize(tl.trans(k), tl.trans(k_scale), tl.trans(k_shift),
                         PACKED_PER_VAL, IS_HIP).to(dtype)
        k = tl.trans(k_t)

    return k, v


@triton.jit
def cast_uint32_to_half2(scale_shift):
    scale = scale_shift & 0xFFFF
    shift = scale_shift >> 16
    scale = scale.to(tl.uint16).to(tl.float16, bitcast=True)
    shift = shift.to(tl.uint16).to(tl.float16, bitcast=True)
    return scale, shift


@triton.jit
def dequantize(x_, scale, shift, PACKED_PER_VAL: tl.constexpr, IS_HIP: tl.constexpr):
    '''
    INT4: 每个 int32 包含 8 个 4-bit 值
    提取: x >> [0, 4, 8, ..., 28] & 0xF → 8 个 uint4
    反量化技巧: 将 4-bit uint 直接 bitcast 为 fp16 位模式（低 4 位落在 mantissa），
    乘以 32768(=2^15) 得到 [0, 15] 的浮点值，再乘 scale*512(=2^9) + shift
    总效果: dequant = uint4_value * scale * 2^24 + shift
    避免了整数→浮点转换指令
    '''
    BLOCK_N: tl.constexpr = x_.shape[0]
    BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[1]
    offsets = tl.arange(0, PACKED_PER_VAL) * (32 // PACKED_PER_VAL)
    quant_offset = (x_[:, :, None, :] >> offsets)
    quant_offset = tl.reshape(quant_offset,
                              (BLOCK_N, BLOCK_DMODEL_PACKED * PACKED_PER_VAL))

    if PACKED_PER_VAL == 4:
        fp8_type = tl.float8e4b8 if torch.version.hip is not None else tl.float8e4nv
        dequant = (quant_offset.to(tl.uint8).to(fp8_type, bitcast=True).to(scale.dtype)
                   * scale + shift)
    else:
        quant_offset = ((quant_offset & 0xF).to(tl.uint16)
                        .to(tl.float16, bitcast=True))
        quant_offset = (quant_offset * 32768.0).to(tl.float16)
        scale_512 = scale * 512
        dequant = quant_offset * scale_512 + shift
    return dequant
```

AMD（HIP）路径差异：K 的反量化使用 `dequantize_k_hip`（reshape 顺序 `[D_packed, N]` → `[D, N]` 而非转置）；fp8 用 `tl.float8e4b8`；中间计算在 float32（MI300 无 fp8→bf16 直接指令）。

## 4. 分页注意力（Paged Attention）

分页注意力在主循环内部改变了 K/V 的寻址方式——逻辑连续的 KV 序列物理上分散存储在不同的 page 中，每次迭代需要查 `block_table` 做地址转换。

```python
    '''
    分页模式与非分页模式的核心区别：
    非分页: K_block_ptr 在循环外创建，循环内用 tl.advance 线性移动
    分页:   每次迭代根据 block_table 重建 K_block_ptr

    block_table: [batch_size, max_num_pages]，存储逻辑页→物理页的映射
    K/V shape: [1, max_pages * page_size, num_heads, head_dim]
    '''
    if PAGE_SIZE > 0:
        BLOCKS_IN_PAGE = PAGE_SIZE // BLOCK_N

        '''
        Split-K 边界对齐到 BLOCK_N（避免跨页起始位置不对齐）
        Gappy bias: KV 不从 0 开始，需要跳过 start_kv_idx 之前的位置
        '''
        lo = (tl.maximum(chunk_lo, start_kv_idx) // BLOCK_N) * BLOCK_N
        hi = ((chunk_hi + shift) // BLOCK_N) * BLOCK_N
        hi = tl.minimum(hi, kv_len + start_kv_idx)
        logical_block_idx = lo // BLOCK_N

    for start_n in range(lo, hi, BLOCK_N):
        if PAGE_SIZE > 0:
            '''
            地址转换链：
            logical_block_idx → 页内偏移 + 逻辑页号
            逻辑页号 → 查 block_table → 物理页号
            物理偏移 = 物理页号 × PAGE_SIZE + 页内偏移 × BLOCK_N
            '''
            block_offset_in_page = logical_block_idx % BLOCKS_IN_PAGE
            logical_page_idx = logical_block_idx // BLOCKS_IN_PAGE
            physical_page_idx = tl.load(
                block_table + stride_blocktablesl * logical_page_idx
            ).to(tl.int32)
            offset = physical_page_idx * PAGE_SIZE + block_offset_in_page * BLOCK_N

            K_block_ptr = tl.make_block_ptr(
                base=k_base,
                shape=(PACKED_D_PER_GROUP, offset + current_block_size),
                strides=(stride_kk, stride_kn),
                offsets=(0, offset),
                block_shape=(PACKED_D_PER_GROUP, BLOCK_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=v_base,
                shape=(offset + current_block_size, PACKED_D_PER_GROUP),
                strides=(stride_vn, stride_vk),
                offsets=(offset, 0),
                block_shape=(BLOCK_N, PACKED_D_PER_GROUP),
                order=(1, 0),
            )
            logical_block_idx += 1

        # ... 后续 QK^T + online softmax 与非分页完全相同 ...
```

## 5. Split-K 归约 Kernel

**源码位置**: [splitk_kernels.py#L908-L1020](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/_triton/splitk_kernels.py#L908-L1020)

split_k > 1 时，前向 kernel 输出 `split_k` 份 partial attention $\mathbf{o}_s$ 和 LSE $\ell_s$，需要归约合并（对照 §1.3 的公式）。

```python
@triton.jit
def _splitK_reduce(
    Out_splitK,    # [B, G, H, split_k, M, D]
    LSE_splitK,    # [B, G, H, split_k, M]
    Out,           # [B, M, G, H, D]
    LSE,           # [B, G, H, M]
    split_k, splitK_pow2, ...,
    WRITE_LSE: tl.constexpr,
):
    '''
    grid = (M, B*G*H, 1)
    每个 program 处理一个 query position 的所有 split_k 份 partial 结果

    splitK_pow2 是 split_k 向上取 2 的幂（Triton 向量操作要求 2 的幂维度），
    多余位置用 mask 填 -inf/0
    '''
    off_m = tl.program_id(0).to(tl.int64)
    off_zhg = tl.program_id(1).to(tl.int64)
    off_z = off_zhg // (H * G)
    off_h = (off_zhg // G) % H
    off_g = off_zhg % G

    head_dim_mask = tl.arange(0, head_dim_pow_2) < head_dim

    '''
    一次性 batch load 所有 chunk 的 partial attention 和 LSE：
    指针构造: 行方向 = split_k 索引（步长 stride_osk_s），列方向 = head_dim
    '''
    Out_splitK_ptr = (
        Out_splitK
        + stride_osk_z * off_z + stride_osk_g * off_g
        + stride_osk_h * off_h + stride_osk_m * off_m
        + tl.arange(0, head_dim_pow_2)[None, :]                # [1, D]
        + stride_osk_s * tl.arange(0, splitK_pow2)[:, None]    # [S, 1]
    )
    LSE_splitK_ptr0 = (
        LSE_splitK
        + stride_lsek_z * off_z + stride_lsek_g * off_g
        + stride_lsek_h * off_h + stride_lsek_m * off_m
        + stride_lsek_s * tl.arange(0, splitK_pow2)
    )

    if splitK_pow2 > split_k:
        mask_1d = tl.arange(0, splitK_pow2) < split_k
        mask_2d = mask_1d[:, None] & head_dim_mask[None, :]
        lse_splitk = tl.load(LSE_splitK_ptr0, mask=mask_1d, other=float("-inf"))
        out_splitk = tl.load(Out_splitK_ptr, mask=mask_2d, other=0)
    else:
        lse_splitk = tl.load(LSE_splitK_ptr0)
        out_splitk = tl.load(Out_splitK_ptr)

    '''
    归约核心（对照 §1.3 公式）：
    m* = max(LSE_s)
    w_s = exp(LSE_s - m*) = 2^{(LSE_s - m*) · log2(e)}
    o = Σ(w_s · o_s) / Σ(w_s)
    LSE_global = m* + ln(Σ w_s)
    '''
    lse_max = tl.max(lse_splitk)
    sumexp_normalized_splitk = tl.math.exp2(
        (lse_splitk - lse_max).to(tl.float32) * 1.44269504
    )                                                           # [splitK_pow2]
    sumexp_normalized = tl.sum(sumexp_normalized_splitk, axis=0) # scalar
    numerator_normalized = tl.sum(
        out_splitk * sumexp_normalized_splitk[:, None], axis=0
    )                                                           # [head_dim_pow_2]
    acc = numerator_normalized / sumexp_normalized
    acc = tl.where(lse_max == float("-inf"), 0.0, acc)

    Out_ptr = (Out + stride_oz * off_z + stride_oh * off_h
               + stride_og * off_g + stride_om * off_m
               + tl.arange(0, head_dim_pow_2))
    tl.store(Out_ptr, acc, mask=head_dim_mask)

    if WRITE_LSE:
        l_ptrs = (LSE + off_z * stride_lse_z + off_g * stride_lse_g
                  + off_h * stride_lse_h + off_m * stride_lse_m)
        to_store = lse_max + tl.math.log2(sumexp_normalized) / 1.44269504
        to_store = tl.where(lse_max == float("-inf"), lse_max, to_store)
        tl.store(l_ptrs, to_store)
```

另有 `_splitK_reduce_varargs` 版本接受列表形式的 partial 结果（用于合并不同 attention 调用的输出，如 prefix caching 场景），通过 `unroll_varargs` AST 变换将 `VAR_ARGS_ARRAY` 参数展开为固定数量的具名参数解决 Triton 不支持张量列表的限制。

## 6. FwOp.apply 入口

**源码位置**: [triton_splitk.py#L606-L1012](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py#L606-L1012)

Python 端 `FwOp.apply` 负责将用户输入统一到 kernel 期望的形状，然后 launch kernel。核心预处理：

- **Bias 分流**：`torch.Tensor` 类型的 additive bias 直接传给 kernel；结构化 bias（`BlockDiagonal*Mask`）提取 `seq_len`、`seq_starts_k/q`、因果/局部标记
- **GQA head-swapping**：当 K/V 的 head 维度 stride=0（MQA/GQA 广播），将 $H_q$ 个查询头映射到序列维度 `Q: (B, Mq, G, Hq, D) → (B, Hq×Mq, G, 1, D)`，K/V 只保留 1 个 head，避免展开复制
- **量化检测**：`k.dtype == int32` 时根据是否有 `k_fp8_scale_shift` 区分 FP8/INT4，设置 `PACKED_PER_VAL`
- **split_k 选择**：`get_split_k()` 启发式——prefill（Mq>1, B×G×H>64）返回 1，decode 场景按 `max(Mk,1024) / (B×H)` 计算后 halving 到合理范围
- **block 参数选择**：`get_extra_args()` 按硬件（CUDA/AMD）、batch size、head dim、是否量化选择 BLOCK_M/N、num_warps/stages

kernel launch 后，split_k>1 时调用 `merge_attentions` 执行归约，然后做 head-swapping 的逆变换恢复原始 shape。
