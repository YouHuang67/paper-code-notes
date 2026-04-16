---
tags:
  - CUDA
  - CUTLASS
  - Flash Attention
---
# Flash Attention V2：反向传播

本文拆解 FA2 反向传播的核心策略与实现。

**源码**: [flash_bwd_kernel.h](src/flash_bwd_kernel_h.md)、[flash_bwd_preprocess_kernel.h](src/flash_bwd_preprocess_kernel_h.md)

## 反向传播的数学

给定前向输出 $O = \text{softmax}(QK^T / \sqrt{d}) \cdot V$ 和损失梯度 $dO$，需要计算 $dQ, dK, dV$。

关键中间量：

$$
S = QK^T / \sqrt{d}, \quad P = \text{softmax}(S), \quad O = PV
$$

$$
dV = P^T \cdot dO, \quad dP = dO \cdot V^T
$$

$$
D_i = \text{rowsum}(dO_i \odot O_i), \quad dS = P \odot (dP - D)
$$

$$
dQ = dS \cdot K / \sqrt{d}, \quad dK = dS^T \cdot Q / \sqrt{d}
$$

## 核心策略：Recomputation

FA2 反向**不存储** attention matrix $P$。前向只保存 $O$ 和 $LSE$（log-sum-exp），反向时**重新计算** $S → P$：

1. 从 $Q, K$ 重新计算 $S = QK^T$
2. 从 $LSE$ 恢复 $P$：$P_{ij} = \exp(S_{ij} \cdot \text{scale} - LSE_i)$

这用 $O(N)$ 额外计算换取 $O(N^2)$ 内存节省。

## 辅助函数

[flash_bwd_kernel.h:L42-L89](src/flash_bwd_kernel_h.md#__codelineno-0-42)

```cpp
/*
 * make_tiled_copy_B_warpcontiguousN:
 * 为反向的 B 操作数 (K/V) 构造 warp 连续的 copy atom
 * 标准 make_tiled_copy_B 可能让 warp 在 N 维不连续
 * 这里手动构造 tile layout 确保 N 维 warp 连续
 */
template<int MMA_N, class... Args, class TiledMMA>
auto make_tiled_copy_B_warpcontiguousN(
    Copy_Atom<Args...> const &copy_atom,
    TiledMMA const &tiled_mma)
{
    constexpr int TileShape_N =
        decltype(tiled_mma.template tile_size_mnk<1>())::value;
    constexpr int TileShape_K =
        decltype(tiled_mma.template tile_size_mnk<2>())::value;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N =
        decltype(size<1>(AtomShape_MNK{}))::value;
    constexpr int kNWarpsN =
        TileShape_N / AtomShape_N / 2;             // N 维 warp 数
    constexpr int MMAStride_N =
        MMA_N * AtomShape_N * 2;                   // N 维步长

    auto t = make_tile(
        Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,
               Stride<_1, Int<MMAStride_N>, _8>>{},
        make_layout(Int<TileShape_K>{}));
    return make_tiled_copy_impl(
        copy_atom, tiled_mma.get_layoutB_TV(), t);
}

// make_tiled_copy_C_warpcontiguousN: 类似, 用于 C 操作数 (P/dS → smem)
```

## Step 1：预处理（compute_dot_do_o）

[flash_bwd_preprocess_kernel.h:L57-L80](src/flash_bwd_preprocess_kernel_h.md#__codelineno-0-57)

计算 $D_i = \text{rowsum}(dO_i \odot O_i)$，写入 `dsoftmax_sum`。

```cpp
/*
 * dot_do_o: 对每行计算 dO · O 的点积
 * 先 reshape dO 和 O 的 MMA 布局
 * 线程内求和 + Allreduce 跨线程同步
 * 结果写到 dsoftmax_sum
 */
__device__ void dot_do_o(
    Tensor do_, Tensor o, Tensor dP_sum, ...)
{
    for (int mi ...) {
        float dP_sum_cur = 0;
        for (int ni ...) {
            dP_sum_cur +=
                do_fp32(mi, ni) * o_fp32(mi, ni);
        }
        dP_sum_cur = Allreduce<THREADS_PER_ROW>::run(
            dP_sum_cur, sum_op) * scale;
        dP_sum(mi * col_stride + tidx / THREADS_PER_ROW)
            = dP_sum_cur;
    }
}
```

如果 `Clear_dQaccum=true`，同时清零 `dq_accum_ptr`（反向累积缓冲）。

## Step 2：主反向 kernel（compute_dq_dk_dv_1colblock）

[flash_bwd_kernel.h:L93-L841](src/flash_bwd_kernel_h.md#__codelineno-0-93)

### 函数签名

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal,
         bool Is_local, bool Has_alibi, bool Is_even_MN,
         bool Is_even_K, bool Is_softcap,
         bool Is_first, bool Is_last,
         bool Seq_parallel=false, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock(
    const Params &params, const int bidb,
    const int bidh, const int n_block)
```

与前向相比多了 `Is_first`/`Is_last`（多 split 控制）和 `Seq_parallel`（序列并行）。

### 初始化与边界

[flash_bwd_kernel.h:L96-L141](src/flash_bwd_kernel_h.md#__codelineno-0-96)

```cpp
    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    // MMA_N_SdP: SdP 计算中 N 维的 MMA 迭代次数
    constexpr int MMA_N_SdP =
        kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}
            .template tile_size_mnk<1>())::value;
    constexpr int AtomLayoutMS =
        Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer =
        !Kernel_traits::No_double_buffer;          // Q/dO 双缓冲

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k)
        return;                                    // 超出 K 序列长度

    /*
     * 与前向的关键区别: 外循环方向翻转
     * 前向: 外循环 Q 行块, 内循环 K/V 块
     * 反向: 外循环 K/V 列块 (每个 TB 固定一个), 内循环 Q/dO 块
     * 因为 dK/dV 需要对所有 Q 行求和, 固定 KV 块可在 reg 中累积
     */
    int m_block_max =
        cute::ceil_div(binfo.actual_seqlen_q, kBlockM);
    if (Is_local) {
        m_block_max = std::min(m_block_max,
            cute::ceil_div(
                (n_block + 1) * kBlockN
                + binfo.actual_seqlen_q
                - binfo.actual_seqlen_k
                + params.window_size_left, kBlockM));
    }
```

### 偏移计算与 Tensor 构建

[flash_bwd_kernel.h:L121-L167](src/flash_bwd_kernel_h.md#__codelineno-0-121)

```cpp
    // 行偏移: 从 m_block_max - 1 开始 (从后向前迭代)
    const index_t row_offset_q =
        binfo.q_offset(...) + (m_block_max - 1) * kBlockM
        * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k =
        binfo.k_offset(...) + n_block * kBlockN    // K 固定
        * params.k_row_stride
        + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    // ... row_offset_v, row_offset_do, row_offset_o,
    //     row_offset_dq, row_offset_dq_accum 类似

    // dq_accum: fp32 累积, deterministic 模式每个 TB 独立分片
    const index_t row_offset_dq_accum = ...
        + (!params.deterministic ? 0
           : blockIdx.x * params.dq_accum_split_stride);
    // LSE 和 D (dsoftmax_sum) 偏移
    const index_t row_offset_lse = ...;
    const index_t row_offset_dpsum = ...;

    // Global Memory Tensor (与前向类似, 但多了 dO, O, dQ, dQaccum)
    Tensor gQ = make_tensor(...,
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(...,                   // K 块 (固定)
        Shape<Int<kBlockN>, Int<kHeadDim>>{}, ...);
    Tensor gV = make_tensor(..., ...);             // V 块 (固定)
    Tensor gdO = make_tensor(..., ...);            // dO 块 (迭代)
    Tensor gO = make_tensor(..., ...);             // O 块 (迭代, 用于 dropout)
    Tensor gdQ = make_tensor(..., ...);            // dQ 输出 (fp16)
    Tensor gdQaccum = make_tensor(...,             // dQ 累积 (fp32)
        make_stride(params.h * params.d_rounded, _1{}));
    Tensor gLSE = make_tensor(...,                 // LSE (1D)
        Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(...,               // D 值 (1D)
        Shape<Int<kBlockM>>{}, Stride<_1>{});
```

### Shared Memory 布局

[flash_bwd_kernel.h:L169-L191](src/flash_bwd_kernel_h.md#__codelineno-0-169)

```cpp
    /*
     * smem 分配 (与 kernel_traits 中 kSmemSize 对应):
     * sQ  (+ double buffer) → sQ 的 2 或 3 份
     * sdO (1 份)
     * sK, sV               → 整个 Q 迭代中不变
     * sdS                  → dS = P ⊙ (dP - D) 中间结果
     * sP / sdQ             → 共享同一片 smem (不同时使用)
     * 每个 tensor 都有对应的转置视图 (零开销 layout 重解释)
     */
    Tensor sQ = make_tensor(
        make_smem_ptr(reinterpret_cast<Element*>(smem_)),
        typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(),
        typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(),
        typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    // 双缓冲: sQ 占 2 或 3 份空间
    Tensor sdO = make_tensor(
        sQ.data() + (Double_buffer ? 2 : 1) * size(sQ),
        typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(),
        typename Kernel_traits::SmemLayoutQdOtransposed{});
    // K, V 紧接 dO 之后
    Tensor sK = make_tensor(
        sdO.data() + size(sdO),
        typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(
        sK.data() + size(sK),
        typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(),
        typename Kernel_traits::SmemLayoutKtransposed{});
    // dS 紧接 V (或 K, 如果 Is_V_in_regs)
    Tensor sdS = make_tensor(
        !Kernel_traits::Is_V_in_regs
            ? sV.data() + size(sV) : sK.data() + size(sK),
        typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(),
        typename Kernel_traits::SmemLayoutPdStransposed{});
    // P 和 dQ 共享 smem
    Tensor sP = make_tensor(
        sdS.data() + size(sdS),
        typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdQ = make_tensor(sP.data(),            // 与 sP 重叠
        typename Kernel_traits::SmemLayoutdQ{});
```

### Copy 与 MMA Partition

[flash_bwd_kernel.h:L192-L301](src/flash_bwd_kernel_h.md#__codelineno-0-192)

```cpp
    // Gmem copy: QKV 共用, dO 根据 Is_first 选不同 copy
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV =
        gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO = std::conditional_t<
        Is_first,
        typename Kernel_traits::GmemTiledCopydO,   // 首次: 普通 copy
        typename Kernel_traits::GmemTiledCopyQKV>;  // 非首次: cp.async
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO =
        gmem_tiled_copy_dO.get_thread_slice(tidx);
    // dQaccum: Seq_parallel 时用 atomic add copy
    using GmemLayoutAtomdQaccum = std::conditional_t<
        !Seq_parallel,
        typename Kernel_traits::GmemTiledCopydQaccum,
        typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd>;

    // Gmem partition: Q, dO, K, V, dQ, dQaccum
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    Tensor tdQgdQaccum =
        gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    /*
     * 三组 TiledMMA + reg fragment
     */
    // TiledMmaSdP: S = Q·K^T, dP = dO·V^T
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    // TiledMmadKV: dK = dS^T·Q, dV = P^T·dO
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdKrdSt =
        thr_mma_dkv.partition_fragment_A(sdStNoSwizzle);
    Tensor tdKrQt =
        thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);
    Tensor tdVrPt =
        thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);
    Tensor tdVrdO =
        thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle);

    // TiledMmadQ: dQ = dS·K
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);
    Tensor tdQrKt =
        thr_mma_dq.partition_fragment_B(sKtNoSwizzle);

    // dK, dV 累积器
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv,
        Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv,
        Shape<Int<kBlockN>, Int<kHeadDim>>{});

    /*
     * Copy Atom Retiling (6 组, 对应三组 MMA 的 A/B 操作数)
     * 反向使用 warpcontiguousN 变体确保 N 维连续
     */
    // SdP 的 A (Q/dO) 和 B (K/V)
    auto smem_tiled_copy_QdO = make_tiled_copy_A(
        typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_tiled_copy_KV =
        make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(
            typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    // SdP 的 C → P/dS 写 smem
    auto smem_tiled_copy_PdS =
        make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(
            typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    // dKV 的 A (P^T/dS^T) 和 B (Q^T/dO^T), 用转置 copy atom
    auto smem_tiled_copy_PdSt = make_tiled_copy_A(
        typename Kernel_traits::SmemCopyAtomTransposed{},
        tiled_mma_dkv);
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(
        typename Kernel_traits::SmemCopyAtomTransposed{},
        tiled_mma_dkv);
    // dQ 的 A (dS) 和 B (K^T)
    auto smem_tiled_copy_dS = make_tiled_copy_A(
        typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_tiled_copy_Kt = make_tiled_copy_B(
        typename Kernel_traits::SmemCopyAtomTransposed{},
        tiled_mma_dq);
    // dQ → smem
    auto smem_tiled_copy_dQ = make_tiled_copy_C(
        typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
```

### 主循环流程

对于固定的 K/V 列块 `n_block`，从后向前迭代所有 Q 块：

```
初始化：加载 K, V → smem（整个循环不变）
clear(acc_dk), clear(acc_dv)

for m_block = m_block_max-1 → m_block_min:
  1. 加载 Q[m_block], dO[m_block] → smem (双缓冲)
  2. 加载 LSE[m_block], D[m_block] → register

  3. GEMM: acc_s = Q · K^T                  (TiledMmaSdP)
  4. Recompute P: P = exp(S * scale - LSE)
  5. 应用 mask、dropout

  6. GEMM: acc_dp = dO · V^T                (TiledMmaSdP)
  7. dS = P ⊙ (dP - D)                      # 逐元素
  8. 写 dS → smem, 写 P → smem

  9. GEMM: acc_dv += P^T · dO               (TiledMmadKV)
  10. GEMM: acc_dk += dS^T · Q              (TiledMmadKV)

  11. GEMM: dQ_block = dS · K               (TiledMmadQ)
  12. dQ_block → smem → global (atomic add)
```

### dQ 的 Atomic Add

$dQ$ 的计算分散在多个 K/V 列块中（每个列块贡献一部分 dQ），需要跨 thread block 累积。FA2 使用两种策略：

- **非确定性模式**: `atomicAdd` 到 `dq_accum_ptr`（fp32 缓冲），最后转回 fp16
- **确定性模式**: 每个 thread block 写入独立的 `dq_accum` 分片（通过 `dq_accum_split_stride` 偏移），最后规约
- **Seq_parallel**: 使用 `GmemTiledCopydQaccumAtomicAdd`（每 thread 1 val/store）

### Double Buffer

Q 和 dO 使用双缓冲（`Double_buffer = !No_double_buffer`）：当前 Q 块在计算时，下一个 Q 块已在异步加载。smem 中分配 3 份 QdO 空间（2 份 Q 双缓冲 + 1 份 dO）。

## 调度入口

[flash_bwd_kernel.h 尾部](src/flash_bwd_kernel_h.md#__codelineno-0-800)

```cpp
template<...>
__global__ void compute_dq_dk_dv(const Params &params) {
    const int n_block = blockIdx.x;                // 外循环: K/V 列块
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;
    compute_dq_dk_dv_1colblock<...,
        Is_first=true, Is_last=true>(
        params, bidb, bidh, n_block);
}
```

Grid 维度：`(ceil_div(seqlen_k, kBlockN), batch, nheads)`，每个 thread block 处理一个 K/V 列块。

## 与前向的对比

| 方面 | 前向 | 反向 |
|------|------|------|
| 外循环 | Q 行块 (blockIdx.x) | K/V 列块 (blockIdx.x) |
| 内循环 | K/V 块 | Q/dO 块 |
| 累积器 | acc_o (输出 O) | acc_dk, acc_dv (梯度) |
| Smem 驻留 | Q (固定) + K/V (迭代) | K/V (固定) + Q/dO (迭代) |
| 额外 GEMM | 无 | dQ = dS · K |
| MMA 组数 | 1 组 | 3 组 |
| Smem 用量 | ~64 KB | ~128-192 KB |
| 跨 block 同步 | 无 | dQ atomic add |
