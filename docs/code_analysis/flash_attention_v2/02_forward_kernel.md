---
tags:
  - CUDA
  - CUTLASS
  - Flash Attention
---
# Flash Attention V2：前向核心

本文详细拆解 FA2 前向的核心函数 `compute_attn_1rowblock`，这是整个代码库最关键的约 500 行代码。

**源码**: [flash_fwd_kernel.h](src/flash_fwd_kernel_h.md)

## 总体流程

每个 thread block 负责计算输出 O 的一个行块（kBlockM 行），通过迭代所有 K/V 块完成：

```
对于 Q 的第 m_block 个块（kBlockM 行）:
  1. 加载 Q[m_block] → shared memory
  2. 从后向前迭代 K/V 块 (n_block = n_block_max-1 → n_block_min):
     a. 加载 K[n_block] → smem (cp.async)
     b. GEMM: acc_s = Q · K^T                    ← Tensor Core
     c. 应用 mask (causal / local)
     d. Online softmax: 更新 max, sum, rescale acc_o
     e. 加载 V[n_block] → smem (cp.async)
     f. GEMM: acc_o += softmax(S) · V             ← Tensor Core
  3. Normalize: acc_o /= sum, 写出 O 和 LSE
```

## 辅助函数：get_lse_tile

[flash_fwd_kernel.h:L43-L61](src/flash_fwd_kernel_h.md#__codelineno-0-43)

```cpp
/*
 * 获取 LSE 的 gmem tile, 支持三种布局:
 * - 标准: (b, h, seqlen_q)
 * - varlen + ngroups_swapped: (h, seqlen_q, b)
 * - varlen: (h, b, seqlen_q)
 */
template<typename ElementAccum, typename Params,
         int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_lse_tile(
    const Params &params,
    const int bidb, const int bidh,                // batch/head 索引
    const int m_block,                             // Q 行块索引
    const BlockInfo<!Is_even_MN> &binfo)
{
    // varlen 模式: 各 batch 序列长度不等, 需按 offset 寻址
    const bool varlen_q =
        params.unpadded_lse                        // 使用紧凑 LSE 布局
        && !params.seqlenq_ngroups_swapped;        // 且未做 ngroups 交换
    auto lse_offset = varlen_q
        ? binfo.q_offset(params.seqlen_q, 1, bidb) // varlen: 按实际偏移
        : 0;                                        // padded: 无需偏移
    auto gmem_ptr_lse = make_gmem_ptr(
        reinterpret_cast<ElementAccum*>(           // LSE 为 fp32
            params.softmax_lse_ptr) + lse_offset);

    // shape: (batch, head, seqlen_q) 或 varlen 等价形式
    auto lse_shape = varlen_q
        ? make_shape(1, params.h, params.total_q)  // varlen: batch=1, 用 total_q
        : make_shape(params.b, params.h,
                     params.seqlen_q);             // padded: 标准 3D
    // stride: 决定 (b,h,s) 如何映射到线性地址
    auto lse_stride = params.seqlenq_ngroups_swapped
        ? make_stride(1,                           // b stride=1
                      params.seqlen_q * params.b,  // h stride
                      params.b)                    // s stride → (h,s,b) 布局
        : (params.unpadded_lse
           ? make_stride(params.h * params.total_q,
                         params.total_q, 1)        // varlen: (b,h,s) 连续
           : make_stride(params.h * params.seqlen_q,
                         params.seqlen_q, 1));     // 标准: (b,h,s) 连续

    Tensor mLSE = make_tensor(gmem_ptr_lse,
        make_layout(lse_shape, lse_stride));       // 全局 3D LSE tensor
    auto mLSE_slice = varlen_q
        ? mLSE(0, bidh, _)                         // varlen: batch 维=0
        : mLSE(bidb, bidh, _);                     // 标准: 取 (bidb, bidh) 切片
    return local_tile(mLSE_slice,                  // 按 kBlockM 分块
        Shape<Int<kBlockM>>{}, make_coord(m_block)); // 取第 m_block 块
}
```

## 函数签名与模板参数

[flash_fwd_kernel.h:L64-L65](src/flash_fwd_kernel_h.md#__codelineno-0-64)

```cpp
template<typename Kernel_traits,
         bool Is_dropout,                              // 启用 dropout
         bool Is_causal,                               // causal mask
         bool Is_local,                                // sliding window
         bool Has_alibi,                               // ALiBi 位置编码
         bool Is_even_MN,                              // seqlen 是 block 整数倍
         bool Is_even_K,                               // head_dim 是 kBlockKSmem 整数倍
         bool Is_softcap,                              // score 上限截断
         bool Return_softmax,                          // 返回 P 矩阵 (debug)
         typename Params>
inline __device__ void compute_attn_1rowblock(
    const Params &params,
    const int bidb,                                    // blockIdx.y → batch
    const int bidh,                                    // blockIdx.z → head
    const int m_block)                                 // blockIdx.x → Q 行块
{
    using Element = typename Kernel_traits::Element;    // fp16/bf16
    using ElementAccum = typename Kernel_traits::ElementAccum; // fp32
    using index_t = typename Kernel_traits::index_t;   // int32/int64
```

所有 bool 参数在编译期确定（通过 `static_switch.h` 的宏展开），避免运行时分支。`bidb`/`bidh` 来自 `blockIdx.y`/`blockIdx.z`，`m_block` 来自 `blockIdx.x`。

## 初始化与边界检查

[flash_fwd_kernel.h:L72-L141](src/flash_fwd_kernel_h.md#__codelineno-0-72)

```cpp
    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;    // Q 块行数, 通常 128
    constexpr int kBlockN = Kernel_traits::kBlockN;    // KV 块行数, 通常 64
    constexpr int kHeadDim = Kernel_traits::kHeadDim;  // head 维度
    constexpr int kNWarps = Kernel_traits::kNWarps;    // warp 数, 通常 4

    /*
     * Dropout 初始化: Philox RNG + 保存 seed/offset 供反向复现
     * 仅 block(0,0,0) 的 thread 0 写 rng_state
     */
    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    Dropout dropout(std::get<0>(seed_offset),
                    std::get<1>(seed_offset),
                    params.p_dropout_in_uint8_t,
                    bidb, bidh, tidx, params.h);
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0
        && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);  // seed
        params.rng_state[1] = std::get<1>(seed_offset);  // offset
    }

    /*
     * 边界计算: 确定需要迭代的 K/V 块范围 [n_block_min, n_block_max)
     * - n_block_max: 不超过 seqlen_k 的最后一个块
     * - causal: 进一步限制到 m_block 对角线以内
     * - local: 同时考虑左右窗口边界
     */
    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q)
        return;                                        // 超出序列长度

    const int n_block_min = !Is_local ? 0
        : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k
                        - binfo.actual_seqlen_q
                        - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
            cute::ceil_div((m_block + 1) * kBlockM
                           + binfo.actual_seqlen_k
                           - binfo.actual_seqlen_q
                           + params.window_size_right, kBlockN));
    }

    /*
     * 提前退出: n_block_max <= n_block_min 时该行块不 attend 任何 KV
     * 写 0 到 gO, 写 INFINITY 到 gLSE, 然后返回
     */
    if ((Is_causal || Is_local || !Is_even_MN)
        && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                + binfo.q_offset(params.o_batch_stride,
                                 params.o_row_stride, bidb)),
            make_shape(binfo.actual_seqlen_q, params.h, params.d),
            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _),
            Shape<Int<kBlockM>, Int<kHeadDim>>{},
            make_coord(m_block, 0));
        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM,
                                    Is_even_MN>(
            params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O =
            gmem_tiled_copy_O.get_thread_slice(tidx);  // 按线程分工
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO); // gmem 目标分区
        Tensor tOrO = make_tensor<Element>(shape(tOgO));// reg 暂存
        clear(tOrO);                                   // 填 0

        /*
         * Identity tensor + predicate: 边界检查
         * cO 映射 (blk_m, blk_k) → 逻辑坐标, 用于判断是否越界
         * tOpO: k 维谓词, head_dim 不对齐时启用
         */
        Tensor cO = make_identity_tensor(
            make_shape(size<0>(gO), size<1>(gO)));     // 坐标映射 tensor
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO); // 按线程分区
        Tensor tOpO = make_tensor<bool>(
            make_shape(size<2>(tOgO)));                // k 维谓词
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(0, 0, k))
                    < params.d;                        // 列 < head_dim ?
            }
        }
        // 写 0 到 gmem, 带 M/K 维边界检查
        copy<Is_even_MN, Is_even_K, false, false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
            binfo.actual_seqlen_q - m_block * kBlockM);// M 维有效行数

        // LSE 写 INFINITY (表示该行无效, softmax 未定义)
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));     // 当前线程对应的行号
            if (row < binfo.actual_seqlen_q - m_block * kBlockM
                && get<1>(tOcO(0, m, 0)) == 0) {       // 仅列=0 的线程写
                gLSE(row) = INFINITY;
            }
        }
        return;
    }
```

## Tensor 构建：Global → Shared → Register

[flash_fwd_kernel.h:L148-L200](src/flash_fwd_kernel_h.md#__codelineno-0-148)

这一段是理解 FA2 如何使用 CuTe 的关键。参见 [CuTe Tensor](../cute/03_tensor.md)。

```cpp
    /*
     * Global Memory Tensor 构建
     * make_tensor: 从原始指针 + shape + stride 构建逻辑 tensor
     * local_tile: 按 block 大小切分, 取第 m_block/n_block 个 tile
     * (维度变换机制详见 CuTe Tensor 教程 local_tile 章节)
     */
    const index_t row_offset_p =
        ((bidb * params.h + bidh) * params.seqlen_q_rounded
         + m_block * kBlockM) * params.seqlen_k_rounded
        + (n_block_max - 1) * kBlockN;            // P 矩阵 debug 偏移

    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
            + binfo.q_offset(params.q_batch_stride,
                             params.q_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.d),
        make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_coord(m_block, 0));                   // (kBlockM, kHeadDim)

    Tensor mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
            + binfo.k_offset(params.k_batch_stride,
                             params.k_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
        make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(
        mK(_, bidh / params.h_h_k_ratio, _),      // GQA: 多 Q head 共享 KV
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_coord(_, 0));                         // (kBlockN, kHeadDim, nblocksN)
                                                   // 保留第 3 维用于迭代

    Tensor mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
            + binfo.k_offset(params.v_batch_stride,
                             params.v_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
        make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(
        mV(_, bidh / params.h_h_k_ratio, _),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_coord(_, 0));                         // (kBlockN, kHeadDim, nblocksN)

    // P 矩阵: debug/Return_softmax 用
    Tensor gP = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.p_ptr)
            + row_offset_p),
        Shape<Int<kBlockM>, Int<kBlockN>>{},
        make_stride(params.seqlen_k_rounded, _1{}));

    /*
     * Shared Memory Tensor 构建
     * sQ, sK, sV 在 smem 中按 SmemLayout 排列
     * Share_Q_K_smem: Q 和 K 共享同一片 smem (节省空间)
     * sVt/sVtNoSwizzle: V 的转置视图 (零开销, 仅 layout 变换)
     *   .data() 返回带 swizzle 的 smem 指针, .data().get() 剥离 swizzle
     *   MMA retile 需要无 swizzle 版本以匹配 MMA atom 的数据布局
     */
    Tensor sQ = make_tensor(
        make_smem_ptr(reinterpret_cast<Element*>(smem_)),
        typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(
        sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
        typename Kernel_traits::SmemLayoutKV{});   // 共享时与 sQ 重叠
    Tensor sV = make_tensor(
        sK.data() + size(sK),
        typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(),             // V^T 视图, 同一 smem 指针
        typename Kernel_traits::SmemLayoutVtransposed{});  // 转置 layout (含 swizzle)
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), // .get() 剥离 swizzle 得到裸指针
        typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{}); // 转置 layout (无 swizzle, 供 MMA retile)

    /*
     * Copy 的 partition: 按线程切分 global↔shared 的搬运任务
     */
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV =
        gmem_tiled_copy_QKV.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);  // global Q 每线程视图
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);  // shared Q 每线程视图
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    /*
     * MMA 的 partition: 按 MMA 线程切分寄存器 fragment
     */
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);    // Q 寄存器 fragment
    Tensor tSrK = thr_mma.partition_fragment_B(sK);    // K 寄存器 fragment
    Tensor tOrVt = thr_mma.partition_fragment_B(
        sVtNoSwizzle);                                 // V^T 寄存器 fragment
    Tensor tSgS = thr_mma.partition_C(gP);             // P 矩阵 (debug)

    Tensor acc_o = partition_fragment_C(tiled_mma,
        Shape<Int<kBlockM>, Int<kHeadDim>>{});         // 输出累积器 (fp32)
```

### Copy Atom Retiling

[flash_fwd_kernel.h:L205-L218](src/flash_fwd_kernel_h.md#__codelineno-0-205)

MMA 和 copy 对数据排布有不同要求。retiling 创建中间视图使 shared memory → register 的拷贝满足 MMA 输入要求。参见 [CuTe GEMM 教程](../cute/06_gemm_tutorial.md)。

```cpp
    // make_tiled_copy_A/B: 将 SmemCopyAtom (ldmatrix) 与 TiledMMA 对齐
    auto smem_tiled_copy_Q = make_tiled_copy_A(
        typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q =
        smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);    // smem→reg 源视图

    auto smem_tiled_copy_K = make_tiled_copy_B(
        typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K =
        smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    // V 用转置 copy atom (ldmatrix.trans)
    auto smem_tiled_copy_V = make_tiled_copy_B(
        typename Kernel_traits::SmemCopyAtomTransposed{},
        tiled_mma);
    auto smem_thr_copy_V =
        smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
```

### Predicates（边界谓词）

[flash_fwd_kernel.h:L228-L258](src/flash_fwd_kernel_h.md#__codelineno-0-228)

```cpp
    /*
     * Identity tensor: 将 (blk_m, blk_k) 逻辑坐标映射到物理位置
     * 用于判断 copy 中每个元素是否在有效范围内
     */
    Tensor cQ = make_identity_tensor(
        make_shape(size<0>(sQ), size<1>(sQ)));     // (BLK_M, BLK_K) → 坐标
    Tensor cKV = make_identity_tensor(
        make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);  // 按 copy 线程切分
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    // k 维谓词: head_dim 不是 kBlockKSmem 整数倍时需要
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) {
            tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
        }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
        }
    }
```

## Prologue：Q 和首个 K 块加载

[flash_fwd_kernel.h:L262-L296](src/flash_fwd_kernel_h.md#__codelineno-0-262)

```cpp
    // 1. 异步加载 Q → shared memory
    copy<Is_even_MN, Is_even_K>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
        binfo.actual_seqlen_q - m_block * kBlockM);    // 边界行数
    if (Kernel_traits::Is_Q_in_regs) {
        cute::cp_async_fence();
    }

    // 2. Share_Q_K_smem: 等 Q 完成 → 拷到寄存器 → 释放 smem 给 K
    if (Kernel_traits::Share_Q_K_smem) {
        cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view =
            smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ,
                   tSrQ_copy_view);                    // smem → register
        __syncthreads();
    }

    // 3. 异步加载首个 K 块 (n_block_max - 1)
    //    从后向前迭代, 最后一个块可能需要边界 mask
    int n_block = n_block_max - 1;
    copy<Is_even_MN, Is_even_K>(
        gmem_tiled_copy_QKV,
        tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
        binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    // 4. Is_Q_in_regs 但不共享 smem: 等 Q → 拷到寄存器
    if (Kernel_traits::Is_Q_in_regs
        && !Kernel_traits::Share_Q_K_smem) {
        cp_async_wait<1>();                            // 保留 1 flight = K
        __syncthreads();
        Tensor tSrQ_copy_view =
            smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ,
                   tSrQ_copy_view);
    }

    clear(acc_o);

    // Softmax 状态 (row_max, row_sum) 和 Mask 对象
    Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi
        || params.alibi_slopes_ptr == nullptr ? 0.0f
        : reinterpret_cast<float*>(params.alibi_slopes_ptr)
            [bidb * params.alibi_slopes_batch_stride + bidh]
          / params.scale_softmax;
    Mask<Is_causal, Is_local, Has_alibi> mask(
        binfo.actual_seqlen_k, binfo.actual_seqlen_q,
        params.window_size_left, params.window_size_right,
        alibi_slope);
```

`cp.async` 异步拷贝与 `cp_async_fence`/`cp_async_wait` 的配合实现了 global → shared memory 的流水线。参见 [sgemm_sm80](../cute/09_sgemm_sm80.md)。

## 主循环：Masking 迭代

[flash_fwd_kernel.h:L311-L388](src/flash_fwd_kernel_h.md#__codelineno-0-311)

循环分为两阶段：需要 masking 的迭代和不需要 masking 的迭代。

```cpp
    /*
     * n_masking_steps: 需要 mask 的迭代次数
     * - 非 causal: 仅 1 次 (最后一个块可能越界)
     * - causal + even_MN: ceil_div(kBlockM, kBlockN) 次 (对角线块)
     * - causal + !even_MN: 多 1 次 (序列末尾可能截断到块中间)
     */
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal)
           ? cute::ceil_div(kBlockM, kBlockN)
           : cute::ceil_div(kBlockM, kBlockN) + 1);

    #pragma unroll
    for (int masking_step = 0;
         masking_step < n_masking_steps;
         ++masking_step, --n_block)
    {
        Tensor acc_s = partition_fragment_C(tiled_mma,
            Shape<Int<kBlockM>, Int<kBlockN>>{});   // S 累积器 (fp32)
        clear(acc_s);

        // 等待上一轮 K 的 cp.async 完成
        cp_async_wait<0>();
        __syncthreads();

        /*
         * 预取 V: 异步加载当前 n_block 的 V → smem
         * 第一次迭代 (masking_step==0) 处理边界, Clear_OOB_MN=true
         */
        if (masking_step > 0) {
            copy<true, Is_even_K>(
                gmem_tiled_copy_QKV,
                tVgV(_, _, _, n_block), tVsV,
                tKVcKV, tKVpKV);
        } else {
            copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV,
                tVgV(_, _, _, n_block), tVsV,
                tKVcKV, tKVpKV,
                binfo.actual_seqlen_k - n_block * kBlockN);
        }
        cute::cp_async_fence();

        /*
         * GEMM 1: acc_s = Q · K^T
         * Q 可能在寄存器 (Is_Q_in_regs) 或 shared memory
         * K 始终在 shared memory
         */
        gemm<Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK,
            tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);

        // Softcap (可选): 将 score 限制在 [-softcap, softcap]
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }

        // Mask: 对超出 causal/local 边界的位置设为 -inf
        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN,
            m_block * kBlockM + (tidx / 32) * 16
                + (tidx % 32) / 4,                 // MMA 线程映射行号
            kNWarps * 16);

        // 等待 V 就绪, 同时预取下一个 K 块
        cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            copy<true, Is_even_K>(
                gmem_tiled_copy_QKV,
                tKgK(_, _, _, n_block - 1), tKsK,
                tKVcKV, tKVpKV);
            cute::cp_async_fence();                // fence 必须在 if 内
        }

        // Online Softmax: 更新 max, sum, rescale acc_o
        masking_step == 0
            ? softmax.template softmax_rescale_o<
                  /*Is_first=*/true,
                  /*Check_inf=*/Is_causal || Is_local>(
                  acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o<
                  /*Is_first=*/false,
                  /*Check_inf=*/Is_causal || Is_local>(
                  acc_s, acc_o, params.scale_softmax_log2);

        // 类型转换: acc_s (fp32) → rP (fp16/bf16)
        Tensor rP = convert_type<Element>(acc_s);
        int block_row_idx =
            m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);

        // Return_softmax: 保存 P 矩阵到 gmem (debug/可视化)
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template apply_dropout<
                /*encode_dropout_in_sign_bit=*/true>(
                rP_drop, block_row_idx, block_col_idx, kNWarps);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN); // 指针后退
        }

        // Dropout (可选)
        if (Is_dropout) {
            dropout.apply_dropout(
                rP, block_row_idx, block_col_idx, kNWarps);
        }

        /*
         * GEMM 2: acc_o += P · V
         * rP reshape: (MMA=4, MMA_M, MMA_N) → ((4,2), MMA_M, MMA_N/2)
         * 适配 m16n8k16 的 A 操作数 layout
         */
        Tensor tOrP = make_tensor(rP.data(),
            convert_layout_acc_Aregs<
                typename Kernel_traits::TiledMma>(rP.layout()));
        gemm_rs(acc_o, tOrP, tOrVt, tOsVt,
                tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        // 提前退出检查 (n_block 已到达 n_block_min)
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }
```

### 流水线调度

每轮迭代中，K 和 V 的加载与计算交替进行：

```
迭代 i:
  [compute] 等待 K[i] 就绪 → Q·K[i]^T → mask → softmax
  [memory]  异步加载 V[i]
  [compute] 等待 V[i] 就绪 → P·V[i]
  [memory]  异步加载 K[i-1]（下一轮用）
```

## 主循环：无 Masking 迭代

[flash_fwd_kernel.h:L391-L442](src/flash_fwd_kernel_h.md#__codelineno-0-391)

结构与 masking 迭代完全相同，关键区别：

```cpp
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma,
            Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        cp_async_wait<0>();
        __syncthreads();

        // V 总是完整块, 不需要 Clear_OOB_MN
        copy<true, Is_even_K>(
            gmem_tiled_copy_QKV,
            tVgV(_, _, _, n_block), tVsV,
            tKVcKV, tKVpKV);
        cute::cp_async_fence();

        // GEMM 1: Q · K^T
        gemm<Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK,
            tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }

        cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            copy<true, Is_even_K>(
                gmem_tiled_copy_QKV,
                tKgK(_, _, _, n_block - 1), tKsK,
                tKVcKV, tKVpKV);
            cute::cp_async_fence();
        }

        // Mask: Causal_mask=false, 仅 local 窗口边界
        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN,
            m_block * kBlockM + (tidx / 32) * 16
                + (tidx % 32) / 4,
            kNWarps * 16);

        // Is_first=false (softmax 已初始化)
        softmax.template softmax_rescale_o<false,
            /*Check_inf=*/Is_local>(
            acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = convert_type<Element>(acc_s);
        int block_row_idx =
            m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template apply_dropout<true>(
                rP_drop, block_row_idx, block_col_idx, kNWarps);
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.apply_dropout(
                rP, block_row_idx, block_col_idx, kNWarps);
        }

        // GEMM 2: acc_o += P · V
        Tensor tOrP = make_tensor(rP.data(),
            convert_layout_acc_Aregs<
                typename Kernel_traits::TiledMma>(rP.layout()));
        gemm_rs(acc_o, tOrP, tOrVt, tOsVt,
                tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }
```

## Epilogue：归一化与输出

[flash_fwd_kernel.h:L444-L506](src/flash_fwd_kernel_h.md#__codelineno-0-444)

```cpp
    // 1. 最终归一化: acc_o /= softmax_sum, 计算 LSE
    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(
        acc_o, params.scale_softmax, params.rp_dropout);

    // 2. 类型转换: acc_o (fp32) → rO (fp16/bf16)
    Tensor rO = convert_type<Element>(acc_o);

    /*
     * 3. Register → Shared Memory → Global Memory (两跳写出)
     *    MMA 输出的 register 排布分布在不同线程, 不能直接连续写 gmem
     *    先写入 smem 重新排列, 再统一写出
     */
    Tensor sO = make_tensor(sQ.data(),
        typename Kernel_traits::SmemLayoutO{});    // 复用 Q 的 smem
    auto smem_tiled_copy_O = make_tiled_copy_C(
        typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O =
        smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);    // retile 适配
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);  // reg → smem

    // 构建 O 的 gmem tensor 和 copy
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
            + binfo.q_offset(params.o_batch_stride,
                             params.o_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_coord(m_block, 0));
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM,
                                Is_even_MN>(
        params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O =
        gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    // smem → register (重排布局)
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    /*
     * 行索引提取: 用于 LSE 写出时判断哪个线程负责哪一行
     * identity tensor → MMA partition → logical_divide 取行
     */
    Tensor caccO = make_identity_tensor(
        Shape<Int<kBlockM>, Int<kHeadDim>>{});          // 坐标 tensor
    Tensor taccOcO = thr_mma.partition_C(caccO);        // 按 MMA 线程分区
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // 取行索引: ((2,2), MMA_M, MMA_K) → 只取行
    Tensor taccOcO_row =
        logical_divide(taccOcO, Shape<_2>{})(           // 拆分 (2,2) → ((2),(2))
            make_coord(0, _), _, 0);                    // 取第一个 2, 保留 MMA_M

    Tensor cO = make_identity_tensor(
        make_shape(size<0>(sO), size<1>(sO)));          // O 的坐标映射
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);      // 按 copy 线程分区
    Tensor tOpO = make_tensor<bool>(
        make_shape(size<2>(tOgO)));                     // k 维谓词
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(0, 0, k))
                < params.d;                             // 列 < head_dim ?
        }
    }

    // register → global (带边界检查)
    copy<Is_even_MN, Is_even_K, false, false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
        binfo.actual_seqlen_q - m_block * kBlockM);     // M 维有效行数

    // 4. 写出 LSE: 每行一个标量, 仅由该行列=0 的线程写
    if (get<1>(taccOcO_row(0)) == 0) {                  // 仅列=0 线程
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));    // 当前线程的行号
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
                gLSE(row) = lse(mi);                    // 写 LSE 标量
            }
        }
    }
}
```

## Split-KV 变体

[flash_fwd_kernel.h:L498-L1071](src/flash_fwd_kernel_h.md#__codelineno-0-498)

`compute_attn_1rowblock_splitkv` 是 split-KV 版本，用于推理时的长序列。核心区别：

- 多个 thread block 分担同一行 Q 的不同 K/V 范围
- 每个 split 输出 partial O 和 partial LSE 到 `oaccum` / `lseaccum`
- 由 `combine_attn_seqk_parallel`（L1110）合并所有 split 的结果

Split-KV 还支持 **Paged KV Cache**：通过 `block_table` 间接寻址物理页，以及 **Append KV**：在推理时将新 token 的 K/V append 到 cache 并可选应用 RoPE。
