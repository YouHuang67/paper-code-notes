---
tags:
  - CUDA
  - CUTLASS
  - Flash Attention
---
# Flash Attention V2：计算原语

本文拆解 FA2 的四个计算原语文件：online softmax、mask、warp reduce 工具函数、dropout。

## softmax.h：Online Softmax

**源码**: [softmax.h](src/softmax_h.md)

### 核心思想

标准 softmax 需要两遍扫描（第一遍求 max，第二遍求 exp 和 sum）。FA2 的 online softmax 在每个 K 块迭代中增量更新 `row_max` 和 `row_sum`，只需单遍扫描。

对应公式（参考 [FlashAttention 论文](https://arxiv.org/abs/2205.14135)）：

$$
m_i^{new} = \max(m_i^{old}, \max_j(S_{ij})), \quad
l_i^{new} = e^{m_i^{old} - m_i^{new}} \cdot l_i^{old} + \sum_j e^{S_{ij} - m_i^{new}}
$$

$$
O_i^{new} = e^{m_i^{old} - m_i^{new}} \cdot O_i^{old} + e^{S_{ij} - m_i^{new}} \cdot V_j
$$

### Reduce 辅助函数

[softmax.h:L36-L76](src/softmax_h.md#__codelineno-0-36)

```cpp
/*
 * thread_reduce_: 每个线程对自己持有的 2D tensor 按行求 max/sum
 * zero_init=true: 从 tensor 首列初始化; false: 与已有 summary 合并
 */
template<bool zero_init=true, typename Engine0, typename Layout0,
         typename Engine1, typename Layout1, typename Operator>
__device__ void thread_reduce_(
    Tensor<Engine0, Layout0> const &tensor,        // (nrow, ncol)
    Tensor<Engine1, Layout1> &summary,             // (nrow,)
    Operator &op)
{
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init
            ? tensor(mi, 0)
            : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

/*
 * quad_allreduce_: 4 线程间通过 shfl_xor 交换数据
 * m16n8k16 MMA 中同一行的数据分布在 lane_id % 4 的 4 个线程
 * Allreduce<4> → shfl_xor(2) → shfl_xor(1), 两次 butterfly exchange
 */
template<typename Engine0, typename Layout0,
         typename Engine1, typename Layout1, typename Operator>
__device__ void quad_allreduce_(
    Tensor<Engine0, Layout0> &dst,
    Tensor<Engine1, Layout1> &src,
    Operator &op)
{
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

/*
 * reduce_: thread_reduce_ + quad_allreduce_ (完整 reduce)
 * reduce_max: reduce_ 的 max 特化
 * reduce_sum: 仅 thread_reduce_, 不做跨线程 allreduce
 *   (sum 的 allreduce 推迟到最终 normalize 时才做, 减少同步)
 */
template<bool zero_init=true>
__device__ void reduce_max(tensor, max) {
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);       // thread + allreduce
}
template<bool zero_init=true>
__device__ void reduce_sum(tensor, sum) {
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op); // 仅 thread 内
}
```

### scale_apply_exp2：exp2 优化

[softmax.h:L80-L105](src/softmax_h.md#__codelineno-0-80)

```cpp
/*
 * 关键优化: exp2 替代 exp
 * exp(x * scale - max * scale) = exp2(x * scale_log2 - max * scale_log2)
 * exp2 映射到单条 PTX 指令 ex2.approx, exp 需要多条
 * 配合 ffma (fused multiply-add) 进一步提速
 */
template<bool Scale_max=true>
__device__ void scale_apply_exp2(
    Tensor &tensor,                                // (nrow, ncol)
    Tensor const &max,                             // (nrow,)
    const float scale)                             // softmax_scale_log2
{
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // -inf 行特殊处理: 避免 (-inf) - (-inf) = NaN
        const float max_scaled = max(mi) == -INFINITY
            ? 0.f
            : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni) {
            tensor(mi, ni) =
                exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}
```

### Softmax 结构体与 softmax_rescale_o

[softmax.h:L141-L200](src/softmax_h.md#__codelineno-0-141)

```cpp
template <int kNRows>
struct Softmax {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;                      // 每行 max 和 sum

    /*
     * softmax_rescale_o: 每次 K 块迭代调用
     * acc_s: 当前 K 块的 score (MMA=4, MMA_M, MMA_N)
     * acc_o: 累积输出 (MMA=4, MMA_M, MMA_K)
     */
    template<bool Is_first, bool Check_inf=false>
    __device__ void softmax_rescale_o(
        Tensor0 &acc_s, Tensor1 &acc_o,
        float softmax_scale_log2)
    {
        // reshape 为 (nrow, ncol) 方便按行操作
        Tensor scores = make_tensor(acc_s.data(),
            convert_layout_acc_rowcol(acc_s.layout()));

        if (Is_first) {
            // 首次迭代: 直接初始化
            reduce_max<true>(scores, row_max);     // max (thread + allreduce)
            scale_apply_exp2(scores, row_max,
                             softmax_scale_log2);  // scores = exp2(...)
            reduce_sum<true>(scores, row_sum);     // sum (仅 thread 内)
        } else {
            // 非首次: 需要 rescale 之前累积的 acc_o
            Tensor scores_max_prev =
                make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            reduce_max<false>(scores, row_max);    // 更新 max

            Tensor acc_o_rowcol = make_tensor(
                acc_o.data(),
                convert_layout_acc_rowcol(acc_o.layout()));

            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY
                       ? 0.0f : row_max(mi));
                // rescale 因子: exp2(old_max - new_max)
                float scores_scale = exp2f(
                    (scores_max_prev(mi) - scores_max_cur)
                    * softmax_scale_log2);
                row_sum(mi) *= scores_scale;       // rescale 旧 sum
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                    acc_o_rowcol(mi, ni) *= scores_scale;  // rescale 旧 O
                }
            }
            scale_apply_exp2(scores, row_max,
                             softmax_scale_log2);
            reduce_sum<false>(scores, row_sum);    // 累加新 sum
        }
    };

    /*
     * normalize_softmax_lse: 所有 K 块迭代完成后调用
     * 1. 跨线程 allreduce row_sum (之前只做了 thread 内)
     * 2. acc_o /= row_sum
     * 3. LSE = row_max * scale + log(row_sum)
     */
    template<bool Is_dropout=false, bool Split=false>
    __device__ TensorT normalize_softmax_lse(
        Tensor0 &acc_o, float softmax_scale,
        float rp_dropout=1.0)
    {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op); // 4 线程 allreduce
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(
            acc_o.data(),
            convert_layout_acc_rowcol(acc_o.layout()));

        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            // sum==0 或 NaN: 无效行 (全被 mask 掉)
            float inv_sum = (sum == 0.f || sum != sum)
                ? 1.f : 1.f / sum;                    // 1/sum 归一化因子
            lse(mi) = (sum == 0.f || sum != sum)
                ? (Split ? -INFINITY : INFINITY)       // Split: -inf 便于 combine
                : row_max(mi) * softmax_scale
                  + __logf(sum);                       // LSE = max*scale + log(sum)
            float scale = !Is_dropout
                ? inv_sum
                : inv_sum * rp_dropout;                // dropout: 乘以 1/p 补偿
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= scale;
            }
        }
        return lse;
    };
};
```

`kNRows = 2 * MMA_M`，即每个线程负责的行数（m16n8k16 的每个线程持有 2 行）。

## mask.h：Attention Mask

**源码**: [mask.h](src/mask_h.md)

### 独立 mask 函数

[mask.h:L27-L95](src/mask_h.md#__codelineno-0-27)

```cpp
/*
 * apply_mask: 仅列维度越界检查 (col_idx >= max_seqlen_k → -inf)
 * tensor shape: (nrow=(2, MMA_M), ncol=(2, MMA_N))
 * 列索引计算: col_idx_offset + (lane_id % 4) * 2 + nj * 8 + j
 *   lane_id % 4: MMA 线程映射, 每 4 个 lane 持有同一行不同列
 *   nj * 8: MMA_N 的外层步长
 *   j: MMA_N 的内层 (0 或 1)
 */
__device__ void apply_mask(tensor, max_seqlen_k, col_idx_offset_);

/*
 * apply_mask_local: 行列双边界检查
 * 左边界: col < row + seqlen_k - seqlen_q - window_size_left
 * 右边界: col >= row + 1 + seqlen_k - seqlen_q + window_size_right
 * 行索引: row_idx_offset + mi * warp_row_stride + i * 8
 */
template<bool HasWSLeft=true>
__device__ void apply_mask_local(
    tensor, col_idx_offset_, max_seqlen_k,
    row_idx_offset, max_seqlen_q, warp_row_stride,
    window_size_left, window_size_right);

/*
 * apply_mask_causal: local 的特例
 * window_size_left = -1 (无限), window_size_right = 0
 */
__device__ void apply_mask_causal(tensor, col_idx_offset_,
    max_seqlen_k, row_idx_offset, max_seqlen_q,
    warp_row_stride) {
    apply_mask_local<false>(tensor, col_idx_offset_,
        max_seqlen_k, row_idx_offset,
        max_seqlen_q, warp_row_stride, -1, 0);
}
```

### Mask 结构体

[mask.h:L124-L223](src/mask_h.md#__codelineno-0-124)

```cpp
template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {
    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;

    /*
     * apply_mask: 统一入口, 输入 (MMA=4, MMA_M, MMA_N)
     * 编译期分两条路径:
     */
    template<bool Causal_mask=false, bool Is_even_MN=true>
    __device__ void apply_mask(tensor_, col_idx_offset_,
                                row_idx_offset, warp_row_stride)
    {
        static constexpr bool Need_masking =
            Has_alibi || Causal_mask || Is_local || !Is_even_MN;
        if constexpr (!Need_masking) return;

        // reshape (MMA=4, MMA_M, MMA_N) → (nrow=(2,MMA_M), ncol=(2,MMA_N))
        Tensor tensor = make_tensor(tensor_.data(),
            convert_layout_acc_rowcol(tensor_.layout()));

        // 快速路径: 只需列索引 (无 causal/local, 无非 causal alibi)
        static constexpr bool Col_idx_only =
            !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask;

        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset =
            col_idx_offset_ + (lane_id % 4) * 2;

        if constexpr (Col_idx_only) {
            // 仅遍历列: ALiBi 加偏置 + seqlen_k 越界 mask
            for (int nj ...; int j ...) {
                const int col_idx = col_idx_offset + nj * 8 + j;
                for (int mi ...) {
                    if constexpr (Has_alibi)
                        tensor(mi, make_coord(j, nj)) +=
                            alibi_slope * col_idx;
                    if constexpr (!Is_even_MN)
                        if (col_idx >= max_seqlen_k)
                            tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        } else {
            // 完整路径: 需要行索引
            for (int mi ...; int i ...) {
                const int row_idx =
                    row_idx_offset + mi * warp_row_stride + i * 8;
                const int col_idx_limit_left = std::max(0,
                    row_idx + max_seqlen_k - max_seqlen_q
                    - window_size_left);
                const int col_idx_limit_right = std::min(
                    max_seqlen_k,
                    row_idx + 1 + max_seqlen_k - max_seqlen_q
                    + window_size_right);

                for (int nj ...; int j ...) {
                    const int col_idx =
                        col_idx_offset + nj * 8 + j;
                    // ALiBi: causal 加正偏置, 非 causal 减距离
                    if constexpr (Has_alibi) {
                        if constexpr (Is_causal)
                            tensor(...) += alibi_slope * col_idx;
                        else
                            tensor(...) -= alibi_slope
                                * abs(row_idx + max_seqlen_k
                                      - max_seqlen_q - col_idx);
                    }
                    // Causal: 右越界 → -inf
                    if constexpr (Causal_mask)
                        if (col_idx >= col_idx_limit_right)
                            tensor(...) = -INFINITY;
                    // Local: 左右越界 → -inf
                    if constexpr (Is_local)
                        if (col_idx >= col_idx_limit_right
                            || col_idx < col_idx_limit_left)
                            tensor(...) = -INFINITY;
                    // 非 causal/local 的 MN 越界
                    if constexpr (!Causal_mask && !Is_local
                                  && !Is_even_MN)
                        if (col_idx >= max_seqlen_k)
                            tensor(...) = -INFINITY;
                }
            }
        }
    };
};
```

关联：mask 的 sliding window 机制与 [Native Sparse Attention 滑动窗口](../native_sparse_attention/05_sliding_window.md) 概念一致。

## utils.h：工具函数

**源码**: [utils.h](src/utils_h.md)

### Allreduce：Warp 级规约

[utils.h:L124-L144](src/utils_h.md#__codelineno-0-124)

```cpp
template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16
                  || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(
            uint32_t(-1), x, OFFSET));             // butterfly exchange
        return Allreduce<OFFSET>::run(x, op);      // 递归
    }
};
// 递归终止
template<> struct Allreduce<2> {
    static __device__ T run(T x, Operator &op) {
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        return x;
    }
};
// Allreduce<4> → shfl_xor(2) → Allreduce<2> → shfl_xor(1)
```

### MaxOp / SumOp

[utils.h:L104-L120](src/utils_h.md#__codelineno-0-104)

```cpp
template<typename T>
struct MaxOp {
    __device__ T operator()(T const &x, T const &y) {
        return x > y ? x : y;
    }
};
template<> struct MaxOp<float> {
    __device__ float operator()(float const &x, float const &y) {
        return max(x, y);                          // __fmax_rn
    }
};
template<typename T>
struct SumOp {
    __device__ T operator()(T const &x, T const &y) {
        return x + y;
    }
};
```

### gemm / gemm_rs：GEMM 封装

[utils.h:L148-L195](src/utils_h.md#__codelineno-0-148)

```cpp
/*
 * gemm: A 和 B 都从 shared memory 加载 (用于 Q·K^T)
 * K 维迭代: 先 copy 下一个 K slice, 再执行当前 MMA
 * A_in_regs: Q 已在寄存器时跳过 A 的 smem→reg copy
 */
template<bool A_in_regs=false, bool B_in_regs=false>
__device__ void gemm(
    Tensor0 &acc,                                  // 累积器
    Tensor1 &tCrA, Tensor2 &tCrB,                 // reg fragment
    Tensor3 const &tCsA, Tensor4 const &tCsB,     // smem 视图
    TiledMma tiled_mma,
    TiledCopyA smem_tiled_copy_A,
    TiledCopyB smem_tiled_copy_B,
    ThrCopyA smem_thr_copy_A,
    ThrCopyB smem_thr_copy_B)
{
    Tensor tCrA_copy_view =
        smem_thr_copy_A.retile_D(tCrA);           // retile 适配
    Tensor tCrB_copy_view =
        smem_thr_copy_B.retile_D(tCrB);

    // 预取第 0 个 K slice
    if (!A_in_regs)
        cute::copy(smem_tiled_copy_A,
                   tCsA(_, _, _0{}),
                   tCrA_copy_view(_, _, _0{}));
    if (!B_in_regs)
        cute::copy(smem_tiled_copy_B,
                   tCsB(_, _, _0{}),
                   tCrB_copy_view(_, _, _0{}));

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {              // 预取下一个
            if (!A_in_regs) cute::copy(
                smem_tiled_copy_A,
                tCsA(_, _, i+1), tCrA_copy_view(_, _, i+1));
            if (!B_in_regs) cute::copy(
                smem_tiled_copy_B,
                tCsB(_, _, i+1), tCrB_copy_view(_, _, i+1));
        }
        cute::gemm(tiled_mma,
                   tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

/*
 * gemm_rs: A 在寄存器, B 从 shared memory 加载 (用于 P·V)
 * P 已在寄存器 (从 softmax 输出直接使用)
 * 只需要 B (V^T) 的 smem→reg copy
 */
template<typename Tensor0, typename Tensor1, ...>
__device__ void gemm_rs(
    Tensor0 &acc, Tensor1 &tCrA,
    Tensor2 &tCrB, Tensor3 const &tCsB, ...)
{
    Tensor tCrB_copy_view =
        smem_thr_copy_B.retile_D(tCrB);
    cute::copy(smem_tiled_copy_B,
               tCsB(_, _, _0{}),
               tCrB_copy_view(_, _, _0{}));        // 预取第 0 个
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1)
            cute::copy(smem_tiled_copy_B,
                       tCsB(_, _, i+1),
                       tCrB_copy_view(_, _, i+1));
        cute::gemm(tiled_mma,
                   tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}
```

### Layout 转换

[utils.h:L199-L237](src/utils_h.md#__codelineno-0-199)

```cpp
/*
 * convert_layout_acc_rowcol:
 * (MMA=4, MMA_M, MMA_N) → (nrow=(2, MMA_M), ncol=(2, MMA_N))
 * 将 MMA 维度的 4 拆为 2×2 对应行列
 */
__device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    auto l = logical_divide(acc_layout, Shape<_2>{});
    // l: ((2, 2), MMA_M, MMA_N)
    return make_layout(
        make_layout(get<0, 1>(l), get<1>(l)),      // (2, MMA_M) → nrow
        make_layout(get<0, 0>(l), get<2>(l)));     // (2, MMA_N) → ncol
};

/*
 * convert_layout_acc_Aregs:
 * C 累积器格式 → MMA A 操作数格式, 用于 P·V GEMM
 * m16n8k16: (4, MMA_M, MMA_N) → ((4,2), MMA_M, MMA_N/2)
 * m16n8k8:  不变
 */
template<typename MMA_traits>
__device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    constexpr int mma_shape_K =
        get<2>(typename MMA_traits::Shape_MNK{});
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout,
            Shape<Underscore, Underscore, _2>{});
        return make_layout(
            make_layout(get<0>(l), get<2, 0>(l)),  // (4, 2) → 8
            get<1>(l),                             // MMA_M
            get<2, 1>(l));                         // MMA_N/2
    }
};

/*
 * convert_layout_acc_dropout:
 * 同 convert_layout_acc_Aregs (k16), 用于 dropout reshape
 * (4, MMA_M, MMA_N) → ((4,2), MMA_M, MMA_N/2) = (8, MMA_M, MMA_N/2)
 */

/*
 * convert_type: fp32 ↔ fp16/bf16 类型转换
 * 使用 cutlass::NumericArrayConverter 批量转换
 */
template<typename To_type>
__device__ auto convert_type(Tensor const &tensor) {
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel>
        convert_op;
    auto frag = convert_op(
        *reinterpret_cast<const cutlass::Array<From_type, numel>*>(
            tensor.data()));
    return make_tensor(
        make_rmem_ptr<To_type>(&frag), tensor.layout());
}
```

### copy 函数（带边界检查）

[utils.h 后半部分](src/utils_h.md#__codelineno-0-270)

带边界检查的 copy 封装，通过 identity tensor（`cQ`/`cKV`）判断每个元素是否越界：

- `Is_even_MN=true`：序列长度恰好对齐 block 大小，无需边界检查
- `Is_even_K=true`：head dim 恰好对齐，无需 K 维边界检查
- `Clear_OOB_MN=true`：越界元素写 0（用于 V 加载，避免脏数据参与计算）

## dropout.h：Dropout

**源码**: [dropout.h](src/dropout_h.md)

[dropout.h:L26-L106](src/dropout_h.md#__codelineno-0-26)

```cpp
struct Dropout {
    const unsigned long long seed, offset;
    const uint8_t p_dropout_in_uint8_t;

    /*
     * 构造: offset 编码 (batch, head, lane_id)
     * 确保不同 batch/head/线程 使用不同 RNG 子序列
     */
    __device__ Dropout(
        unsigned long long seed,
        unsigned long long offset,
        uint8_t p_dropout_in_uint8_t,
        int bid, int hid, int tid, int nheads)
        : seed(seed)
        , offset(offset + (bid * nheads + hid) * 32
                 + tid % 32)                       // 编码位置
        , p_dropout_in_uint8_t(p_dropout_in_uint8_t) {}

    /*
     * apply_dropout: 对 attention 概率矩阵应用 dropout
     * 1. reshape (4, MMA_M, MMA_N) → (8, MMA_M, MMA_N/2)
     * 2. 用 (block_row, block_col) 作为 Philox 坐标生成随机数
     * 3. 每次 Philox 调用生成 uint4 = 16 个 uint8 随机数
     * 4. 与 p_dropout 比较:
     *    - 标准模式: rnd <= threshold → 保留, 否则 → 0
     *    - sign_bit 模式: 不保留时取负值 (Return_softmax 用)
     */
    template<bool encode_dropout_in_sign_bit=false>
    __device__ void apply_dropout(
        Tensor &tensor_,
        int block_row_start,
        int block_col_start,
        int block_row_stride)
    {
        Tensor tensor = make_tensor(tensor_.data(),
            convert_layout_acc_dropout(tensor_.layout()));

        // fp16 快速路径: set.le.u32.f16x2 PTX 指令
        // 将 threshold 复制到 uint32 的高低 16 位
        // 一条指令同时比较两个 fp16 元素
        const uint32_t p_dropout_8bit_in_uint32_t =
            (uint32_t(uint16_t(p_dropout_in_uint8_t)) << 16)
            | uint32_t(uint16_t(p_dropout_in_uint8_t));

        #pragma unroll
        for (int m = 0; m < size<1>(tensor);
             ++m, block_row_start += block_row_stride)
        {
            uint2 rowcol =
                make_uint2(block_row_start, block_col_start);
            #pragma unroll
            for (int n = 0; n < size<2>(tensor) / 2;
                 ++n, ++rowcol.y)
            {
                // Philox RNG: 确定性, seed+offset+rowcol → 4×uint32
                uint4 random_uint4 = philox(seed,
                    reinterpret_cast<unsigned long long&>(rowcol),
                    offset);
                uint8_t (&rnd_8)[16] =
                    reinterpret_cast<uint8_t(&)[16]>(random_uint4);

                // fp16/bf16 快速路径: set.le.u32.f16x2 + AND
                if (!encode_dropout_in_sign_bit
                    && (is_half || is_bf16)) {
                    uint16_t rnd_16[16];
                    for (int i = 0; i < 16; i++)
                        rnd_16[i] = uint16_t(rnd_8[i]);
                    uint32_t (&rnd_32)[8] =
                        reinterpret_cast<uint32_t(&)[8]>(rnd_16);
                    for (int j = 0; j < 2; j++) {
                        Tensor tensor_uint32 =
                            recast<uint32_t>(tensor(_, m, n*2+j));
                        for (int i = 0; i < 4; i++) {
                            uint32_t mask;
                            asm volatile(
                                "set.le.u32.f16x2 %0, %1, %2;\n"
                                : "=r"(mask)
                                : "r"(rnd_32[j*4+i]),
                                  "r"(p_dropout_8bit_in_uint32_t));
                            tensor_uint32(i) &= mask;  // 位 AND
                        }
                    }
                } else {
                    // 通用路径: 逐元素比较
                    for (int j = 0; j < 2; j++) {
                        for (int i = 0; i < 8; i++) {
                            tensor(i, m, n*2+j) = encode_dropout(
                                rnd_8[j*8+i] <= p_dropout_in_uint8_t,
                                tensor(i, m, n*2+j));
                        }
                    }
                }
            }
        }
    }
};
```

Dropout mask 不存储，前向和反向都用相同的 seed + offset 实时生成，确保一致性。
