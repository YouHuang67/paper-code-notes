---
tags:
  - CUDA
  - CUTLASS
  - Flash Attention
---
# Flash Attention V2：参数结构与硬件配置

本文拆解 FA2 的两个基础头文件：`flash.h` 定义了前向/反向的所有参数，`kernel_traits.h` 定义了 GPU 硬件相关的计算与内存访问配置。

如果你更关心这些配置背后的性能动机，可先结合两篇附录阅读：

- [CUDA 基础：性能模型与 Occupancy](../cuda_foundations/03_cuda_performance_model_and_occupancy.md)
- [CUDA 基础：分块、数据搬运与局部性](../cuda_foundations/04_cuda_tiling_data_movement_and_locality.md)

## flash.h：参数结构体

**源码**: [flash.h](src/flash_h.md)

### 继承关系

```
Qkv_params          ← Q/K/V 指针与 stride
  └── Flash_fwd_params  ← 前向特有参数
        └── Flash_bwd_params  ← 反向额外参数（继承前向全部）
```

### Qkv_params（基类）

[flash.h:L34-L57](src/flash_h.md#__codelineno-0-34)

```cpp
namespace FLASH_NAMESPACE {
constexpr int TOTAL_DIM = 0;              // 维度索引常量
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

struct Qkv_params {
    using index_t = int64_t;
    void *__restrict__ q_ptr;                 // Q 矩阵指针
    void *__restrict__ k_ptr;                 // K 矩阵指针
    void *__restrict__ v_ptr;                 // V 矩阵指针

    // 布局: (batch, seqlen, head, dim)
    // stride 支持任意内存排列
    index_t q_batch_stride;                   // batch 维 stride
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;                     // seqlen 维 stride
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;                    // head 维 stride
    index_t k_head_stride;
    index_t v_head_stride;

    int h, h_k;                               // h: Q head 数, h_k: KV head 数
    int h_h_k_ratio;                          // h / h_k, GQA/MQA 快速索引
};
```

### Flash_fwd_params（前向）

[flash.h:L61-L156](src/flash_h.md#__codelineno-0-61)

```cpp
struct Flash_fwd_params : public Qkv_params {

    void * __restrict__ o_ptr;                // 输出矩阵 O
    void * __restrict__ oaccum_ptr;           // split-KV 中间累积（float32）
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    void * __restrict__ p_ptr;                // attention 概率矩阵（debug 用）
    void * __restrict__ softmax_lse_ptr;      // log-sum-exp, (b, h, seqlen_q)
    void * __restrict__ softmax_lseaccum_ptr; // split-KV 的 LSE 累积

    // 维度参数
    int b, seqlen_q, seqlen_k, seqlen_knew;   // batch, 序列长度
    int d, seqlen_q_rounded, seqlen_k_rounded; // head dim, 对齐后的长度
    int d_rounded, rotary_dim, total_q;

    // 缩放因子
    float scale_softmax;                      // 1/√d
    float scale_softmax_log2;                 // log₂(e)/√d, 用 exp2 替代 exp

    // 变长序列 (Varlen)
    int * __restrict__ cu_seqlens_q;          // cumulative lengths, (b+1,)
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;             // 左侧 padding
    int * __restrict__ seqused_k;             // 实际 K 序列长度
    int *__restrict__ blockmask;

    // KV Cache: append 新 KV
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // RoPE 旋转位置编码
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    int * __restrict__ cache_batch_idx;       // KV cache 索引

    // Paged KV Cache
    int * __restrict__ block_table;           // 页表
    index_t block_table_batch_stride;
    int page_block_size;

    // Dropout
    float p_dropout;                          // dropout 概率
    uint8_t p_dropout_in_uint8_t;
    float rp_dropout;                         // 1 / (1 - p_dropout)
    float scale_softmax_rp_dropout;
    at::PhiloxCudaState philox_args;          // Philox RNG 状态
    uint64_t * rng_state;

    // Local / Causal / Softcap
    int window_size_left, window_size_right;  // local attention 窗口
    float softcap;                            // softmax capping
    bool is_bf16;
    bool is_causal;                           // causal mask 标志

    bool is_seqlens_k_cumulative;
    bool is_rotary_interleaved;
    int num_splits;                           // split-KV 切分数

    // ALiBi
    void * __restrict__ alibi_slopes_ptr;     // ALiBi 斜率
    index_t alibi_slopes_batch_stride;

    bool unpadded_lse;                        // varlen: LSE 用 [nheads, total_q]
    bool seqlenq_ngroups_swapped;             // GQA 转置标志
};
```

### Flash_bwd_params（反向）

[flash.h:L160-L198](src/flash_h.md#__codelineno-0-160)

```cpp
struct Flash_bwd_params : public Flash_fwd_params {

    // 梯度张量
    void *__restrict__ do_ptr;                // dO 梯度
    void *__restrict__ dq_ptr;                // dQ 梯度
    void *__restrict__ dk_ptr;                // dK 梯度
    void *__restrict__ dv_ptr;                // dV 梯度

    // dQ 累积缓冲 (float32, 跨 block 原子加需要高精度)
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // dO/dQ/dK/dV stride
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // softmax 反向: Dᵢ = rowsum(dOᵢ ⊙ Oᵢ)
    void *__restrict__ dsoftmax_sum;

    bool deterministic;                       // 确定性模式
    index_t dq_accum_split_stride;
};
```

## kernel_traits.h：硬件配置

**源码**: [kernel_traits.h](src/kernel_traits_h.md)

### 继承关系

```
Flash_kernel_traits          ← 基础 MMA/Copy Atom
  ├── Flash_fwd_kernel_traits  ← 前向特有 Layout 与 Copy
  └── Flash_bwd_kernel_traits  ← 反向特有 Layout 与 Copy
```

### Flash_kernel_traits（基类）

[kernel_traits.h:L28-L59](src/kernel_traits_h.md#__codelineno-0-28)

```cpp
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using Element = elem_type;                // SM80+: 支持 bf16
    static constexpr bool Has_cp_async = true;
#else
    using Element = cutlass::half_t;          // SM75: 仅 fp16
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;               // 累积器始终 fp32
    using index_t = int64_t;

    /*
     * MMA Atom: m16n8k16 Tensor Core 指令
     * 每条指令计算 16×8×16 矩阵乘, 输入 fp16/bf16, 累积 fp32
     * 参见 CuTe MMA Atom (../cute/05_mma_atom.md)
     */
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,  // fp16
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>  // bf16
    >;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif

    /*
     * Shared memory → register 的 copy atom
     * ldmatrix 批量加载, 满足 Tensor Core 数据排布要求
     * 参见 CuTe 算法 (../cute/04_algorithms.md)
     */
    using SmemCopyAtom =                      // ldmatrix 正常读
        Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed =            // ldmatrix 转置读
        Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
};
```

### Flash_fwd_kernel_traits（前向）

[kernel_traits.h:L62-L172](src/kernel_traits_h.md#__codelineno-0-62)

```cpp
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false,
         typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_,
                                           kNWarps_, elem_type>>
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;     // 总线程数
    static constexpr int kBlockM = kBlockM_;            // Q 行 block
    static constexpr int kBlockN = kBlockN_;            // KV 列 block
    static constexpr int kHeadDim = kHeadDim_;          // head 维度
    static_assert(kHeadDim % 32 == 0);

    // Smem 内层维度与 Swizzle 位数
    static constexpr int kBlockKSmem =                  // smem 内层维度
        kHeadDim % 64 == 0 ? 64 : 32;                  // 64B 或 32B
    static constexpr int kBlockKGmem =                  // gmem 加载粒度
        kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle =                     // swizzle 位数
        kBlockKSmem == 32 ? 2 : 3;                      // 2→4路, 3→8路

    /*
     * TiledMMA: kNWarps 个 warp 沿 M 维排列
     * 总覆盖: M = 16*kNWarps (通常 64/128), N = 16, K = 16
     */
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,             // warp 排布
        Tile<Int<16 * kNWarps>, _16, _16>>;            // tile 尺寸

    /*
     * Shared Memory Layout (Swizzle 消除 bank conflict)
     * 基本原子: 8 行 × kBlockKSmem 列
     * Swizzle<B,M,S> 重排地址位, 使 warp 内线程访问不同 bank
     * 参见 GEMM 优化 (../cutlass_gemm_blog/03_tensorcore_and_cutlass.md)
     */
    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(         // 平铺到完整 block
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(        // Q/KV 共用 atom
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    /*
     * V 转置 Layout: P·V 矩阵乘中 V 作为 B 操作数需要转置
     * 不搬运数据, layout composition 逻辑重解释
     * (kBlockN, kHeadDim) → (kHeadDim, kBlockN)
     * 配合 SmemCopyAtomTransposed (ldmatrix.trans) 零开销转置
     */
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{},
                    make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{},
                                GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(
        get_nonswizzle_portion(SmemLayoutVtransposed{}));

    // O 的 smem layout (写回时中转)
    using SmemLayoutO = decltype(tile_to_shape(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}),
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO =                                  // O smem→gmem
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    using SmemCopyAtomOaccum =                             // split-KV fp32 输出
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

    // Shared Memory 总大小
    static constexpr int kSmemQSize =
        size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKVSize =
        size(SmemLayoutKV{}) * 2 * sizeof(Element);    // K + V
    static constexpr int kSmemSize = Share_Q_K_smem
        ? std::max(kSmemQSize, kSmemKVSize)            // Q/K 共享
        : kSmemQSize + kSmemKVSize;                    // Q + K + V
    // d=128, kBlockM=128, kBlockN=64 时: 64 KB

    /*
     * Global Memory Copy: cp.async 128-bit 异步拷贝
     * global → shared 绕过寄存器, CACHEGLOBAL 策略
     * 参见 sgemm_sm80 实战 (../cute/09_sgemm_sm80.md)
     */
    static constexpr int kGmemElemsPerLoad =
        sizeof(cute::uint128_t) / sizeof(Element);     // 8 elem
    static constexpr int kGmemThreadsPerRow =
        kBlockKSmem / kGmemElemsPerLoad;               // 每行线程数
    using GmemLayoutAtom = Layout<
        Shape <Int<kNThreads / kGmemThreadsPerRow>,
               Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;

    using Gmem_copy_struct = std::conditional_t<
        Base::Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,    // cp.async
        AutoVectorizingCopyWithAssumedAlignment<128>>;  // fallback
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));      // 8 elem/thread
    using GmemTiledCopyO = decltype(
        make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,
                       Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{}));                  // O 写回

    // split-KV: Oaccum (float32) 的 gmem copy
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_16, _8>,                         // 8 threads/row
               Stride< _8, _1>>,
        Layout<Shape <_8, _16>,                         // 16 threads/row
               Stride< _16, _1>>>;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,
                       ElementAccum>{},
            GmemLayoutAtomOaccum{},
            Layout<Shape<_1, _4>>{}));                  // 4 fp32/thread

    // RoPE: rotary cos/sin 的 gmem copy
    using GmemTiledCopyRotcossin = decltype(            // interleaved
        make_tiled_copy(
            Copy_Atom<UniversalCopy<uint64_t>, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _4>>{}));                  // 4 elem/load
    using GmemTiledCopyRotcossinCont = decltype(        // contiguous
        make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,
                       Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{}));                  // 8 elem/load
};
```

### Flash_bwd_kernel_traits（反向）

[kernel_traits.h:L176-L355](src/kernel_traits_h.md#__codelineno-0-176)

反向比前向多三个模板参数控制 warp 排布：`AtomLayoutMSdP`, `AtomLayoutNdKV`, `AtomLayoutMdQ`。

```cpp
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2,
         int AtomLayoutMdQ=2,
         bool Is_V_in_regs_=false, bool No_double_buffer_=false,
         typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_,
                                           kNWarps_, elem_type>>
struct Flash_bwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Is_V_in_regs = Is_V_in_regs_;   // 减 smem, 增 reg
    static constexpr bool No_double_buffer = No_double_buffer_;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem =
        kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem =
        kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle =
        kBlockKSmem == 32 ? 2 : 3;

    /*
     * 三组 TiledMMA, 对应反向三种矩阵乘法
     * 每组 warp 排列不同, 平衡 M/N 维尺寸
     */

    // S = Q·Kᵀ, dP = dO·Vᵀ
    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutNdKV == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);
    using TiledMmaSdP = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutMSdP>,               // M warp
                     Int<kNWarps / AtomLayoutMSdP>,
                     _1>>,
        Tile<Int<16 * AtomLayoutMSdP>,                   // M tile
             Int<16 * kNWarps / AtomLayoutMSdP>,         // N tile
             _16>>;

    // dK = dSᵀ·Q, dV = Pᵀ·dO
    using TiledMmadKV = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutNdKV>,
                     Int<kNWarps / AtomLayoutNdKV>,
                     _1>>,
        Tile<Int<16 * AtomLayoutNdKV>,
             Int<16 * kNWarps / AtomLayoutNdKV>,
             _16>>;

    // dQ = dS·K
    using TiledMmadQ = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutMdQ>,
                     Int<kNWarps / AtomLayoutMdQ>,
                     _1>>,
        Tile<Int<16 * AtomLayoutMdQ>,
             Int<16 * kNWarps / AtomLayoutMdQ>,
             _16>>;

    /*
     * Smem Layout: Q/dO, K/V, P/dS, dK/dV, dQ 各自独立
     * 反向需更多 smem 同时驻留多个矩阵
     */
    using SmemLayoutQdO = decltype(tile_to_shape(       // Q 和 dO
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}),
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutKV = decltype(tile_to_shape(         // K 和 V
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<kBlockM / kNWarps_>,
                                 Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}),
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    // K/QdO 的转置 (逻辑重解释, 零开销)
    using SmemLayoutKtransposed = decltype(
        composition(SmemLayoutKV{},
                    make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{},
                                GenRowMajor{})));
    using SmemLayoutKtransposedNoSwizzle = decltype(
        get_nonswizzle_portion(SmemLayoutKtransposed{}));
    using SmemLayoutQdOtransposed = decltype(
        composition(SmemLayoutQdO{},
                    make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{},
                                GenRowMajor{})));
    using SmemLayoutQdOtransposedNoSwizzle = decltype(
        get_nonswizzle_portion(SmemLayoutQdOtransposed{}));

    // P/dS 中间结果暂存
    static_assert(kBlockN >= 32);
    static constexpr int kPBlockN =
        kBlockN >= 64 ? 64 : 32;
    static constexpr int kSwizzlePdS = 3;
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<kSwizzlePdS, 3, 3>{},
                    Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
                           Stride<Int<kPBlockN>, _1>>{}));
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    using SmemLayoutPdStransposed = decltype(
        composition(SmemLayoutPdS{},
                    make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{},
                                GenRowMajor{})));
    using SmemLayoutPdStransposedNoSwizzle = decltype(
        get_nonswizzle_portion(SmemLayoutPdStransposed{}));
    using SmemCopyAtomPdS =                                // P/dS smem copy
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, elem_type>;

    // dK/dV 和 dQ
    using SmemLayoutdKV = decltype(tile_to_shape(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}),
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdKV =                                // dK/dV smem copy
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, elem_type>;
    using SmemLayoutdQ = decltype(tile_to_shape(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}),
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdQ =                                 // dQ smem copy
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, elem_type>;

    /*
     * Shared Memory 总大小
     * 同时驻留: Q/dO + K/V + dS + max(P, dQ)
     * 用量约为前向的 2-3 倍
     */
    static constexpr int kSmemQdOSize =                 // triple buffer
        size(SmemLayoutQdO{}) * (No_double_buffer_ ? 2 : 3)
        * sizeof(Element);
    static constexpr int kSmemKVSize =
        size(SmemLayoutKV{}) * 2 * sizeof(Element);     // K + V
    static constexpr int kSmemdSSize =
        size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemPSize =
        size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemdQSize =
        size(SmemLayoutdQ{}) * sizeof(Element);
    // Is_V_in_regs 时 V 常驻寄存器, K/V smem 可部分复用
    static constexpr int kSmemSize = kSmemQdOSize
        + (!Is_V_in_regs
           ? kSmemKVSize + kSmemdSSize
             + std::max(kSmemPSize, kSmemdQSize)
           : std::max(kSmemKVSize,
                      kSmemKVSize / 2 + kSmemdSSize
                      + std::max(kSmemPSize, kSmemdQSize)));
    // 单列 block 变体 (无 dQ 累积, 用于 deterministic 路径)
    static constexpr int kSmemSize1colblock = kSmemQdOSize
        + (!Is_V_in_regs
           ? kSmemKVSize + kSmemdSSize + kSmemPSize
           : std::max(kSmemKVSize,
                      kSmemKVSize / 2 + kSmemdSSize + kSmemPSize));

    /*
     * Global Memory Copy: 与前向相同的 cp.async 策略
     */
    static constexpr int kGmemElemsPerLoad =
        sizeof(cute::uint128_t) / sizeof(Element);     // 8 elem
    static constexpr int kGmemThreadsPerRow =
        kBlockKSmem / kGmemElemsPerLoad;
    using GmemLayoutAtom = Layout<
        Shape <Int<kNThreads / kGmemThreadsPerRow>,
               Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;

    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,    // cp.async
        AutoVectorizingCopyWithAssumedAlignment<128>>;  // fallback
    using GmemTiledCopyQKV = decltype(                  // Q/K/V 加载
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));      // 8 elem/thread
    using GmemTiledCopydO = decltype(                   // dO 加载
        make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,
                       elem_type>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{}));
    using GmemTiledCopydKV = decltype(                  // dK/dV 写回
        make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,
                       elem_type>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{}));
    using GmemTiledCopydQ = decltype(                   // dQ 写回
        make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,
                       elem_type>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{}));

    // dQ float32 累积的 gmem copy
    using GmemLayoutAtomdQaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,                         // 8 threads/row
               Stride< _8, _1>>,
        Layout<Shape <_16, _16>,                        // 16 threads/row
               Stride< _16, _1>>>;
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,
                       ElementAccum>{},
            GmemLayoutAtomdQaccum{},
            Layout<Shape<_1, _4>>{}));                  // 4 fp32/thread

    // dQ 原子加 (deterministic 模式)
    using GmemTiledCopydQaccumAtomicAdd = decltype(
        make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,
                       ElementAccum>{},
            Layout<Shape <_8, _32>,                     // 32 threads/row
                   Stride<_32, _1>>{},
            Layout<Shape<_1, _1>>{}));                  // 1 val/store
};
```
