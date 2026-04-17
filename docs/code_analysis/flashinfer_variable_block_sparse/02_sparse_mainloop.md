---
tags:
  - CUDA
  - CUTLASS
  - Sparse Attention
  - Flash Attention
---

# Hopper Sparse Mainloop

**源码位置**: [sparse_mainloop.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/hopper/sparse_mainloop.cuh)

## 定位

`SparseCollectiveMainloop` 是 Hopper（SM90）架构下 FlashInfer paged attention 的 K/V 加载主循环。它与标准 dense mainloop 的唯一区别在于 **K/V 的加载方式**：dense 版本用 TMA 连续加载，sparse 版本用 `cp.async` 按 page table 逐 token gather。

MMA 计算部分（Q·K^T 和 P·V）完全相同，都是 dense Tensor Core 运算。

## 模板结构

```cpp
template <typename AdditionalParams, typename Ktraits, bool CAUSAL, bool MULTIITEMSCORING = false>
struct SparseCollectiveMainloop {
    // tile 大小
    static constexpr int CTA_Q  = get<0>(TileShape_QKD{});   // 如 64 或 128
    static constexpr int CTA_KV = get<1>(TileShape_QKD{});   // 如 64 或 128

    // Q: TMA 连续加载
    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;

    // K/V: cp.async sparse gather（注意用的是 SM80 指令，不是 TMA）
    using GmemCopyAtomKV = cute::Copy_Atom<
        SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<AlignmentTypeKV>, DTypeKV>;

    static constexpr bool USE_TMA_LOAD_KV = false;  // 明确标记不用 TMA 加载 KV
};
```

Q 走 TMA 是因为 Q 在内存中连续（每个 Q block 是连续的 token 段）。K/V 则通过 page table 间接寻址，物理上不连续，无法用 TMA。

## 核心参数

```cpp
struct Arguments {
    DTypeQ const* Q_ptr;
    DTypeKV const* K_ptr;
    DTypeKV const* V_ptr;
    IdType const* kv_indices;      // page table（token 级索引数组）
    int window_left;
    int64_t k_page_stride;         // page 间步长（page_size=1 时等于 stride_n）
    int64_t v_page_stride;
    uint32_t page_size;            // page 大小（variable block sparse 下为 1）
};
```

当 `page_size = 1` 时，`kv_indices` 就是一个 token 级索引数组：`kv_indices[i]` 直接指向第 i 个 KV token 在全局 K/V buffer 中的位置。

## K/V 加载的两个核心 lambda

### prefetch_kv_offset：预计算全局偏移

```cpp
'''
对当前 tile 内的每个 KV position，通过 page table 查找其在全局内存中的偏移。
每个线程负责一个 KV position，预计算结果存入 my_kv_offset[parity] 双缓冲。
'''
auto prefetch_kv_offset = [&](int kv_tile_idx, bool use_predicate) {
    int kv_base_idx = kv_tile_idx * CTA_KV;
    int kv_idx_read = kv_base_idx + group_id + thread_in_group * KV_STRIDE;
    bool valid_read = thread_in_group < NUM_ITERS_PER_GROUP
                      && (!use_predicate || kv_idx_read < kv_len);
    if (valid_read) {
        uint32_t page_iter, entry_idx;
        mainloop_params.page_size.divmod(kv_idx_read, page_iter, entry_idx);
        IdType page_idx = kv_indices_ptr[page_iter];       // page table 查找
        my_kv_offset[parity] = page_idx * k_page_stride    // page 起始地址
                             + entry_idx * k_stride_n;     // page 内偏移
    } else {
        my_kv_offset[parity] = 0;
    }
};
```

当 `page_size = 1` 时，`divmod` 退化为：`page_iter = kv_idx_read`，`entry_idx = 0`。所以偏移直接就是 `kv_indices[kv_idx_read] * k_page_stride`。

### load_kv_with_gather：稀疏加载到 shared memory

```cpp
'''
每个线程根据自己负责的 KV position，从全局内存 gather 加载数据到 shared memory。
关键技巧：用 __shfl_sync 从拥有该 position 偏移的线程广播预计算的 base_offset。
'''
auto load_kv_with_gather = [&](auto&& tXsX, auto&& tXcX,
                               DTypeKV* base_ptr, int kv_tile_idx,
                               int stage_idx, bool use_predicate) {
    using Vec = AlignmentTypeKV;                            // 128-bit 向量类型
    constexpr int VecSize = sizeof(Vec) / sizeof(DTypeKV);  // 如 bf16 下为 8
    int kv_base_idx = kv_tile_idx * CTA_KV;

    auto dst = recast<Vec>(flatten(tXsX(_, _, _, stage_idx)));  // smem 目标
    auto c = flatten(tXcX(_, _, _, kv_tile_idx));               // 坐标映射

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(dst); ++i) {
        auto coord = c(VecSize * i);
        int kv_offset = get<0>(coord);                     // tile 内 KV 偏移
        int d_idx = get<1>(coord);                         // head_dim 维偏移
        int kv_idx = kv_base_idx + kv_offset;
        bool guard = !use_predicate || kv_idx < kv_len;

        '''
        __shfl_sync: 从持有该 KV position 预计算偏移的线程获取 base_offset。
        每个 group 内的 thread 0 持有 group_id 对应位置的偏移，
        src_thread 计算出目标线程 ID，通过 warp shuffle 读取。
        '''
        int src_thread = group_id * THREADS_PER_GROUP + kv_offset / KV_STRIDE;
        int64_t base_offset = __shfl_sync(FULL_MASK, my_kv_offset[parity], src_thread);

        Vec const* src_ptr = reinterpret_cast<Vec const*>(
            base_ptr + base_offset + d_idx);               // 最终全局地址
        cutlass::arch::cp_async_zfill<sizeof(Vec),
            cutlass::arch::CacheOperation::Global>(
            &dst(i), src_ptr, guard);                      // 异步拷贝到 smem
    }
};
```

## 线程组织

```cpp
constexpr int NUM_KV_PER_ITER = decltype(size<1>(tKcK))::value;  // 如 12
constexpr int KV_STRIDE = CTA_KV / NUM_KV_PER_ITER;              // 96/12 = 8
constexpr int NUM_GROUPS = KV_STRIDE;                             // 8 groups
constexpr int THREADS_PER_GROUP = NUM_COPY_THREADS / NUM_GROUPS;  // 128/8 = 16
```

128 个 producer 线程分为 8 组，每组 16 个线程。每组负责 CTA_KV 中等间距的若干 KV position。`prefetch_kv_offset` 阶段每组中有 12 个线程实际执行 page table 查找，`load_kv_with_gather` 阶段通过 `__shfl_sync` 让组内所有线程共享预计算的偏移。

## 双缓冲流水线

`my_kv_offset[2]` 是一个双缓冲数组，`parity` 变量在 0/1 之间切换：

```
迭代 N:
  parity=0: prefetch tile N 的 K 偏移 → my_kv_offset[0]
  parity=0: load K tile N from my_kv_offset[0]
  parity=1: prefetch tile N-1 的 K 偏移 → my_kv_offset[1]
  parity=1: load K tile N-1
  parity=0: load V tile N from my_kv_offset[0]（复用之前预取的偏移）
```

这样当加载 K tile N 时，可以同时预取 tile N-1 的偏移，实现 prefetch 与 load 的重叠。配合 CUTLASS pipeline（`pipeline_k` / `pipeline_v`），K 和 V 的加载也在不同 stage 中流水。

## 主循环结构

```cpp
template <bool LEFT_SLIDING_WINDOW, ...>
CUTLASS_DEVICE void load(Params const& mainloop_params,
                          MainloopPipeline pipeline_k,
                          MainloopPipeline pipeline_v,
                          ...) {
    '''
    加载最后一个 K tile（从后向前遍历）
    '''
    prefetch_kv_offset(kv_tile_idx, true);
    pipeline_k.producer_acquire(smem_pipe_write_k);
    load_kv_with_gather(tKsK, tKcK, K_ptr_base, kv_tile_idx, ...);
    pipeline_k.producer_commit(smem_pipe_write_k, cpasync_barrier_arrive);

    '''
    等待 Q 加载完成后，加载 Q tile（TMA，仅 warp 0 lane 0 发起）
    '''
    if (warp_idx_in_warpgroup == 0) {
        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
            shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
            copy(mainloop_params.tma_load_Q.with(...), tQgQ, tQsQ);
        }
    }

    '''
    从后向前遍历剩余 KV tiles：
    每次迭代交替加载 K tile 和 V tile，双缓冲偏移预取重叠
    '''
    #pragma unroll 2
    for (; kv_tile_idx > swa_begin_kv_tile_idx; kv_tile_idx = decrement(kv_tile_idx)) {
        // prefetch next K tile offset
        parity ^= 1;
        prefetch_kv_offset(kv_tile_k, false);
        // load K tile
        pipeline_k.producer_acquire(smem_pipe_write_k);
        load_kv_with_gather(tKsK, tKcK, K_ptr_base, kv_tile_k, ...);
        pipeline_k.producer_commit(smem_pipe_write_k, ...);
        // load V tile (reuse previous offset)
        parity ^= 1;
        pipeline_v.producer_acquire(smem_pipe_write_v);
        load_kv_with_gather(tVsV, tVcV, V_ptr_base, kv_tile_idx, ...);
        pipeline_v.producer_commit(smem_pipe_write_v, ...);
        parity ^= 1;
    }
}
```

## 与 dense mainloop 的对比

| 维度 | Dense Mainloop | Sparse Mainloop |
|---|---|---|
| Q 加载 | TMA (`SM90_TMA_LOAD`) | TMA (`SM90_TMA_LOAD`)——相同 |
| K/V 加载 | TMA 连续加载 | `cp.async` gather + page table |
| 寻址方式 | 直接指针偏移 | `kv_indices` 间接寻址 |
| 额外开销 | 无 | `prefetch_kv_offset` + `__shfl_sync` |
| MMA 计算 | dense Tensor Core | dense Tensor Core——相同 |
| Pipeline | K/V 双流水线 | K/V 双流水线——相同 |

核心差异只在 K/V 加载路径。加载到 shared memory 后，后续的 MMA 计算完全一样——这就是 **sparse loading + dense computation** 范式。

## FP8 变体

**源码位置**: [mainloop_sparse_load.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/hopper/quantization/mainloop_sparse_load.cuh)

`FP8SparseCollectiveMainloop` 采用完全相同的 sparse gather load 模式（`prefetch_kv_offset` + `load_kv_with_gather`），额外处理 FP8 量化的数据类型转换。这进一步说明 sparse loading 是 FlashInfer Hopper 路径的通用设计，不是单一 kernel 的特殊优化。

## 小结

Hopper sparse mainloop 的设计非常聚焦：

1. **Q 连续，K/V 稀疏**：Q 用 TMA 一次加载整个 tile，K/V 通过 page table 逐 token gather
2. **预计算 + 广播**：每个线程预计算自己负责的 KV position 的全局偏移，通过 `__shfl_sync` 让组内线程共享
3. **加载后即稠密**：数据到 shared memory 后，MMA 看到的就是连续的 tile，无需感知稀疏性
4. **双缓冲流水**：偏移预取与数据加载重叠，K 和 V 分 pipeline stage 流水

整个设计的代价是 K/V 加载带宽略低于 TMA（因为 gather 效率 < 连续加载），但省去了把不需要的 KV token 加载进来的浪费。在稀疏度足够高时（如 SVG2 的 15-30% 密度），净效果是显著加速。
