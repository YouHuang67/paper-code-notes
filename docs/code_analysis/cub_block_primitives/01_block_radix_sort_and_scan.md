---
tags:
  - CUDA
---

# CUB Block 级原语：BlockRadixSort 与 BlockScan

## 概述

CUB（CUDA UnBound）是 NVIDIA 提供的 CUDA C++ 模板库，提供线程（thread）、warp、block、device 四个层级的并行原语。本文聚焦 **block 级** 原语，它们是单个 thread block 内的 collective operation，所有线程协作完成一次计算。

本文重点介绍两个核心原语及其依赖链：

- **BlockRadixSort**：块内基数排序，支持 key-only 和 key-value pair
- **BlockScan**：块内并行前缀扫描（prefix scan / prefix sum）

以及它们依赖的辅助原语：BlockRadixRank、BlockExchange、BlockLoad/BlockStore。

**源码位置**: `/usr/local/cuda-12.4/include/cub/block/`

## CUB Block 级原语的共性设计

### 模板参数与特化

所有 block 级原语遵循相同的模板参数模式：

```cpp
template <
    typename    T,                // 数据类型
    int         BLOCK_DIM_X,      // block X 维度线程数
    int         ITEMS_PER_THREAD, // 每线程处理的元素数（部分原语有此参数）
    /* 算法选择、可选维度等 */
    int         BLOCK_DIM_Y = 1,
    int         BLOCK_DIM_Z = 1>
class BlockXxx { ... };
```

关键设计点：

- **BLOCK_DIM_X × ITEMS_PER_THREAD = SORT_SIZE**：总处理元素数在编译期确定，所有循环可 `#pragma unroll`
- **BLOCK_DIM_Y / Z**：支持 2D/3D thread block，内部统一转换为 `linear_tid`
- **算法策略**：通过模板枚举参数（如 `BlockScanAlgorithm`）在编译期选择实现策略

### TempStorage 与 SMEM union 复用

每个原语定义一个嵌套类型 `TempStorage`，封装该原语所需的 shared memory：

```cpp
using BlockSort = cub::BlockRadixSort<float, 128, 4, int>;
__shared__ typename BlockSort::TempStorage temp_storage;
BlockSort(temp_storage).SortDescending(keys, values);
```

多个原语的 TempStorage 可用 union 复用同一块 SMEM，因为它们的生命期不重叠（中间用 `__syncthreads()` 隔开）：

```cpp
__shared__ union {
    typename BlockLoad::TempStorage   load;
    typename BlockSort::TempStorage   sort;
    typename BlockScan::TempStorage   scan;
    typename BlockStore::TempStorage  store;
} temp;
```

**源码位置**: BlockRadixSort 内部的 `_TempStorage` union 复用了 AscendingBlockRadixRank / DescendingBlockRadixRank / BlockExchangeKeys / BlockExchangeValues 四种存储。
[block_radix_sort.cuh:288-294](https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_radix_sort.cuh)

### 数据布局：Blocked vs Striped

CUB 的 block 级原语围绕两种数据布局工作：

**Blocked 布局**（默认）：每个线程持有连续的 ITEMS_PER_THREAD 个元素

```
Thread 0: [0, 1, 2, 3]    Thread 1: [4, 5, 6, 7]    Thread 2: [8, 9, 10, 11] ...
```

**Striped 布局**：元素按线程 ID 交错分布

```
Thread 0: [0, 128, 256, 384]    Thread 1: [1, 129, 257, 385] ...
```

Blocked 布局适合 block 内 collective 操作（排序、scan），Striped 布局适合全局内存的 coalesced 访问。BlockExchange 负责两种布局之间的转换，BlockLoad/BlockStore 的 `TRANSPOSE` 模式也隐式包含了这种转换。

## BlockRadixSort

### 接口与模板参数

**源码位置**: [block_radix_sort.cuh](https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_radix_sort.cuh)

```cpp
template <
    typename    KeyT,               // key 类型（int/float/double/__half 等）
    int         BLOCK_DIM_X,        // block X 维度线程数
    int         ITEMS_PER_THREAD,   // 每线程持有的 key 数量
    typename    ValueT = NullType,  // value 类型（NullType 表示 key-only）
    int         RADIX_BITS = 4,     // 每个 digit 的位数（默认 4 = 16 路桶）
    bool        MEMOIZE_OUTER_SCAN = true,
    BlockScanAlgorithm INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS,
    /* SMEM bank config, BLOCK_DIM_Y/Z 等可选参数 */ >
class BlockRadixSort;
```

关键参数说明：

- **BLOCK_DIM_X × ITEMS_PER_THREAD = SORT_SIZE**：块内排序的总元素数，编译期确定
- **RADIX_BITS = 4**：每 pass 处理 4 bit，即 16 个桶。32 位 float 需要 32/4 = 8 个 pass
- **MEMOIZE_OUTER_SCAN**：是否将 raking segment 缓存到寄存器，减少 SMEM 读取，增加寄存器压力
- **ValueT = NullType**：指定 NullType 时为 key-only 排序，否则为 key-value pair 排序

主要公开接口：

| 方法 | 说明 |
|------|------|
| `Sort(keys, [values], [begin_bit], [end_bit])` | 升序排序（blocked → blocked） |
| `SortDescending(keys, [values], ...)` | 降序排序（blocked → blocked） |
| `SortBlockedToStriped(keys, [values], ...)` | 升序排序（blocked → striped） |
| `SortDescendingBlockedToStriped(...)` | 降序排序（blocked → striped） |

`begin_bit / end_bit` 允许只排序 key 的部分 bit 位，减少 pass 数量。例如排序 float 的高 16 位：`Sort(keys, values, 16, 32)`。

### 浮点数的 bit 保序映射

Radix sort 本质上是对无符号整数的 bit 模式进行排序。要对 signed 和 float 类型排序，需要先将它们转换为保序的无符号表示，排序完成后再转换回来。

CUB 通过 `radix::traits_t<KeyT>::bit_ordered_conversion_policy` 处理这种转换：

- **unsigned 整数**：bit 模式直接保序，无需转换
- **signed 整数**：翻转符号位（最高位）。因为二进制补码中负数的最高位为 1，翻转后变为 0，排在正数前面
- **float（正数）**：翻转符号位。IEEE 754 正浮点数的 bit 模式本身就是保序的（指数在高位，尾数在低位）
- **float（负数）**：翻转全部 bit。因为负浮点数的 bit 模式与其绝对值是反序的

```cpp
// 排序前：to_bit_ordered 转换
unsigned_keys[KEY] = bit_ordered_conversion::to_bit_ordered(decomposer, unsigned_keys[KEY]);

// ... 多轮 radix pass ...

// 排序后：from_bit_ordered 逆转换
unsigned_keys[KEY] = bit_ordered_conversion::from_bit_ordered(decomposer, unsigned_keys[KEY]);
```

**源码位置**: [radix_rank_sort_operations.cuh](https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/radix_rank_sort_operations.cuh)

特殊处理 `-0.0`：IEEE 754 中 `+0.0` 和 `-0.0` 的 bit 模式不同（符号位不同），但语义相等。CUB 的 `BaseDigitExtractor<KeyT, FLOATING_POINT>::ProcessFloatMinusZero` 将 `-0.0` 的 twiddled 表示映射为 `+0.0`，确保两者排序位置相同，保证稳定性。

### 内部流程：多轮 Rank-Exchange

`SortBlocked` 是所有排序方法的核心实现。每一轮处理 RADIX_BITS 个 bit（默认 4 bit），总共需要 `ceil((end_bit - begin_bit) / RADIX_BITS)` 轮。

```cpp
template <int DESCENDING, int KEYS_ONLY, class DecomposerT>
__device__ void SortBlocked(KeyT (&keys)[ITEMS_PER_THREAD],
                            ValueT (&values)[ITEMS_PER_THREAD],
                            int begin_bit, int end_bit, ...)
{
    '''
    to_bit_ordered：将 key 转换为保序的无符号表示
    对 float：正数翻转符号位，负数翻转全部 bit
    '''
    #pragma unroll
    for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        unsigned_keys[KEY] = bit_ordered_conversion::to_bit_ordered(...);

    '''
    多轮 radix pass：每轮处理 RADIX_BITS 个 bit
    每轮两步：
      1. RankKeys：根据当前 digit 计算每个 key 的目标位置 ranks[]
      2. BlockExchange::ScatterToBlocked：按 ranks 重排 keys 和 values
    '''
    while (true) {
        int pass_bits = CUB_MIN(RADIX_BITS, end_bit - begin_bit);
        auto digit_extractor = traits::digit_extractor(begin_bit, pass_bits);

        int ranks[ITEMS_PER_THREAD];
        RankKeys(unsigned_keys, ranks, digit_extractor, is_descending);
        begin_bit += RADIX_BITS;

        CTA_SYNC();
        BlockExchangeKeys(temp_storage.exchange_keys).ScatterToBlocked(keys, ranks);
        ExchangeValues(values, ranks, ...);

        if (begin_bit >= end_bit) break;
        CTA_SYNC();
    }

    '''
    from_bit_ordered：逆转换，恢复原始类型表示
    '''
    #pragma unroll
    for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        unsigned_keys[KEY] = bit_ordered_conversion::from_bit_ordered(...);
}
```

每轮的两个关键步骤：

1. **RankKeys**（由 BlockRadixRank 实现）：根据当前 digit 值，为每个 key 计算其在排序结果中的目标位置
2. **ScatterToBlocked**（由 BlockExchange 实现）：按照计算出的 rank，通过 SMEM 将 keys/values 重排到正确位置

### BlockRadixRank：digit 计数与排名

BlockRadixRank 是 BlockRadixSort 内部最核心的组件。它负责在一轮 radix pass 中，为每个 key 计算其目标位置（rank）。

**源码位置**: [block_radix_rank.cuh](https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_radix_rank.cuh)

#### SMEM 计数器布局

BlockRadixRank 在 SMEM 中维护一个 digit 计数器矩阵。每个线程有自己独立的一列计数器，用于统计自己持有的 key 中各 digit 值出现的次数：

```cpp
// RADIX_BITS = 4 → 16 个 digit 值
// 用 unsigned short (2B) 作为计数器，一个 unsigned int (4B) 可以 pack 2 个计数器
// PACKING_RATIO = sizeof(PackedCounter) / sizeof(DigitCounter) = 4/2 = 2
// COUNTER_LANES = 2^max(RADIX_BITS - LOG_PACKING_RATIO, 0) = 2^(4-1) = 8
// PADDED_COUNTER_LANES = COUNTER_LANES + 1 = 9 （加 1 避免 bank conflict）

DigitCounter digit_counters[PADDED_COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO];
// 等价视图（reinterpret_cast）：
PackedCounter raking_grid[BLOCK_THREADS][RAKING_SEGMENT];
```

两个关键的布局设计：

- **打包（packing）**：将多个 `DigitCounter`（unsigned short）pack 进一个 `PackedCounter`（unsigned int），scan 时可以并行处理多个计数器
- **Padding**：`PADDED_COUNTER_LANES = COUNTER_LANES + 1`，确保不同线程的同一 counter lane 不会落入同一 SMEM bank，避免 bank conflict

#### RankKeys 流程

```cpp
template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT>
__device__ void RankKeys(UnsignedBits (&keys)[KEYS_PER_THREAD],
                         int (&ranks)[KEYS_PER_THREAD],
                         DigitExtractorT digit_extractor)
{
    DigitCounter  thread_prefixes[KEYS_PER_THREAD];
    DigitCounter* digit_counters[KEYS_PER_THREAD];

    '''
    Phase 1: 计数
    每个线程遍历自己的 KEYS_PER_THREAD 个 key，
    提取当前 pass 的 digit 值，在对应的 SMEM 计数器上 +1。
    同时保存该线程在该 digit 上的 thread-exclusive prefix。
    '''
    ResetCounters();
    #pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
        uint32_t digit = digit_extractor.Digit(keys[ITEM]);
        uint32_t sub_counter = digit >> LOG_COUNTER_LANES;
        uint32_t counter_lane = digit & (COUNTER_LANES - 1);

        digit_counters[ITEM] = &temp_storage.digit_counters[counter_lane][linear_tid][sub_counter];
        thread_prefixes[ITEM] = *digit_counters[ITEM];  // 读取当前计数（排在我前面的同 digit 元素数）
        *digit_counters[ITEM] = thread_prefixes[ITEM] + 1;  // 计数 +1
    }
    CTA_SYNC();

    '''
    Phase 2: Scan（前缀扫描）
    对 SMEM 中的 packed 计数器做 block-wide exclusive prefix sum。
    scan 完成后，每个线程的每个 counter 位置存储的是"所有 thread ID 更小的线程"
    中同一 digit 值的总数。
    这一步通过 raking_grid 视图操作 packed counter，利用 BlockScan 完成。
    '''
    ScanCounters();  // Upsweep reduce → BlockScan::ExclusiveSum → Downsweep scatter
    CTA_SYNC();

    '''
    Phase 3: 提取 rank
    每个 key 的最终 rank = thread_prefix（线程内同 digit 的先序计数）
                         + scan 后的 counter 值（跨线程同 digit 的先序计数）
    '''
    #pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
        ranks[ITEM] = thread_prefixes[ITEM] + *digit_counters[ITEM];
}
```

#### ScanCounters 细节

计数完成后，需要对所有线程的计数器做 exclusive prefix sum，使得每个计数器位置存储的是全局前缀计数。这通过 raking 模式实现：

```cpp
__device__ void ScanCounters()
{
    '''
    Upsweep：每个线程将自己的 raking segment（PADDED_COUNTER_LANES 个 packed counter）
    做 sequential reduction，得到一个 packed partial sum。
    如果 MEMOIZE_OUTER_SCAN=true，同时将 segment 缓存到寄存器。
    '''
    PackedCounter raking_partial = Upsweep();

    '''
    BlockScan：对所有线程的 packed partial sum 做 block-wide exclusive scan。
    注意这里 scan 的是 PackedCounter 类型，加法在 packed 的各个子 counter 上独立进行
    （因为子 counter 不会溢出，所以不会产生跨 sub-counter 的进位）。
    PrefixCallBack 处理跨 packing 边界的累加传播。
    '''
    PackedCounter exclusive_partial;
    PrefixCallBack prefix_call_back;
    BlockScan(temp_storage.block_scan)
        .ExclusiveSum(raking_partial, exclusive_partial, prefix_call_back);

    '''
    Downsweep：每个线程用 exclusive_partial 作为种子，
    对自己的 raking segment 做 sequential exclusive scan，
    将全局前缀写回 SMEM。
    '''
    ExclusiveDownsweep(exclusive_partial);
}
```

这里 `PrefixCallBack` 做了一个精妙的操作：它将 block aggregate 在 packed counter 的各个子 counter 之间传播。例如对于 PACKING_RATIO=2，一个 `uint32_t` 包含两个 `uint16_t` 计数器 `[low, high]`，low 的总数需要加到 high 的前缀上：

```cpp
struct PrefixCallBack {
    __device__ PackedCounter operator()(PackedCounter block_aggregate) {
        PackedCounter block_prefix = 0;
        #pragma unroll
        for (int PACKED = 1; PACKED < PACKING_RATIO; PACKED++)
            block_prefix += block_aggregate << (sizeof(DigitCounter) * 8 * PACKED);
        return block_prefix;
    }
};
```

### BlockExchange：按 rank 重排数据

每轮 radix pass 计算出 ranks 后，需要将 keys 和 values 按 rank 搬移到目标位置。BlockExchange 通过 SMEM 实现这一步：

**源码位置**: [block_exchange.cuh](https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_exchange.cuh)

```cpp
// ScatterToBlocked：按 ranks 将数据写入 SMEM，再以 blocked 顺序读出
__device__ void ScatterToBlocked(InputT (&items)[ITEMS_PER_THREAD],
                                  int (&ranks)[ITEMS_PER_THREAD])
{
    // 步骤 1：每个线程将自己的 items 写入 SMEM 的 ranks[i] 位置
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        temp_storage.buff[ranks[ITEM]] = items[ITEM];

    CTA_SYNC();

    // 步骤 2：每个线程从 SMEM 的 blocked 位置读回数据
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        items[ITEM] = temp_storage.buff[linear_tid * ITEMS_PER_THREAD + ITEM];
}
```

SMEM 大小 = `BLOCK_THREADS × ITEMS_PER_THREAD × sizeof(T)` + padding（避免 bank conflict，当 ITEMS_PER_THREAD 为 2 的幂且 > 4 时插入 padding）。

### 降序排序的实现

BlockRadixSort 的降序不是通过翻转 bit 实现的（那是 DeviceRadixSort 的做法），而是在 BlockRadixRank 内部将 digit 值和 counter lane 做镜像映射：

```cpp
if (IS_DESCENDING) {
    sub_counter = PACKING_RATIO - 1 - sub_counter;
    counter_lane = COUNTER_LANES - 1 - counter_lane;
}
```

这使得 digit 值越大的 key 获得越小的 rank，从而实现降序。

### 完整排序示例

以 `BlockRadixSort<float, 32, 16, int32_t>` 降序排序 512 个 float-int KV pair 为例：

| 参数 | 值 |
|------|-----|
| SORT_SIZE | 32 × 16 = 512 |
| BLOCK_THREADS | 32 |
| ITEMS_PER_THREAD | 16 |
| RADIX_BITS | 4（默认） |
| 总 pass 数 | 32 / 4 = 8 |
| 每 pass 桶数 | 2^4 = 16 |

每个 pass 的步骤：
1. `to_bit_ordered`：float → uint32（仅第一个 pass 前执行一次）
2. `BFEDigitExtractor::Digit`：用 `BFE`（Bit Field Extract）指令提取当前 4-bit digit
3. `BlockRadixRank::RankKeys`：计数 + packed scan + 提取 rank
4. `BlockExchangeKeys::ScatterToBlocked`：按 rank 重排 keys
5. `BlockExchangeValues::ScatterToBlocked`：按 rank 重排 values
6. `from_bit_ordered`：uint32 → float（仅最后一个 pass 后执行一次）

## BlockScan

### 接口与模板参数

**源码位置**: [block_scan.cuh](https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_scan.cuh)

```cpp
template <
    typename            T,              // 数据类型
    int                 BLOCK_DIM_X,    // block X 维度线程数
    BlockScanAlgorithm  ALGORITHM = BLOCK_SCAN_RAKING,  // 算法策略
    int                 BLOCK_DIM_Y = 1,
    int                 BLOCK_DIM_Z = 1>
class BlockScan;
```

BlockScan 实现块内并行前缀扫描（prefix scan）。给定每个线程的输入值，计算所有线程输入的前缀和/前缀归约。

主要接口分为三大类：

| 类别 | 方法 | 说明 |
|------|------|------|
| Exclusive Sum | `ExclusiveSum(input, output)` | exclusive 前缀和，初始值为 0 |
| Inclusive Sum | `InclusiveSum(input, output)` | inclusive 前缀和 |
| Generic Scan | `ExclusiveScan(input, output, init, scan_op)` | 自定义二元运算 + 初始值 |

每类又有三个变体：

- 基础版：只计算 scan 结果
- + `block_aggregate`：额外返回全 block 的归约总和
- + `block_prefix_callback_op`：支持 callback 函数注入 block 前缀（用于多 tile 的跨 block 级联 scan）

### 三种算法策略

BlockScan 提供三种编译期可选的算法策略：

#### BLOCK_SCAN_RAKING（默认）

高吞吐的 "raking reduce-then-scan" 算法，分 5 个阶段：

1. **Upsweep 寄存器归约**：每个线程将自己的 ITEMS_PER_THREAD 个元素做 sequential reduction，得到一个 partial sum，写入 SMEM
2. **Upsweep SMEM raking 归约**：一个 warp 内的线程 rake 过 SMEM 中各段 partial reductions，做 sequential reduction
3. **Warp-synchronous Kogge-Stone scan**：在 raking warp 内做 warp-level exclusive scan
4. **Downsweep SMEM raking scan**：raking warp 将 scan 结果 scatter 回 SMEM 的各段
5. **Downsweep 寄存器 scan**：每个线程用 SMEM 中的 prefix 对自己的 ITEMS_PER_THREAD 个元素做 sequential scan

特点：占用率充足时吞吐量最高，因为 raking 模式最大化了串行工作量、最小化了同步次数。

#### BLOCK_SCAN_RAKING_MEMOIZE

与 RAKING 相同的算法，但在 upsweep 阶段将 SMEM segment 缓存到寄存器，downsweep 时直接从寄存器读取而不需要再次访问 SMEM。以更高的寄存器压力换取更少的 SMEM 读取。

#### BLOCK_SCAN_WARP_SCANS

低延迟的 "tiled warpscans" 算法，分 4 个阶段：

1. **Upsweep 寄存器归约**：同 RAKING
2. **Warp-level Kogge-Stone scan**：每个 warp 内部独立做 warp scan（利用 `__shfl` 指令，无需 SMEM）
3. **跨 warp 传播**：收集每个 warp 的 aggregate，sequential 累加得到各 warp 的 prefix
4. **Downsweep 寄存器 scan**：每个线程用 warp prefix 修正自己的输出

特点：延迟更低（因为 warp scan 使用 shuffle 指令，不需要 SMEM 和 `__syncthreads`），但 warp scan 本身的工作效率较低（Kogge-Stone 是 O(N log N) 工作量）。适合 GPU 负载不足时追求低延迟。

**约束**：BLOCK_SCAN_WARP_SCANS 要求 BLOCK_THREADS 是 warp size 的倍数，否则自动回退到 RAKING。

### BLOCK_SCAN_WARP_SCANS 内部实现

这是 BlockRadixSort 默认使用的 scan 策略（通过 `INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS`）。以 ExclusiveSum 为例，详细分析其实现。

**源码位置**: [block_scan_warp_scans.cuh](https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/specializations/block_scan_warp_scans.cuh)

#### SMEM 布局

```cpp
struct _TempStorage {
    T warp_aggregates[WARPS];               // 每个 warp 的 aggregate
    typename WarpScanT::TempStorage warp_scan[WARPS];  // 每个 warp 的 scan 临时空间
    T block_prefix;                          // block 级前缀
};
```

#### ExclusiveScan 流程

```cpp
template <typename ScanOp>
__device__ void ExclusiveScan(T input, T &exclusive_output,
                              const T &initial_value, ScanOp scan_op,
                              T &block_aggregate)
{
    '''
    Step 1: Warp-level scan
    每个 warp 内部独立做 inclusive + exclusive scan。
    WarpScanT 基于 Kogge-Stone 算法，使用 __shfl_up_sync 实现，
    O(log W) 步完成（W = warp size = 32）。
    '''
    T inclusive_output;
    WarpScanT(temp_storage.warp_scan[warp_id])
        .Scan(input, inclusive_output, exclusive_output, scan_op);

    '''
    Step 2: 跨 warp 前缀计算
    每个 warp 的最后一个 lane（lane_id == 31）将 inclusive 结果
    （即该 warp 的 aggregate）写入 SMEM 的 warp_aggregates[]。
    然后 sequential 累加所有 warp 的 aggregate，得到：
      - 每个 warp 的 prefix（该 warp 之前所有 warp 的 aggregate 之和）
      - 整个 block 的 aggregate
    '''
    T warp_prefix = ComputeWarpPrefix(scan_op, inclusive_output,
                                       block_aggregate, initial_value);

    '''
    Step 3: 应用 warp prefix
    每个线程的最终结果 = scan_op(warp_prefix, exclusive_output)
    lane 0 特殊处理：直接使用 warp_prefix（因为 lane 0 的 exclusive output 未定义）
    '''
    exclusive_output = scan_op(warp_prefix, exclusive_output);
    if (lane_id == 0)
        exclusive_output = warp_prefix;
}
```

#### ComputeWarpPrefix 的模板展开

跨 warp 前缀计算通过编译期模板递归展开实现，避免运行时循环：

```cpp
template <typename ScanOp, int WARP>
__device__ void ApplyWarpAggregates(T &warp_prefix, ScanOp scan_op,
                                     T &block_aggregate,
                                     Int2Type<WARP>)
{
    if (warp_id == WARP)
        warp_prefix = block_aggregate;  // 当前 warp 的 prefix = 前面所有 warp 的 aggregate 之和

    T addend = temp_storage.warp_aggregates[WARP];
    block_aggregate = scan_op(block_aggregate, addend);

    ApplyWarpAggregates(warp_prefix, scan_op, block_aggregate, Int2Type<WARP + 1>());
}

// 递归终止
template <typename ScanOp>
__device__ void ApplyWarpAggregates(..., Int2Type<WARPS>) {}
```

对于 WARPS=4（128 线程 / 32），编译器展开为：

```
block_aggregate = warp_aggregates[0]
warp 1 的 prefix = warp_aggregates[0]
block_aggregate = warp_aggregates[0] + warp_aggregates[1]
warp 2 的 prefix = warp_aggregates[0] + warp_aggregates[1]
block_aggregate = warp_aggregates[0] + warp_aggregates[1] + warp_aggregates[2]
warp 3 的 prefix = warp_aggregates[0] + warp_aggregates[1] + warp_aggregates[2]
block_aggregate = warp_aggregates[0] + ... + warp_aggregates[3]
```

这个 O(WARPS) 的 sequential 累加在 warp 数较少时（通常 ≤ 32）完全可接受，且因为模板展开在编译期完成，没有任何分支开销。

### BlockScan 在 BlockRadixRank 中的特殊用法

在 BlockRadixRank 的 `ScanCounters()` 中，BlockScan 被用来对 `PackedCounter`（unsigned int）类型做 ExclusiveSum。这里有一个精妙之处：packed counter 内部的多个 sub-counter（unsigned short）通过整数加法"并行" scan。

```cpp
// PackedCounter = uint32_t，包含 2 个 uint16_t sub-counter
// 例如：[count_digit_0, count_digit_1]

// BlockScan 对 PackedCounter 做加法时：
// [a0, a1] + [b0, b1] = [a0+b0, a1+b1]  (只要 sub-counter 不溢出)
```

这成立的前提是每个 sub-counter 的值不会溢出到相邻的 sub-counter。对于 KEYS_PER_THREAD ≤ 32 的配置，每个线程最多 32 个 key，总共 BLOCK_THREADS × 32 ≤ 4096 个 key，每个 digit 的最大计数不超过 4096 < 65535（uint16_t max），不会溢出。

但 packed scan 的跨 sub-counter 前缀传播（digit 0 的总数需要加到 digit 1 的前缀上）由 `PrefixCallBack` 处理，这在前面 BlockRadixRank 一节已经分析过。

## PyTorch 中的使用范例

### radixSortKVInPlace（torch.sort 块内路径）

PyTorch 的 `torch.sort` 对于中等规模（≤ 4096 元素）的排序使用 CUB BlockRadixSort。

**源码位置**: `ATen/native/cuda/SortUtils.cuh` → `radixSortKVInPlace`

```cpp
template <int KeyDims, int ValueDims,
          int block_size, int items_per_thread,
          typename K, typename V, typename IndexType>
__global__ void radixSortKVInPlace(...) {
    using key_t = typename at::cuda::cub::detail::cuda_type<K>::type;
    using LoadKeys   = cub::BlockLoad<K, block_size, items_per_thread, BLOCK_LOAD_TRANSPOSE>;
    using LoadValues = cub::BlockLoad<V, block_size, items_per_thread, BLOCK_LOAD_TRANSPOSE>;
    using Sort       = cub::BlockRadixSort<key_t, block_size, items_per_thread, V>;
    using StoreKeys  = cub::BlockStore<K, block_size, items_per_thread, BLOCK_STORE_TRANSPOSE>;
    using StoreValues= cub::BlockStore<V, block_size, items_per_thread, BLOCK_STORE_TRANSPOSE>;

    '''
    SMEM union 复用：Load/Sort/Store 三个阶段各自的 TempStorage
    不会同时使用，用 union 共享同一块 SMEM
    '''
    __shared__ union {
        typename LoadKeys::TempStorage   load_keys;
        typename LoadValues::TempStorage load_values;
        typename Sort::TempStorage       sort;
        typename StoreKeys::TempStorage  store_keys;
        typename StoreValues::TempStorage store_values;
    } tmp_storage;

    '''
    invalid_key 处理：当实际数据量 < SORT_SIZE 时，
    用 MAX_KEY（升序）或 LOWEST_KEY（降序）填充，
    使 padding 元素始终排到尾部
    '''
    const K invalid_key = [descending] {
        using radix_t = typename cub::Traits<key_t>::UnsignedBits;
        union { K key; radix_t radix; } tmp;
        tmp.radix = descending ? cub::Traits<key_t>::LOWEST_KEY
                               : cub::Traits<key_t>::MAX_KEY;
        return tmp.key;
    }();

    K local_keys[items_per_thread];
    V local_values[items_per_thread];

    '''
    五步流水线：Load → Load → Sort → Store → Store
    每步之间需要 __syncthreads()（因为 union 共享 SMEM）
    '''
    LoadKeys(tmp_storage.load_keys).Load(keys_iter, local_keys, keySliceSize, invalid_key);
    __syncthreads();
    LoadValues(tmp_storage.load_values).Load(values_iter, local_values, keySliceSize, invalid_value);
    __syncthreads();

    if (descending) {
        Sort(tmp_storage.sort).SortDescending(
            reinterpret_cast<key_t (&)[items_per_thread]>(local_keys), local_values);
    } else {
        Sort(tmp_storage.sort).Sort(
            reinterpret_cast<key_t (&)[items_per_thread]>(local_keys), local_values);
    }
    __syncthreads();

    StoreKeys(tmp_storage.store_keys).Store(keys_iter, local_keys, keySliceSize);
    __syncthreads();
    StoreValues(tmp_storage.store_values).Store(values_iter, local_values, keySliceSize);
}
```

关键设计点：

- **BLOCK_LOAD_TRANSPOSE / BLOCK_STORE_TRANSPOSE**：从全局内存以 striped 模式 coalesced 读取，然后在 SMEM 中转置为 blocked 布局。等价于 BlockLoad + BlockExchange::StripedToBlocked
- **reinterpret_cast**：PyTorch 使用 `at::Half` / `at::BFloat16` 等自定义类型，需要 cast 为 CUB 认识的 CUDA 原生类型（`__half` / `__nv_bfloat16`）
- **StridedRandomAccessor**：处理非 contiguous tensor（stride ≠ 1）的排序

### PyTorch ScanUtils 中的 Sklansky 并行 scan

PyTorch 的 `torch.cumsum` / `torch.cumprod` 对 innermost dimension 的实现使用了自定义的 SMEM-based parallel scan，而非直接调用 CUB BlockScan。

**源码位置**: `ATen/native/cuda/ScanUtils.cuh` → `tensor_kernel_scan_innermost_dim_with_indices`

其实现模式是经典的 Blelloch / Sklansky 并行 scan：

1. 每个线程加载 2 个元素到 SMEM
2. **Up-sweep**（reduce phase）：`log2(N)` 步，每步间距翻倍，将 SMEM 中的元素做 pairwise reduction
3. **Down-sweep**（distribute phase）：`log2(N)` 步，每步间距减半，将前缀分发回各个位置

这与 CUB BlockScan 的 RAKING 策略不同——PyTorch 使用的是 work-efficient 的 Blelloch scan（O(N) work），而 CUB 的 WARP_SCANS 策略使用的是 Kogge-Stone scan（O(N log N) work 但更低延迟）。选择取决于场景：

- CUB BlockScan 优化了小规模（一个 block 内）、需要最低延迟的场景
- PyTorch ScanUtils 需要处理 row_size 远大于 block size 的情况（通过多个 tile 级联）

