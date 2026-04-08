---
tags:
  - CUTLASS
  - CUDA
---
# CUDA 与 CUTLASS 核心概念

本文汇总 CUTLASS Memory Efficient Attention kernel 中涉及的 CUDA 编程模型和 CUTLASS 模板体系知识，供[主文档](00_overview.md)交叉引用。

## 1. CUDA 执行模型基础

### 1.1 Thread / Warp / Block / Grid

GPU 的线程组织层次：

- **Thread**：最小执行单元，拥有私有寄存器
- **Warp**：32 个线程组成一个 warp，以 SIMT 方式锁步执行同一条指令
- **Block**（ThreadBlock / CTA）：若干 warp 组成一个 block，共享同一块 Shared Memory，可通过 `__syncthreads()` 同步
- **Grid**：所有 block 组成 grid，block 之间无同步原语（除 atomic）

CUTLASS attention kernel 的 grid 配置：

```cpp
dim3 blocks(
    ceil_div(num_queries, kQueriesPerBlock),  // Q 序列分块
    num_heads,                                 // 每个 head 一个 block
    num_batches                                // batch 维度
);
dim3 threads(kWarpSize, kNumWarpsPerBlock, 1);  // x=lane, y=warp
```

block 内每个线程的身份由 `threadIdx.x`（lane_id，0~31）和 `threadIdx.y`（warp_id）决定。

### 1.2 `__launch_bounds__`

```cpp
__global__ void __launch_bounds__(maxThreadsPerBlock, minBlocksPerSm) kernel(...);
```

向编译器声明 kernel 的线程配置约束：

- `maxThreadsPerBlock`：每个 block 的最大线程数
- `minBlocksPerSm`：每个 SM 至少同时驻留的 block 数

编译器据此优化寄存器分配：当 `minBlocksPerSm` 较高时，编译器会限制每个线程的寄存器用量以容纳更多 block（提升 occupancy），必要时将寄存器溢出到 local memory。

CUTLASS attention 的配置：

```cpp
static constexpr int kNumThreads = kWarpSize * kNumWarpsPerBlock;  // 如 128
static constexpr int kMinBlocksPerSm =
    getWarpsPerSmFw<scalar_t, ArchTag>() / kNumWarpsPerBlock;      // 如 16/4=4
```

SM80 + f16 场景目标是每个 SM 驻留 16 个 warp，即 `kMinBlocksPerSm = 16 / kNumWarpsPerBlock`。

### 1.3 `__syncthreads()`

block 级屏障同步。block 内所有线程必须到达同一个 `__syncthreads()` 后才能继续执行。主要用途：

- 确保 Shared Memory 写入对所有线程可见（如 MM0 写 SMEM 后、MM1 读之前）
- 确保 `m_prime` / `s_prime` 等共享状态已更新

注意：warp 内的 32 个线程天然同步（SIMT），不需要 `__syncthreads()`。

### 1.4 Shared Memory

```cpp
extern __shared__ char smem_buffer[];
SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);
```

- **片上 SRAM**，延迟 ~20 cycles（远低于 GMEM ~400 cycles）
- 一个 block 内的所有线程共享，但不同 block 之间独立
- 大小有限（SM80 最多 164KB/SM，通过 `cudaFuncSetAttribute` 配置动态 shared memory）

CUTLASS attention 使用 `union` 复用 shared memory：

```cpp
union {
    typename MM0::Mma::SharedStorage mm0;      // QK^T 的 GEMM 使用
    SharedStorageAfterMM0 after_mm0;           // 包含 si(P矩阵) + MM1 存储
    typename MM1::DefaultEpilogue::SharedStorage epilogue;  // 输出写回
};
```

三个阶段不会同时使用，用 union 让它们占用同一块 SMEM，大幅减少 SMEM 消耗。完整的 SMEM 布局见[主文档 §4](00_overview.md#4-shared-memory-布局)。

对照 Triton：Triton 的 shared memory 由编译器自动管理，程序员无需手动 union 复用。但代价是编译器可能不如手动布局高效，尤其在多阶段流水线场景。

### 1.5 浮点原子操作 Trick

CUDA 不提供原生的 `atomicMaxFloat`，kernel 通过整数 bitcast 实现：

```cpp
static CUTLASS_DEVICE float atomicMaxFloat(float* addr, float value) {
    return !signbit(value)
        ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
        : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}
```

原理：IEEE 754 浮点数的正数部分与整数的大小比较顺序一致（exponent 在高位），因此可以把 float bitcast 为 int 后用 `atomicMax` 比较。负数的位模式是"反序"的，所以用 `atomicMin` + unsigned 处理。

用于 Online Softmax 中跨 warp 更新行最大值 `mi[accum_m]`（见[主文档 §6 iterative_softmax Step 1](00_overview.md#6-iterative_softmax-详解)）。

对照 Triton：Triton 中 `m_i_new = tl.maximum(m_i, tl.max(qk, 1))` 不需要 atomic——每个 program 独立处理一行，而 CUTLASS 中一行由多个 warp 的不同线程持有不同列的 fragment，必须跨 warp 归约。

### 1.6 Warp Shuffle：`__shfl_sync`

```cpp
p.asInt = __shfl_sync(0xffffffff, (unsigned)p.asInt, 0);
```

warp 内的线程可以直接读取其他线程的寄存器值，无需经过 shared memory。参数含义：

- `0xffffffff`：mask，表示 warp 内所有 32 个线程都参与
- `(unsigned)p.asInt`：要广播的值
- `0`：源 lane id（从 lane 0 读）

这里用于 `warp_uniform()`——将一个值从 lane 0 广播到 warp 内所有线程，让编译器知道该值在 warp 内一致，从而启用 warp-uniform 优化（减少分支、合并内存访问）。

### 1.7 `exp2f` vs `expf`

GPU 硬件有原生 `exp2`（以 2 为底的指数）指令，但 `exp`（以 e 为底）通常由编译器实现为 `exp2(x * log2(e))`，多一次乘法。

- CUTLASS attention 在 `iterative_softmax` 中预乘 `kLog2e = 1.4426950408889634`，然后全部使用 `exp2f`
- Triton Split-K 同理：`qk_scale = sm_scale * log2e`，后续用 `tl.math.exp2`

两者在数学上等价：$e^x = 2^{x \cdot \log_2 e}$。

## 2. CUTLASS 模板体系

### 2.1 核心概念

CUTLASS 是 NVIDIA 的 C++ 模板库，将 GEMM 分解为层次化的组件：

```
Device level    →  完整 GEMM 调用（grid launch）
Kernel level    →  threadblock 级 tiling
Threadblock MMA →  一个 block 内的矩阵乘累加
Warp MMA        →  一个 warp 的 Tensor Core 指令
```

每一层都用模板参数组合来特化，编译期确定所有 shape、数据类型、内存布局。

### 2.2 `GemmShape<M, N, K>`

定义 GEMM 在某一层的 tile 尺寸：

```cpp
// ThreadblockShape: 一个 block 处理的 tile
using ThreadblockShape = GemmShape<kQueriesPerBlock, kKeysPerBlock, ThreadK>;
// 如 GemmShape<64, 128, 32>：block 处理 64×128 的输出 tile，K 维度每次 32

// WarpShape: 一个 warp 处理的 tile
using WarpShape = GemmShape<32, 32, WarpK>;
// 如 GemmShape<32, 32, 32>

// InstructionShape: 一条 Tensor Core 指令的 shape
using InstructionShape = GemmShape<16, 8, 8>;
// Sm80 f16: 一条 mma.m16n8k8 指令
```

从 ThreadblockShape 到 InstructionShape 的除法决定了循环次数：`WarpCount = ThreadblockShape / WarpShape`，`Mma iterations = WarpShape / InstructionShape`。

### 2.3 `OpClass` 与 `ArchTag`

**OpClass** 决定使用哪种计算单元：

- `OpClassSimt`：CUDA Core（FMA 指令），任何 GPU
- `OpClassTensorOp`：Tensor Core（MMA 指令），Volta+

**ArchTag** 标记目标架构：

- `cutlass::arch::Sm50`：Maxwell（无 Tensor Core）
- `cutlass::arch::Sm70`：Volta（fp16 Tensor Core，mma.m8n8k4）
- `cutlass::arch::Sm75`：Turing（INT8/INT4 Tensor Core，mma.m16n8k8）
- `cutlass::arch::Sm80`：Ampere（bf16/tf32 Tensor Core，异步 cp.async）

`DefaultGemmType` 根据 `(ArchTag, scalar_t)` 选择 `OpClass`、`InstructionShape` 等参数：

```cpp
// Sm80 + f16/bf16 → TensorOp + GemmShape<16,8,8>
// Sm80 + f32     → TensorOp + GemmShape<16,8,8> + OpMultiplyAddFastF32 (TF32)
// Sm50 + 任意    → Simt + GemmShape<1,1,1>（FMA）
```

### 2.4 MMA 流水线

CUTLASS MMA（Matrix Multiply-Accumulate）的执行流程：

```
Global Memory → [IteratorA/B 加载] → Shared Memory → [WarpIterator 加载] → Registers → MMA 指令 → Accumulator
```

关键组件：

- **IteratorA / IteratorB**：负责从 Global Memory 到 Shared Memory 的 tile 加载，处理边界检查和对齐
- **Mma（ThreadblockMma）**：编排多个 warp 的计算和数据搬运，实现计算-访存流水线
- **多阶段流水线**（`kStages`）：Sm80 上支持 `cp.async` 异步拷贝，可以同时执行当前阶段的计算和下一阶段的数据加载（software pipelining）

Attention kernel 的两个 GEMM：

| | MM0: Q @ K^T | MM1: P @ V |
|--|------------|-----------|
| A 来源 | Q (Global Memory) | P (Shared Memory) |
| B 来源 | K (Global Memory) | V (Global Memory) |
| 输出 | 寄存器 → iterative_softmax → SMEM | 寄存器 → Epilogue → GMEM |

MM1 的特殊之处：A 矩阵（P）从 Shared Memory 读取（`MmaFromSharedMemory`），而非常规的 Global Memory。

### 2.5 Epilogue

Epilogue 负责将 MMA 累加结果从寄存器写回 Global Memory，同时执行后处理操作：

```
Accumulator (寄存器)
  → AccumulatorFragmentIterator (将寄存器切分为 tile)
  → WarpTileIterator (写入 SMEM)
  → SharedLoadIterator (从 SMEM 读回)
  → EpilogueOutputOp (后处理: rescale / normalize / cast)
  → OutputTileIterator (写入 GMEM)
```

在 attention kernel 中，`MemoryEfficientAttentionNormalize` 作为 `EpilogueOutputOp`，执行 Online Softmax 的输出归一化（详见[主文档 §7](00_overview.md#7-epilogue-输出归一化)）。

### 2.6 `kSingleValueIteration` 与输出累加策略

这是 CUTLASS attention 相比 Triton Split-K 能支持大 head_dim 的关键机制（见[主文档 §2 模板参数](00_overview.md#2-模板参数与-kernel-变体)）。

当 `head_dim ≤ kKeysPerBlock`（如 64 或 128）时，`kSingleValueIteration = true`：

- V 矩阵在 dim 维度上一次迭代就能处理完
- 输出 `accum_o` 常驻寄存器（`kKeepOutputInRF = true`），整个 KV 循环只在最后一次写回 GMEM
- 避免了中间结果的 GMEM 读写

当 `head_dim > kKeysPerBlock`（如 kMaxK=65536）时：

- V 矩阵需要多次迭代处理 dim 维度
- 每次 KV 迭代的中间结果必须经 Epilogue 写回 GMEM buffer（`output_accum_ptr`）
- 下一次迭代读回 buffer 做 rescale 累加
- 最后一次迭代才做 `1/s_prime` 归一化并写入最终输出
