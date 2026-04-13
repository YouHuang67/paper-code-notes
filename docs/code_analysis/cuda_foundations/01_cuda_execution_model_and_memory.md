---
tags:
  - CUDA
---
# CUDA 执行模型与内存访问

本文整理 CUDA kernel 中最常见的执行模型、共享内存、同步原语和 warp 级通信，作为各类 CUDA / CUTLASS / CUB 代码分析的公共引用页。

CUTLASS / CuTe 的模板体系与编程抽象见 [CUDA CUTLASS/CuTe 编程模型](02_cuda_cutlass_cute_programming_model.md)。

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
