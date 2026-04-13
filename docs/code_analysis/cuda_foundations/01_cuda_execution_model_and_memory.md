---
tags:
  - CUDA
---
# CUDA 基础：执行模型与内存访问

本文整理 CUDA kernel 中最常见的执行模型、共享内存、同步原语和 warp 级通信，作为各类 CUDA / CUTLASS / CUB 代码分析的公共引用页。

CUTLASS / CuTe 的模板体系与编程抽象见 [CUDA 基础：CUTLASS/CuTe 编程模型](02_cuda_cutlass_cute_programming_model.md)。

## 信息源范围

本文只保留能在本地 CUDA 运行时头文件、CUTLASS 头文件和已分析 kernel 实现中核实的内容。主要依据包括：

- CUDA 运行时头文件：`vector_types.h`、`device_launch_parameters.h`、`crt/host_defines.h`、`crt/device_functions.h`、`sm_30_intrinsics.h/.hpp`、`sm_32_atomic_functions.h`、`crt/math_functions.h`
- CUTLASS 头文件：`cutlass/detail/helper_macros.hpp`、`cutlass/device_kernel.h`、`cutlass/gemm/kernel/gemm.h`
- 项目实现：PyTorch ATen `mem_eff_attention/kernel_forward.h`

没有在这些本地材料中直接看到的结论，本文会改用保守表述，不把经验判断写成硬性事实。

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

这是一个向编译器提供 launch 配置信息的属性。它会影响寄存器分配和可并发驻留的 block 数，但具体结果仍取决于目标架构、编译器和 kernel 本身。

CUTLASS attention 的配置：

```cpp
static constexpr int kNumThreads = kWarpSize * kNumWarpsPerBlock;  // 如 128
static constexpr int kMinBlocksPerSm =
    getWarpsPerSmFw<scalar_t, ArchTag>() / kNumWarpsPerBlock;      // 如 16/4=4
```

在 CUTLASS attention 这类实现中，`kMinBlocksPerSm` 往往由“期望每个 SM 驻留多少个 warp”反推得到，例如 `16 / kNumWarpsPerBlock` 这样的写法。

### 1.3 `__syncthreads()`

block 级屏障同步。block 内所有线程必须到达同一个 `__syncthreads()` 后才能继续执行。主要用途：

- 确保 Shared Memory 写入对所有线程可见（如 MM0 写 SMEM 后、MM1 读之前）
- 确保 `m_prime` / `s_prime` 等共享状态已更新

注意：在没有分支分歧、且只做 warp 范围内寄存器级协作的场景里，通常不需要 block 级的 `__syncthreads()`；但这不等于任何 warp 内代码都可以忽略同步语义。

### 1.4 Shared Memory

```cpp
extern __shared__ char smem_buffer[];
SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);
```

- **片上 SRAM**，访问范围是 block 内线程共享
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

三个阶段不会同时使用，用 union 让它们占用同一块 SMEM。完整的 SMEM 布局见 [CUTLASS Memory Efficient Attention §4](../cutlass_mem_eff_attention/00_overview.md#4-shared-memory-布局)。

对照 Triton：Triton 的 shared memory 布局通常由编译器负责推导，源码层面较少直接写出这种 union 复用。

### 1.5 浮点原子操作 Trick

在本文查阅的本地 CUDA 运行时头文件中，未看到 `atomicMax(float*)` 这类声明；因此这里分析的 attention kernel 通过整数 bitcast 自行封装了一个 `atomicMaxFloat` helper：

```cpp
static CUTLASS_DEVICE float atomicMaxFloat(float* addr, float value) {
    return !signbit(value)
        ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
        : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}
```

原理：IEEE 754 浮点数的正数部分与整数的大小比较顺序一致（exponent 在高位），因此可以把 float bitcast 为 int 后用 `atomicMax` 比较。负数的位模式是"反序"的，所以用 `atomicMin` + unsigned 处理。

用于 Online Softmax 中跨 warp 更新行最大值 `mi[accum_m]`（见 [CUTLASS Memory Efficient Attention §6](../cutlass_mem_eff_attention/00_overview.md#6-iterative_softmax-详解)）。

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

从本地 CUDA 头文件可以确认，`expf` 和 `exp2f` 都是设备数学函数。本文这里只分析 kernel 为什么主动改写成 `exp2f` 形式，而不对底层具体指令序列做更强断言。

- CUTLASS attention 在 `iterative_softmax` 中预乘 `kLog2e = 1.4426950408889634`，然后全部使用 `exp2f`
- Triton Split-K 同理：`qk_scale = sm_scale * log2e`，后续用 `tl.math.exp2`

两者在数学上等价：$e^x = 2^{x \cdot \log_2 e}$。
