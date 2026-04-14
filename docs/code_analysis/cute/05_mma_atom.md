---
tags:
  - CUTLASS
  - CUDA
---

# CuTe MMA 指令支持

> **原文出处**: [NVIDIA/cutlass - media/docs/cute/0t_mma_atom.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0t_mma_atom.md)
> **许可证**: BSD-3-Clause, Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES

本文详细解释 CuTe 如何支持 GPU 的矩阵乘累加（Matrix Multiply-Accumulate, MMA）硬件指令。

MMA 是架构特定的。不同代的 GPU 架构引入不同的 MMA 指令集。然而，CuTe 的 `Layout` 等功能使得在通用 CUDA C++ 代码中暴露 MMA 成为可能。我们通过多个步骤实现：

1. 将每个 MMA 的 PTX 指令封装在一个 **Operation** 结构体中
2. 为每个 Operation 结构体定义一个 **Traits** 结构体，包含使用该 Operation 所需的所有元信息
3. **Atom** = Operation + Traits，提供构造 `cute::Tensor` "fragment" 和使用该 Operation 操作现有 Tensor 的方法
4. **TiledMMA** 组合多个 Atom，提供构建更复杂分区模式的工具——创建 Atom 的 layout 和交错排列

## CuTe MMA Atom

CuTe 将每个 MMA 暴露为一对结构体：Operation 结构体和以 Operation 类型为模板参数的 `MMA_Traits` 结构体。

### Operation 结构体

**位置**: [`include/cute/arch`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch) 目录中以 `mma` 开头的头文件。

**命名**主要编码其封装的 PTX 指令，通常包括：首个支持架构、M/N/K 维度、操作类型、A/B 输入的排列方式。

例如 `SM70_8x8x4_F32F16F16F32_NT`（定义在 [`include/cute/arch/mma_sm70.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch/mma_sm70.hpp)）：

- "SM70" — Volta 架构
- "8x8x4" — M=8, N=8, K=4，quadpair 执行的 MMA 维度
- "F32F16F16F32" — D/A/B/C 四个矩阵操作数的元素类型（MMA 计算 D = C + A*B）
- "NT" — A 为 M-major（非转置，列主序），B 为 N-major（转置，行主序）

**内容**——四个公共类型别名 `DRegisters`、`ARegisters`、`BRegisters`、`CRegisters`，以及一个 `static void fma` 成员函数。例如：

```c++
using DRegisters = float[8];
using ARegisters = uint32_t[2];
using BRegisters = uint32_t[2];
using CRegisters = float[8];
```

这表明每个线程为 C 和 D 传入 8 个 F32 值，为 A 和 B 各传入 4 个 F16 值（两个 16 位 F16 值打包在每个 32 位 `uint32_t` 中）。

### Traits 结构体

**位置**: [`include/cute/atom`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom) 目录中以 `mma_traits` 开头的头文件。

`MMA_Traits` 特化定义以下公共类型别名：

- `ValTypeD`/`ValTypeA`/`ValTypeB`/`ValTypeC`：D/A/B/C 矩阵的逻辑计算类型
- `Shape_MNK`：MMA 操作的逻辑 MxNxK shape
- `ThrID`：单个 MMA 操作内的逻辑线程映射
- `ALayout`：(thread, value) 对到 MxK A 矩阵坐标的映射
- `BLayout`：(thread, value) 对到 NxK B 矩阵坐标的映射
- `CLayout`：(thread, value) 对到 MxN C 矩阵坐标的映射

示例（`SM70_8x8x4_F32F16F16F32_NT`）：

```c++
template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Col;
  using BLayout = SM70_8x4_Col;
  using CLayout = SM70_8x8_32b;
};
```

## Volta

Volta 架构实现 HMMA 指令，其中 8 个线程组成的 quadpair（QP）协作执行 8x8x4 矩阵乘累加（因为 warp 有 32 个线程，它会跨 4 个 QP 执行 16x16x4 的 MMA）。

### 线程 ID

32 线程 warp 中逻辑索引为 [0...31]，HMMA 使用线程 [0,1,2,3]∪[16,17,18,19]（第 0 个 quadpair）。线程映射 layout 将 8 个逻辑线程 id [0,8) 映射到 quadpair 线程索引 [0,4)∪[16,20)：

```cpp
// (逻辑线程 id) -> (线程索引)
using ThrID = Layout<Shape <_4, _2>,
                     Stride<_1,_16>>;
```

### 累加器映射

CLayout 构造 `(logical_thr_id, logical_val_id)` 到 C 矩阵 `(m,n)` 坐标的映射。对于 F32 累加器：

```cpp
// (T8,V8) -> (m,n)
using CLayout = Layout<Shape <Shape <_2, _2,_2>, Shape <_2,_2, _2>>,
                       Stride<Stride<_1,_16,_4>, Stride<_8,_2,_32>>>;
```

对于 F16 累加器（更简单，每行由单个线程持有）：

```cpp
using CLayout = Layout<Shape <_8,_8>,
                       Stride<_1,_8>>;
```

### A 和 B 矩阵 Layout 映射

A 和 B 的 layout 取决于源是否转置。以 TN 的 A 矩阵为例——8 个逻辑线程各持有 4 个元素，M 方向步长 1，K 方向步长 8：

```cpp
// (T8,V4) -> (m,k)
using ALayout = Layout<Shape <_8,_4>,
                       Stride<_1,_8>>;
```

NT 情况下 A 矩阵 layout 更复杂：

```cpp
// (T8,V4) -> (m,k)
using ALayout = Layout<Shape <Shape <_4,_2>,_4>,
                       Stride<Stride<_8,_4>,_1>>;
```

## Hopper

Hopper 引入了 GMMA（Group MMA）操作，在 128 线程（4 个 warp 的 warpgroup）粒度上运行。

### 线程 ID

```cpp
using ThrID = Layout<_128, _1>;
```

### 累加器映射

GMMA 的累加器从 core matrix 概念出发层次化构建。对于 fp16 累加器，64x8 的 CLayout：

```cpp
// (T128,V4) -> (M64,N8)
using CLayout = Layout<Shape <Shape <  _4, _8,  _4>, Shape < _2, _2>>,
                       Stride<Stride<_128, _1, _16>, Stride<_64, _8>>>;
```

64x128 的 CLayout（16 份 64x8 tile 的复制）：

```cpp
// (T128,V64) -> (M64,N128)
using CLayout = Layout<Shape <Shape <  _4, _8,  _4>, Shape < _2, _2,  _16>>,
                       Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>>;
```

### A 和 B Layout 映射

从共享内存直接消费 A 和 B 的 GMMA atom 比较特殊——GMMA Descriptor 构建在整个 A/B 数据 tile 上，而非按线程分区。所有线程映射到同一个 `(m,k)=(0,0)=0` 元素，值保持不变：

```cpp
// (T128,V64x16) -> (M64,K16)
using ALayout = Layout<Shape <_128, Shape <_64,_16>>,
                       Stride<  _0, Stride< _1,_64>>>;
```

## TiledMMA

通过组合和交错多个 atom 构建更复杂的模式。

从单个 `SM70_8x8x4_F32F16F16F32_NT` atom 开始：

```cpp
MMA_Atom mma = MMA_Atom<SM70_8x8x4_F32F16F16F32_NT>{};
```

创建类 WMMA 的 16x16x4 操作——使用四个 quadpair MMA：

```cpp
TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                              Layout<Shape <_2,_2>,
                                     Stride<_2,_1>>{});   // 2x2 n-major atom layout
```

将 tile 大小扩展到 32x32x4：

```cpp
TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                              Layout<Shape <_2,_2>,
                                     Stride<_2,_1>>{},    // 2x2 n-major atom layout
                              Tile<_32,_32,_4>{});         // 32x32x4 tiler
```

这通过跨值（而非跨线程）复制 TiledMMA 来扩展 tile 大小。

还可以对 M 模式应用排列使线程的 A 矩阵访问在 m 坐标上连续——在设计共享内存或寄存器 layout 时很方便：

```cpp
TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                              Layout<Shape <_2,_2>,
                                     Stride<_2,_1>>{},
                              Tile<Layout<Shape <_4,_4,_2>,
                                          Stride<_1,_8,_4>>,  // M 上的排列，大小 32
                                   _32,                        // N 上的排列，大小 32 恒等
                                   _4>{});                     // K 上的排列，大小 4 恒等
```

关于如何使用 TiledMMA 分区数据 tensor，参见 [GEMM 教程](06_gemm_tutorial.md)。
