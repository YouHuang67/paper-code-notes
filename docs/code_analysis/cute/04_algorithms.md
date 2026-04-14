---
tags:
  - CUTLASS
  - CUDA
---

# CuTe Tensor 算法

> **原文出处**: [NVIDIA/cutlass - media/docs/cute/04_algorithms.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/04_algorithms.md)
> **许可证**: BSD-3-Clause, Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES

本节概述在 Tensor 上执行的常见数值算法的接口与实现。

这些算法的实现位于 [`include/cute/algorithm/`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/) 目录。

## `copy`

CuTe 的 `copy` 算法将源 Tensor 的元素拷贝到目标 Tensor 的元素中。`copy` 的各种重载位于 [`include/cute/algorithm/copy.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/copy.hpp)。

### 接口与特化机会

Tensor 在编译时封装了数据类型、数据位置，以及可能的 shape 和 stride。因此，`copy` 可以且确实会根据其参数的类型分发到各种同步或异步硬件拷贝指令。

`copy` 算法有两个主要重载。第一个仅接受源 Tensor 和目标 Tensor：

```c++
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

第二个额外接受一个 `Copy_Atom`：

```c++
template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

两参数重载根据两个 Tensor 参数的类型选择默认实现。`Copy_Atom` 重载让调用者通过指定非默认 `copy` 实现来覆盖默认行为。

### 并行性和同步取决于参数类型

无论是默认实现还是 Copy_Atom 重载选择的实现，都可能使用零到全部可用的并行性，并具有多种同步语义。行为取决于 `copy` 的参数类型，用户需要根据其运行的架构来判断。

`copy` 算法可以是每线程顺序的，也可以跨线程集合（如 block 或 cluster）并行。如果 `copy` 是并行的，参与的线程集合可能需要同步后才能假设拷贝操作已完成。例如，若参与线程构成一个 thread block，用户必须调用 `__syncthreads()` 后才能使用 `copy` 的结果。

`copy` 算法可能使用异步拷贝指令如 `cp.async`。此时用户需要执行底层实现所需的额外同步后才能使用 `copy` 算法的结果。

### 泛型 copy 实现

一个简单的泛型 `copy` 实现示例：

```c++
template <class TA, class ALayout,
          class TB, class BLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<TA, ALayout> const& src,
     Tensor<TB, BLayout>      & dst)
{
  for (int i = 0; i < size(dst); ++i) {
    dst(i) = src(i);
  }
}
```

此泛型算法用一维逻辑坐标访问两个 Tensor，以逻辑列主序遍历。合理的架构无关优化包括：

1. 如果两个 Tensor 具有已知内存空间且有优化访问指令（如 `cp.async`），则分发到自定义指令
2. 如果两个 Tensor 具有静态 layout 且可以证明元素向量化有效（例如四个 `ld.global.b32` 可合并为单个 `ld.global.b128`），则向量化源和目标 tensor
3. 如果可能，验证使用的拷贝指令适合源和目标 tensor

CuTe 的优化 copy 实现可以做到以上所有。

## `copy_if`

`copy_if` 算法与 `copy` 位于同一头文件。它额外接受一个与输入输出形状相同的"谓词 Tensor"。仅当对应谓词 Tensor 元素非零时才拷贝源 Tensor 元素。

关于为何以及如何使用 `copy_if`，请参考[谓词教程](07_predication.md)。

## `gemm`

### gemm 的计算内容

`gemm` 算法接受三个 Tensor A、B 和 C，其行为取决于参数的模式数量。我们用字母表示模式：

- **V** — "向量"，独立元素的模式
- **M** 和 **N** — BLAS GEMM 中结果矩阵 C 的行数和列数
- **K** — GEMM 的"归约模式"，即求和的模式

使用 `(...) x (...) => (...)` 记法列出 A、B 和 C 的模式：

1. `(V) x (V) => (V)`：向量逐元素积 $C_v \mathrel{+}= A_v B_v$。分发到 FMA 或 MMA
2. `(M) x (N) => (M,N)`：向量外积 $C_{mn} \mathrel{+}= A_m B_n$。分发到 (4) 且 V=1
3. `(M,K) x (N,K) => (M,N)`：矩阵乘积 $C_{mn} \mathrel{+}= A_{mk} B_{nk}$。对每个 K 分发到 (2)
4. `(V,M) x (V,N) => (V,M,N)`：批量向量外积 $C_{vmn} \mathrel{+}= A_{vm} B_{vn}$。优化寄存器复用，对每个 M、N 分发到 (1)
5. `(V,M,K) x (V,N,K) => (V,M,N)`：批量矩阵乘积 $C_{vmn} \mathrel{+}= A_{vmk} B_{vnk}$。对每个 K 分发到 (4)

CuTe 的模式排列惯例：K 总是在最右（最外），V 总是在最左（最内）。

### 分发到优化实现

与 `copy` 一样，CuTe 的 `gemm` 实现根据 Tensor 参数类型分发到适当优化的实现。同样接受可选的 `MMA_Atom` 参数让调用者覆盖默认的 `FMA` 指令。

关于 `MMA_Atom` 和 `gemm` 在不同架构上的特化，请参考 [MMA 教程](05_mma_atom.md)。

## `axpby`

`axpby` 算法位于 [`include/cute/algorithm/axpby.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/axpby.hpp)。它将 $y$ 赋值为 $\alpha x + \beta y$ 的结果，其中 $\alpha$ 和 $\beta$ 是标量，$x$ 和 $y$ 是 Tensor。名称代表 "Alpha times X Plus Beta times Y"，是原始 BLAS "AXPY" 例程的推广。

## `fill`

`fill` 算法位于 [`include/cute/algorithm/fill.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/fill.hpp)。它用给定标量值覆写输出 Tensor 的元素。

## `clear`

`clear` 算法位于 [`include/cute/algorithm/clear.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/clear.hpp)。它用零覆写输出 Tensor 的元素。

## 其他算法

CuTe 还提供其他算法，头文件位于 [`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm) 目录。
