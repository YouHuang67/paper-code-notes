---
tags:
  - CUTLASS
  - CUDA
---

# CuTe 快速入门

> **原文出处**: [NVIDIA/cutlass - media/docs/cute/00_quickstart.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
> **许可证**: BSD-3-Clause, Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES

CuTe 是一组 C++ CUDA 模板抽象的集合，用于定义和操作**层次化多维**的线程与数据布局。CuTe 提供 `Layout` 和 `Tensor` 对象，将数据的类型、形状、内存空间和布局紧凑地打包在一起，同时为用户完成复杂的索引计算。这使得程序员可以专注于算法的逻辑描述，而由 CuTe 处理机械化的簿记工作。借助这些工具，我们可以快速设计、实现和修改所有稠密线性代数运算。

CuTe 的核心抽象是**层次化多维布局**（hierarchically multidimensional layouts），它可以与数据数组组合来表示张量。布局的表示能力足以涵盖我们在实现高效稠密线性代数时所需的几乎所有内容。布局还可以通过函数组合（functional composition）进行组合与变换，在此基础上我们构建了大量常用操作，如分块（tiling）和分区（partitioning）。

## 系统要求

CuTe 与 CUTLASS 3.x 共享相同的软件要求，包括 NVCC 和支持 C++17 的主机编译器。

## 前置知识

CuTe 是一个纯头文件（header-only）的 CUDA C++ 库，要求 C++17（2017 年发布的 C++ 标准修订版）。

本教程假设读者具备中级 C++ 经验。例如，读者需要知道如何读写模板函数和模板类，以及如何使用 `auto` 关键字推导函数返回类型。

同时假设读者具备中级 CUDA 经验。例如，读者必须知道设备代码（device code）与主机代码（host code）的区别，以及如何启动 kernel。

## 构建测试与示例

CuTe 的测试和示例作为 CUTLASS 正常构建流程的一部分进行编译和运行。

- 单元测试位于 [`test/unit/cute`](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute) 子目录
- 示例位于 [`examples/cute`](https://github.com/NVIDIA/cutlass/tree/main/examples/cute) 子目录

## 库组织结构

CuTe 是纯头文件 C++ 库，无需构建源代码。库头文件包含在顶层 [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) 目录中，库的各组件按语义分组到不同目录。

| 目录 | 内容 |
|------|------|
| [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) | 顶层每个头文件对应 CuTe 的一个基本构建块，如 [`Layout`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp) 和 [`Tensor`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/tensor.hpp) |
| [`include/cute/container`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container) | 类 STL 对象的实现，如 tuple、array 和 aligned array |
| [`include/cute/numeric`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric) | 基础数值数据类型，包括非标准浮点类型、非标准整数类型、复数和整数序列 |
| [`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm) | 实用算法实现，如 copy、fill 和 clear，可在支持时自动利用架构特定功能 |
| [`include/cute/arch`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch) | 架构特定的矩阵乘法和拷贝指令的封装 |
| [`include/cute/atom`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom) | `arch` 中指令的元信息，以及分区和分块等工具 |

## 教程目录

本教程系列包含以下文档，翻译自 [CUTLASS 官方 CuTe 教程](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute/)：

- [00 快速入门](00_quickstart.md) — 本文，CuTe 概述与入门指引
- [01 Layout](01_layout.md) — `Layout`，CuTe 的核心抽象
- [02 Layout 代数](02_layout_algebra.md) — 高级 `Layout` 运算与 CuTe Layout 代数
- [03 Tensor](03_tensor.md) — `Tensor`，将 `Layout` 与数据组合的多维数组抽象
- [04 算法](04_algorithms.md) — 操作 `Tensor` 的泛型算法概要
- [05 MMA Atom](05_mma_atom.md) — GPU 架构特定 MMA 指令的元信息与接口
- [06 GEMM 教程](06_gemm_tutorial.md) — 使用 CuTe 从零构建 GEMM
- [07 谓词](07_predication.md) — 分块不整除时的处理方法
- [08 TMA Tensor](08_tma_tensors.md) — CuTe 用于支持 TMA 加载/存储的高级 `Tensor` 类型

## 调试技巧

### 如何在主机或设备端打印 CuTe 对象？

`cute::print` 函数为几乎所有 CuTe 类型提供了重载，包括 Pointer、Integer、Stride、Shape、Layout 和 Tensor。遇到不确定的类型时，尝试对其调用 `print`。

CuTe 的打印函数在主机和设备端均可使用。注意在设备端打印开销很大。即使只是在设备端保留打印代码（即使实际不会执行，例如放在运行时不会进入的 `if` 分支中），也可能导致生成更慢的代码。因此，调试完成后务必移除设备端的打印代码。

你可能只想在每个 threadblock 的线程 0 或 grid 的 threadblock 0 上打印。`thread0()` 函数仅在 kernel 的全局线程 0（即 threadblock 0 的线程 0）上返回 true。一个常用的惯例是仅在全局线程 0 上打印 CuTe 对象：

```c++
if (thread0()) {
  print(some_cute_object);
}
```

某些算法依赖特定的线程或 threadblock，因此可能需要在非零线程或 threadblock 上打印。头文件 [`cute/util/debug.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/util/debug.hpp) 中提供了 `bool thread(int tid, int bid)` 函数，当运行在线程 `tid` 且 threadblock `bid` 上时返回 `true`。

### 其他输出格式

部分 CuTe 类型提供了使用不同输出格式的专用打印函数：

- `cute::print_layout`：以纯文本表格显示任意 rank-2 layout，便于可视化坐标到索引的映射
- `cute::print_tensor`：以纯文本多维表格显示 rank-1 到 rank-4 的 tensor 值，用于验证数据拷贝后的正确性
- `cute::print_latex`：输出可通过 `pdflatex` 编译的 LaTeX 命令，生成格式化的彩色表格。适用于 `Layout`、`TiledCopy` 和 `TiledMMA`，对理解 CuTe 内部的布局模式和分区模式非常有用
