---
tags:
  - CUTLASS
  - CUDA
---

# CuTe TMA Tensor

> **原文出处**: [NVIDIA/cutlass - media/docs/cute/0z_tma_tensors.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0z_tma_tensors.md)
> **许可证**: BSD-3-Clause, Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES

你可能会遇到打印为如下形式的 CuTe Tensor：

```
ArithTuple(0,_0,_0,_0) o ((_128,_64),2,3,1):((_1@0,_1@1),_64@1,_1@2,_1@3)
```

什么是 `ArithTuple`？那些 tensor stride 是什么？这是做什么用的？

本文回答这些问题，介绍 CuTe 的一些高级特性。

## TMA 指令简介

Tensor Memory Accelerator（TMA）是一组在全局内存和共享内存之间拷贝可能多维数组的指令，由 Hopper 架构引入。单条 TMA 指令可以一次性拷贝整个数据 tile，硬件不再需要计算单个内存地址并为 tile 的每个元素发出单独的拷贝指令。

TMA 指令接收一个 **TMA 描述符**（TMA descriptor），它是全局内存中 1 到 5 维 tensor 的紧凑表示，包含：

- tensor 的基地址指针
- 元素数据类型
- 每个维度的大小和步长
- 其他标志（smem box 大小、smem swizzle 模式、越界访问行为）

描述符必须在 kernel 执行前在主机端创建，被所有发出 TMA 指令的 thread block 共享。kernel 内部 TMA 执行时接收：

- TMA 描述符指针
- SMEM 指针
- GMEM tensor 在 TMA 描述符视图中的坐标

关键观察：TMA 指令**不直接消费全局内存指针**。全局内存指针包含在描述符中且被视为常量，不是 TMA 指令的单独参数。TMA 消费的是进入 TMA 全局内存视图的 **TMA 坐标**。

因此，存储 GMEM 指针并计算偏移/新 GMEM 指针的普通 CuTe Tensor 对 TMA 没用。

## 构建 TMA Tensor

### 隐式 CuTe Tensor

所有 CuTe Tensor 都是 Layout 和 Iterator 的组合。普通全局内存 tensor 的迭代器是全局内存指针。但 CuTe Tensor 的迭代器不一定是指针——可以是任何随机访问迭代器。

**计数迭代器**（counting iterator）就是一例——表示从某个值开始的可能无限整数序列。序列不显式存储在内存中，迭代器只存储当前值。

```cpp
Tensor A = make_tensor(counting_iterator<int>(42), make_shape(4,5));
```

输出：

```
counting_iter(42) o (4,5):(_1,4):
   42   46   50   54   58
   43   47   51   55   59
   44   48   52   56   60
   45   49   53   57   61
```

此 tensor 将逻辑坐标映射到即时计算的整数。因为仍是 CuTe Tensor，仍可像普通 tensor 一样被分块、分区和切片。

但 TMA 消费的不是指针或整数，而是**坐标**。能否创建一个隐式 TMA 坐标的 tensor？

### ArithTupleIterator 和 ArithTuple

构建一个类似计数迭代器的 TMA 坐标迭代器，支持：

- 解引用得到 TMA 坐标
- 按另一个 TMA 坐标偏移

`ArithmeticTupleIterator` 存储一个坐标（整数 tuple），表示为 `ArithmeticTuple`——一个 `cute::tuple` 的子类，重载了 `operator+` 使两个 tuple 相加为各元素之和。

```cpp
ArithmeticTupleIterator citer_1 = make_inttuple_iter(42, Int<2>{}, Int<7>{});
ArithmeticTupleIterator citer_2 = citer_1 + make_tuple(Int<0>{}, 5, Int<2>{});
print(*citer_2);   // (42,7,_9)
```

TMA Tensor 使用此类迭代器存储当前 TMA 坐标"偏移"（不是普通的一维数组偏移或指针）。

总结：为*整个全局内存 tensor* 创建一个 TMA 描述符。TMA 描述符定义该 tensor 的视图，指令接受该视图中的 TMA 坐标。为了生成和追踪这些 TMA 坐标，我们定义一个隐式 CuTe Tensor（TMA 坐标的 tensor），可以与普通 CuTe Tensor 完全相同的方式被分块、切片和分区。

### Stride 不只是整数

普通 tensor 的 layout 将逻辑坐标 `(i,j)` 映射为一维线性索引 `k`——坐标与步长的内积。

TMA Tensor 的迭代器持有 TMA 坐标。因此 TMA Tensor 的 Layout 必须将逻辑坐标映射为 TMA 坐标而非一维索引。

为此，可以抽象化步长的含义。步长不必是整数，而是任何支持与整数（逻辑坐标）做内积的代数对象——即**整数模**（integer-module）。

#### 基向量元素

CuTe 的基向量元素位于 `cute/numeric/arithmetic_tuple.hpp`。使用 `E` 类型别名创建可作为步长的归一化基向量元素：

| C++ 对象 | 描述 | 字符串表示 |
| --- | --- | --- |
| `E<>{}` | `1` | `1` |
| `E<0>{}` | `(1,0,...)` | `1@0` |
| `E<1>{}` | `(0,1,0,...)` | `1@1` |
| `E<0,0>{}` | `((1,0,...),0,...)` | `1@0@0` |
| `E<1,0>{}` | `(0,(1,0,...),0,...)` | `1@0@1` |

基向量元素可以嵌套、缩放和相加。例如 `5*E<1>{}` 打印为 `5@1`，意为 `(0,5,0,...)`。

#### 步长的线性组合

步长为 `(1@0, 1@1)` 时，坐标 `(i,j)` 与步长的内积为 `i@0 + j@1 = (i,j)`——得到 TMA 坐标 `(i,j)`。若要反转坐标，用 `(1@1, 1@0)` 得到 `i@1 + j@0 = (j,i)`。

### TMA Tensor 应用

```cpp
Tensor a = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<0>{}, E<1>{}));
print_tensor(a);

Tensor b = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<1>{}, E<0>{}));
print_tensor(b);
```

输出：

```
ArithTuple(0,0) o (4,5):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)

ArithTuple(0,0) o (4,5):(_1@1,_1@0):
  (0,0)  (1,0)  (2,0)  (3,0)  (4,0)
  (0,1)  (1,1)  (2,1)  (3,1)  (4,1)
  (0,2)  (1,2)  (2,2)  (3,2)  (4,2)
  (0,3)  (1,3)  (2,3)  (3,3)  (4,3)
```

tensor `a` 的 mode-0 映射到 TMA 坐标 0，mode-1 映射到 TMA 坐标 1。tensor `b` 则反转。这些 TMA Tensor 可以像普通 Tensor 一样被分块、切片和分区，始终产生正确的 TMA 坐标。
