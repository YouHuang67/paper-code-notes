---
tags:
  - CUTLASS
  - CUDA
---

# CuTe Tensor

> **原文出处**: [NVIDIA/cutlass - media/docs/cute/03_tensor.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/03_tensor.md)
> **许可证**: BSD-3-Clause, Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES

本文描述 `Tensor`——CuTe 的核心容器，部署了前面描述的 Layout 概念。

从根本上说，`Tensor` 表示一个多维数组。Tensor 抽象掉数组元素如何组织和存储的细节，使用户可以编写通用的多维数组访问算法，并可能针对 Tensor 的特征进行特化。例如，可以对 Tensor 的 rank 进行分发，检查数据的 Layout，以及验证数据的类型。

Tensor 由两个模板参数表示：`Engine` 和 `Layout`。关于 Layout 请参考 [Layout 部分](01_layout.md)。Tensor 提供与 Layout 相同的 shape 和访问运算符，并使用 Layout 计算的结果来偏移和解引用 Engine 持有的随机访问迭代器。即数据布局由 Layout 提供，实际数据由迭代器提供。数据可以驻留在任何类型的内存中——全局内存、共享内存、寄存器内存——甚至可以被即时变换或生成。

## 基本操作

CuTe Tensor 提供类容器的元素访问操作：

- `.data()`：此 Tensor 持有的迭代器
- `.size()`：此 Tensor 的总逻辑大小
- `.operator[](Coord)`：访问逻辑坐标 Coord 对应的元素
- `.operator()(Coord)`：访问逻辑坐标 Coord 对应的元素
- `.operator()(Coords...)`：访问逻辑坐标 `make_coord(Coords...)` 对应的元素

CuTe Tensor 提供与 Layout 类似的层次化操作：

- `rank<I...>(Tensor)`：Tensor 第 I... 个模式的 rank
- `depth<I...>(Tensor)`：Tensor 第 I... 个模式的 depth
- `shape<I...>(Tensor)`：Tensor 第 I... 个模式的 shape
- `size<I...>(Tensor)`：Tensor 第 I... 个模式的 size
- `layout<I...>(Tensor)`：Tensor 第 I... 个模式的 layout
- `tensor<I...>(Tensor)`：Tensor 第 I... 个模式对应的子 tensor

## Tensor Engine

`Engine` 概念是迭代器或数据数组的封装，使用精简的 `std::array` 接口来呈现迭代器：

```c++
using iterator     =  // 迭代器类型
using value_type   =  // 迭代器 value-type
using reference    =  // 迭代器 reference-type
iterator begin()      // 迭代器
```

通常用户不需要自己构造 Engine。当构造 Tensor 时，会自动构造合适的 engine——通常是 `ArrayEngine<T,N>`、`ViewEngine<Iter>` 或 `ConstViewEngine<Iter>`。

### 带标签的迭代器

任何随机访问迭代器都可以用于构造 Tensor，但用户还可以给任何迭代器"打标签"标记内存空间——例如标记此迭代器访问的是全局内存或共享内存。通过 `make_gmem_ptr(g)` 或 `make_gmem_ptr<T>(g)` 标记为全局内存迭代器，`make_smem_ptr(s)` 或 `make_smem_ptr<T>(s)` 标记为共享内存迭代器。

标记内存使得 CuTe 的 Tensor 算法可以针对特定类型的内存使用最快的实现。在使用 Tensor 执行特定操作时，还允许这些运算符验证标签与预期是否匹配。例如某些优化的拷贝操作要求源是全局内存、目标是共享内存。标记使得 CuTe 可以分发到这些拷贝操作和/或验证这些拷贝操作。

## Tensor 创建

Tensor 可以构造为**持有型**（owning）或**非持有型**（nonowning）。

**持有型** Tensor 行为类似 `std::array`。拷贝 Tensor 时会深拷贝其元素，Tensor 析构时释放元素数组。

**非持有型** Tensor 行为类似裸指针。拷贝 Tensor 不拷贝元素，销毁 Tensor 不释放元素数组。

### 非持有型 Tensor

Tensor 通常是现有内存的非持有型视图。通过 `make_tensor` 传入随机访问迭代器和 Layout（或构造 Layout 的参数）来创建：

```cpp
float* A = ...;

// 无标签指针
Tensor tensor_8   = make_tensor(A, make_layout(Int<8>{}));  // 用 Layout 构造
Tensor tensor_8s  = make_tensor(A, Int<8>{});               // 用 Shape 构造
Tensor tensor_8d2 = make_tensor(A, 8, 2);                   // 用 Shape 和 Stride 构造

// 全局内存（静态或动态 layout）
Tensor gmem_8s     = make_tensor(make_gmem_ptr(A), Int<8>{});
Tensor gmem_8d     = make_tensor(make_gmem_ptr(A), 8);
Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{},16));
Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape (      8  ,Int<16>{}),
                                                   make_stride(Int<16>{},Int< 1>{}));

// 共享内存（静态或动态 layout）
Layout smem_layout = make_layout(make_shape(Int<4>{},Int<8>{}));
__shared__ float smem[decltype(cosize(smem_layout))::value];   // 仅静态分配
Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
Tensor smem_4x8_row = make_tensor(make_smem_ptr(smem), shape(smem_layout), LayoutRight{});
```

### 持有型 Tensor

Tensor 也可以是持有内存的数组。通过 `make_tensor<T>` 创建（T 为元素类型），传入 Layout 或构造 Layout 的参数。数组分配类似于 `std::array<T,N>`，因此持有型 Tensor 必须使用具有静态 shape 和静态 stride 的 Layout。CuTe 在 Tensor 中不进行动态内存分配，因为这在 CUDA kernel 中不常见也不高效。

```c++
// 寄存器内存（仅静态 layout）
Tensor rmem_4x8_col = make_tensor<float>(Shape<_4,_8>{});
Tensor rmem_4x8_row = make_tensor<float>(Shape<_4,_8>{}, LayoutRight{});
Tensor rmem_4x8_pad = make_tensor<float>(Shape <_4, _8>{}, Stride<_32,_2>{});
Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);
```

`make_tensor_like` 函数创建一个与输入 Tensor 具有相同 value type 和 shape 的寄存器内存持有型 Tensor，并尽量使用相同的步长顺序。

## 访问 Tensor

用户通过 `operator()` 和 `operator[]` 使用逻辑坐标的 IntTuple 来访问 Tensor 元素。

Tensor 访问时使用其 Layout 将逻辑坐标映射为迭代器可访问的偏移。`operator[]` 的实现：

```c++
template <class Coord>
decltype(auto) operator[](Coord const& coord) {
  return data()[layout()(coord)];
}
```

示例——使用自然坐标、可变参数 `operator()` 和容器式 `operator[]`：

```c++
Tensor A = make_tensor<float>(Shape <Shape < _4,_5>,Int<13>>{},
                              Stride<Stride<_12,_1>,    _64>{});
float* b_ptr = ...;
Tensor B = make_tensor(b_ptr, make_shape(13, 20));

// 通过自然坐标 op[] 填充 A
for (int m0 = 0; m0 < size<0,0>(A); ++m0)
  for (int m1 = 0; m1 < size<0,1>(A); ++m1)
    for (int n = 0; n < size<1>(A); ++n)
      A[make_coord(make_coord(m0,m1),n)] = n + 2 * m0;

// 使用可变参数 op() 将 A 转置到 B
for (int m = 0; m < size<0>(A); ++m)
  for (int n = 0; n < size<1>(A); ++n)
    B(n,m) = A(m,n);

// 像数组一样将 B 拷贝到 A
for (int i = 0; i < A.size(); ++i)
  A[i] = B[i];
```

## 分块 Tensor

很多 [Layout 代数操作](02_layout_algebra.md)也可以应用于 Tensor：

```cpp
   composition(Tensor, Tiler)
logical_divide(Tensor, Tiler)
 zipped_divide(Tensor, Tiler)
  tiled_divide(Tensor, Tiler)
   flat_divide(Tensor, Tiler)
```

上述操作允许从 Tensor 中"提取"任意子 tensor。这在 threadgroup 分块、MMA 分块和为线程重排数据 tile 时非常常用。

注意 `_product` 操作未在 Tensor 上实现，因为它们通常会产生增大陪域的 layout，意味着 Tensor 会访问远超其之前边界的元素。Layout 可以用于乘积，但 Tensor 不行。

## 切片 Tensor

访问 Tensor 时传入坐标返回一个元素，而切片 Tensor 返回被切模式中所有元素的子 tensor。

切片通过与元素访问相同的 `operator()` 执行。传入 `_`（下划线字符，`cute::Underscore` 类型的实例）与 Fortran 或 Matlab 中的 `:`（冒号）效果相同：保留 tensor 的该模式，就好像未使用坐标。

切片 tensor 执行两个操作：

- Layout 对部分坐标求值，结果偏移累积到迭代器——新迭代器指向新 tensor 的起始位置
- 与 `_` 元素对应的 Layout 模式用于构造新 layout

新迭代器和新 layout 一起构造新 tensor。

```cpp
// ((_3,2),(2,_5,_2)):((4,1),(_2,13,100))
Tensor A = make_tensor(ptr, make_shape (make_shape (Int<3>{},2), make_shape (       2,Int<5>{},Int<2>{})),
                            make_stride(make_stride(       4,1), make_stride(Int<2>{},      13,     100)));

// ((2,_5,_2)):((_2,13,100))
Tensor B = A(2,_);

// ((_3,_2)):((4,1))
Tensor C = A(_,5);

// (_3,2):(4,1)
Tensor D = A(make_coord(_,_),5);

// (_3,_5):(4,13)
Tensor E = A(make_coord(_,1),make_coord(0,_,1));

// (2,2,_2):(1,_2,100)
Tensor F = A(make_coord(2,_),make_coord(_,3,_));
```

注意 tensor C 和 D 包含相同元素，但由于使用 `_` 与 `make_coord(_,_)` 的区别，它们具有不同的 rank 和 shape。每种情况下，结果的 rank 等于切片坐标中 Underscore 的数量。

## 分区 Tensor

为实现通用的 Tensor 分区，我们应用 composition 或分块后接切片。这可以多种方式执行，但有三种特别有用：内分区（inner-partitioning）、外分区（outer-partitioning）和 TV-layout 分区。

### 内分区与外分区

以分块示例展示：

```cpp
Tensor A = make_tensor(ptr, make_shape(8,24));  // (8,24)
auto tiler = Shape<_4,_8>{};                    // (_4,_8)

Tensor tiled_a = zipped_divide(A, tiler);       // ((_4,_8),(2,3))
```

假设要给每个 threadgroup 一个 4x8 数据 tile，用 threadgroup 坐标索引第二个模式：

```cpp
Tensor cta_a = tiled_a(make_coord(_,_), make_coord(blockIdx.x, blockIdx.y));  // (_4,_8)
```

这称为**内分区**——保留内部"tile"模式。这种"应用 tiler + 通过索引 rest 模式切出 tile"的模式已封装为 `inner_partition(Tensor, Tiler, Coord)`，也称 `local_tile(Tensor, Tiler, Coord)`。`local_tile` 分区器常用于 threadgroup 级别将 tensor 按 tile 分配给 threadgroup。

**`local_tile` 的维度变换机制（shape division）**

`local_tile` 的核心是**形状除法**：将输入 tensor 的每个维度按 tiler 对应维度整除，产生"块内 + 块外"两层坐标。以 Flash Attention 中 K tensor 的处理为例：

```
原始 mK: (SeqLen_K, HeadDim, NumHeads_K)  — 3D，来自 make_tensor
```

**第一步：预切片降维。** 用 head 索引切出单个 head，3D → 2D：

```
mK(_, bidh/ratio, _) → 2D: (SeqLen_K, HeadDim)
```

**第二步：`local_tile` 做形状除法。** tiler 为 `Shape<kBlockN, kHeadDim>`，将 2D 的每个维度整除：

```
SeqLen_K ÷ kBlockN  → 块内 kBlockN × 块外 Num_Blocks_N
HeadDim  ÷ kHeadDim → 块内 kHeadDim × 块外 1（整除，只有 1 个块）
```

逻辑上产生 4 个维度：`(kBlockN, kHeadDim, Num_Blocks_N, 1)`，即 `zipped_divide` 的结果 `((kBlockN, kHeadDim), (Num_Blocks_N, 1))`。

**第三步：`make_coord(_, 0)` 选择/折叠块外维度。** `_`（Underscore）保留该维度，字面量 `0` 折叠该维度（取第 0 个块）：

```cpp
local_tile(mK_2d, tiler, make_coord(_, 0))
//                              ^    ^
//              保留 Num_Blocks_N    折叠 HeadDim 的块外维度 (只有 1 块，取第 0 个)
```

最终结果为 3D：`(kBlockN, kHeadDim, Num_Blocks_N)`——块内两个维度 + 保留的一个块外维度。后续用 `n_block` 索引第三个维度即可取出某个 K/V 块。实际应用参见 [Flash Attention V2 前向核心](../flash_attention_v2/02_forward_kernel.md) 中 Global Memory Tensor 构建部分。

或者，假设有 32 个线程，要给每个线程分配这些 4x8 tile 中的一个元素，用线程索引第一个模式：

```cpp
Tensor thr_a = tiled_a(threadIdx.x, make_coord(_,_)); // (2,3)
```

这称为**外分区**——保留外部"rest"模式。封装为 `outer_partition(Tensor, Tiler, Coord)`，也有 `local_partition(Tensor, Layout, Idx)` 的变体。

### Thread-Value 分区

另一种常用的分区策略叫 Thread-Value 分区。在此模式中，构造一个 Layout 表示所有线程和每个线程接收的所有值到目标数据坐标的映射。用 `composition` 将目标数据 layout 按 TV-layout 变换，然后用线程索引切入 thread 模式。

```cpp
// 构造 TV-layout：将 8 个线程索引和 4 个值索引映射到 4x8 tensor 内的 1D 坐标
// (T8,V4) -> (M4,N8)
auto tv_layout = Layout<Shape <Shape <_2,_4>,Shape <_2, _2>>,
                        Stride<Stride<_8,_1>,Stride<_4,_16>>>{}; // (8,4)

// 构造任意 layout 的 4x8 tensor
Tensor A = make_tensor<float>(Shape<_4,_8>{}, LayoutRight{});    // (4,8)
// 用 tv_layout 组合 A 以变换其 shape 和顺序
Tensor tv = composition(A, tv_layout);                           // (8,4)
// 切片使每个线程有 4 个值，shape 和顺序由 tv_layout 规定
Tensor  v = tv(threadIdx.x, _);                                  // (4)
```

关于如何构造和使用这些分区模式，参见 [MMA Atom 教程](05_mma_atom.md)。

## 示例

### 从全局内存拷贝子 tile 到寄存器

以下示例将矩阵的行（任意 Layout）从全局内存拷贝到寄存器内存，然后对寄存器中的行执行 `do_something`：

```c++
Tensor gmem = make_tensor(ptr, make_shape(Int<8>{}, 16));  // (_8,16)
Tensor rmem = make_tensor_like(gmem(_, 0));                // (_8)
for (int j = 0; j < size<1>(gmem); ++j) {
  copy(gmem(_, j), rmem);
  do_something(rmem);
}
```

此代码除了知道 `gmem` 是 rank-2 且第一个模式大小为静态外，不需要关于其 Layout 的任何其他知识。

使用分块工具扩展此示例，可以用几乎相同的代码拷贝 tensor 的任意子 tile：

```c++
Tensor gmem = make_tensor(ptr, make_shape(24, 16));         // (24,16)

auto tiler         = Shape<_8,_4>{};                        // 8x4 tiler
Tensor gmem_tiled  = zipped_divide(gmem, tiler);            // ((_8,_4),Rest)
Tensor rmem        = make_tensor_like(gmem_tiled(_, 0));    // ((_8,_4))
for (int j = 0; j < size<1>(gmem_tiled); ++j) {
  copy(gmem_tiled(_, j), rmem);
  do_something(rmem);
}
```

## 小结

- Tensor 由 Engine 和 Layout 定义
    - Engine 是可偏移和解引用的迭代器
    - Layout 定义 tensor 的逻辑域并将坐标映射为偏移
- 使用与 Layout 相同的方法对 Tensor 进行分块
- 通过切片 Tensor 获取子 tensor
- 分区 = 分块和/或组合 + 切片
