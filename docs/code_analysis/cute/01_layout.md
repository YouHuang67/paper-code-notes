---
tags:
  - CUTLASS
  - CUDA
---

# CuTe Layout

> **原文出处**: [NVIDIA/cutlass - media/docs/cute/01_layout.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md)
> **许可证**: BSD-3-Clause, Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES

本文描述 `Layout`——CuTe 的核心抽象。从根本上说，`Layout` 将**坐标空间**（coordinate space）映射到**索引空间**（index space）。

`Layout` 为多维数组访问提供统一接口，抽象掉数组元素在内存中如何组织的细节。这使用户可以编写通用的多维数组访问算法，当布局改变时无需修改用户代码。例如，行主序（row-major）的 MxN layout 和列主序（column-major）的 MxN layout 可以在软件中被同等对待。

CuTe 还提供一套 **"`Layout` 代数"**。`Layout` 可以被组合与变换，以构造更复杂的布局，并将一个布局分块到另一个布局上。这有助于实现诸如将数据布局在线程布局上进行分区等操作。

## 基本类型与概念

### 整数

CuTe 大量使用动态（仅在运行时已知）和静态（在编译时已知）整数。

- **动态整数**（运行时整数）就是普通的整型，如 `int`、`size_t`、`uint16_t`。任何被 `std::is_integral<T>` 接受的类型在 CuTe 中都被视为动态整数
- **静态整数**（编译时整数）是 `std::integral_constant<Value>` 等类型的实例化。这些类型将值编码为 `static constexpr` 成员，同时支持转换为底层动态类型，因此可以在表达式中与动态整数混用。CuTe 定义了自己的 CUDA 兼容静态整数类型 `cute::C<Value>` 及重载的数学运算符，使得静态整数之间的运算结果仍为静态整数。CuTe 定义了快捷别名 `Int<1>`、`Int<2>`、`Int<3>` 和 `_1`、`_2`、`_3` 作为便利写法

CuTe 试图统一处理静态和动态整数。在后续示例中，所有动态整数都可以替换为静态整数，反之亦然。当我们在 CuTe 中说"整数"时，几乎总是指静态**或**动态整数。

CuTe 提供的整数 traits 包括：

- `cute::is_integral<T>`：检查 `T` 是否为静态或动态整数类型
- `cute::is_std_integral<T>`：检查 `T` 是否为动态整数类型，等价于 `std::is_integral<T>`
- `cute::is_static<T>`：检查 `T` 是否为空类型（实例化不依赖任何动态信息），等价于 `std::is_empty`
- `cute::is_constant<N,T>`：检查 `T` 是否为静态整数且其值等于 `N`

更多信息参见 [`integral_constant` 实现](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric/integral_constant.hpp)。

### Tuple

Tuple 是零个或多个元素的有限有序列表。[`cute::tuple` 类](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container/tuple.hpp)行为类似 `std::tuple`，但可在设备端和主机端使用。它对模板参数施加限制并精简实现以提升性能和简洁性。

### IntTuple

CuTe 将 IntTuple 概念定义为：一个整数，或一个 IntTuple 的 tuple。注意这是递归定义。在 C++ 中，我们定义了 [IntTuple 上的操作](https://github.com/NVIDIA/cutlass/tree/main/include/cute/int_tuple.hpp)。

IntTuple 的示例：

- `int{2}`，动态整数 2
- `Int<3>{}`，静态整数 3
- `make_tuple(int{2}, Int<3>{})`，动态 2 与静态 3 的 tuple
- `make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{})`，嵌套 tuple

CuTe 将 IntTuple 概念复用于多种用途，包括 Shape、Stride、Step 和 Coord（参见 [`include/cute/layout.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp)）。

IntTuple 上定义的操作：

- `rank(IntTuple)`：IntTuple 的元素个数。单个整数的 rank 为 1，tuple 的 rank 为 `tuple_size`
- `get<I>(IntTuple)`：IntTuple 的第 `I` 个元素（`I < rank`）。对单个整数，`get<0>` 就是该整数本身
- `depth(IntTuple)`：层次化 IntTuple 的深度。单个整数深度为 0，整数的 tuple 深度为 1，包含整数 tuple 的 tuple 深度为 2，以此类推
- `size(IntTuple)`：IntTuple 所有元素的乘积

我们用圆括号表示层次结构。例如 `6`、`(2)`、`(4,3)` 和 `(3,(6,2),8)` 都是 IntTuple。

### Shape 与 Stride

`Shape` 和 `Stride` 都是 IntTuple 概念。

### Layout

`Layout` 是 (`Shape`, `Stride`) 的 tuple。语义上，它实现从 Shape 内任意坐标经由 Stride 到索引的映射。

### Tensor

`Layout` 可以与数据（如指针或数组）组合来创建 `Tensor`。Layout 生成的索引用于对迭代器进行下标访问以获取相应数据。关于 Tensor 的详细说明请参见 [Tensor 部分](03_tensor.md)。

## Layout 的创建与使用

`Layout` 是一对 IntTuple：`Shape` 和 `Stride`。第一个元素定义 Layout 的抽象**形状**，第二个元素定义**步长**，将形状内的坐标映射到索引空间。

Layout 上定义的操作（类似于 IntTuple 的操作）：

- `rank(Layout)`：Layout 的模式（mode）数量，等于 Layout Shape 的 tuple 大小
- `get<I>(Layout)`：Layout 的第 `I` 个子 layout（`I < rank`）
- `depth(Layout)`：Layout Shape 的深度
- `shape(Layout)`：Layout 的 Shape
- `stride(Layout)`：Layout 的 Stride
- `size(Layout)`：Layout 函数定义域的大小，等于 `size(shape(Layout))`
- `cosize(Layout)`：Layout 函数陪域的大小（不一定是值域），等于 `A(size(A) - 1) + 1`

### 层次化访问函数

IntTuple 和 Layout 可以任意嵌套。为方便起见，上述某些函数定义了接受整数序列（而非单个整数）的版本，从而可以更容易地访问嵌套 IntTuple 或 Layout 中的元素。例如 `get<I...>(x)` 中 `I...` 是一个"C++ 参数包"，表示零个或多个整数模板参数：

- `get<I0,I1,...,IN>(x) := get<IN>(...(get<I1>(get<I0>(x)))...)`：提取 x 的第 I0 个元素的第 I1 个...的第 IN 个元素
- `rank<I...>(x) := rank(get<I...>(x))`
- `depth<I...>(x) := depth(get<I...>(x))`
- `shape<I...>(x) := shape(get<I...>(x))`
- `size<I...>(x) := size(get<I...>(x))`

在后续示例中，你会看到 `size<0>` 和 `size<1>` 用于确定 layout 或 tensor 第 0 和第 1 模式的循环边界。

### 构造 Layout

Layout 可以通过多种方式构造，可以包含编译时（静态）整数和运行时（动态）整数的任意组合。

```c++
Layout s8 = make_layout(Int<8>{});
Layout d8 = make_layout(8);

Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
Layout s2xd4 = make_layout(make_shape(Int<2>{},4));

Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
                             make_stride(Int<12>{},Int<1>{}));
Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
                               LayoutLeft{});
Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
                               LayoutRight{});

Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
                           make_stride(4,make_stride(2,1)));
Layout s2xh4_col = make_layout(shape(s2xh4),
                               LayoutLeft{});
```

`make_layout` 函数返回一个 `Layout`，它推导函数参数的类型并返回具有相应模板参数的 Layout。类似地，`make_shape` 和 `make_stride` 函数分别返回 Shape 和 Stride。CuTe 经常使用这些 `make_*` 函数，因为构造函数模板参数推导（CTAD）存在限制，且避免重复书写静态或动态整数类型。

当省略 `Stride` 参数时，它会根据提供的 `Shape` 以 `LayoutLeft` 为默认值生成。`LayoutLeft` 标签从左到右构造 Shape 的排他前缀积作为步长，不考虑 Shape 的层次结构，可视为"广义列主序步长生成"。`LayoutRight` 标签从右到左构造 Shape 的排他前缀积作为步长。对于深度为 1 的 shape，这可视为"行主序步长生成"，但对于层次化 shape，结果步长可能出人意料。

对上述每个 layout 调用 `print` 的结果：

```
s8        :  _8:_1
d8        :  8:_1
s2xs4     :  (_2,_4):(_1,_2)
s2xd4     :  (_2,4):(_1,_2)
s2xd4_a   :  (_2,4):(_12,_1)
s2xd4_col :  (_2,4):(_1,_2)
s2xd4_row :  (_2,4):(4,_1)
s2xh4     :  (2,(2,2)):(4,(2,1))
s2xh4_col :  (2,(2,2)):(_1,(2,4))
```

`Shape:Stride` 记法在 Layout 中非常常用。`_N` 记法是静态整数的缩写，其他整数为动态整数。可以看到 Shape 和 Stride 都可以由静态和动态整数混合组成。

注意 Shape 和 Stride 被假定为**一致的**（congruent），即 Shape 和 Stride 具有相同的 tuple 结构。Shape 中的每个整数都有 Stride 中对应的整数。可以用以下断言检查：

```cpp
static_assert(congruent(my_shape, my_stride));
```

### 使用 Layout

Layout 的基本用途是在 Shape 定义的坐标空间与 Stride 定义的索引空间之间进行映射。例如，打印任意 rank-2 layout 的 2-D 表格：

```c++
template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}
```

对上述示例的输出：

```
> print2D(s2xs4)
  0    2    4    6
  1    3    5    7
> print2D(s2xd4_a)
  0    1    2    3
 12   13   14   15
> print2D(s2xh4_col)
  0    2    4    6
  1    3    5    7
> print2D(s2xh4)
  0    2    1    3
  4    6    5    7
```

这里可以看到静态、动态、行主序、列主序和层次化布局的打印结果。语句 `layout(m,n)` 提供逻辑二维坐标 (m,n) 到一维索引的映射。

有趣的是，`s2xh4` 既不是行主序也不是列主序。而且它有三个模式但仍被解释为 rank-2，我们使用的是二维坐标。具体来说，`s2xh4` 在第二个模式中有一个二维多模式（multi-mode），但我们仍然可以对该模式使用一维坐标。进一步推广，使用一维坐标并将所有模式视为单个多模式：

```c++
template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}
```

输出：

```
> print1D(s2xs4)
  0    1    2    3    4    5    6    7
> print1D(s2xd4_a)
  0   12    1   13    2   14    3   15
> print1D(s2xh4_col)
  0    1    2    3    4    5    6    7
> print1D(s2xh4)
  0    4    2    6    1    5    3    7
```

Layout 的任何多模式（包括整个 layout 本身）都可以接受一维坐标。

CuTe 提供了更多可视化 Layout 的打印工具。`print_layout` 函数生成 Layout 映射的格式化二维表格：

```text
> print_layout(s2xh4)
(2,(2,2)):(4,(2,1))
      0   1   2   3
    +---+---+---+---+
 0  | 0 | 2 | 1 | 3 |
    +---+---+---+---+
 1  | 4 | 6 | 5 | 7 |
    +---+---+---+---+
```

`print_latex` 函数生成可用 `pdflatex` 编译的 LaTeX 代码，产出相同二维表格的带颜色向量图形。

### 向量 Layout

我们将 `rank == 1` 的任何 Layout 定义为向量。例如，layout `8:1` 可解释为索引连续的 8 元素向量：

```
Layout:  8:1
Coord :  0  1  2  3  4  5  6  7
Index :  0  1  2  3  4  5  6  7
```

类似地，layout `8:2` 是索引以 2 为步长的 8 元素向量：

```
Layout:  8:2
Coord :  0  1  2  3  4  5  6  7
Index :  0  2  4  6  8 10 12 14
```

根据上述 rank-1 定义，我们**也**将 layout `((4,2)):((2,1))` 解释为向量，因为其 shape 是 rank-1。内层 shape 看起来像 4x2 行主序矩阵，但额外的一层括号表明我们可以将这两个模式解释为一维 8 元素向量。步长告诉我们前 4 个元素以 2 为步长，然后有 2 份这样的前几个元素以 1 为步长：

```
Layout:  ((4,2)):((2,1))
Coord :  0  1  2  3  4  5  6  7
Index :  0  2  4  6  1  3  5  7
```

再看 layout `((4,2)):((1,4))`——同样是 4 个以 1 为步长的元素，然后 2 份以 4 为步长：

```
Layout:  ((4,2)):((1,4))
Coord :  0  1  2  3  4  5  6  7
Index :  0  1  2  3  4  5  6  7
```

作为从整数到整数的函数，它与 `8:1` 完全相同——就是恒等函数。

### 矩阵示例

推广一下，我们将任何 rank-2 的 Layout 定义为矩阵。例如：

```
Shape :  (4,2)
Stride:  (1,4)
  0   4
  1   5
  2   6
  3   7
```

这是 4x2 列主序 layout，列方向步长为 1，行方向步长为 4。

```
Shape :  (4,2)
Stride:  (2,1)
  0   1
  2   3
  4   5
  6   7
```

这是 4x2 行主序 layout，列方向步长为 2，行方向步长为 1。主序性简单取决于哪个模式的步长为 1。

与向量 layout 一样，矩阵的每个模式也可以拆分为**多模式**。这使我们能表达行主序和列主序之外的更多布局。例如：

```
Shape:  ((2,2),2)
Stride: ((4,1),2)
  0   2
  4   6
  1   3
  5   7
```

这在逻辑上也是 4x2，行方向步长为 2，但列方向有多步长。列方向前 2 个元素步长为 4，然后有一个步长为 1 的副本。由于此 layout 逻辑上是 4x2（如列主序和行主序示例），我们仍然可以使用二维坐标来索引它。

## Layout 概念

本节介绍 Layout 接受的坐标集合，以及坐标映射和索引映射的计算方式。

### Layout 兼容性

如果 layout A 的 shape 与 layout B 的 shape 兼容，则称 layout A 与 layout B **兼容**（compatible）。Shape A 与 Shape B 兼容当且仅当：

- A 的 size 等于 B 的 size，且
- A 内的所有坐标都是 B 内的有效坐标

示例：

- Shape 24 与 Shape 32 **不**兼容
- Shape 24 与 Shape (4,6) 兼容
- Shape (4,6) 与 Shape ((2,2),6) 兼容
- Shape ((2,2),6) 与 Shape ((2,2),(3,2)) 兼容
- Shape 24 与 Shape ((2,2),(3,2)) 兼容
- Shape 24 与 Shape ((2,3),4) 兼容
- Shape ((2,3),4) 与 Shape ((2,2),(3,2)) **不**兼容
- Shape ((2,2),(3,2)) 与 Shape ((2,3),4) **不**兼容
- Shape 24 与 Shape (24) 兼容
- Shape (24) 与 Shape 24 **不**兼容
- Shape (24) 与 Shape (4,6) **不**兼容

即 *compatible* 是 Shape 上的弱偏序关系：自反、反对称、传递。

### Layout 坐标

基于上述兼容性概念，每个 Layout 接受多种坐标。每个 Layout 接受任何与其兼容的 Shape 的坐标。CuTe 通过**逆字典序**（colexicographical order）提供这些坐标集之间的映射。

因此，所有 Layout 提供两个基本映射：

- 通过 Shape 将输入坐标映射到对应的自然坐标（natural coordinate）
- 通过 Stride 将自然坐标映射到索引

#### 坐标映射

输入坐标到自然坐标的映射是在 Shape 内应用逆字典序（从右到左读取，而非"字典序"的从左到右）。

以 shape `(3,(2,3))` 为例，此 shape 有三组坐标集：一维坐标、二维坐标和自然（层次化）坐标。

|  1-D  |   2-D   |   自然坐标   | |  1-D  |   2-D   |   自然坐标   |
| ----- | ------- | ----------- |-| ----- | ------- | ----------- |
|  `0`  | `(0,0)` | `(0,(0,0))` | |  `9`  | `(0,3)` | `(0,(1,1))` |
|  `1`  | `(1,0)` | `(1,(0,0))` | | `10`  | `(1,3)` | `(1,(1,1))` |
|  `2`  | `(2,0)` | `(2,(0,0))` | | `11`  | `(2,3)` | `(2,(1,1))` |
|  `3`  | `(0,1)` | `(0,(1,0))` | | `12`  | `(0,4)` | `(0,(0,2))` |
|  `4`  | `(1,1)` | `(1,(1,0))` | | `13`  | `(1,4)` | `(1,(0,2))` |
|  `5`  | `(2,1)` | `(2,(1,0))` | | `14`  | `(2,4)` | `(2,(0,2))` |
|  `6`  | `(0,2)` | `(0,(0,1))` | | `15`  | `(0,5)` | `(0,(1,2))` |
|  `7`  | `(1,2)` | `(1,(0,1))` | | `16`  | `(1,5)` | `(1,(1,2))` |
|  `8`  | `(2,2)` | `(2,(0,1))` | | `17`  | `(2,5)` | `(2,(1,2))` |

shape `(3,(2,3))` 中的每个坐标都有两个**等价**坐标，所有等价坐标映射到同一个自然坐标。再次强调，因为上述所有坐标都是有效输入，具有 Shape `(3,(2,3))` 的 Layout 可以：

- 使用一维坐标当作 18 元素的一维数组
- 使用二维坐标当作 3x6 元素的二维矩阵
- 使用层次化坐标当作 3x(2x3) 元素的层次化张量

函数 `cute::idx2crd(idx, shape)` 负责坐标映射，将 shape 内的任意坐标转换为该 shape 的等价自然坐标：

```cpp
auto shape = Shape<_3,Shape<_2,_3>>{};
print(idx2crd(   16, shape));                                // (1,(1,2))
print(idx2crd(_16{}, shape));                                // (_1,(_1,_2))
print(idx2crd(make_coord(   1,5), shape));                   // (1,(1,2))
print(idx2crd(make_coord(_1{},5), shape));                   // (_1,(1,2))
print(idx2crd(make_coord(   1,make_coord(1,   2)), shape));  // (1,(1,2))
print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape));  // (_1,(1,_2))
```

#### 索引映射

自然坐标到索引的映射通过自然坐标与 Layout 的 Stride 做**内积**来完成。

以 layout `(3,(2,3)):(3,(12,1))` 为例，自然坐标 `(i,(j,k))` 将产生索引 `i*3 + j*12 + k*1`。下表以 `i` 作为行坐标、`(j,k)` 作为列坐标展示该 layout 计算的索引：

```
       0     1     2     3     4     5     <== 1-D 列坐标
     (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)   <== 2-D 列坐标 (j,k)
    +-----+-----+-----+-----+-----+-----+
 0  |  0  |  12 |  1  |  13 |  2  |  14 |
    +-----+-----+-----+-----+-----+-----+
 1  |  3  |  15 |  4  |  16 |  5  |  17 |
    +-----+-----+-----+-----+-----+-----+
 2  |  6  |  18 |  7  |  19 |  8  |  20 |
    +-----+-----+-----+-----+-----+-----+
```

函数 `cute::crd2idx(c, shape, stride)` 负责索引映射，将 shape 内的任意坐标转换为自然坐标（如果尚未转换），然后计算与步长的内积：

```cpp
auto shape  = Shape <_3,Shape<  _2,_3>>{};
auto stride = Stride<_3,Stride<_12,_1>>{};
print(crd2idx(   16, shape, stride));       // 17
print(crd2idx(_16{}, shape, stride));       // _17
print(crd2idx(make_coord(   1,   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},_5{}), shape, stride));  // _17
print(crd2idx(make_coord(   1,make_coord(   1,   2)), shape, stride));  // 17
print(crd2idx(make_coord(_1{},make_coord(_1{},_2{})), shape, stride));  // _17
```

## Layout 操作

### 子 Layout

通过 `layout<I...>` 获取子 layout：

```cpp
Layout a   = Layout<Shape<_4,Shape<_3,_6>>>{}; // (4,(3,6)):(1,(4,12))
Layout a0  = layout<0>(a);                     // 4:1
Layout a1  = layout<1>(a);                     // (3,6):(4,12)
Layout a10 = layout<1,0>(a);                   // 3:4
Layout a11 = layout<1,1>(a);                   // 6:12
```

或 `select<I...>`：

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = select<1,3>(a);                   // (3,7):(2,30)
Layout a01 = select<0,1,3>(a);                 // (2,3,7):(1,2,30)
Layout a2  = select<2>(a);                     // (5):(6)
```

或 `take<ModeBegin, ModeEnd>`：

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = take<1,3>(a);                     // (3,5):(2,6)
Layout a14 = take<1,4>(a);                     // (3,5,7):(2,6,30)
// take<1,1> 不允许，不允许空 layout
```

### 拼接

可以将 Layout 传递给 `make_layout` 进行包装和拼接：

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout row = make_layout(a, b);                 // (3,4):(1,3)
Layout col = make_layout(b, a);                 // (4,3):(3,1)
Layout q   = make_layout(row, col);             // ((3,4),(4,3)):((1,3),(3,1))
Layout aa  = make_layout(a);                    // (3):(1)
Layout aaa = make_layout(aa);                   // ((3)):((1))
Layout d   = make_layout(a, make_layout(a), a); // (3,(3),3):(1,(1),1)
```

或使用 `append`、`prepend`、`replace` 组合：

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout ab = append(a, b);                       // (3,4):(1,3)
Layout ba = prepend(a, b);                      // (4,3):(3,1)
Layout c  = append(ab, ab);                     // (3,4,(3,4)):(1,3,(1,3))
Layout d  = replace<2>(c, b);                   // (3,4,4):(1,3,3)
```

### 分组与展平

Layout 模式可以用 `group<ModeBegin, ModeEnd>` 分组，用 `flatten` 展平：

```cpp
Layout a = Layout<Shape<_2,_3,_5,_7>>{};  // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout b = group<0,2>(a);                 // ((_2,_3),_5,_7):((_1,_2),_6,_30)
Layout c = group<1,3>(b);                 // ((_2,_3),(_5,_7)):((_1,_2),(_6,_30))
Layout f = flatten(b);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout e = flatten(c);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
```

分组、展平和模式重排允许就地将张量重新解释为矩阵、将矩阵重新解释为向量、将向量重新解释为矩阵等。

### 切片

Layout 可以被切片，但切片操作更适合在 Tensor 上执行。详见 [Tensor 部分](03_tensor.md)。

## 小结

- Layout 的 **Shape** 定义其坐标空间
    - 每个 Layout 都有一维坐标空间，可用于按逆字典序遍历坐标空间
    - 每个 Layout 都有 R 维坐标空间（R 为 layout 的 rank），R 维坐标的逆字典序枚举对应上述一维坐标
    - 每个 Layout 都有层次化（h-D）的自然坐标空间，按逆字典序排列，其枚举对应一维坐标。自然坐标与 Shape **一致**，使坐标的每个元素与 Shape 的对应元素匹配

- Layout 的 **Stride** 将坐标映射到索引
    - 自然坐标的元素与 Stride 的元素的内积产生结果索引

对于每个 Layout，存在一个与之兼容的整数 Shape，即 `size(layout)`。由此可以观察到：

> **Layout 是从整数到整数的函数。**

如果你熟悉 C++23 的 `mdspan`，这是 `mdspan` layout mapping 与 CuTe Layout 之间的重要区别。在 CuTe 中，Layout 是一等公民，原生支持层次化以自然表示行主序和列主序之外的函数，并且可以用层次化坐标进行索引。`mdspan` 的输入坐标必须与 `mdspan` 具有相同的 shape；多维 `mdspan` 不接受一维坐标。
