---
tags:
  - CUTLASS
  - CUDA
---

# CuTe Layout 代数

> **原文出处**: [NVIDIA/cutlass - media/docs/cute/02_layout_algebra.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md)
> **许可证**: BSD-3-Clause, Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES

CuTe 提供一套 **"Layout 代数"**，支持以不同方式组合 layout。该代数包括：

- Layout 的**函数组合**（functional composition）
- Layout 的**乘积**（product）——按照一个 layout 复制另一个 layout
- Layout 的**除法**（divide）——按照一个 layout 拆分另一个 layout

从简单 layout 构建复杂 layout 的常用工具依赖 Layout 乘积。将 layout（如数据 layout）按另一个 layout（如线程 layout）进行分区的常用工具依赖 Layout 除法。所有这些工具都依赖 Layout 的函数组合。

## Coalesce

在上一节中，我们总结道：

> Layout 是从整数到整数的函数。

`coalesce` 操作是对"从整数到整数的函数"的**化简**。如果我们只关心输入整数，就可以操纵 Layout 的 shape 和模式数量而不改变其作为函数的行为。`coalesce` 唯一不能改变的是 Layout 的 `size`。

具体的后置条件（参见 [`coalesce` 单元测试](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/core/coalesce.cpp)）：

```cpp
// @post size(@a result) == size(@a layout)
// @post depth(@a result) <= 1
// @post for all i, 0 <= i < size(@a layout), @a result(i) == @a layout(i)
Layout coalesce(Layout const& layout)
```

例如：

```cpp
auto layout = Layout<Shape <_2,Shape <_1,_6>>,
                     Stride<_1,Stride<_6,_2>>>{};
auto result = coalesce(layout);    // _12:_1
```

结果具有更少的模式且更"简洁"。这可以节省坐标映射和索引映射中的若干操作（如果这些操作是动态执行的话）。

那么，如何实现？

- 列主序 Layout 如 `(_2,_4):(_1,_2)` 对一维坐标的行为与 `_8:_1` 完全相同
- 大小为 static-1 的模式总是产生 static-0 的自然坐标，无论步长如何都可以忽略

推广来看，考虑只有两个整数模式 s0:d0 和 s1:d1 的 layout。将合并结果记为 `s0:d0 ++ s1:d1`，有四种情况：

1. `s0:d0 ++ _1:d1 => s0:d0`：忽略大小为 static-1 的模式
2. `_1:d0 ++ s1:d1 => s1:d1`：忽略大小为 static-1 的模式
3. `s0:d0 ++ s1:s0*d0 => s0*s1:d0`：如果第二个模式的步长等于第一个模式的大小与步长的乘积，则可以合并
4. `s0:d0 ++ s1:d1 => (s0,s1):(d0,d1)`：否则无法合并，必须分开处理

就是这样！我们可以展平任何 layout 并对每一对相邻模式依次应用上述二元操作来"合并"layout 的模式。

### 按模式 Coalesce

显然有时我们**确实**关心 Layout 的 shape，但仍想合并。例如，我有一个二维 Layout，希望结果仍是二维的。

为此，`coalesce` 提供了一个接受额外参数的重载：

```cpp
// 在 trg_profile 的终端处应用 coalesce
Layout coalesce(Layout const& layout, IntTuple const& trg_profile)
```

用法：

```cpp
auto a = Layout<Shape <_2,Shape <_1,_6>>,
                Stride<_1,Stride<_6,_2>>>{};
auto result = coalesce(a, Step<_1,_1>{});   // (_2,_6):(_1,_2)
// 等价于
auto same_r = make_layout(coalesce(layout<0>(a)),
                          coalesce(layout<1>(a)));
```

此函数递归进入 `Step<_1,_1>{}`，在遇到整数（值不重要，只是标记）而非 tuple 时，对对应的子 layout 应用 `coalesce`。

> 这种"先定义将 Layout 视为一维整数到整数函数的操作，再推广到任意形状 layout"的主题将是一个常见模式！

## Composition（组合）

Layout 的函数组合是 CuTe 的核心，几乎所有高级操作都用到它。

再次从"Layout 是从整数到整数的函数"出发，我们可以定义产生另一个 Layout 的函数组合。先看一个例子：

```text
函数组合，R := A o B
R(c) := (A o B)(c) := A(B(c))

示例
A = (6,2):(8,2)
B = (4,3):(3,1)

R( 0) = A(B( 0)) = A(B(0,0)) = A( 0) = A(0,0) =  0
R( 1) = A(B( 1)) = A(B(1,0)) = A( 3) = A(3,0) = 24
R( 2) = A(B( 2)) = A(B(2,0)) = A( 6) = A(0,1) =  2
R( 3) = A(B( 3)) = A(B(3,0)) = A( 9) = A(3,1) = 26
R( 4) = A(B( 4)) = A(B(0,1)) = A( 1) = A(1,0) =  8
R( 5) = A(B( 5)) = A(B(1,1)) = A( 4) = A(4,0) = 32
R( 6) = A(B( 6)) = A(B(2,1)) = A( 7) = A(1,1) = 10
R( 7) = A(B( 7)) = A(B(3,1)) = A(10) = A(4,1) = 34
R( 8) = A(B( 8)) = A(B(0,2)) = A( 2) = A(2,0) = 16
R( 9) = A(B( 9)) = A(B(1,2)) = A( 5) = A(5,0) = 40
R(10) = A(B(10)) = A(B(2,2)) = A( 8) = A(2,1) = 18
R(11) = A(B(11)) = A(B(3,2)) = A(11) = A(5,1) = 42
```

一个绝妙的观察是，上面定义的函数 `R(c) = k` 可以写成另一个 Layout：

```
R = ((2,2),3):((24,2),8)
```

**并且** `compatible(B, R)`，即 B 的每个坐标也可以用作 R 的坐标。这是函数组合的预期性质，因为 B 定义了 R 的**定义域**。

后置条件（参见 [`composition` 单元测试](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/core/composition.cpp)）：

```cpp
// @post compatible(@a layout_b, @a result)
// @post for all i, 0 <= i < size(@a layout_b), @a result(i) == @a layout_a(@a layout_b(i)))
Layout composition(LayoutA const& layout_a, LayoutB const& layout_b)
```

### 计算 Composition

几个观察：

- `B = (B_0, B_1, ...)`：layout 可以表示为其子 layout 的拼接
- `A o B = A o (B_0, B_1, ...) = (A o B_0, A o B_1, ...)`：当 B 是单射时，组合对拼接是左分配的

基于以上，不失一般性地假设 `B = s:d` 是具有整数 shape 和 stride 的 layout，且 A 是展平并合并后的 layout。

当 A 是整数 `A = a:b` 时，结果很直接：`R = A o B = a:b o s:d = s:(b*d)`。组合 R 是 A 中以 d 为步长的前 s 个元素。

当 A 是多模式时需要更仔细。`A o B = A o s:d`（s 和 d 为整数）意味着：

**1. 确定产生 A 中每第 d 个元素的 layout**

中间 layout 的 shape 通过从 A 的 shape 从左开始逐步"除去"前 d 个元素来计算。例如：

- `(6,2) / 2 => (3,2)`
- `(6,2) / 3 => (2,2)`
- `(6,2) / 6 => (1,2)`
- `(3,6,2,8) / 3 => (1,6,2,8)`
- `(3,6,2,8) / 72 => (1,1,1,4)`

步长通过上述操作的余数来缩放 A 的步长。例如 `(3,6,2,8):(w,x,y,z) / 72` 产生步长 `(72*w,24*x,4*x,2*z)`。

只有特定值才能除以 shape 并得到合理结果，这称为**步长可整除条件**，CuTe 在可能时会静态检查。

**2. 保留新步长 A 的前 s 个元素使结果与 B shape 兼容**

通过从 A 的 shape 从左开始逐步"取模"前 s 个元素来计算。例如：

- `(6,2) % 2 => (2,1)`
- `(6,2) % 3 => (3,1)`
- `(6,2) % 12 => (6,2)`
- `(3,6,2,8) % 6 => (3,2,1,1)`

此操作同样必须满足 **shape 可整除条件**。

从上述例子可以构造组合 `(3,6,2,8):(w,x,y,z) o 16:9 = (1,2,2,4):(9*w,3*x,y,z)`。

---

#### 示例 1 — 组合的详细计算

两个多模式 layout 的组合：

```
A = (6,2):(8,2)
B = (4,3):(3,1)

1. 利用左分配律和拼接性质：
R = A o B
  = (6,2):(8,2) o (4,3):(3,1)
  = ((6,2):(8,2) o 4:3, (6,2):(8,2) o 3:1)

---
计算 (6,2):(8,2) o 4:3
- 步长化 layout：(6,2):(8,2) / 3 = (6/3,2):(8*3,2) = (2,2):(24,2)
- 保持 shape 兼容：(2,2):(24,2) % 4 = (2,2):(24,2)

---
计算 (6,2):(8,2) o 3:1
- 步长化 layout：(6,2):(8,2) / 1 = (6,2):(8,2)
- 保持 shape 兼容：(6,2):(8,2) % 3 = (3,1):(8,2)

---
合并并按模式 coalesce：
R = A o B = ((2,2),3):((24,2),8)
```

#### 示例 2 — 将 layout 重塑为矩阵

`20:2 o (5,4):(4,1)`：将 layout `20:2` 解释为行主序的 5x4 矩阵。

1. `= 20:2 o (5:4, 4:1)`
2. `= (20:2 o 5:4, 20:2 o 4:1)`
    - `20:2 o 5:4 => 5:8`（平凡情况）
    - `20:2 o 4:1 => 4:2`（平凡情况）
3. `= (5:8, 4:2)`
4. `= (5,4):(8,2)`（最终组合 layout）

#### 示例 3 — 将 layout 重塑为矩阵

`(10,2):(16,4) o (5,4):(1,5)`：将 layout `(10,2):(16,4)` 解释为列主序的 5x4 矩阵。

1. `= (10,2):(16,4) o (5:1, 4:5)`
2. `= ((10,2):(16,4) o 5:1, (10,2):(16,4) o 4:5)`
    - `(10,2):(16,4) o 5:1 => (5,1):(16,4)`
    - `(10,2):(16,4) o 4:5 => (2,2):(80,4)`
3. `= ((5,1):(16,4), (2,2):(80,4))`
4. `= (5:16, (2,2):(80,4))`（按模式 coalesce）
5. `= (5,(2,2)):(16,(80,4))`（最终组合 layout）

使用编译时 shape 和 stride 的 C++ 代码打印 `(_5,(_2,_2)):(_16,(_80,_4))`。使用动态整数则打印 `((5,1),(2,2)):((16,4),(80,4))`。结果看起来不同但在数学上相同——size-1 模式不影响 layout 作为数学函数的行为。

### 按模式 Composition

有时我们关心 A layout 的 shape，想对各个模式单独应用 `composition`。为此，`composition` 的第二个参数可以是 **Tiler**——一个 layout 或 layout 的 tuple。

```cpp
// (12,(4,8)):(59,(13,1))
auto a = make_layout(make_shape (12,make_shape ( 4,8)),
                     make_stride(59,make_stride(13,1)));
// <3:4, 8:2>
auto tiler = make_tile(Layout<_3,_4>{},   // 对 mode-0 应用 3:4
                       Layout<_8,_2>{});  // 对 mode-1 应用 8:2

// (_3,(2,4)):(236,(26,1))
auto result = composition(a, tiler);
// 等价于
auto same_r = make_layout(composition(layout<0>(a), get<0>(tiler)),
                          composition(layout<1>(a), get<1>(tiler)));
```

为方便起见，CuTe 也将 Shape 解释为 tiler——Shape 被解释为步长为 1 的 layout tuple：

```cpp
auto tiler = make_shape(Int<3>{}, Int<8>{});
// 等价于 <3:1, 8:1>
```

## Composition Tiler

总结，**Tiler** 是以下对象之一：

1. 一个 Layout
2. 一个 Tiler 的 tuple
3. 一个 Shape（将被解释为步长为 1 的 Layout tiler）

以上任何一种都可以作为 `composition` 的第二个参数。情况 (1) 中组合视为两个整数到整数函数之间的操作；情况 (2) 和 (3) 中组合按对应模式逐一应用。

## Complement（补）

在进入"乘积"和"除法"之前，还需要一个操作。`composition` 可看作 layout B "选择" layout A 中的某些坐标。但那些**未被选择**的坐标呢？为实现通用分块，我们需要选择任意元素——即 tile——并描述这些 tile 的 layout——即"剩余"或"其余"。

Layout 的 `complement` 试图找到表示"其余"的另一个 layout——即原 layout 未涉及的元素。

后置条件（参见 [`complement` 单元测试](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/core/complement.cpp)）：

```cpp
// @post cosize(make_layout(@a layout_a, @a result))) >= size(@a cotarget)
// @post cosize(@a result) >= round_up(size(@a cotarget), cosize(@a layout_a))
// @post 对所有 i, 1 <= i < size(@a result), @a result(i-1) < @a result(i)
// @post 对所有 i, 1 <= i < size(@a result),
//         对所有 j, 0 <= j < size(@a layout_a),
//           @a result(i) != @a layout_a(j)
Layout complement(LayoutA const& layout_a, Shape const& cotarget)
```

即 layout A 关于 Shape M 的补 R 满足：

1. R 的大小和陪大小被 `size(M)` **限定**
2. R 是**有序的**——R 的步长为正且递增，因此 R 是唯一的
3. A 和 R 的陪域**不相交**——R 试图"完成" A 的陪域

### Complement 示例

- `complement(4:1, 24)` = `6:4`。`(4,6):(1,4)` 的陪大小为 24。layout `4:1` 被 `6:4` 有效地重复了 6 次
- `complement(6:4, 24)` = `4:1`。`6:4` 的"空洞"被 `4:1` 填充
- `complement((4,6):(1,4), 24)` = `1:0`。无需追加
- `complement(4:2, 24)` = `(2,3):(1,8)`。`4:2` 的"空洞"先被 `2:1` 填充，然后整体被 `3:8` 重复 3 次
- `complement((2,4):(1,6), 24)` = `3:2`
- `complement((2,2):(1,6), 24)` = `(3,2):(2,12)`

最后一个例子的可视化：原 layout `(2,2):(1,6)` 的像用灰色标记，补操作有效地"重复"了原 layout（用其他颜色展示），使结果的陪域大小为 24。补 `(3,2):(2,12)` 可视为"重复的 layout"。

## Division（除法/分块）

最终，我们可以定义 Layout 之间的除法。将 layout 分解为各部分的函数是分块和分区 layout 的基础。

`logical_divide(A, B)` 将 layout A 拆分为两个模式——第一个模式包含 B 指向的所有元素，第二个模式包含 B 未指向的所有元素。

形式化定义：

$$A \oslash B := A \circ (B, B^*)$$

实现：

```cpp
template <class LShape, class LStride,
          class TShape, class TStride>
auto logical_divide(Layout<LShape,LStride> const& layout,
                    Layout<TShape,TStride> const& tiler)
{
  return composition(layout, make_layout(tiler, complement(tiler, size(layout))));
}
```

注意这仅用拼接、组合和补来定义。

- "第一个模式包含 B 指向的所有元素"——这就是组合 `A o B`
- "第二个模式包含 B 未指向的所有元素"——B 的补 `B*`（在 A 的 size 范围内），描述的是"B 的重复的 layout"。如果 B 是"tiler"，那么 `B*` 就是 tile 的 layout

### Logical Divide 一维示例

对一维 layout `A = (4,2,3):(2,1,8)` 使用 tiler `B = 4:2` 分块。即：有一个 24 元素的一维向量，其存储顺序由 A 定义，我们要提取以 2 为步长的 4 元素 tile。

- 补：`B = 4:2` 在 `size(A) = 24` 下的补为 `B* = (2,3):(1,8)`
- 拼接：`(B,B*) = (4,(2,3)):(2,(1,8))`
- 组合：`A` 与 `(B,B*)` 的组合为 `((2,2),(2,3)):((4,1),(2,8))`

分块后，结果的第一个模式是数据的 tile，第二个模式遍历每个 tile。

### Logical Divide 二维示例

使用上面定义的 Tiler 概念，此操作立即推广到多维分块——简单地对二维 layout 的行和列分别按模式应用 `logical_divide`。

### Zipped, Tiled, Flat Divide

为了更方便地操作 tile，CuTe 提供了 `logical_divide` 的多种便利变体：

```text
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
```

`zipped_divide` 将 `logical_divide` 的"子 tile"汇集为单个模式，"rest"汇集为单个模式：

```cpp
auto layout_a = make_layout(make_shape (Int< 9>{}, make_shape (Int< 4>{}, Int<8>{})),
                            make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{})));
auto tiler = make_tile(Layout<_3,_3>{},
                       Layout<Shape <_2,_4>, Stride<_1,_8>>{});

// ((TileM,RestM), (TileN,RestN)) shape ((3,3), (8,4))
auto ld = logical_divide(layout_a, tiler);
// ((TileM,TileN), (RestM,RestN)) shape ((3,8), (3,4))
auto zd = zipped_divide(layout_a, tiler);
```

第 3 个 tile 的偏移是 `zd(0,3)`，第 7 个是 `zd(0,7)`，第 (1,2) 个是 `zd(0,make_coord(1,2))`。tile 本身始终是 `layout<0>(zd)`。事实上始终有：

`layout<0>(zipped_divide(a, b)) == composition(a, b)`

注意 `logical_divide` 保留模式的**语义**——A 的 M 模式仍是结果的 M 模式。而 `zipped_divide` 不保留——mode-0 是 Tile 本身，mode-1 是 tile 的 layout。

## Product（乘积/分块）

Layout 乘积的定义。`logical_product(A, B)` 产生一个两模式 layout：第一个模式是 layout A，第二个模式是 layout B 但每个元素被替换为 layout A 的一个"唯一复制"。

形式化定义：

$$A \otimes B := (A, A^* \circ B)$$

实现：

```cpp
template <class LShape, class LStride,
          class TShape, class TStride>
auto logical_product(Layout<LShape,LStride> const& layout,
                     Layout<TShape,TStride> const& tiler)
{
  return make_layout(layout, composition(complement(layout, size(layout)*cosize(tiler)), tiler));
}
```

注意这仅用拼接、组合和补来定义。

- "第一个模式是 layout A"——直接复制 A
- "第二个模式是 B，但每个元素替换为 A 的唯一复制"——A 的补 `A*`（在 B 的 cosize 范围内），描述的是"A 的重复的 layout"

### Logical Product 一维示例

将一维 layout `A = (2,2):(4,1)` 按 `B = 6:1` 复制——即有一个 4 元素的一维 layout，我们要复制它 6 次。

- 补：`A` 在 `6*4 = 24` 下的补为 `A* = (2,3):(2,8)`
- 组合：`A* = (2,3):(2,8)` 与 `B = 6:1` 的组合为 `(2,3):(2,8)`
- 拼接：`(A, A* o B) = ((2,2),(2,3)):((4,1),(2,8))`

结果与一维 Logical Divide 示例完全相同。

当然，可以通过改变 B 来改变 tile 的数量和顺序。

### Blocked 和 Raked Product

`blocked_product(LayoutA, LayoutB)` 和 `raked_product(LayoutA, LayoutB)` 是在一维 `logical_product` 之上的**rank 敏感**变换，让我们表达更直观的 Layout 乘积。

关键观察——`logical_product` 的兼容性后置条件：

```
// @post rank(result) == 2
// @post compatible(layout_a, layout<0>(result))
// @post compatible(layout_b, layout<1>(result))
```

因为 A 始终与结果的 mode-0 兼容，B 始终与 mode-1 兼容，如果让 A 和 B 同 rank，就可以在乘积后"重新关联"同类模式——A 的"列"模式可与 B 的"列"模式组合，A 的"行"模式可与 B 的"行"模式组合，等等。

这正是 `blocked_product` 和 `raked_product` 所做的。`blocked_product` 中，结果的"列"模式由 A 的"列"模式接 B 的"列"模式构成——产生**块分布**（block distribution）。`raked_product` 则将 B 的模式放在 A 的模式前面——产生**交错分布**（raked/cyclic distribution）。

### Zipped 和 Tiled Product

类似 `zipped_divide` 和 `tiled_divide`：

```text
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_product : ((M,TileM), (N,TileN), L, ...)
zipped_product  : ((M,N), (TileM,TileN,L,...))
tiled_product   : ((M,N), TileM, TileN, L, ...)
flat_product    : (M, N, TileM, TileN, L, ...)
```
