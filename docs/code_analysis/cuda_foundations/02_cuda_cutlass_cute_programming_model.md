---
tags:
  - CUDA
  - CUTLASS
---
# CUDA CUTLASS/CuTe 编程模型

本文汇总 CUTLASS 的模板分发体系与 CuTe 的 Tensor/Layout/Atom 抽象，作为各类 CUDA kernel 分析的公共引用页。

阅读顺序建议：

- 先看 [CUDA 执行模型与内存访问](01_cuda_execution_model_and_memory.md)，再回到本文看 CUTLASS / CuTe 的模板与布局抽象
- 在项目文档中只关心某个 kernel 的时候，优先从对应项目文档跳转到本文相关小节，不必通读全文

本文基于 CUTLASS 源码（随 tilelang 打包的版本）整理。CuTe 相关源码引用均指向本地路径 `$CUTLASS/include/cute/`。

**源码根目录**: `/home/hy/anaconda3/envs/fta/lib/python3.11/site-packages/tilelang/3rdparty/cutlass/include/cute/`

---

## CUTLASS 模板体系

### 核心概念

CUTLASS 是 NVIDIA 的 C++ 模板库，将 GEMM 分解为层次化的组件：

```
Device level    →  完整 GEMM 调用（grid launch）
Kernel level    →  threadblock 级 tiling
Threadblock MMA →  一个 block 内的矩阵乘累加
Warp MMA        →  一个 warp 的 Tensor Core 指令
```

每一层都用模板参数组合来特化，编译期确定 shape、数据类型、内存布局和指令类别。

### `GemmShape<M, N, K>`

`GemmShape` 用类型携带某一层 GEMM tile 的编译期尺寸：

```cpp
using ThreadblockShape = GemmShape<kQueriesPerBlock, kKeysPerBlock, ThreadK>;
using WarpShape = GemmShape<32, 32, WarpK>;
using InstructionShape = GemmShape<16, 8, 8>;
```

- `ThreadblockShape`：一个 block 负责的输出 tile
- `WarpShape`：一个 warp 负责的输出 tile
- `InstructionShape`：一条 MMA 指令处理的 tile

从 threadblock 到 warp，再到 instruction 的逐层切分，决定了 warp 数量、迭代次数和寄存器 fragment 的形状。

### `OpClass` 与 `ArchTag`

`OpClass` 决定用哪类计算单元：

- `OpClassSimt`：CUDA Core / FMA 路径
- `OpClassTensorOp`：Tensor Core / MMA 路径

`ArchTag` 标记目标架构：

- `cutlass::arch::Sm50`
- `cutlass::arch::Sm70`
- `cutlass::arch::Sm75`
- `cutlass::arch::Sm80`

CUTLASS 通过模板偏特化按 `(OpClass, ArchTag, dtype)` 选择默认配置，例如 `DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm80, ...>` 会给出对应的 `ThreadblockShape`、`WarpShape`、`InstructionShape` 和 `kStages`。

### MMA 流水线

CUTLASS threadblock MMA 的典型数据流：

```
Global Memory
  → IteratorA / IteratorB
  → Shared Memory
  → WarpIterator
  → Registers
  → MMA instruction
  → Accumulator
```

关键组件：

- `IteratorA / IteratorB`：从 global memory 取 tile，处理边界与对齐
- `Mma`：组织 warp 级计算和数据搬运
- `kStages`：多阶段流水线深度；SM80 上通常与 `cp.async` 配合

在 attention kernel 中：

- MM0: `Q @ K^T` 走标准 global → smem → register → MMA 路径
- MM1: `P @ V` 中的 A 操作数 `P` 来自 shared memory，因此会出现 `MmaFromSharedMemory` 这类特殊路径

### Epilogue

Epilogue 负责将 MMA 累加结果从寄存器写回 global memory，并执行输出后处理：

```
Accumulator
  → AccumulatorFragmentIterator
  → WarpTileIterator
  → SharedLoadIterator
  → EpilogueOutputOp
  → OutputTileIterator
```

在 attention kernel 中，`MemoryEfficientAttentionNormalize` 就是 `EpilogueOutputOp`，负责把 Online Softmax 的累计结果做最终 rescale / normalize。

### `kSingleValueIteration` 与输出累加策略

这是 CUTLASS attention 能覆盖大 `head_dim` 的关键模板分支：

- `kSingleValueIteration = true`
  - `head_dim <= kKeysPerBlock`
  - V 在 dim 维度一次迭代完成
  - 输出 `accum_o` 常驻寄存器，最后统一写回
- `kSingleValueIteration = false`
  - V 需要多次迭代
  - 中间结果通过 Epilogue 写入 GMEM buffer
  - 下次迭代再读回做 rescale 累加

这个分支本质上是“寄存器容量够不够支撑整段输出常驻”的编译期判断。

---

## 1. 核心概念：Tensor = Pointer + Layout

CuTe 的 `Tensor` 将**存储**（Engine）和**索引方式**（Layout）正交分离。一个 Tensor 不包含任何关于数据排列的隐式假设——所有内存访问模式都由 Layout 在编译时完全确定。

**源码位置**: [tensor_impl.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor_impl.hpp)

### 1.1 Tensor 类定义

```cpp
template <class Engine, class Layout>
struct Tensor : private cute::tuple<Engine, Layout> {
    // Engine: 数据存储（ViewEngine 或 ArrayEngine）
    // Layout: 坐标 → 线性偏移的映射
};
```

- **`ViewEngine<Iterator>`**：非拥有视图，包装一个指针。`make_tensor(ptr, layout)` 创建 ViewEngine Tensor
- **`ArrayEngine<T, N>`**：拥有存储，内部是对齐数组。`make_tensor<T>(layout)` 创建 ArrayEngine Tensor（用于寄存器 fragment）

关键方法：

| 方法 | 功能 |
|------|------|
| `tensor(coord)` | 通过 Layout 将逻辑坐标映射为线性偏移，返回 `engine[offset]` |
| `tensor(c0, c1, ...)` | 多维坐标的便捷语法，等价于 `tensor(make_coord(c0, c1, ...))` |
| `tensor(_, i)` | 下划线切片——固定第 2 维为 `i`，返回子 Layout 的新 Tensor |
| `compose(other_layout)` | Layout 组合：`composition(this->layout(), other_layout)` |
| `data()` | 返回底层指针/迭代器 |

**地址空间标记**：`make_smem_ptr` 和 `make_gmem_ptr` 为指针附加地址空间信息，CuTe 据此自动选择不同的 PTX 指令（global load、smem load、cp.async 等）。

### 1.2 Layout：Shape + Stride

**源码位置**: [layout.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp)

Layout 是 CuTe 最核心的抽象——一个从多维逻辑坐标到一维线性偏移的纯函数：

```cpp
template <class Shape, class Stride = LayoutLeft::Apply<Shape>>
struct Layout : private cute::tuple<Shape, Stride> {
    static constexpr int rank = rank_v<Shape>;

    // 坐标 → 线性偏移
    auto operator()(Coord const& coord) const {
        return crd2idx(coord, shape(), stride());
    }

    // 切片（Coord 中包含 Underscore 时）
    auto operator()(Coord const& coord) const {   // has_underscore 分支
        return slice(coord, *this);
    }
};
```

**`crd2idx` 的计算规则**：内积 $\text{offset} = \sum_i \text{coord}_i \times \text{stride}_i$。对于嵌套的 Shape/Stride，递归展开后求内积。

**类型别名**——Shape、Stride、Step、Coord、Tile 全部是 `cute::tuple` 的别名：

```cpp
template <class... Shapes>  using Shape  = cute::tuple<Shapes...>;
template <class... Strides> using Stride = cute::tuple<Strides...>;
template <class... Strides> using Step   = cute::tuple<Strides...>;
template <class... Coords>  using Coord  = cute::tuple<Coords...>;
template <class... Layouts> using Tile   = cute::tuple<Layouts...>;
```

**编译时常量**：CuTe 使用 `Int<N>` (即 `cute::constant<int, N>`) 表示编译时整数。例如 `_64{}` 是 `Int<64>` 的实例。当 Shape 和 Stride 全部由 `Int<N>` 构成时，Layout 是**完全静态**的——所有偏移计算在编译时完成，运行时零开销。

### 1.3 Layout 示例

**行主序 4×8 矩阵**：

```
Layout<Shape<_4, _8>, Stride<_8, _1>>

逻辑坐标 (i, j) → 偏移 = i * 8 + j

(0,0)→0   (0,1)→1   ...  (0,7)→7
(1,0)→8   (1,1)→9   ...  (1,7)→15
(2,0)→16  (2,1)→17  ...  (2,7)→23
(3,0)→24  (3,1)→25  ...  (3,7)→31
```

**列主序 4×8 矩阵**：

```
Layout<Shape<_4, _8>, Stride<_1, _4>>

逻辑坐标 (i, j) → 偏移 = i * 1 + j * 4

(0,0)→0   (0,1)→4   ...  (0,7)→28
(1,0)→1   (1,1)→5   ...  (1,7)→29
(2,0)→2   (2,1)→6   ...  (2,7)→30
(3,0)→3   (3,1)→7   ...  (3,7)→31
```

**嵌套 Shape**（用于 Tensor Core 的线程-值映射）：

```
Layout<Shape <Shape <_4, _8>, Shape <_2, _2>>,
       Stride<Stride<_32, _1>, Stride<_16, _8>>>

Shape 的第 0 维是 (4, 8) = 32 个线程
Shape 的第 1 维是 (2, 2) = 4 个值/线程

坐标 ((thr_row, thr_col), (val_row, val_col))
  → 偏移 = thr_row * 32 + thr_col * 1 + val_row * 16 + val_col * 8
```

### 1.4 Layout 操作

**Composition（组合）**：`composition(layout_a, layout_b)` 将 `layout_b` 的输出喂给 `layout_a` 的输入，产生新的复合 Layout。直观理解：`layout_b` 做坐标变换，`layout_a` 做地址计算。这是 Swizzle 和逻辑转置的基础。

**Tiled Divide（分块）**：`tiled_divide(layout, tile)` 将 Layout 的每一维按 tile 大小分块，产生 `((tile_shape), (num_tiles))` 的嵌套 Layout。用于将全局 Tensor 切分为每个线程/warp 处理的子块。

**Slice（切片）**：`layout(coord_with_underscore)` 固定某些维度，保留下划线维度，返回子 Layout。例如 `layout(_, 3)` 固定第 2 维为 3，返回第 1 维的子 Layout。

**`get_hier_coord` / `get_flat_coord`**：反向映射——从线性偏移恢复逻辑坐标。`idx2crd` 计算层次化坐标，`crd2crd` 展平为一维坐标。仅对 compact layout 有效。

---

## 2. MMA Atom：Tensor Core 指令抽象

CuTe 将硬件 MMA 指令封装为三层抽象：底层 PTX 结构体 → MMA_Atom → TiledMMA。

### 2.1 底层 PTX 结构体

**源码位置**: [arch/mma_sm80.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm80.hpp)

每个 PTX MMA 指令被封装为一个结构体，定义了寄存器类型和内联汇编：

```cpp
// SM80 16×8×16 矩阵乘加，bf16 输入、fp32 累加
struct SM80_16x8x16_F32BF16BF16F32_TN {
    using DRegisters = float[4];       // 输出：4 个 fp32
    using ARegisters = uint32_t[4];    // A 操作数：4 个 uint32（每个 = 2 个 bf16）
    using BRegisters = uint32_t[2];    // B 操作数：2 个 uint32（每个 = 2 个 bf16）
    using CRegisters = float[4];       // 累加器：4 个 fp32

    static void fma(float& d0, float& d1, float& d2, float& d3,
                    uint32_t const& a0, ..., uint32_t const& a3,
                    uint32_t const& b0, uint32_t const& b1,
                    float const& c0, ..., float const& c3) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3},"      // D: 4×fp32 输出
            "{%4, %5, %6, %7},"      // A: 4×uint32 = 8×bf16
            "{%8, %9},"              // B: 2×uint32 = 4×bf16
            "{%10, %11, %12, %13};"  // C: 4×fp32 累加器
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
               "r"(b0), "r"(b1),
               "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    }
};
```

**PTX 指令语义**：`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`

| 字段 | 含义 |
|------|------|
| `mma.sync` | 矩阵乘加，warp 内 32 线程同步执行 |
| `.aligned` | 所有线程到达同一条指令（无分支分歧） |
| `.m16n8k16` | 计算 $D_{16\times8} = A_{16\times16} \cdot B_{16\times8} + C_{16\times8}$ |
| `.row.col` | A 行主序（T=Transposed）、B 列主序（N=Normal），即 TN 布局 |
| `.f32.bf16.bf16.f32` | D=fp32, A=bf16, B=bf16, C=fp32 |

SM80 还提供 16×8×8 变体（K 维减半，寄存器减少），以及 fp16、tf32、int8、int4 等数据类型组合。

### 2.2 寄存器分布：Warp 内 32 线程如何分担

单条 `mma.sync.aligned.m16n8k16` 指令由 warp 内 32 个线程**协作**计算 16×8 的输出矩阵。每个线程持有 4 个 fp32 输出元素（d0-d3），对应矩阵中 2 行×2 列的子块：

```
D 矩阵 16×8（每格 = 1 个 fp32 元素）:

  col:    0    1    2    3    4    5    6    7
row 0:  [t0 ][t0 ][t1 ][t1 ][t2 ][t2 ][t3 ][t3 ]  ← group 0, quad 0
row 1:  [t4 ][t4 ][t5 ][t5 ][t6 ][t6 ][t7 ][t7 ]  ← group 0, quad 1
row 2:  [t0 ][t0 ][t1 ][t1 ][t2 ][t2 ][t3 ][t3 ]
row 3:  [t4 ][t4 ][t5 ][t5 ][t6 ][t6 ][t7 ][t7 ]
  ...          （group 0 覆盖 row 0-7）
row 8:  [t0 ][t0 ][t1 ][t1 ][t2 ][t2 ][t3 ][t3 ]  ← group 1, quad 0
row 9:  [t4 ][t4 ][t5 ][t5 ][t6 ][t6 ][t7 ][t7 ]  ← group 1, quad 1
  ...          （group 1 覆盖 row 8-15）
row 15: [t4 ][t4 ][t5 ][t5 ][t6 ][t6 ][t7 ][t7 ]

t0-t3 = quad 0 的 4 个线程（lane 0,1,2,3）
t4-t7 = quad 1 的 4 个线程（lane 4,5,6,7）
每个 quad 有 4 组（lane 8-31 中每 8 个 lane 一组）
每线程持有 4 个 fp32：(d0, d1) 来自 group 0，(d2, d3) 来自 group 1
```

这意味着：同一行的 8 列分布在 quad 的 4 个线程中（每线程 2 列）。因此 softmax 等行级归约需要 quad 内通信（`__shfl_xor_sync` 跨 4 线程）。

### 2.3 MMA_Traits：线程-值布局的编译时描述

**源码位置**: [atom/mma_traits_sm80.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm80.hpp)

`MMA_Traits` 为每种 PTX 指令特化，用 Layout 描述 32 个线程和每线程的值如何映射到 M×N×K 的矩阵坐标：

```cpp
template <>
struct MMA_Traits<SM80_16x8x16_F32BF16BF16F32_TN> {
    using ValTypeD = float;
    using ValTypeA = bfloat16_t;
    using ValTypeB = bfloat16_t;
    using ValTypeC = float;

    // 线程-值 → (M, K) 坐标的映射
    // Shape<(4 threads × 8 threads), (2 × 2 × 2 values)>
    using ALayout = Layout<
        Shape <Shape <_4, _8>, Shape <_2, _2, _2>>,
        Stride<Stride<_32, _1>, Stride<_16, _8, _128>>>;

    // 线程-值 → (N, K) 坐标的映射
    using BLayout = Layout<
        Shape <Shape <_4, _8>, Shape <_2, _2>>,
        Stride<Stride<_32, _1>, Stride<_16, _8>>>;

    // 线程-值 → (M, N) 坐标的映射
    using CLayout = Layout<
        Shape <Shape <_4, _8>, Shape <_2, _2>>,
        Stride<Stride<_32, _1>, Stride<_16, _8>>>;
};
```

每个 Layout 有两层嵌套 Shape：

- **外层**：线程维度。`Shape<_4, _8>` 表示 32 个线程被组织为 4×8 的二维网格
- **内层**：值维度。`Shape<_2, _2>` 表示每个线程持有 2×2 = 4 个值

Stride 将 `(thread_id, value_id)` 映射到矩阵中的线性偏移。这些布局完全是编译时常量，不产生任何运行时开销。

### 2.4 MMA_Atom：Traits 的可调用包装

**源码位置**: [atom/mma_atom.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_atom.hpp)

`MMA_Atom` 继承 `MMA_Traits`，增加 `call()` 方法和 fragment 创建接口：

```cpp
template <class MMA_Op>
struct MMA_Atom : MMA_Traits<MMA_Op> {
    // 从 Traits 继承 ValTypeA/B/C/D 和 ALayout/BLayout/CLayout

    // 执行单条 MMA 指令
    void call(D& d, A const& a, B const& b, C const& c) {
        MMA_Op::fma(d, a, b, c);
    }

    // 创建 fragment tensor
    auto make_fragment_C(Tensor<...> const& tCgC) {
        return make_tensor<FrgTypeC>(shape(tCgC));
    }
};
```

### 2.5 TiledMMA：从单条指令到完整 Tile

**源码位置**: [atom/mma_atom.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_atom.hpp)

单条 MMA 只计算 16×8×16。实际 kernel 需要更大的 tile（如 64×64×16）。`TiledMMA` 通过**线程复制**将多个 MMA atom 铺满目标 tile：

```cpp
template <class MMA_Atom, class AtomLayoutMNK, class PermutationMNK>
struct TiledMMA {
    // AtomLayoutMNK: 如何在 M/N/K 维度复制 atom
    //   例如 Layout<Shape<_4, _1, _1>> → 4 个 warp 沿 M 维排列
    //
    // PermutationMNK: 可选的 MNK 重排列
    //   例如 Tile<_64, _16, _16> → 目标 tile 尺寸

    // 从线程索引获取该线程的 MMA 操作器
    auto get_slice(int thread_idx);

    // 将 Tensor 从 (M,N)/(M,K)/(N,K) 空间变换到线程-fragment 布局
    auto thrfrg_A(Tensor const& gA);  // (M,K) → ((ThrV,(ThrM,ThrK)), (FrgV,(RestM,RestK)))
    auto thrfrg_B(Tensor const& gB);  // (N,K) → ((ThrV,(ThrN,ThrK)), (FrgV,(RestN,RestK)))
    auto thrfrg_C(Tensor const& gC);  // (M,N) → ((ThrV,(ThrM,ThrN)), (FrgV,(RestM,RestN)))
};
```

**实际使用示例**（Sparse Attention kernel 中）：

```cpp
using TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>,   // 基础 atom: 16×8×16
    Layout<Shape<Int<kNWarps>, _1, _1>>,          // kNWarps 个 warp 沿 M 维复制
    Tile<Int<16 * kNWarps>, _16, _16>             // 目标 tile: kBlockM × 16 × 16
>;
```

当 `kNWarps = 4` 时：
- 4 个 warp × 32 线程/warp = 128 线程
- 每个 warp 负责 M 维的 16 行：warp 0 → M[0,16), warp 1 → M[16,32), warp 2 → M[32,48), warp 3 → M[48,64)
- 结果 tile 为 64×16×16
- 完整的 GEMM（如 64×64×K）需要沿 N 以 16 为步长迭代 4 次，沿 K 以 16 为步长迭代 K/16 次

**`get_slice(thread_idx)` → `ThrMMA`**：返回一个 per-thread 的 MMA 操作器，提供：
- `partition_A/B/C(tensor)`：将全局/共享内存 Tensor 切分为该线程负责的分片
- `partition_fragment_A/B/C(tensor)`：创建正确形状的寄存器 fragment
- fragment 的布局由 MMA_Traits 中的 ALayout/BLayout/CLayout 自动决定

---

## 3. Copy Atom：内存搬运指令抽象

CuTe 将不同路径的数据搬运（global→smem、smem→register 等）统一封装为 Copy_Atom → TiledCopy 的两层抽象，与 MMA 侧的 MMA_Atom → TiledMMA 完全对称。

### 3.1 底层 PTX 指令

CuTe 封装了三类核心搬运指令：

**cp.async（SM80，global → smem）**

**源码位置**: [arch/copy_sm80.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm80.hpp)

```cpp
// SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>
// 16 字节（128 bit）异步搬运，绕过寄存器直达 smem
asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
    :: "r"(smem_int_ptr),      // 目标：smem 地址（uint32 编码）
       "l"(gmem_ptr),          // 源：global 地址（64-bit 指针）
       "n"(sizeof(TS)));       // 搬运字节数（编译时常量，16 字节）
```

四个变体：

| 结构体 | PTX | 缓存策略 | 特殊功能 |
|--------|-----|----------|----------|
| `SM80_CP_ASYNC_CACHEALWAYS<TS>` | `cp.async.ca.shared.global` | 缓存到 L1+L2 | 支持 4/8/16 字节 |
| `SM80_CP_ASYNC_CACHEGLOBAL<TS>` | `cp.async.cg.shared.global` | 仅缓存到 L2 | 仅 16 字节 |
| `SM80_CP_ASYNC_CACHEALWAYS_ZFILL<TS>` | `cp.async.ca` + pred | L1+L2 | pred=false 时填零 |
| `SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<TS>` | `cp.async.cg` + pred | 仅 L2 | pred=false 时填零 |

**ZFILL 变体**带有 `bool pred` 参数——当 `pred=false` 时 `src_size=0`，指令不读全局内存，目标 smem 位置填零。这用于越界保护：对超出有效范围的位置填零而非读越界地址。

```cpp
// ZFILL 变体的关键区别
int src_size = pred ? sizeof(TS) : 0;  // pred=false → 不搬运，填零
asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n"
    :: "r"(smem_int_ptr), "l"(gmem_ptr),
       "n"(sizeof(TS)),                   // 最大字节数（编译时常量）
       "r"(src_size));                    // 实际字节数（运行时，0 = 填零）
```

**cp.async vs 传统路径**：

```
传统路径：global ──ld.global──→ register ──st.shared──→ smem
          两条指令，寄存器占用增加，线程阻塞等待 ld.global 完成

cp.async：global ─────cp.async────────────────────→ smem
          一条指令，绕过寄存器，线程不阻塞（异步）
          需要 commit_group / wait_group 同步
```

**`.cg` vs `.ca` 缓存策略**：
- `.ca`（cache all）：数据缓存到 L1 和 L2。适合会被多次读取的小数据
- `.cg`（cache global）：数据只缓存到 L2，不污染 L1。适合大块数据搬运（如 KV tile），避免 L1 被大量数据刷新

**ldmatrix（SM75，smem → register）**

**源码位置**: [arch/copy_sm75.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm75.hpp)

`ldmatrix` 是专为 Tensor Core 设计的 smem→register 加载指令。它的特点是将 8×8 的 16-bit 矩阵片段加载到寄存器中，布局直接匹配 `mma.sync` 指令的寄存器要求。

```cpp
// SM75_U32x4_LDSM_N：加载 4 个 8×8 矩阵片段（Normal，不转置）
struct SM75_U32x4_LDSM_N {
    using SRegisters = uint128_t[1];   // 源：smem 地址
    using DRegisters = uint32_t[4];    // 目标：4 个 32-bit 寄存器

    static void copy(uint128_t const& smem_src,
                     uint32_t& dst0, uint32_t& dst1,
                     uint32_t& dst2, uint32_t& dst3) {
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
            :  "r"(smem_int_ptr));
    }
};
```

**ldmatrix 家族**：

| 结构体 | PTX | 加载量 | 用途 |
|--------|-----|--------|------|
| `SM75_U32x1_LDSM_N` | `ldmatrix.sync.aligned.x1.m8n8.shared.b16` | 1×32bit | 单片段 |
| `SM75_U32x2_LDSM_N` | `ldmatrix.sync.aligned.x2.m8n8.shared.b16` | 2×32bit | 双片段 |
| `SM75_U32x4_LDSM_N` | `ldmatrix.sync.aligned.x4.m8n8.shared.b16` | 4×32bit = 16B | 四片段（最常用） |
| `SM75_U16x2_LDSM_T` | `ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16` | 1×32bit | 转置加载 |
| `SM75_U16x4_LDSM_T` | `ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16` | 2×32bit | 转置加载 |
| `SM75_U16x8_LDSM_T` | `ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16` | 4×32bit = 16B | 转置加载（最常用） |

**`.trans` 后缀**：在从 smem 加载到寄存器时自动转置。这意味着 V 矩阵可以在 smem 中保持行主序存储，用 `ldmatrix.trans` 加载时完成硬件转置，无需额外的内存搬运或 smem 中的物理转置。

**ldmatrix 的工作方式**：warp 内 32 个线程各提供一个 smem 地址（同一 128-bit 行内）。`.x4` 时从 4 组地址中各加载 128 bit，按 Tensor Core 要求分发到各线程的寄存器中。这保证了加载后的寄存器分布直接匹配 `mma.sync` 的操作数布局。

### 3.2 Copy_Traits：线程-值布局描述

**源码位置**: [atom/copy_traits_sm80.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm80.hpp)

与 MMA_Traits 类似，`Copy_Traits` 为每种搬运指令描述线程和值的布局：

```cpp
template <class S, class D>
struct Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<S, D>> {
    using ThrID     = Layout<_1>;          // 单线程指令
    using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;  // (1 thread, N bits)
    using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;
    using RefLayout = SrcLayout;           // 参考布局 = 源布局
};
```

`cp.async` 是单线程指令（`ThrID = Layout<_1>`），每次一个线程搬运 `sizeof(TS)` 字节。值布局按 bit 粒度描述，由上层 recast 到实际数据类型。

ZFILL 变体的 Copy_Traits 额外存储 `bool pred` 成员，通过 `with(pred)` 工厂方法设置。

### 3.3 Copy_Atom 与 TiledCopy

**源码位置**: [atom/copy_atom.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_atom.hpp)

**Copy_Atom**：包装 Copy_Traits，提供 `call(src, dst)` 执行接口。类似 MMA_Atom 之于 MMA_Traits。

```cpp
template <class Copy_Op, class ValType>
struct Copy_Atom : Copy_Traits<Copy_Op> {
    // 从 Traits 继承 ThrID、SrcLayout/DstLayout/RefLayout
    // Recast bit layouts to ValType granularity

    using ValLayoutSrc = /* SrcLayout recast to ValType */;
    using ValLayoutDst = /* DstLayout recast to ValType */;

    // 执行搬运
    void call(Tensor const& src, Tensor& dst) {
        if (size matches instruction)
            copy_unpack(src, dst);     // 直接调用 PTX
        else
            recursively partition and copy;
    }
};
```

**TiledCopy**：将 Copy_Atom 与线程排列布局组合，描述多个线程如何协作搬运一个完整 tile：

```cpp
template <class Copy_Atom, class LayoutCopy_TV, class ShapeTiler_MN>
struct TiledCopy {
    // LayoutCopy_TV: (thread_id, value_id) → (M, N) 坐标
    // ShapeTiler_MN: tile 大小

    auto get_slice(int thread_idx);  // → ThrCopy

    // 变换到线程-fragment 布局
    auto tidfrg_S(Tensor const& src);  // → ((ThrV, ThrX), FrgV, (RestM, RestN))
    auto tidfrg_D(Tensor const& dst);
};
```

**`make_tiled_copy`**：便捷工厂函数，从 atom + 线程布局 + 值布局构造 TiledCopy：

```cpp
auto tiled_copy = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>{},
    Layout<Shape<_16, _8>>{},    // 线程排列：16×8 = 128 线程
    Layout<Shape<_1, _8>>{}      // 每线程每次搬运 8 个元素
);
```

**`ThrCopy`（per-thread 切片）**：
- `partition_S(tensor)`：将源 Tensor 切分为该线程负责的分片
- `partition_D(tensor)`：将目标 Tensor 切分为该线程负责的分片
- `retile_S/D(tensor)`：重排 Tensor 的模式以匹配搬运布局

### 3.4 cute::copy 的分发逻辑

CuTe 的高层 `copy(atom, src, dst)` 根据 src 和 dst 的**地址空间**自动选择 PTX 指令：

| 源 | 目标 | 自动选择的指令 |
|-----|------|--------------|
| global（`make_gmem_ptr`） | smem（`make_smem_ptr`） | `cp.async`（SM80+） |
| smem | register | `ldmatrix`（如果 atom 匹配） |
| register | smem | 普通 `st.shared` |
| register | register | 寄存器拷贝 |

这就是为什么 CuTe 代码中看不到手动选择指令——地址空间标记让编译器自动分发到最优路径。

---

## 4. cp.async 异步流水线

### 4.1 commit_group 与 wait_group

`cp.async` 指令是异步的——发射后线程立即继续执行，数据搬运在后台进行。通过 **group** 机制管理同步：

```cpp
// 步骤 1：发射多条 cp.async（它们属于同一个"未提交"的 group）
cute::copy(atom, gmem_src_0, smem_dst_0);
cute::copy(atom, gmem_src_1, smem_dst_1);
// ...

// 步骤 2：提交当前 group（将上面的 cp.async 打包为一个 group）
cute::cp_async_fence();
// PTX: cp.async.commit_group;

// 步骤 3：等待 group 完成
cute::cp_async_wait<0>();
// PTX: cp.async.wait_group 0;   等待所有 group 完成
__syncthreads();

// 步骤 4：现在可以安全读取 smem 数据
```

**`cp.async.wait_group N`** 的语义：等待直到未完成的 group 数 $\leq N$。

| N | 含义 | 场景 |
|---|------|------|
| 0 | 等待**所有** group 完成 | 最简单，本 kernel 使用 |
| 1 | 允许最多 1 个 group 还在进行中 | 双缓冲流水线 |
| 2 | 允许最多 2 个 group 还在进行中 | 三级流水线（如 FlashAttention-2） |

### 4.2 流水线模式

**简单模式**（本 kernel 使用）：每次加载后 `wait<0>` 全部等待完成。清晰正确，但加载和计算无法重叠。

```
fence    wait<0>   sync      计算
  │        │        │        │
K[0] ═══╗  │        │        │
        ║──┘        │        │
                    └──── GEMM-I ────→ fence    wait<0>   sync
                                         │        │        │
                               V[0] ═══╗ │        │        │
                                       ║──┘       │        │
                                                  └──── GEMM-II
```

**双缓冲模式**（高级，如 FlashAttention-2）：在计算当前 tile 的同时，预加载下一个 tile 到另一块 smem。使用 `wait<1>` 允许一个 group 还在飞行中：

```
fence(Q)  fence(K[0])  wait<0>+sync  fence(V[0])  wait<1>+sync
  │          │            │              │              │
Q ══╗   K[0]══╗           │         V[0]══╗        GEMM-I(K[0])
    ║        ║            │              ║         同时 K[1] 在飞
    ║        ║────────────┘              ║
    ║────────┘   Q→reg                   ║─────────┘
```

这需要两倍的 smem 缓冲区（KV 各两份），但能显著提升吞吐量。

### 4.3 __syncthreads 的必要性

`cp.async.wait_group` 只保证**本线程**发射的 cp.async 完成。但 smem 是 block 内共享的——线程 A 写入的 smem 位置可能被线程 B 读取。因此在 `wait` 之后必须加 `__syncthreads()` 确保**所有线程**的写入对所有线程可见。

```cpp
cute::cp_async_fence();      // commit
cute::cp_async_wait<0>();    // 等待本线程的 cp.async 完成
__syncthreads();             // 等待所有线程完成 + smem 可见性
// 现在安全读取 smem
```

---

## 5. Swizzle 与共享内存 Bank Conflict

### 5.1 Bank Conflict 问题

GPU 共享内存被组织为 **32 个 bank**，每个 bank 4 字节宽。同一 warp 内 32 个线程**同一时钟周期**访问 smem 时：

- **无冲突**：32 个线程各访问不同 bank → 一个周期完成
- **冲突**：多个线程访问同一 bank 的不同地址 → 串行化，性能骤降
- **广播**：多个线程访问同一 bank 的同一地址 → 无冲突（硬件广播）

朴素行主序布局的典型冲突场景：

```
smem_k [64 × 128] (bf16)，每行 = 256 字节 = 64 个 bank 槽（每槽 4B）

行 0: [bank0][bank1]...[bank31][bank0][bank1]...[bank31]
行 1: [bank0][bank1]...[bank31][bank0][bank1]...[bank31]
...
行 7: [bank0][bank1]...[bank31][bank0][bank1]...[bank31]

当 ldmatrix 加载某一列时：
  行 0 的 col 0 → bank 0
  行 1 的 col 0 → bank 0   ← 冲突！
  行 2 的 col 0 → bank 0   ← 冲突！
  ...
8 行同列全部映射到 bank 0 → 8-way bank conflict
```

### 5.2 Swizzle 原理

**源码位置**: [swizzle.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp)

Swizzle 通过 **XOR 变换**将相邻行的列地址"搅乱"，使原本映射到同一 bank 的元素分散到不同 bank。

```cpp
template <int BBits, int MBase, int SShift = BBits>
struct Swizzle {
    static constexpr int num_bits = BBits;   // mask 宽度（BBits 个 bit）
    static constexpr int num_base = MBase;   // 最低 MBase 位保持不变
    static constexpr int num_shft = SShift;  // YYY 与 ZZZ 之间的位距

    // 位域掩码
    using bit_msk = constant<int, (1 << num_bits) - 1>;
    using yyy_msk = constant<int, bit_msk{} << (num_base + max(0, num_shft))>;
    using zzz_msk = constant<int, bit_msk{} << (num_base - min(0, num_shft))>;

    // 核心操作：ZZZ ^= YYY
    static auto apply(Offset const& offset) {
        return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});
    }
};
```

**位域示意**：

```
地址 offset 的二进制表示：
  0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                                ^--^ MBase：最低 MBase 位保持不变
                   ^-^       ^-^     BBits：YYY 和 ZZZ 各有 BBits 位
                     ^---------^     SShift：YYY 相对 ZZZ 的偏移距离

swizzle(offset) = offset，但 ZZZ 位被替换为 ZZZ XOR YYY
```

YYY 位通常对应行号（高位），ZZZ 位对应列地址中决定 bank 的位（低位）。XOR 后，不同行的同一列被映射到不同 bank。

**`make_swizzle<Y, Z>()`**：从两个 bit mask 自动推导 `Swizzle<BBits, MBase, SShift>` 参数：

```cpp
make_swizzle<0b1000, 0b0100>()         // → Swizzle<1, 2, 1>
make_swizzle<0b11000000, 0b00000110>() // → Swizzle<2, 1, 5>
```

**`composition(Swizzle, Swizzle)`**：合并两个相同 shift 的 Swizzle。两个 Swizzle 的 YYY/ZZZ mask 做 XOR 合并。

### 5.3 实际配置示例：Swizzle<3, 3, 3>

以 D=128、Element=bf16 为例：

```
每行字节数 = 128 × 2B = 256 字节
256 / 128 = 2 → 按 128 字节对齐
kBlockKGmem = 128 / 2 = 64 个元素

kSwizzle = 3（覆盖 8 个值 = 16 字节 = 4 个 bank）
kSwizzleBase = 3（跳过最低 3 位 = 8 字节 = 4 个 bf16）

→ Swizzle<3, 3, 3>
```

**BBits=3**：mask 有 3 位，覆盖 $2^3 = 8$ 个值

**MBase=3**：跳过最低 3 位。这 3 位对应 $2^3 = 8$ 字节 = 4 个 bf16 = 1 个 bank 组（bank 内偏移），它们不参与 swizzle（同一 bank 内的不同字节不会冲突）

**SShift=3**：YYY 在 ZZZ 上方 3 位

```
地址的 bit 分解（以 bf16 元素为单位）：
  bit: ... [8 7 6] [5 4 3] [2 1 0]
              YYY     ZZZ    base（不变）

swizzle 后: bit[5:3] ^= bit[8:6]

效果（以 col 0 为例）：
  行 0 col 0 → bank = (000 XOR 000) = 0
  行 1 col 0 → bank = (000 XOR 001) = 1
  行 2 col 0 → bank = (000 XOR 010) = 2
  行 3 col 0 → bank = (000 XOR 011) = 3
  行 4 col 0 → bank = (000 XOR 100) = 4
  行 5 col 0 → bank = (000 XOR 101) = 5
  行 6 col 0 → bank = (000 XOR 110) = 6
  行 7 col 0 → bank = (000 XOR 111) = 7
  → 连续 8 行的同一列完全分散到 8 个不同 bank，零冲突
```

**SmemLayoutAtom**：Swizzle 与一个 8×64 的基础 atom 组合，然后通过 `tile_to_shape` 平铺到完整的 64×128 tile。

### 5.4 SmemLayoutVt：零开销逻辑转置

GEMM-II 计算 $O \mathrel{+}= P \cdot V$。按 `TN` 布局要求，B 操作数必须是列主序。V 在 smem 中是行主序 `[kBlockN, kHeadDim]`——需要以 `[kHeadDim, kBlockN]` 的方式访问。

```cpp
using SmemLayoutVt = decltype(
    composition(SmemLayoutV{},
                make_ordered_layout(
                    make_shape(Int<kHeadDim>{}, Int<kBlockN>{}),
                    Step<_2, _1>{})));
```

`make_ordered_layout(..., Step<_2, _1>{})` 创建 `[kHeadDim, kBlockN]` 形状的列主序逻辑布局（第 2 维的 stride 更大）。`composition(SmemLayoutV{}, ...)` 将这个逻辑布局映射到 SmemLayoutV 的物理 swizzled 地址上。

结果：`sVt(d, n)` 访问的物理地址等于 `sV(n, d)` 的地址——**不移动任何数据**，只是改变了索引方式。配合 `ldmatrix.trans` 指令，在从 smem 加载到寄存器时完成硬件转置。整个转置操作零内存开销。

### 5.5 SharedStorage：Union 复用 smem

典型的 Attention kernel 使用 `union` 让不同阶段的数据复用同一块 smem：

```cpp
union SharedStorage {
    struct {
        Element smem_q[kBlockM * kHeadDim];   // Q tile（整个 kernel 生命周期）
        Element smem_k[kBlockN * kHeadDim];   // K tile（GEMM-I 阶段）
    };
    struct {
        Element __pad[kBlockM * kHeadDim];    // 占位（Q 不被覆盖）
        Element smem_v[kBlockN * kHeadDim];   // V tile（GEMM-II 阶段，复用 K 的空间）
    };
    struct {
        Element smem_o[kBlockM * kHeadDim];   // O tile（输出阶段，复用 Q+K 的空间）
    };
};
```

`union` 保证三个结构体共享同一起始地址。K 和 V 不会同时使用，因此复用同一块内存。Q 在整个 KV block 循环中需要保持，所以放在前面不被覆盖。输出阶段 Q/K/V 都不再需要，O 复用全部空间。

这种设计将 smem 用量从 $\text{Q} + \text{K} + \text{V} + \text{O}$ 降为 $\max(\text{Q}+\text{K},\ \text{Q}+\text{V},\ \text{O})$。

---

## 6. 完整数据流

以下是一个典型的 CuTe-based Attention kernel 的端到端数据流，展示上述所有概念如何协同工作：

```
全局内存                          共享内存                          寄存器                  Tensor Core
─────────                     ─────────                      ──────────               ────────────

Q [kBlockM×D]                 smem_q [kBlockM×D]             tSrQ [fragment]
  │                              │                              │
  ├── cp.async (TiledCopy) ──→  │ (Swizzled layout)            │
  │   fence + wait<0> + sync    │                              │
  │                              ├── ldmatrix (Copy_Atom) ───→ │
  │                              │                              │
  │                              │                              │
K[i] [64×D]                   smem_k [64×D]                  tSrK [fragment]
  │                              │                              │
  ├── cp.async.cg.zfill ──────→│ (Swizzled layout)            │
  │   fence + wait<0> + sync    │                              │
  │                              ├── ldmatrix ────────────────→│
  │                              │                              ├── mma.sync ──→ tSrS (S=Q·K^T)
  │                              │                              │                   │
  │                              │                    mask ────→│                   │
  │                              │                              │  online softmax   │
  │                              │                              │  (exp2f + rescale)│
  │                              │                              ├── tSrP (fp32→bf16)│
  │                              │                              │                   │
V[i] [64×D]                   smem_v [64×D]                  tOrV [fragment]       │
  │                              │                              │                   │
  ├── cp.async.cg.zfill ──────→│ sVt = composition 逻辑转置   │                   │
  │   fence + wait<0> + sync    │                              │                   │
  │                              ├── ldmatrix.trans ──────────→│                   │
  │                              │                              ├── mma.sync ──→ tOrO (O+=P·V)
  │                              │                              │
                                smem_o [kBlockM×D]              │
O [q_bs×D] ←── global ←────── │ ←── register store ←──────── │
LSE [q_bs]  ←── global ←──────────── register 直接写 ←─────── softmax.row_sum
```

**关键设计决策**：

- **cp.async + ZFILL**：global→smem 异步搬运，越界位置自动填零
- **Swizzle**：smem 布局经过 XOR 变换，消除 `ldmatrix` 按列加载时的 bank conflict
- **ldmatrix + ldmatrix.trans**：smem→register 加载，寄存器分布直接匹配 mma.sync 要求。V 的转置由 composition + ldmatrix.trans 零开销完成
- **TiledMMA**：多个 warp 沿 M 维并行，通过 partition_fragment 自动分配寄存器
- **union SharedStorage**：K/V 复用 smem 空间，降低内存占用
- **Online Softmax + exp2f**：单遍流式处理，exp2f 单条 PTX 指令替代 expf 的多条指令

