---
tags:
  - CUTLASS
  - CUDA
---

# CuTe GEMM 教程

> **原文出处**: [NVIDIA/cutlass - media/docs/cute/0x_gemm_tutorial.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md)
> **许可证**: BSD-3-Clause, Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES

本节回顾 [CUTLASS 示例](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/)中的几个独立、单文件的稠密矩阵乘法实现，全部仅使用 CuTe。

## `sgemm_1.cu`

最简单的教程示例，涵盖将全局内存分块到 CTA（也称 CUDA threadblock）、将数据 tile 分区给 CTA 内的线程、以及使用 `cute::copy` 和 `cute::gemm` 编写主循环的基础知识。

### 高层接口

Kernel 入口 `gemm_device` 的模板参数概览：

- `ProblemShape`：矩阵乘法的 MxNxK 问题 shape
- `CtaTiler`：CuTe [tiler 概念](02_layout_algebra.md)，决定如何从问题 shape 中提取数据 tile
- `TA const* A`、`TB const* B`、`TC* C`：A/B/C 数据的类型和指针
- `AStride`、`BStride`、`CStride`：对应 ProblemShape 的 layout 步长
- `ASmemLayout`、`BSmemLayout`、`CSmemLayout`：各阶段共享内存的 layout
- `AThreadLayout`、`BThreadLayout`、`CThreadLayout`：分区各阶段时使用的线程 layout
- `Alpha alpha`、`Beta beta`：GEMM 标量常数 $C = \alpha \cdot A \cdot B + \beta \cdot C$

### 完整 Tensor：Shape、Stride 和数据

问题维度 M、N、K 打包为单个 IntTuple：

```cpp
auto M = int(m);
auto N = int(n);
auto K = int(k);
auto prob_shape = make_shape(M, N, K);    // (M, N, K)
```

在 kernel 内部构造完整矩阵：

```cpp
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);  // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);  // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);  // (M,N)
```

注意 B 采用 `(N,K)` 惯例而非 `(K,N)`。CuTe 的模式语义惯例：A 为 `(M,K)`、B 为 `(N,K)`、C 为 `(M,N)`。

#### 补充说明：M-major、N-major、K-major

CuTe 不使用 BLAS 的"非转置"(N)/"转置"(T) 标志，而是直接说矩阵在哪个模式上步长为 1：

| BLAS | A 主序 | A Layout | B 主序 | B Layout |
| --- | --- | --- | --- | --- |
| NT | M-major | `(M,K):(1,ldA)` | N-major | `(N,K):(1,ldB)` |
| TN | K-major | `(M,K):(ldA,1)` | K-major | `(N,K):(ldB,1)` |
| NN | M-major | `(M,K):(1,ldA)` | K-major | `(N,K):(ldB,1)` |
| TT | K-major | `(M,K):(ldA,1)` | N-major | `(N,K):(1,ldB)` |

### CTA 分区

最高层将工作分配给 CTA。定义 CTA tile 大小并使用 tiler：

```cpp
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int<  8>{};
auto cta_tiler = make_shape(bM, bN, bK);  // (BLK_M, BLK_N, BLK_K)
```

分区 tensor：

```cpp
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
```

`local_tile` 本质上是 `zipped_divide` + 切片：先用 tiler 分块，再用坐标索引 rest 模式提取当前 CTA 的 tile。

对于 tensor A，结果是 shape 为 `(BLK_M,BLK_K,k)` 的 rank-3 tensor。前两个模式是 CTA tile 的模式，最后一个模式遍历该 CTA 负责归约的所有 tile。

### SMEM Tensor

共享内存 layout 作为参数传入（静态 layout），用于暂存 A 和 B 的数据 tile。NT 情况下是简单的 M-major 和 N-major layout，TN 情况下使用 K-major layout。

静态 layout 的优势：静态分配共享内存、通常更高效、易于正确性证明和检查。

```cpp
__shared__ TA smemA[cosize_v<ASmemLayout>];
__shared__ TB smemB[cosize_v<BSmemLayout>];
Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);  // (BLK_M,BLK_K)
Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);  // (BLK_N,BLK_K)
```

注意用 `cosize` 而非 `size` 来确定分配大小——Layout 作为函数的陪域大小保证所有偏移有效。

### Copy 分区

将全局内存 tile 和共享内存 tile 按线程 layout 分区：

```cpp
Tensor tAgA = local_partition(gA, tA, threadIdx.x);    // (THR_M,THR_K,k)
Tensor tAsA = local_partition(sA, tA, threadIdx.x);    // (THR_M,THR_K)
```

`local_partition` 类似 `local_tile`，但坐标切入 tile 模式（第一个模式）而非 rest 模式。每个线程在每个线程 tile 中获得一个数据元素，该线程 tile 重复以覆盖整个数据 tile。

命名惯例 `tAsA` 读作"分区模式 tA 应用于 tensor sA"。通过对 sA 和 gA 应用相同的分区模式 tA，保留两个 tensor 的**逻辑一致性**。

所有线程参与拷贝：

```cpp
copy(tAgA(_,_,0), tAsA);
```

### Math 分区

用 16x16 线程 layout `tC` 分区输出 tile gC，每个线程计算自己的 8x8 子 tensor：

```cpp
// 按 tC 的行分区 sA (M,K)
Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
// 按 tC 的列分区 sB (N,K)
Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
// 按 tC 分区 gC (M,N)
Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

// 分配累加器
Tensor tCrC = make_tensor_like(tCgC);                                // (THR_M,THR_N)
```

所有线程参与计算：

```cpp
gemm(tCsA, tCsB, tCrC);
```

### 主循环

主循环遍历全局内存 tile，读入共享内存，然后执行矩阵乘累加：

```c++
auto K_TILE_MAX = size<2>(tAgA);

for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
{
  // 用 tA|tB 线程分区 tensor 从 gmem 拷贝到 smem
  copy(tAgA(_,_,k_tile), tAsA);
  copy(tBgB(_,_,k_tile), tBsB);

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  // 用 tC 线程分区的 smem 执行 gemm
  gemm(tCsA, tCsB, tCrC);
  __syncthreads();
}
```

## `sgemm_2.cu`

使用更复杂的 `TiledMMA` 和 `TiledCopy` 替代 tA/tB/tC 线程 layout 进行分区。强调共享内存 layout、分区模式和 PTX 指令可以独立指定。

### TiledCopy

用 `TiledCopy` 替代 tA 分区，提供更复杂的分区模式和到特定拷贝指令的校验分发：

```cpp
TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
                                  Layout<Shape<_32,_8>>{},       // 32x8 m-major 线程 layout
                                  Layout<Shape< _4,_1>>{});      // 4x1 m-major 值 layout
```

每个线程读取 4x1 个 TA 元素，有 32x8 个线程。`UniversalCopy<uint128_t>` 强制使用 128 位拷贝指令。

使用：

```cpp
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);            // (CPY,CPY_M,CPY_K,k)
Tensor tAsA = thr_copy_a.partition_D(sA);            // (CPY,CPY_M,CPY_K)
```

`partition_S` 应用源 tensor 分区，`partition_D` 应用目标 tensor 分区。第一个模式 `CPY` 包含单条指令消费的所有元素。

### TiledMMA

用 `TiledMMA` 替代 tC 分区：

```cpp
TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                               Layout<Shape<_16,_16,_1>>{});  // 16x16x1 UniversalFMA
```

使用：

```cpp
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA);        // (MMA,MMA_M,MMA_K)
Tensor tCsB = thr_mma.partition_B(sB);        // (MMA,MMA_N,MMA_K)
Tensor tCgC = thr_mma.partition_C(gC);        // (MMA,MMA_M,MMA_N)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA,MMA_M,MMA_N)
```

### 其他改进

此版本中 `gemm_tn` 的共享内存 layout 从 K-major 改为加 padding 的 M-major/N-major 以避免 bank conflict：

```cpp
auto sA = make_layout(make_shape (      bM,          bK),
                      make_stride(Int<1>{}, bM+Int<1>{}));  // padded m-major
```

## `sgemm_sm70.cu`

针对 Volta SM70 架构的优化主循环，流水线化共享内存和寄存器内存。

## `sgemm_sm80.cu`

针对 Ampere SM80 架构的优化主循环，显式使用异步全局内存读取流水线化共享内存。包含 SIMT（NT/TN）和 Tensor Core（TN HGEMM）三个版本，涵盖 cp.async 多级流水线、Swizzle SMEM layout、ldmatrix retiling 等完整特性。

详细拆解参见 [sgemm_sm80 实战拆解](09_sgemm_sm80.md)。

## GETT 作为 GEMM

"GETT" 即"广义张量×张量"（generalized tensor times tensor），一种张量收缩。

CuTe 允许矩阵具有嵌套 Layout，这意味着我们可以通过按类别分组模式来将 Tensor 折叠为"矩阵"。因此可以使用现有 GEMM 实现来计算 GETT。

示例——使用与 `sgemm_1.cu` 相同的 device kernel 计算具有两个 m 模式的 GETT：

```cpp
auto M = make_shape(m0, m1);                               // (m0,m1) 多模式 M
auto N = int(n);
auto K = int(k);
auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

auto dA = make_stride(make_stride(Int<1>{}, ldAm1), ldAk); // (dM, dK)
auto dC = make_stride(make_stride(Int<1>{}, ldCm1), ldCn); // (dM, dN)

auto bM = Shape<_64, _2>{};    // 从 m0 取 _64 个元素，从 m1 取 _2 个元素
auto bN = Int<128>{};
auto bK = Int<  8>{};
auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
```

唯一的改变是 M shape 变为多模式 `(m0,m1)`、对应步长和 CTA Tiler `bM = <_64,_2>` 变为多模式。kernel 本身无需修改。

## 下一步

上述所有示例假设 CTA tile 大小整除问题大小，因此全局内存加载不需要谓词化。关于不整除情况的处理，请参见[谓词教程](07_predication.md)。
