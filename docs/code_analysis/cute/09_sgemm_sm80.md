---
tags:
  - CUTLASS
  - CUDA
---

# CuTe 实战：sgemm_sm80.cu 拆解

> **源码**: [NVIDIA/cutlass - examples/cute/tutorial/sgemm_sm80.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu)
> **许可证**: BSD-3-Clause, Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES

[GEMM 教程](06_gemm_tutorial.md)概述了 `sgemm_1.cu`（手动线程 layout 分区）和 `sgemm_2.cu`（TiledCopy + TiledMMA），但对 `sgemm_sm80.cu` 只有一句话。本文以该文件的 **TN HGEMM**（half_t，Tensor Core）路径为主线，完整拆解一个面向 SM80（Ampere）的高性能 GEMM 实现，串联 CuTe 各核心概念。

## 文件总览

`sgemm_sm80.cu` 包含三个 GEMM 路径，**共用同一个 `gemm_device` kernel 模板**（第 54-310 行），所有差异都通过模板参数在 host 侧传入。这正是 CuTe 的核心设计：同一套 `local_tile` / `copy` / `gemm` 抽象，通过不同的 Atom 和 Layout 参数适配完全不同的硬件路径。

### 三个版本

**NT SIMT**（`gemm_nt<TA,TB,TC>`，第 432-503 行）：最简单的版本。

- A 是 M-major `(M,K):(1,ldA)`，B 是 N-major `(N,K):(1,ldB)` — 即 BLAS 的 NT（Non-Transposed × Transposed）
- CTA tile 128×128×**8**（BK=8，标量 FMA 吞吐量低，不需要大 K tile）
- SMEM layout：最简单的列主序 `make_layout(make_shape(bM, bK, bP))`。M-major 下连续线程沿 M 访问连续地址，天然无 bank conflict，无需 swizzle 或 padding
- TiledCopy G→S：`SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` + 32×8 M-major 线程布局 + 4×1 值布局。每线程搬 4 个元素 = 16B = 128b 向量化
- TiledMMA：`UniversalFMA<TC,TA,TB>` + 16×16×1 线程布局 = 256 线程各做标量 FMA
- S→R：`AutoVectorizingCopy`（普通寄存器拷贝，不用 ldmatrix）

**TN SIMT**（`gemm_tn<TA,TB,TC>`，第 505-580 行）：K-major 下的标量路径。

- A 是 K-major `(M,K):(ldA,1)`，B 是 K-major `(N,K):(ldB,1)` — 即 BLAS 的 TN
- CTA tile 128×128×**8**
- SMEM layout：**padding** 避免 bank conflict — `make_stride(Int<1>{}, bM+Int<1>{})`。K-major 下同一行的连续 K 元素被不同线程从 smem 读取时会落到相同 bank，加 1 个 padding 元素错开 bank 分配
- TiledCopy G→S：`SM80_CP_ASYNC_CACHEALWAYS<TA>` + 32×8 **K-major** 线程布局 + **1×1 值布局**。注意这里是**标量逐元素**搬运而非向量化——K-major 下连续线程沿 M 方向跨行访问 global memory，地址不连续，无法向量化
- TiledMMA / S→R：同 NT SIMT

**TN HGEMM**（`gemm_tn` 的 half_t 特化，第 325-430 行）：最完整的高性能路径。

- A/B 均 K-major，元素类型限定为 `half_t`
- CTA tile 128×128×**64**（BK=64，是 SIMT 版的 8 倍——Tensor Core 一条 mma.sync 就消费 16 个 K 元素，大 BK 维持高算术强度）
- SMEM layout：`Swizzle<3,3,3>` + `composition` + `tile_to_shape` 构建 swizzle 后的布局，消除 ldmatrix 列访问时的 bank conflict（详见[编程模型 §5 Swizzle](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)）
- TiledCopy G→S：`SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` + 16×8 K-major 线程布局 + **1×8 值布局**（每线程搬 8 个 half_t = 16B = 128b）。虽然同为 K-major，half_t 比 float 窄一半，8 个元素刚好 128b，可以向量化
- TiledMMA：`SM80_16x8x16_F16F16F16F16_TN` + **2×2 atom layout** + `Tile<_32,_32,_16>` = 4 个 MMA atom 铺满 32×32×16（详见 [MMA Atom](05_mma_atom.md)、[编程模型 §2.5 TiledMMA](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)）
- S→R：`SM75_U32x4_LDSM_N`（ldmatrix 指令，直接将 smem 数据加载到 Tensor Core 要求的寄存器布局，详见[编程模型 §3.1 ldmatrix](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)）
- **双层流水线**：SMEM 层 3 级 cp.async + 寄存器层 K_BLOCK 级预取

### CuTe 特性对比

| CuTe 特性 | NT SIMT | TN SIMT | TN HGEMM |
|-----------|---------|---------|----------|
| Tensor 构造 + CTA 分区 | ✓ | ✓ | ✓ |
| TiledCopy G→S（cp.async） | 128b 向量化 | 标量逐元素 | 128b 向量化 |
| TiledMMA | UniversalFMA 16×16×1 | UniversalFMA 16×16×1 | SM80_16x8x16 + 2×2 atom |
| SMEM 防 bank conflict | 无需 | padding (+1) | Swizzle<3,3,3> |
| ldmatrix S→R retiling | ✗ | ✗ | ✓ SM75_U32x4_LDSM_N |
| 多级 SMEM 流水线 | 3 级 cp.async | 3 级 cp.async | 3 级 cp.async |
| 寄存器级流水线 | ✗ | ✗ | ✓ K_BLOCK 预取 |
| BK（K tile 大小） | 8 | 8 | 64 |

TN HGEMM 覆盖了几乎所有 CuTe 核心特性，下文以此为主线展开。NT/TN SIMT 版本在[末尾](#simt-版本补充说明)做简要补充。

### 数据流全景

```
Global Memory
  │  cp.async (SM80_CP_ASYNC_CACHEALWAYS<uint128_t>)
  │  TiledCopy: 128 线程 × 128b/线程 = 2KB/次
  ▼
Shared Memory（3 级 pipeline，Swizzle<3,3,3> 消除 bank conflict）
  │  ldmatrix (SM75_U32x4_LDSM_N)
  │  make_tiled_copy_A/B: retile 到 MMA 寄存器布局
  ▼
Registers（K_BLOCK 级 pipeline，预取下一个 k_block）
  │  mma.sync (SM80_16x8x16_F16F16F16F16_TN)
  │  TiledMMA: 2×2 atom = 32×32×16
  ▼
Accumulator Registers
  │  axpby: C = α·acc + β·C
  ▼
Global Memory
```

---

## Host 侧配置：gemm_tn HGEMM

`gemm_tn` 的 half_t 特化版本（第 325-430 行）在 host 侧构造所有静态参数，然后传给通用 `gemm_device` kernel。以下逐一拆解每个参数的含义和设计意图。

### Stride 与 CTA Tile

```cpp
'''
TN 布局：A 和 B 都是 K-major
- A (M,K):(ldA, 1) → 沿 K 方向步长为 1，沿 M 方向跨 ldA 个元素
- B (N,K):(ldB, 1) → 同理
- C (M,N):(1, ldC) → M-major 输出
'''
auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int< 64>{};
auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
auto bP = Int<3>{};                                         // 3 级 pipeline
```

CuTe 的 stride 惯例与 BLAS 的 T/N 标记不同——直接用步长说明哪个维度连续。`Int<1>{}` 是编译时常量，编译器可以生成更高效的地址计算。关于 M-major / K-major 的完整对照，参见 [GEMM 教程 §M-major/K-major 表](06_gemm_tutorial.md)。

BK=64 远大于 SIMT 版的 8。原因：一条 `mma.sync.m16n8k16` 就消费 16 个 K 元素，128 个线程 × 4 个 MMA atom 每次内循环迭代消费 BK=64 的 K tile 中的 16 个，需要 K_BLOCK=64/16=4 次内循环迭代。大 BK 保证了足够的算术强度（compute-to-memory ratio）。

### SMEM Layout：Swizzle

```cpp
'''
Swizzle atom 构造：
1. 基础 atom: 8×64 的 K-major 布局 (Shape<_8, Shape<_8,_8>>, Stride<_8, Stride<_1,_64>>)
   - 外层 8 行，内层 8×8 = 64 列（分成两组各 8 列）
   - K-major: stride<_8, stride<_1, _64>> → 沿 K 方向步长 1
2. composition(Swizzle<3,3,3>{}, atom): XOR 变换消除 bank conflict
3. tile_to_shape: 将 8×64 atom 平铺到 128×64×3 (bM×bK×bP)
'''
auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                Layout<Shape <_8,Shape <_8, _8>>,
                                       Stride<_8,Stride<_1,_64>>>{});

auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
auto sC = make_layout(make_shape(bM, bN));
```

为什么需要 Swizzle？K-major 布局下，ldmatrix 需要沿 M/N 方向读取列数据。连续行的同一列地址会落到相同 bank，导致严重 bank conflict。`Swizzle<3,3,3>` 对地址的 bit[5:3]（决定 bank 分配的位）做 XOR bit[8:6]（行号位），使连续 8 行的同一列分散到 8 个不同 bank。详细原理参见[编程模型 §5.3 Swizzle<3,3,3> 示例](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)。

`tile_to_shape` 将 8×64 的 swizzle atom 平铺到目标 shape。第三维 `bP=3` 是 pipeline 深度——SMEM 中同时存在 3 个 tile 的缓冲区，支持 cp.async 的多级流水线。

### TiledCopy：Global → Shared Memory

```cpp
'''
G→S 拷贝配置：
- Copy_Atom: SM80_CP_ASYNC_CACHEALWAYS<uint128_t> — 每条指令搬 16 字节
- Thr layout: 16×8 K-major (Stride<_8,_1>) — 128 线程排成 16 行×8 列
  - 16 行覆盖 M/N 方向的 16 个位置
  - 8 列覆盖 K 方向的 8 个位置
- Val layout: 1×8 K-major — 每线程每次搬 8 个 half_t = 16B = 128b
  - 8 个值沿 K 方向连续，匹配 K-major 内存布局

一次 copy 调用搬运：16行 × (8列×8元素) = 16×64 = 1024 个 half_t = 2KB
需要 128/16 = 8 次迭代覆盖 BLK_M=128 行
'''
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
    Layout<Shape< _1,_8>>{});               // Val layout  1x8 k-major
TiledCopy copyB = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
    Layout<Shape<_16,_8>,Stride<_8,_1>>{},
    Layout<Shape< _1,_8>>{});
```

`make_tiled_copy` 的三个参数（参见 [算法 §copy](04_algorithms.md)、[编程模型 §3.3 TiledCopy](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)）：

- **Copy_Atom**：底层搬运指令。`SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` 使用 `cp.async.ca.shared.global` 指令搬运 128b，绕过寄存器直达 smem
- **Thr layout**：128 线程在 2D tile 上的排列方式。`Stride<_8,_1>` 表示列方向（K）步长为 1，即 K-major 排列——连续线程 ID 先填满 K 方向的 8 列，再换行
- **Val layout**：每个线程每次搬运的元素排列。`Shape<_1,_8>` 即 1 行 × 8 列，8 个 half_t 连续排列在 K 方向

线程布局为什么是 K-major？因为 global memory 中 A/B 是 K-major 存储。让连续线程在 K 方向排列，每线程沿 K 搬 8 个连续元素，保证了 **global memory 合并访问**（coalesced access）。

### TiledMMA：Tensor Core 计算

```cpp
'''
MMA 配置：
- MMA Atom: SM80_16x8x16_F16F16F16F16_TN — 单条 mma.sync 计算 16×8×16
  - 32 线程（1 个 warp）协作执行
  - TN: A 行主序（即 K-major），B 列主序（即 K-major）
  - F16F16F16F16: 输入 fp16，累加 fp16（本 kernel 的选择；也可用 F32 累加）
- Atom layout: 2×2 — 沿 M 复制 2 份、沿 N 复制 2 份
  - 4 个 atom 需要 4×32 = 128 线程 = 4 个 warp
- Tile<_32,_32,_16>: 目标 tile 32×32×16
  - 2 个 atom 沿 M: 2×16 = 32
  - 2 个 atom 沿 N: 2×8 = 16... 但 Tile 指定 N=32
  - 差值由 MMA_N 迭代补齐：每个 atom 在 N 方向迭代 32/8/2=2 次

完整 CTA tile (128×128) 需要:
  M 方向: 128/32 = 4 次 MMA_M 迭代
  N 方向: 128/32 = 4 次 MMA_N 迭代
  K 方向: 64/16 = 4 次 K_BLOCK 迭代（内循环）
'''
TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                               Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
                               Tile<_32,_32,_16>{});      // 32x32x16 Tiled MMA for LDSM
```

`make_tiled_mma` 的三个参数（参见 [MMA Atom](05_mma_atom.md)、[编程模型 §2.5 TiledMMA](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)）：

- **MMA Atom**：底层 PTX 指令 `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`，32 线程协作计算 16×8 输出
- **Atom layout** `<_2,_2>`：2×2 排列 = 4 个 atom → 4 个 warp → 128 线程。沿 M 铺 2 个（16×2=32），沿 N 铺 2 个（8×2=16）
- **Tile** `<_32,_32,_16>`：最终 tile 32×32×16。结合 atom 的 16×8×16，每个 atom 需要在 N 方向迭代 32/(8×2)=2 次

线程数 `size(mmaC)` = 128，直接用于 `dimBlock`。Grid 大小按 M/N 方向的 tile 数量计算。

### S→R Copy Atom：ldmatrix

```cpp
'''
S→R 搬运选择：SM75_U32x4_LDSM_N
- ldmatrix.sync.aligned.x4.m8n8.shared.b16: 每线程加载 4×32bit = 16B
- 32 线程协作加载 4 个 8×8 的 16-bit 矩阵片段
- 加载后的寄存器分布直接匹配 mma.sync 操作数布局
- 无需额外的寄存器 shuffle 或重排
'''
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;
```

为什么不直接从 smem 读到寄存器？因为 `mma.sync` 要求操作数在寄存器中的分布必须匹配特定模式（参见[编程模型 §2.2 寄存器分布](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)）。`ldmatrix` 指令（SM75 引入）专门将 smem 数据按 Tensor Core 要求的寄存器布局加载，一步到位。普通 `ld.shared` 加载后还需要线程间 shuffle 才能匹配 MMA 布局。

注意 s2r_atom 只是 Copy_Atom，不是 TiledCopy。它在 kernel 内部通过 `make_tiled_copy_A(s2r_atom_a, mma)` 与 TiledMMA 绑定后才构成完整的 S→R TiledCopy（见下节 [Copy Atom Retiling](#copy-atom-retiling)）。

### Kernel 启动

```cpp
int smem_size = int(sizeof(SharedStorage<cute::half_t, cute::half_t, decltype(sA), decltype(sB)>));
dim3 dimBlock(size(mmaC));                                  // 128 线程
dim3 dimGrid(size(ceil_div(M, bM)),
             size(ceil_div(N, bN)));

cudaFuncSetAttribute(kernel_fptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
cudaFuncSetAttribute(kernel_fptr,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);   // L1 全部划给 SMEM
```

`SharedStorage` 是一个简单结构体，包含 A 和 B 的 SMEM buffer。`cosize_v<SmemLayout>` 计算 Layout 的陪域大小（即最大偏移 + 1），确保分配足够空间。`PreferredSharedMemoryCarveout = 100` 将 L1/SMEM 可配置区域全部划给 SMEM，最大化可用 SMEM 容量。

---

## Kernel：Tensor 构造与分区

`gemm_device`（第 54-310 行）是一个高度泛型的 kernel 模板。所有类型和 Layout 都由模板参数决定，kernel 本身不包含任何硬件特定代码。

### 全局 Tensor 与 CTA 分区

```cpp
'''
从原始指针 + shape + stride 构造完整矩阵 Tensor
- make_gmem_ptr: 为指针标记 global memory 地址空间
- select<0,2>(shape_MNK): 从 (M,N,K) 中取 (M,K)
- CuTe 惯例：A=(M,K), B=(N,K), C=(M,N)
'''
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

'''
CTA 分区：每个 thread block 负责一个 (BLK_M, BLK_N) 的输出 tile
- cta_coord: (blockIdx.x, blockIdx.y, _) — M 和 N 固定, K 用下划线保留
- local_tile: 先用 cta_tiler 做 zipped_divide 分块，再用 cta_coord 切片
- Step 掩码控制哪些维度参与分区：
  - gA: Step<_1, X, _1> → 取 M 和 K，跳过 N → (BLK_M, BLK_K, k)
  - gB: Step< X, _1, _1> → 取 N 和 K，跳过 M → (BLK_N, BLK_K, k)
  - gC: Step<_1, _1, X> → 取 M 和 N，跳过 K → (BLK_M, BLK_N)
'''
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
```

gA 的第三维 `k` 是该 CTA 沿 K 方向需要迭代的 tile 数量（= ceil(K / BLK_K)）。主循环通过 `k_tile_next` 索引遍历。关于 `local_tile` 的详细语义，参见 [Layout 代数](02_layout_algebra.md) 和 [GEMM 教程 §CTA 分区](06_gemm_tutorial.md)。

### SMEM Tensor

```cpp
'''
动态共享内存 → 类型化 Tensor
- SharedStorage 包含 ArrayEngine<ElementA, cosize_v<ASmemLayout>> A 和 B
- sA_layout 是 host 侧传入的 swizzled layout: (BLK_M, BLK_K, PIPE)
  - 前两维 (128, 64): 一个 tile 的数据
  - 第三维 PIPE=3: 三级流水线的三个缓冲区
'''
extern __shared__ char shared_memory[];
using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)
```

`make_smem_ptr` 标记指针为 shared memory 地址空间。后续 `copy` 调用时，CuTe 根据源（gmem_ptr）和目标（smem_ptr）的地址空间标记自动选择 `cp.async` 指令（参见[编程模型 §3.4 copy 分发逻辑](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)）。

### G→S Copy 分区

```cpp
'''
将 global tile 和 smem tile 按 TiledCopy 的线程-值映射分区
- get_slice(threadIdx.x): 获取当前线程的 per-thread copy 操作器
- partition_S: 将源 tensor (gA) 按 copy 的源布局分区
  → (CPY, CPY_M, CPY_K, k)
  - CPY: 单条 cp.async 指令消费的元素数（8 个 half_t = 16B）
  - CPY_M: 该线程在 M 方向负责的迭代次数 (128/16=8)
  - CPY_K: 该线程在 K 方向负责的迭代次数 (64/64=1)
  - k: 沿 K 方向的全局 tile 数
- partition_D: 将目标 tensor (sA) 按 copy 的目标布局分区
  → (CPY, CPY_M, CPY_K, PIPE)
  - 前三维与 partition_S 一一对应
  - 第四维是 PIPE=3，对应 smem 的三个 pipeline buffer
'''
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)
```

`partition_S` 和 `partition_D` 的核心作用：将一个完整 tile 按线程 layout 和值 layout 切分后，每个线程只看到自己负责搬运的那部分数据。源和目标使用相同的分区模式，保证 `copy(tAgA(_, _, _, k), tAsA(_, _, _, pipe))` 将正确的全局数据搬运到正确的 smem 位置。

### MMA 分区与累加器

```cpp
'''
MMA 分区：将输出 tile 和操作数按 TiledMMA 的线程-值映射分区
- partition_C(gC): 将 (BLK_M, BLK_N) 输出按 MMA 布局分区
  → (MMA, MMA_M, MMA_N)
  - MMA: 单个 MMA atom 产生的元素数（每线程的值数，如 4 个 fp16）
  - MMA_M: M 方向的迭代次数 (128/32=4)
  - MMA_N: N 方向的迭代次数 (128/32=4)
- partition_fragment_A(sA(_,_,0)): 创建 A 操作数的寄存器 fragment
  → (MMA, MMA_M, MMA_K)
  - MMA_K: K 方向的迭代次数 (64/16=4)，即 K_BLOCK_MAX
- make_fragment_C: 创建累加器（ArrayEngine，寄存器存储）
'''
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));               // (MMA,MMA_N,MMA_K)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

clear(tCrC);                                                          // 累加器清零
```

`partition_fragment_A/B` 传入 `sA(_,_,0)` 而非整个 sA，因为只需要单个 pipeline stage 的 shape 来确定 fragment 大小。创建的是 **ArrayEngine** Tensor（寄存器存储），不是视图。

这里有一个关键的 shape 对应关系：

- `tCrA` 的 `MMA_K = size<2>(tCrA) = 4`（即 K_BLOCK_MAX）— 64/16 = 4 个 k_block
- `tCrC` 的 `MMA_M` 和 `MMA_N` 与 `tCgC` 匹配 — 决定了 epilogue 写回时的遍历次数

### Copy Atom Retiling

这是本 kernel 最微妙的部分。MMA 分区产生的 `tCrA`（寄存器 fragment）的内部布局是按 MMA 指令要求排列的。但从 smem 读数据到寄存器时，需要用 `ldmatrix` 指令，它有自己的线程-值映射。**两者的布局不同**——需要 retiling 来桥接。

```cpp
'''
S→R retiling：将 ldmatrix 的分区模式与 MMA fragment 对齐
1. make_tiled_copy_A(s2r_atom_a, mma):
   - 根据 MMA 的 A 操作数布局（ALayout）和 ldmatrix 的 Copy_Atom
   - 自动构建一个 TiledCopy，其线程排列匹配 MMA 的线程映射
   - 其值排列匹配 ldmatrix 的加载粒度

2. partition_S(sA): 按 ldmatrix 的源布局分区 smem
   → (CPY, MMA_M, MMA_K, PIPE)
   - CPY: ldmatrix 单次加载的元素
   - MMA_M/MMA_K: 与 tCrA 的后两维对应

3. retile_D(tCrA): 将 MMA fragment 的布局重排为 ldmatrix 的目标布局
   → (CPY, MMA_M, MMA_K)
   - 不创建新存储，只是改变 tCrA 的逻辑视图
   - 保证 copy(tXsA_p(_,_,k), tXrA(_,_,k)) 将 smem 数据
     按 ldmatrix 布局加载到 MMA 期望的寄存器位置
'''
TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)
```

为什么需要 retiling？直觉上：

- **MMA 视角**：`tCrA(mma, mma_m, mma_k)` — 按"第几个 atom、第几次 M 迭代、第几次 K 迭代"组织
- **ldmatrix 视角**：`tXrA(cpy, mma_m, mma_k)` — 按"一次 ldmatrix 加载的数据块、第几次 M 迭代、第几次 K 迭代"组织
- 两者指向**同一块寄存器**，只是第一维（MMA vs CPY）的逻辑切分方式不同
- `retile_D` 将 MMA 的第一维重新解释为 ldmatrix 的第一维，不移动任何数据

`make_tiled_copy_A/B` 的关键作用是自动计算这个重新解释需要的布局变换——它查询 MMA 的 ALayout/BLayout（线程-值到矩阵坐标的映射），与 ldmatrix 的 Copy_Atom 布局做对齐，生成兼容两者的 TiledCopy。

---

## Kernel 核心：双层流水线主循环

主循环是本 kernel 的核心，实现了 **SMEM 层 + 寄存器层** 的双层流水线。理解其结构需要先明确几个关键变量：

| 变量 | 含义 | 值（TN HGEMM） |
|------|------|------|
| `K_PIPE_MAX` | SMEM pipeline 深度 = `size<3>(tAsA)` | 3（bP） |
| `K_BLOCK_MAX` | 寄存器 pipeline 深度 = `size<2>(tCrA)` | 4（BLK_K/16 = 64/16） |
| `k_tile_count` | 剩余待处理的全局 K tile 数 | 初始 = ceil(K/BLK_K) |
| `k_tile_next` | 下一个要从 global 加载的 K tile 索引 | 初始 = 0 |
| `smem_pipe_read` | 当前从 smem 读取的 pipeline stage | 0 |
| `smem_pipe_write` | 当前向 smem 写入的 pipeline stage | K_PIPE_MAX-1 = 2 |

### Prologue：SMEM 预填充

在进入主循环之前，先将前 `K_PIPE_MAX - 1 = 2` 个 K tile 异步加载到 SMEM：

```cpp
'''
SMEM 预填充：将 global 的前 2 个 K tile 异步加载到 smem pipe 0 和 pipe 1
- 每次 copy 调用发起一组 cp.async 指令
- cp_async_fence() 将这组指令提交为一个 async group
- 不等待完成就继续下一组加载（异步）
'''
CUTE_UNROLL
for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
}
```

此时 smem 的 pipe 0 和 pipe 1 正在异步填充，pipe 2 空闲。主循环第一次迭代会将 pipe 2 的加载也发起出去。

### Prologue：寄存器预填充

如果 `K_BLOCK_MAX > 1`（本例 = 4），在主循环开始前预加载第一个 k_block 到寄存器：

```cpp
if (K_BLOCK_MAX > 1) {
    '''
    等待 pipe 0 的 cp.async 完成（wait_group 允许 K_PIPE_MAX-2=1 个 group 未完成）
    → pipe 0 就绪，pipe 1 仍可能在飞行中
    '''
    cp_async_wait<K_PIPE_MAX-2>();                                    // wait_group 1
    __syncthreads();

    '''
    用 ldmatrix 将 smem pipe 0 的第 0 个 k_block 加载到寄存器
    - tXsA_p 指向 smem_pipe_read=0 的 slice
    - Int<0>{} 是编译时常量，确保静态分发
    '''
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
}
```

此时状态：smem pipe 0 的 k_block 0 已在寄存器中，可以立即开始计算。

### 主循环结构

主循环是一个**外循环（SMEM 层）嵌套内循环（寄存器层）** 的结构：

```cpp
CUTE_NO_UNROLL
while (k_tile_count > -(K_PIPE_MAX-1))
{
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
        // ... 三个关键操作，位置不同 ...
    }
}
```

**外循环终止条件** `k_tile_count > -(K_PIPE_MAX-1)`：看起来反直觉，但这是因为 prologue 已经预加载了 `K_PIPE_MAX-1` 个 tile。`k_tile_count` 在 prologue 中已递减，主循环会继续递减。当 `k_tile_count` 降到 `-(K_PIPE_MAX-1)+1 = -1` 时，所有 tile 都已加载并计算完毕。

**内循环** `k_block = 0..K_BLOCK_MAX-1`：遍历一个 BLK_K=64 内的 4 个 16-element 子块，每次执行一个 mma.sync。

### 内循环详解

内循环中有三个关键操作，分布在不同的 `k_block` 位置：

```cpp
CUTE_UNROLL
for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
{
    '''
    操作 1（k_block == K_BLOCK_MAX-1 时）：切换 SMEM 读指针 + 等待 cp.async
    当前 k_tile 的最后一个 k_block 即将计算完毕，
    需要准备好下一个 k_tile 的 smem 数据：
    - 切换 tXsA_p/tXsB_p 到下一个 smem pipe stage
    - cp_async_wait 确保该 stage 的数据已到达
    '''
    if (k_block == K_BLOCK_MAX - 1)
    {
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);

        cp_async_wait<K_PIPE_MAX-2>();                                // wait_group 1
        __syncthreads();
    }

    '''
    操作 2（每个 k_block）：寄存器预取
    将下一个 k_block 的数据从 smem 加载到寄存器
    - k_block_next = (k_block + 1) % K_BLOCK_MAX
    - 当 k_block = K_BLOCK_MAX-1 时，k_block_next = 0
      此时 tXsA_p 已指向新的 smem pipe（操作 1 刚切换）
      → 预取的是下一个 k_tile 的第 0 个 k_block
    '''
    auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;          // 编译时模运算
    copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
    copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));

    '''
    操作 3（k_block == 0 时）：发起下一个 k_tile 的 G→S cp.async
    在内循环的第一次迭代，即刚开始处理当前 k_tile 时，
    就启动下一个 k_tile 从 global 到 smem 的异步搬运：
    - 写入 smem_pipe_write（之前已读完的 stage）
    - 推进 smem 读/写指针（循环 buffer）
    '''
    if (k_block == 0)
    {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();

        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }

        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
    }

    '''
    操作 4（每个 k_block）：执行 MMA 计算
    使用当前 k_block 的寄存器数据执行 mma.sync
    - tCrA(_,_,k_block): 当前 k_block 的 A fragment
    - tCrB(_,_,k_block): 当前 k_block 的 B fragment  
    - tCrC: 累加器，跨所有 k_block 和 k_tile 累加
    '''
    gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
}
```

### 时序图

以 K_PIPE_MAX=3、K_BLOCK_MAX=4 为例，展示前几个 k_tile 的操作时序：

```
时间线 →

Prologue:
  G→S:  [pipe0: tile0] [pipe1: tile1]
  S→R:  [pipe0: blk0]

主循环 k_tile=0 (smem_pipe_read=0):
  k_block=0: S→R(blk1)  G→S(pipe2:tile2)  MMA(blk0)
  k_block=1: S→R(blk2)                     MMA(blk1)
  k_block=2: S→R(blk3)                     MMA(blk2)
  k_block=3: 切换→pipe1  wait  S→R(next blk0)  MMA(blk3)

主循环 k_tile=1 (smem_pipe_read=1):
  k_block=0: S→R(blk1)  G→S(pipe0:tile3)  MMA(blk0)
  k_block=1: S→R(blk2)                     MMA(blk1)
  k_block=2: S→R(blk3)                     MMA(blk2)
  k_block=3: 切换→pipe2  wait  S→R(next blk0)  MMA(blk3)

主循环 k_tile=2 (smem_pipe_read=2):
  k_block=0: S→R(blk1)  G→S(pipe1:tile4)  MMA(blk0)
  ...
```

**重叠关系**：

- **G→S 与 MMA 重叠**：k_block=0 发起的 cp.async 在后续 k_block 的 MMA 计算期间在后台完成
- **S→R 与 MMA 重叠**：每个 k_block 先预取下一个 k_block 的寄存器数据，再执行当前 k_block 的 MMA
- **三级 smem pipeline**：当一个 stage 在被读取（S→R）时，另一个在被写入（G→S），第三个在等待消费——三者完全重叠

关于 cp.async 的 commit/wait 语义，参见[编程模型 §4 cp.async 异步流水线](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)。

---

## Epilogue

主循环结束后，`tCrC` 累加了所有 K tile 的结果。Epilogue 将累加器写回 global memory：

```cpp
axpby(alpha, tCrC, beta, tCgC);
```

`axpby` 是 CuTe 提供的算法（参见[算法](04_algorithms.md)），逐元素计算 `tCgC = alpha * tCrC + beta * tCgC`。`tCrC` 是寄存器 fragment，`tCgC` 是 global memory 的分区视图。两者 shape 相同（`(MMA, MMA_M, MMA_N)`），由 `thr_mma.partition_C` 保证一一对应。

这是一个极简 epilogue——没有 smem 中转、没有向量化写回。实际高性能 kernel（如 CUTLASS 的 `CollectiveEpilogue`）会通过 smem 做跨线程数据重排后再向量化写回。

---

## SIMT 版本补充说明

### NT SIMT（gemm_nt<TA,TB,TC>）

NT 布局下 A 是 M-major、B 是 N-major，数据在内存中的主方向与 tile 的非归约维度一致。这带来几个简化：

- **SMEM layout**：直接用 `make_layout(make_shape(bM, bK, bP))`，列主序。M-major 下连续线程沿 M 方向读取 smem，天然跨不同 bank，无需 swizzle 或 padding
- **TiledCopy G→S**：线程布局 32×8 M-major，值布局 4×1 M-major。每线程搬 4 个元素 = 128b 向量化。因为 A 在 global 也是 M-major，连续线程读取连续地址，合并访问
- **TiledMMA**：`UniversalFMA<TC,TA,TB>` + 16×16×1 = 256 线程，每线程做标量 FMA。无 Tensor Core
- **S→R**：`AutoVectorizingCopy`，普通 `ld.shared`，不用 ldmatrix
- **BK=8**：标量 FMA 吞吐量低，大 BK 无收益

### TN SIMT（gemm_tn<TA,TB,TC>）

TN 布局下 A/B 都是 K-major，这是更有挑战性的情况：

- **SMEM layout**：使用 **padding** — `make_stride(Int<1>{}, bM+Int<1>{})`。物理上仍然 M-major 存储，但 K 方向的 stride 从 bM 变为 bM+1。这使得沿 M 方向连续元素的 bank 偏移错开，避免 bank conflict。Padding 比 swizzle 简单但浪费少量 smem
- **TiledCopy G→S**：`SM80_CP_ASYNC_CACHEALWAYS<TA>`（注意模板参数是 TA 而非 uint128_t）+ 32×8 K-major 线程布局 + **1×1 值布局**。逐元素标量搬运，因为 K-major 下连续线程沿 M 跨行访问 global，地址间距 = ldA，无法向量化
- **TiledMMA / S→R**：同 NT SIMT

注意两个 SIMT 版本使用了完全相同的 `gemm_device` kernel——差异仅在 host 侧传入的 layout、TiledCopy 和 TiledMMA 参数。kernel 模板对这些参数是完全泛型的，CuTe 的类型系统在编译期完成所有特化。
