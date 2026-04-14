---
tags:
  - CUTLASS
  - CUDA
---

# CuTe 实战：sgemm_sm80.cu 拆解

> **源码**: [NVIDIA/cutlass - examples/cute/tutorial/sgemm_sm80.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu)
> **许可证**: BSD-3-Clause, Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES

本文按源码顺序完整拆解 `sgemm_sm80.cu`，以 **TN HGEMM** 路径为主线。TN 表示 A 转置、B 不转置（即 A 和 B 在内存中都是 K 维度连续存储）；HGEMM 表示 half（fp16）精度的 GEMM；这条路径使用 Tensor Core 硬件加速。这个文件用一个通用 kernel 模板 `gemm_device`（L54-310）实现了三种 GEMM 路径（NT SIMT / TN SIMT / TN HGEMM），差异全部通过 host 侧模板参数传入。

---

## 存储空间布局

GPU 的存储层次从远到近分为三级：**Global Memory**（显存，容量大但延迟高）→ **Shared Memory**（片上 SRAM，同一 CTA 内的线程共享，低延迟）→ **Registers**（每个线程私有，最快）。高性能 GEMM 的核心就是让数据沿这三级逐步搬运、逐步复用：先从 Global 加载大块数据到 SMEM 供整个 CTA 共享，再从 SMEM 取小块到寄存器供 Tensor Core 计算。

以下按这三级存储，说明 TN HGEMM 路径中每一级存放了哪些数据、shape 是什么。

### 第一级：Global Memory（显存中的矩阵）

GEMM 计算 $C_{M \times N} = A_{M \times K} \times B_{K \times N}$（CuTe 中 B 用 shape (N,K) 表示，数学上等价于 $B^T$ 参与乘法）。K 是**归约维度**——A 的一行与 B 的一行沿 K 做点积求和，K 不出现在输出 C 中。

数据在 Global Memory 中有两层视图：

- **完整矩阵**（`mA`, `mB`, `mC`）：对应整个 A/B/C 矩阵，是 kernel 的原始输入输出
- **CTA tile**（`gA`, `gB`, `gC`）：从完整矩阵中切出当前 CTA 负责的子块。每个 CTA 负责 C 的一个 128×128 输出块；为了计算这个输出块，需要沿归约维度 K 将 A 和 B 切成 64 一段的 tile，逐个加载到 SMEM 做乘累加

| 变量 | Shape | 说明 |
|------|-------|------|
| mA | (M, K) | A 完整矩阵，K-major（K 维度步长为 1，即同一行的 K 方向元素在内存中连续），stride = (ldA, 1) |
| mB | (N, K) | B 完整矩阵，K-major，stride = (ldB, 1) |
| mC | (M, N) | C 完整矩阵，M-major（M 维度步长为 1，即同一列的 M 方向元素连续），stride = (1, ldC) |
| gA | (128, 64, k) | 从 mA 中切出的当前 CTA 的 tile。128 行(M) × 64 列(K)，第三维 k = ceil(K/64) 表示归约方向共有多少个这样的 tile |
| gB | (128, 64, k) | 从 mB 中切出，同理 |
| gC | (128, 128) | 从 mC 中切出的当前 CTA 的输出块 |

### 第二级：Shared Memory（CTA 内共享的片上缓存）

gA/gB 的每个 128×64 tile 不能一次全部放进寄存器（太大），需要先搬到 Shared Memory 作为中转。为了让 Global→SMEM 搬运和计算重叠（流水线），SMEM 中为 A 和 B 各分配了 **3 个 buffer**（称为 pipeline stage）——当一个 buffer 正在被读取计算时，另一个在接收新数据，第三个在等待：

| 变量 | Shape | Layout | 说明 |
|------|-------|--------|------|
| sA | (128, 64, 3) | Swizzle\<3,3,3\> | A 的 SMEM buffer。前两维 (128,64) 与 gA 的单个 tile 对应，第三维 3 = pipeline 深度 |
| sB | (128, 64, 3) | Swizzle\<3,3,3\> | B 的 SMEM buffer，同理 |

每个 128×64 tile 占 128×64×2B = 16KB（half_t 每个 2 字节），A+B 共 3 个 stage = 6×16KB = 96KB。Swizzle 通过对地址做 XOR 变换，使原本会映射到同一个 shared memory bank 的访问分散到不同 bank，消除 `ldmatrix` 沿列读取时的 bank conflict（多个线程同时访问同一 bank 会串行化，降低带宽）。详见[编程模型 §5 Swizzle](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)。

### 第三级：Registers（每个线程的私有寄存器）

Tensor Core 的 `mma.sync` 指令从寄存器中读取操作数、将结果写入寄存器累加器。SMEM 中的一个 128×64 tile 仍然太大，不能一次全部加载到寄存器，所以沿 K 维度再切成 4 个 16-element 的子块（k_block），在内循环中依次从 SMEM 加载到寄存器并计算。这里同样做了流水线：计算当前 k_block 的同时，预取下一个 k_block 的数据。

每个线程持有以下寄存器 Tensor：

| 变量 | Shape | 说明 |
|------|-------|------|
| tCrA | (MMA, MMA_M=4, MMA_K=4) | A 操作数 fragment。MMA_K=4 对应 4 个 k_block，MMA_M=4 对应 M 方向 4 次迭代 |
| tCrB | (MMA, MMA_N=4, MMA_K=4) | B 操作数 fragment。MMA_N=4 对应 N 方向 4 次迭代 |
| tCrC | (MMA, MMA_M=4, MMA_N=4) | 累加器，跨所有归约 tile 持续累加，最终写回 Global Memory |

各维度的来源：

- MMA_M=4：CTA 输出块有 128 行，TiledMMA 一次覆盖 32 行，需要 128/32 = 4 次迭代
- MMA_N=4：同理，128/32 = 4 次
- MMA_K=4：SMEM tile 有 64 列(K)，`mma.sync` 一次消费 16 列，需要 64/16 = 4 次内循环
- MMA（第 0 维）：单条 `mma.sync` 指令中，当前线程负责的输出元素个数（由硬件指令格式决定）

### 数据流总览

```
Global Memory ──cp.async 128b──→ SMEM (3 级 pipe) ──ldmatrix──→ Registers (4 级 k_block) ──mma.sync──→ Accumulator ──axpby──→ Global
```

| 搬运 | 硬件指令 | CuTe 抽象 | 参与线程 |
|------|---------|-----------|---------|
| Global→SMEM | cp.async.ca.shared.global 128b | TiledCopy + SM80_CP_ASYNC_CACHEALWAYS\<uint128_t\> | 128 线程协作 |
| SMEM→Register | ldmatrix.sync.x4.m8n8.b16 | make_tiled_copy_A/B + SM75_U32x4_LDSM_N | 32 线程/warp |
| Register 计算 | mma.sync.m16n8k16.f16 | TiledMMA + SM80_16x8x16_F16F16F16F16_TN | 32 线程/warp |

---

## Host 侧配置：gemm_tn HGEMM（L325-430）

Kernel 的所有行为由 host 侧传入的模板参数决定。先看 host 侧如何构造这些参数，再进入 kernel。

```cpp
auto dA = make_stride(ldA, Int<1>{});                      // A (M,K):(ldA,1) K-major
auto dB = make_stride(ldB, Int<1>{});                      // B (N,K):(ldB,1) K-major
auto dC = make_stride(Int<1>{}, ldC);                      // C (M,N):(1,ldC) M-major

auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int< 64>{};                                      // BK=64: 一条 mma.sync 消费 K 方向 16 个元素, 64/16=4 次内循环
auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
auto bP = Int<3>{};                                         // SMEM pipeline 深度

// ── SMEM Layout: Swizzle ──
// 基础 atom: 8 行 × 64 列的 K-major 布局
// composition(Swizzle<3,3,3>{}, atom): 对地址做 XOR, 避免 ldmatrix 沿列读取时
// 多个线程访问同一 shared memory bank (bank conflict 会导致串行化)
// tile_to_shape: 将 8×64 atom 重复平铺到 128×64×3
auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                Layout<Shape <_8,Shape <_8, _8>>,
                                       Stride<_8,Stride<_1,_64>>>{});
auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
auto sC = make_layout(make_shape(bM, bN));

// ── TiledCopy: Global → SMEM ──
// Copy_Atom: cp.async 128b, 绕过寄存器直达 smem
// Thr layout 16x8 K-major: 连续线程沿 K 排列 → global 合并访问
// Val layout 1x8: 每线程搬 8 个 half_t = 16B = 128b
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
    Layout<Shape<_16,_8>,Stride<_8,_1>>{},                 // 16x8 K-major
    Layout<Shape< _1,_8>>{});                              // 1x8 K-major
TiledCopy copyB = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
    Layout<Shape<_16,_8>,Stride<_8,_1>>{},
    Layout<Shape< _1,_8>>{});

// ── TiledMMA: Tensor Core ──
// MMA Atom: mma.sync.m16n8k16, 32 线程(1 warp)计算 16×8 输出
// 2×2 atom layout: 4 个 warp = 128 线程, 覆盖 32×16
// Tile<32,32,16>: 每个 atom 在 N 方向迭代 32/(8×2)=2 次
TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                               Layout<Shape<_2,_2>>{},     // 2x2 atom layout
                               Tile<_32,_32,_16>{});       // 目标 tile

// ── S→R Copy Atom: ldmatrix ──
// ldmatrix.x4: 32 线程协作加载 4 个 8×8 矩阵片段到 mma 寄存器布局
// 只是 Atom, 在 kernel 内通过 make_tiled_copy_A/B 与 MMA 绑定
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;

// ── Kernel 启动 ──
int smem_size = int(sizeof(SharedStorage<cute::half_t, cute::half_t,
                                         decltype(sA), decltype(sB)>));
dim3 dimBlock(size(mmaC));                                  // 128 线程
dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
cudaFuncSetAttribute(kernel_fptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
cudaFuncSetAttribute(kernel_fptr,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);   // L1 全部划给 SMEM
kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>
    (prob_shape, cta_tiler,
     A, dA, sA, copyA, s2r_atom_A,
     B, dB, sB, copyB, s2r_atom_B,
     C, dC, sC, mmaC, alpha, beta);
```

关键设计选择说明：

- **Stride 惯例**：CuTe 不用 BLAS 的 T/N 标记，直接用 stride 指定哪个维度连续。`Int<1>{}` 是编译时常量，编译器可优化地址计算。参见 [GEMM 教程 §M-major/K-major 表](06_gemm_tutorial.md)
- **Swizzle 原理**：K-major 下 ldmatrix 需要沿列（M/N 方向）读取 SMEM。不做 swizzle 时，连续行的同一列地址差值恰好是 K 方向的 stride，容易落到同一 bank 造成串行化。`Swizzle<3,3,3>` 对 SMEM 地址的行号部分和列号部分做 XOR，使连续 8 行的同一列自动分散到 8 个不同 bank。详见[编程模型 §5.3](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- **G→S 线程布局为 K-major**：global 是 K-major 存储（K 方向连续），连续线程沿 K 排列 + 每线程搬 8 个连续 K 元素 = 合并访问（相邻线程访问相邻地址，GPU 可合并为一次内存事务）。一次 copy 调用覆盖 16 行 × (8 线程 × 8 元素) = 16×64 = 1024 个 half_t = 2KB，CTA 有 128 行需要搬运，所以需要 128/16 = 8 次迭代。参见[算法 §copy](04_algorithms.md)、[编程模型 §3.3 TiledCopy](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- **TiledMMA 参数**：单个 atom 计算 16(M)×8(N)×16(K) 的输出，2×2 排列后 4 个 atom 覆盖 32(M)×16(N)×16(K)。但 Tile 指定目标为 32×32×16，所以每个 atom 在 N 方向需要迭代 32/(8×2)=2 次来补齐。完整 CTA tile 128×128 需要 MMA_M=128/32=4, MMA_N=128/32=4 次外层迭代，K 方向 MMA_K=64/16=4 次内循环。参见 [MMA Atom](05_mma_atom.md)、[编程模型 §2.5 TiledMMA](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- **ldmatrix 而非 ld.shared**：`mma.sync` 要求操作数寄存器分布匹配特定模式，`ldmatrix` 一步到位加载到正确布局，省去线程间 shuffle。参见[编程模型 §3.1](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)

---

## gemm_device Kernel（L54-310）

以下按 kernel 代码从上到下的顺序走。所有 shape 标注基于 TN HGEMM 参数值。

CuTe 的变量命名遵循前缀规则：`t` 表示线程分区后的视图，紧跟的字母标识分区模式（如 `A` = G→S copy 的 A 分区，`C` = MMA 分区，`X` = S→R copy 分区），再后面标识数据来源（`g` = Global Memory，`s` = Shared Memory，`r` = Register）。例如 `tAgA` 读作"用 copy_A 的分区模式分区 Global Memory 中的 A"，`tCrC` 读作"MMA 分区模式下的寄存器累加器 C"。

### Tensor 构造、分区与 Retiling（L68-184）

这一阶段在三级存储中构造 Tensor，并为每个线程建立 G→S 和 S→R 的分区视图。

```cpp
// ── 第一级: Global Memory ──
// mA/mB/mC: 完整矩阵 → gA/gB/gC: 当前 CTA 负责的子块
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);               // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});   // (128,64,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});   // (128,64,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});   // (128,128)

// ── 第二级: Shared Memory ──
// sA/sB: 3 个 pipeline stage 的 SMEM buffer, 数据从 gA/gB 搬入
extern __shared__ char shared_memory[];
using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);    // (128,64,3)
Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);    // (128,64,3)

// ── G→S Copy 分区: 每个线程看到自己负责从 gA→sA 搬运的部分 ──
// tAgA/tBgB: 线程视角的 Global 源 (从 gA/gB 分区而来)
// tAsA/tBsB: 线程视角的 SMEM 目标 (从 sA/sB 分区而来)
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);                              // (CPY,CPY_M,CPY_K,k)
Tensor tAsA = thr_copy_a.partition_D(sA);                              // (CPY,CPY_M,CPY_K,PIPE)
ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
Tensor tBgB = thr_copy_b.partition_S(gB);                              // (CPY,CPY_N,CPY_K,k)
Tensor tBsB = thr_copy_b.partition_D(sB);                              // (CPY,CPY_N,CPY_K,PIPE)

// ── 第三级: Registers ──
// tCrA/tCrB: MMA 操作数 fragment (从 sA/sB 加载到寄存器)
// tCrC: 累加器 (始终在寄存器, 最终写回 gC)
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCgC = thr_mma.partition_C(gC);                                 // (MMA,MMA_M,MMA_N) Global C 的线程视图
Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                 // (MMA,MMA_M,MMA_K) 寄存器
Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                 // (MMA,MMA_N,MMA_K) 寄存器
Tensor tCrC = thr_mma.make_fragment_C(tCgC);                           // (MMA,MMA_M,MMA_N) 寄存器累加器
clear(tCrC);                                                            // 累加器清零

// ── S→R Copy Atom Retiling: 桥接 SMEM(第二级) 与 Registers(第三级) 的布局差异 ──
// tXsA/tXsB: ldmatrix 视角的 SMEM 源 (从 sA/sB 重新分区)
// tXrA/tXrB: ldmatrix 视角的寄存器目标 (与 tCrA/tCrB 共享同一块寄存器, 仅逻辑分组不同)
TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                          // (CPY,MMA_M,MMA_K,PIPE)
Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                           // (CPY,MMA_M,MMA_K)
TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                          // (CPY,MMA_N,MMA_K,PIPE)
Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                           // (CPY,MMA_N,MMA_K)
```

关键概念说明：

- **local_tile + Step 掩码**：`Step<_1, X, _1>` 表示从 (M,N,K) 中取 M 和 K 维度参与分区，跳过 N。本质是 `zipped_divide` 分块 + 坐标切片。参见 [Layout 代数](02_layout_algebra.md)、[GEMM 教程 §CTA 分区](06_gemm_tutorial.md)
- **make_gmem_ptr / make_smem_ptr**：标记指针的地址空间。后续 `copy` 调用根据源（gmem）和目标（smem）自动选择 `cp.async` 指令。参见[编程模型 §3.4 copy 分发](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- **partition_S / partition_D 的 shape**：CPY 是单条 `cp.async` 指令搬运的元素数（Val layout 指定 1×8 = 8 个 half_t = 128b），CPY_M 是该线程在 M 方向的迭代次数（128 行 / 16 行线程布局 = 8），CPY_K 是 K 方向迭代次数（64 / 64 = 1，因为 8 线程 × 8 元素已覆盖 64 列）。源和目标用相同分区模式，保证 `copy(src, dst)` 一一对应
- **partition_fragment_A/B**：传入 `sA(_,_,0)` 而非整个 sA——只需单个 pipe stage 的 shape 确定 fragment 大小。创建的是 ArrayEngine Tensor（寄存器存储），不是视图
- **Retiling 原理**：`tCrA`（MMA 视角）和 `tXrA`（ldmatrix 视角）指向**同一块寄存器**，但对这些寄存器有不同的逻辑分组方式。MMA 按"第几个 atom 的第几个元素"分组 `(mma, mma_m, mma_k)`，ldmatrix 按"一次加载指令的第几个元素"分组 `(cpy, mma_m, mma_k)`。`retile_D` 只改变第一维的逻辑解释，不移动任何数据。`make_tiled_copy_A/B` 负责自动计算这个重解释——它查询 MMA 的线程-值映射，与 ldmatrix 的 Copy_Atom 对齐，生成兼容两者的 TiledCopy

### Prologue + 双层流水线主循环（L135-301）

```cpp
// ═══════════════════════════════════════════════════════════
// Prologue: SMEM 预填充
// ═══════════════════════════════════════════════════════════

auto K_PIPE_MAX  = size<3>(tAsA);                                      // 3 (pipeline 深度)
int  k_tile_count = size<3>(tAgA);                                     // 总 K tile 数
int  k_tile_next  = 0;                                                 // 下一个待加载的 K tile

// 将前 2 个 K tile 异步加载到 smem pipe 0 和 pipe 1
CUTE_UNROLL
for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));        // G→S: A tile
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));        // G→S: B tile
    cp_async_fence();                                                   // 提交为一个 async group
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
}

// ═══════════════════════════════════════════════════════════
// Prologue: 寄存器预填充
// ═══════════════════════════════════════════════════════════

int smem_pipe_read  = 0;                                               // 当前读取的 pipe stage
int smem_pipe_write = K_PIPE_MAX-1;                                    // 当前写入的 pipe stage

Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);                           // pipe 0 的 smem slice
Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

auto K_BLOCK_MAX = size<2>(tCrA);                                     // 4 (寄存器 pipeline 深度)

if (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE_MAX-2>();                                     // wait<1>: pipe 0 就绪
    __syncthreads();
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));       // ldmatrix: k_block 0
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
}

// ═══════════════════════════════════════════════════════════
// 双层流水线主循环
//   外循环: 遍历 SMEM pipeline stages (K tiles)
//   内循环: 遍历 Register pipeline (K blocks within one tile)
//
// 重叠关系:
//   G→S cp.async 在后台完成 (与 MMA 重叠)
//   S→R ldmatrix 预取下一个 k_block (与 MMA 重叠)
//   MMA 计算当前 k_block
// ═══════════════════════════════════════════════════════════

CUTE_NO_UNROLL
while (k_tile_count > -(K_PIPE_MAX-1))                                 // 含排空迭代(消费 prologue 预加载的 tile)
{
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
        if (k_block == K_BLOCK_MAX - 1)                                // 最后一个 k_block
        {
            tXsA_p = tXsA(_,_,_,smem_pipe_read);                      // 切换到下一个 pipe stage
            tXsB_p = tXsB(_,_,_,smem_pipe_read);
            cp_async_wait<K_PIPE_MAX-2>();                             // 确保该 stage 数据就绪
            __syncthreads();
        }

        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;       // 编译时模运算
        copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next)); // S→R 预取
        copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));

        if (k_block == 0)                                             // 第一个 k_block
        {
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write)); // G→S 下一 tile
            copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
            cp_async_fence();                                          // 提交 async group

            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_next; }

            smem_pipe_write = smem_pipe_read;                          // 推进 smem 指针
            smem_pipe_read  = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
        }

        gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);       // MMA 计算当前 k_block
    }
}

// ═══════════════════════════════════════════════════════════
// Epilogue
// ═══════════════════════════════════════════════════════════

axpby(alpha, tCrC, beta, tCgC);                                        // C = α·acc + β·C
```

主循环结构说明：

**外循环终止条件** `k_tile_count > -(K_PIPE_MAX-1)`：prologue 预加载了 2 个 tile 并递减 `k_tile_count`。主循环继续递减直到 `-1`。为什么不是 `> 0`？因为 prologue 已经把 tile 加载到了 SMEM 但还没计算，主循环需要额外 2 次迭代来消费 prologue 加载的最后 2 个 tile（这些"排空"迭代中 G→S 加载的是越界地址，但不影响正确性因为结果不会被使用）。

**内循环**遍历 BLK_K=64 内的 4 个 16-element 子块（k_block），每次迭代中四个操作的位置精心安排以最大化重叠：

- `k_block == K_BLOCK_MAX-1`：切换 smem 读指针 + `cp_async_wait` 确保新 stage 就绪。放在最后一个 k_block 是因为此时当前 tile 即将处理完毕，需要为下一个 tile 准备好 smem 数据
- **每个 k_block**：`ldmatrix` 预取下一个 k_block 到寄存器。当 `k_block == K_BLOCK_MAX-1` 时 `k_block_next == 0`，此时 `tXsA_p` 已指向新 pipe（刚切换），预取的是下一个 k_tile 的第 0 个 k_block
- `k_block == 0`：发起下一个 k_tile 的 G→S `cp.async`。放在第一个 k_block 是为了尽早启动异步搬运，让它在后续 k_block 的 MMA 计算期间完成
- **每个 k_block**：`gemm` 执行 `mma.sync`，使用当前 k_block 的寄存器数据

**cp_async_wait\<1\> 语义**：允许最多 1 个 async group 未完成。prologue 和主循环不断向流水线提交 group，wait<1> 确保至少倒数第二个 group 完成——正好是当前要读取的 stage。参见[编程模型 §4 cp.async 流水线](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)。

**时序示意**（K_PIPE_MAX=3, K_BLOCK_MAX=4）：

```
Prologue:
  G→S:  [pipe0: tile0] [pipe1: tile1]
  S→R:  [pipe0: blk0]

主循环 k_tile=0 (pipe_read=0):
  k_block=0: S→R(blk1)  G→S(pipe2:tile2)  MMA(blk0)
  k_block=1: S→R(blk2)                     MMA(blk1)
  k_block=2: S→R(blk3)                     MMA(blk2)
  k_block=3: 切换→pipe1  wait  S→R(blk0')  MMA(blk3)

主循环 k_tile=1 (pipe_read=1):
  k_block=0: S→R(blk1)  G→S(pipe0:tile3)  MMA(blk0)
  ...
```

**Epilogue**：`axpby` 逐元素计算 `tCgC = α·tCrC + β·tCgC`。tCrC 是寄存器中的累加结果，tCgC 是 global memory 中 C 矩阵对应位置的视图，两者 shape 一致 `(MMA,MMA_M,MMA_N)`。这是极简写回——没有通过 SMEM 做跨线程数据重排，也没有向量化写回优化。

---

## NT SIMT 与 TN SIMT 版本

两个 SIMT 版本使用完全相同的 `gemm_device` kernel，差异仅在 host 侧参数。

### NT SIMT（L432-503）

```cpp
auto dA = make_stride(Int<1>{}, ldA);                      // A: M-major (M,K):(1,ldA)
auto dB = make_stride(Int<1>{}, ldB);                      // B: N-major (N,K):(1,ldB)
auto bK = Int<  8>{};                                      // BK=8, 标量 FMA 吞吐低
auto sA = make_layout(make_shape(bM, bK, bP));             // 列主序, M-major 无 bank conflict

TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
    Layout<Shape<_32,_8>>{},                               // 32x8 M-major 线程布局
    Layout<Shape< _4,_1>>{});                              // 4x1 值布局 → 128b 向量化

TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                               Layout<Shape<_16,_16,_1>>{}); // 256 线程标量 FMA
// S→R: Copy_Atom<AutoVectorizingCopy, TA>{} (普通 ld.shared)
```

M-major 下连续线程沿 M 访问连续 smem 地址，天然无 bank conflict，SMEM 不需要 swizzle 或 padding。

### TN SIMT（L505-580）

```cpp
auto dA = make_stride(ldA, Int<1>{});                      // A: K-major (M,K):(ldA,1)
auto sA_atom = make_layout(make_shape (      bM,          bK),
                           make_stride(Int<1>{}, bM+Int<1>{})); // padding: stride=bM+1

TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>{},        // 标量搬运, 非 uint128_t
    Layout<Shape<_32,_8>,Stride<_8,_1>>{},                 // 32x8 K-major 线程布局
    Layout<Shape< _1,_1>>{});                              // 1x1 逐元素
// TiledMMA, S→R: 同 NT SIMT
```

K-major 下连续线程沿 M 方向分布，但 global memory 中同一 K 位置、不同 M 行的元素地址相差 ldA（整行宽度），远非连续，无法合并为一次内存事务，只能逐元素标量搬运。SMEM 用 padding（stride = bM+1 而非 bM）使同一 K 列的相邻 M 行元素不再落到同一 bank。

### 三版本对比

| 特性 | NT SIMT | TN SIMT | TN HGEMM |
|-----|---------|---------|----------|
| G→S 搬运 | 128b 向量化 | 标量逐元素 | 128b 向量化 |
| SMEM 防 bank conflict | 无需 | padding (+1) | Swizzle\<3,3,3\> |
| MMA 指令 | UniversalFMA 16×16×1 | UniversalFMA 16×16×1 | SM80_16x8x16 Tensor Core |
| ldmatrix retiling | 否 | 否 | SM75_U32x4_LDSM_N |
| 寄存器级流水线 | 否 | 否 | K_BLOCK 预取 |
| BK | 8 | 8 | 64 |

三个版本共用一个 kernel 模板。差异全部在 host 侧 Atom 和 Layout 参数——CuTe 在编译期将同一套抽象（Tensor / copy / gemm）特化为完全不同的 PTX 指令路径。

---

## main 与分发（L582-718）

```cpp
void gemm(char transA, char transB, ...) {
    if (transA == 'N' && transB == 'T') return gemm_nt(...);           // → NT SIMT
    if (transA == 'T' && transB == 'N') return gemm_tn(...);           // → TN HGEMM (half_t 特化)
    assert(false && "Not implemented");
}
```

`gemm_tn` 对 `half_t` 有特化重载（L325），优先匹配 TN HGEMM 路径；其他类型走泛型 TN SIMT（L505）。`main` 默认参数 M=5120, N=5120, K=4096, transA='N', transB='T'。
