---
tags:
  - CUTLASS
  - CUDA
---

# CuTe 实战：sgemm_sm80.cu 拆解

> **源码**: [NVIDIA/cutlass - examples/cute/tutorial/sgemm_sm80.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu)
> **许可证**: BSD-3-Clause, Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES

本文按源码顺序完整拆解 `sgemm_sm80.cu`，以 **TN HGEMM**（half_t，Tensor Core）路径为主线。这个文件用一个通用 kernel 模板 `gemm_device`（L54-310）实现了三种 GEMM 路径（NT SIMT / TN SIMT / TN HGEMM），差异全部通过 host 侧模板参数传入。

---

## 存储空间布局

TN HGEMM 路径中数据经过三级存储，每级存储中数据的 shape 和含义如下：

### Global Memory

每个 CTA 负责 C 矩阵的一个 128×128 输出 tile，需要遍历 K 方向的所有 tile 做归约：

| Tensor | Shape | 含义 |
|--------|-------|------|
| mA | (M, K) | 完整 A 矩阵，K-major，stride = (ldA, 1) |
| mB | (N, K) | 完整 B 矩阵，K-major，stride = (ldB, 1) |
| mC | (M, N) | 完整 C 矩阵，M-major，stride = (1, ldC) |
| gA | (128, 64, k) | 当前 CTA 的 A tile，第三维 k = ceil(K/64) 是 K 方向 tile 数 |
| gB | (128, 64, k) | 当前 CTA 的 B tile |
| gC | (128, 128) | 当前 CTA 的 C tile |

### Shared Memory（3 级 Pipeline）

SMEM 中为 A 和 B 各分配 3 个 buffer（pipeline stage），每个 buffer 存一个 128×64 tile：

| Tensor | Shape | Layout | 含义 |
|--------|-------|--------|------|
| sA | (128, 64, 3) | Swizzle\<3,3,3\> | A 的 SMEM buffer，3 级流水线 |
| sB | (128, 64, 3) | Swizzle\<3,3,3\> | B 的 SMEM buffer |

每个 128×64 tile 占 128×64×2B = 16KB（half_t），A+B 共 3 个 stage = 6×16KB = 96KB。Swizzle 通过 XOR 变换消除 `ldmatrix` 列访问时的 bank conflict（详见[编程模型 §5 Swizzle](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)）。

### Registers（K_BLOCK 级 Pipeline）

每个线程持有 MMA 操作数 fragment 和累加器，fragment 按 K_BLOCK 维度做寄存器级流水线：

| Tensor | Shape | 含义 |
|--------|-------|------|
| tCrA | (MMA, MMA_M=4, MMA_K=4) | A 操作数寄存器 fragment |
| tCrB | (MMA, MMA_N=4, MMA_K=4) | B 操作数寄存器 fragment |
| tCrC | (MMA, MMA_M=4, MMA_N=4) | 累加器，跨所有 k_tile 累加 |

- MMA_M=4：M 方向 128/32 = 4 次迭代
- MMA_N=4：N 方向 128/32 = 4 次迭代
- MMA_K=4：K 方向 64/16 = 4 个 k_block（内循环长度）
- MMA（第 0 维）：单个 MMA atom 每线程持有的元素数

### 数据流总览

```
Global Memory ──cp.async 128b──→ SMEM (3 级 pipe) ──ldmatrix──→ Registers (4 级 k_block) ──mma.sync──→ Accumulator ──axpby──→ Global
```

每级之间的搬运操作和对应的 CuTe 抽象：

| 搬运 | 指令 | CuTe 抽象 | 线程协作 |
|------|------|-----------|---------|
| Global→SMEM | cp.async.ca.shared.global 128b | TiledCopy + SM80_CP_ASYNC_CACHEALWAYS\<uint128_t\> | 128 线程 |
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
auto bK = Int< 64>{};                                      // BK=64: 一条 mma 消费 16K，4 次 k_block 迭代
auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
auto bP = Int<3>{};                                         // SMEM pipeline 深度

// ── SMEM Layout: Swizzle ──
// 基础 atom: 8×64 K-major, composition 做 XOR 消除 bank conflict
// tile_to_shape 将 8×64 atom 平铺到 128×64×3
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
- **Swizzle 原理**：K-major 下 ldmatrix 沿列读取，连续行同列落到相同 bank。`Swizzle<3,3,3>` 对 bit[5:3] 做 XOR bit[8:6]，使连续 8 行同列分散到 8 个 bank。详见[编程模型 §5.3](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- **G→S 线程布局为 K-major**：global 是 K-major 存储，连续线程沿 K 排列 + 每线程搬 8 个连续 K 元素 = 合并访问。一次 copy 搬 16×64 = 1024 个 half_t，需 128/16=8 次迭代覆盖 128 行。参见[算法 §copy](04_algorithms.md)、[编程模型 §3.3 TiledCopy](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- **TiledMMA 参数**：atom 16×8×16 + 2×2 layout = 32×16×16，Tile 指定 32×32×16，差值由 N 方向迭代 2 次补齐。完整 CTA tile 128×128 需要 MMA_M=4, MMA_N=4 次迭代，K 方向 MMA_K=4 次内循环。参见 [MMA Atom](05_mma_atom.md)、[编程模型 §2.5 TiledMMA](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- **ldmatrix 而非 ld.shared**：`mma.sync` 要求操作数寄存器分布匹配特定模式，`ldmatrix` 一步到位加载到正确布局，省去线程间 shuffle。参见[编程模型 §3.1](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)

---

## gemm_device Kernel（L54-310）

以下按 kernel 代码从上到下的顺序走。所有 shape 标注基于 TN HGEMM 参数值。

### Tensor 构造、分区与 Retiling（L68-184）

```cpp
// ── 全局 Tensor 构造 + CTA 分区 ──
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);               // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});   // (128,64,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});   // (128,64,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});   // (128,128)

// ── SMEM Tensor ──
extern __shared__ char shared_memory[];
using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);    // (128,64,3)
Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);    // (128,64,3)

// ── G→S Copy 分区：每个线程看到自己负责搬运的部分 ──
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);                              // (CPY,CPY_M,CPY_K,k)
Tensor tAsA = thr_copy_a.partition_D(sA);                              // (CPY,CPY_M,CPY_K,PIPE)
ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
Tensor tBgB = thr_copy_b.partition_S(gB);                              // (CPY,CPY_N,CPY_K,k)
Tensor tBsB = thr_copy_b.partition_D(sB);                              // (CPY,CPY_N,CPY_K,PIPE)

// ── MMA 分区 + 寄存器 Fragment ──
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCgC = thr_mma.partition_C(gC);                                 // (MMA,MMA_M,MMA_N)
Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                 // (MMA,MMA_M,MMA_K)
Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                 // (MMA,MMA_N,MMA_K)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);                           // (MMA,MMA_M,MMA_N)
clear(tCrC);                                                            // 累加器清零

// ── S→R Copy Atom Retiling ──
// make_tiled_copy_A/B: 根据 MMA 的操作数布局 + ldmatrix Atom
// 自动构建 TiledCopy, 线程排列匹配 MMA, 值排列匹配 ldmatrix
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
- **partition_S / partition_D 的 shape**：CPY 是单条指令消费的元素数（8 个 half_t），CPY_M/CPY_K 是该线程的迭代次数。源和目标用相同分区模式，保证 `copy(src, dst)` 一一对应
- **partition_fragment_A/B**：传入 `sA(_,_,0)` 而非整个 sA——只需单个 pipe stage 的 shape 确定 fragment 大小。创建的是 ArrayEngine Tensor（寄存器存储），不是视图
- **Retiling 原理**：`tCrA` 和 `tXrA` 指向**同一块寄存器**，第一维的逻辑切分不同。MMA 视角是 `(mma, mma_m, mma_k)`，ldmatrix 视角是 `(cpy, mma_m, mma_k)`。`retile_D` 将第一维重新解释，不移动数据。`make_tiled_copy_A/B` 查询 MMA 的线程-值映射，与 ldmatrix 的 Copy_Atom 对齐，生成兼容两者的 TiledCopy

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
while (k_tile_count > -(K_PIPE_MAX-1))                                 // 含 drain 阶段
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

**外循环终止条件** `k_tile_count > -(K_PIPE_MAX-1)`：prologue 预加载了 2 个 tile 并递减 `k_tile_count`。主循环继续递减直到 `-1`——此时最后 2 个 tile 已在 prologue 阶段加载但尚未计算，外循环的最后 2 次迭代负责消费它们（drain 阶段）。

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

**Epilogue**：`axpby` 逐元素计算 `tCgC = α·tCrC + β·tCgC`。tCrC 是寄存器 fragment，tCgC 是 global memory 视图，shape 一致。这是极简 epilogue——无 smem 中转、无向量化写回。

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

K-major 下连续线程沿 M 跨行访问 global（地址间距 = ldA），无法向量化，只能标量搬运。SMEM 用 padding（stride = bM+1）错开 bank 分配。

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
