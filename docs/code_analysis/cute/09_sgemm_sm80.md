---
tags:
  - CUTLASS
  - CUDA
---

# CuTe 实战：sgemm_sm80.cu 拆解

> **源码**: [NVIDIA/cutlass - examples/cute/tutorial/sgemm_sm80.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu)
> **许可证**: BSD-3-Clause, Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES

本文按源码顺序完整拆解 `sgemm_sm80.cu`。这个文件用一个通用 kernel 模板 `gemm_device` 实现了三种 GEMM 路径（NT SIMT / TN SIMT / TN HGEMM），差异全部通过 host 侧模板参数传入。本文以 **TN HGEMM**（half_t，Tensor Core，最复杂的路径）为主线，涵盖 CuTe 的 Tensor 构造、TiledCopy、TiledMMA、Swizzle SMEM、ldmatrix retiling、双层流水线等核心特性。

文件结构概览：

- L44-52：`SharedStorage` 结构体
- L54-310：`gemm_device` kernel 模板（通用，三个版本共用）
- L325-430：`gemm_tn` half_t 特化（TN HGEMM host 侧配置）
- L432-503：`gemm_nt` 泛型（NT SIMT host 侧配置）
- L505-580：`gemm_tn` 泛型（TN SIMT host 侧配置）
- L582-600：`gemm` 分发函数
- L603-718：`main`

---

## SharedStorage（L44-52）

```cpp
template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage
{
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};
```

共享内存的存储结构。`cosize_v<Layout>` 返回 Layout 作为函数的**陪域大小**（最大偏移 + 1），而非 `size`（定义域大小）。Swizzle 等非线性 Layout 的 cosize 可能大于 size，用 cosize 保证所有偏移都在合法范围内。这个结构体会在 kernel 内通过 `reinterpret_cast` 映射到动态共享内存。

---

## Host 侧配置：gemm_tn HGEMM（L325-430）

Kernel 的所有行为由 host 侧传入的模板参数决定。先看 host 侧如何构造这些参数，再进入 kernel 按代码顺序走。

```cpp
// TN 布局：A 和 B 都是 K-major
// - A (M,K):(ldA, 1) → 沿 K 方向步长为 1，沿 M 方向跨 ldA 个元素
// - B (N,K):(ldB, 1) → 同理
// - C (M,N):(1, ldC) → M-major 输出
auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int< 64>{};
auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
auto bP = Int<3>{};                                         // 3 级 pipeline
```

CuTe 不用 BLAS 的 T/N 标记，而是直接用 stride 说明哪个维度连续。`Int<1>{}` 是编译时常量，编译器生成更高效的地址计算。BK=64 远大于 SIMT 版的 8——一条 `mma.sync.m16n8k16` 消费 16 个 K 元素，BK=64 内含 4 个 k_block 迭代，保证足够的算术强度。关于 M-major / K-major 的完整对照，参见 [GEMM 教程 §M-major/K-major 表](06_gemm_tutorial.md)。

### SMEM Layout：Swizzle

```cpp
// Swizzle atom 构造：
// 1. 基础 atom: 8×64 的 K-major 布局
//    (Shape<_8, Shape<_8,_8>>, Stride<_8, Stride<_1,_64>>)
//    外层 8 行，内层 8×8 = 64 列（分成两组各 8 列）
// 2. composition(Swizzle<3,3,3>{}, atom): XOR 变换消除 bank conflict
// 3. tile_to_shape: 将 8×64 atom 平铺到 128×64×3 (bM×bK×bP)
auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                Layout<Shape <_8,Shape <_8, _8>>,
                                       Stride<_8,Stride<_1,_64>>>{});

auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
auto sC = make_layout(make_shape(bM, bN));
```

K-major 布局下，`ldmatrix` 需要沿 M/N 方向读取列数据。连续行的同一列地址落到相同 bank，导致严重 bank conflict。`Swizzle<3,3,3>` 对地址的 bit[5:3]（bank 分配位）做 XOR bit[8:6]（行号位），使连续 8 行的同一列分散到 8 个不同 bank。第三维 `bP=3` 是 pipeline 深度——SMEM 中同时存 3 个 tile 的缓冲区。详细原理参见[编程模型 §5 Swizzle](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)。

### TiledCopy：Global → Shared Memory

```cpp
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
    Layout<Shape< _1,_8>>{});               // Val layout  1x8 k-major
TiledCopy copyB = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
    Layout<Shape<_16,_8>,Stride<_8,_1>>{},
    Layout<Shape< _1,_8>>{});
```

`make_tiled_copy` 三个参数（参见 [算法 §copy](04_algorithms.md)、[编程模型 §3.3 TiledCopy](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)）：

- **Copy_Atom**：`SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` 使用 `cp.async.ca.shared.global` 搬运 128b，绕过寄存器直达 smem
- **Thr layout**：16×8，`Stride<_8,_1>` 即 K-major 排列——连续线程 ID 先填 K 方向的 8 列再换行。128 线程 = 16 行 × 8 列
- **Val layout**：`Shape<_1,_8>` 即每线程搬 1 行 × 8 列 = 8 个 half_t = 16B = 128b

线程布局为什么是 K-major？因为 global memory 中 A/B 是 K-major 存储，让连续线程在 K 方向排列、每线程搬 8 个连续 K 元素，保证 global memory 合并访问。一次 copy 调用搬 16×(8×8) = 1024 个 half_t = 2KB，需要 128/16=8 次迭代覆盖 BLK_M=128 行。

### TiledMMA：Tensor Core 计算

```cpp
TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                               Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
                               Tile<_32,_32,_16>{});      // 32x32x16 Tiled MMA
```

`make_tiled_mma` 三个参数（参见 [MMA Atom](05_mma_atom.md)、[编程模型 §2.5 TiledMMA](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)）：

- **MMA Atom**：PTX 指令 `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`，32 线程（1 warp）协作计算 16×8 输出
- **Atom layout** `<_2,_2>`：2×2 排列 = 4 个 atom → 4 个 warp → 128 线程。沿 M 铺 2 个（16×2=32），沿 N 铺 2 个（8×2=16）
- **Tile** `<_32,_32,_16>`：最终 tile 32×32×16。2 个 atom 沿 N 覆盖 16，但 Tile 指定 32，差值由每个 atom 在 N 方向迭代 32/(8×2)=2 次补齐

完整 CTA tile 128×128 需要：M 方向 128/32=4 次 MMA_M 迭代，N 方向 128/32=4 次 MMA_N 迭代，K 方向 64/16=4 次 K_BLOCK 迭代。

### S→R Copy Atom：ldmatrix

```cpp
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;
```

`ldmatrix.sync.aligned.x4.m8n8.shared.b16`：每线程加载 4×32bit = 16B，32 线程协作加载 4 个 8×8 的 16-bit 矩阵片段，寄存器分布直接匹配 `mma.sync` 操作数布局，无需额外 shuffle。

注意这里只是 Copy_Atom，不是 TiledCopy。它在 kernel 内通过 `make_tiled_copy_A(s2r_atom_a, mma)` 与 TiledMMA 绑定后才构成完整的 S→R TiledCopy（见 kernel 内的 [Copy Atom Retiling](#copy-atom-retiling) 部分）。

为什么不直接 `ld.shared`？因为 `mma.sync` 要求操作数在寄存器中的分布匹配特定模式（参见[编程模型 §2.2 寄存器分布](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)），`ldmatrix` 一步到位，普通 `ld.shared` 还需要线程间 shuffle。

### Kernel 启动

```cpp
int smem_size = int(sizeof(SharedStorage<cute::half_t, cute::half_t,
                                         decltype(sA), decltype(sB)>));
dim3 dimBlock(size(mmaC));                                  // 128 线程
dim3 dimGrid(size(ceil_div(M, bM)),
             size(ceil_div(N, bN)));

cudaFuncSetAttribute(kernel_fptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
cudaFuncSetAttribute(kernel_fptr,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);   // L1 全部划给 SMEM

kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>
    (prob_shape, cta_tiler,
     A, dA, sA, copyA, s2r_atom_A,
     B, dB, sB, copyB, s2r_atom_B,
     C, dC, sC, mmaC,
     alpha, beta);
```

`size(mmaC)` = 128（4 个 warp × 32 线程/warp）。`PreferredSharedMemoryCarveout = 100` 将 L1/SMEM 可配置区域全部划给 SMEM。所有静态参数（Layout、TiledCopy、TiledMMA、Copy_Atom）都通过模板参数传入 kernel——kernel 本身完全泛型。

---

## gemm_device Kernel（L54-310）

以下按 kernel 代码从上到下的顺序逐段讲解。所有 shape 标注基于 TN HGEMM 的参数值。

### 模板参数与静态断言（L54-91）

```cpp
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout,
            TiledCopyA copy_a, S2RAtomA s2r_atom_a,
            TB const* B, BStride dB, BSmemLayout sB_layout,
            TiledCopyB copy_b, S2RAtomB s2r_atom_b,
            TC      * C, CStride dC, CSmemLayout,
            TiledMma mma,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  // 静态断言：rank 检查、线程数一致性、Layout 尺寸匹配
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)
  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // 线程数一致
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));
  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  // ... 更多尺寸和步长兼容性检查
```

`__launch_bounds__` 使用 `decltype(size(TiledMma{}))::value` 从 TiledMMA 静态推导线程数上限（128），帮助编译器优化寄存器分配。所有断言在**编译期**执行，不产生运行时开销——这是 CuTe 类型系统的核心优势。

### Tensor 构造与 CTA 分区（L96-112）

```cpp
// 从原始指针 + shape + stride 构造完整矩阵 Tensor
// CuTe 惯例：A=(M,K), B=(N,K), C=(M,N)
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

// CTA 分区：每个 thread block 负责一个 (BLK_M, BLK_N) 输出 tile
// local_tile: 先用 cta_tiler 做 zipped_divide 分块，再用 cta_coord 切片
// Step 掩码控制参与分区的维度：
//   gA: Step<_1, X, _1> → 取 M 和 K，跳过 N
//   gB: Step< X, _1, _1> → 取 N 和 K，跳过 M
//   gC: Step<_1, _1, X> → 取 M 和 N，跳过 K
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

// 动态共享内存 → 类型化 Tensor
extern __shared__ char shared_memory[];
using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)
```

gA 的第三维 `k` 是该 CTA 沿 K 方向需要迭代的 tile 数量（= ceil(K / BLK_K)）。sA/sB 的第三维 PIPE=3 是流水线缓冲区。`make_gmem_ptr` / `make_smem_ptr` 标记地址空间，后续 `copy` 调用据此自动选择 `cp.async` 指令（参见[编程模型 §3.4 copy 分发逻辑](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)）。关于 `local_tile` 的语义，参见 [Layout 代数 §zipped_divide](02_layout_algebra.md) 和 [GEMM 教程 §CTA 分区](06_gemm_tutorial.md)。

### G→S Copy 分区（L118-129）

```cpp
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)
```

`partition_S` / `partition_D` 将完整 tile 按线程-值映射切分，每个线程只看到自己负责搬运的部分。对于 TN HGEMM：

- **CPY**（第 0 维）：单条 cp.async 指令消费的元素数 = 8 个 half_t = 16B = 128b
- **CPY_M**（第 1 维）：该线程在 M 方向的迭代次数 = 128/16 = 8
- **CPY_K**（第 2 维）：该线程在 K 方向的迭代次数 = 64/64 = 1
- **k / PIPE**（第 3 维）：全局 K tile 数 / SMEM pipeline stage 数

源和目标使用相同的分区模式，保证 `copy(tAgA(_,_,_,k), tAsA(_,_,_,pipe))` 一一对应。

### SMEM 预填充（L135-150）

```cpp
auto K_PIPE_MAX = size<3>(tAsA);                                     // 3（pipeline 深度）

int k_tile_count = size<3>(tAgA);                                    // 总 K tile 数
int k_tile_next = 0;                                                 // 下一个待加载的 K tile

// 将前 K_PIPE_MAX-1 = 2 个 K tile 异步加载到 smem pipe 0 和 pipe 1
// 每次 copy 发起一组 cp.async，cp_async_fence() 提交为一个 async group
// 不等待完成就继续下一组（异步）
CUTE_UNROLL
for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
}
```

Prologue 阶段。此时 pipe 0 和 pipe 1 正在异步填充，pipe 2 空闲。主循环第一次迭代会发起 pipe 2 的加载。关于 cp.async 的 commit/wait 语义，参见[编程模型 §4 cp.async 异步流水线](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)。

### MMA 分区与累加器（L156-170）

```cpp
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

// 创建寄存器 fragment（ArrayEngine，寄存器存储，不是视图）
// partition_fragment_A/B 传入 sA(_,_,0)：只需单个 pipe stage 的 shape 来确定 fragment 大小
Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));               // (MMA,MMA_N,MMA_K)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

clear(tCrC);                                                          // 累加器清零
```

shape 对应关系：

- **MMA**（第 0 维）：单个 MMA atom 每线程产生的元素数
- **MMA_M** = 128/32 = 4：M 方向迭代次数
- **MMA_N** = 128/32 = 4：N 方向迭代次数
- **MMA_K** = 64/16 = 4：K 方向迭代次数（= K_BLOCK_MAX，内循环长度）

### Copy Atom Retiling（L176-184）

这是本 kernel 最微妙的部分。MMA 分区产生的 `tCrA`（寄存器 fragment）的布局按 MMA 指令排列，但从 smem 读数据用 `ldmatrix` 指令，它有自己的线程-值映射。**两者布局不同**，需要 retiling 桥接。

```cpp
// make_tiled_copy_A: 根据 MMA 的 A 操作数布局和 ldmatrix 的 Copy_Atom
// 自动构建 TiledCopy，线程排列匹配 MMA，值排列匹配 ldmatrix
TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
// partition_S: 按 ldmatrix 的源布局分区 smem
Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
// retile_D: 将 MMA fragment 的布局重排为 ldmatrix 的目标布局
// 不创建新存储，只改变 tCrA 的逻辑视图
Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)
```

直觉理解：`tCrA` 和 `tXrA` 指向**同一块寄存器**，只是第一维的逻辑切分方式不同。MMA 视角是 `(mma, mma_m, mma_k)`，ldmatrix 视角是 `(cpy, mma_m, mma_k)`。`retile_D` 将 MMA 的第一维重新解释为 ldmatrix 的第一维，不移动任何数据。

`make_tiled_copy_A/B` 的关键作用是查询 MMA 的 ALayout/BLayout（线程-值到矩阵坐标的映射），与 ldmatrix 的 Copy_Atom 布局对齐，生成兼容两者的 TiledCopy。

### 流水线变量初始化（L224-246）

```cpp
int smem_pipe_read  = 0;                                              // 当前读取的 pipe stage
int smem_pipe_write = K_PIPE_MAX-1;                                   // 当前写入的 pipe stage (= 2)

// 获取 pipe 0 的 smem slice
Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

auto K_BLOCK_MAX = size<2>(tCrA);                                    // 4（寄存器 pipeline 深度）

// 寄存器预填充：等待 pipe 0 就绪，加载第 0 个 k_block 到寄存器
if (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE_MAX-2>();                                    // wait_group 1
    __syncthreads();

    // 用 ldmatrix 将 smem pipe 0 的 k_block 0 加载到寄存器
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
}
```

`cp_async_wait<K_PIPE_MAX-2>()` 即 `cp_async_wait<1>()`：允许最多 1 个 async group 未完成。prologue 提交了 2 个 group（pipe 0 和 pipe 1），wait<1> 确保 pipe 0 就绪，pipe 1 仍可能在飞行中。

此时状态：pipe 0 的 k_block 0 已在寄存器中，可以立即开始计算。

### 双层流水线主循环（L261-301）

```cpp
CUTE_NO_UNROLL
while (k_tile_count > -(K_PIPE_MAX-1))
{
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
        // ── 操作 1：切换 SMEM 读指针 + 等待 cp.async ──
        // 当前 k_tile 的最后一个 k_block 即将计算完毕
        // 准备好下一个 k_tile 的 smem 数据
        if (k_block == K_BLOCK_MAX - 1)
        {
            tXsA_p = tXsA(_,_,_,smem_pipe_read);
            tXsB_p = tXsB(_,_,_,smem_pipe_read);

            cp_async_wait<K_PIPE_MAX-2>();                            // wait_group 1
            __syncthreads();
        }

        // ── 操作 2：寄存器预取（每个 k_block 都做）──
        // 将下一个 k_block 的数据从 smem 加载到寄存器
        // 当 k_block == K_BLOCK_MAX-1 时，k_block_next == 0
        // 此时 tXsA_p 已指向新 pipe（操作 1 刚切换）
        // → 预取的是下一个 k_tile 的第 0 个 k_block
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // 编译时模运算
        copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
        copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));

        // ── 操作 3：发起下一个 k_tile 的 G→S cp.async（仅 k_block==0）──
        // 刚开始处理当前 k_tile 时，就启动下一个 k_tile 的异步搬运
        if (k_block == 0)
        {
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
            copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
            cp_async_fence();

            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_next; }

            // 推进 smem 读/写指针（循环 buffer）
            smem_pipe_write = smem_pipe_read;
            smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1)
                           ? 0 : smem_pipe_read+1;
        }

        // ── 操作 4：执行 MMA 计算（每个 k_block 都做）──
        // 使用当前 k_block 的寄存器数据执行 mma.sync
        // tCrC 跨所有 k_block 和 k_tile 累加
        gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }
}
```

**外循环终止条件** `k_tile_count > -(K_PIPE_MAX-1)`：prologue 已预加载 2 个 tile 并递减 `k_tile_count`。主循环继续递减，当降到 `-1` 时所有 tile 都已处理完毕。

**内循环**遍历一个 BLK_K=64 内的 4 个 16-element 子块。每次迭代中四个操作的位置精心安排以最大化重叠：

- 操作 1 放在 `k_block == K_BLOCK_MAX-1`：当前 k_tile 最后一个 k_block 时切换 smem 读指针并等待新数据
- 操作 2 每个 k_block 都做：预取下一个 k_block 的寄存器数据（S→R 与 MMA 重叠）
- 操作 3 放在 `k_block == 0`：刚开始当前 k_tile 时就启动下一个 k_tile 的 G→S 搬运（G→S 与 MMA 重叠）
- 操作 4 每个 k_block 都做：MMA 计算

时序示意（K_PIPE_MAX=3, K_BLOCK_MAX=4）：

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

三层重叠：G→S cp.async 在后台完成、S→R ldmatrix 预取下一个 k_block、MMA 计算当前 k_block——三者同时进行。

### Epilogue（L305-310）

```cpp
axpby(alpha, tCrC, beta, tCgC);
```

逐元素计算 `tCgC = alpha * tCrC + beta * tCgC`。`tCrC` 是寄存器 fragment，`tCgC` 是 global memory 的分区视图，两者 shape 一致 `(MMA,MMA_M,MMA_N)`。这是极简 epilogue——无 smem 中转、无向量化写回。实际高性能 kernel（如 CUTLASS 的 `CollectiveEpilogue`）会通过 smem 重排后向量化写回。

---

## NT SIMT 与 TN SIMT 版本

两个 SIMT 版本使用完全相同的 `gemm_device` kernel，差异仅在 host 侧参数。

### NT SIMT（L432-503）

NT 布局下 A 是 M-major `(M,K):(1,ldA)`、B 是 N-major `(N,K):(1,ldB)`：

```cpp
auto dA = make_stride(Int<1>{}, ldA);                      // M-major
auto dB = make_stride(Int<1>{}, ldB);                      // N-major

auto bK = Int<  8>{};                                      // BK=8（标量 FMA 吞吐低）
auto sA = make_layout(make_shape(bM, bK, bP));             // 列主序，无 swizzle

TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
    Layout<Shape<_32,_8>>{},                               // 32x8 M-major 线程布局
    Layout<Shape< _4,_1>>{});                              // 4x1 M-major 值布局 → 128b 向量化

TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                               Layout<Shape<_16,_16,_1>>{}); // 256 线程标量 FMA

// S→R: AutoVectorizingCopy（普通 ld.shared，不用 ldmatrix）
// kernel 启动时传入 Copy_Atom<AutoVectorizingCopy, TA>{}
```

M-major 下连续线程沿 M 访问连续 smem 地址，天然无 bank conflict，SMEM 不需要 swizzle 或 padding。线程布局 32×8 M-major + 值布局 4×1 M-major，每线程搬 4 个元素 = 128b 向量化。

### TN SIMT（L505-580）

TN 布局下 A/B 都是 K-major `(M,K):(ldA,1)`：

```cpp
auto dA = make_stride(ldA, Int<1>{});                      // K-major

// SMEM: padding 避免 bank conflict
auto sA_atom = make_layout(make_shape (      bM,          bK),
                           make_stride(Int<1>{}, bM+Int<1>{})); // stride = bM+1

TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>{},        // 注意是 TA 不是 uint128_t
    Layout<Shape<_32,_8>,Stride<_8,_1>>{},                 // 32x8 K-major 线程布局
    Layout<Shape< _1,_1>>{});                              // 1x1 标量搬运
```

与 NT SIMT 的关键差异：K-major 下连续线程沿 M 方向跨行访问 global，地址间距 = ldA，无法向量化，只能逐元素标量搬运（`Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>` 而非 `uint128_t`）。SMEM 用 padding（stride = bM+1 而非 bM）使沿 M 方向连续元素的 bank 偏移错开。TiledMMA 和 S→R 与 NT 版相同。

### 三版本 CuTe 特性对比

| CuTe 特性 | NT SIMT | TN SIMT | TN HGEMM |
|-----------|---------|---------|----------|
| G→S 搬运 | 128b 向量化 | 标量逐元素 | 128b 向量化 |
| SMEM 防 bank conflict | 无需 | padding (+1) | Swizzle<3,3,3> |
| MMA 指令 | UniversalFMA 16×16×1 | UniversalFMA 16×16×1 | SM80_16x8x16 Tensor Core |
| ldmatrix retiling | 否 | 否 | SM75_U32x4_LDSM_N |
| 寄存器级流水线 | 否 | 否 | K_BLOCK 预取 |
| BK | 8 | 8 | 64 |

三个版本共用一个 kernel 模板，差异全部在 host 侧模板参数。这正是 CuTe 的核心设计：同一套抽象（Tensor / copy / gemm），通过不同的 Atom 和 Layout 参数，在编译期特化为完全不同的 PTX 指令路径。

---

## main 与分发（L582-718）

```cpp
void gemm(char transA, char transB, ...) {
    if (transA == 'N' && transB == 'T') {
        return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    } else if (transA == 'T' && transB == 'N') {
        return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    assert(false && "Not implemented");
}
```

`main` 函数解析命令行参数（M/N/K/transA/transB），初始化随机数据，调用 `gemm` 分发。`gemm_tn` 对 `half_t` 有特化重载（L325），会优先匹配 TN HGEMM 路径；其他类型走泛型 TN SIMT 路径（L505）。默认参数 M=5120, N=5120, K=4096, transA='N', transB='T'（走 NT 路径）。
