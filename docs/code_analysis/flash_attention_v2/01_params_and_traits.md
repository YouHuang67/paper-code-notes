---
tags:
  - CUDA
  - CUTLASS
  - Flash Attention
---
# Flash Attention V2：参数结构与硬件配置

本文拆解 FA2 的两个基础头文件：`flash.h` 定义了前向/反向的所有参数，`kernel_traits.h` 定义了 GPU 硬件相关的计算与内存访问配置。

## flash.h：参数结构体

**源码**: [flash.h](src/flash_h.md)

### 继承关系

```
Qkv_params          ← Q/K/V 指针与 stride
  └── Flash_fwd_params  ← 前向特有参数
        └── Flash_bwd_params  ← 反向额外参数（继承前向全部）
```

### Qkv_params（基类）

[flash.h:L21-L44](src/flash_h.md#__codelineno-0-21)

```cpp
struct Qkv_params {
    using index_t = int64_t;
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    index_t q_batch_stride;    // batch 维 stride
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;      // seqlen 维 stride
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;     // head 维 stride
    index_t k_head_stride;
    index_t v_head_stride;

    int h, h_k;                // h: query head 数, h_k: key/value head 数
    int h_h_k_ratio;           // h / h_k，用于 GQA/MQA
};
```

Q/K/V 按 `(batch, seqlen, head, dim)` 布局，stride 支持任意内存排列。`h_h_k_ratio` 预计算供 GQA 快速索引。

### Flash_fwd_params（前向）

[flash.h:L48-L143](src/flash_h.md#__codelineno-0-48)

关键字段分组：

**输出与 softmax**
- `o_ptr`：输出矩阵 O
- `softmax_lse_ptr`：每行的 log-sum-exp，shape `(b, h, seqlen_q)`，反向传播需要
- `oaccum_ptr` / `softmax_lseaccum_ptr`：split-KV 模式的中间累积

**维度**
- `b, seqlen_q, seqlen_k, d`：batch、序列长度、head dim
- `seqlen_q_rounded, seqlen_k_rounded, d_rounded`：对齐到 block 大小的 rounded 值

**缩放**
- `scale_softmax`：$1/\sqrt{d}$
- `scale_softmax_log2`：$\log_2(e) / \sqrt{d}$，用于 `exp2` 替代 `exp` 提升性能

**变长序列（Varlen）**
- `cu_seqlens_q/k`：cumulative sequence lengths，shape `(b+1,)`
- `leftpad_k`：左侧 padding

**KV Cache**
- `knew_ptr / vnew_ptr`：append 新 KV 到 cache
- `block_table`：Paged KV Cache 的页表

**可选特性**
- `p_dropout`：dropout 概率
- `window_size_left/right`：local attention 窗口
- `softcap`：softmax capping
- `rotary_cos_ptr/sin_ptr`：RoPE
- `alibi_slopes_ptr`：ALiBi
- `is_causal`：causal mask 标志

### Flash_bwd_params（反向）

[flash.h:L147-L185](src/flash_h.md#__codelineno-0-147)

继承 `Flash_fwd_params` 全部字段，额外增加：

- `do_ptr, dq_ptr, dk_ptr, dv_ptr`：梯度张量指针
- `dq_accum_ptr`：dQ 的 float32 累积缓冲（跨 block 原子加需要高精度）
- `dsoftmax_sum`：softmax 反向的中间值 $D_i = \text{rowsum}(dO_i \odot O_i)$
- `deterministic`：确定性模式标志

## kernel_traits.h：硬件配置

**源码**: [kernel_traits.h](src/kernel_traits_h.md)

### 继承关系

```
Flash_kernel_traits          ← 基础 MMA/Copy Atom
  ├── Flash_fwd_kernel_traits  ← 前向特有 Layout 与 Copy
  └── Flash_bwd_kernel_traits  ← 反向特有 Layout 与 Copy
```

### Flash_kernel_traits（基类）

[kernel_traits.h:L15-L46](src/kernel_traits_h.md#__codelineno-0-15)

```cpp
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;           // 累积器始终 float32

    // SM80: m16n8k16 Tensor Core 指令
    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

    // Shared memory → register 的 copy atom
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;           // ldmatrix 正常读
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>; // ldmatrix 转置读
};
```

核心选择：
- **MMA Atom**: `SM80_16x8x16_F32F16F16F32_TN` — 每条指令计算 16×8×16 的矩阵乘，输入 fp16/bf16，累积 fp32。参见 [CuTe MMA Atom](../cute/05_mma_atom.md)
- **SmemCopyAtom**: `SM75_U32x4_LDSM_N` — 使用 `ldmatrix` 指令批量加载 shared memory 到寄存器，满足 Tensor Core 的数据排布要求。参见 [CuTe 算法](../cute/04_algorithms.md)

### Flash_fwd_kernel_traits（前向）

[kernel_traits.h:L49-L159](src/kernel_traits_h.md#__codelineno-0-49)

模板参数：`kHeadDim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem`

**TiledMMA 构建**

```cpp
using TiledMma = TiledMMA<
    MMA_Atom_Arch,
    Layout<Shape<Int<kNWarps>,_1,_1>>,    // kNWarps 个 warp 沿 M 维排列
    Tile<Int<16 * kNWarps>, _16, _16>>;   // 每 warp 处理 16 行
```

4 个 warp 沿 M 维排列，总计覆盖 `kBlockM = 16 * kNWarps` 行（通常 64 或 128），N 和 K 维各 16。

**Shared Memory Layout（Swizzle）**

```cpp
static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

using SmemLayoutAtomQ = composition(
    Swizzle<kSwizzle, 3, 3>{},
    Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{});
using SmemLayoutQ = tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{});
```

Swizzle 消除 bank conflict：将 8 行 × kBlockKSmem 列的基本原子通过 `Swizzle<B,M,S>` 重排地址位，使同一 warp 内的线程访问不同 bank。参见 [GEMM 优化：Tensor Core 与 CUTLASS](../cutlass_gemm_blog/03_tensorcore_and_cutlass.md)

**V 的转置 Layout**

```cpp
using SmemLayoutVtransposed = composition(SmemLayoutKV{},
    make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{}));
```

P·V 的矩阵乘中 V 作为 B 操作数需要转置。这里不实际搬运数据，而是通过 CuTe 的 layout composition 在逻辑上将 KV 的 `(kBlockN, kHeadDim)` 重解释为 `(kHeadDim, kBlockN)`，配合 `SmemCopyAtomTransposed`（`ldmatrix.trans`）实现零开销转置。

**Global Memory Copy**

```cpp
using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;  // cp.async 128-bit
using GmemTiledCopyQKV = make_tiled_copy(
    Copy_Atom<Gmem_copy_struct, Element>{},
    GmemLayoutAtom{},                    // 线程排布
    Layout<Shape<_1, _8>>{});            // 每线程 8 个元素 = 128 bit
```

使用 SM80 的 `cp.async` 指令直接从 global memory 异步拷贝到 shared memory，绕过寄存器。`CACHEGLOBAL` 策略适合不会被同一 thread block 重复读取的数据。参见 [sgemm_sm80 实战](../cute/09_sgemm_sm80.md)

**Shared Memory 大小**

```cpp
static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);  // K + V
static constexpr int kSmemSize = Share_Q_K_smem
    ? max(kSmemQSize, kSmemKVSize)       // Q 和 K 共享 smem
    : kSmemQSize + kSmemKVSize;          // Q 独占 + K/V 独占
```

以 `d=128, kBlockM=128, kBlockN=64` 为例：`kSmemSize = 128×128×2 + 64×128×2×2 = 65536` bytes = 64 KB。

### Flash_bwd_kernel_traits（反向）

[kernel_traits.h:L163-L342](src/kernel_traits_h.md#__codelineno-0-163)

反向比前向多定义了三组 TiledMMA（对应三个矩阵乘法）：

- `TiledMmaSdP`：计算 $S = Q \cdot K^T$ 和 $dP = dO \cdot V^T$
- `TiledMmadKV`：计算 $dK = dP^T \cdot Q$ 和 $dV = P^T \cdot dO$
- `TiledMmadQ`：计算 $dQ = dP \cdot K$

每组 MMA 的 warp 排列不同（通过 `AtomLayoutMSdP/NdKV/MdQ` 控制），以平衡不同矩阵乘法的 M/N 维尺寸。

额外的 shared memory 分配：

```cpp
static constexpr int kSmemSize = kSmemQdOSize
    + kSmemKVSize + kSmemdSSize + max(kSmemPSize, kSmemdQSize);
```

反向需要同时驻留 Q/dO + K/V + dS + max(P, dQ)，shared memory 用量约为前向的 2-3 倍。
