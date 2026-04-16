---
tags:
  - CUDA
  - CUTLASS
  - Flash Attention
---
# Flash Attention V2：前向核心

本文详细拆解 FA2 前向的核心函数 `compute_attn_1rowblock`，这是整个代码库最关键的 1294 行代码。

**源码**: [flash_fwd_kernel.h](src/flash_fwd_kernel_h.md)

## 总体流程

每个 thread block 负责计算输出 O 的一个行块（kBlockM 行），通过迭代所有 K/V 块完成：

```
对于 Q 的第 m_block 个块（kBlockM 行）:
  1. 加载 Q[m_block] → shared memory
  2. 从后向前迭代 K/V 块 (n_block = n_block_max-1 → n_block_min):
     a. 加载 K[n_block] → smem (cp.async)
     b. GEMM: acc_s = Q · K^T                    ← Tensor Core
     c. 应用 mask (causal / local)
     d. Online softmax: 更新 max, sum, rescale acc_o
     e. 加载 V[n_block] → smem (cp.async)
     f. GEMM: acc_o += softmax(S) · V             ← Tensor Core
  3. Normalize: acc_o /= sum, 写出 O 和 LSE
```

## 函数签名与模板参数

[flash_fwd_kernel.h:L51-L52](src/flash_fwd_kernel_h.md#__codelineno-0-51)

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local,
         bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap,
         bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(
    const Params &params, const int bidb, const int bidh, const int m_block)
```

所有 bool 参数在编译期确定（通过 `static_switch.h` 的宏展开），避免运行时分支。`bidb`/`bidh` 来自 `blockIdx.y`/`blockIdx.z`，`m_block` 来自 `blockIdx.x`。

## 初始化与边界检查

[flash_fwd_kernel.h:L54-L128](src/flash_fwd_kernel_h.md#__codelineno-0-54)

```cpp
'''
基本设置：shared memory 指针、线程索引、block 常量
'''
extern __shared__ char smem_[];
const int tidx = threadIdx.x;

constexpr int kBlockM = Kernel_traits::kBlockM;   // Q 块行数，通常 128
constexpr int kBlockN = Kernel_traits::kBlockN;   // K/V 块行数，通常 64
constexpr int kHeadDim = Kernel_traits::kHeadDim;  // head 维度
constexpr int kNWarps = Kernel_traits::kNWarps;    // warp 数，通常 4

'''
边界计算：确定需要迭代的 K/V 块范围 [n_block_min, n_block_max)
- n_block_max: 不超过 seqlen_k 的最后一个块
- 若 causal: 进一步限制到 m_block 对角线以内
- 若 local: 同时考虑左右窗口边界
'''
const BlockInfo<!Is_even_MN> binfo(params, bidb);
if (m_block * kBlockM >= binfo.actual_seqlen_q) return;  // 超出序列长度，直接返回

const int n_block_min = !Is_local ? 0
    : max(0, (m_block * kBlockM + seqlen_k - seqlen_q - window_size_left) / kBlockN);
int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
if (Is_causal || Is_local) {
    n_block_max = min(n_block_max,
        ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q + window_size_right, kBlockN));
}
```

如果 `n_block_max <= n_block_min`（该行块不需要 attend 任何 KV），提前退出并写入 0/INFINITY。

## Tensor 构建：Global → Shared → Register

[flash_fwd_kernel.h:L138-L206](src/flash_fwd_kernel_h.md#__codelineno-0-138)

这一段是理解 FA2 如何使用 CuTe 的关键。参见 [CuTe Tensor](../cute/03_tensor.md)。

```cpp
'''
Global Memory Tensor 构建
- make_tensor: 从原始指针 + shape + stride 构建逻辑 tensor
- local_tile: 按 block 大小切分，取第 m_block/n_block 个 tile
'''
Tensor mQ = make_tensor(make_gmem_ptr(q_ptr + binfo.q_offset(...)),
                        make_shape(seqlen_q, h, d),
                        make_stride(q_row_stride, q_head_stride, _1{}));
Tensor gQ = local_tile(mQ(_, bidh, _),
                        Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));   // (kBlockM, kHeadDim)

Tensor gK = local_tile(mK(_, bidh / h_h_k_ratio, _),
                        Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));          // (kBlockN, kHeadDim, nblocksN)
                                                    //  注意：K 保留第 3 维用于迭代
Tensor gV = local_tile(mV(_, bidh / h_h_k_ratio, _),
                        Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));          // (kBlockN, kHeadDim, nblocksN)

'''
Shared Memory Tensor 构建
- sQ, sK, sV 在 smem 中按 SmemLayout 排列
- sVt: V 的转置视图（零开销，仅 layout 变换）
'''
Tensor sQ = make_tensor(make_smem_ptr(smem_), SmemLayoutQ{});
Tensor sK = make_tensor(sQ.data() + size(sQ), SmemLayoutKV{});
Tensor sV = make_tensor(sK.data() + size(sK), SmemLayoutKV{});
Tensor sVt = make_tensor(sV.data(), SmemLayoutVtransposed{});  // 转置视图

'''
Copy 与 MMA 的 partition
- gmem_thr_copy_QKV.partition_S/D: 按线程切分 global↔shared 的搬运任务
- thr_mma.partition_fragment_A/B: 按 MMA 线程切分寄存器 fragment
'''
auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);   // global Q 的每线程视图
Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);   // shared Q 的每线程视图
Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);   // (KCPY, KCPY_N, KCPY_K, nblocksN)
Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

auto thr_mma = tiled_mma.get_thread_slice(tidx);
Tensor tSrQ = thr_mma.partition_fragment_A(sQ);     // Q 的寄存器 fragment
Tensor tSrK = thr_mma.partition_fragment_B(sK);     // K 的寄存器 fragment
Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);  // V^T 的寄存器 fragment

Tensor acc_o = partition_fragment_C(tiled_mma,
    Shape<Int<kBlockM>, Int<kHeadDim>>{});           // 输出累积器 (fp32)
```

### Copy Atom Retiling

[flash_fwd_kernel.h:L192-L206](src/flash_fwd_kernel_h.md#__codelineno-0-192)

MMA 和 copy 操作对数据排布有不同要求。retiling 创建中间视图使 shared memory → register 的拷贝满足 MMA 输入要求：

```cpp
auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);   // smem→reg copy 的源视图

auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
Tensor tSsK = smem_thr_copy_K.partition_S(sK);

auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);  // V 用转置 copy atom
```

`make_tiled_copy_A/B` 将 `SmemCopyAtom`（ldmatrix 指令）与 `TiledMMA` 对齐，确保从 smem 读取的数据排布直接满足 Tensor Core 输入要求。参见 [CuTe GEMM 教程](../cute/06_gemm_tutorial.md)。

## Prologue：Q 和首个 K 块加载

[flash_fwd_kernel.h:L247-L283](src/flash_fwd_kernel_h.md#__codelineno-0-247)

```cpp
'''
1. 异步加载 Q → shared memory
'''
flash::copy<Is_even_MN, Is_even_K>(
    gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    binfo.actual_seqlen_q - m_block * kBlockM);      // 边界：实际行数
if (Kernel_traits::Is_Q_in_regs) { cp_async_fence(); }

'''
2. 如果 Share_Q_K_smem，等待 Q 加载完成后拷贝到寄存器，然后释放 smem 给 K 复用
'''
if (Kernel_traits::Share_Q_K_smem) {
    cp_async_wait<0>();
    __syncthreads();
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);  // smem → register
    __syncthreads();
}

'''
3. 异步加载首个 K 块 (n_block_max - 1) → shared memory
   注意：从后向前迭代，最后一个块可能需要边界 mask
'''
int n_block = n_block_max - 1;
flash::copy<Is_even_MN, Is_even_K>(
    gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    binfo.actual_seqlen_k - n_block * kBlockN);
cp_async_fence();

'''
4. 如果 Is_Q_in_regs 但不共享 smem，等 Q 加载完后拷贝到寄存器
'''
if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
    cp_async_wait<1>();    // 等 Q（保留 1 个 flight = K）
    __syncthreads();
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
}

clear(acc_o);
```

`cp.async` 异步拷贝与 `cp_async_fence`/`cp_async_wait` 的配合实现了 global → shared memory 的流水线。参见 [sgemm_sm80](../cute/09_sgemm_sm80.md)。

## 主循环：Masking 迭代

[flash_fwd_kernel.h:L298-L375](src/flash_fwd_kernel_h.md#__codelineno-0-298)

循环分为两阶段：需要 masking 的迭代和不需要 masking 的迭代。

```cpp
'''
第一阶段：需要 masking 的迭代
- 非 causal: 仅 1 次（最后一个块可能越界）
- causal: ceil_div(kBlockM, kBlockN) 次（对角线附近的块需要 causal mask）
'''
constexpr int n_masking_steps = (!Is_causal && !Is_local)
    ? 1
    : (Is_even_MN && Is_causal)
        ? ceil_div(kBlockM, kBlockN)
        : ceil_div(kBlockM, kBlockN) + 1;

#pragma unroll
for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
    Tensor acc_s = partition_fragment_C(tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});   // S 累积器 (fp32)
    clear(acc_s);

    '''
    流水线同步：等待上一轮 K 的 cp.async 完成
    '''
    cp_async_wait<0>();
    __syncthreads();

    '''
    预取 V：异步加载当前 n_block 的 V → shared memory
    第一次迭代需要处理边界（Clear_OOB_MN=true），后续迭代 V 总是完整的
    '''
    if (masking_step > 0) {
        flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV,
            tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
    } else {
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
            binfo.actual_seqlen_k - n_block * kBlockN);
    }
    cp_async_fence();

    '''
    GEMM 1: acc_s = Q · K^T
    - Q 可能在寄存器 (Is_Q_in_regs) 或 shared memory
    - K 始终在 shared memory
    - 结果 acc_s 在寄存器，shape (MMA=4, MMA_M, MMA_N)
    '''
    flash::gemm<Kernel_traits::Is_Q_in_regs>(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
        smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K);

    '''
    Softcap（可选）：将 attention score 限制在 [-softcap, softcap]
    '''
    if constexpr (Is_softcap) {
        flash::apply_softcap(acc_s, params.softcap);
    }

    '''
    Mask：对超出 causal/local 边界的位置设为 -inf
    '''
    mask.apply_mask<Is_causal, Is_even_MN>(
        acc_s, n_block * kBlockN,
        m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
        kNWarps * 16);

    '''
    流水线同步：等待 V 的 cp.async 完成
    同时预取下一个 K 块
    '''
    cp_async_wait<0>();
    __syncthreads();
    if (n_block > n_block_min) {
        flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV,
            tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
        cp_async_fence();
    }

    '''
    Online Softmax：更新 max, sum, rescale acc_o
    '''
    masking_step == 0
        ? softmax.softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal || Is_local>(
            acc_s, acc_o, params.scale_softmax_log2)
        : softmax.softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(
            acc_s, acc_o, params.scale_softmax_log2);

    '''
    类型转换 + Dropout
    - acc_s (fp32) → rP (fp16/bf16)
    - 可选：应用 dropout mask
    '''
    Tensor rP = flash::convert_type<Element>(acc_s);
    if (Is_dropout) {
        dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
    }

    '''
    GEMM 2: acc_o += P · V
    - P 在寄存器，需要 reshape 以匹配 MMA 的 A 操作数 layout
    - V 在 shared memory（转置视图 sVt）
    '''
    Tensor tOrP = make_tensor(rP.data(),
        flash::convert_layout_acc_Aregs<TiledMma>(rP.layout()));
    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma,
        smem_tiled_copy_V, smem_thr_copy_V);
}
```

### 关键细节：流水线调度

每轮迭代中，K 和 V 的加载与计算交替进行：

```
迭代 i:
  [compute] 等待 K[i] 就绪 → Q·K[i]^T → mask → softmax
  [memory]  异步加载 V[i]
  [compute] 等待 V[i] 就绪 → P·V[i]
  [memory]  异步加载 K[i-1]（下一轮用）
```

这种 K-ahead-of-V 的流水线确保了 compute 和 memory 的重叠。

## 主循环：无 Masking 迭代

[flash_fwd_kernel.h:L377-L429](src/flash_fwd_kernel_h.md#__codelineno-0-377)

结构与 masking 迭代完全相同，只是：
- mask 调用使用 `Causal_mask=false`
- V 加载不需要 `Clear_OOB_MN`
- `Is_first=false`（softmax 已经初始化）

## Epilogue：归一化与输出

[flash_fwd_kernel.h:L431-L493](src/flash_fwd_kernel_h.md#__codelineno-0-431)

```cpp
'''
1. 最终归一化：acc_o /= softmax_sum，同时计算 LSE
'''
Tensor lse = softmax.normalize_softmax_lse<Is_dropout>(
    acc_o, params.scale_softmax, params.rp_dropout);

'''
2. 类型转换：acc_o (fp32) → rO (fp16/bf16)
'''
Tensor rO = flash::convert_type<Element>(acc_o);

'''
3. Register → Shared Memory → Global Memory（两跳写出）
   不能直接 register → global，因为 MMA 输出的 register 排布不连续
'''
Tensor sO = make_tensor(sQ.data(), SmemLayoutO{});  // 复用 Q 的 smem 空间
auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);    // register → smem

__syncthreads();

cute::copy(gmem_tiled_copy_O, tOsO, tOrO);          // smem → register (重排)
flash::copy<Is_even_MN, Is_even_K, false, false>(    // register → global (带边界检查)
    gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
    binfo.actual_seqlen_q - m_block * kBlockM);

'''
4. 写出 LSE (log-sum-exp)
   每行一个标量，仅由该行的第一个线程写
'''
if (get<1>(taccOcO_row(0)) == 0) {
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccOcO_row(mi));
        if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
            gLSE(row) = lse(mi);
        }
    }
}
```

输出写回的两跳路径（register → smem → global）是因为 MMA 输出寄存器的数据分布在不同线程中，不能直接以连续方式写到 global memory。先写入 smem 让数据重新排列，再统一写出。

## Split-KV 变体

[flash_fwd_kernel.h:L498-L1071](src/flash_fwd_kernel_h.md#__codelineno-0-498)

`compute_attn_1rowblock_splitkv` 是 split-KV 版本，用于推理时的长序列。核心区别：

- 多个 thread block 分担同一行 Q 的不同 K/V 范围
- 每个 split 输出 partial O 和 partial LSE 到 `oaccum` / `lseaccum`
- 由 `combine_attn_seqk_parallel`（L1110）合并所有 split 的结果

Split-KV 还支持 **Paged KV Cache**：通过 `block_table` 间接寻址物理页，以及 **Append KV**：在推理时将新 token 的 K/V append 到 cache 并可选应用 RoPE。
