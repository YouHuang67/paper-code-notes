---
tags:
  - CUDA
  - CUTLASS
  - Flash Attention
---
# Flash Attention V2：计算原语

本文拆解 FA2 的四个计算原语文件：online softmax、mask、warp reduce 工具函数、dropout。

## softmax.h：Online Softmax

**源码**: [softmax.h](src/softmax_h.md)

### 核心思想

标准 softmax 需要两遍扫描（第一遍求 max，第二遍求 exp 和 sum）。FA2 的 online softmax 在每个 K 块迭代中增量更新 `row_max` 和 `row_sum`，只需单遍扫描。

对应公式（参考 [FlashAttention 论文](https://arxiv.org/abs/2205.14135)）：

$$
m_i^{new} = \max(m_i^{old}, \max_j(S_{ij})), \quad
l_i^{new} = e^{m_i^{old} - m_i^{new}} \cdot l_i^{old} + \sum_j e^{S_{ij} - m_i^{new}}
$$

$$
O_i^{new} = e^{m_i^{old} - m_i^{new}} \cdot O_i^{old} + e^{S_{ij} - m_i^{new}} \cdot V_j
$$

### Softmax 结构体

[softmax.h:L128-L187](src/softmax_h.md#__codelineno-0-128)

```cpp
template <int kNRows>
struct Softmax {
    TensorT row_max, row_sum;    // 每行维护的 max 和 sum，shape (kNRows,)
};
```

`kNRows = 2 * MMA_M`，即每个线程负责的行数（MMA 输出的行数 × 2，因为 m16n8k16 的每个线程持有 2 行）。

### softmax_rescale_o：核心更新函数

[softmax.h:L136-L167](src/softmax_h.md#__codelineno-0-136)

```cpp
'''
acc_s: 当前 K 块的 attention score (MMA=4, MMA_M, MMA_N)
acc_o: 累积输出 (MMA=4, MMA_M, MMA_K)
softmax_scale_log2: log2(e) / sqrt(d)

关键优化：使用 exp2 替代 exp
  exp(x * scale - max * scale) = exp2(x * scale_log2 - max * scale_log2)
  exp2 映射到单条 PTX 指令，exp 需要多条
'''
template<bool Is_first, bool Check_inf=false>
void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    Tensor scores = make_tensor(acc_s.data(),
        convert_layout_acc_rowcol(acc_s.layout()));  // reshape 为 (nrow, ncol)

    if (Is_first) {
        reduce_max<true>(scores, row_max);           // row_max = max(scores) 跨线程
        scale_apply_exp2(scores, row_max, softmax_scale_log2);  // scores = exp2(...)
        reduce_sum<true>(scores, row_sum);            // row_sum = sum(scores) 线程内
    } else {
        '''
        非首次迭代：需要 rescale 之前累积的 acc_o
        '''
        copy(row_max, scores_max_prev);
        reduce_max<false>(scores, row_max);           // row_max = max(row_max, max(scores))

        for (int mi = 0; mi < size(row_max); ++mi) {
            float scores_scale = exp2f(
                (scores_max_prev(mi) - row_max(mi)) * softmax_scale_log2);
            row_sum(mi) *= scores_scale;              // rescale 旧 sum
            for (int ni ...) { acc_o(mi, ni) *= scores_scale; }  // rescale 旧 output
        }

        scale_apply_exp2(scores, row_max, softmax_scale_log2);
        reduce_sum<false>(scores, row_sum);           // row_sum += sum(new scores)
    }
}
```

### normalize_softmax_lse：最终归一化

[softmax.h:L169-L186](src/softmax_h.md#__codelineno-0-169)

```cpp
'''
所有 K 块迭代完成后调用：
  1. 跨线程 allreduce row_sum（迭代中只做了线程内 reduce）
  2. acc_o /= row_sum
  3. 计算 LSE = row_max * scale + log(row_sum)
'''
TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout) {
    quad_allreduce_(row_sum, row_sum, sum_op);    // 4 线程 allreduce
    for (int mi ...) {
        float inv_sum = 1.f / row_sum(mi);
        lse(mi) = row_max(mi) * softmax_scale + __logf(row_sum(mi));
        for (int ni ...) { acc_o(mi, ni) *= inv_sum; }
    }
    return lse;
}
```

### Reduce 操作

[softmax.h:L23-L63](src/softmax_h.md#__codelineno-0-23)

reduce 分两步：

1. **thread_reduce_**：每个线程对自己持有的元素求 max/sum
2. **quad_allreduce_**：4 个线程间通过 `__shfl_xor_sync` 交换数据

为什么是 4 个线程？因为 m16n8k16 MMA 中，同一行的数据分布在 4 个线程中（`lane_id % 4` 的 4 个线程共享同一行）。`Allreduce<4>` 递归展开为 `shfl_xor(2)` + `shfl_xor(1)` 两次交换。

## mask.h：Attention Mask

**源码**: [mask.h](src/mask_h.md)

### Mask 结构体

[mask.h:L111-L210](src/mask_h.md#__codelineno-0-111)

```cpp
template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {
    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;
};
```

### apply_mask 核心逻辑

输入 tensor 的 shape 为 `(MMA=4, MMA_M, MMA_N)`，先 reshape 为 `(nrow=(2,MMA_M), ncol=(2,MMA_N))`。

根据每个元素的 `(row_idx, col_idx)` 判断是否需要 mask：

- **Causal**: `col_idx >= row_idx + 1 + seqlen_k - seqlen_q + window_size_right` → 设为 `-INFINITY`
- **Local**: 额外检查左边界 `col_idx < row_idx + seqlen_k - seqlen_q - window_size_left`
- **ALiBi**: 不 mask，而是加偏置 `alibi_slope * col_idx`（causal）或减去 `alibi_slope * |row - col|`（非 causal）
- **!Is_even_MN**: `col_idx >= max_seqlen_k` → 设为 `-INFINITY`

行列索引的计算考虑了 MMA 的线程映射：每个线程持有特定行列位置的元素，通过 `tidx / 32`（warp id）和 `tidx % 32`（lane id）推算全局位置。

关联：mask 的 sliding window 机制与 [Native Sparse Attention 滑动窗口](../native_sparse_attention/05_sliding_window.md) 概念一致。

## utils.h：工具函数

**源码**: [utils.h](src/utils_h.md)

### Allreduce：Warp 级规约

[utils.h:L111-L131](src/utils_h.md#__codelineno-0-111)

```cpp
template<int THREADS>
struct Allreduce {
    template<typename T, typename Operator>
    static T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};
```

递归模板：`Allreduce<4>` → `shfl_xor(2)` → `Allreduce<2>` → `shfl_xor(1)`。两次 butterfly exchange 完成 4 线程 allreduce。

### gemm / gemm_rs：GEMM 封装

[utils.h:L135-L182](src/utils_h.md#__codelineno-0-135)

两个 GEMM 函数封装了 smem → register copy + MMA 的流水线：

```cpp
'''
gemm: A 和 B 都从 shared memory 加载（用于 Q·K^T）
- A_in_regs: 如果 Q 已在寄存器中，跳过 A 的 smem→reg copy
- K 维迭代：先 copy 下一个 K slice，再执行当前 K slice 的 MMA
'''
void gemm(acc, tCrA, tCrB, tCsA, tCsB, tiled_mma, ...) {
    copy(smem_tiled_copy_A, tCsA(_, _, 0), tCrA(_, _, 0));  // 预取第 0 个 K slice
    copy(smem_tiled_copy_B, tCsB(_, _, 0), tCrB(_, _, 0));
    for (int i = 0; i < K_tiles; ++i) {
        if (i < K_tiles - 1) {
            copy(tCsA(_, _, i+1), tCrA(_, _, i+1));  // 预取下一个
            copy(tCsB(_, _, i+1), tCrB(_, _, i+1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);  // 当前 MMA
    }
}

'''
gemm_rs: A 在寄存器，B 从 shared memory 加载（用于 P·V）
- P 已在寄存器中（从 softmax 输出直接使用）
- 只需要 B (V^T) 的 smem→reg copy
'''
void gemm_rs(acc, tCrA, tCrB, tCsB, tiled_mma, ...) { ... }
```

### Layout 转换

[utils.h:L186-L220](src/utils_h.md#__codelineno-0-186)

- `convert_layout_acc_rowcol`：MMA 累积器 `(MMA=4, MMA_M, MMA_N)` → `(nrow=(2,MMA_M), ncol=(2,MMA_N))`，将 MMA 维度的 4 拆为 2×2 对应行列
- `convert_layout_acc_Aregs`：将 softmax 输出（C 累积器格式）转换为 MMA 的 A 操作数格式，供 P·V 的 GEMM 使用

### copy 函数

[utils.h 后半部分](src/utils_h.md#__codelineno-0-220)

带边界检查的 copy 封装，通过 identity tensor（`cQ`/`cKV`）判断每个元素是否越界：

- `Is_even_MN=true`：序列长度恰好对齐 block 大小，无需边界检查
- `Is_even_K=true`：head dim 恰好对齐，无需 K 维边界检查
- `Clear_OOB_MN=true`：越界元素写 0（用于 V 加载，避免脏数据参与计算）

## dropout.h：Dropout

**源码**: [dropout.h](src/dropout_h.md)

FA2 使用 **Philox PRNG** 生成确定性 dropout mask。关键设计：

- **Seed + offset** 初始化 RNG，确保前向和反向生成相同的 dropout pattern
- offset 编码了 `(batch, head, lane_id)`，subsequence 编码了 attention matrix 中的 16×32 block 位置
- dropout mask 不存储，在前向/反向都实时生成

```cpp
struct Dropout {
    unsigned long long seed, offset;
    uint8_t p_dropout_uint8;

    void apply_dropout(Tensor &rP, int block_row_idx, int block_col_idx, int kNWarps) {
        // 根据 (block_row_idx, block_col_idx) 生成 Philox 随机数
        // 与 p_dropout 比较，小于阈值的元素乘以 rp_dropout (1/(1-p))
        // 大于阈值的元素设为 0
    }
};
```
