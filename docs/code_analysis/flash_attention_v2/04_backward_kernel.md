---
tags:
  - CUDA
  - CUTLASS
  - Flash Attention
---
# Flash Attention V2：反向传播

本文拆解 FA2 反向传播的核心策略与实现。

**源码**: [flash_bwd_kernel.h](src/flash_bwd_kernel_h.md)、[flash_bwd_preprocess_kernel.h](src/flash_bwd_preprocess_kernel_h.md)

## 反向传播的数学

给定前向输出 $O = \text{softmax}(QK^T / \sqrt{d}) \cdot V$ 和损失梯度 $dO$，需要计算 $dQ, dK, dV$。

关键中间量：

$$
S = QK^T / \sqrt{d}, \quad P = \text{softmax}(S), \quad O = PV
$$

$$
dV = P^T \cdot dO, \quad dP = dO \cdot V^T
$$

$$
D_i = \text{rowsum}(dO_i \odot O_i), \quad dS = P \odot (dP - D)
$$

$$
dQ = dS \cdot K / \sqrt{d}, \quad dK = dS^T \cdot Q / \sqrt{d}
$$

## 核心策略：Recomputation

FA2 反向**不存储** attention matrix $P$。前向只保存 $O$ 和 $LSE$（log-sum-exp），反向时**重新计算** $S → P$：

1. 从 $Q, K$ 重新计算 $S = QK^T$
2. 从 $LSE$ 恢复 $P$：$P_{ij} = \exp(S_{ij} \cdot \text{scale} - LSE_i)$

这用 $O(N)$ 额外计算换取 $O(N^2)$ 内存节省。

## 两步执行

### Step 1：预处理（compute_dot_do_o）

[flash_bwd_preprocess_kernel.h:L57-L80](src/flash_bwd_preprocess_kernel_h.md#__codelineno-0-57)

计算 $D_i = \text{rowsum}(dO_i \odot O_i)$，写入 `dsoftmax_sum`。

```cpp
'''
dot_do_o: 对每行计算 dO · O 的点积
- 先 reshape dO 和 O 的 MMA 布局
- 线程内求和 + Allreduce 跨线程同步
- 结果写到 global memory 的 dsoftmax_sum
'''
void dot_do_o(Tensor do_, Tensor o, Tensor dP_sum, ...) {
    for (int mi ...) {
        float dP_sum_cur = 0;
        for (int ni ...) { dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni); }
        dP_sum_cur = Allreduce<THREADS_PER_ROW>::run(dP_sum_cur, sum_op) * scale;
        dP_sum(mi * col_stride + tidx / THREADS_PER_ROW) = dP_sum_cur;
    }
}
```

如果 `Clear_dQaccum=true`，同时清零 `dq_accum_ptr`（反向累积缓冲）。

### Step 2：主反向 kernel（compute_dq_dk_dv_1colblock）

[flash_bwd_kernel.h:L80-L841](src/flash_bwd_kernel_h.md#__codelineno-0-80)

**与前向的关键区别：外循环方向翻转**

- 前向：外循环 Q 行块，内循环 K/V 块
- 反向：**外循环 K/V 列块**（每个 thread block 固定一个 K/V 块），**内循环 Q/dO 块**

这是因为 $dK$ 和 $dV$ 需要对所有 Q 行求和：$dK_j = \sum_i dS_{ij}^T Q_i$，固定 K/V 块可在寄存器中累积 `acc_dk` 和 `acc_dv`。

### Shared Memory 布局

[flash_bwd_kernel.h:L156-L177](src/flash_bwd_kernel_h.md#__codelineno-0-156)

反向需要同时驻留多个矩阵：

```
smem 分配：
├── sQ   (+ double buffer)     # Q 块，可能双缓冲
├── sdO  (+ double buffer)     # dO 块
├── sK                         # K 块（固定在整个 Q 迭代中）
├── sV                         # V 块（固定）
├── sdS                        # dS = P ⊙ (dP - D) 中间结果
├── sP / sdQ                   # P 和 dQ 共享同一 smem（不同时使用）
```

### 三组 TiledMMA

反向涉及 5 个矩阵乘法，分为三组 MMA（参见 [01_params_and_traits.md](01_params_and_traits.md)）：

| MMA 组 | 用途 | 操作 |
|--------|------|------|
| `TiledMmaSdP` | S 和 dP 计算 | $S = Q \cdot K^T$, $dP = dO \cdot V^T$ |
| `TiledMmadKV` | dK 和 dV 累积 | $dK += dS^T \cdot Q$, $dV += P^T \cdot dO$ |
| `TiledMmadQ` | dQ 计算 | $dQ = dS \cdot K$ |

### 主循环流程

对于固定的 K/V 列块 `n_block`，从后向前迭代所有 Q 块：

```
初始化：加载 K, V → smem（整个循环不变）
clear(acc_dk), clear(acc_dv)

for m_block = m_block_max-1 → m_block_min:
  1. 加载 Q[m_block], dO[m_block] → smem
  2. 加载 LSE[m_block], D[m_block] → register

  3. GEMM: acc_s = Q · K^T                  (TiledMmaSdP)
  4. Recompute P: P = exp(S * scale - LSE)
  5. 应用 mask、dropout

  6. GEMM: acc_dp = dO · V^T                (TiledMmaSdP)
  7. dS = P ⊙ (dP - D)                      # 逐元素
  8. 写 dS → smem, 写 P → smem

  9. GEMM: acc_dv += P^T · dO               (TiledMmadKV)
  10. GEMM: acc_dk += dS^T · Q              (TiledMmadKV)

  11. GEMM: dQ_block = dS · K               (TiledMmadQ)
  12. dQ_block → smem → global (atomic add)
```

### dQ 的 Atomic Add

$dQ$ 的计算分散在多个 K/V 列块中（每个列块贡献一部分 dQ），需要跨 thread block 累积。FA2 使用两种策略：

- **非确定性模式**: `atomicAdd` 到 `dq_accum_ptr`（fp32 缓冲），最后转回 fp16
- **确定性模式**: 每个 thread block 写入独立的 `dq_accum` 分片，最后规约

### Double Buffer

Q 和 dO 使用双缓冲（`Double_buffer = !No_double_buffer`）：当前 Q 块在计算时，下一个 Q 块已在异步加载。smem 中分配 3 份 Q 空间（2 份双缓冲 + 1 份 dO）。

## 调度入口

[flash_bwd_kernel.h 尾部](src/flash_bwd_kernel_h.md#__codelineno-0-800)

```cpp
template<...>
void compute_dq_dk_dv(const Params &params) {
    const int n_block = blockIdx.x;    // 外循环：K/V 列块
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;
    compute_dq_dk_dv_1colblock<..., Is_first=true, Is_last=true>(
        params, bidb, bidh, n_block);
}
```

Grid 维度：`(ceil_div(seqlen_k, kBlockN), batch, nheads)`，每个 thread block 处理一个 K/V 列块。

## 与前向的对比

| 方面 | 前向 | 反向 |
|------|------|------|
| 外循环 | Q 行块 (blockIdx.x) | K/V 列块 (blockIdx.x) |
| 内循环 | K/V 块 | Q/dO 块 |
| 累积器 | acc_o (输出 O) | acc_dk, acc_dv (梯度) |
| Smem 驻留 | Q (固定) + K/V (迭代) | K/V (固定) + Q/dO (迭代) |
| 额外 GEMM | 无 | dQ = dS · K |
| MMA 组数 | 1 组 | 3 组 |
| Smem 用量 | ~64 KB | ~128-192 KB |
| 跨 block 同步 | 无 | dQ atomic add |
