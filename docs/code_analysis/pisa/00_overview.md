# PISA 代码实现：Piecewise Sparse Attention

对应论文：[PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers](../../paper_reading/pisa.md)

官方代码位于 `refs/codes/piecewise-sparse-attention`，当前解析提交：
`655648df86cafd75042dc32d6b6be78c4ea0eca8`。公开仓库提供的是 attention backend 和 FLUX 示例，不是完整 Wan/Hunyuan 视频推理工程；因此本页重点解释开源 Triton 实现如何把论文中的 piecewise approximation 落到 GPU kernel。

## 先建立实现目标

PISA 的实现目标不是“做一个稀疏 mask 后调用普通 sparse attention”，而是把完整 softmax attention 拆成三类可在同一个 kernel 中累积的项：

$$
O_i =
\frac{
N_i^{exact} + N_i^{0th} + N_i^{1st}
}{
D_i^{exact} + D_i^{0th}
}
$$

实现上必须同时满足四个条件：

1. **精确项仍按 token 级计算**：Top-K KV block 不能近似，否则质量会退化。
2. **未选块不能直接丢弃**：它们需要以块均值、value sum 和全局一阶矩阵进入分子/分母。
3. **归一化必须一致**：exact 和 approximate 不能分别 softmax 后相加，而要共享 online softmax 的 `m/l/acc`。
4. **一阶修正不能逐 block 读 `D x D` 矩阵**：否则理论项虽然更准，但 kernel 会变成 HBM memory-bound。

开源实现围绕这四点组织成两段：准备阶段生成块统计与路由，主前向 kernel 同时处理 exact、zero-order 和 global first-order。

## 代码结构不是三份文件的罗列

这份仓库可以理解为一个“稀疏 attention backend + 一个模型接入器”：

- `piecewise_sparse_attn_hyd.py` 是论文主方法。它实现 hybrid approximation 的 inference forward：块统计、top-k 路由、三阶段 Triton 前向。
- `piecewise_sparse_attn_0th.py` 是可反传的工程变体。它去掉全局一阶修正，只保留 exact + zero-order，并实现 backward。
- `flux_processor.py` 是 Diffusers FLUX 的 attention processor patch。它负责把模型中的 Q/K/V、RoPE、文本/图像 token 顺序整理成 PISA kernel 需要的 `[B,H,T,D]`。

真正的执行链路是：

```text
FLUX transformer block
  -> FluxAttnProcessor.__call__
    -> Q/K/V projection, QK norm, RoPE, image-token reorder
    -> piecewise_sparse_attention(q, k, v, density, block_size)
      -> fused_chunk_reduce(q, k, v)
      -> block-level score + topk indices
      -> piecewise_sparse_attention_fwd_kernel(...)
```

源码交叉引用集中放在附录 A，正文只在关键位置解释为什么这些代码这样写。

## 张量与中间量：先看数据契约

PISA kernel 假设 Q/K/V 均为 `[B, H, T, D]`，batch 和 head 在 kernel 中合并为 `BH`。序列被切成 `NT=ceil(T/BT)` 个 block，默认 `BT=64`。

| 符号 | 代码张量 | 形状 | 用途 |
|---|---|---:|---|
| `Q,K,V` | `q,k,v` | `[B,H,T,D]` | 原始 token 级输入 |
| `\bar{Q}` | `qc` | `[B,H,NT,D]` | query block 均值，用于路由 |
| `\bar{K}` | `kc` | `[B,H,NT,D]` | key block 均值，用于路由和零阶近似 |
| `\hat{V}` | `vc` | `[B,H,NT,D]` | value block sum，用于零阶分子 |
| `H_j` | `hc` | `[B,H,NT,D,D]` | 每块一阶矩阵，仅准备阶段临时保存 |
| `H` | `h` | `[B,H,D,D]` | 全局一阶矩阵，主 kernel 只读它 |
| `S_i` | `indices` | `[B,H,NT,NS]` | 每个 query block 的 exact KV block id |

这张表是理解实现的关键：PISA 主 kernel 不再需要对未选块读取 token 级 K/V，也不逐块读取 `H_j`。未选块的信息被压缩到 `kc/vc/h` 三个对象里。

## 阶段一：块统计预扫描

`fused_chunk_reduce` 的职责是一次扫描 Q/K/V，写出 `qc/kc/vc/hc`。它的 Triton grid 是：

```python
grid = (ceil(K / BK) * ceil(V / BV), NT, B * H)
```

三个 program id 的含义是：

- `program_id(2) = i_bh`：一个 batch-head；
- `program_id(1) = i_t`：一个 token block；
- `program_id(0) = i_kv`：同时编码 K tile 和 V tile，拆成 `i_k` 与 `i_v`。

这种布局有一个明确目标：**每个 program 只负责一个 block 的一个 `(K tile, V tile)`，所有输出都可以独立写回，不需要跨 CTA 归约**。其中：

```text
b_qc = sum(Q_block) / block_len
b_kc = sum(K_block) / block_len
b_vc = sum(V_block)
b_hc = (K_block - b_kc)^T @ V_block
```

`b_hc` 是 `[BK,BV]` tile。代码没有在 Triton kernel 内完成跨 token block 的全局 `H=sum_j H_j`，而是在 Python 侧 `hc.sum(dim=2)`。这看起来多了一次临时 `hc` 写读，但换来预扫描 kernel 无需原子加或跨 block reduction，逻辑简单且并行度充足。

### 为什么 `hc` 先按 block 存，再 sum

直接在 kernel 里把所有 `H_j` 累到全局 `H` 需要多个 program 同时写同一个 `[D,D]` 矩阵 tile，要么用 atomic，要么做多级 reduction。对 head_dim 64/128 的小矩阵来说，atomic 写回会破坏吞吐；多级 reduction 又会让实现复杂化。当前实现选择“先写 `hc`，再用 PyTorch sum”，是以显存峰值换工程稳定性。

这也解释了 PISA 的一个限制：当 `B*H*NT*D*D` 很大时，`hc` 临时张量会影响显存峰值。论文主推的是 inference acceleration，这个 trade-off 是可接受的；若要生产化到更大视频分辨率，`H` 的流式 reduction 是值得优化的方向。

## 阶段二：路由把误差大的块留给 exact path

路由发生在 Python 入口 `piecewise_sparse_attention`：

```python
score = einsum(qc, kc * scale)
if use_bias:
    score = softmax(score + log(bias + eps), dim=-1)
indices = topk(score, k=top_k, dim=-1).indices
```

这里有两个层次：

1. `qc @ kc` 估计 query block 对 KV block 的平均注意力强度。
2. `use_bias=True` 时，`bias` 来自 `hc` 范数，近似表示这个 block 的一阶结构偏离程度；偏离越大，越不适合被全局一阶近似替代，应该进入 exact path。

这正对应论文的 covariance-aware selection。实现不是完整计算 `||H_j-\bar{H}||`，而是用 `hc` norm 作为更直接的 routing bias。FLUX processor 默认使用 `use_bias=True`，所以图像示例走的是协方差感知路由。

`top_k=max(1,int(density*NT))` 是一个重要的性能约束：所有 query block 的 exact block 数相同，因此主 kernel 的 exact loop 次数固定。很多稀疏 kernel 的负载不均来自每行非零数量不同；PISA 在这一点上更接近 regular computation。

## 阶段三：主前向 kernel 的整体调度

主 kernel grid 是：

```python
grid = (ceil(V / BV), NT, B * H)
```

一个 Triton program 处理一个 `(batch-head, query block, value tile)`：

- `i_v`: 当前 value/head_dim tile；
- `i_t`: 当前 query block；
- `i_bh`: 当前 batch-head。

这意味着同一个 query block 如果 `D > BV`，会被多个 program 沿 value 维并行处理。每个 program 都加载完整 `Q_i [BT,D]`，但只写 `O_i` 的一个 value tile `[BT,BV]`。这和 FlashAttention 的 row-block 思路类似，只是 PISA 的 KV 访问分成 exact 和 approximate 两类。

program 内部的状态有三个：

- `m_i [BT]`: online softmax 的行最大值；
- `l_i [BT]`: exact + zero-order 的归一化分母累积；
- `acc [BT,BV]`: 当前 value tile 的分子累积。

PISA 的关键是三阶段都复用这套状态，而不是分别算一个 sparse attention output 和一个 approximation output。

## Phase 1：Top-K 块精确计算

Exact phase 对每个 selected block 执行：

```text
load selected block id j
load K_j [BT,D]
load V_j [BT,BV]
S = Q_i @ K_j^T
online_softmax_update(m_i, l_i, acc, S, V_j)
```

这里 `K_j/V_j` 是 token 级数据，和 dense attention 没有数学区别。差异只在于只遍历 `NS` 个 block，而不是所有 `NT` 个 block。

GPU 利用率方面，Phase 1 有三个特点：

- `Q_i` 在 program 开始加载一次，后续 exact 和 approximate 阶段复用。
- `NS` 固定，CTA 工作量一致。
- 每轮是 64x64 tile dot，计算密度高，适合 Tensor Core/Triton dot 路径。

因此 exact path 的主要成本来自被选中 K/V block 的非连续访问，而不是低算术强度。

## Phase 2：未选块零阶扫描

Zero-order phase 不再读 token 级 K/V，而是扫描 `kc/vc`：

```text
for KV block group G:
    load K centroid kc_G [GROUP_SIZE,D]
    score_mean = Q_i @ kc_G^T
    mask blocks already selected by exact phase
    load V sum vc_G [GROUP_SIZE,BV]
    acc += exp(score_mean) @ vc_G
    g_l += exp(score_mean) * block_len
```

这里 `g_l` 是 tail mass，即未选块贡献的分母部分。`vc` 已经是块内 `sum(V)`，所以分子零阶项不需要乘 `block_len`；分母需要乘 `block_len`，代码用 `current_lens` 处理最后一个短 block。

这一阶段的设计意图是：**用一次规则小 GEMM 近似一组 KV block 的 token 级贡献**。它仍扫描所有 `NT` 个 centroid，但每个 centroid 只有一个 D 向量和一个 value-sum 向量，而不是 64 个 token。和 keep-or-drop sparse attention 相比，它多做了尾部扫描；和 dense attention 相比，它少做了大量 token 级 K/V dot 与 value gather。

### selected mask 为什么在 group 内做

Phase 2 会扫描所有 block group，其中包含 Phase 1 已经 exact 计算过的 block。代码提前把 `indices` 加载到 `loaded_indices`，在每个 group 内计算：

```text
mask_is_selected = chunk_indices in loaded_indices
valid_mask = chunk_indices < NT and not selected
```

这样 exact 和 approximate 不会重复计入同一个 block。这个 mask 是小规模比较，代价远低于维护一个 `[NT]` 全局 bool map，也避免额外显存读写。

## Phase 3：全局一阶注入

如果严格按块级一阶近似，Phase 3 应该对每个未选 block 读取 `H_j [D,D]` 并计算 `Q_i H_j`。这会非常慢，因为每个 query block 都要流式读取大量 `D x D` 矩阵，算术强度低，HBM 带宽压力大。

PISA 的 hybrid 设计把它改成：

```text
load global H tile [D,BV]
R = Q_i @ H
acc += R * (g_l / T) * scale
```

这一步只读一次全局 `H` 的当前 value tile。`g_l` 来自 Phase 2，是当前 query token 分配到未选块的总概率质量。于是全局一阶修正等价于“把 tail mass 乘到共享的一阶方向上”。

这就是论文里 hybrid approximation 的工程意义：不是为了让公式更漂亮，而是把原本 `O(NT * D^2)` 的矩阵读压成 `O(D^2)`，把 memory-bound 的逐块一阶修正变成可接受的一次 tile GEMM。

## 归一化：为什么不是后处理残差

最后代码执行：

```text
l_i += g_l
acc /= l_i
```

注意 Phase 3 只更新分子 `acc`，分母的一阶项在 Taylor 展开中抵消；Phase 2 的 `g_l` 则是未选块零阶分母。因此 PISA 的输出仍然是一个整体 softmax approximation：

```text
exact denominator + approximate denominator
exact numerator + zero-order numerator + first-order numerator
```

如果把 approximation 作为后处理残差加到 sparse output 上，会破坏 softmax 归一化，也无法解释论文中的误差界。当前实现的 `m_i/l_i/acc` 共享正是为了保持数学形式自洽。

## FLUX 接入：模型侧做了什么

`FluxAttnProcessor` 的作用不是简单调用 kernel，而是让模型输入满足 block 稀疏近似的前提。

执行步骤是：

1. 从 FLUX attention module 取 Q/K/V；如果存在 encoder hidden states，则先拼接文本/条件 token。
2. 做 QK norm 和 RoPE，保证 sparse backend 接收到的是原 attention 会使用的同一组 Q/K。
3. 对 4096 图像 token 做 `(h p1 w p2) -> (h w p1 p2)` 重排，使一个 64-token block 更接近二维局部 patch。
4. 根据 `processors_id >= start_layer_idx` 决定当前层走 PISA 还是 PyTorch SDPA。
5. 输出后再把图像 token 顺序还原，并接回 `to_out`。

这个重排非常重要。PISA 的路由单位是连续 token block；如果连续 block 在空间上不局部，块均值 `kc` 的语义会变混，zero-order approximation 误差更大。FLUX 接入先让 block 在空间上更有意义，再让 kernel 做稀疏近似。

## 0th 版本：为什么保留一个不完全等同论文的方法

`piecewise_sparse_attn_0th.py` 保留了 backward，是为了训练或微调场景。它的前向只有 exact + zero-order，不做全局一阶注入。路由时额外计算：

$$
k\_var = E[||k||^2] - ||E[k]||^2
$$

然后用类似 `mean_logits + log(k_var)` 的分数选择 exact block。这个标量方差 proxy 比 `H_j` 便宜得多，也更适合 backward 里传播。

Backward 被拆成三部分：

- `bwd_dq`: exact block 与 centroid approximation 都贡献到 `dQ`；
- `bwd_approx_dkdv`: 处理近似路径对 `kc/vc` 的梯度；
- `bwd_exact_dkdv`: 处理 exact selected block 的 token 级 `dK/dV`，并把 centroid 梯度分摊回 token。

因此 0th 版本不是论文主实验 kernel 的简化注释，而是一个不同工程目标的实现：牺牲 hybrid 一阶质量，换取训练可用性。

## 性能链路：从建模到结果

PISA 的实验结果可以从实现链路解释：

1. **质量来自“不丢尾部”**：非 Top-K block 仍通过 `kc/vc/h` 进入输出，所以高稀疏率下比 keep-or-drop 稳。
2. **速度来自“token 级只算 Top-K”**：Phase 1 只对 `density * NT` 个 block 读原始 K/V。
3. **短序列仍有效来自规则计算**：Phase 2 虽扫描 centroid，但是连续块统计和小 GEMM，不是碎片化 sparse gather。
4. **hybrid 一阶有效来自带宽规避**：逐块一阶质量高但慢；全局一阶接近其质量，同时只读一次 `H`。

论文中 `PISA-1st` 质量接近 hybrid 但速度只有 0.96x，`PISA-hyd` 恢复到 1.24x 左右，就是这条带宽链路的直接体现。Wan2.1-14B 中 0th 到 hyd 的 PSNR 从 21.68 提到 22.69，而 speedup 只从 1.93x 轻微降到 1.91x，也说明全局一阶的额外成本很小。

## 工程限制与可改进点

- hyd 版本只有 forward，没有 backward。
- `hc` 临时张量显存峰值较高，生产化可考虑在 Triton 内做分层 reduction 得到 `H`。
- 公开仓库没有 Wan2.1/Hunyuan-Video pipeline patch，视频复现需要自行接入对应 DiT attention。
- `use_bias=True` 使用的是 `hc` norm proxy，不是完整暴露论文所有 warmup/layer schedule。
- 最佳性能依赖 Hopper 和较新 Triton；非 Hopper 上 TensorDescriptor/TMA 优势会下降。

## 附录 A：源码交叉引用

- hyd 入口与路由：[piecewise_sparse_attn_hyd.py#L303-L347](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/piecewise_attn/kernels/piecewise_sparse_attn_hyd.py#L303-L347)
- hyd 预扫描 kernel：[piecewise_sparse_attn_hyd.py#L96-L183](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/piecewise_attn/kernels/piecewise_sparse_attn_hyd.py#L96-L183)
- hyd 前向 Triton kernel：[piecewise_sparse_attn_hyd.py#L196-L299](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/piecewise_attn/kernels/piecewise_sparse_attn_hyd.py#L196-L299)
- 0th forward/backward 版本：[piecewise_sparse_attn_0th.py#L106-L263](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/piecewise_attn/kernels/piecewise_sparse_attn_0th.py#L106-L263)
- FLUX attention processor：[flux_processor.py#L37-L169](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/piecewise_attn/models/flux/flux_processor.py#L37-L169)

## 附录 B：主 kernel 伪代码

```text
program(i_bh, i_t, i_v):
    Q = load Q block i_t
    m = -inf
    l = 0
    acc = 0

    for j in topk_indices[i_bh, i_t]:
        K, V = load token-level block j
        S = Q @ K.T
        m, l, acc = online_softmax_update(m, l, acc, S, V)

    g_l = 0
    for group in all KV centroid groups:
        Kc = load kc[group]
        Vc = load vc[group]
        S_mean = Q @ Kc.T
        S_mean = mask_selected_blocks(S_mean)
        m, l, acc = online_softmax_update_zero_order(m, l, acc, S_mean, Vc)
        g_l += tail_denominator_mass(S_mean, block_len)

    H = load global H tile
    acc += (Q @ H) * (g_l / T) * scale
    l += g_l
    store acc / l
```
