---
tags:
  - Sparse Attention
  - Video Generation
  - CUDA
  - Triton
  - Flash Attention
---

# SVOO 代码实现：总览

对应论文：[SVOO: Attention Sparsity is Input-Stable](../../paper_reading/svoo.md)

当前分析基于本地仓库 `refs/codes/svoo`。这一页的重点不是重复论文结论，而是把开源实现真正的执行链条拆开：离线 profile 怎样进入在线预算，Q/K co-clustering 怎样避免中间大张量，动态块稀疏 attention 怎样落到 Triton 与 FlashInfer。

## 先建立实现目标

SVOO 的工程目标不是“给 attention 加一个稀疏 mask”。它真正要解决的是三件事：

1. **预算要稳定**：不同 layer/head 的可剪枝程度差异很大，所以预算必须来自离线 profile，而不是统一 ratio。
2. **分块要贴近真实 attention**：Q/K 独立聚类会破坏高注意力对应关系，所以运行时要做双向 co-clustering。
3. **稀疏执行要真省时间**：路由、重排、plan 构造的开销不能高过省下的 attention FLOPs。

因此代码实现自然分成一条四段式链路：

```text
offline profiling
  -> per-step / per-layer / per-head sparsity csv
  -> online attention processor
    -> qk co-clustering + block routing
    -> cluster permutation + variable block sparse attention
```

如果把它看成“几个 Triton kernel”，会漏掉真正的关键：SVOO 的价值在于它把 **预算标定、路由重排、稀疏执行元数据** 串成了一条一致的系统路径。

## 1. 实现结构

建议按下面四层理解，而不是按文件名顺序阅读。

### 1.1 模型接入层：替换 Wan / Hunyuan 的 attention processor

Wan 路径中的 processor 负责：

- 取出 `Q/K/V`、做 QK norm、RoPE 与输出投影；
- 判断当前 layer / diffusion step 是否继续走 dense warm-up；
- 读取离线 profile 得到每个 head 的最小 key-cluster 保留预算；
- 触发 co-clustering、block routing、FlashInfer sparse attention；
- 最后把输出逆置换回原 token 顺序。

这里的主控逻辑并不是“kernel wrapper”，而是一个调度器：它决定何时 dense、何时 sparse、何时复用上一次聚类结果，以及 block budget 如何传给稀疏执行后端。

### 1.2 预算层：离线 profile 生成可查询的稀疏日程

离线脚本用 calibration prompts 跑完整 attention，并把每个 `(step, layer, head)` 在累计 95% 注意力质量时所需的 key-token 比例写成 CSV。

在线阶段不会重新估计这些量，而是把它们当作 **head-aware budget floor** 使用。也就是说，profile 不是一个“建议值”，而是直接约束每个 query cluster 至少保留多少 key clusters。

### 1.3 路由层：co-clustering + dynamic map

这层完成两件事：

- 用 Q-centroid space 更新 K labels，再用 K-centroid space 更新 Q labels；
- 用 centroid-level attention 为每个 query cluster 选出可见 key clusters。

产物不是最终输出，而是三类 metadata：

- `qlabels / klabels`：token 属于哪个 cluster；
- `qcluster_sizes / kcluster_sizes`：每个 cluster 的真实长度；
- `dynamic_map[B,H,Qc,Kc]`：query cluster 能访问哪些 key clusters。

### 1.4 执行层：cluster permutation + variable block sparse attention

这一层的关键不是 mask，而是 **重排后连续的物理布局**。

SVOO 会先按 `qlabels / klabels` 把 Q/K/V 排成 cluster-contiguous layout。这样每个 selected block pair 就变成真实连续的 token span，后端才能把它当作一个小 dense tile 去算。默认路径不是自写 Triton kernel，而是 FlashInfer 的 `VariableBlockSparseAttentionWrapper`，Triton 版本更多是 fallback 与验证实现。

## 2. 离线 profile 如何进入在线预算

SVOO 的离线 profile 不是泛泛地“统计稀疏度”，而是要产出在线阶段可直接使用的 `per-step / per-layer / per-head` 稀疏预算。

离线过程可以概括成：

1. 对 calibration prompts 逐层逐头计算完整 attention。
2. 对每个 query row，找到覆盖阈值 `tau=0.95` 所需的最短 key 前缀。
3. 对同一个 `(step, layer, head)`，跨 prompt 取保守统计量，写入 CSV。

在线接入时，processor 会按当前 diffusion step 和 layer 读取整头的稀疏 ratio，然后把它转成每个 head 的 `min_kc_ratio`。这个量进入 `identify_dynamic_map` 后，不是用于缩放 logits，而是用于约束排序后的 key-cluster 保留长度：

```text
preserve_length = int(min_kc_ratio[h] * kc_num)
```

这意味着：

- 对敏感 head，哪怕 centroid attention 很尖锐，也至少要保留足够多的 key clusters；
- 对冗余 head，可以把可见 key-cluster 数压得更低。

所以 SVOO 的 profile 本质上是 **在线 block routing 的硬预算接口**，而不是一个松散的启发式先验。

## 3. Co-clustering 不是普通 KMeans

SVOO 的 co-clustering 真正重要的地方，不是“用了 KMeans”，而是聚类空间不是 token embedding 空间，而是 **opposite-side centroid profile space**。

对 key 更新时，token `k_n` 的表示不是它自己的 `D` 维向量，而是：

$$
\phi_K(k_n) = k_n C_Q^\top \in \mathbb{R}^{Q_c}
$$

对 query 更新时同理：

$$
\phi_Q(q_n) = q_n C_K^\top \in \mathbb{R}^{K_c}
$$

这相当于说：

- key 是否应该并到同一块，看的是它们对 query centroids 的响应是否相似；
- query 是否应该并到同一块，看的是它们对 key centroids 的响应是否相似。

这比独立在 embedding 空间做 k-means 更贴近最终 attention 结构。

## 4. Triton 低内存 co-clustering 设计

开源实现最值得看的部分是：它没有物化 `[B, N, K]` 级 profile 张量，而是把 profile 构造和最近中心分配拆成两个流式 kernel。

### 4.1 `profile_norm_triton`：先算 profile 行范数，不落大矩阵

目标是计算：

$$
\|\phi(n)\|_2 = \|x_n C^\top\|_2
$$

其中 `x` 是 tokens，`C` 是 opposite-side centroids。

直观做法会生成 `x @ C^T`，即 `[B,N,K]` 大张量，再沿 `K` 求 L2 norm。这在视频 token 很长、centroid 数很多时，HBM 压力非常大。

Triton kernel 的 grid 是：

```text
program_id(0) = token tile over N
program_id(1) = batch-head
```

也就是一个 program 负责一个 `(bh, BLOCK_N)` token tile。它的执行方式是：

1. 一次性把 `x_tile[BLOCK_N, D]` 载入寄存器；
2. 沿 `K` 维分批加载 `BLOCK_K` 个 centroids；
3. 做 `x_tile @ c_tile^T`；
4. 直接把这一小段 profile 的平方和累到 `norms`；
5. 最终只写回 `[B,N]` 的范数。

这个 kernel 的要点有两个：

- **不生成 profile 矩阵**：中间结果留在寄存器 / 临时 tile 内消化；
- **重用 `x_tile`**：同一批 tokens 在遍历全部 centroids 时只加载一次。

从 GPU 视角看，它更像一个 streaming reduction kernel，而不是标准 GEMM。性能关键不是 Tensor Core 峰值，而是让 `BLOCK_N * D` 的 token tile 足够大，以便 amortize 对 centroids 的多轮扫描。

### 4.2 `fused_cocluster_assign_triton`：边构造 profile，边做最近中心搜索

第二个关键 kernel 不再显式生成 profile 后再做距离计算，而是直接做 fused nearest-centroid assignment。

目标是给每个 token 找到 profile-space 里最近的聚类中心 `profile_centroids[J,K]`。实现上依然使用：

```text
program_id(0) = token tile over N
program_id(1) = batch-head
```

每个 program 内部的循环逻辑是：

1. 固定一批 tokens `x_tile[BLOCK_N,D]`；
2. 沿 opposite centroids 的维度 `K` 分块；
3. 计算当前 tokens 对这一小段 centroids 的归一化响应；
4. 再与 profile centroids 的一小段 `BLOCK_J` 做点积；
5. 累积得到 token 对候选 profile-centroid 的相似度；
6. 在寄存器里维护 `best_dist / best_idx`。

本质上，这里在做：

$$
\arg\min_j \left\|\frac{\phi(n)}{\|\phi(n)\|_2} - \bar \phi_j\right\|_2^2
=
\arg\max_j \left\langle \frac{\phi(n)}{\|\phi(n)\|_2}, \bar \phi_j \right\rangle
$$

所以 kernel 最终不需要保存完整 `phi(n)`，而只需要逐块累积与候选中心的内积。

这类 fused 设计的真正收益是内存，而不是算力：

- 如果先生成 `[B,N,K]` profile，再和 `[B,J,K]` 做匹配，HBM 会成为主瓶颈；
- 现在只把最终标签 `[B,N]` 写回，HBM traffic 大幅下降。

### 4.3 `triton_centroid_update_sorted_euclid`：先排序，再按 run 聚合

更新 centroids 的常见问题是 atomic 冲突。SVOO 的处理方式很直接：先按 `cluster_id` 排序，再让 kernel 以连续 chunk 扫描。

grid 仍是：

```text
program_id(0) = chunk over sorted tokens
program_id(1) = batch-head
```

一个 program 处理 `BLOCK_N` 个已经按 label 排好序的 tokens。由于同一 cluster 的 token 在排序后往往形成连续 run，kernel 对每个 run 只做一次：

- 局部求和；
- 局部计数；
- 对 `sum_ptr / count_ptr` 做一次 atomic add。

这样原本“每个 token 一次 atomic”的模式，被降成“每个 run 一次 atomic”。如果 cluster 较大、局部连续性好，atomic 冲突会显著减轻。

这也是一个很典型的 GPU 工程选择：先用一次并行 sort 改善数据布局，再让后续 reduction kernel 变得规则。

## 5. Dynamic map 的预算生成逻辑

完成 co-clustering 后，SVOO 会基于 centroid attention 生成 `dynamic_map[B,H,Qc,Kc]`。

先算：

$$
S = \frac{C_Q C_K^\top}{\sqrt{D}}
$$

然后不是直接做 softmax，而是带上 `k_cluster_sizes` 作为权重：

$$
P_{q,k} \propto |K_k| \cdot \exp(S_{q,k})
$$

原因很直接：同样的 centroid score，大 key cluster 代表更多真实 token 质量，不能与小 cluster 一视同仁。

接着对每个 query cluster 按概率降序累计，直到达到保留预算 `p`。再叠加两类约束：

- `min_kc_ratio`：由离线 profile 决定的最低保留比例；
- `max_kc_ratio`：可选的上界，防止极端 head 退回太 dense。

所以 `dynamic_map` 不是“纯注意力 top-p”，而是：

```text
centroid attention ranking
  + cluster-size weighting
  + profile-derived per-head floor
  + optional per-head cap
```

## 6. Triton fallback sparse attention：一个 program 对一个 query cluster

SVOO 自带了 `dynamic_block_sparse_fwd_triton` 作为 fallback。这个 kernel 很适合理解动态块稀疏 attention 的基本执行形态。

grid 是：

```text
program_id(0) = flattened (batch, head, query_cluster)
```

也就是一个 program 负责一个 `(b, h, q_cluster)`。这并不是最激进的并行策略，但它有两个现实好处：

1. **动态块大小好处理**：query cluster 的真实起止位置直接从 `qc_cum_size` 读取。
2. **在线 softmax 状态局部化**：同一个 query cluster 的 `m_i / l_i / acc_o` 完全留在当前 program 内。

program 内部再做两层循环：

- 沿 query cluster 内部按 `BLOCK_M` 切分 query rows；
- 对所有 active key clusters，再按 `BLOCK_N` 切分 key/value rows。

### 6.1 在线 softmax 怎样保持数值一致

这个 kernel 不是对每个 block pair 单独 softmax 再拼接，而是标准 online softmax 累积：

- `m_i`：当前已访问 key chunks 的逐行最大值；
- `l_i`：对应的归一化分母；
- `acc_o`：未归一化输出。

处理一个新的 key chunk 后，用：

$$
m_{new} = \max(m_i, m_{ij})
$$

并把旧的 `l_i / acc_o` 按 `exp(m_i - m_new)` 重缩放，再加上当前 chunk 的贡献。

这样即使 active key clusters 是离散跳跃访问的，最终结果依然与“把所有被保留 token 拼成一个长序列后再做一次 softmax”一致。

### 6.2 这个 kernel 为什么更像 fallback

从 GPU 利用率看，这个实现有明显局限：

- 一个 program 串行遍历当前 query cluster 的所有 active key clusters；
- 不同 query cluster 的活跃 block 数可能不同，负载并不完全均衡；
- CTA 数量取决于 `B * H * Qc`，而不是更细粒度的 `(q_cluster, k_cluster)` 对。

所以它的优势是通用、直接、容易验证；真正追求论文级吞吐时，默认还是走 FlashInfer 后端。

## 7. FlashInfer patch 的关键意义

SVOO 最有工程味道的一部分，不在 attention math，而在它如何把动态块描述转换成 FlashInfer 能高效执行的 paged sparse metadata。

### 7.1 不是直接把 bool mask 丢给 FlashInfer

输入的 block 描述是：

- `block_mask_map[BH, Qc, Kc]`
- `block_row_sz[BH, Qc]`
- `block_col_sz[BH, Kc]`

但 FlashInfer 执行时真正需要的是 CSR-like paged 描述：

- `qo_indptr`
- `kv_indptr`
- `kv_indices`

因此 SVOO 先把 block mask 展开成 token-level key index 列表。

### 7.2 `_fill_variable_block_kv_indices_kernel` 的 grid 与负载形态

展开 kernel 的 grid 是：

```text
program_id(0) = segment tile
program_id(1) = token-offset tile
```

一个 program 同时处理一小批 selected segments 和一段 token offsets。它做的事情很直接：

1. 读取每个 segment 的 `base / start / length`；
2. 对合法的 token offset 生成连续的 `base + offset`；
3. 写入 `kv_indices`。

这里的设计重点是把“每个 active key cluster 对应一段连续 token span”这个结构利用起来。因为 cluster permutation 已经保证每个 cluster 在物理上连续，所以展开索引不需要复杂 gather，只是写连续区间。

### 7.3 `_memory_efficient_plan` 为什么要 patch

SVOO 没有完全依赖原始 `VariableBlockSparseAttentionWrapper.plan`，而是注入了自己的 `_memory_efficient_plan`。原因是：

- 原始接口更偏通用；
- SVOO 的 block sizes 是运行时动态的，而且 head 之间可以不同；
- 如果 metadata 展开和 plan 生成不够紧凑，planning 开销会吃掉 sparse attention 的收益。

patch 后的 plan 路径会：

1. 构造 query 侧 `qo_indptr`；
2. 用自定义 kernel 生成 `kv_indptr / kv_indices`；
3. 准备 FlashInfer 需要的 workspace 与 backend module；
4. 调用 batch prefill plan，生成后续 run 的调度信息。

这条路径的意义是：SVOO 不是只借用 FlashInfer 的 compute kernel，而是连 **稀疏元数据展开与计划生成** 都针对自己的动态块结构做了优化。

## 8. 为什么 clustering reuse 很重要

从 FLOPs 角度看，稀疏 attention 省了很多计算；但从系统角度看，如果每个 diffusion step 都完整 co-clustering，实际端到端速度未必会更快。

SVOO 的处理有两层：

### 8.1 warm-up

早期 diffusion steps 与前几层继续使用 dense attention。理由不是实现偷懒，而是这部分：

- attention 结构更不稳定；
- co-clustering 结果可复用性更差；
- 总体在端到端时长中的占比没那么大。

### 8.2 reuse

进入稳定区间后，不是每一步都重聚类，而是缓存：

- `qlabels / qcentroids / qcluster_sizes`
- `klabels / kcentroids / kcluster_sizes`

只有当 `current_step` 落到 `reuse_interval` 指定的位置时才重算，否则直接复用。

这相当于把 co-clustering 视为一种可摊销的路由预处理。论文之所以能拿到端到端加速，不只是 sparse kernel 快，而是路由成本被这种 reuse 机制压住了。

## 9. 开源实现还暴露了一个更有意思的方向：EAR 补偿

SVOO 代码里集成了 `dynamic_block_sparse_prune_fwd_flashinfer` 这条更激进的路径。它不是简单丢掉未选 block，而是：

1. 对 selected token blocks 用 FlashInfer 做精确 sparse attention；
2. 对 pruned key clusters，用 centroid 级 K/V 计算补偿项；
3. 仍然用统一的 online softmax 状态把两部分融合。

这说明作者实际上也意识到：单纯 keep-or-drop 的块稀疏策略，在高 sparsity 下会很快进入 recall 瓶颈。SVOO 的主线论文先停在 co-clustering + block sparse 上，但开源代码已经把“精确块 + centroid 补偿”的方向铺出来了。

## 10. 关键实现结论

SVOO 这份代码最值得记住的，不是某一个 Triton kernel，而是三条系统级设计原则：

1. **profile 要变成预算接口**：离线统计只有进入在线 per-head block budget 才有工程意义。
2. **聚类要围绕 attention 关系，而不是 token embedding 本身**：opposite-side profile space 是这篇方法的真正核心。
3. **稀疏执行必须连同 metadata 与重排一起设计**：cluster permutation、variable block sizes、FlashInfer plan patch、reuse scheduling 缺一不可。

如果只复现其中一部分，比如只做 co-clustering 或只做 block sparse kernel，通常拿不到论文里的端到端收益。SVOO 的强点就在于它把这几部分真正接上了。
