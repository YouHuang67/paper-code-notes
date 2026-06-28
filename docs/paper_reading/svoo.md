---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
  - Triton
---

# SVOO: Attention Sparsity is Input-Stable

- 论文：https://arxiv.org/abs/2603.18636
- 代码：https://github.com/Mutual-Luo/SVOO
- 团队：Beihang University, Zhongguancun Academy, Peking University, Chinese Academy of Sciences, USTC, Tsinghua University

## 概述

SVOO 解决的是现有 video DiT training-free sparse attention 的两个偏差：

- **层间异质性被忽略**：不同 transformer layer / head 对注意力剪枝的容忍度差别很大，但许多方法用统一 sparsity ratio。
- **Q-K 耦合被忽略**：block sparse attention 先把 Q/K 分块再选 block pair，如果 Q 和 K 独立聚类，可能把真正高注意力的对应关系打散。

论文的核心观察是：每一层的 attention sparsity 在不同输入之间相当稳定，更像 layer/head 的内在属性，而不是 prompt 专属属性。因此 SVOO 采用两阶段：

1. **Offline sparsity profiling**：用少量 calibration prompt 统计每个 `(step, layer, head)` 在覆盖 0.95 attention mass 时需要保留的 key token 比例，得到保守的 layer/head sparsity schedule。
2. **Online QK co-clustering**：推理时对 Q/K 做双向协同聚类，得到语义和注意力偏好更一致的 query/key blocks，再按离线 schedule 选择 top block pairs，并用 dynamic block-size sparse attention 执行。

工程上，SVOO 可以看作 SVG2 的一个更强版本：SVG2 主要解决“语义相似 token 重排成连续块”，SVOO 进一步让 K 的分块依赖 Q centroid、Q 的分块依赖 K centroid，并把离线 per-layer/head 预算接入在线 block 选择。开源实现的主路径是 Wan/Hunyuan attention processor 替换 + Triton co-clustering kernels + FlashInfer `VariableBlockSparseAttentionWrapper` 动态块稀疏执行。

实验上，SVOO 在 7 个 720p video generation 设置上取得更好的质量-速度折中。论文主表中 Wan2.1-1.3B-T2V 从 417s 降到 216s，端到端 **1.93×** 加速，PSNR 29.986；HunyuanVideo-T2V 和 HunyuanVideo-I2V 均达到 2.17×。消融显示：去掉 offline profiling 会降低效率，去掉 online co-clustering 会损伤质量。

## 稀疏思路

### Attention Sparsity 是 Layer/Head 的稳定属性

SVOO 先统计每层 attention density：对每个 query row，把 attention probability 降序排序，找到覆盖阈值 $\tau=0.95$ 累积概率所需的最短前缀长度，再除以 key token 总数。对 layer $\ell$、head $h$、calibration input $x_j$：

$$
d^{(j)}_{\ell,h} = \frac{1}{n}\sum_i \frac{|S^{(j)}_{\ell,h}(i)|}{n}
$$

论文发现：

- 同一模型不同 layer 的 density 差别明显，说明不同层可剪枝程度不同。
- 同一 layer 在不同 prompt 上的 density 曲线接近，说明可以离线估计。

离线阶段对每个 `(layer, head)` 的 calibration densities 拟合高斯分布，并取 0.95 quantile：

$$
\hat d_{\ell,h} = \mu_{\ell,h} + z_{0.95}\sigma_{\ell,h}
$$

它是一个保守估计：宁愿多保留一点 key，也不要把敏感层剪坏。开源仓库进一步把 schedule 存成 `Step,Layer,Head,Sparsity` CSV，例如 `sparsity_profiles/sparsity_wan_1.3B_t2v.csv`，推理时按当前 denoising step、layer、head 查表。

### 双向 QK Co-Clustering

普通 k-means 只看 Q 自己或 K 自己的 embedding 距离。SVOO 的判断是：K token 是否应该在同一个 key block，不只取决于 K 间距离，还取决于它们对不同 Q block 的相对响应是否相似。

算法交替执行两步：

1. **Query-aware key clustering**：把每个 key token 投影到 query centroid 空间，形成 affinity profile $K C_q^\top$，再把 profile 相似的 key 聚到一起。
2. **Key-aware query clustering**：更新 key centroids 后，把每个 query token 投影到 key centroid 空间，形成 $Q C_k^\top$，再把 attention preference 相似的 query 聚到一起。

这样得到的块有两个性质：

- 一个 query block 内的 query 对 key blocks 有相似偏好。
- 一个 key block 内的 keys 对 query blocks 呈现相似 relevance。

这比 SVG2 独立 Q/K k-means 更适合后续 block-pair 选择。论文 Figure 9 中，SVOO 的 co-clustering 在 Wan2.1-1.3B / Wan2.1-14B / Wan2.2-14B 上的 attention recall 都高于 SVG2 k-means。

### Block Pair Selection

完成聚类后，SVOO 用 centroid-level attention 估计每个 `(query cluster, key cluster)` 的重要性：

$$
\bar A = C_q C_k^\top
$$

代码中的 `identify_dynamic_map` 会对 centroid logits 做带 key cluster size 的 weighted softmax。key cluster size 作为权重很重要：一个大 key cluster 即使 centroid score 相同，也代表更多真实 token 的 attention mass。

然后对每个 query cluster 按概率降序累加，保留 top-p block pairs，并用 `min_kc_ratio` 给每个 head 设置最低 key cluster 覆盖量。论文公式中还用离线 schedule 和 recall threshold 平衡选择比例；开源实现里这个 per-step/layer/head 信息通过 CSV 变成 `min_kc_ratio` 或动态 clamp 后的 per-head ratio，直接影响每个 query cluster 至少保留多少 key clusters。

## 代码实现

### 实现结构

SVOO 的开源实现分成四层，而不是一组互不相关的 kernel：

1. **模型接入层**：Wan/Hunyuan attention processor 替换原始 attention，负责取 Q/K/V、QK norm、RoPE、输出投影，并判断哪些 step/layer 继续走 dense warm-up。
2. **预算层**：离线脚本生成 sparsity CSV；推理时 `SparsityLookup` 根据 `(step, layer, head)` 查到 per-head sparsity ratio。
3. **路由层**：`co_cluster_tokens` 对 Q/K 做双向 co-clustering，`identify_dynamic_map` 或 `identify_dynamic_map_estimated` 生成 `[B,H,Qc,Kc]` block mask。
4. **执行层**：先按 cluster label 把 Q/K/V 重排成连续 cluster layout，再用 FlashInfer variable block sparse attention；Triton attention kernel 是 fallback / debug 路径，真正默认加速依赖 FlashInfer 动态块大小 kernel。

在 Wan 路径里，`WanAttn_SAPAttn_Processor.attention_core_logic` 是主控逻辑。它先根据 `first_layers_fp` 和 `first_times_fp` 保留早期层或早期 diffusion steps 的 dense attention；进入稀疏路径后调用 `semantic_aware_permutation`，得到：

- `q_perm/k_perm/v_perm`：按 Q/K cluster label 排列后的连续 token。
- `dyn_map`：每个 query cluster 可见哪些 key clusters。
- `qc_sz_s/kc_sz_s`：每个 cluster 的真实 token 数，支持动态块大小。
- `q_sorted_indices`：用于 attention 后逆置换回原序列。

这条路径的核心是“重排后再稀疏计算”。如果只把 `dyn_map` 当 dense mask 用，计算上不会快；只有让同一 cluster 的 token 在物理内存里连续，FlashInfer 才能把一个可见 block pair 当作真实的 dense tile 来算。

### 离线 Schedule 如何进入在线选择

仓库的 `scripts/offline/generate_sparsity_profiles.sh` 为不同模型生成 canonical CSV，例如 Wan2.1-T2V-1.3B 对应：

```text
sparsity_profiles/sparsity_wan_1.3B_t2v.csv
```

CSV 字段是 `Step,Layer,Head,Sparsity`。`SparsityLookup.get_sparsity_batch(step, layer, num_heads)` 一次返回当前层所有 head 的值。Wan processor 的 `_get_min_kc_ratio_for_heads_from_step` 会：

1. 判断当前 step/layer/head 是否有完整 CSV 数据。
2. 读取每个 head 的 sparsity ratio。
3. 用 `dynamic_min_kc_ratio_min/max` 做上下限裁剪。
4. 返回长度为 `num_heads` 的 list，传给 `identify_dynamic_map`。

`identify_dynamic_map` 里这个 list 被当作 per-head `min_kc_ratio`：对每个 head，排序后的 key clusters 至少保留 `ratio * kc_num` 个。这样离线 profile 不是一个全局开关，而是直接变成每个 head 的 block budget 下界；敏感 head 会被迫多看 key clusters，冗余 head 则可以更激进。

### Co-Clustering 的 Triton 设计

`co_cluster_tokens` 是 SVOO 区别于普通 SVG2 k-means 的关键实现。输入先展平为 `[B*H, S, D]`，每个 batch-head 独立聚类。默认配置里 query clusters 为 256、key clusters 为 1024，论文实验只跑 2 次迭代，并每 20 个 diffusion steps 才重算一次。

一次 co-clustering 迭代包含四个高开销操作，代码都尽量避免 materialize 大矩阵：

**1. Profile norm kernel**

`_profile_norm_kernel` 计算：

$$
\|x_n C^\top\|_2
$$

这里 $C$ 是 opposite-side centroids。直观做法会生成 `[B, N, K]` profile 矩阵，N 是视频 token 数，K 是 centroid 数，HBM 压力很大。SVOO 的 Triton kernel 采用 grid：

$$
(\lceil N / BLOCK\_N \rceil,\ B)
$$

每个 program 处理一个 batch-head 的 `BLOCK_N` 个 token。它把 `x_tile[BLOCK_N,D]` 只加载一次，然后沿 K 维按 `BLOCK_K` 分块加载 centroid tile，做 `x_tile @ kc_tile`，累加平方和，最后只写 `[B,N]` norms。这样 profile 大矩阵不落 HBM。

**2. Fused profile-space assignment**

`_fused_cocluster_assign_kernel` 做的是“在 profile 空间找最近 centroid”。它同样 grid 为：

$$
(\lceil N / BLOCK\_N \rceil,\ B)
$$

每个 program 固定一段 tokens，外层遍历 profile centroids `J`，内层遍历 opposite centroids `K`：

- 加载 `x_tile[BLOCK_N,D]` 一次。
- 计算 `dot_xkc = x_tile @ kc_tile / norm`，得到当前 token 对一小段 opposite centroids 的归一化 affinity。
- 加载 `profile_centroids[BLOCK_K,BLOCK_J]`。
- 累加 `dot_nj += dot_xkc @ pc_tile`。

最终距离写成 `2 - 2 * dot_nj`，等价于归一化 profile 的 cosine 距离。关键点是它没有生成 `[B,N,K]` 或 `[B,N,J]` 中间张量，而是在一个 program 内边算边维护 `best_dist/best_idx`。

**3. Sorted centroid update**

聚类 assignment 后需要按 label 求均值。`triton_centroid_update_sorted_euclid` 先对 cluster ids 排序，再调用 `_centroid_update_chunk_kernel`。kernel 的 grid 是：

$$
(\lceil N / BLOCK\_N \rceil,\ B)
$$

因为 token 已按 cluster id 排序，同一 cluster 在一个 chunk 内通常形成连续 run。kernel 对每个 run 做局部求和，只对 `sum_ptr/count_ptr` 做一次 atomic add，而不是每个 token 做 atomic。这个设计减少了 centroid update 中最容易拖慢的原子写冲突。

**4. Q/K 交替更新**

`co_cluster_tokens` 的循环顺序是：

1. 用当前 Q centroids 为 K 建 profile，更新 K labels 和 K centroids。
2. 用更新后的 K centroids 为 Q 建 profile，更新 Q labels 和 Q centroids。

这就是论文的 bidirectional co-clustering。它不是对 Q 和 K 各跑一次普通 k-means，而是让每一侧的聚类空间由另一侧当前 centroids 决定。

### 动态块稀疏执行

SVOO 有一个自写 Triton sparse attention fallback：`dynamic_block_sparse_fwd_triton`。它的 kernel grid 是：

$$
(B \times H \times Qc)
$$

每个 program 负责一个 `(batch, head, query cluster)`。由于 cluster size 不固定，kernel 先用 `qc_cum_size/kc_cum_size` 读出当前 query cluster 的起止位置，再在 cluster 内按 `BLOCK_M` 切 query，在每个 active key cluster 内按 `BLOCK_N` 切 K/V。

kernel 内部是 FlashAttention 风格 online softmax：

- 对每个 Q chunk 维护 `m_i`，即已处理 key chunks 的 row-wise max。
- 维护 `l_i`，即稳定 softmax denominator。
- 维护 `acc_o`，即未归一化的 weighted V 累积。
- 每处理一个 active K chunk，就用 `m_new = max(m_i, m_ij)` 重缩放旧 accumulator。

这保证 block sparse attention 不需要 materialize attention matrix，也不会因为分块遍历 key clusters 而破坏 softmax 归一化。不过这个 Triton kernel 更像通用 fallback：一个 program 对一个 query cluster 串行遍历 active key clusters，性能不如专门优化过的 FlashInfer dynamic block sparse kernel。

默认路径是 `dynamic_block_sparse_fwd_flashinfer`。它把 `[B,H,Qc,Kc]` 的 bool mask 和可变 block sizes 交给 FlashInfer `VariableBlockSparseAttentionWrapper`。SVOO 还 patch 了 FlashInfer plan：

- `block_mask_map_to_expanded_indices` 先把 block-level mask 展开成 token-level `kv_indptr/kv_indices`。
- `_fill_variable_block_kv_indices_kernel` 的 grid 是：

$$
(\lceil num\_segments / 16 \rceil,\ \lceil max\_length / 128 \rceil)
$$

每个 program 同时处理 16 个 selected block segments 和 128 个 token offsets，把连续 key cluster 展开成 FlashInfer 需要的 page indices。
- `_memory_efficient_plan` 构造 `qo_indptr`、`kv_indptr`、`kv_indices`，并调用 FlashInfer batch prefill module plan。

这里的加速来自两个层面：一是 `dyn_map` 减少了参与 QK/V 的 block pairs；二是 Q/K/V 已被 cluster permutation 排成连续段，FlashInfer 可以对每个 selected variable-size block 做真实 tile 计算，而不是在稠密矩阵里 mask 掉无效元素。

### Clustering Reuse 与 Warm-Up

Co-clustering 本身有成本，尤其是 256/1024 个 clusters 下的 profile assignment。SVOO 通过两个策略摊薄：

- **Dense warm-up**：早期 diffusion steps 和第一层使用 dense attention。Wan 脚本里 `first_times_fp=0.2`，`first_layers_fp=0.03`；Hunyuan 系列使用 10% warm-up。
- **Reuse clustering**：`start_reuse_step` 之后只在 `reuse_interval` 整除时重算 cluster，否则复用缓存的 `qlabels/qcentroids/qcluster_sizes/klabels/kcentroids/kcluster_sizes`。论文设置是每 20 个 diffusion steps 重算；开源脚本按模型设为 20 或 40。

论文 Figure 8 显示 cluster assignment 在 diffusion steps 间 mutual-information similarity 很高，因此 reuse 通常不会明显降质。这个设计很关键：如果每个 layer、每个 step 都重新完整 co-clustering，路由开销会抵消 sparse attention 省下的时间。

### EAR 变体

代码里还有 `use_ear` 路径：`dynamic_block_sparse_prune_fwd_flashinfer`。它不仅保留 selected token blocks，还用 pruned key/value centroids 做补偿。FlashInfer 返回 token-level sparse attention output 和 LSE 后，`_fused_qc_kernel_opt` 再把 centroid attention 合并进去：

- grid 为 `(query_cluster, batch_head)`。
- 对 FlashInfer 输出的 log-sum-exp 先还原成自然 log 空间。
- 对被剪掉的 key clusters 用 K/V centroid 计算近似 attention。
- 用 online softmax 的同一套 `m_i/l_i/acc` 合并 token attention 与 centroid attention。

这条路径说明 SVOO 代码不是简单丢弃低分 block，也预留了“剪枝后用低秩/centroid 项补偿”的实现。不过论文主线和默认开源 SVOO 主要是 co-clustering + block sparse attention。

## 实验结果

论文在 NVIDIA H200、720p 设置下评测 7 个模型：

| 设置 | Dense latency | SVOO latency | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| Wan2.1-1.3B-T2V | 417s | 216s | 1.93× | 29.986 |
| Wan2.1-14B-T2V | 1982s | 1203s | 1.64× | 27.786 |
| Wan2.2-A14B-T2V | 1608s | 984s | 1.63× | 24.846 |
| HunyuanVideo-T2V | 1783s | 821s | 2.17× | 24.879 |
| Wan2.1-14B-I2V | 1658s | 954s | 1.74× | 27.545 |
| Wan2.2-A14B-I2V | 1605s | 994s | 1.61× | 29.678 |
| HunyuanVideo-I2V | 1761s | 810s | 2.17× | 25.155 |

与 SVG2 相比，SVOO 通常同时更快、PSNR 更高。例如 Wan2.1-1.3B-T2V 上 SVG2 是 241s / 1.73× / PSNR 29.268，SVOO 是 216s / 1.93× / PSNR 29.986。差异主要来自两点：离线 schedule 让冗余层/head 更敢剪，co-clustering 让保留下来的 block pairs recall 更高。

消融结果也符合设计预期：

- 去掉 offline profiling，使用固定 recall threshold，质量接近但 latency 变慢，说明 layer/head schedule 提供了更有效预算。
- 去掉 online co-clustering，改成独立聚类，速度接近但质量下降，说明 Q-K 耦合主要贡献 recall 和生成质量。

## 关键启示

- **Sparse ratio 不应该全层统一**：video DiT 的 attention sparsity 有明显 layer/head 异质性，离线 profile 可以把这个信息固化成低成本 schedule。
- **稀疏块的质量取决于分块方式**：block sparse 不是选 top blocks 就完事，Q/K block partition 本身会决定 coarse attention 估计是否可靠。
- **Q/K 需要联合建模**：SVOO 的 co-clustering 本质上是在 opposite-side centroid profile 空间聚类，比 embedding 空间独立 k-means 更贴近 attention 关系。
- **动态稀疏要服务 kernel**：cluster permutation、variable block sizes、FlashInfer plan 和 online softmax 是端到端加速的关键；只在 Python 里生成 mask 不会得到论文级速度。
- **路由开销必须被摊薄**：2 次 co-clustering iteration + 每 20/40 steps reuse，是 SVOO 能端到端加速而不是只省 attention FLOPs 的重要原因。
