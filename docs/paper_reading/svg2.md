---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
  - Triton
---

# Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation

[arXiv 2505.18875](https://arxiv.org/abs/2505.18875) | [代码](https://github.com/svg-project/Sparse-VideoGen) | UC Berkeley, MIT, Stanford

## 概述

现有视频生成稀疏注意力方法存在两个核心缺陷：(1) 基于位置的 token 聚类导致关键 token 识别不准确，(2) 关键 token 在 tensor 中分散分布导致 GPU 计算浪费。SVG2 通过**语义感知排列**（semantic-aware permutation）同时解决这两个问题，达到质量-效率的 Pareto 最优。

核心方法：对 Q 和 K 分别做 k-means 聚类，将语义相似的 token 重排为连续布局。这使得 (1) 聚类质心能精确表示簇内 token 的语义，提升关键 token 识别准确率，(2) 关键 token 在物理内存中连续排布，消除计算浪费。配合 centroid-based top-p 动态预算控制和定制化动态块大小 kernel，实现端到端加速。

主要结果：
- HunyuanVideo 720P T2V：**2.30×** 端到端加速，PSNR 30.45
- Wan2.1 720P T2V：**1.89×** 端到端加速（Turbo 模式），PSNR 23.68
- Wan2.1 720P I2V：**1.84×** 端到端加速（Turbo 模式），PSNR 24.51
- 在任意计算预算下均优于 SVG、SpargeAttn、XAttention，位于 Pareto 前沿

## 动机：现有方法与 Oracle 差距

### 注意力的内在稀疏性

Wan2.1-I2V-14B 上的统计：仅 13% 的计算（按 oracle 策略选择）即可达到 95% 的注意力 recall，维持近无损的 PSNR 27。

### 两个失败原因

**识别不准确**：现有方法（SpargeAttn 等）按位置聚类（每 128 个 Q token / 64 个 K token 为一块），用 mean pooling 生成块表示来近似 $P$。但位置相邻的 token 不保证语义相似（如画面中相邻的苹果和蛋糕），块表示质量差，识别准确率低。

**计算浪费**：即使完美识别了关键 token，它们在 tensor 中分散分布。GPU tensor core 优化的是稠密矩阵乘法，分散的关键 token 必须 pad 非关键 token 来维持连续布局，导致大量无效计算。实测中，89% recall 下实际有效计算仅 26.4%。

### SVG2 的解法

语义感知排列后：90% recall，28% 计算预算，86.6% 有效计算率。

## 方法

### Semantic-Aware Permutation with k-means

对每个注意力头和 Transformer 层，独立对 Q 和 K 做 k-means 聚类：
- Q: $N_q$ 个 token → $C_q$ 个簇（默认 100）
- K: $N_k$ 个 token → $C_k$ 个簇（默认 500）

然后按簇将 token 重排为连续布局。数学上可证明排列不改变注意力输出：

$$O' = \pi_q^\top \text{softmax}\left(\frac{(\pi_q Q)(\pi_k K)^\top}{\sqrt{d}}\right) \pi_k V = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V = O$$

其中 $\pi_q, \pi_k$ 为置换矩阵，K 和 V 共享同一置换 $\pi_k$ 保证输出等价。

### Centroid-Based Top-p Selection

**关键度估计**：用簇质心近似注意力分数。预 softmax 分数 $S_{ij} = \text{centroid}(Q_i) \cdot \text{centroid}(K_j)^\top / \sqrt{d_k}$，加权后得到近似注意力分数：

$$P'_{ij} = \frac{|K_j| \exp(S_{ij})}{\sum_k |K_k| \exp(S_{ik})}$$

由于簇内 token 语义一致，质心表示精确，估计可靠。簇数量 < 1024，计算开销 < 全注意力的 1%。

**动态预算**：对所有簇按 $P'$ 降序排列，累积到预定目标即停止选择，实现 per-head 的自适应计算预算。

### System-Algorithm Co-design

#### Fast k-means with Centroid Cache

k-means++ 从头收敛需要 100+ 迭代，耗时可达注意力计算的 50%。但 DiT 连续去噪步之间激活高度相似，因此可以：
- 缓存上一步的质心作为下一步的初始化
- 减少 k-means 运行时间 **76×**

#### 动态块大小稀疏注意力 Kernel

现有高效注意力实现（FlashAttention / FlexAttention / FlashInfer）仅支持静态块大小（如 128×128）。k-means 聚类后簇大小天然不同，如 Q 簇 128 token + K 簇 32 token 的 128×32 计算需要 pad 到 128×128，浪费 75%。

SVG2 实现了支持**动态块大小**的定制注意力 kernel，同时支持 FA2（A100）和 FA3（H100）：

**FA3 kernel 实现细节**：
- 使用 **wgmma（m64n64k16）** 指令执行稠密计算，最大化 H100 硬件效率
- **Q token 加载**：从同一簇加载连续 token，排列后天然连续，无额外开销
- **K/V token 加载**：不同簇大小导致 K/V 在全局内存中可能不连续，使用 **per-token address offset** 实现稀疏加载，加载后在 shared memory 中重排为连续布局
- 核心设计：**sparse loading + dense computation**，避免昂贵的 K/V padding

性能：达到理论最大性能的 **85%+**（上界 = 稀疏密度 × 稠密 FlashAttention-3 运行时间）。

相比 FlashInfer（静态块大小），SVG2 kernel 在实际工作负载下平均减少 1.48× 计算浪费，Cq=100, Ck=500 配置下减少 1.88×。

## 实验

### 主要结果

| Model | Config | PSNR↑ | SSIM↑ | Density↓ | Speedup↑ |
|---|---|---|---|---|---|
| **Wan2.1 I2V** | Full Attn | - | - | 100% | 1× |
| | SpargeAttn | 21.18 | 0.665 | 38.99% | 1.47× |
| | SVG | 24.06 | 0.813 | 30.25% | 1.56× |
| | **SVG2** | **26.56** | **0.861** | **31.28%** | **1.58×** |
| | SVG2-Turbo | 24.51 | 0.812 | 14.13% | 1.84× |
| **Wan2.1 T2V** | Full Attn | - | - | 100% | 1× |
| | SpargeAttn | 20.52 | 0.623 | 42.03% | 1.44× |
| | SVG | 22.99 | 0.785 | 30.25% | 1.58× |
| | **SVG2** | **25.81** | **0.854** | **29.51%** | **1.60×** |
| | SVG2-Turbo | 23.68 | 0.789 | 12.87% | 1.89× |
| **Hunyuan T2V** | Full Attn | - | - | 100% | 1× |
| | SpargeAttn | 27.89 | 0.884 | 42.62% | 1.53× |
| | SVG | 29.16 | 0.905 | 29.86% | 1.91× |
| | **SVG2** | **30.45** | **0.910** | **25.45%** | **2.30×** |

SVG2 在更低密度下实现更高 PSNR 和更大加速。SVG2-Turbo 在与 SVG 相当的 PSNR 下实现 2.5× 更低的密度。

### Kernel 效率

- Centroid cache：k-means 收敛速度提升 76×（从 50+ 迭代降到 1-2 迭代）
- 动态块大小 kernel vs FlashInfer 静态块：计算 FLOPs 减少 1.48-1.88×
- 端到端：支持与 FP8 量化叠加（SVG + FP8 → 2.3×，SVG2 + FP8 → 2.55×）

### 消融

- 语义感知排列在任意密度下均提升 attention recall（vs 不排列）
- 启用排列后计算浪费平均减少 36%（相同关键 token 集合下）
- Pareto 前沿：在 10%-45% 密度范围内，SVG2 的 PSNR 始终高于 SVG 和 SpargeAttn

## 关键启示

- **语义聚类 > 位置聚类**：k-means 按激活值聚类生成的簇质心比位置平均池化的块表示准确得多，这是 SVG2 在低密度下仍保持高质量的根本原因
- **排列的双重收益**：一次 k-means + 排列同时解决识别准确性和计算浪费两个问题，且排列不改变最终输出（数学等价）
- **Centroid cache 利用了 DiT 的步间连续性**：76× 的 k-means 加速使得在线语义聚类从不可行变为可行，这个技巧在其他需要跨步聚类的方法中同样适用
- **动态块大小 kernel 设计**：sparse Q load（连续）+ sparse K/V load（per-token offset → shared memory 重排）+ dense wgmma compute 的组合，是处理非均匀块大小稀疏注意力的高效范式，达到理论性能的 85%+

## 代码实现分析

仓库地址：[Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen)。仓库同时保留 SVG v1 和 SVG2/SAP 两套实现，核心代码结构不是“一个新 attention kernel 替换全部”，而是一个分层流水线：

1. 模型接入层把 Hunyuan/Wan 原始 attention processor 替换为 SVG/SAP processor，并把 timestep、layer id、prompt/video 长度等运行时信息传进 attention。
2. 稀疏策略层决定当前层当前步是跑 full attention、SVG v1 的固定 mask，还是 SVG2 的 semantic-aware permutation。
3. SAP 层先对 Q/K 做独立 k-means，生成 cluster label、centroid、cluster size，再用 centroid attention 生成 block graph。
4. 数据布局层把 Q/K/V 按 cluster label 重排成连续段，让“语义簇”变成后端能执行的 variable block。
5. 后端执行层在开源版本中主要调用 FlashInfer `VariableBlockSparseAttentionWrapper.plan/run`；仓库里的 Triton dynamic block sparse attention 更像机制原型，论文描述的定制 FA2/FA3 dynamic block kernel 没有开源。

这个分层很关键：SVG2 的加速不是单靠“少算一些边”，而是先把语义相关 token 聚到连续内存，再把不规则 token 稀疏问题改写成后端更容易调度的变长 block sparse attention。

### 模型接入：Wan 和 Hunyuan 的两条路径

Wan 路径相对直接：`WanAttn_SAPAttn_Processor.attention_core_logic()` 在非 warmup 阶段对完整序列执行 `semantic_aware_permutation()`，随后调用 `dynamic_block_sparse_fwd_flashinfer()`，最后用 `apply_inverse_permutation_triton()` 把 Q 侧输出还原到原 token 顺序。

Hunyuan 路径更复杂：`Hunyuan_SAPAttn_Processor2_0.attention_core_logic()` 先用 `prepare_video_part()` 只切出 video token 做 SAP，context token 不参与 k-means。`dynamic_map_post_processing()` 再把 prompt 和 unused prompt 作为额外 block 拼到 block graph 后面：

- video cluster 由 k-means 产生，block size 是真实 cluster size。
- prompt block 与 video block 双向可见，保证文本条件仍能参与视频注意力。
- unused prompt block 只和自己相连，避免 padding 语义污染 video token。

因此 Hunyuan 送入 FlashInfer 的并不是“纯视频 cluster 图”，而是一个由 video clusters、prompt block、unused prompt block 共同组成的更大 block sparse graph。后端不理解文本/视频语义，这些约束全部在 wrapper 之前通过 `dyn_map` 和 `qc_sz_s/kc_sz_s` 改写完成。

两条路径都有限制：SAP processor 中显式 `assert cfg == 1`，说明当前开源实现主要面向 batch size 1 的视频生成推理；同时稀疏参数大多是类变量，例如 `num_q_centroids`、`num_k_centroids`、`top_p_kmeans`，需要通过替换函数在模型级统一设置。

### SAP 主流程：从 Q/K 到可执行 block graph

SAP 的核心函数是 `semantic_aware_permutation()`，它把一次 attention 拆成四个状态转换：

1. `query/key: [B, H, S, D] -> [B * H, S, D]`，每个 batch-head 独立做 Q/K k-means。
2. k-means 返回 `qlabels/klabels`、`qcentroids/kcentroids`、`qcluster_sizes/kcluster_sizes`。
3. `identify_dynamic_map()` 用 Q/K centroid 估计 cluster 级注意力，生成 `[B, H, Cq, Ck]` 的 boolean 邻接图。
4. `permute_tensor_by_labels_triton()` 按 label 排序 Q、K、V，使每个 cluster 成为序列中的连续段。

这里的 Q 和 K 是分开聚类的，不是共用一套 token 分组。原因是注意力矩阵行侧和列侧承担的角色不同：Q cluster 决定输出行如何分段，K cluster 决定每个行段要访问哪些列段。V 必须复用 K 的排序索引，否则 `softmax(QK^T)V` 的 K/V 行对应关系会被破坏；输出只需要按 Q 的 `q_sorted_indices` 做逆重排。

这也解释了论文中 permutation 等价性的工程含义：K/V 共享同一置换、Q 输出再逆置换，才不改变 full attention 的数学结果；稀疏化只发生在后续 `dynamic_map` 裁剪的 block 连接上。

### K-means 设计：centroid cache 和两段 Triton 加速

开源实现采用 `batch_kmeans_Euclid()`，输入形状是 `[B * H, S, D]`。每个 batch-head 是独立样本组，因此不同 head 可以形成完全不同的语义簇。首次进入稀疏阶段时，如果没有缓存质心，就随机从 token 中采样初始 centroid 并跑 `kmeans_iter_init` 轮；后续 step 使用上一轮 `self.q_centroids/self.k_centroids` 作为初始化，只跑 `kmeans_iter_step` 轮。这就是论文里的 centroid cache：利用 DiT 去噪相邻 step 激活变化小的性质，把在线 k-means 从高迭代成本压成一两轮局部修正。

Hunyuan 版本用 `q_centroids/k_centroids` 字典按 `layer_idx` 缓存；Wan 版本用单个 `q_centroids/k_centroids` 成员，依赖 processor 实例与 layer 绑定。两者都支持 `zero_step_kmeans_init`：即使当前 warmup 仍跑 full attention，也先对 video token 做 k-means，等切到 sparse 阶段时 centroid cache 已经存在，避免第一步稀疏推理突然承担完整初始化成本。

k-means 每一轮由两个关键操作组成：assignment 和 centroid update。

**Assignment kernel。** `_euclid_assign_kernel` 的 grid 是 `(ceil(N / BLOCK_N), B)`，这里的 `B` 实际是 flatten 后的 `B * H`。一个 Triton program 负责某个 batch-head 的 `BLOCK_N` 个 token：

- 先加载 `x_tile: [BLOCK_N, D]` 和预计算好的 `x_sq: [BLOCK_N]`。
- 沿 centroid 维度以 `BLOCK_K` 分块加载 `c_tile: [D, BLOCK_K]`。
- 用 `tl.dot(x_tile, c_tile)` 得到 cross term，再通过 `||x||^2 + ||c||^2 - 2 x c^T` 计算欧氏距离。
- 每个 token 在所有 centroid chunk 上维护 `best_dist/best_idx`，最后写出 nearest cluster id。

这个设计把 assignment 变成 tile GEMM 风格计算，主要收益来自两点：D 维向量在一个 program 内复用，`BLOCK_N x BLOCK_K` 距离矩阵用 tensor-core 友好的 dot 形式求 cross term；同时 `BLOCK_N/BLOCK_K/num_warps` 走 Triton autotune，适配不同 token 数和 cluster 数。

**Centroid update kernel。** 朴素做法是每个 token 对所属 centroid 做一次 `atomic_add`，视频序列长、head 多时 atomic 冲突会很重。仓库保留了 per-token atomic 的 `_centroid_update_kernel`，但 Euclidean 主路径实际调用 `triton_centroid_update_sorted_euclid()`：

- 先对每个 batch-head 的 `cluster_ids` 排序，得到 `sorted_cluster_ids` 和原 token 索引 `sorted_idx`。
- `_centroid_update_chunk_kernel` 的 grid 是 `(ceil(N / BLOCK_N), B)`，每个 program 处理排序后连续的 `BLOCK_N` 个 token。
- 因为同 cluster id 在排序后形成连续 run，kernel 在 chunk 内按 cluster id 范围遍历，先把一个 run 的 feature sum 在 program 内归约，再对 centroid sum/count 做一次 atomic add。

这样 atomic 粒度从“每 token 一次”降到“每个连续 cluster run 一次”。当 cluster 较大时，atomic 数量接近 cluster 数而不是 token 数，这是 SVG2 能把在线聚类放进推理循环的关键工程点。空 cluster 用旧 centroid 回填，避免下一轮因为缺失 centroid 产生不稳定标签；`cluster_sizes` 则直接成为后续 variable block 的真实行高/列宽。

### Dynamic map：不是固定密度，而是 centroid 级 top-p 邻接图

`identify_dynamic_map()` 不直接看 token 级注意力，而是在 centroid 上做近似：

$$S_{ij} = Q^c_i {K^c_j}^\top / \sqrt{D}$$

随后对 K cluster size 加权 softmax：

$$P'_{ij} = \frac{|K_j| \exp(S_{ij})}{\sum_t |K_t| \exp(S_{it})}$$

这个权重项很重要：一个大 K 簇即使 centroid logit 与小簇相同，对总注意力质量的影响也更大。代码中 `k_cluster_sizes.unsqueeze(-2)` 作为 `weighted_softmax()` 的权重进入分母和分子。

top-p 裁剪是逐 query cluster 做的：对每个 Q cluster，把所有 K cluster 的 $P'_{ij}$ 降序排序，累积概率超过 `top_p_kmeans` 后的 cluster 标为删除；`min_kc_ratio` 则强制保留至少一定比例的 K cluster。最终输出 `dynamic_map: [B, H, Cq, Ck]`。

因此 `top_p_kmeans` 控制的是每个 Q cluster 保留的近似注意力质量，不是直接控制 FLOPs 密度。真实密度由 `density_calculation()` 用 `q_cluster_sizes[:, :, :, None] * k_cluster_sizes[:, :, None, :]` 加权计算：保留一个 `128 x 160` 的 block 和保留一个 `16 x 24` 的 block 成本完全不同。

### Semantic-aware permutation：把离散 token 集合变成连续 block

如果只知道某个 cluster 有 96 个 token，但这些 token 散落在原序列里，后端要么维护复杂 gather/scatter index，要么 pad 回规则块，都会抵消稀疏收益。SVG2 的 permutation 就是把这个问题提前解决：按 cluster label 排序，让同簇 token 在物理序列维度上连续。

`permute_tensor_by_labels_triton()` 的实现很直接但很关键：

- 输入限制为 `[B, H, S, D]` 且 `dim == 2`，先 flatten 成 `[B * H, S, D]`。
- 如果没有传入 `sorted_indices`，先用 `torch.argsort(labels, dim=-1)` 生成每个 batch-head 的 cluster 排序。
- Triton `_permute_kernel` 的 grid 是 `(B * H, ceil(S / BLOCK_S))`，每个 program 负责一个 batch-head 的一段 `BLOCK_S=64` token。
- program 加载这 64 个目标位置对应的原 token index，然后一次性搬运完整 D 维向量到输出。

K 和 V 的处理有一个关键约束：V 调用 `permute_tensor_by_labels_triton(value, klabels, sorted_indices=k_sorted_indices)`，强制复用 K 的排序结果。这样 K/V 的第 t 行仍然对应同一个原 token。输出阶段 `apply_inverse_permutation_triton()` 使用 Q 的 `q_sorted_indices` 做 scatter 式逆重排，grid 同样是 `(B * H, ceil(S / 64))`。

这个 permutation kernel 本身不是稀疏 attention 计算，但它决定了后端是否能把 `q_cluster_sizes/k_cluster_sizes` 当成连续区间长度。换句话说，SVG2 的语义排列同时服务两个目标：让 centroid 更能代表簇内 token，从而选边更准；也让被选中的簇在内存中连续，从而让 block sparse 后端有机会高效执行。

### FlashInfer variable block sparse：开源主路径的实际执行方式

开源 SAP 主路径调用 `dynamic_block_sparse_fwd_flashinfer()`。进入这个函数时，Q/K/V 已经完成 permutation，`dyn_map` 和 cluster sizes 已经生成。函数先检查所有 head 的 block size 之和一致，然后 reshape：

- `q/k/v: [B, H, S, D] -> [B * H, S, D]`
- `block_mask_map: [B, H, Cq, Ck] -> [B * H, Cq, Ck]`
- `block_row_sz: [B, H, Cq] -> [B * H, Cq]`
- `block_col_sz: [B, H, Ck] -> [B * H, Ck]`

之后创建 `flashinfer.sparse.VariableBlockSparseAttentionWrapper(float_workspace_buffer, backend="auto")`，并额外分配 `vector_sparse_indices_buffer`。`plan()` 接收 block graph、row/col block size、head 数、head dim 和 dtype；`run(q, k, v)` 执行真正注意力。

从接口和输入组织可以看出，`plan()` 的职责是把高层描述编译成后端调度元数据：根据 block size 前缀和得到每个 block 的起止 offset，把 dense boolean `block_mask_map` 转成更适合 kernel 消费的稀疏索引结构，并为 flatten 后的每个 batch-head 生成 active block 列表。`run()` 阶段则在这些 active block 上做 dense attention 子任务：稀疏的是 block 连接，block 内仍是 dense matmul、online softmax 和 value accumulation。

一个最小例子：如果某个 head 的 Q block size 是 `[96, 64, 128]`，K block size 是 `[80, 160, 48, 96]`，而 `dynamic_map` 只保留 `(Q0,K0)`、`(Q0,K1)`、`(Q1,K1)`、`(Q2,K1)`、`(Q2,K3)`，那么后端需要执行的是 `96x80`、`96x160`、`64x160`、`128x160`、`128x96` 这五个 dense 子问题，而不是把所有块 pad 成统一 `128x128` 后全算。

### FlashInfer patch：为什么连 plan 也要优化

仓库有一个容易忽略但很重要的 `flashinfer_patch.py`。它不是改数学结果，而是 gated monkey patch FlashInfer 的 `VariableBlockSparseAttentionWrapper.plan()`：只有在 `with flashinfer_patch_enabled():` 中才启用。

原 plan 中有一段逻辑会根据每个 active block 的 `base` 和 `lengths` 展开 `kv_indices`，典型写法是 `torch.repeat_interleave(base, lengths) + offsets_within`，还可能把 `kv_indices_host` 拷到 CPU。Sparse-VideoGen 用源码重写的方式把这段替换为 `_svg_expand_kv_indices()`：

- 如果 `lengths/base` 在 CUDA 上，就启动 Triton `_svg_kvidx_kernel`，grid 是 `(num_blocks,)`。
- 每个 program 负责一个 active block，根据 `base`、`base_off`、`length` 写出该 block 内连续的 kv index。
- `MAX_BLOCK_SIZE2` 取 `next_power_of_2(lengths.max())`，用 mask 处理不同 block 长度。
- 同时把 `kv_indices_host` 替换为 GPU 侧 `kv_indices`，避免不必要的 host copy。

这说明开源实现的性能瓶颈不只在 attention `run()`。对于每步每层都变化的 `dynamic_map`，planning 也在推理热路径上；如果索引展开留在 CPU 或用高开销 PyTorch repeat，会吞掉一部分稀疏节省。这个 patch 是把 variable block sparse 的调度准备也尽量留在 GPU 上。

### 自写 Triton dynamic block sparse：机制原型而非主路径

`kmeans_utils.py` 里还有 `dynamic_block_sparse_fwd_triton()`。它比 FlashInfer wrapper 更直观地展示了 variable block sparse attention 的算法：

- 先对 `qc_size/kc_size` 做 cumsum，得到每个 Q/K block 的起止 offset。
- Triton kernel grid 是 `(B * H * Cq,)`，一个 program 对应一个 batch-head 下的一个 query block。
- program 内再把 query block 按 `BLOCK_M` 分片，把每个激活的 key block 按 `BLOCK_N` 分片。
- 对每个激活 K chunk 计算 `QK^T`，用 online softmax 维护 `m_i/l_i/acc_o`，最后写回当前 query block 输出。

这个实现解释了算法形态，但它不是当前 SAP processor 调用的生产路径。它的局限也比较明显：一个 program 覆盖一个 query block，遇到特别大的 query cluster 时只能在 program 内串行遍历 Q chunk；不同 query block 大小差异也会带来负载不均。实际运行选择 FlashInfer，是为了复用更成熟的 block sparse attention 调度和 kernel 后端。

### 论文 FA2/FA3 kernel 与开源代码的边界

论文中真正强调的 system contribution 是定制 dynamic block-size FA2/FA3 kernel：FA3 版本使用 H100 的 `wgmma(m64n64k16)` 做 dense 计算，Q 因为 permutation 后同簇连续可以连续加载，K/V 则通过 per-token offset 做 sparse load，进入 shared memory 后重排成 dense tile，再用 dense compute 路径完成注意力。这是典型的 **sparse loading + dense computation**：全局内存访问允许不规则，但 tensor core 看到的是规整 tile。

开源仓库没有包含这套论文 FA2/FA3 kernel。仓库中的 CUDA extension `_kernels` 主要暴露 RMSNorm、LayerNorm 和 RoPE 相关算子；SAP attention 主计算走 FlashInfer wrapper。`svg/kernels/ops/attention_ops_wan_dyn_blk.py` 也只是测试 FlashInfer variable block sparse attention 的封装，不是论文自研 FA3 kernel。

所以读实验结果时要分清两层：

- 论文系统结果中的 85%+ 理论上界、相对 FlashInfer static block 1.48-1.88x 计算浪费减少，来自未开源的定制 dynamic block-size FA2/FA3 kernel。
- 开源代码验证的是算法和工程流水线：k-means 语义聚类、Triton permutation、centroid top-p block graph、FlashInfer variable block sparse 执行，以及 plan 阶段 GPU 化 patch。

这不是小差异。算法层面的 sparsity pattern 和 permutation 可以复现，但如果要复现论文的 kernel-level 性能，需要额外实现或获得论文中的 dynamic block-size FA2/FA3 kernel。

### SVG v1 对比：为什么 SVG2 不再依赖固定模式

仓库里的 SVG v1 仍值得对照。Hunyuan SVG processor 先在 warmup 后用 `sample_mse()` 随机采样若干 Q 行，分别评估 spatial mask 和 temporal mask 与 full attention 的 MSE，再按 head 选择更好的 pattern；temporal head 通过 placement kernel 做 frame-major/token-major 重排，让固定 mask 更接近对角结构；最后用 `torch.compile(flex_attention)` 加预编译 `block_mask` 执行。

这条路线的核心假设是视频注意力可以用少数位置模式近似。SVG2 放弃这个假设，改为每层每头在线从 Q/K 激活中学习语义 cluster，并用 centroid top-p 生成非规则 block graph。代价是多了 k-means、permutation 和 planning；收益是选边更贴近当前样本当前 step 的语义结构，也能通过 permutation 把原本分散的关键 token 收拢成可执行的连续 block。

### 加速链条总结

SVG2 的端到端加速来自一条连续的工程链，而不是某个孤立技巧：

- k-means 让 cluster centroid 比位置 pooling 更能代表注意力语义，降低低密度下的错误删边。
- centroid cache 把在线聚类成本从“重新收敛”变成“跨 step 微调”。
- assignment kernel 用 `BLOCK_N x BLOCK_K` tile dot 计算欧氏距离，centroid update 用排序 run 降低 atomic 冲突。
- semantic-aware permutation 把语义簇转成连续内存段，使 cluster size 能直接成为 variable block size。
- weighted top-p dynamic map 在 cluster 级选择 K block，真实 FLOPs 由 Q/K cluster size 加权决定。
- FlashInfer wrapper 负责开源主路径的 variable block sparse execution，patch 则把 plan 阶段的 kv index 展开尽量留在 GPU 上。
- 论文未开源 FA3 kernel 进一步把 K/V sparse load 和 shared-memory 重排接到 dense wgmma compute 上，解决变长 block 直接映射到 tensor core 时的 padding 浪费。

这也是 SVG2 相比 SVG v1、SpargeAttn 等方法的本质优势：它同时优化“选哪些 token/block”和“这些 token/block 在 GPU 上如何被连续、稠密地计算”。
