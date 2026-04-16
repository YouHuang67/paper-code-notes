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

仓库地址：[Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen)。代码同时包含 SVG（v1）和 SVG2（SAP）两套方案，支持 HunyuanVideo、CogVideoX、Wan、Cosmos 四个模型。以 HunyuanVideo 为主线分析。

### 整体架构

代码通过 **monkey-patch** 替换原模型的注意力处理器：

1. `replace_hyvideo_flashattention(pipe)`：先将原始 FSDP+mask 注意力替换为 FlashAttention varlen 实现
2. `replace_hyvideo_attention(pipe, pattern="SAP")`：再替换为稀疏注意力处理器，通过 `pattern` 参数选择 SVG 或 SAP
3. `replace_sparse_forward()`：替换 Transformer block 的 forward 方法，注入 timestep 参数传递

每个注意力层的处理器是独立实例，但**稀疏参数通过类变量全局共享**（如 `AttnModule.num_q_centroids = 50`）。

### SVG（v1）实现：在线 MSE 采样 + FlexAttention

**源码位置**: [attention.py Hunyuan_SVGAttn_Processor2_0](https://github.com/svg-project/Sparse-VideoGen/blob/main/svg/models/hyvideo/attention.py)

核心流程（`attention_core_logic`）：

- **Warmup 判断**：前 `first_layers_fp` 层或前 `first_times_fp` 步 → 全注意力（FlashAttention varlen）
- **在线模式选择**：`sample_mse()` 随机采样 32 行 Q，分别用 spatial mask 和 temporal mask 做 masked attention，与全注意力结果算 MSE，逐 (cfg, head) 选最小 → `best_mask_idx`（0=spatial, 1=temporal）
- **Head Placement**：根据 `best_mask_idx`，temporal head 做 frame-major → token-major 重排（Triton kernel），spatial head 直接复制。重排后两种模式都变成对角线结构
- **稀疏注意力**：统一调用 `torch.compile(flex_attention)` + 预编译的 `block_mask`
- **逆重排**：temporal head 输出从 token-major 重排回 frame-major

`block_mask` 由 `generate_temporal_head_mask_mod` 生成，本质是 tri-diagonal 模式：`|q_idx - kv_idx| < 2 * frame_size`，加上 text token 的全连接。

### SAP（v2）实现：KMeans 聚类 + 动态 Block Sparse

**源码位置**: [attention.py Hunyuan_SAPAttn_Processor2_0](https://github.com/svg-project/Sparse-VideoGen/blob/main/svg/models/hyvideo/attention.py)

核心流程（`attention_core_logic`）：

**1. Warmup 阶段**：同 SVG，但可选 `zero_step_kmeans_init` 在全注意力步骤中预初始化 KMeans 质心

**2. KMeans 聚类**（`kmeans_clustering`）：

- 仅对 **video token**（排除 text token）做聚类
- Q 和 K 分别独立聚类，默认 Q 50 簇、K 200 簇
- 初始化：第一次用随机初始化 + `kmeans_iter_init` 轮迭代；后续步骤复用上一步质心（centroid cache）+ `kmeans_iter_step` 轮迭代
- KMeans 迭代的两个核心操作均用 **Triton kernel 加速**：
  - Assignment：`euclid_assign_triton` — 分块计算欧氏距离，BLOCK_N × BLOCK_K 的 tile 遍历质心
  - Centroid update：`triton_centroid_update_sorted_euclid` — 先按 cluster ID 排序，chunk kernel 利用排序后的连续性，每个 run 只做一次 atomic add（而非每 token 一次）

**3. Dynamic Map 生成**（`identify_dynamic_map`）：

- 计算 Q-centroid 与 K-centroid 的注意力分数：$S_{ij} = Q_c \cdot K_c^\top / \sqrt{d}$
- 加权 softmax：$P'_{ij} = |K_j| \cdot \exp(S_{ij}) / \sum_k |K_k| \cdot \exp(S_{ik})$
- Top-p 截断：按 $P'$ 降序累积，达到 `top_p_kmeans` 阈值后停止
- 可选 `min_kc_ratio` 保留最少比例的 K 簇
- 输出：`[B, H, qc_num, kc_num]` 的 boolean mask

**4. 后处理**（`dynamic_map_post_processing`）：

HunyuanVideo 的 context token 分为 prompt（有效）和 unprompt（padding）两部分：
- 将 prompt 和 unprompt 作为额外的两个"簇"追加到 dynamic map
- prompt 簇与所有 video 簇互相可见，unprompt 簇仅自注意

**5. Token 重排 + Block Sparse Attention**：

- `permute_tensor_by_labels_triton`：按 cluster label 排序 Q/K/V，使同簇 token 物理连续
- 调用 FlashInfer 的 `VariableBlockSparseAttentionWrapper`：
  - `plan()`：根据 dynamic map 和各簇大小规划稀疏计算
  - `run()`：执行变长 block sparse attention
- `apply_inverse_permutation_triton`：将输出还原回原始 token 顺序

### Kernel 实现总结

**自写 Triton kernel**（辅助性操作）：

| Kernel | 文件 | 功能 |
|---|---|---|
| `hunyuan_sparse_head_placement_kernel` | [placement.py](https://github.com/svg-project/Sparse-VideoGen/blob/main/svg/models/hyvideo/placement.py) | SVG 的 token 重排（frame↔token major） |
| `hunyuan_hidden_states_placement_kernel` | [placement.py](https://github.com/svg-project/Sparse-VideoGen/blob/main/svg/models/hyvideo/placement.py) | SVG 的输出逆重排 |
| `_euclid_assign_kernel` | [kmeans_utils.py](https://github.com/svg-project/Sparse-VideoGen/blob/main/svg/kmeans_utils.py) | KMeans nearest-centroid assignment（autotuned） |
| `_centroid_update_kernel` | [kmeans_utils.py](https://github.com/svg-project/Sparse-VideoGen/blob/main/svg/kmeans_utils.py) | KMeans centroid update（per-token atomic） |
| `_centroid_update_chunk_kernel` | [kmeans_utils.py](https://github.com/svg-project/Sparse-VideoGen/blob/main/svg/kmeans_utils.py) | KMeans centroid update（sorted chunk，减少 atomic） |
| `permute_tensor_by_labels_triton` | [permute.py](https://github.com/svg-project/Sparse-VideoGen/blob/main/svg/kernels/triton/permute.py) | 通用 token 重排 |
| `apply_inverse_permutation_triton` | [permute.py](https://github.com/svg-project/Sparse-VideoGen/blob/main/svg/kernels/triton/permute.py) | 通用逆重排 |

**自写 CUDA kernel**（`_kernels` 模块，可选加速）：

- `rms_norm_forward`：QK normalization 的 RMSNorm
- `apply_qk_rope_inplace_cossin_txtlast`：RoPE 应用（跳过 text token）

**核心注意力计算完全依赖第三方库**：

- SVG 模式 → `torch.compile(flex_attention)` + `block_mask`
- SAP 模式 → FlashInfer `VariableBlockSparseAttentionWrapper`
- Full attention → `flash_attn_varlen_func`

论文中提到的**动态块大小 FA2/FA3 kernel**（sparse loading + dense wgmma）在开源仓库中**未包含**，SAP 模式实际使用 FlashInfer 的通用变长 block sparse 实现。这意味着论文中报告的 85%+ 理论性能和 1.48-1.88× 计算浪费减少的数据来自未开源的定制 kernel。
