---
tags:
  - Sparse Attention
  - LLM Inference
---

# IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse

- 论文：[arXiv 2603.12201](https://arxiv.org/abs/2603.12201)（2026.03）
- 团队：清华大学 + Z.ai（GLM 团队）
- 模型：GLM-4.7-Flash（30B-A3B MoE）、GLM-5（744B-40B）

## 概述

IndexCache 针对 DeepSeek Sparse Attention (DSA) 的推理效率瓶颈——lightning indexer 的跨层冗余——提出跨层索引复用方案。DSA 用轻量 indexer 在每层独立选出 top-k token 供核心注意力计算，将核心注意力从 $O(L^2)$ 降到 $O(Lk)$，但 indexer 本身仍是 $O(L^2)$，N 层累计 $O(NL^2)$，在长上下文下占据主要延迟（200K 时 prefill 阶段 indexer 占 81%）。

核心观察：相邻层的 top-k 索引高度相似（70%-100% 重叠），呈现明显的层间块结构。IndexCache 将 N 层划分为少量 Full 层（保留 indexer）和多数 Shared 层（直接复用最近 Full 层的 top-k），推理时仅需一个条件分支。

两种互补方案：

- **Training-free**：贪心搜索算法，逐步将 F 层转为 S 层，以 LM loss 为代理指标选择最优 pattern。保留 1/4 indexer 即可匹配原始 DSA 性能
- **Training-aware**：多层蒸馏损失，训练每个保留的 indexer 同时服务多层。在此目标下，即使简单的均匀交替 pattern 也能达到 full-indexer 精度

30B 模型上保留 1/4 indexer：prefill 1.82x 加速、decode 1.48x 加速（200K 上下文），9 个基准上质量无显著退化。GLM-5（744B）初步实验：保留 1/2 indexer 实现约 1.2x 端到端加速。

## 背景：DSA 的 Indexer 瓶颈

DSA 将每层注意力分为两阶段：

1. **Lightning indexer**：轻量模块，用多头 ReLU 门控点积对所有前置 token 评分，选出 top-k（k=2048）位置
2. **Core attention**：仅在 top-k 子集上计算 MLA 注意力

Indexer 设计：少量 head、低秩投影、FP8 算术，单次计算量远低于主注意力。但 indexer 仍需在每层独立对全序列评分（$O(L^2)$），N 层累计 $O(NL^2)$。

实测延迟占比（30B DSA 模型）：

| 上下文长度 | Indexer 占 Prefill 延迟 | Indexer 占 Decode 延迟 |
|-----------|----------------------|----------------------|
| 10K       | 27%                  | 27%                  |
| 60K       | 50%                  | 31%                  |
| 120K      | 68%                  | 38%                  |
| 200K      | 81%                  | 41%                  |

关键趋势：随上下文增长，indexer 占比急剧上升（尤其 prefill），而其他计算部分增长温和。这说明 **indexer 是长上下文 DSA 推理的主要瓶颈**。

## 跨层 Top-k 稳定性

论文验证了 DSA indexer 输出的跨层稳定性（附录 A）：

- 对 30B DSA 模型（47 层），在 200K 长度的 768 个样本上计算所有层对的 top-k 重叠率 $|T^{(i)} \cap T^{(j)}| / k$
- **相邻层重叠 70%-100%**，说明大部分 indexer 计算是冗余的
- 热力图呈现块状结构：层 3-5、6-8、17-30、31-36 等形成内部高重叠的功能簇
- 早期层和晚期层之间重叠率低（$\leq 0.4$），关注的 token 集合本质不同

这一观察是跨层索引复用方案的经验基础。

## 方法

### 形式化

- $N$：transformer 层数，$L$：序列长度，$k$：每个 query 选取的 token 数（k=2048）
- 层 $\ell$ 的 indexer 输出分数 $I_t^{(\ell)} \in \mathbb{R}^L$，top-k 索引集 $T_t^{(\ell)} = \text{Top-k}(I_t^{(\ell)})$
- 聚合注意力分布 $p_t^{(\ell)}$（跨 head 平均 softmax 权重），indexer 输出分布 $q_t^{(\ell)} = \text{Softmax}(I_t^{(\ell)})$
- **Pattern string** $c = c_1 c_2 \cdots c_N$，$c_\ell \in \{F, S\}$：
    - F (Full)：保留 indexer，计算新的 $T_t^{(\ell)}$，缓存为 $T_{\text{cache}}$
    - S (Shared)：无 indexer，直接 $T_t^{(\ell)} \leftarrow T_{\text{cache}}$（从最近的 F 层继承）
- 第一层始终为 F

推理修改极其简洁：标准 DSA 循环中加一个条件分支，F 层运行 indexer 并缓存索引，S 层跳过 indexer 直接复用缓存。$T_{\text{cache}}$ 只是一个临时缓冲区，每遇到 F 层就被覆盖，不需要额外 GPU 内存。

### Training-free IndexCache

#### 均匀交替为何不行

最简单的策略是每 r 层保留一个 indexer（如 FSSSFSSS...）。但 indexer 重要性在层间差异很大——早期层和过渡区域的层对 indexer 移除更敏感。均匀交替可能移除关键 indexer 而保留冗余的，导致显著质量下降。

#### 贪心层选择算法

用 LM loss 作为代理指标，在小型校准集上增量搜索最优 pattern：

1. 初始化所有层为 F
2. 每步遍历当前所有 F 层（第一层除外），尝试将每个翻转为 S
3. 选择导致 LM loss 最小增加的层，提交翻转
4. 重复 K 步（K 为目标 S 层数量，如 $K = 3N/4$ 保留 1/4）

校准集：从训练数据缓存 B 个 mini-batch，所有候选 pattern 在相同 batch 上评估，确保 loss 差异仅反映 pattern 变化。

搜索复杂度：完整搜索需 $N(N-1)/2$ 次前向传播。用流水线并行（P 个 stage）可加速约 $P$ 倍。

贪心解的性质：

- 在相同保留率下优于均匀交替
- 每步 LM loss 曲线呈现"容易层"（前 20 步）和"关键层"（35 步后）的清晰分离
- 结果在不同校准集上稳定，说明 indexer 重要性排序是模型固有属性
- LM loss 与下游任务性能正相关

### Training-aware IndexCache：多层蒸馏

当可以训练 DSA 模型时，可以显式训练每个保留的 indexer 同时服务多层。

#### 从单层到多层蒸馏

标准 DSA 训练中，每个 indexer 用 KL 散度对自己层的聚合注意力分布蒸馏：

$$\mathcal{L}_I = \sum_t D_{KL}(p_t^{(\ell)} \| q_t^{(\ell)})$$

多层蒸馏推广为：设层 $\ell$ 是 F 层，$\ell+1, \ldots, \ell+m$ 是后续 S 层，多层蒸馏损失为：

$$\mathcal{L}_I^{\text{multi}} = \sum_{j=0}^{m} \frac{1}{m+1} \sum_t D_{KL}(p_t^{(\ell+j)} \| q_t^{(\ell)})$$

#### 梯度等价性

论文证明 $\mathcal{L}_I^{\text{multi}}$ 与对平均目标分布的蒸馏 $\mathcal{L}_I^{\text{avg}} = \sum_t D_{KL}(\bar{p}_t \| q_t^{(\ell)})$（其中 $\bar{p}_t = \frac{1}{m+1}\sum_{j=0}^{m} p_t^{(\ell+j)}$）具有**完全相同的梯度**：

$$\nabla_\theta \mathcal{L}_I^{\text{multi}} = \nabla_\theta \mathcal{L}_I^{\text{avg}}$$

证明：$q_t^{(\ell)}$ 是唯一依赖参数的项，$p$ 的熵在微分下消失，展开即得。

解释：多层蒸馏不是启发式正则化——它精确等价于让 indexer 向所有服务层注意力分布的质心蒸馏，学习一个共识 top-k 覆盖所有服务层的重要 token。

实践中采用 $\mathcal{L}_I^{\text{multi}}$ 而非 $\mathcal{L}_I^{\text{avg}}$，因为 S 层只需接收当前层的 $q^{(\ell)}$，而 $\mathcal{L}_I^{\text{avg}}$ 还需传递 $p^{(\ell)}$，带来额外内存开销。

#### 训练流程

沿用标准 DSA 两阶段训练：

1. **Warm-up**（1000 步）：用 $\mathcal{L}_I^{\text{multi}}$ 训练 F 层的 indexer，其他参数冻结
2. **Sparse training**（4000 步）：继续用 $\mathcal{L}_I^{\text{multi}}$（仅在 top-k token 上计算 KL）训练 indexer，同时用 LM loss 训练其余参数

## 实验结果

### 端到端推理加速（30B DSA，H100，SGLang，dp_size=8）

| 指标 | 10K | 60K | 120K | 200K |
|------|-----|-----|------|------|
| **Prefill 延迟 (s)** | | | | |
| DSA | 0.57 | 3.38 | 8.57 | 19.5 |
| +IndexCache (1/2) | 0.47 | 2.86 | 6.57 | 13.7 |
| +IndexCache (1/4) | 0.45 | 2.59 | 5.66 | **10.7** |
| **Decode 吞吐 (tok/s/request)** | | | | |
| DSA | 73.5 | 67.0 | 63.0 | 58.0 |
| +IndexCache (1/2) | 84.5 | 80.0 | 77.0 | 73.0 |
| +IndexCache (1/4) | 91.0 | 89.5 | 88.0 | **86.0** |
| **Decode 吞吐 (tok/s, full KV cache)** | | | | |
| DSA | 2700 | 613 | 341 | 197 |
| +IndexCache (1/2) | 3070 | 750 | 431 | 253 |
| +IndexCache (1/4) | 3310 | 840 | 498 | **297** |

关键结论：

- Prefill：上下文越长加速越显著，200K 时 1/4 保留率达 **1.82x**
- Decode per-request：200K 时 58→86 tok/s（**1.48x**）
- Decode full KV：200K 时 197→297 tok/s（**1.51x**）

### Training-free 结果（30B）

| 配置 | Long Avg | G&R Avg | MRCR | GW | LB2 | RULER | LCR | AIME | GPQA | LCB | IFB |
|------|----------|---------|------|----|-----|-------|-----|------|------|-----|-----|
| 原始 DSA | 50.2 | 74.6 | 24.5 | 49.6 | 45.5 | 87.9 | 43.6 | 91.0 | 77.6 | 71.4 | 58.4 |
| 1/2 均匀 | 47.4 | 74.3 | 22.0 | 46.6 | 46.0 | 83.6 | 38.6 | 92.2 | 76.4 | 69.7 | 59.0 |
| 1/2 搜索 | **50.3** | 74.4 | 24.7 | 49.5 | 46.3 | 87.8 | 43.2 | 91.9 | 76.3 | 71.3 | 58.2 |
| 1/4 均匀 | 43.0 | 73.8 | 17.7 | 37.2 | 43.1 | 79.2 | 37.8 | 91.3 | 75.7 | 69.4 | 58.9 |
| 1/4 搜索 | **49.9** | **74.9** | 25.1 | 47.4 | 45.7 | 87.6 | 43.8 | 92.6 | 78.6 | 70.0 | 58.3 |
| 1/8 均匀 | 35.3 | 70.0 | 12.9 | 33.1 | 37.7 | 68.8 | 24.0 | 89.1 | 74.1 | 58.7 | 58.0 |
| 1/8 搜索 | 46.1 | 73.7 | 21.7 | 43.8 | 42.3 | 82.0 | 40.8 | 90.7 | 76.5 | 69.6 | 58.1 |

关键发现：

- **搜索 pattern 远优于均匀交替**：1/4 均匀 Long Avg 掉 7.2 分（50.2→43.0），搜索仅掉 0.3 分（50.2→49.9）
- **"保留哪些层"比"保留多少层"更重要**
- 推理能力完全保留：G&R Avg 在 1/4 搜索下甚至微升（74.6→74.9），AIME 从 91.0→92.6
- 1/8 保留率退化明显（Long Avg 35.3/46.1），是当前方法的极限

### Training-aware 结果（30B）

| 配置 | Long Avg | G&R Avg | MRCR | GW | LB2 | RULER | LCR | AIME | GPQA | LCB | IFB |
|------|----------|---------|------|----|-----|-------|-----|------|------|-----|-----|
| 原始 DSA | 51.0 | 74.2 | 24.7 | 49.1 | 46.9 | 87.3 | 47.0 | 88.8 | 79.4 | 70.5 | 57.9 |
| 1/2 均匀 | **51.6** | 74.5 | 23.8 | 50.2 | 47.2 | 87.0 | 49.8 | 89.3 | 76.7 | 72.2 | 59.9 |
| 1/2 搜索 | 50.6 | 73.6 | 23.9 | 48.1 | 47.1 | 87.5 | 46.6 | 89.6 | 78.6 | 68.5 | 57.7 |
| 1/2 无跨层 loss | 49.8 | 74.5 | 24.6 | 48.3 | 45.0 | 87.1 | 44.0 | 88.8 | 79.4 | 71.7 | 58.0 |
| 1/4 均匀 | 50.6 | 74.1 | 23.7 | 48.1 | 46.9 | 86.1 | 48.4 | 89.3 | 78.0 | 70.5 | 58.7 |

关键发现：

- **Training-aware 消除了 pattern 敏感性**：均匀交替在 training-free 下严重退化，但经过多层蒸馏训练后反而优于搜索 pattern（Long Avg 51.6 vs 50.6）
- 原因：重训后 S 层学会适应继承的索引，保留的 indexer 学会产生跨层通用的 top-k，联合适应消除了层特异性敏感度
- **跨层蒸馏损失有实质贡献**：去掉后 Long Avg 从 51.6 降到 49.8，AA-LCR 从 49.8 降到 44.0

### GLM-5 缩放实验（744B，Training-free）

| 配置 | Long Avg | MRCR | GW | LB2 | RULER | LCR |
|------|----------|------|----|-----|-------|-----|
| 原始 DSA | 78.4 | 71.1 | 92.7 | 64.5 | 97.7 | 66.2 |
| 1/2 均匀 | 78.1 | 72.8 | 90.2 | 65.1 | 97.6 | 64.6 |
| 1/2 搜索 | **78.7** | 72.3 | 90.8 | 66.0 | 97.3 | 67.2 |
| 1/4 均匀 | 72.7 | 65.8 | 74.9 | 62.2 | 96.2 | 64.6 |
| 1/4 搜索 | **78.0** | 70.8 | 90.3 | 63.7 | 97.6 | 67.6 |

趋势与 30B 一致：搜索 pattern 在 1/4 保留率下仅掉 0.4 分，均匀交替掉 5.7 分。

## 负面结果：相似度搜索的失败

论文诚实报告了一个失败的尝试——基于层间注意力输出余弦相似度的 pattern 搜索（附录 C）：

- 方法：构建 $N \times N$ 相似度矩阵 $S_{i,j}$（层 $i$ 用层 $j$ 的索引时 vs 用自己索引时的注意力输出余弦相似度），用 DP 求最优 pattern
- 结果：与均匀交替性能相当，远逊于贪心 loss 搜索
- 原因：逐层相似度是**局部指标**，无法捕捉小扰动在后续层中的级联传播效应。两层注意力输出几乎相同（$S_{i,j} \approx 1$）但遗漏的少量关键 token 可能在后续推理步骤中产生不可忽视的质量退化

这与 Kascade 使用 DP 基于相似度矩阵选择 anchor 层的成功形成对比——区别在于 Kascade 的 anchor 层计算 full attention，而 IndexCache 的 F 层只运行轻量 indexer，误差容忍度更低。

## 与 Kascade 的对比

| 维度 | Kascade | IndexCache |
|------|---------|------------|
| 目标 | 加速 full attention 推理 | 加速 DSA indexer 推理 |
| Oracle | Full attention（anchor 层计算完整注意力） | Lightning indexer（F 层运行轻量 indexer） |
| 复用内容 | Anchor 层的 full attention top-k 索引 | F 层的 indexer top-k 索引 |
| 选层策略 | DP（基于 attention score 相似度矩阵） | 贪心搜索（基于 LM loss）或均匀+训练 |
| 训练需求 | Training-free only | Training-free 或 Training-aware |
| Head 处理 | Head remapping（关键贡献） | 无需（indexer 本身 head 数少） |

核心区别：**Kascade 需要全量注意力作为 oracle，IndexCache 的 oracle 是更轻量的 indexer**——DSA 已经移除了全量注意力，因此 Kascade 的方法无法直接适用。IndexCache 在此基础上进一步压缩 indexer 开销。

## 关键启示

- **跨层稳定性是 transformer 的固有属性**：不仅 full attention 的 top-k 选择跨层稳定（Kascade 等前人工作），DSA 的轻量 indexer 输出同样稳定。这一原则可推广到任何涉及动态 token 选择的稀疏注意力方法（如 MoBA、NSA 的 block 级选择）
- **全局评估优于局部代理**：层间相似度（局部指标）无法替代端到端 LM loss（全局指标）作为 pattern 搜索的代理。小扰动的级联效应使局部相似度具有欺骗性
- **训练可以消除架构敏感性**：training-free 下 pattern 选择至关重要（均匀 vs 搜索差距巨大），但训练后简单均匀 pattern 即可达到最优。这意味着如果有训练预算，无需复杂的 pattern 搜索
- **Indexer 开销是长上下文 DSA 的下一个瓶颈**：随着 sparse attention 成为前沿 LLM 默认配置（DeepSeek-V3.2、GLM-5），跨层索引复用有望成为高效推理的标准组件
