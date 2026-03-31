---
tags:
  - Sparse Attention
  - LLM Inference
---
# Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference

**论文**: [arXiv 2512.16391](https://arxiv.org/abs/2512.16391) | **代码**: [GitHub](https://github.com/microsoft/kascade) | **团队**: Microsoft Research India

## 概述

Kascade 是微软研究院提出的 training-free 动态稀疏注意力方法，用于加速长上下文 LLM 推理中的 attention 计算。

核心问题：长上下文推理中，attention 是延迟瓶颈。Prefill 阶段 attention 是 $O(n^2)$（MLP 只有 $O(n)$），Decode 阶段 attention 是 $O(n)$（MLP 只有 $O(1)$），且 decode attention 是 memory bandwidth bound，batching 也难以缓解。

核心观察：
- Post-softmax 注意力权重天然稀疏：256 个 token（约 10%）即可覆盖 95% 以上的 attention mass
- Top-k 稀疏模式在相邻层间高度稳定：layer $i$ 的 Top-k 能覆盖 layer $i+1$、$i+2$ 98% 以上的 attention mass

解决方案：将层分为 anchor layer（计算完整 attention 并提取 Top-k 索引）和 reuse layer（复用前一个 anchor layer 的 Top-k 索引做稀疏 attention）。通过动态规划自动选择最优 anchor layer 集合，并引入 head remapping 处理 GQA 下的跨层 head 对齐问题。

主要结果：
- Decode attention 相比 FlashAttention-3 加速达 4.1x，prefill attention 加速达 2.2x（H100）
- 在 LongBench 上精度与 dense attention 接近，在 AIME-24 上以 10% Top-k 显著优于 LessIsMore、OmniKV、Quest 等方法（47.92 vs 36.25/39.58/7.50，DeepSeek-R1-Distill-Llama-8B）
- 无需训练，通过 development set 自动配置 anchor layer 和 head mapping，可快速部署到新模型

---

## 1. Oracle Top-k 可行性分析

Kascade 的出发点是验证 Top-k 稀疏近似的理论上限。

**Softmax 的稀疏化效应**：softmax 指数放大大值、压制小值，使 post-softmax 注意力向量 $p = \text{softmax}(q_t \cdot K^\top / \sqrt{d})$ 天然稀疏。实验证实（Llama-3.1-8B-Instruct, MuSiQue），除 layer 0 外，几乎所有层和头中 top 256 个 token 即覆盖 95%+ 的 attention mass。

**Oracle Top-k**：假设已知每个 token 的 Top-k 注意力权重索引（需要先算完整 softmax），只对这 k 个 token 计算 attention output。实验显示，在 2WikiMultihopQA 上，即使 $k/N = 2.5\%$，Oracle Top-k 的精度就已匹配 dense attention（前提是 layer 0 保持 dense）。

这说明核心挑战不在"能不能稀疏"，而在"如何高效获取 Top-k 索引"。

---

## 2. 跨层相似性与索引复用

**跨层相似性度量**：对 token $q$，定义 layer $a$ 的 Top-k 索引集合 $I_q^a = \text{topk}(P_q^a, k)$，其中 $P_q^a$ 是 layer $a$ 所有 head 的 post-softmax 注意力的平均。层间相似度定义为：

$$\text{sim}(a, b)_q = \frac{\sum_{i=1}^{k} P_q^b[I_q^a[i]]}{\sum_{i=1}^{k} P_q^b[I_q^b[i]]}$$

即用 layer $a$ 的 Top-k 索引去查 layer $b$ 的 attention 分布，看能恢复多少 oracle attention mass。

**实验结论**（Llama-3.1-8B-Instruct, MuSiQue, $k=256$）：
- 相邻层对的相似度普遍 > 0.98
- 相似度随层间距离衰减，但短距离内保持很高
- Layer 16 的 Top-k 能覆盖 layer 17/18 的 99% attention mass

这为 anchor-reuse 架构提供了理论基础：少量 anchor layer 计算精确 Top-k，中间的 reuse layer 直接复用。

---

## 3. Anchor Layer 选择

给定 anchor 数量预算 $M$，目标是选出一组 anchor layers 使得所有 reuse layer 与对应 anchor 的累积相似度最大。

**动态规划算法**（Algorithm 1）：

输入为相似度矩阵 $S \in \mathbb{R}^{L \times L}$ 和预算 $M$。

$$dp[m][j] = \max_{i=m-1}^{j-1} \left( dp[m-1][i] + \sum_{l=i}^{j-1} S[i][l] \right)$$

通过回溯 $\text{path}[M+1][L+1]$ 恢复最优 anchor 集合。

**相似度矩阵的构建细节**：
- 对每个 prompt 内的所有 token 计算 $\text{sim}(a,b)_q$，取 **token 维度的最小值**（而非均值），确保最差情况 token 也满足要求
- 跨 development set 多个 prompt 取平均
- 用 $k=64$ 计算相似度，实验验证跨场景鲁棒

**层重要性加权**：深层 attention 对表征的改变可能很小。定义 layer $l$ 的重要性：

$$w_l = 1 - \text{CosineSim}(x_l, y_l)$$

其中 $x_l, y_l$ 是 attention block 的输入输出。如果 attention 几乎不改变表征（高余弦相似度），说明该层不重要。将重要性权重乘入相似度矩阵：$S[i][j] = w_j \cdot S[i][j]$。

实验显示（Llama-3.1-8B-Instruct, MuSiQue）深层重要性明显下降，layer 0 重要性最高。

**实际选择结果**（development set = MuSiQue, 5 个 anchor）：
- Llama-3.1-8B-Instruct（32层）: [0, 2, 8, 13, 14]
- Qwen3-8B（36层）: [0, 2, 7, 14, 23]
- DeepSeek-R1-Distill-Llama-8B 复用 Llama-3.1 的配置

---

## 4. Query Pooling（Tile 级 Top-k 共享）

现代 attention kernel 以 tile 为单位计算 $QK^\top$：
- Decode 阶段：GQA 中共享同一 KV head 的多个 query head 组成一个 tile
- Prefill 阶段：连续 token 的 query 组成一个 tile（共享前缀 key）

为保持 kernel 效率，tile 内所有 query 必须共享同一组 Top-k 索引，因此需要将 tile 内的 attention 分布 pooling 成一个。

**两种 pooling 策略对比**：
- **Pre-Softmax pooling**：先平均 tile 内 query 向量，再用 pooled query 计算一次 attention → 随 tile 增大精度显著下降
- **Post-Softmax pooling**：每个 query 独立计算 post-softmax 分布，然后跨 tile pooling → tile 增大到 256 精度仍然稳定

Kascade 采用 Post-Softmax pooling：
- Decode：仅跨 GQA 组内 query heads pooling（tile = GQA group size）
- Prefill：跨 128 个 query 的完整 tile pooling（含 GQA grouping），与 FlashAttention 的 tile 大小一致

---

## 5. Head Remapping

GQA 中每个 KV head 对应多个 query head。anchor layer 为每个 KV head 计算一组 Top-k 索引，但 reuse layer 的 head $i$ 不一定与 anchor layer 的 head $i$ 对应（transformer 不保证跨层 head 顺序一致）。

**三种策略对比**：
- 1:1 mapping（anchor head $i$ → reuse head $i$）：效果最差
- 全 head pooling（所有 head 共享一组 Top-k）：中等，但在低 Top-k% 时不稳定
- Head remapping：为每个 reuse layer 的每个 head 找 anchor layer 中最相似的 head，允许多对一映射 → 一致最优且鲁棒

Head mapping 在 development set 上通过 head-level 相似度计算，随 anchor layer 选择一并确定。

实验对比（Llama-3.1-8B-Instruct, MuSiQue, tile=128）：
- Head remapping 在所有 Top-k% 下一致最优
- 全 head pooling 在高 Top-k% 时接近但低 Top-k% 时显著下降
- 无 remapping 的 1:1 mapping 一致最差

---

## 6. Kernel 实现

基于 TileLang（tile-level GPU 编程语言）修改 FlashAttention kernel 实现。

### 6.1 Reuse Layer（大部分层）

直接使用 anchor layer 传入的 Top-k 索引和 head mapping 加载非连续 key。虽然 key 加载不连续，但每个 key 约 256 字节，实测无明显开销（与 block sparse attention 方法的声称相反）。

### 6.2 Anchor Layer（多 pass）

Post-Softmax pooling 要求先独立计算每个 query 的完整 softmax，再跨 tile pooling。由于 softmax 需要完整 row sum，无法单 pass 完成：

- **Pass 1**：计算完整 $QK^\top$ 矩阵和行和向量 $\sum_j QK^\top_{ij}$。Decode 写出两者到 HBM，prefill 只写行和（attention 矩阵太大）
- **Pass 2**：输出 pooled post-softmax 权重。Decode 读 pass 1 结果做 softmax+pool；prefill 需重算 attention 权重（因为只有行和），结合行和计算 post-softmax 并 pool
- **Pass 3**：在 pass 2 输出上计算 Top-k 索引
- **Pass 4**：用 Top-k 索引计算稀疏 attention（与 reuse layer 相同）

Layer 0 特殊处理：执行 dense attention（pass 1 直接算出结果），省略 pass 4。

Prefill 的 pass 2 重算是主要额外开销，限制了 prefill 加速比。

---

## 7. 实验评估

### 7.1 设置

- 模型：Llama-3.1-8B-Instruct、Qwen3-8B、DeepSeek-R1-Distill-Llama-8B
- Benchmark：LongBench（21 个长上下文任务，6 大类）、AIME-24（30 道数学推理题）
- 对比方法：Dense attention、StreamingLLM（30% sliding window + 4 sink tokens）、Quest、OmniKV、LessIsMore
- 默认 Top-k = 10%，最小 128（$k = \min(\max(0.1 \cdot L, 128), L)$）
- Kascade 在 prefill 阶段也使用稀疏 attention（滚动式，每个 tile 仅 attend 前 10% tokens），其他方法仅优化 decode

### 7.2 精度结果

**LongBench**（prefill-heavy）：

| 模型 | 方法 | SQA | MQA | Summ. | Fewshot | Synthetic | Code | Avg. |
|------|------|-----|-----|-------|---------|-----------|------|------|
| Llama-3.1-8B | Dense | 48.43 | 43.18 | 25.99 | 63.22 | 34.83 | 59.89 | 45.92 |
| | LessIsMore (decode-only) | 48.15 | 42.71 | 25.38 | 63.05 | 34.67 | 59.16 | 45.52 |
| | OmniKV (decode-only) | 48.22 | 43.05 | 25.97 | 63.22 | 34.72 | 59.33 | 45.75 |
| | Kascade | 47.41 | 39.84 | 25.21 | 61.32 | 33.67 | 62.70 | 45.02 |
| | Kascade (All Heads Pooled) | 47.83 | 40.50 | 25.34 | 63.09 | 34.50 | 62.95 | 45.70 |
| Qwen3-8B | Dense | 47.56 | 41.35 | 24.15 | 64.32 | 34.83 | 65.90 | 46.35 |
| | Kascade | 44.19 | 40.38 | 23.02 | 60.83 | 35.00 | 63.98 | 44.57 |
| | Kascade (All Heads Pooled) | 44.87 | 42.34 | 23.74 | 61.99 | 34.50 | 62.71 | 45.02 |

LongBench 是 prefill-heavy 任务，Quest/OmniKV/LessIsMore 仅优化 decode 因此精度几乎无损。Kascade 在 prefill 也用稀疏 attention，仍保持接近 dense 的精度。StreamingLLM 精度大幅下降（avg 33-34）。

**AIME-24**（decode-heavy reasoning）：

| 方法 | DeepSeek-R1-Distill-Llama-8B | Qwen3-8B |
|------|------|------|
| Dense | 50.42 (11.3k) | 73.75 (14.4k) |
| StreamingLLM | 0.00 (7.5k) | 0.00 (6.9k) |
| LessIsMore | 36.25 (14.8k) | 60.83 (17.9k) |
| OmniKV | 39.58 (12.5k) | - |
| Quest | 7.50 (22.9k) | 25.33 (28.8k) |
| **Kascade** | **47.92** (14.6k) | **70.42** (15.9k) |
| Kascade (All Heads Pooled) | 41.25 (14.0k) | 65.83 (17.9k) |

括号内为平均 decode 长度。AIME-24 是复杂推理任务，attention pattern 复杂，StreamingLLM 完全失败。Kascade 在两个模型上均大幅领先其他方法：DeepSeek-R1 上比 LessIsMore 高 11.67 点，比 OmniKV 高 8.34 点；Qwen3 上比 LessIsMore 高 9.59 点。

Top-k 增大到 20% 时，Kascade 在 DeepSeek-R1 上达 49.2（baseline 50.42），decode 长度降至约 13% 高于 baseline。

### 7.3 效率结果

硬件：单卡 H100，fp16。模型配置：32 heads, 8 KV heads, head dim 128（Llama-3.1-8B 设置）。

**Decode attention**（batch=64, Top-k=10%）：

| 序列长度 | FA3 (ms) | Kascade (ms) | 加速比 |
|----------|----------|-------------|--------|
| 8k | 0.70 | 0.24 | 2.91x |
| 32k | 2.93 | 0.74 | 3.97x |
| 64k | 5.85 | 1.43 | 4.08x |
| 128k | 11.68 | 2.83 | 4.12x |
| 256k | 21.77 | 5.34 | 4.08x |
| 512k | 21.85 | 5.33 | 4.10x |

长序列下 reuse layer 仅需 dense 的 ~11% 时间。加速比在 32k 后趋于稳定 ~4.1x。

**Prefill attention**（batch=1, Top-k=10%）：

| 序列长度 | FA3 (ms) | TileLang (ms) | Kascade (ms) | 加速比 vs FA3 | 加速比 vs TL |
|----------|----------|--------------|-------------|--------------|-------------|
| 8k | 0.76 | 1.00 | 0.62 | 1.23x | 1.62x |
| 32k | 12.28 | 17.13 | 6.42 | 1.91x | 2.67x |
| 64k | 53.77 | 64.65 | 24.63 | 2.18x | 2.62x |
| 128k | 215.76 | 262.21 | 98.55 | 2.19x | 2.66x |
| 256k | 864.02 | 1048.01 | 408.30 | 2.12x | 2.57x |

Prefill 加速比低于 decode，主要因为 anchor layer 的 pass 2 需要重算 attention 权重。TileLang baseline 本身比 FA3 慢 ~20%，因此 vs TileLang 的加速比更高。

**时间分解**（128k, Llama-3.1-8B 设置）：
- Anchor layer 0 最耗时（dense attention + Top-k 计算）
- 其他 anchor layer：pass 1 + pass 2（重算）+ Top-k + reuse
- Reuse layer：仅约 dense 的 11% 时间
- 加权平均按 1/32（layer 0）+ 4/32（其他 anchor）+ 27/32（reuse）计算

---

## 8. 局限性

- 需要 development set 计算 anchor layers 和 head mappings，可能引入对该数据集的偏差（实验中发现跨数据集鲁棒）
- 不减少 KV cache 的内存容量需求，长序列仍受显存限制
- 对已预训练内置稀疏 attention 的架构（如 Gemma），收益有限

---

## 关键启示

- **跨层 Top-k 复用是高效稀疏 attention 的关键 enabler**：直接算 Top-k 需要完整 softmax，成本与 dense 相当；复用相邻层的 Top-k 索引将这个成本分摊到少数 anchor layer
- **Head-aware 设计不可省略**：全 head 共享 Top-k 损失显著，head remapping（允许多对一映射）是低 Top-k 比例下维持精度的关键
- **Tile-level 设计约束需要前置考虑**：稀疏方法的精度优势如果与 GPU kernel 的 tile 结构冲突（如要求每个 token 独立 Top-k），将无法转化为实际加速。Post-Softmax pooling 在精度和效率间取得了好的平衡
- **DP 自动选层使部署可行**：手动选择 anchor layer 难以泛化到新模型，动态规划 + 层重要性加权使得配置自动化，降低部署门槛
