---
tags:
  - Video Generation
  - Reinforcement Learning
  - GRPO
---

# DPP-GRPO: Diverse Video Generation with Determinantal Point Process-Guided Policy Optimization

- 论文：https://arxiv.org/abs/2511.20647
- 代码：https://diverse-video.github.io/
- 团队：Virginia Tech

## 概述

DPP-GRPO 解决视频生成的多样性不足问题——同一 prompt 反复生成相似视频。核心思路：将多样性视频生成建模为集合级策略优化，结合行列式点过程（DPP）的递减收益机制与 GRPO 的组内反馈。关键设计选择是在 prompt 空间而非视频 latent 空间优化：训练一个 prompt 策略模型生成多样化的 prompt 变体，然后送入任意 T2V 生成器。方法是模型无关的（model-agnostic），在 Wan2.1、CogVideoX、Veo3 上均有效，且推理开销仅 +0.67%。

## 动机

- 现有 T2V 模型在同一 prompt 下生成的视频高度相似（mode collapse）
- 图像领域的多样性方法（熵采样、数据覆盖、噪声注入）不适用于视频：
  - 需要昂贵的 test-time 优化
  - 需要完整训练集访问或架构修改
  - 忽略视频特有的多样性维度：运动、镜头运动、场景构图
- 没有先前工作直接针对视频生成的多样性问题

## 方法

### 在 Prompt 空间优化的三个理由

1. **效率**：无需通过视频采样反向传播
2. **电影化表达力**：镜头运动、场景构图天然可通过 prompt 控制
3. **即插即用**：生成的多样化 prompt 可直接喂给开源或闭源模型

### DPP 多样性 Reward

用 L-ensemble DPP 量化集合多样性：

$$\text{Div}(p_{1:k}) = \log \det(L_\phi(p_{1:k}) + I)$$

$L_\phi[p_i, p_j] = \cos(\phi(p_i), \phi(p_j))$ 为归一化余弦相似度核矩阵，$\phi(\cdot)$ 为 Sentence-BERT embedding。log 行列式衡量 embedding 张成的 log-volume——多样性高则体积大。

每个候选 prompt 的边际多样性增益（核心的递减收益机制）：

$$\Delta(p_i | R_q) = \log \det(L_\phi(R_q \cup \{p_i\})) - \log \det(L_\phi(R_q))$$

$R_q$ 为参考集（策展的 ground-truth 多样性变体）。第一个 dolly shot 得高分，重复的 dolly shot 收益递减。

### 相关性约束

$$R_{\text{rel}} = \frac{1}{|R_q|} \sum_{g \in R_q} \cos(\phi(p_i), \phi(q)) \cdot \cos(\phi(p_i), \phi(g))$$

联合约束：生成的 prompt 必须同时与原始 query $q$ 和参考变体 $g$ 保持高相似度。

### 综合 Reward

$$R(p | q, g) = \lambda_{\text{div}} \cdot \Delta(p_i | R_q) + \lambda_{\text{rel}} \cdot R_{\text{rel}}$$

默认 $\lambda_{\text{div}} = \lambda_{\text{rel}} = 0.5$。

### 训练流程

1. **SFT 热启动**：30K prompt-变体对（GPT-5-nano 生成 3K base prompt，每个扩展 10 个变体），50 iterations，LR $2 \times 10^{-5}$
2. **GRPO 训练**：约 1200 iterations，LR $2 \times 10^{-7}$，每个 query 采样 G 个 response，用组合 reward 更新策略
3. **推理**：自回归集合扩展——初始化空参考集，逐步生成 $p_1, p_2, \ldots, p_K$，每步将新 prompt 加入参考集

### 数据集

- 30K 多样性 prompt 数据集（首个视频多样性基准）
- 两阶段构建：GPT-5-nano 生成 base prompt → architect agent 提出变体 → critic agent 用视频级指标（TIE/TCE/CLIP/VideoScore）筛选

## 实验

- **骨干模型**：Wan2.1, CogVideoX（定量），Veo3（定性）
- **策略模型**：Qwen2-7b-Instruct
- **硬件**：4× NVIDIA L40S
- **评估**：200 prompts（VBench），每 prompt 20 个视频 = 4000 视频/方法

### 定量结果

| 模型 | TCE | TIE | VENDI | CLIP |
|------|:---:|:---:|:---:|:---:|
| Wan2.1 baseline | 19.76 | 0.28 | 9.2 | 0.973 |
| DPP-GRPO (Wan2.1) | **31.95** | **0.311** | **11.29** | **0.976** |
| CogVideoX baseline | 22.21 | 0.292 | 8.10 | 0.961 |
| DPP-GRPO (CogVideoX) | **27.59** | **0.310** | **10.30** | **0.964** |

- TCE（语义多样性）提升 62%（Wan2.1），CLIP 对齐反而提升
- VideoScore 同步提升（Wan2.1: 39.7→49.09），多样性不牺牲质量

### 人工评估（5 分 Likert）

| 方法 | Diversity | Alignment |
|------|:---:|:---:|
| CogVideoX | 2.55 | 3.75 |
| Wan2.1 | 3.10 | 3.80 |
| **DPP-GRPO** | **4.07** | **4.28** |

### 推理效率

| 方法 | Overhead |
|------|:---:|
| DPP-GRPO | +0.67% |
| Promptist | +0.60% |
| Prompt-A-Video | +12% |
| GPT-5 | +26% |

### 消融实验

- 仅 relevance：CLIP 高但多样性崩溃（TCE 20.06）
- 仅 DPP：多样性强但 CLIP 下降（0.961）
- Full model：两项均最优（TCE 31.95, CLIP 0.976）
- 最优参考集大小 $|R_q| = 5\text{-}8$

## 关键启示

- **Prompt 空间优化的实用性**：在 prompt 层面操作而非视频 latent，实现模型无关 + 即插即用 + 近零开销的多样性提升
- **DPP 递减收益是天然的多样性建模工具**：log 行列式优雅编码"第一个 dolly shot 有价值，第五个冗余"的直觉，比简单的 pairwise distance 更有效
- **多样性 ≠ 保真度的对立面**：联合优化多样性和相关性反而提升了 CLIP 对齐（0.973→0.976），挑战了常见的多样性-保真度权衡假设
