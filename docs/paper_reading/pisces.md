---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - GRPO
---

# PISCES: Annotation-free Text-to-Video Post-Training via Optimal Transport-Aligned Rewards

- 论文：https://arxiv.org/abs/2602.01624
- 团队：Microsoft, Stony Brook University

## 概述

PISCES 提出一种无标注的文本到视频后训练方法，核心创新是双重最优传输（OT）对齐 Rewards 模块。现有无标注方法依赖预训练 VLM（如 InternVideo2）的嵌入空间来计算 reward，但文本和视频嵌入存在分布失配，导致 reward 信号不准确。PISCES 通过两层 OT 对齐解决此问题：(1) 分布级 OT 对齐质量 Reward——学习一个神经 OT 映射将文本嵌入传输到真实视频嵌入流形上，然后计算余弦相似度作为视觉质量评分；(2) 离散 token 级 OT 对齐语义 Reward——构建包含语义、时间、空间三项代价的代价矩阵，通过 entropic Sinkhorn 求解 partial OT 传输计划，将其融入 VLM 的 cross-attention 层，增强文本 token 到视频 patch 的精准对应。在 VideoCrafter2（短视频）和 HunyuanVideo（长视频）上验证，PISCES 在 VBench 所有指标上超越现有有标注和无标注方法，且兼容直接反传和 GRPO 两种优化范式。

## 动机

- 有标注方法（VideoReward-DPO、IPO、UnifiedReward）需要大规模人工偏好数据集，成本高且扩展性差
- 无标注方法（T2V-Turbo）使用 VLM 余弦相似度作为 reward，但 VLM 训练目标（逐点匹配、对比学习）未对齐文本与真实视频分布
- 核心瓶颈：VLM 嵌入空间中文本和视频的**分布失配**导致 reward 信号不可靠

## 方法

### 分布级 OT 对齐质量 Reward

模拟人类评价视频整体质量的过程——判断生成视频是否"看起来像真实视频"。

- 将文本嵌入 $Y$ 到真实视频嵌入 $X$ 的对齐建模为 Monge-Kantorovich OT 问题
- 用 Neural OT (NOT) 学习传输映射 $T: Y \to X$：

$$\sup_f \inf_T \int_X f(x) d\nu(x) + \int_Y (c(y, T(y)) - f(T(y))) d\mu(y)$$

- $T_\psi$（传输映射）和 $f_\omega$（势函数）均为 3 层 MLP，在 1 块 A100 上训练 1 天
- OT 映射将文本嵌入投影到真实视频流形上，使 $T^*(y)$ 成为 $x_{\text{real}}$ 的代理
- 质量 Reward = OT 对齐后文本 [CLS] token 与生成视频 [CLS] token 的余弦相似度：

$$R_{\text{OT-quality}} = \frac{T^*(y_{[\text{CLS}]})^T \cdot \hat{x}_{[\text{CLS}]}}{\|T^*(y_{[\text{CLS}]})\| \|\hat{x}_{[\text{CLS}]}\|}$$

### 离散 token 级 OT 对齐语义 Reward

模拟人类判断"关键词是否真正反映在视频中"的过程。

将 partial OT 集成到 InternVideo2 的 cross-attention 层：

**代价矩阵构造**（三项）：
- 语义相似度：$1 - \cos(y_i, \hat{x}_j)$
- 时间约束：$|\tau(y_i) - t_j|$，其中 $\tau(y_i) = \sum_k A_{ik} \cdot t_k$ 是文本 token 在注意力加权下的期望帧索引
- 空间约束：$|\pi(y_i) - s_j|_2$，其中 $\pi(y_i) = \sum_k A_{ik} \cdot s_k$ 是期望空间位置

$$C_{ij} = \text{semantic}(i,j) + \gamma \cdot \text{temporal}(i,j) + \eta \cdot \text{spatial}(i,j)$$

**Partial OT 求解**：用 entropic unbalanced Sinkhorn 算法，传输质量分数 $m=0.9$（允许 10% token 不匹配），得到传输计划 $P^*$。

**注意力融合**：在对数空间将 $P^*$ 与原始 cross-attention $A$ 融合：

$$\tilde{A} \propto \exp\left(\log(A + \epsilon) + \log(P^* + \epsilon)\right)$$

$P^*$ 作为 detached 结构先验，梯度通过 $A$ 流动。

**语义 Reward**：用 OT 增强的注意力聚合视频特征，通过 InternVideo2 的预训练 Video-Text Matching 分类器输出正匹配概率：

$$R_{\text{OT-semantic}} = \text{softmax}(\text{VTM}(\tilde{A} \cdot \hat{x}))_{\text{idx}=1}$$

### 后训练

**直接反传**：与一致性蒸馏结合，$\mathcal{L} = \mathcal{L}_{\text{CD}} - R_{\text{OT-quality}} - R_{\text{OT-semantic}}$

**GRPO**：对每个 prompt 采样一组视频，用双重 OT reward 计算组内标准化 advantage，clipped surrogate objective 更新策略。

## 实验

### 主要结果（VBench）

| 方法 | VideoCrafter2 Total | Quality | Semantic | HunyuanVideo Total | Quality | Semantic |
|------|-------------------|---------|----------|--------------------|---------|----------|
| Vanilla | 80.44 | 82.20 | 73.42 | 83.24 | 85.09 | 75.82 |
| T2V-Turbo-v2 | 81.87 | 83.26 | 76.30 | 84.25 | 85.93 | 77.52 |
| VideoReward-DPO | 80.75 | 82.11 | 75.29 | 83.54 | 85.02 | 77.63 |
| VideoDPO | 81.93 | 83.07 | 77.38 | 84.13 | 85.71 | 77.83 |
| **PISCES** | **82.75** | **84.05** | **77.54** | **85.45** | **86.73** | **80.33** |

PISCES 在所有指标上超越有标注（VideoReward-DPO、VideoDPO）和无标注（T2V-Turbo-v2）方法。HunyuanVideo 上 Semantic 分数提升 +4.51，效果尤为突出。

### 人类评估

在视觉质量、运动质量、语义对齐三个维度上，PISCES 一致优于 HunyuanVideo、T2V-Turbo-v2、VideoReward-DPO（偏好率 57-69%）。

### 消融

- **OT 对齐的关键性**：去掉 OT 后 Semantic Score 从 77.63 降至 75.82，证实分布对齐是核心
- **质量 vs 语义 Reward 互补**：Quality Reward 主导整体一致性和美学，Semantic Reward 主导物体存在和动作正确性，两者梯度近似正交（余弦相似度 ≈ 0）
- **Partial OT 的 mass 参数**：$m=0.9$ 最优（VTM +8.11%），$m=0.5$ 过度丢弃关键 token，$m=1.0$ 引入噪声匹配
- **时空约束的作用**：$\gamma=0.2, \eta=0.2$ 较纯语义 OT 额外提升 1.82%
- **一致性蒸馏防止 reward hacking**：去掉 CD loss 后 Quality 从 86.84 降至 86.51

### 训练效率

- OT 映射训练：24 A100 GPU-hours（vs VideoReward-DPO 的 72 A800 GPU-hours 训练 reward 模型）
- 后训练总成本：8×A100 上约 30 小时（直接反传）/ 78 小时（GRPO）
- 推理无额外开销，GRPO 变体通过一致性蒸馏将去噪步数从 50 降至 16（3x 加速）

## 关键启示

- **VLM 嵌入空间的分布失配是无标注 reward 的核心瓶颈**：OT 通过结构保持的分布对齐将文本嵌入投影到视频流形上，使余弦相似度成为有意义的质量度量
- **离散 token 级 OT 可增强 cross-attention 的语义对应精度**：Partial OT + 时空代价矩阵将 VTM 准确率提升 8.11%，远优于普通 cross-attention
- **无标注方法可以超越有标注方法**：关键在于 reward 信号本身的质量而非标注数据量，OT 对齐后的 VLM 嵌入提供了比人工偏好标注更有效的监督
- **双重 reward 提供正交监督**：分布级质量 reward 和 token 级语义 reward 的梯度近似正交，联合训练不产生冲突
- **一致性蒸馏既是效率工具也是正则化手段**：约束模型不偏离教师分布，同时减少去噪步数
