---
tags:
  - Video Generation
  - Reinforcement Learning
  - Reward Model
---

# VR-Thinker: Boosting Video Reward Models through Thinking-with-Image Reasoning

- 论文：https://arxiv.org/abs/2510.10518
- 代码：https://github.com/vr-thinker/vrthinker
- 团队：CUHK MMLab, Kling Team (Kuaishou), Nanjing University

## 概述

VR-Thinker 提出 Thinking-with-Image 框架，将视频 reward 模型从被动评估器升级为主动视觉推理器。现有 VLM 基础的 reward 模型有两个固有限制：(1) 视觉输入占用大量 context，迫使降采样丢失细节；(2) 所有视觉信息打包在初始 prompt 中，后续 CoT 推理纯文本进行，加剧遗忘和幻觉。VR-Thinker 赋予 reward 模型视觉推理操作（如帧选取）和可配置的视觉记忆窗口，使其能在推理过程中主动检索和更新视觉证据。训练分三阶段：Cold Start（CoT 格式初始化）→ Rejection Fine-Tuning（筛选高质量推理轨迹 SFT）→ GRPO（强化推理能力）。基于 Qwen2.5-VL-7B，在 VideoGen-RewardBench 上达到 80.5%，GenAI-Bench 上 82.3%，MJ-Bench-Video 上 75.6%，平均超越其他开源模型 11.4%。

## 动机

VLM 基础的视频 reward 模型存在两个根本问题：

1. **视觉上下文消耗大**：单帧约 500 token，8 帧约 4000 token（10 倍于文本部分），迫使降采样导致关键帧信息丢失
2. **推理过程无法回溯视觉证据**：所有视觉信息在初始输入时一次性注入，CoT 推理过程中纯文本进行，无法重新检查或获取新的视觉证据，加剧遗忘和幻觉

## 方法

### Thinking-with-Image 框架

核心设计：将未采样的视频帧保留为可操作工作空间，模型可在推理过程中主动检索。

**工具调用**：初始输入降采样帧后，模型进行多轮推理。每轮可通过 `select_frames` 工具从完整视频中检索特定帧（如分析指法动作时检索中间帧），获取新视觉证据后更新推理。

**窗口记忆**：工具返回的视觉内容仅保留 $p$ 轮（窗口宽度），超出后删除。总 token 数：

$$T_{\text{total}} \approx (N_{\text{in}} + p \cdot N_{\text{ex}}) \cdot V_t$$

其中 $N_{\text{in}}$ 是初始帧数，$N_{\text{ex}}$ 是每次检索帧数，$V_t$ 是每帧 token 数。关键性质：$T_{\text{total}}$ 与推理步数 $t$ 近似无关，保持 context 预算稳定。

**推理格式**：
- `<snapshot>`：每轮将视觉证据压缩为语言摘要，缓解窗口记忆下的信息丢失
- `<think>`：推理内容
- `<recommend answer>`：非最终轮的临时答案 + 置信度
- `<answer>`：最终判断
- `<tool_call>`：帧检索调用

### 三阶段训练

**Stage 1: Cold Start**

用 GPT-4o 生成高质量多模态 CoT 数据（1.2k 样本），两阶段过滤：(1) 推理格式完全符合规范；(2) 所有维度判断和整体偏好与 ground truth 完全一致。标准 SFT 损失训练，工具执行输出的 token 从 loss 中 mask。

**Stage 2: Rejection Sampling Fine-Tuning**

混合多个偏好数据集，用 Stage 1 模型采样多条 CoT 轨迹，保留所有判断均正确的样本进行 SFT。此阶段大幅提高高质量推理轨迹的比例，为 RL 奠定基础。

**Stage 3: GRPO 强化**

四种 reward 信号：

- **Format Reward**：推理结构正确性（标签、答案格式）
- **Accuracy Reward**：$r_{\text{acc}} = \alpha \cdot r_{\text{acc\_all}} + \bar{\alpha} \cdot r_{\text{acc\_dim}}$，同时评估整体偏好和各维度判断。答案空间 $3^{d+1}$（$d$ 个维度），远大于传统仅 3 选项，减少偶然正确但推理错误的样本
- **CoT Gain Reward**：$r_{\text{cot}} = k \cdot \sum_{i=1}^{t-1} \Delta r_i$，奖励推理更新带来的准确率提升，鼓励模型通过视觉推理获取更多证据
- **Exploratory Incentive**：$r_{\text{explo}} = \max(\omega - R(X), 0) \cdot \mathbb{1}_{\text{mul}}(R)$，对多模态推理比例设置下限 $\omega$，防止模型退化为纯文本推理（VLM 文本推理能力天然更强，容易陷入局部最优）

## 实验

### 训练配置

- 基础模型：Qwen2.5-VL-7B
- Cold Start：1.2k GPT-4o 蒸馏数据
- 训练数据：VideoGen-Reward (182k) + MJ-Bench-Video (8.7k) + Text2Video-Human Preferences (2.6k)
- 评估：GenAI-Bench（短视频）、VideoGen-RewardBench（现代 T2V）、MJ-Bench-Video

### 主要结果

| 模型 | 类型 | GenAI-Bench (diff) | VideoGen-Reward (diff) | MJ-Bench-Video (diff) |
|------|------|-------------------|----------------------|---------------------|
| VideoScore (7B) | 分类 | 70.9 | 50.2 | 63.5 |
| VideoReward (2B) | 分类 | 73.1 | 73.8 | 62.6 |
| VisionReward (13B) | 分类 | 72.7 | 68.4 | 65.2 |
| UnifiedReward (7B) | 生成 | 76.8 | 78.6 | 69.5 |
| UnifiedReward-Think (7B) | 推理 | 80.4 | 79.1 | 71.9 |
| **VR-Thinker (7B)** | 推理 | **82.3** | **80.5** | **75.6** |

VR-Thinker 在所有基准上达到开源 SOTA，且仅在视频偏好数据上训练（不像 UnifiedReward 还用了图像数据）。

### 消融实验

- **视觉推理 vs 随机检索**：随机检索帧性能显著下降，证明视觉推理（而非仅增加帧数）是关键
- **训练阶段**：GRPO 贡献最大；Cold Start 和 Rejection FT 提供关键推理基础，尤其 Rejection FT 增益显著（提高高质量轨迹比例，提升 GRPO 效率）
- **辅助 Reward**：去除 CoT Gain Reward 影响最大（鼓励视觉推理的核心信号）；去除 Exploratory Incentive 也明显下降（模型退化为纯文本推理）
- **Accuracy Reward 设计**：同时评估整体+各维度（$3^{d+1}$ 答案空间）远优于仅评估整体偏好（3 答案空间），因为扩大答案空间减少偶然猜对的概率

## 关键启示

- **视频 reward 模型需要主动视觉推理而非被动降采样**：Thinking-with-Image 框架让模型在推理过程中按需检索帧，突破固定帧数限制
- **窗口记忆 + 快照压缩是处理长视频上下文的实用方案**：保持 context 预算稳定的同时不丢失关键信息
- **扩大 accuracy reward 的答案空间是提升 GRPO 效率的关键**：从 3 选项扩展到 $3^{d+1}$，减少偶然正确但推理错误的奖励信号
- **CoT Gain Reward 和 Exploratory Incentive 对视觉推理能力至关重要**：前者奖励每轮推理的进步，后者防止退化为纯文本推理
- **三阶段训练（SFT → Rejection FT → GRPO）层层递进**：Cold Start 教格式，Rejection FT 提高高质量比例，GRPO 强化探索
