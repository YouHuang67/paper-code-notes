---
tags:
  - Video Generation
  - VLM
  - Reward Model
---

# Video-Bench: Human-Aligned Video Generation Benchmark

**论文**: [arXiv 2504.04907](https://arxiv.org/abs/2504.04907)
**代码**: [Video-Bench/Video-Bench](https://github.com/Video-Bench/Video-Bench)
**团队**: SJTU, Stanford, CMU, PKU, Fudan, PolyU, Soochow, Glasgow, CityU HK, Westlake, NUS, LiveX AI
**发表**: CVPR 2025

## 概述

Video-Bench 是首个系统性利用 MLLM 进行全维度视频生成评估的 benchmark，CVPR 2025 接收。现有评估方法分两派：传统指标（FVD、CLIP score 等）与人类判断对齐差；LLM-based 方法受限于跨模态比较困难和评分尺度模糊。Video-Bench 提出两大技术创新解决这些问题：(1) **Chain-of-Query**：将跨模态对齐评估转化为"视频描述 → 多轮提问 → 验证回答"的文本对文本比较过程，避免 MLLM 直接跨模态比较的幻觉问题；(2) **Few-shot Scoring**：对视频质量维度，用同 prompt 下多个视频作为上下文参照，建立相对质量标尺，解决文本评分标准模糊导致 MLLM 通通打平均分的问题。

评估覆盖 9 个子维度（视频-条件对齐 5 个 + 视频质量 4 个），419 条 prompt，7 个 T2V 模型（含开源和商业），35k+ 人工标注。与人类判断的 Spearman 相关性达 0.73，远超 VBench、EvalCrafter、CompBench 等方法，与人类标注者间一致性（0.52）相当。

## 评估维度体系

分两大类 9 个子维度：

### 视频-条件对齐（5 维度）

1. **Object Class Consistency（物体类别一致性）**：视频中物体是否与 prompt 描述匹配，物体外观和结构是否符合客观现实。3 分制
2. **Action Consistency（动作一致性）**：动作准确性和清晰度，是否符合 prompt 描述和物理规律。3 分制
3. **Color Consistency（颜色一致性）**：物体颜色是否与 prompt 一致，颜色是否跨帧稳定。3 分制
4. **Scene Consistency（场景一致性）**：场景元素是否与 prompt 对齐，空间布局是否合理。3 分制
5. **Video-Text Consistency（视频-文本整体一致性）**：所有核心元素（人物/动物/动作/物体/场景/风格/空间关系/数量关系）整体对齐度。5 分制

### 视频质量（4 维度）

1. **Imaging Quality（成像质量）**：单帧技术质量——噪声、模糊、过曝、伪影等。5 分制
2. **Aesthetic Quality（美学质量）**：艺术吸引力、构图、整体视觉协调性。5 分制
3. **Temporal Consistency（时间一致性）**：视觉特征（色彩/亮度/纹理）跨帧平滑过渡 + 语义（物体/主体/场景）跨帧稳定。5 分制
4. **Motion Quality（运动质量）**：运动合理性（符合物理规律）+ 运动幅度（与 prompt 描述匹配，不过度或过微弱）。5 分制

### Prompt Suite

419 条 prompt，约 70-90 条/维度。动态维度（Action/Temporal/Motion）结合 Kinetics-400 动作数据和 VBench 的刚体/动物运动数据。每个 prompt 采样 3 次去偏差。

## 评估框架

### 挑战 1：跨模态比较困难 → Chain-of-Query

MLLM 直接比较视频（视觉信号）和文本（语义概念）时容易出现幻觉和文本偏见。Video-Bench 将跨模态比较转化为文本对文本比较：

1. **Video Description**：MLLM 先生成视频的完整描述和单句摘要
2. **Query Chain Generation**：LLM 根据视频 prompt 和初始描述，按预定义策略生成 N 组问题。以颜色维度为例——Q1："考拉的颜色是否与 prompt 匹配？"；Q2："考拉的棕色是否被海浪颜色混淆？"
3. **Answer Chain Generation**：MLLM 重新观看视频进行反思，然后逐一回答 query
4. **Final Scoring**：MLLM 结合视频内容和多轮对话历史，按文本指南打分

### 挑战 2：评分标准模糊 → Few-shot Scoring

仅靠文本描述区分"有点模糊"和"非常模糊"不够，MLLM 习惯给所有视频打平均分。解决方案：同一 prompt 的多个生成视频批量处理，前一个视频的评分作为后一个的隐式参照，建立相对质量标尺。MLLM 通过比较学习区分质量等级。

## 实验

### 评估模型

4 个开源 + 3 个商业：LaVie, Show-1, VideoCrafter2, CogVideoX-5B, Pika-Beta, Kling, Gen3。

### Leaderboard（Table 1）

| 模型 | Video Quality Avg | Video-Condition Align Avg | Overall Rank |
|------|-------------------|--------------------------|-------------|
| Gen3 | 4.46 | 2.80 | 1 |
| CogVideoX | 3.85 | 2.87 | 2 |
| VideoCrafter2 | 3.61 | 2.77 | 3 |
| Kling | 3.89 | 2.71 | 4 |
| Show-1 | 3.35 | 2.72 | 5 |
| PiKa-Beta | 3.38 | 2.47 | 6 |
| LaVie | 2.84 | 2.68 | 7 |

Gen3 综合最佳，CogVideoX 视频-文本一致性最优。

### 与人类对齐（Table 2，Spearman 相关系数）

| 维度 | CompBench | Video-Bench (Ours) | 提升 |
|------|-----------|-------------------|------|
| Imaging Quality | — | 0.733 | — |
| Aesthetic Quality | — | 0.702 | — |
| Video-Text Consist. | 0.633 | 0.732 | +0.099 |
| Object-Class Consist. | 0.611 | 0.735 | +0.124 |
| Color Consist. | 0.696 | 0.750 | +0.054 |
| Action Consist. | 0.633 | 0.718 | +0.085 |
| Scene Consist. | 0.631 | 0.733 | +0.102 |

Video-Bench 在所有维度上超出现有方法，平均 Spearman ~0.73。

### 与人类评估者一致性（Table 3, Krippendorff's α）

| 对比 | Avg α |
|------|-------|
| 人类-人类 | 0.52 |
| 人类-GPT（朴素） | 0.41 |
| 人类-Video-Bench | **0.50** |

Video-Bench 与人类的一致性接近人类标注者间的自一致性，远超朴素 GPT-4o 评估。

### 消融实验（Table 4）

Few-shot scoring 对视频质量维度贡献显著：Imaging Quality 从 0.639→0.733，Aesthetic Quality 从 0.627→0.702。

## 关键启示

- **跨模态比较应转化为文本对文本**：MLLM 在视觉-文本跨模态比较中容易幻觉，Chain-of-Query 先描述再提问让 MLLM 只需做文本推理，大幅提升对齐评估准确性
- **相对评分优于绝对评分**：Few-shot scoring 用同类视频互相参照，解决了绝对评分标准模糊导致 MLLM"全打 3 分"的问题。这一思路也可用于 reward 模型训练数据的自动标注
- **MLLM 评估已接近人类标注者间一致性**：Video-Bench α=0.50 vs 人类间 α=0.52，表明自动化评估有潜力替代部分人工评估
- **视频质量和视频-条件对齐需要不同的评估策略**：前者用 few-shot 比较，后者用 chain-of-query 文本化，一刀切的单一 prompt 评估效果差
- **Benchmark 设计对 reward 模型的启发**：9 维度细粒度评估体系 + MLLM 评估策略，可为训练视频 reward 模型提供标注框架和数据生成思路
