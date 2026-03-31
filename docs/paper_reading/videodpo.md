---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - DPO
---

# VideoDPO: Omni-Preference Alignment for Video Diffusion Generation

- 论文：https://arxiv.org/abs/2412.14167
- 代码：https://videodpo.github.io/
- 团队：HKUST, Renmin University of China, Johns Hopkins University

## 概述

VideoDPO 是首个将 DPO（Direct Preference Optimization）系统性地适配到视频扩散模型的工作。核心创新有三点：(1) 提出 OmniScore，一个同时评估视觉质量和语义对齐的综合评分体系；(2) 基于 OmniScore 自动生成偏好对数据，免去人工标注；(3) 引入基于频率直方图的偏好对重加权策略，让模型更关注区分度大的样本对。在 VideoCrafter2、T2V-Turbo、CogVideo 三个开源模型上验证，VideoDPO 在 VBench 总分上均带来提升（如 VC2 从 80.44% 提升至 81.93%），同时在 HPS(V) 和 PickScore 等人类偏好指标上也有改善。

## 方法

### OmniScore：综合偏好评分

与以往只关注单一维度（如美学或语义）的 reward model 不同，OmniScore 同时评估三个主要维度：

**帧内质量（Intra-frame）**：
- Image Quality：使用 MUSIQ 模型评估画面失真程度（过曝、噪声、模糊）
- Aesthetic Quality：使用 LAION aesthetic predictor 评估画面美学

**帧间质量（Inter-frame）**：
- Subject Consistency：使用 DINO 特征跨帧相似度评估主体一致性
- Temporal Flickering：通过 RAFT 静态帧的平均绝对差计算闪烁程度
- Motion Smoothness：使用视频帧插值模型的运动先验评估动作平滑度
- Dynamic Degree：通过 RAFT 估计视频动态程度

**文本-视频语义对齐**：
- 使用 ViCLIP 计算视频-文本整体一致性

各维度归一化到 [0, 1] 后加权求和，质量维度权重为 4，语义对齐权重为 1。

论文发现各质量子维度之间相关性很低（如 dynamic degree 与 temporal flickering 相关系数为 -0.65），说明优化单一维度无法自动改善其他维度，因此综合评分是必要的。

### 偏好对数据生成

流程：
1. 从 VidProm 数据集取 K=10,000 个人类编写的 prompt
2. 对每个 prompt 用待对齐的模型生成 N=4 个视频
3. 用 OmniScore 对每个视频打分
4. 取最高分视频作为 winning sample $v_W$，最低分作为 losing sample $v_L$，构成偏好对

策略选择实验表明，"Best-vs-Worst"（只取极端对）优于其他策略（如 better-vs-worse 生成多对），说明偏好对的质量比数量更重要。

### OmniScore-Based Re-Weighting

直接使用偏好对训练 DPO 的问题：部分 winning 和 losing 样本的分数差异很小，模型难以从中有效学习。

解决方案是基于频率直方图的重加权：
1. 对所有 $K \times N$ 个视频的 OmniScore 构建频率直方图
2. 对每个偏好对，计算其采样概率的几何均值：$\text{prob}(s_W, s_L) = \sqrt{p(s_W) \cdot p(s_L)}$
3. 重加权因子：$w_{\text{pair}} = (\beta / \text{prob}(s_W, s_L))^\alpha$
4. 最终损失：$L_{\text{video}} = L_{\text{DPO}}(p, v_W, v_L) \cdot w_{\text{pair}}$

其中 $\beta$ 为最高频样本的概率，$\alpha$ 为超参数。$\alpha=0$ 时退化为普通 DPO。实验表明 $\alpha=1.0$ 效果最佳。

直觉：低频的偏好对通常对应更具区分度的样本，给它们更高权重帮助模型学习更有意义的偏好信号。

## 实验

### 训练配置
- 3000 步，全局 batch size 8，AdamW 优化器，学习率 6e-6
- $\alpha=0.72$，$\beta=1$，OmniScore 直方图 bin width 0.01
- 4 张 A100 GPU

### 主要结果

| 模型 | VBench Total | Quality | Semantics | HPS(V) | PickScore |
|------|-------------|---------|-----------|--------|-----------|
| VC2 Baseline | 80.44 | 82.20 | 73.42 | 0.258 | 20.65 |
| VC2 + VADER | 80.59 | 82.46 | 73.09 | 0.259 | 20.62 |
| VC2 + VideoDPO | **81.93** | **83.07** | **77.38** | **0.261** | 20.65 |
| Turbo Baseline | 80.95 | 82.71 | 73.93 | 0.262 | 21.15 |
| Turbo + VideoDPO | **81.80** | **83.80** | 73.81 | 0.260 | **21.18** |
| CogVideo Baseline | 79.30 | 82.35 | 67.10 | - | 19.81 |
| CogVideo + VideoDPO | **79.80** | **83.00** | 66.99 | - | 19.79 |

与 SFT（仅用 winning sample 微调）相比，VideoDPO 有明显优势，说明从 negative sample 学习对减少低质量生成很重要。

### 消融实验

- **N 的影响**：N 越大，偏好对区分度越高。N=4 时 VBench Total 81.93%，N=2 时降至 80.89%
- **数据过滤**：过滤掉区分度低的偏好对反而降低性能（从 81.93% 降至 80.08%~81.42%），因为减少了 prompt 多样性
- **训练数据规模**：25% 数据时 VBench 80.21%，50% 时 80.83%，说明更多数据有利于泛化
- **单维度 vs 多维度**：仅用语义分数训练得 80.20%，仅用美学得 79.65%，OmniScore 得 81.93%

## 关键启示

- **综合评分优于单维度**：视频质量的各个子维度（动态性、平滑度、美学等）相关性很低，优化单一 reward 会顾此失彼
- **偏好对质量 > 数量**：Best-vs-Worst 策略只取最极端的一对，但效果最好
- **频率重加权有效**：给低频（高区分度）偏好对更高权重，比 vanilla DPO 提升约 1%
- **不要过滤低区分度数据**：虽然这些偏好对信号弱，但对应的 prompt 提供了训练多样性
- **DPO 优于 SFT 和单一 reward 梯度优化**：VADER 只能优化单个可微 reward，多 reward 训练计算成本高；DPO 通过偏好对间接融合多维度评价
