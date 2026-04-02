---
tags:
  - Video Generation
  - Reward Model
  - Reinforcement Learning
  - GRPO
  - VLM
---

# VideoScore2: Think Before You Score in Generative Video Evaluation

- 论文：https://arxiv.org/abs/2509.22799
- 团队：UIUC, University of Waterloo, M-A-P, University of Toronto

## 概述

VideoScore2 是一个基于 VLM 的视频评估/奖励模型，核心创新在于引入可解释的 chain-of-thought (CoT) 推理——模型先生成详细分析推理，再给出多维度评分。评估三个维度：视觉质量（VQ）、文本对齐（TA）、物理/常识一致性（PC）。训练采用两阶段：SFT 建立格式和任务理解 → GRPO 强化学习提升评分准确性。构建 VideoFeedback2 数据集（27,168 视频，81,504 评分+推理），覆盖 20+ T2V 模型。在 4 个 OOD 基准上平均性能超出次优 +4.32 点，可作为视频生成对齐的有效 reward 模型。

## 动机

- 现有视频评估器的关键局限：
  - 单一不透明分数，缺乏可解释性（VideoReward、VideoPhy2、Q-Insight）
  - 无法捕捉视频质量的多维度本质（视觉保真度、语义对齐、物理合理性）
  - OOD 泛化差，仅 SFT 训练的模型在新数据分布上表现衰退
  - 缺乏推理痕迹用于问责和理解

## 方法

### 三维度评估框架

| 维度 | 定义 |
|------|------|
| Visual Quality (VQ) | 分辨率、清晰度、平滑性、亮度稳定性、畸变和伪影 |
| Text Alignment (TA) | 与 prompt 在主体、动作、细节、风格、事件序列上的一致性 |
| Physical Consistency (PC) | 物理定律和常识的遵守程度，异常伪影的有无 |

### VideoFeedback2 数据集

- **27,168 视频**，每视频 3 维度评分 + 推理文本 = 81,504 条标注
- **2,933 unique prompts**（VidProM 真实用户查询 + Koala-36M 结构化描述 + 手动构建 OCR/多动作/镜头运动）
- **20+ T2V 模型**分四档：
  - Tier 1（完美/现代）10.36%：Kling、Sora、Pika、StepVideo
  - Tier 2（优质）33.53%
  - Tier 3（中等）41.77%
  - Tier 4（差/早期）12.54%：ModelScope、ZeroScope
- 分辨率 256×256 到 1920×982，8-30 fps，1-6 秒

### 推理文本构建流程

1. **人工标注**：15 名标注员给 1-5 分 + 简短评语（IAA 93.3%-96.7%）
2. **LLM 扩展**（Claude Sonnet 4 + thinking）：从人工评语生成 200-600 词的 CoT 推理
3. **分数校准**：模型分与人工分差 ≤1 保留人工分，差 2 取均值，差 ≥3 重新评分（最多 3 轮，<10% 丢弃）
4. **文本对齐**（GPT-4-mini）：确保推理文本与最终分数一致

### 两阶段训练

**Stage 1: SFT**

- 基础模型：Qwen2.5-VL-7B-Instruct
- 配置：LR $5 \times 10^{-5}$，2 epochs，2 fps 采样，最大 960×720
- 8×A800 GPU，约 6h/epoch
- 目的：建立格式遵守和任务熟悉度

**Stage 2: GRPO 强化学习**

- Reward 函数（基于与 ground-truth 的匹配度）：

$$R_{\text{acc}} = \begin{cases} 1.0 & \text{三个维度全部精确匹配} \\ 0.7 & \text{两个匹配，一个差 1} \\ 0.4 & \text{一个匹配，两个差 1} \\ 0.1 & \text{三个均差 1} \\ 0 & \text{其他} \end{cases}$$

- 格式 reward：$R_{\text{fmt}} = 1$（含 `<think>` 标签）或 $0$
- 总 reward：$R = R_{\text{acc}} + \lambda R_{\text{fmt}}$（从 SFT checkpoint 出发 $\lambda = 0$，从 base model 出发 $\lambda = 0.3$）
- 配置：LR $2 \times 10^{-6}$，每 rollout 8 个生成，4×A100，约 8h/100 steps
- 最优 checkpoint：300 RL steps（超过后 in-domain 性能下降）

### 推理设置

- Temperature 0.7（增加多样性）
- 整数预测 {1,2,3,4,5} 转浮点分数（softmax 加权）
- 2 fps 帧采样

## 实验

### In-Domain: VideoScore-Bench-V2（500 hold-out 视频）

| 指标 | Visual | Align | Physical | Avg |
|------|:---:|:---:|:---:|:---:|
| Accuracy | 50.10 | 43.88 | 39.08 | **44.35** |
| Relaxed Acc | 92.99 | 91.38 | 87.98 | **90.78** |
| PLCC | 60.13 | 62.60 | 52.73 | **60.37** |

- 相比 SFT-only：Accuracy +5.94, Relaxed Acc +4.01, PLCC +8.32

### OOD 泛化（4 个基准）

| 基准 | 性能 |
|------|:---:|
| VideoGen-Reward (偏好) | 51.53% |
| T2VQA-DB (偏好) | 50.60% |
| MJ-Bench-Video (点评分) | 65.77% |
| VideoPhy2-test (点评分) | 33.58% |
| **OOD 平均** | **50.37%** (+4.32 over next best) |

### 作为 Reward Model: Best-of-N 采样

- 在 6 个中等/差质量 T2V 模型上做 BoN (n=5)
- VBench 各维度一致性提升，验证可用作视频生成优化的有效 reward

## 关键启示

- **CoT 推理提升评估质量**：先推理后评分比直接评分更准确，且提供可解释性——用户可理解扣分原因
- **GRPO 提升 OOD 泛化**：SFT 倾向于过拟合训练分布，300 步 GRPO 显著提升跨域泛化（+4.32 OOD 平均）
- **多维度评估的必要性**：VQ/TA/PC 三维度捕捉视频质量的不同方面，单一分数无法区分"画质好但物理不合理"和"物理合理但语义偏移"
- **分层 reward 设计**：匹配度递减的阶梯式 reward（1.0→0.7→0.4→0.1→0）比二值 reward 提供更精细的梯度信号
- **可作为通用视频 reward**：OOD 泛化能力使其适合在新的 T2V 模型上做 RLHF/GRPO 对齐，无需针对每个模型重新训练评估器
