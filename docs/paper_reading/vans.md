---
tags:
  - Video Generation
  - Reinforcement Learning
  - GRPO
  - Flow Matching
  - VLM
---

# VANS: Video-as-Answer with Joint-GRPO for Next Event Prediction

- 论文：https://arxiv.org/abs/2511.16669
- 代码：https://github.com/KlingTeam/VANS
- 团队：City University of Hong Kong, Kling Team (Kuaishou)

## 概述

VANS 开创 Video-Next-Event Prediction (VNEP) 任务——给定场景描述，生成展示下一个事件的视频（而非传统文本回答）。核心创新是 Joint-GRPO：两阶段 RL 对齐 VLM（Qwen2.5-VL-3B 生成描述）和 VDM（Wan-2.1-1.3B 生成视频）的联合系统。Stage 1 用视频保真度 reward 反向引导 VLM 生成"可视化友好"的描述；Stage 2 固定 VLM 做锚点，用语义一致性 reward 优化 VDM。构建 VANS-Data-100K 数据集。程序性任务 ROUGE-L +29.1%，CLIP-T +19.4%，FVD 从 85.34 降到 78.32。

## 动机

- 传统 Next-Event Prediction 仅生成文本答案，无法表达空间布局、运动和时序等视觉信息
- 级联 VLM→VDM 管道存在语义-视觉鸿沟：VLM 生成语言正确但视觉不可实现的描述
- 统一模型在理解和生成之间存在 trade-off

## 方法

### 架构

1. **VLM**：Qwen2.5-VL-3B，输入 ViT 视觉特征 + 文本问题，输出带 CoT 推理的事件描述
2. **VDM**：Wan-2.1-1.3B，条件：VLM 描述 + VAE 编码的 $n=6$ 帧低层特征，输出 352×640、33 帧视频
3. **Joint-GRPO**：两阶段 RL 对齐

### Stage 1: Visualization-Friendly VLM Tuning

冻结 VDM，优化 VLM 策略。对每个输入采样 $G$ 个描述 $\{s_i\}_{i=1}^G$，经冻结 VDM 生成视频，计算联合 reward：

$$r_1(s_i, v_i^{\text{out}}) = \lambda_f r_f(s_i) + \lambda_{t1} r_{t1}(s_i, s^{gt}) + \lambda_{v1} r_{v1}(v_i^{\text{out}}, v^{gt})$$

- $r_f$：格式合规（是否遵循 `[Think][/Think][Ans][/Ans]` 模板）
- $r_{t1}$：ROUGE-L 文本保真度
- $r_{v1}$：CLIP Similarity 视频保真度

关键：$r_{t1}$ 单独使用导致视觉不可实现的描述，$r_{v1}$ 单独使用信号过于间接，二者结合引导 VLM 生成"可执行"的描述。

训练：800 steps，LR $5 \times 10^{-5}$，LoRA (rank=8, $\alpha=32$)

### Stage 2: Context-Faithful VDM Adaptation

冻结 VLM 做锚点模型，优化 VDM。VLM 生成锚描述（过滤 ROUGE-L < 0.6），VDM 采样 $G$ 个视频：

$$r_2(v_i^{\text{out}}, s^{\text{anchor}}) = \lambda_{v2} r_{v2}(v_i^{\text{out}}, v^{gt}) + \lambda_{c2} r_{c2}(v_i^{\text{out}}, s^{\text{anchor}})$$

- $r_{v2}$：CLIP 视频保真度
- $r_{c2}$：CLIPScore 视频-文本语义一致性（防止 reward hacking 产生静态帧）

ODE 转 SDE 做 GRPO 训练。配置：1000 steps，LR $5 \times 10^{-5}$，KL 系数 $\beta=0.004$，clip range $1 \times 10^{-3}$，$G=8$

### VANS-Data-100K

- 30k 程序性（YouCook2 9k + COIN 21k）+ 70k 预测性（Video-Holmes、ActivityNet、YouTube 等）
- Gemini-2.5-Flash 筛选片段 + CoT 推理生成 QA 对
- 1k 高质量样本手动挑选用于 RL 后训练

## 实验

### 主要结果

**程序性基准**（400 样本）：

| 模型 | ROUGE-L | FVD | CLIP-V | CLIP-T |
|------|:---:|:---:|:---:|:---:|
| VANS (SFT) | 0.2812 | 85.34 | 0.7655 | 0.3202 |
| **VANS (Joint-GRPO)** | **0.3631** | **78.32** | **0.8021** | **0.3824** |
| Gemini-FilmWeaver | 0.2802 | 110.54 | 0.7102 | 0.2773 |
| Omni-Video | 0.1075 | 236.38 | 0.6293 | 0.2323 |

**预测性基准**（400 样本）：

| 模型 | ROUGE-L | FVD | CLIP-V | CLIP-T |
|------|:---:|:---:|:---:|:---:|
| VANS (SFT) | 0.2435 | 94.12 | 0.7512 | 0.3038 |
| **VANS (Joint-GRPO)** | **0.3058** | **86.85** | **0.7872** | **0.3759** |

Joint-GRPO vs SFT 提升：ROUGE-L +29.1%，CLIP-T +19.4%，FVD 85.34→78.32

### 消融

- VLM-only GRPO：文本提升但视觉对齐缺失
- VDM-only GRPO：无语义引导，提升极小
- VLM+VDM 级联 GRPO：次优（reward 归因模糊）
- All-in-one 联合 GRPO：更差（优化不稳定）
- **两阶段 Joint-GRPO**：全指标最优

去掉 $r_{c2}$（Stage 2 语义一致性）导致 reward hacking（静态帧）

### 人类评估（30 评估者，20 样本，1-5 分）

- 语义正确性：VANS (Joint-GRPO) 4.7 vs Gemini-FilmWeaver 3.9
- 视觉一致性：VANS (Joint-GRPO) 4.6 vs VANS (SFT) 3.9
- 推理时间：VANS ~39s（4s 描述 + 35s 视频）vs Omni-Video ~50s

## 关键启示

- **VLM-VDM 联合对齐比级联或统一模型更有效**：两阶段设计解决 reward 归因问题——Stage 1 让 VLM 内化 VDM 约束，Stage 2 让 VDM 锚定在语义正确的描述上
- **复合 Reward 防止 reward hacking**：文本保真度 + 视频保真度 + 语义一致性三重 reward 缺一不可
- **"以视频为答案"的范式价值**：程序性学习（如系鞋带）视频示范远优于文本描述，开辟新的人机交互模式
