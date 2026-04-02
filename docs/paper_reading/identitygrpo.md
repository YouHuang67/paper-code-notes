---
tags:
  - Video Generation
  - Reinforcement Learning
  - GRPO
  - Flow Matching
  - Reward Model
---

# Identity-GRPO: Multi-Human Identity-Preserving Video Generation via RL

- 论文：https://arxiv.org/abs/2510.14256
- 团队：Alibaba Group, Fudan University

## 概述

Identity-GRPO 首次将 GRPO 应用于多人身份保持视频生成（MH-IPV）。现有 T2V reward（HPS、CLIP、VideoAlign）依赖高层语义，无法区分个体身份一致性；ArcFace 等人脸相似度指标与非身份因素高度相关，导致"复制粘贴"效果。方法包括：(1) 构建 15k 身份偏好数据集（10k 自动标注 + 5k 人工标注），(2) 用 Bradley-Terry-with-Ties 目标训练身份一致性 RM（Qwen2.5-VL-3B，达 89% 准确率），(3) GRPO 训练时采用差异化初始噪声 + 大组采样（16 组×8 个=128 视频）稳定训练。在 VACE-1.3B 上身份一致性 +18.9%，人类偏好胜率 76%。

## 动机

- VACE、Phantom 等模型可生成高保真视频，但多人场景常出现"身份交换"
- T2V reward 模型设计用于整体语义对齐，不足以分离身份保持与动态运动
- ArcFace 人脸相似度在帧间与非身份因素高相关，导致模型学会"复制粘贴"参考图
- 多模态条件（多参考图 + 文本）引入巨大方差，GRPO 训练不稳定

## 方法

### 偏好数据集构建

**自动标注数据（10k 对）**：
- OpenHumanVid 过滤：Qwen3 解析标题限制 ≤3 人，Qwen2.5-VL 筛选清晰正面人脸
- GroundingDINO + SAM2 精确人体分割
- Flux.1 合成多视角参考图（>80% 成功率）
- 5 个模型（VACE-1.3B/14B, Phantom-1.3B/14B, MAGREF）推理生成
- 多次 VLM 投票标注，GME 向量模型过滤不一致样本

**人工标注数据（5k 对）**：
- CelebA-HQ + OpenHumanVid，3 名标注员投票
- 评估维度：面部一致性、视觉质量、文本对齐

### 身份一致性 Reward Model

基于 Qwen2.5-VL-3B + LoRA ($\alpha=32$, $r=32$)，Bradley-Terry-with-Ties (BTT) 目标：

$$P(y_A \succ y_B | x, t) = \frac{e^{r(x,t,y_A)}}{e^{r(x,t,y_A)} + \theta e^{r(x,t,y_B)}}$$

$$P(y_A = y_B | x, t) = \frac{(\theta^2 - 1)e^{r(x,t,y_A)}e^{r(x,t,y_B)}}{(e^{r(x,t,y_A)} + \theta e^{r(x,t,y_B)})(\theta e^{r(x,t,y_A)} + e^{r(x,t,y_B)})}$$

$\theta=5$（tie 倾向参数）。

**两阶段训练**：
1. 先在人工标注数据上训练 RM_teacher
2. RM_teacher 过滤自动标注数据（~48% 保留率）
3. 余弦退火采样策略联合训练：$\alpha_t = 0.5(1 + \cos(\pi t/T))$，从高自动标注比例逐渐过渡到人工标注主导

结果：89.0% 偏好准确率（ArcFace 77.2%，Qwen2.5VL-3B 基线 43%，InternVL3.5-38B 68.5%）

### Identity-GRPO 训练

Flow matching 模型 ODE 转 SDE：

$$dz_t = \left[v_\theta(z_t, c, t) + \frac{\sigma_t^2}{2t}(z_t + (1-t)v_\theta(z_t, c, t))\right]dt + \sigma_t dw$$

**训练稳定策略**：
- **Prompt 微调**：Qwen2.5-VL-7B 生成精确人物描述，解决不同模型对 prompt vs 参考图的敏感度差异
- **差异化初始噪声**：组内使用不同初始化噪声，扩大探索空间（超越 SDE 随机性）
- **大组采样**：16 组×8 个=128 视频/更新，用分辨率（416×240）和帧数（33 帧）换取采样数量

配置：$G=8$，$\epsilon=1 \times 10^{-3}$，采样 25 步（训练）/ 50 步（评估），8×A100

## 实验

### RM 偏好准确率（500 对人工标注基准）

| 模型 | 准确率 |
|------|:---:|
| ArcFace | 77.2% |
| Qwen2.5VL-3B | 43.0% |
| InternVL3.5-38B | 68.5% |
| **Identity-GRPO RM** | **89.0%** |

Smooth sampling vs alternatives：人工标注 only 85.3%，自动标注 raw 66.4%，Random 82.4%，**Smooth 89.0%**

### GRPO 训练结果（100 测试样本）

| 模型 | ID-Consistency | Aesthetics | Winning Rate |
|------|:---:|:---:|:---:|
| VACE-1.3B | 2.606 | 45.58% | 24% |
| + Identity-GRPO | **3.099** (+18.9%) | 47.56% | **76%** |
| Phantom-1.3B | 3.809 | 44.13% | 37% |
| + Identity-GRPO | **4.056** (+6.5%) | 47.03% | **63%** |

### 稳定性消融

| 组数 | 差异化噪声 | ID-Consistency |
|:---:|:---:|:---:|
| 4 | 否 | 2.588 |
| 4 | 是 | 2.749 |
| 16 | 否 | 2.718 |
| 16 | 是 | **3.099** |

16 组+差异化噪声比 4 组+相同噪声提升 19.8%

### SFT vs GRPO

| 方法 | ArcFace Similarity | ID-Consistency |
|------|:---:|:---:|
| 基线 | 0.235 | 2.606 |
| SFT | 0.261 (+11.1%) | 2.774 |
| **Identity-GRPO** | **0.298** (+26.8%) | **3.099** (+18.9%) |

## 关键启示

- **细粒度任务需要专用 RM**：通用 VLM reward 对身份一致性任务完全失效（43-68.5%），针对性偏好数据+BTT 训练达 89%
- **数据质量>数量**：48% 保留率的过滤自动标注+人工标注余弦退火混合，优于任一单独使用
- **多模态条件下 GRPO 需要大量采样**：128 视频/更新才能稳定训练，差异化初始噪声扩大探索空间是关键
- **GRPO 优于 SFT**：偏好优化在 ArcFace 上 +26.8%，SFT 仅 +11.1%——RL 在细粒度目标上的优势
