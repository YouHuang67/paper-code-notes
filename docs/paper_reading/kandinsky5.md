---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - Reward Model
---

# Kandinsky 5.0: A Family of Foundation Models for Image and Video Generation

- 论文：https://arxiv.org/abs/2511.14993
- 团队：Kandinsky Lab

## 概述

Kandinsky 5.0 是开源的生成模型家族，包含 Image Lite (6B)、Video Lite (2B)、Video Pro (19B) 三个规模，共享 CrossDiT backbone。支持文本到图像/视频、图像到视频和指令编辑，视频最长 10 秒 HD 分辨率。后训练流程为 SFT → RL（仅图像），其中 RL 采用 DRaFT-K 直接 reward 微调。在 MovieGen benchmark 人类评估中，Video Lite 在视觉质量和运动动态上优于 Sora 和 Wan 2.1/2.2。

## 架构

- **CrossDiT**：使用显式 cross-attention（分离 self-attention + cross-attention + MLP），兼容稀疏注意力
- **文本编码**：Qwen2.5-VL-7B（主编码器，256 token）+ CLIP ViT-L/14（全局嵌入与时间步相加）
- **NABLA 稀疏注意力**：HD 视频（>512px, >5s）下 90% 稀疏度，计算量减少 2.7x，无可测量质量损失

## 后训练

### SFT

- 图像：153K 高质量图像，按子域独立微调后**权重平均**（SFT-soup），避免跨域干扰
- 视频：~2.8K 视频 + 45K 图像，同样使用 SFT-soup 策略

### Reward 模型（仅图像）

基于 Qwen2.5-VL-7B 初始化，训练为成对比较判别器。利用自然质量序构造偏好数据，**无需人工标注**：

$$\text{预训练输出} < \text{SFT 输出} < \text{真实 SFT 数据集图像}$$

三元组提供免费的偏好信号。Reward 模型输出 $R(x_1, x_2, y) = P(\text{"Yes"} | x_1, x_2, y)$（$x_1$ 优于 $x_2$的概率）。

通过 KDE 监控分数分布防止过拟合，最优 checkpoint 在 1300 步。

### RL 微调：DRaFT-K

Direct Reward Fine-Tuning 变体，仅反传梯度通过最后 $K=10$ 步去噪：

$$\mathcal{L} = \mathcal{L}_{\text{RL}} + \beta_{\text{KL}} \cdot \text{KL}(p_{\text{RL}} \| p_{\text{SFT}})$$

其中 $\mathcal{L}_{\text{RL}} = 1 - R(x_\nabla, x_{\text{real}}, y)$，$x_{\text{real}}$ 是 SFT 数据集的真实图像（非第二次生成）。

Flow matching 的 KL 约束简化为速度场 L2 距离：

$$\text{KL}(p_{\text{RL}} \| p_{\text{SFT}}) = \sum_t \|v_{\text{RL}}(x_t, t) - v_{\text{SFT}}(x_t, t)\|^2$$

最优 $\beta_{\text{KL}} = 2 \times 10^{-2}$，$K = 10$。

**注意**：RL 后训练目前仅应用于图像模型，视频模型停留在 SFT + 蒸馏。

## 实验

### 人类评估（MovieGen benchmark）

- Video Lite vs Sora：~65K 成对评判，视觉质量和运动动态上 Video Lite 被偏好
- Video Lite vs Wan 2.1/2.2：视觉质量和运动动态上 Video Lite 被偏好
- Video Pro vs Veo 3：视觉质量和运动动态优于 Veo 3，prompt following 弱于 Veo 3
- Image Lite vs FLUX.1 [dev]：视觉质量更强，prompt following 持平

### 蒸馏

Flash 版本将 NFE 从 100 降至 16，质量有中等损失。

## 关键启示

- **启发式质量序消除标注成本**：预训练 < SFT < 真实数据的自然序列提供免费偏好信号，是大规模 RLHF 的可扩展替代方案
- **真实图像作为 RL 参考优于自比较**：与真实 SFT 数据集图像比较（非两次生成比较）提供稳定的高质量锚点
- **SFT-soup（权重平均）处理领域多样性**：独立子域微调后权重平均，避免跨域干扰同时保持各内容类型的质量
- **Flow matching 的 KL 约束自然简化为速度场 L2 距离**：计算廉价且易调参（单标量 $\beta_{\text{KL}}$）
- **视频 RL 后训练仍是开放问题**：尽管图像 RL 效果明显，但视频对齐尚未应用 RL——SFT-soup + 蒸馏是当前视频质量的实际上限
