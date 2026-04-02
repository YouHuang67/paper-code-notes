---
tags:
  - Video Generation
  - DPO
  - Diffusion Model
---

# LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion

- 论文：https://arxiv.org/abs/2601.04068
- 团队：Harbin Institute of Technology, Alibaba Group (Taobao & Tmall)

## 概述

LocalDPO 提出局部化偏好优化框架，解决传统 DPO 的三个问题：(1) 需要多次采样+外部评分的高成本；(2) 全局评分导致模糊/冲突的监督信号；(3) 忽略区域级偏好线索。方法核心：用高质量真实视频作为正样本，通过随机时空掩码局部腐蚀后由冻结 base model 修复生成负样本（每 prompt 仅需一次推理），再用 region-aware DPO loss 将偏好学习限制在腐蚀区域。在 CogVideoX-5B 上 VideoAlign 从 9.72 提升到 10.29；Wan2.1-1.3B 上从 7.83 提升到 7.96。

## 动机

- 传统 DPO：每个 prompt 需生成多个视频 + 人工/模型标注排序，成本高且标注模糊
- 全局评分忽略局部质量差异：高总分视频可能在特定区域有伪影
- 同一 prompt 不同 seed 的视频在局部区域质量差异显著（Fig. 2），全局评分无法捕捉

## 方法

### 偏好对构建

**正样本**：高质量真实视频

**负样本生成（每 prompt 一次推理）**：
1. **随机时空掩码**：用多条 Bezier 曲线首尾相连构成闭合区域，在视频的空间+时间维度上生成随机掩码
2. **局部腐蚀**：对真实视频加噪，冻结 base model 仅对掩码区域做去噪修复，保持未掩码区域不变
3. 结果：负样本保持全局语义但在局部区域存在质量退化（细节丢失、时间闪烁）

**优势**：正样本质量在每个维度上都一致优于负样本（真实视频 vs 模型修复），消除标注歧义。

### 训练损失

三部分联合训练：

$$\mathcal{L}_{\text{total}} = \lambda_{\text{RA}} \mathcal{L}_{\text{RA-DPO}} + \lambda_{\text{DPO}} \mathcal{L}_{\text{DPO}} + \lambda_{\text{SFT}} \mathcal{L}_{\text{SFT}}$$

- **$\mathcal{L}_{\text{RA-DPO}}$（Region-Aware DPO）**：将偏好学习限制在掩码腐蚀区域，使模型精确学习局部质量差异
- **$\mathcal{L}_{\text{DPO}}$**：标准 Diffusion DPO loss，作为全局正则
- **$\mathcal{L}_{\text{SFT}}$**：真实视频上的 SFT loss，锚定高质量数据分布

消融显示 RA-DPO 是核心贡献（几乎所有指标提升归因于此），DPO 和 SFT 提供微弱辅助。

## 实验

### VBench Prompts 定量对比

| 方法 | Aesthetic | Imaging | HPS-v2 | VideoAlign Overall |
|------|:---:|:---:|:---:|:---:|
| **CogVideoX-2B** | | | | |
| Baseline | 0.6279 | 0.6589 | 0.2655 | 7.79 |
| Vanilla DPO | 0.6304 | 0.6598 | 0.2654 | 7.79 |
| DenseDPO | 0.6325 | 0.6606 | 0.2652 | 7.82 |
| **LocalDPO** | **0.6499** | **0.7080** | **0.2738** | **7.86** |
| **CogVideoX-5B** | | | | |
| Baseline | 0.6110 | 0.6631 | 0.2692 | 9.72 |
| SFT | 0.6132 | 0.6860 | 0.2728 | 9.36 |
| Vanilla DPO | 0.5953 | 0.6534 | 0.2658 | 9.59 |
| DenseDPO | 0.6233 | 0.6962 | 0.2674 | 9.57 |
| **LocalDPO** | **0.6274** | **0.7107** | **0.2782** | **10.29** |
| **Wan2.1-1.3B** | | | | |
| Baseline | 0.6363 | 0.6296 | 0.2727 | 7.83 |
| SFT | 0.6373 | 0.6342 | 0.2730 | 7.73 |
| DenseDPO | 0.6375 | 0.6356 | 0.2728 | 7.84 |
| **LocalDPO** | **0.6416** | **0.6412** | **0.2754** | **7.96** |

- Imaging Quality 提升最显著（局部细节优化的直接体现）
- SFT 在多个模型上导致 VideoAlign 下降（缺乏偏好信号），LocalDPO 持续提升
- Vanilla DPO 在 CogVideoX-5B 上反而导致性能下降（全局评分模糊性的负面影响）

### 偏好对构建效率

CogVideoX-5B 上构建 1000 对偏好数据：
- Vanilla DPO：需要多次采样 + 标注
- LocalDPO：每 prompt 仅一次推理 + 自动掩码

## 关键启示

- **局部偏好优化比全局更精确**：视频质量差异主要体现在局部区域（伪影、闪烁、细节丢失），region-aware loss 直接定位并优化这些区域
- **真实视频 vs 模型修复天然构成高置信偏好对**：无需外部评分模型，正样本在每个维度上都一致优于负样本
- **单次推理生成偏好对大幅降低成本**：冻结 base model 局部修复，无需多轮采样和人工标注
- **SFT 对视频生成的局限性**：等权处理所有训练样本，无法学习相对质量差异——偏好学习（DPO）在这类场景更有效
