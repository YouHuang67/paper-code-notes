---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - Flow Matching
---

# PhysMaster: Physics Representation Learning for Video Generation

- 论文：https://arxiv.org/abs/2510.13809
- 团队：Tsinghua University, Kuaishou Technology

## 概述

PhysMaster 提出通过学习物理表征（physical representation）来提升 I2V 模型的物理一致性。核心组件 PhysEncoder 是一个即插即用的物理编码器，基于 DINOv2（从 Depth Anything 初始化），通过附加的 physical head 从输入图像中提取物理嵌入，与图像嵌入拼接后作为 DiT 的额外条件。训练分三阶段：(1) SFT 联合训练 DiT 和 PhysEncoder；(2) Flow-DPO 优化 DiT（LoRA）；(3) Flow-DPO 单独优化 PhysEncoder。关键发现：DPO 阶段对 DiT 和 PhysEncoder 分别顺序优化优于联合优化，且先 DiT 后 PhysEncoder 的顺序效果最好。在代理任务（Kubric 自由落体）上验证方法有效性后，推广到通用物理场景（WISA-80K，17 类物理事件），在 VIDEOPHY 基准上达到 PC 0.40 / SA 0.67。

## 动机

现有 I2V 模型生成的视频在物理一致性方面表现差——物体运动不符合物理规律、碰撞无正确反馈、重力效果不真实。原因：(1) 训练数据中物理信息隐式存在，模型难以显式学习物理规律；(2) 直接用 DPO 优化 DiT 在物理一致性上改进有限，因为 DiT 同时负责视觉质量和物理一致性，两个目标可能冲突。

PhysMaster 的思路：将物理知识从 DiT 中解耦出来，由专门的 PhysEncoder 负责编码，DiT 专注于视觉生成。

## 方法

### PhysEncoder 架构

- **骨干**：DINOv2-Large（从 Depth Anything V2 初始化），提取通用视觉特征
- **Physical Head**：附加在 DINOv2 之上的轻量头部，将视觉特征映射为物理嵌入
- **条件注入**：物理嵌入与图像嵌入（来自 CLIP）拼接，通过 cross-attention 注入 DiT

初始化选择 Depth Anything 而非原始 DINOv2，因为深度信息与物理属性（如重力方向、空间关系）高度相关。

### 代理任务：自由落体

用 Kubric 引擎构建合成数据集，物体在重力作用下自由下落。优势：
- 物理规律完全已知（牛顿力学），可精确计算 ground truth 轨迹
- 自动生成 win/lose 对：与 GT 轨迹偏差小的为 win，偏差大的为 lose
- 评估指标客观：L2 距离、Chamfer Distance（CD）、IoU

### 三阶段训练

**Stage I: SFT**

联合训练 DiT（全参数）和 PhysEncoder，使用标准 flow matching 损失。此阶段让 PhysEncoder 学会提取有用的物理特征，DiT 学会利用这些特征。

**Stage II: Flow-DPO for DiT**

冻结 PhysEncoder，用 LoRA（rank 128）对 DiT 进行 Flow-DPO 优化。采用 VideoAlign 提出的常数 $\beta$（而非时间相关 $\beta_t$）。偏好数据来自 Stage I 模型的生成结果，按物理准确度排序构造 win/lose 对。

**Stage III: Flow-DPO for PhysEncoder**

冻结 DiT（包括 LoRA），单独对 PhysEncoder 进行 Flow-DPO。此阶段让 PhysEncoder 的物理嵌入更精确地编码物理规律。

**为什么顺序优化优于联合优化**：联合 DPO 时两个模块的梯度相互干扰——DiT 的 DPO 梯度通过 cross-attention 影响 PhysEncoder 的输入分布，PhysEncoder 的 DPO 梯度改变 DiT 的条件输入。顺序优化避免了这种耦合。先 DiT 后 PhysEncoder 的顺序更好，因为先让 DiT 学会更好地利用物理嵌入，再让 PhysEncoder 针对优化后的 DiT 调整其输出。

### 通用物理场景

代理任务验证方法有效后，推广到通用场景：

- **WISA-80K 数据集**：80K 视频，覆盖 17 类物理事件（流体、碰撞、弹性、破碎等）
- **偏好数据构造**：用 Gemini 2.0 Flash 作为物理一致性评估器，对生成视频进行 pairwise 排序
- **训练流程**：同样三阶段，但偏好数据来自 VLM 评估而非合成 GT

## 实验

### 训练配置

- 基础模型：Wan2.1-I2V-14B
- PhysEncoder：DINOv2-Large（从 Depth Anything V2 初始化）
- DiT DPO：LoRA rank 128
- 硬件：8 × A800，SFT 20h，DPO-DiT 15h，DPO-PhysEncoder 8h

### 代理任务结果（Kubric 自由落体）

| 方法 | L2 ↓ | CD ↓ | IoU ↑ |
|------|------|------|-------|
| 基础模型 | 144.8 | 138.1 | 14.5 |
| + SFT | 122.3 | 116.7 | 18.2 |
| + SFT + DPO (DiT) | 108.5 | 103.2 | 22.1 |
| + SFT + DPO (DiT) + DPO (PhysEnc) | **95.7** | **91.4** | **26.8** |

每个阶段都有明确增益，DPO-PhysEncoder 阶段提供最后一步显著改进。

### 消融：DPO 优化策略

| DPO 策略 | L2 ↓ | CD ↓ | IoU ↑ |
|---------|------|------|-------|
| 联合 DPO (DiT + PhysEnc) | 112.3 | 107.1 | 20.5 |
| 先 PhysEnc 后 DiT | 103.8 | 98.6 | 24.1 |
| **先 DiT 后 PhysEnc** | **95.7** | **91.4** | **26.8** |

顺序优化显著优于联合优化，且先 DiT 后 PhysEncoder 的顺序最优。

### 通用场景结果（VIDEOPHY 基准）

| 方法 | PC | SA |
|------|-----|-----|
| Wan2.1-I2V-14B | 0.29 | 0.54 |
| + SFT (WISA-80K) | 0.34 | 0.61 |
| + Flow-DPO (DiT + PhysEnc) | **0.40** | **0.67** |

在通用物理场景中同样有效，PC 提升 +0.11，SA 提升 +0.13。

## 关键启示

- **物理知识应从生成模型中解耦**：专门的 PhysEncoder 比让 DiT 隐式学习物理规律更有效，且作为即插即用模块可迁移到不同 DiT
- **DPO 多模块优化应顺序而非联合**：避免梯度干扰，先优化主模型再优化条件编码器的顺序最优
- **代理任务是验证方法有效性的高效途径**：Kubric 自由落体提供精确 GT 和自动偏好标注，快速验证后再推广到通用场景
- **Depth Anything 是物理编码器的良好初始化**：深度信息与物理属性（重力、空间关系）天然相关
- **VLM 可作为通用场景的物理一致性评估器**：当 GT 不可用时，Gemini 2.0 Flash 提供可用的 pairwise 偏好信号
