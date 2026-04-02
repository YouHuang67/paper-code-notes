---
tags:
  - Video Generation
  - Reinforcement Learning
  - GRPO
  - Flow Matching
  - Diffusion Model
  - DPO
  - Reward Model
---

# Advances in GRPO for Generation Models: A Survey

- 论文：https://arxiv.org/abs/2603.06623
- 团队：SJTU, THU, CUHK

## 概述

首个系统综述 Flow-GRPO 及其后续 200+ 篇论文的发展。Flow-GRPO 将 GRPO 从 LLM 扩展到视觉生成，通过 ODE→SDE 转换引入随机性实现扩散/流匹配模型的 RL 对齐。综述沿两个维度组织：(1) 方法论进展——reward 信号设计、credit assignment、采样效率、多样性保持、reward hacking 缓解、ODE/SDE 策略、reward 模型构建；(2) 应用扩展——文生图、视频生成、图像编辑、语音音频、3D、具身 AI、统一多模态、自回归/掩码扩散、图像恢复。

## 背景

### Flow Matching 模型

从噪声分布 $p_0 = \mathcal{N}(0, I)$ 到数据分布 $p_1 = p_{\text{data}}$ 的连续时间变换，速度场 $v_\theta(x_t, t)$ 驱动 ODE：

$$\frac{dx_t}{dt} = v_\theta(x_t, t), \quad t \in [0, 1]$$

### GRPO 原理

无 critic 策略优化。对每个条件 $c$，采样 $G$ 个输出，组归一化优势：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

PPO-style clipped 策略梯度：$\mathcal{L} = -\mathbb{E}\left[\sum_{i=1}^G \min(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i)\right]$

### Flow-GRPO：ODE→SDE

将确定性 ODE 转为 SDE 引入随机探索：

$$dx_t = v_\theta(x_t, t)\,dt + \sigma(t)\,dW_t$$

去噪过程建模为 MDP，每步 log-likelihood：$\log \pi_\theta(a_t|s_t) = -\frac{\|a_t - v_\theta(x_t,t)\Delta t\|^2}{2\sigma^2(t)\Delta t} + \text{const}$

## 方法论进展

### 3.1 Reward 信号设计：从稀疏到密集

标准 Flow-GRPO 用终端稀疏 reward，所有步共享同一优势信号，导致信号稀释和探索低效。

| 方法 | 核心创新 | 关键结果 |
|------|---------|---------|
| **DenseGRPO** | ODE 预测干净样本做步级 reward gain $\Delta r_t = R(\hat{x}_1^{(t)}) - R(\hat{x}_1^{(t-1)})$ + reward 感知自适应随机性 | PickScore 23.1 (vs 22.5) |
| **SuperFlow** | 方差感知动态组大小 + 连续时间步级优势 | 1.7-16% 提升，仅需 5.4-56.3% 训练成本 |
| **VGPO** | 过程感知价值估计 $V_\phi(x_t, t)$ + 绝对值增强组归一化 | PickScore/HPS/GenEval SOTA |
| **Euphonium** | Reward 梯度注入 SDE 漂移项 + 双重 Reward GRPO | 1.66× 更快收敛，统一 Flow-GRPO 和 DanceGRPO |

### 3.2 Credit Assignment：从轨迹到步级

| 方法 | 核心机制 | 关键结果 |
|------|---------|---------|
| **TreeGRPO** | 去噪重构为搜索树，兄弟 reward 比较做步级归因 | 2.4× 训练加速 |
| **BranchGRPO** | 灵活按需分支，深度优势估计 | +16% 对齐，55% 训练时间削减 |
| **G2RPO** | 单步随机注入（其余 ODE）+ 多粒度优势整合 | 强因果链接 |
| **Chunk-GRPO** | 连续步分组为 chunk，chunk 级共享优势 | 低方差 credit 估计 |
| **PCPO** | 按策略梯度范数比例分配 credit | 长轨迹场景显著优于均匀分配 |
| **Multi-GRPO** | MCTS 式时间分组 + 多 reward 独立归一化加权 | 多目标稳定性 |

### 3.3 采样效率与训练加速

| 方法 | 核心机制 | 加速比 |
|------|---------|--------|
| **E-GRPO** | 高熵步识别，低熵步合并用 ODE | 改善信噪比 |
| **MixGRPO** | ODE-SDE 滑动窗口 | Flash 版 71% 时间削减 |
| **Smart-GRPO** | 基于 reward 反馈迭代优化初始噪声分布 | 避免低 reward 区域 |
| **DiffusionNFT** | 前向过程在线 RL，对比正负流匹配 | **25× 加速**，GenEval 0.24→0.98 |
| **AWM** | 优势加权流匹配 $\hat{A}_i \cdot \|v_\theta - \text{target}\|^2$ | **24× 加速** |
| **DGPO** | DPO-style 直接组偏好优化，确定性 ODE | ~20× 加速 |
| **GRPO-ELBO-ODE** | 分解设计空间，ELBO 似然估计主导 | 4.6× 效率提升 |

### 3.4 Mode Collapse 与多样性保持

| 方法 | 核心思路 |
|------|---------|
| **DiverseGRPO** | CLIP 空间谱聚类 + 稀有簇探索 reward + 早期步强 KL |
| **OSCAR** | 注入正交于生成流的随机扰动（训练免费） |
| **DRIFT** | Reward 集中采样 + 随机 prompt 扰动 + 势函数 reward shaping |
| **D2-Align** | 识别 reward 嵌入空间偏差方向并移除 |
| **DisCo** | 组合 reward + 面部相似度惩罚（98.6% Unique Face Acc） |

### 3.5 Reward Hacking 缓解

| 方法 | 核心思路 |
|------|---------|
| **GRPO-Guard** | 比率归一化 + 梯度重加权修复 PPO clipping 失效 |
| **GARDO** | 高不确定性输出惩罚 + EMA 参考更新 + 多样性 reward |
| **DDRL** | 前向 KL 正则化 $D_{KL}(p_{\text{data}} \| \pi_\theta)$ 锚定数据分布 |
| **ConsistentRFT** | 多粒度 rollout + 一致策略梯度（-49% 感知幻觉，-38% 语义幻觉） |
| **CPS** | 系数保持采样器消除过度随机性引入的伪影 |
| **ArtifactReward** | 首个系统分类：审美 hacking=过度风格化，一致性 hacking=物体变形 |

### 3.6 ODE vs SDE 采样策略

从纯 ODE（零噪声）到结构化 SDE 和摊销随机流映射的连续谱。关键发现：ELBO 似然估计是压倒性主导因素，在几乎所有策略梯度目标和采样方案组合下显著优于逐步 log-likelihood。

## 视频生成应用（§4.2）

综述将视频生成方法分为六类：

### Core T2V/I2V

- **TAGRPO**：记忆库对比 I2V 对齐
- **PhysRVG**：物理引擎 reward（刚体轨迹验证）
- **Diffusion-DRF**：冻结 VLM critic（训练免费）
- **Self-Paced GRPO**：能力感知课程 reward
- **PG-DPO**：自适应拒绝缩放稳定高维 DPO
- **DenseDPO**：段级偏好标注（1/3 标注数据）
- **McSc**：多维 self-critic 推理 + 运动校正 DPO

### 身份与一致性

- **Identity-GRPO**：时间感知身份 RM（+18.9%）
- **IPRO**：多角度身份特征池
- **ID-Crafter**：层次身份注意力 + 在线 RL
- **DreamID-V**：身份一致性 RL 换脸

### 运动控制

- **AR-Drag**：RL 增强的 few-step 轨迹控制
- **Camera-Controlled Video**：3D 几何验证 reward
- **MoGAN**：光流判别器（+7.3%）

### 视频编辑

- **VIVA**：Edit-GRPO（编辑精度 + 时间一致性）
- **ReViSE**：推理式语义分解编辑（+32%）

### 视频理解与预测

- **VANS (Video-as-Answer)**：Joint-GRPO 对齐 VLM+VDM
- **What Happens Next**：因果一致性 reward
- **RLIR**：逆动力学可验证 reward

### 前沿系统

- **Seedance 1.0/1.5**：多维 RLHF + 10× 推理加速 + 音视频联合生成
- **Self-Forcing++**：自回归超长视频（4min15s）
- **FSVideo/LongCat-Video**：高压缩/稀疏注意力大规模系统

## 关键启示

- **稀疏→密集 reward 是核心趋势**：终端稀疏 reward 导致信号稀释和探索低效，DenseGRPO/SuperFlow/VGPO/Euphonium 从不同角度实现步级信号，收益显著
- **训练加速的多条路径**：从 ODE-SDE 混合（MixGRPO）、噪声分布优化（Smart-GRPO）、前向过程训练（DiffusionNFT 25×）到完全去除策略梯度（DGPO 20×），加速空间巨大
- **Reward hacking 是系统性问题**：审美 hacking（过度风格化）和一致性 hacking（物体变形）有本质不同，需要针对性缓解策略
- **视频生成需要多维度联合对齐**：时间一致性、运动自然度、身份保持、物理合理性——单一 reward 无法覆盖，需要组合 reward + 课程学习 + 运动校正
- **ELBO > 逐步 log-likelihood**：设计空间分析表明似然估计器是最关键因子，ELBO 在几乎所有配置下主导
