---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - Flow Matching
  - GRPO
---

# CAMVERSE: Taming Camera-Controlled Video Generation with Verifiable Geometry Reward

- 论文：https://arxiv.org/abs/2512.02870
- 团队：AIsphere, NUS, NTU

## 概述

CAMVERSE 是首个用于相机控制视频生成的在线 RL 后训练框架。现有方法全依赖 SFT，无法利用 3D 大模型反馈纠正几何错误。CAMVERSE 提出可验证的几何 reward——将相机轨迹切分为非重叠段，用 3D 大模型（π3）估计外参并逐段计算 relative pose 对齐分，结合 GRPO 进行在线策略优化。在 T2V 和 I2V 上 translation error 分别降低 25.8% 和 15.1%，rotation error 降低 21.0% 和 16.5%。

## 动机

- SFT 被动拟合固定数据，无法从 3D 大模型的反馈中纠正几何错误
- 高维视频输出难以设计可验证的几何 reward
- 相机轨迹是连续信号，传统 instance-level reward（如 CLIP score）粒度太粗，信用分配稀疏低效

## 方法

### 相机条件编码

将外参 $(K_n, R_n, t_n)$ 转为 Plücker embedding $p_{u,v} = (o_n \times d_{u,v},\, d_{u,v}) \in \mathbb{R}^6$，每帧堆叠为 $P_n \in \mathbb{R}^{6 \times h \times w}$，通过轻量 camera network 注入 DiT。

### 可验证几何 Reward

1. **外参估计**：用 3D 大模型 π3 估计生成视频各帧外参 $\tilde{E}_n$，参考外参为 $\hat{E}_n$
2. **Umeyama 对齐**：解决尺度歧义，$(s^*, R^*, t^*) = \text{Umeyama}(\{\tilde{o}_n\}, \{\hat{o}_n\})$
3. **段级 Relative Pose 误差**：将轨迹切分为长度 $L$ 的非重叠段，段内计算 relative transform：

$$\tilde{T}_k = (\tilde{E}'_{n_k})^{-1} \tilde{E}'_{n_k+L}, \quad \hat{T}_k = (\hat{E}_{n_k})^{-1} \hat{E}_{n_k+L}$$

4. **对齐分**：$s_k = -(\lambda_t e_t(k) + \lambda_R e_R(k))$，用 confidence mask $m_k = \mathbf{1}[\omega_k \geq \tau]$ 过滤不可靠段

### GRPO 训练

将 diffusion model 视为 Gaussian 随机策略，每组采样 $G=16$ rollout，取 top-4 正样本 / bottom-4 负样本参与优化：

$$\max_\theta \frac{1}{GTK} \sum_{g,t,k} \left[f(r_t^g, A_k^g, \Delta) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

$A_k^g$ 为段级 z-score 归一化的 group-relative advantage。采样用前 3 步 SDE（noise=0.7）+ 后 11 步 ODE 共 14 步。LoRA rank=64, $\alpha$=128。

## 实验

### 训练配置

- 2B 参数 DiT，315k 视频数据集
- SFT：batch=128，10k iter，lr=5e-5
- GRPO：~3.2k 样本子集，200 iter，32×H200

### 主要结果（RealEstate10K）

| 方法 | T2V Trans↓ | T2V Rot↓ | I2V Trans↓ | I2V Rot↓ |
|------|-----------|---------|-----------|---------|
| AC3D-5B | 0.0428 | 0.9120 | — | — |
| CAMVERSE (SFT) | 0.0395 | 0.6506 | 0.0337 | 0.5613 |
| **CAMVERSE (RL)** | **0.0293** | **0.5140** | **0.0286** | **0.4685** |

### 消融

- Relative pose 误差显著优于 absolute pose 误差
- Dense segment-level reward vs clip-level reward：translation -14%，rotation -12%
- SDE 步数 3 最优（过少探索不足，过多准确率反降）
- 正负样本平衡（4-4）优于非对称组

## 关键启示

- **Relative pose 优于 absolute pose 作为 reward**：绝对误差偏向全局相似但局部错误的轨迹，relative segment 更捕捉局部连续性
- **Dense reward 解决连续控制的信用分配稀疏问题**：将轨迹切分为段逐段打分是粗粒度 reward 精细化的通用思路
- **SDE 步数是 RL 探索与利用的关键超参**：前几步 SDE 引入随机性探索，后续 ODE 确保质量，混合采样是 diffusion RL 的有效范式
- **LoRA + 极少量数据即可有效 RL 后训练**：200 iter、3.2k 样本带来显著提升，reward 设计质量比数据量更关键
- **3D 大模型作为可验证 reward 计算器**：可推广到深度估计、光流等任何可量化视觉信号
