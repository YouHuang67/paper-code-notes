---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - Flow Matching
  - GRPO
---

# VGGRPO: Towards World-Consistent Video Generation with 4D Latent Reward

- 论文：https://arxiv.org/abs/2603.26599
- 代码：https://zhaochongan.github.io/projects/VGGRPO
- 团队：Google, University of Copenhagen, University of Oxford, CREST-ENSAE

## 概述

VGGRPO（Visual Geometry GRPO）解决视频扩散模型生成中的几何一致性问题。现有方法要么通过修改架构（损害泛化能力），要么在 RGB 空间计算几何 reward（需反复 VAE 解码，计算开销大且仅支持静态场景）。VGGRPO 提出两个核心组件：(1) Latent Geometry Model（LGM），通过轻量级 3D 卷积连接层将视频扩散 VAE 的 latent 空间直接接入几何基础模型（Any4D），无需 RGB 解码即可从 latent 预测 4D 场景几何；(2) 基于 latent 空间的 GRPO 训练，使用两个互补 reward——相机运动平滑度 reward 和几何重投影一致性 reward。VGGRPO 在 Wan2.1-1B 和 Wan2.2-5B 上验证，在静态和动态场景基准上均大幅超越 Epipolar-DPO 和 VideoGPA，同时 latent reward 比 RGB reward 减少 24.5% 计算时间和 10.7% 显存。

## 动机

视频扩散模型虽然视觉保真度高，但常出现几何漂移、相机轨迹不稳定、场景结构不一致等问题，影响下游应用（具身 AI、物理仿真）。

现有两种改进范式的局限：
- **架构级集成**：注入点云条件、辅助几何预测模块等，增加复杂度且限制泛化能力，大多仅支持静态场景
- **后训练对齐**：Epipolar-DPO 使用极线约束，VideoGPA 使用稠密对应关系，但它们 (a) 依赖离线偏好数据（off-policy），(b) 在 RGB 空间计算 reward 需反复 VAE 解码，(c) 仅适用于静态场景

关键观察：几何基础模型（VGGT、Any4D）已能从前馈网络恢复稠密几何和相机运动，但它们需要 RGB 输入。能否直接在 latent 空间利用这些几何先验？

## 方法

### Latent Geometry Model（LGM）

核心思想：通过 model stitching 将视频 VAE 的 latent 空间连接到预训练几何基础模型，绕过 RGB 输入路径。

给定视频 VAE 编码器 $E$（将视频 $x$ 映射为 latent $z = E(x)$）和预训练几何模型 $\Phi = T_L \circ T_{L-1} \circ \cdots \circ T_1$（$L$ 层 transformer），LGM 用一个可学习的 3D 卷积连接器 $C_\phi$ 替换 $\Phi$ 的前 $\hat{\ell}$ 层：

$$\hat{\Phi}_\phi = \Phi_{\hat{\ell}+1:L} \circ C_\phi$$

训练分两步：

**Step 1：搜索最优拼接层 + 训练连接器**

在校准集上联合优化拼接层位置 $\hat{\ell}$ 和连接器参数 $\phi$，最小化特征对齐误差：

$$\hat{\ell}, \phi = \arg\min_{\ell, \phi} \frac{1}{M} \sum_{j=1}^{M} \|C_\phi(E(x_j)) - \Phi_{1:\ell}(x_j)\|_2^2$$

**Step 2：端到端微调**

微调连接器 $C_\phi$ 和下游层 $\Phi_{\hat{\ell}+1:L}$，最小化几何预测的对齐损失：

$$\mathcal{L}_{\text{align}}(\phi) = \sum_k \lambda_k \|\hat{\Phi}_{\phi,k}(E(x)) - \Phi_k(x)\|_1$$

其中 $k$ 索引不同几何模态（相机位姿 $C$、深度 $D$、点云 $P$、场景流 $F$）。

最终 LGM 可直接从 latent 预测 4D 几何：

$$\{C_i, D_i, P_i, F_i\}_{i=1}^N = \hat{\Phi}_\phi(z)$$

当基础几何模型选用 Any4D（支持 4D 动态重建）时，LGM 自然支持动态场景，突破先前方法仅限静态场景的限制。

### GRPO 训练

#### 相机运动平滑度 Reward

从 LGM 预测的相机位姿 $C_i$ 中提取世界坐标系下的相机中心 $c_i$，计算离散速度 $v_i = c_{i+1} - c_i$ 和加速度 $a_i = v_i - v_{i-1}$。

平移平滑度（尺度归一化加速度）：

$$S_{\text{trans}}(z_0) = \frac{1}{N-2} \sum_{i=2}^{N-1} \frac{\|a_i\|_2}{\|v_i\|_2 + \|v_{i-1}\|_2}$$

旋转平滑度类似，用角速度 $\omega_i = \log_{SO(3)}(R_i^\top R_{i+1})$ 和角加速度替换平移量。

组合运动 reward：

$$r_{\text{motion}}(z_0) = \frac{1}{2}\left(\frac{1}{1 + S_{\text{trans}}} + \frac{1}{1 + S_{\text{rot}}}\right)$$

映射到 $[0, 1]$，平滑轨迹接近 1，抖动轨迹趋近 0。

#### 几何重投影一致性 Reward

从 LGM 预测的点云 $P_i$、深度 $D_i$、相机参数 $C_i$ 和场景流 $F_i$，构建场景点云（静态场景聚合所有帧，动态场景用场景流过滤动态区域后仅聚合静态点），投影到每个视角得到渲染深度图 $\hat{D}_i$，与预测深度 $D_i$ 比较：

$$E_{\text{geo}}^{(i)}(z_0) = \frac{1}{|\Omega_i|} \sum_{p \in \Omega_i} |\hat{D}_i(p) - D_i(p)|$$

聚焦局部失败：取最差的 3 个视角的平均误差作为 reward：

$$r_{\text{geo}}(z_0) = -\frac{1}{3} \sum_{i \in \text{top-3}} E_{\text{geo}}^{(i)}(z_0)$$

#### Advantage 计算与策略更新

两个 reward 尺度不同，在 group 内分别归一化后取均值作为 advantage：

$$A_j = \frac{1}{2}\left(\frac{r_{\text{motion}}(z_0^j) - \mu_{\text{motion}}}{\sigma_{\text{motion}}} + \frac{r_{\text{geo}}(z_0^j) - \mu_{\text{geo}}}{\sigma_{\text{geo}}}\right)$$

代入标准 GRPO 的 clipped surrogate objective 进行策略更新，所有 reward 计算均在 latent 空间完成，无需 VAE 解码。

## 实验

### 训练配置

- 基础模型：Wan2.1-1B 和 Wan2.2-5B
- LGM 几何基础模型：Any4D（支持 4D 动态重建）
- LGM 训练数据：模型生成视频 + DL3DV / RealEstate10K / MiraData 真实视频，20 epochs
- VGGRPO：LoRA rank 32，scaling factor 64，group size 64，AdamW（lr $1 \times 10^{-4}$，weight decay $1 \times 10^{-4}$）
- 评估：190 静态场景（DL3DV + RealEstate10K）+ 200 动态场景（MiraData）

### 主要结果

**Wan2.1-1B**

| 方法 | VQ↑ (Static) | MQ↑ (Static) | Epi.↓ | VQ↑ (Dynamic) | MQ↑ (Dynamic) |
|------|-------------|-------------|-------|---------------|---------------|
| Base | - | - | 0.133 | - | - |
| SFT | 45.26 | 46.84 | 0.137 | 40.00 | 39.00 |
| Epipolar-DPO | 54.21 | 55.79 | 0.098 | 45.50 | 43.00 |
| VideoGPA | 53.68 | 56.32 | 0.105 | 42.50 | 41.00 |
| VGGRPO | **59.47** | **66.84** | 0.102 | **57.00** | **63.00** |

**Wan2.2-5B**

| 方法 | VQ↑ (Static) | MQ↑ (Static) | Epi.↓ | VQ↑ (Dynamic) | MQ↑ (Dynamic) |
|------|-------------|-------------|-------|---------------|---------------|
| Base | - | - | 0.142 | - | - |
| Epipolar-DPO | 52.11 | 58.95 | 0.101 | 38.00 | 54.50 |
| VideoGPA | 54.74 | 60.53 | 0.098 | 40.00 | 54.00 |
| VGGRPO | **62.63** | **68.42** | **0.093** | **56.50** | **66.00** |

VGGRPO 在动态场景上的优势尤其明显：MQ 比 VideoGPA 高 12+ 个百分点（Wan2.2-5B），因为先前方法在复杂非刚性运动下几何一致性显著退化。

### 消融实验

- **几何基础模型选择**：VGGT（仅静态）在极线误差上略优，Any4D（支持动态）在 VQ/MQ 上更高。两者均优于先前基线
- **Reward 组件**：$r_{\text{motion}}$ 单独使用改善相机稳定性但几何伪影依存；加入 $r_{\text{geo}}$ 后修复重投影不一致，两者互补
- **测试时 reward guidance**：LGM 可微，每 20 步做一次梯度引导即可提升几何一致性，无需训练，提供 training-free 的增强手段
- **泛化性**：在标准 VBench caption 上，VGGRPO 几乎所有指标优于基线，仅 Dynamic Degree 略低（因平滑相机轨迹降低了光流幅度）
- **效率**：latent reward vs RGB reward，计算时间减少 24.5%（41.33s vs 54.73s），显存减少 10.7%（68.57GB vs 76.80GB）

## 关键启示

- **Latent 空间可直接计算可靠的几何 reward**：通过 model stitching 将 VAE latent 接入几何基础模型，避免反复 VAE 解码，是高效 reward 计算的通用范式
- **Model stitching 是连接异构模型空间的轻量方案**：3D 卷积连接器 + 两阶段训练（特征对齐 → 端到端微调），可将任意几何基础模型适配到 latent 空间
- **4D 几何模型使动态场景对齐成为可能**：先前方法（Epipolar-DPO、VideoGPA）受限于静态场景假设，选择支持 4D 重建的几何基础模型（Any4D）自然扩展到动态场景
- **相机运动平滑度和几何重投影一致性是互补的优化目标**：前者消除抖动，后者确保跨视角几何连贯，单独使用任一都不够
- **GRPO 的 on-policy 优势**：相比 DPO 的离线偏好数据，GRPO 从当前策略采样并在 group 内归一化，reward 信号更新更及时
