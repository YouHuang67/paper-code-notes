---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - Flow Matching
---

# PRFL: Video Generation Models Are Good Latent Reward Models

- 论文：https://arxiv.org/abs/2511.21541
- 团队：UCAS, Tencent Hunyuan, PKU, SJTU, THU, NJU

## 概述

PRFL（Process Reward Feedback Learning）提出将预训练视频生成模型本身作为 latent 空间的 reward 模型，实现全去噪轨迹上的过程级监督。现有 RGB 空间的 reward 模型（基于 VLM）存在三个问题：(1) 需要近乎完全去噪 + VAE 解码，计算开销大；(2) 反向传播 VAE 解码器导致显存溢出；(3) 仅监督最终步骤，无法指导早期去噪阶段（运动和结构在此形成）。PRFL 发现视频生成模型的 DiT 特征在任意噪声级别和任意层都编码了运动质量信息，通过 query attention 聚合时空特征构建 Process-Aware Video Reward Model（PAVRM）。训练时随机采样 timestep，单步梯度反传即可优化，无需 VAE 解码。在 Wan2.1-14B 上，PRFL 在 dynamic degree 上提升高达 +56.00，human anatomy +21.52，同时比 RGB ReFL 节省显存且训练速度加快 1.4 倍以上。

## 动机

RGB 空间 ReFL 的三重瓶颈：
1. **评估延迟**：VLM reward 模型需要 RGB 输入，必须先近乎完全去噪再 VAE 解码
2. **显存瓶颈**：对所有视频帧反向传播 VAE 解码器经常导致 OOM
3. **监督不足**：reward 仅在最终步应用，无法直接指导早期生成阶段（运动和结构在此确定）

### 可行性分析：VGM 特征天然适合 reward 建模

三个关键发现：

- **VLM 在噪声输入上失效**：VideoAlign 在不同去噪 timestep 上评分剧烈波动，对高噪声区域泛化能力差
- **VGM 特征均匀编码运动信息**：用 MLP 探针测试 DiT 任意层（L8 到 L40）的特征，均达到 78.8% 准确率，匹配 VLM baseline。这意味着只用前 8 层就足够
- **timestep-aware 微调释放 VGM 潜力**：MLP 探针无论固定或随机 timestep 都无法超越 VLM baseline，但全量微调 + 随机 timestep 采样达到 85.46%（+6.6% 绝对增益），在早期噪声状态（t=0.8）性能最高

## 方法

### PAVRM：Process-Aware Video Reward Model

**架构**：复用预训练 VGM 的前 8 层 DiT block 作为时空特征提取器。给定噪声 latent $x_t \in \mathbb{R}^{F \times H \times W \times C}$ 和 timestep $t$：

$$h = \text{DiT}_\phi(x_t, t, \mathcal{T}(p)) \in \mathbb{R}^{F \times H \times W \times D}$$

**Query attention 聚合**：将时空特征展平为 $\hat{h} \in \mathbb{R}^{N \times D}$（$N = F \cdot H \cdot W$），用可学习 query $q \in \mathbb{R}^{1 \times D}$ 通过注意力机制压缩为固定大小 embedding：

$$z_{\text{obs}} = \text{softmax}(q(\hat{h}W_K)^T / \sqrt{D}) \cdot (\hat{h}W_V) \in \mathbb{R}^{1 \times D}$$

最终表征 $z = z_{\text{obs}} + q$，通过三层 MLP 映射为 reward 分数。残差连接 $+q$ 注入内容无关的质量先验，使模型学习质量相关模式而非内容相关性。

**训练**：二元偏好数据集，构造噪声 latent $x_t = (1-t)x_0 + tx_1$，$t \sim U(0,1)$ 随机采样，二元交叉熵损失：

$$\mathcal{L}_{\text{PAVRM}} = -\mathbb{E}_{t,(V,p,y)}[y\log\sigma(r_\phi) + (1-y)\log(1-\sigma(r_\phi))]$$

关键：$t \sim [0,1]$ 全范围随机采样，使 PAVRM 在整个去噪轨迹上学习阶段性质量评估。Rectified flow 的线性插值保证最终输出的偏好标签可自然传播到中间状态。

### PRFL 训练

两个损失交替训练：

**Process Reward Loss**：随机采样 timestep $s$，从 $t=1$ 无梯度去噪到 $s+\Delta t$，最后一步有梯度：

$$x_s = x_{s+\Delta t} - \Delta t \cdot v_\theta(x_{s+\Delta t}, s+\Delta t, p) \quad \text{(w/ grad)}$$

$$\mathcal{L}_{\text{PRFL}} = -\lambda \mathbb{E}_{s,p}[r_\phi(x_s, s, p)]$$

梯度仅通过最后一个去噪步和 PAVRM 反传，无需 VAE 解码。

**SFT 正则化**：标准 flow matching 损失，防止 reward 过度优化：

$$\mathcal{L}_{\text{SFT}} = \mathbb{E}_{t,(V,p)}\|v_\theta(x_t, t, p) - (x_1 - x_0)\|_2^2$$

## 实验

### 训练配置

- 基础模型：Wan2.1-I2V-14B / Wan2.1-T2V-14B
- PAVRM：前 8 层 DiT block + query attention + 3 层 MLP
- 偏好数据：31k 肖像视频，24k 标注对（运动质量二分类）
- 优化器：AdamW，PAVRM lr $10^{-5}$（query/head）/ $10^{-6}$（DiT），PRFL lr $5 \times 10^{-6}$
- 推理步数 40，训练步数 1000

### 主要结果（T2V, Wan2.1-T2V-14B）

| 方法 | Dynamic Degree | Human Anatomy | Motion Smooth. | Subject Consist. |
|------|---------------|---------------|----------------|-----------------|
| Pretrain (480P) | 22.00 | 84.24 | 99.20 | 97.34 |
| SFT | 44.00 | 92.79 | 98.96 | 96.61 |
| RWR | 60.00 | 91.85 | 98.99 | 95.93 |
| RGB ReFL | 38.00 | 91.68 | 99.20 | 92.26 |
| **PRFL** | **68.00** | **94.73** | 99.05 | 96.34 |
| Pretrain (720P) | 25.00 | 78.73 | 99.09 | 96.69 |
| **PRFL (720P)** | **81.00** | **90.89** | 98.85 | 96.09 |

PRFL 在 dynamic degree 上的提升最为显著（+46.00 / +56.00），同时保持 motion smoothness 和 subject consistency 基本不变。

### 效率对比

| 方法 | 显存 (GB) | 每步时间 (s) | 加速比 |
|------|----------|-------------|--------|
| RGB ReFL (全帧) | OOM | - | - |
| RGB ReFL (首帧) | 55.47 | 72.38 | 1.0× |
| PRFL (全帧) | 66.81 | 51.11 | 1.42× |

RGB ReFL 对全帧直接 OOM，只能退化为首帧优化；PRFL 处理全帧仍比 RGB ReFL 首帧更快。

### Timestep 采样策略消融

| 采样阶段 | Dynamic Degree | Human Anatomy | Avg |
|---------|---------------|---------------|-----|
| 早期（高噪声） | 51.00 | 87.52 | 82.25 |
| 中期 | 51.00 | 89.38 | 87.02 |
| 晚期（低噪声） | 44.00 | 91.54 | 85.01 |
| 全阶段 | **68.00** | **94.73** | **89.58** |

全阶段随机采样效果最好，验证了全轨迹过程级监督的必要性。早期阶段对 dynamic degree 贡献较大，晚期阶段对 human anatomy 贡献较大。

### 人类评估

PRFL vs 各基线（30 位评估者，25 个 prompt）：
- vs SFT：59.33% 胜率
- vs RWR：63.20% 胜率
- vs RGB ReFL：67.47% 胜率

## 关键启示

- **视频生成模型天然是良好的 latent reward 模型**：DiT 特征在任意层和任意 timestep 都编码运动质量信息，无需 VLM 式的 RGB 输入
- **过程级监督优于结果级监督**：全去噪轨迹随机采样 timestep 比仅监督最终步或特定阶段更有效，因为运动结构在早期形成，纹理在晚期形成
- **Query attention 是处理变长时空特征的高效方案**：用单个可学习 query 将 $F \times H \times W$ 的特征压缩为固定大小 embedding，且残差连接注入内容无关先验
- **Latent 空间 reward 避免 VAE 解码瓶颈**：RGB ReFL 对全帧视频直接 OOM，PRFL 处理全帧仍更快更省显存
- **前 8 层 DiT 特征就足够**：运动质量信息均匀分布在网络各层，无需深层特征
