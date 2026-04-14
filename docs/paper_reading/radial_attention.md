---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
---

# Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation

[arXiv 2506.19852](https://arxiv.org/abs/2506.19852) | [代码](https://github.com/mit-han-lab/radial-attention) | MIT, NVIDIA, Princeton, UC Berkeley, Stanford (NeurIPS 2025)

## 概述

视频扩散模型中注意力的二次复杂度使长视频生成的训练和推理成本极高。Radial Attention 发现了**时空能量衰减**（Spatiotemporal Energy Decay）现象：post-softmax 注意力分数随 token 间空间和时间距离增大而指数衰减。基于此，设计了一个**静态** $O(n \log n)$ 稀疏注意力掩码，将能量衰减转化为计算密度衰减——近帧全计算，远帧指数缩减。

与 SVG 等动态方法不同，Radial Attention 使用统一的静态掩码，无需在线模式分类，因此同时适用于推理加速和训练加速。结合 LoRA 微调，可以高效地将预训练模型扩展到更长的视频。

主要结果：
- 默认长度推理：HunyuanVideo **1.9×** 加速，Wan2.1-14B **1.8×** 加速，质量与全注意力相当
- 4× 长度扩展（509 帧）：训练成本降低 **4.4×**，推理加速 **3.7×**，质量匹配甚至优于全注意力 LoRA 微调
- 注意力计算量减少最高 **9×**（509 帧 720p）

## 时空能量衰减

### 观察

对 HunyuanVideo 的 post-softmax 注意力图分析：
- **时间维度**（b1）：同一空间位置、不同帧的 token 之间，注意力分数随帧距离增大而衰减
- **空间维度**（b2）：同一帧内，注意力分数随空间距离增大而衰减

两种衰减均可良好拟合为指数函数 $y = \exp(-ax + b)$，$R^2 > 0.985$。

### 形式化

设视频潜变量由 $f$ 帧、每帧 $s$ 个 token 组成（总 $n = fs$）。对于第 $i_0$ 帧第 $k_0$ 位置的 query token，其 softmax 注意力分数 $p$ 满足：

$$p_{js+l} \le C_{\text{rel}} e^{-\alpha|j-i_0| - \beta|l-k_0|} p_{i_0 s + k_0}$$

- $\alpha$：时间衰减率。$\alpha$ 大 → 空间注意力（高时间衰减，低空间衰减）
- $\beta$：空间衰减率。$\beta$ 大 → 时间注意力（低时间衰减，高空间衰减）

这个统一模型将 SVG 中分离的 spatial/temporal 注意力头纳入同一框架。

## Radial Attention 掩码设计

### 时间密度衰减

沿时间维度，计算密度按指数衰减：帧 $i$ 到帧 $j$ 的计算密度为 $(\frac{1}{2})^{\lfloor \log_2(\max(|i-j|,1)) \rfloor}$。

注意力图被分为 $2\lceil \log_2(\max(f,2)) \rceil - 1$ 个对角 band：
- Band 0（中心）：100% 计算密度
- Band ±1, ±2, ...：每层密度减半，宽度加倍
- 效果：每个 band 的总计算量保持恒定

### 空间密度衰减

每个帧-帧注意力块内保留对角线结构（token 关注空间相似位置）。对角线宽度随时间距离衰减：

$$\text{diagonal width}(i,j) = \left\lfloor \frac{s}{2^{\lfloor \log_2 \max(|i-j|,1) \rfloor}} \right\rfloor$$

当对角线宽度降至 1 以下时，改为降低对角线频率（仅在满足特定模条件的帧对间保留对角线）。

### 形式化掩码

4D 掩码 $\tilde{M} \in \{-\infty, 0\}^{f \times f \times s \times s}$：

$$\tilde{M}_{i,j,k,l} = \begin{cases} 0 & \text{if } 2^{\lfloor \log_2 \max(|i-j|,1) \rfloor} \le s \text{ and } |k-l|+1 \le \frac{s}{2^{\lfloor \log_2 \max(|i-j|,1) \rfloor}} \\ 0 & \text{if } |i-j| \bmod \lceil \frac{2^{\lfloor \log_2 \max(|i-j|,1) \rfloor}}{s} \rceil = 0 \text{ and } k = l \\ -\infty & \text{otherwise} \end{cases}$$

额外添加 **attention sink**（所有 token 关注第一帧所有 token）。

### 与 SVG 的关系

Radial Attention 用单一掩码统一了 SVG 的 spatial/temporal 注意力：
- 中心 band（band 0）= SVG 的 spatial attention（稠密空间交互）
- 远距离 band = temporal attention，但按衰减规律分配计算，避免 SVG 对远帧的过度计算

### 复杂度分析

掩码中零元素数上界：

$$\#\text{zeros} \le 4s^2 f \underbrace{\vphantom{\sum}}_{\text{中心band+sink}} + \underbrace{\sum_{r=1}^{\lfloor \log_2 s \rfloor} \frac{2^{r+2} s^2 f}{2^r}}_{\text{宽度≥1的band}} + \underbrace{(\lfloor \log_2 f \rfloor - \lfloor \log_2 s \rfloor) \cdot 4s^2 f}_{\text{宽度<1的band}} \le 4s^2 f \log_2 f$$

即 $O(sn(\log_2 n - \log_2 s)) = O(n \log n)$（固定分辨率 $s$ 下）。

### 误差界

$$\|\tilde{p} - p\|_1 \le C_{\text{rel}} \left( \frac{8 e^{-\beta(s/2+1)}}{(1-e^{-\alpha})(1-e^{-\beta})} + \frac{4(1+e^{-\beta})}{(1-e^{-\beta})(1-e^{-\alpha})} e^{-\alpha(s+1)} \right) = O(C_{\text{rel}} e^{-\min(\beta/2, \alpha)s})$$

误差随衰减率增大指数减小。实测 Radial Attention MSE 3.9×10⁻³，低于 SVG 的 4.4×10⁻³ 和 STA 的 1.5×10⁻²。

### 硬件友好的块稀疏

注意力按 128×128 块计算（对齐 FlashAttention 块大小），而非逐 token 计算。

## LoRA 长视频适配

Radial Attention 保留了 softmax 注意力的关键 token 关系，预训练权重可大部分保持不变，只需轻量 LoRA 微调。对 Q/K/V/O 投影层添加 LoRA（rank=128）。

关键发现：Radial Attention + LoRA 不仅匹配全注意力 + LoRA 的质量，在长视频扩展时甚至**优于全注意力 + 全参数微调**。原因：LoRA 集中更新最关键的权重，Radial Attention 的结构化稀疏引导模型更有效地学习时间连贯性。

### 实现细节

- 推理：FlashInfer（block-sparse attention）
- 训练：Block-Sparse-Attention + FlashAttention-2 后端
- 前 12 步（25%）使用全注意力 warmup（默认长度）
- 前 2 层 DiT block 保持全注意力（训练时前 2 层，推理默认长度时第 1 层）
- 训练数据：OpenVid-1M 中筛选 2K 高质量视频
- 8× H100 训练，16-21 小时（HunyuanVideo），8-17 小时（Mochi 1）

## 实验

### 默认长度推理加速

| Model | Method | PSNR↑ | Vision Reward↑ | PFLOPs | Speedup |
|---|---|---|---|---|---|
| HunyuanVideo (117帧) | Original | - | 0.141 | 612 | - |
| | STA (FA3) | 26.7 | 0.132 | 331 | 2.29× |
| | SVG | 27.2 | 0.144 | 340 | 1.90× |
| | **Radial** | **27.3** | **0.139** | **339** | **1.88×** |
| Wan2.1-14B (69帧) | Original | - | 0.136 | 560 | - |
| | STA (FA3) | 22.9 | 0.132 | 322 | 2.01× |
| | SVG | 23.2 | 0.114 | 324 | 1.71× |
| | **Radial** | **23.9** | **0.128** | **323** | **1.77×** |

Radial Attention 在 PSNR/SSIM 上优于 STA 和 PowerAttention，匹配 SVG 质量。STA 用 FA3 后端速度更快但质量明显下降。

### 长视频扩展（4× 长度）

| Model | Method | Sparsity | Train Speedup | Infer Speedup | Vision Reward↑ |
|---|---|---|---|---|---|
| HunyuanVideo 509帧 | Full FT | 0% | 1.00× | 1.00× | 0.133 |
| | Spatial | 88.3% | 4.52× | 3.83× | 0.112 |
| | LongLoRA | 88.4% | 4.48× | 3.61× | 0.130 |
| | PA | 88.2% | 4.29× | 3.78× | 0.128 |
| | **Radial + LoRA** | **88.3%** | **4.37×** | **3.71×** | **0.134** |
| Mochi 1 667帧 | Full FT | 0% | 1.00× | 1.00× | 0.099 |
| | **Radial + LoRA** | **85.5%** | **2.83×** | **2.57×** | **0.113** |

Radial + LoRA 在 509 帧下 Vision Reward 0.134，甚至略超全注意力全参数微调的 0.133，同时训练成本降 4.37×，推理加速 3.71×。

### LoRA 兼容性

Radial Attention 的长度扩展 LoRA 可直接与已有风格 LoRA 权重合并，保持视觉质量的同时实现更长视频生成。

### 消融

- **Warmup 步数**（默认长度）：12 步（25%）最优，0 步 PSNR 仅 12.8
- **Warmup 步数**（4× 长度）：2 步最优
- **Dense 层数**（训练）：前 2 层全注意力最优
- **O(n log n) 模式**：vs Harmonic Series Decay（对角线宽度反比衰减），Radial 的指数衰减在所有指标上占优
- **LoRA vs Full FT**：Radial + LoRA 在所有长度上匹配或优于 Radial + Full FT

## 关键启示

- **能量衰减是 vDiT 注意力的内在物理规律**：指数拟合 $R^2 > 0.985$，这不是经验性 heuristic 而是可证明的结构特性，为稀疏掩码设计提供了物理直觉
- **静态掩码的训练兼容性**是 Radial Attention 的独特优势：动态方法（如 SVG）在训练时因在线分类误差被梯度放大而退化，静态掩码避免了这个问题
- **O(n log n) 在 O(n²) 和 O(n) 之间找到了实用的 sweet spot**：线性注意力改变了 softmax 机制需要大量重训练，Radial Attention 保持 softmax 只需 LoRA 微调
- **LoRA + 结构化稀疏的协同**：稀疏掩码限制了权重需要适应的注意力模式范围，使得 LoRA 的低秩更新更高效——这解释了为什么 Radial + LoRA 能超过 Dense + Full FT
- 当前实现使用 FA2，升级到 FA3 可获得与 STA 相当的额外加速（正交优化）
