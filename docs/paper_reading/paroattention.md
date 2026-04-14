---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
---

# PAROAttention: Pattern-Aware ReOrdering for Efficient Sparse and Quantized Attention in Visual Generation Models

[arXiv 2506.16054](https://arxiv.org/abs/2506.16054) | 代码已开源 | 清华大学, ByteDance Seed

## 概述

视觉生成模型中的注意力模式呈现出分散、不规则的特征（block-wise、multi-diagonal、diagonal-in-block 等），这使得现有稀疏化和量化方法在低密度 / 低比特下效果有限。PAROAttention 的核心思路不是设计更复杂的稀疏掩码来适配这些模式，而是**重排列 token 使注意力模式变得规则**，再用简单的块稀疏 + 块量化完成加速。

关键观察：视觉注意力的多种复杂模式本质上是沿不同维度（F/H/W）的局部聚集。通过对 3D token 的 6 种排列 $P_3^3$ 中选择最优排列，可以将这些模式统一为硬件友好的 block-wise 模式，从而同时降低稀疏化误差和量化误差。

主要结果：
- CogVideoX 720P 6s 视频：INT8 + 30% Dense → **5.7× 注意力加速**，**2.72× 端到端加速**
- Flux 1024 图像：INT4 + 75% Dense → **2.5× 注意力加速**，**1.94× 端到端加速**
- 所有配置下指标无损，生成结果与 FP16 全注意力几乎一致
- 20% dense rate 即可匹配其他方法 50% 的质量，加速从 1.4-1.7× 提升到 3.81×

## 核心问题

### 稀疏化困难

视觉注意力模式在类型（block-wise / multi-diagonal / diagonal-in-block）、结构参数（对角线数量/宽度/间距）、时间步和 prompt 之间都在变化。设计一种结构化稀疏模式来适配所有变化极其困难。分散的注意力分布使得很难形成完全稀疏的区域，结构化稀疏不可避免地引入误差。

### 量化困难

Q/K/V 分布本身变化不大，主要挑战在于 post-softmax 注意力矩阵 $P$ 的量化。在 "diagonal-like" 模式中，对角线上的大值在每个量化组内充当 "outlier"，使 scaling factor 过大，多数值被压缩到零附近，产生严重量化误差。

### 统一解法：Token 重排序

与其分别解决两个问题，不如**改变注意力分布本身**。重排序后：
- 稀疏化：块内全稀疏区域更多，结构化稀疏更精确
- 量化：块内值分布更均匀，incoherence 从 ~483（per-row）/ ~93（block-wise）降至 ~22（block-wise + reorder），量化误差显著减小

## 方法

### Pattern-Aware Token Reordering (PARO)

对 3D 视频 token [F, H, W]，搜索空间限定为 6 种排列：[F,H,W] / [F,W,H] / [H,F,W] / [H,W,F] / [W,F,H] / [W,H,F]。每个注意力头独立选择最优排列。

**排列指标**：

稀疏指标 $M_{\text{sparse}}$：对 post-softmax 注意力矩阵 $P$ 分块为 $b \times b$ 子矩阵，统计值 $< \epsilon$ 的比例超过 $\sigma$ 的块占比：

$$M_{\text{sparse}} = \frac{1}{k \times k} \sum_{i,j} \mathbb{I}\left(\frac{n_{ij}^{<\epsilon}}{b \times b} \ge \sigma\right)$$

量化指标 $M_{\text{quant}}$：使用 incoherence $\Psi = \max(|x|) / \text{mean}(|x|)$ 衡量量化难度，取所有块的平均值。

综合指标：对 6 种排列归一化后加权组合，选最低值：

$$M^{\Theta_i} = \alpha \cdot \frac{M_{\text{sparse}}^{\Theta_i}}{\sum_j M_{\text{sparse}}^{\Theta_j}} + (1-\alpha) \cdot \frac{M_{\text{quant}}^{\Theta_i}}{\sum_j M_{\text{quant}}^{\Theta_j}}$$

**开销控制**：
- 排列顺序在不同 timestep 和 prompt 间一致 → **离线确定**，无运行时开销
- 在线排列操作通过 fused CUDA kernel 实现：与前序 kernel（如 LayerNorm）融合，只需调整输出写入地址，开销 < 前序 kernel 的 1%

### Block-wise 稀疏注意力

重排后注意力图为规则 block-wise 模式，稀疏设计简化为：

- **静态掩码**（static，非 dynamic）：利用 post-softmax 的完整信息离线生成，避免在线预测的精度损失和运行时开销
- **Block-sum 阈值**：简单的块求和阈值即可，无需复杂的稀疏指标设计
- **Timestep-aware 掩码共享**：前半段去噪步使用逐步掩码（模式变化大），后半段共享同一掩码
- **块对齐**：稀疏粒度对齐 FlashAttention 块大小（64），可以直接跳过整个块，无需额外分支

存储开销：每个 head 的稀疏掩码以二值 bitmask 存储仅 9.2 KB，推理时预取即可。

### Block-wise 量化注意力

- **块对齐量化分组**：量化 grouping 必须与 FlashAttention 块大小对齐，per-row 分组不仅不兼容 block-wise 处理，还因对角结构产生高 incoherence
- **Token 重排降低 incoherence**：传统的 scaling / rotation 技术不适用于 FlashAttention 中的 $PV$ 计算（$P$ 不显式物化），重排序通过聚集相似注意力值来降低块内 incoherence
- 支持 $QK^T$ INT8/INT4 和 $PV$ INT8/INT4

### CUDA Kernel 实现

基于 **SageAttnV2 kernel** 定制实现，整合稀疏和量化功能：

- 稀疏部分：对齐 FlashAttention 块大小的 block-sparse，整块跳过的实现极其简单
- 量化部分：扩展 SageAttnV2 的量化能力，从 QK (INT4) + PV (FP8) 扩展到 QK + PV 均支持 INT8/INT4
- 排列融合：与 LayerNorm 等前序 kernel 融合，排列只需修改写入地址，几乎零开销
- 在 A100（稀疏对比）和 RTX 4090（量化对比，需 FP8/INT4 支持）上评测

运行时开销对比：
- PAROAttn: < 1%
- SpargeAttn: 6-9%
- SparseVideoGen: 10-15%

## 实验

### CogVideoX 视频生成

| Method | Dense Rate / Bitwidth | PSNR↑ | VQA↑ | CLIPSIM↑ |
|---|---|---|---|---|
| FP16 Full Attn. | 100% | ∞ | 92.53 | 0.203 |
| SpargeAttn | 50% | 16.80 | 87.72 | 0.198 |
| SparseVideoGen | 50% | 18.50 | 90.14 | 0.198 |
| **PAROAttn** | **50%** | **29.14** | **92.56** | **0.203** |
| **PAROAttn** | **30%** | **22.89** | **92.66** | **0.204** |
| PAROAttn | 20% | 19.39 | 92.42 | 0.203 |
| SageAttn | QK INT8, PV FP16 | 29.58 | 92.24 | 0.203 |
| **PAROAttn (INT8)** | QK INT8, PV INT8 | **29.01** | **92.57** | **0.203** |
| SageAttnV2 | QK INT4, PV FP8 | 24.46 | 88.79 | 0.200 |
| **PAROAttn (INT4)** | QK INT4, PV INT4 | **24.16** | **89.24** | **0.200** |
| PAROAttn (0.3+INT8) | 30% + INT8 | 21.49 | 91.68 | 0.201 |
| PAROAttn (0.5+INT4) | 50% + INT4 | 24.34 | 90.42 | 0.200 |

50% dense rate 下 PAROAttn PSNR 29.14 vs SparseVideoGen 18.50，质量差距巨大。30% dense rate 的 PAROAttn 仍然优于 50% 的所有 baseline。

### 加速效率

延迟接近理论上限：
- 50% density → 1.73× 加速（理论极限 2×）
- 30% density → 2.71× 加速
- 20% density → 3.81× 加速

稀疏 + 量化组合的最激进配置（0.5+INT4）相比 baseline 的 1.5-2× 加速达到近 **10× 注意力加速**。

### 消融分析

- 去除 token reorder：稀疏和量化指标均显著恶化
- 去除 timestep sharing（所有步独立掩码）：性能并未提升，说明后期步可共享掩码
- 量化分组改为 row-wise：性能显著下降

排列可视化：6 种排列中，选中的 [H,W,F] 同时展现出 block-sparse 和 block-uniform 特征，而 [H,F,W] 虽然块内均匀但缺乏稀疏性。

## 关键启示

- **"改变分布"优于"适配分布"**：与其设计越来越复杂的稀疏掩码适配不规则模式，不如通过 token 重排序将问题本身简化。这个思路在稀疏化和量化两个维度同时有效
- **排列空间的可控性**：3D token 的 6 种排列提供了足够的表达力来统一多种模式，同时搜索空间极小（6 种而非 $N!$ 种），离线确定不产生运行时开销
- **稀疏 + 量化协同设计**：PARO 统一了两者的优化方向（降低块内 incoherence），使得激进配置（如 30% density + INT8）成为可能
- 基于 SageAttnV2 的 CUDA kernel 实现展示了如何在现有高效注意力 kernel 上叠加稀疏功能，排列操作与前序 kernel 融合后几乎零开销
