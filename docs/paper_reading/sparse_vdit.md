---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - Triton
  - CUDA
---

# Sparse-vDiT: Unleashing the Power of Sparse Attention to Accelerate Video Diffusion Transformers

[arXiv 2506.03065](https://arxiv.org/abs/2506.03065) | [代码](https://github.com/Peyton-Chen/Sparse-vDiT) | 复旦大学, StepFun, CUHK, Imperial College London

## 概述

视频扩散 Transformer（vDiT）中 3D 全注意力是推理的核心瓶颈——HunyuanVideo 120K token 下注意力占总延迟 81%，且比例随序列长度增长。Sparse-vDiT 系统分析了 vDiT 注意力图中的四种稀疏模式，并为每种模式设计了专用稀疏 kernel，通过离线搜索为每个 layer-head 分配最优模式，再融合同层同模式的 head 进一步加速。

核心发现：
- vDiT 注意力存在 **head 级冗余**（3-6% head 可跳过）和**注意力图级冗余**（diagonal / multi-diagonal / vertical-stripe 三种稀疏模式）
- 这些模式与**层深度和 head 位置**强相关，与输入内容几乎无关（t-SNE 可视化确认），因此可离线搜索固定

主要结果：
- CogVideoX1.5：2.09× 理论 FLOP 减少，**1.76×** 实际加速，PSNR 24.13
- HunyuanVideo：2.38× 理论 FLOP 减少，**1.85×** 实际加速，PSNR 27.09
- Wan2.1：1.67× 理论 FLOP 减少，**1.58×** 实际加速，PSNR 22.59

## 注意力冗余分析

### vDiT 注意力图结构

主流 vDiT（CogVideoX、HunyuanVideo）采用 MM-DiT 范式，token 序列由 text token（T）和 video token（V）拼接。V 占 99%+，注意力图的 V-V 区域中：
- 对角块 = self-frame 交互（同帧 token 间）
- 非对角块 = cross-frame 交互（跨帧 token 间）

### Head 跳过

按最小 MSE 准则评估 head 跳过：

| Model | Skip Ratio | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| CogVideoX1.5 | 1% | 36.62 | 0.96 | 0.01 |
| | 3% | 33.31 | 0.95 | 0.02 |
| | **6%** | **30.02** | **0.92** | **0.04** |
| | 10% | 26.87 | 0.85 | 0.09 |
| HunyuanVideo | 1% | 31.84 | 0.95 | 0.02 |
| | **3%** | **28.94** | **0.91** | **0.06** |
| | 6% | 24.21 | 0.81 | 0.12 |

CogVideoX1.5 可跳过 6% head 质量尚可，HunyuanVideo 可跳过 3%。但单靠跳过不足以大幅加速，需要更细粒度的策略。

### 四种注意力模式

1. **Full Attention**：值均匀分布，全局交互，无法稀疏化
2. **Diagonal Pattern**：大值集中在主对角线，对应同帧内邻近 token 交互。可用 window attention 加速
3. **Multi-Diagonal Pattern**：多条等间距对角线，对应跨帧相同空间位置的交互。通过 token 重排可转为 diagonal 结构
4. **Vertical-Stripe Pattern**：垂直条纹，表示存在全局 token 强烈关注所有其他 token。可用专用稀疏 kernel 加速

### 模式不变性

t-SNE 对 50 个 VBench prompt 的注意力图降维可视化：不同层的模式形成明显聚类，而同一层不同 prompt 的模式聚在一起。确认模式由 layer/head 位置决定，与输入无关。

## 方法

### Sparse Computation 预定义

五种计算模式 $M_0 \sim M_4$：

- $M_0$：Full Attention（$S_0 = 0$，无稀疏）
- $M_1$：Skip Head（$S_1 = 1$，输出置零）
- $M_2$：Diagonal Attention（window attention，仅计算主对角线附近块）
- $M_3$：Multi-Diagonal Attention（token 重排后的 window attention）
- $M_4$：Vertical-Stripe Attention（仅计算条纹位置的列）

每种模式的稀疏率 $S_i$ 预定义为固定常数。

### Offline Sparse Diffusion Search

对每个 layer 的每一步，将输入通过 $M_0 \sim M_4$，得到隐状态 $O_0 \sim O_4$。计算损失：

$$L_i = \text{MSE}(O_i - O_0) + \lambda \times (1 - S_i)$$

其中 $\lambda$ 平衡质量和计算成本。决策规则：

$$\text{Attention}(Q,K,V,M) = \begin{cases} M_0(Q,K,V) & \text{if } \bigwedge_{i=1,...,4} (L_i > \epsilon) \\ M_{\arg\min_i\{L_i\}}(Q,K,V) & \text{otherwise} \end{cases}$$

$\epsilon$ 控制整体稀疏率：$\epsilon$ 越大越激进。默认 $\lambda = 0.5$, $\epsilon = 1$。

由于模式与输入无关，搜索仅需少量样本即可离线完成，搜索后每个 head 的模式固定。

### Head Fusion

搜索确定后，同一层中使用相同稀疏模式的 head 可以**融合**为一次 kernel 调用。例如某层有 8 个 head 使用 diagonal attention，可以合并为一次 batched diagonal kernel 执行，减少 kernel launch 开销和内存访问次数。

### 稀疏 Kernel 实现

为三种稀疏模式（diagonal / multi-diagonal / vertical-stripe）设计了**预定义的 Triton/CUDA kernel**：

**Diagonal Attention Kernel**：
- 基于 window attention 实现，仅加载和计算主对角线附近的 KV 块
- 窗口大小由模式分析确定，预定义为固定值
- 对齐 FlashAttention 的 block 粒度，整块跳过非对角区域

**Multi-Diagonal Attention Kernel**：
- 先执行 token 重排（将跨帧同空间位置的 token 排列在一起）
- 重排后转化为 diagonal 结构，复用 diagonal kernel
- 重排操作可与前序操作融合

**Vertical-Stripe Attention Kernel**：
- 识别 "全局 token"（形成垂直条纹的列）
- 仅计算这些列对应的 KV 交互
- 适配为列选择 + 稠密计算的模式

所有 kernel 的稀疏率在搜索阶段确定后固定，推理时无需任何动态决策开销。

评测硬件：CogVideoX1.5 和 HunyuanVideo 在单卡 A800，Wan2.1 在单卡 H800。

## 实验

### 主要结果

| Model | Method | PSNR↑ | SSIM↑ | LPIPS↓ | PFLOPS↓ | Speedup↑ |
|---|---|---|---|---|---|---|
| **CogVideoX1.5** | Original | - | - | - | 147.87 | 1.00× |
| | MInference | 14.63 | 0.61 | 0.37 | 84.89 | 1.42× |
| | SVG | 21.92 | 0.75 | 0.22 | 74.57 | 1.64× |
| | **Sparse-vDiT** | **24.13** | **0.82** | **0.14** | **70.69** | **1.76×** |
| **HunyuanVideo** | Original | - | - | - | 612.37 | 1.00× |
| | SVG | 26.83 | 0.86 | 0.14 | 259.79 | 1.75× |
| | **Sparse-vDiT** | **27.09** | **0.87** | **0.12** | **257.09** | **1.85×** |
| **Wan2.1** | Original | - | - | - | 660.49 | 1.00× |
| | SVG | 21.96 | 0.78 | 0.18 | 403.50 | 1.49× |
| | **Sparse-vDiT** | **22.59** | **0.80** | **0.16** | **397.39** | **1.58×** |

Sparse-vDiT 在所有模型上的 PSNR / SSIM / LPIPS 和加速比均优于所有 baseline。CogVideoX1.5 上不需要前 10 步全注意力的 warmup（其他方法需要）。

### 消融：超参数影响

**λ（质量-效率平衡）**：固定 ε=1，λ=0.5 最优（PSNR 24.13，1.76× 加速）。λ=0 和 λ=0.1 质量略低，λ=1 效率略低。

**ε（整体稀疏率）**：固定 λ=0.5：

| ε | PSNR↑ | Speedup↑ |
|---|---|---|
| 0.5 | 25.49 | 1.68× |
| 1 (default) | 24.13 | 1.76× |
| 3 | 22.70 | 1.81× |
| 5 | 22.02 | 1.87× |
| 10 | 20.84 | 1.91× |

ε=5 时 PSNR 22.02 / 1.87× 加速，仍优于 SVG 的 21.92 / 1.64×。

### 局限性

预定义的稀疏 kernel 稀疏率固定，可能与实际注意力图的稀疏度不完全匹配，存在 under/over-sparsification 风险。自适应稀疏调整是未来改进方向。

## 关键启示

- **模式分类 + 专用 kernel** 是一种实用的稀疏加速路线：不追求通用的动态稀疏，而是识别少数固定模式，为每种模式写高效的专用 kernel，然后离线搜索分配
- **Head fusion** 是被忽视的优化机会：同层同模式的 head 合并执行可显著减少 kernel launch 开销，这在 head 数量多的模型中收益更大
- **模式的位置依赖性**（而非内容依赖性）使得离线搜索方案可行且鲁棒——只需几个样本就能确定整个推理流程的稀疏配置
- 实际加速（1.76-1.85×）与理论 FLOP 减少（2.09-2.38×）之间的 gap 主要来自非注意力模块（FFN 等）和 kernel overhead，未来与量化/缓存等正交技术结合可进一步缩小
