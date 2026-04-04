---
tags:
  - Sparse Attention
  - Diffusion Model
---

# HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation

[arXiv 2506.04421](https://arxiv.org/abs/2506.04421) | Stanford / NVIDIA / CUHK / HKU / NTU / UCSD

## 概述

Visual Auto-Regressive (VAR) 将图像生成表述为逐尺度预测（coarse-to-fine），但存在三个问题：（1）每个尺度内所有 token 并行生成，隐含条件独立假设，质量下降；（2）每步需缓存所有前置尺度的 KV，序列长度随分辨率超线性增长，推理内存大（1024×1024 下 OOM）；（3）采样步数在训练时固定，无法灵活调整。

HMAR 提出三项改进：
1. **Markovian 重表述**：将每步 conditioning 改为仅依赖**前一尺度**的累积重建 $\tilde{x}_{1:k-1}$（等价于所有前序尺度），注意力模式从 block-causal 变为 block-diagonal，稀疏度大幅提升（256×256 下约 5×）
2. **定制 block-diagonal 稀疏注意力 kernel**：Triton 实现 IO-aware 窗口注意力，attention 计算 **>10×** 加速；端到端训练 **2.5×** 加速（1024×1024）；推理 **1.75×** 快，内存 **3×** 小
3. **多步掩码生成（MaskGIT 风格）**：每个尺度内用 $M_k$ 步迭代掩码预测，而非一步并行生成，消除条件独立假设，提升质量；$M_k$ 推理时可调，无需重训练

ImageNet 256×256 结果：HMAR-d30（2.4B）FID 1.95，IS 334.5，与 VAR-d30（2.0B，FID 1.95，IS 303.6）FID 持平，IS 提升 31 分。

## 方法

### VAR 的问题回顾

VAR 生成过程：

$$p(\mathbf{x}) = \prod_{k=1}^K p(r_k | r_1, \ldots, r_{k-1})$$

推理时每步 attending 到所有前序尺度 token，形成 block-causal 掩码（下三角块结构）。问题：
- 序列长度 = 所有尺度 token 数之和，256×256 下约比 next-token 方案长 5.84×
- FlashAttention 不支持 block-causal 掩码，无专有 kernel 加速
- 推理需 KV-cache 保存所有前序尺度，内存随分辨率快速增长

### Markovian 重表述

关键观察：每个尺度 $r_k$ 是残差量化结果，而累积重建 $\tilde{x}_{1:k-1}$ 已编码了前 $k-1$ 尺度的所有信息。因此：

$$p(r_k | r_1, \ldots, r_{k-1}) = p(r_k | \tilde{x}_{1:k-1})$$

条件由"所有前序尺度"替换为"前一步累积重建"，**预测目标不变**（仍预测残差 $r_k$），仅改变 conditioning 来源。

注意力模式变化：从 block-causal（下三角块）→ block-diagonal（仅当前尺度 token 与前一步重建 token 之间的注意力块），稀疏度显著提升。推理时无需 KV-cache 前序尺度，内存从线性增长降为常数。

实证验证（Fig. 9）：VAR 中各 token 对前序尺度的注意力集中在直接前驱尺度，后续尺度的贡献微弱，支持 Markovian 假设。

### IO-Aware Block-Diagonal 稀疏注意力 Kernel

FlexAttention 支持多种掩码但要求序列长度为 128 的倍数，无法直接使用。HMAR 用 Triton 自定义 kernel，继承 FlashAttention 的 IO-aware 分块计算，在 block-diagonal 掩码下跳过非活跃块，实现 >10× attention 加速（相比 VAR 的密集注意力）。

### 多步掩码生成（Intra-Scale Masked Prediction）

VAR 在每个尺度内并行生成所有 token，等价于条件独立假设。HMAR 引入 $M_k$ 步迭代掩码过程：

$$p(r_k | r_{<k}) = \prod_{m=1}^{M_k} p(r_k^m | r_k^1, \ldots, r_k^{m-1}, r_k^0, r_{<k})$$

其中 $r_k^0$ 是 next-scale 模块的初始预测，后续每步掩盖部分 token 并重新预测（MaskGIT 风格）。

- $M_k = 0$：退化为 VAR 的并行生成
- $M_k = H_k \times W_k$：退化为 next-token 逐一生成
- 推理时可不重训练地调整 $M_k$，粗尺度增加步数改善 FID，细尺度增加步数改善感知质量

**训练**：两阶段——先训练 next-scale prediction 模块（block-diagonal 掩码），再 finetune 掩码预测头（随机 mask γ∈[0,1]），两阶段共享主干权重。

### 尺度感知 Loss 加权

VAR 对所有 token 均等加权，但精细尺度 token 数是粗尺度的 256 倍，模型偏向精细尺度。HMAR 对每个尺度 $k$ 赋权 $w(k)$：

$$\mathcal{L}_{\text{train}} = \sum_{k=1}^K w(k) \sum_{(i,j)} \mathcal{L}(r_k^{(i,j)})$$

实验发现 log-normal 加权（与尺度学习难度分布吻合）效果最佳。消融结果：loss 加权使 FID 从 3.76 → 3.42，IS 从 293.3 → 307.9。

## 实验

### ImageNet 256×256（class-conditional）

| 方法 | FID↓ | IS↑ | 参数 | 步数 |
|------|------|-----|------|------|
| DiT-XL/2 | 2.27 | 278.2 | 675M | 250 |
| VAR-d24 | 2.15 | 312.4 | 1.0B | 10 |
| VAR-d30 | 1.95 | 303.6 | 2.0B | 10 |
| HMAR-d24 | **2.10** | **324.3** | 1.3B | 14 |
| HMAR-d30 | **1.95** | **334.5** | 2.4B | 14 |

HMAR IS 比 VAR 高 ~31 分（更高样本多样性），FID 持平或更好。

### 效率（A100，d-24 模型）

| 指标 | 256×256 | 512×512 | 1024×1024 |
|------|---------|---------|-----------|
| 推理加速（vs VAR） | ~1× | 1.3× | **1.75×** |
| 训练 FWD 加速 | ~1× | 1.5× | **2.5×** |
| 内存 | ~1× | 1.5× | **3×（VAR OOM）** |

效率优势随分辨率增长加剧，1024×1024 下 VAR OOM 而 HMAR 正常运行。

### 消融（HMAR-d16，ImageNet 256×256）

| 配置 | FID↓ | IS↑ |
|------|------|-----|
| VAR-d16 (baseline) | 3.50 | 276.0 |
| + Markov 假设 | 3.76 | 293.3 |
| + Loss 加权 | 3.42 | 307.9 |
| + Masked Prediction | **3.01** | **288.6** |

三项改进均有贡献，掩码预测对 FID 提升最大（3.42→3.01）。

## 关键启示

- **Markovian 重表述是核心工程突破**：从数学上证明仅条件化于前一步累积重建等价于条件化于所有前序尺度，使注意力模式从 block-causal 变为 block-diagonal，同时消除推理时的 KV-cache 需求——一个数学等价性直接换来了效率和内存的大幅改善
- **Block-diagonal 掩码需要定制 kernel**：FlashAttention 不支持此模式，Triton 自定义 IO-aware kernel 将 attention 计算加速 >10×，是端到端 2.5× 训练加速的关键
- **MaskGIT 式掩码预测解决 VAR 的条件独立假设**：逐尺度并行生成本质上是一个错误的独立性假设，引入迭代掩码在质量和速度间取得平衡，且 $M_k$ 可在推理时调整无需重训练
- **Log-normal 尺度加权对齐学习难度**：粗尺度 token 少但全局结构重要，细尺度 token 多但容易学习；loss 难度分布近似 log-normal，对应加权使模型更关注结构性尺度
