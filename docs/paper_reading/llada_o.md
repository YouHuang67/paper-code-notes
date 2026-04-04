---
tags:
  - Diffusion Model
  - VLM
---

# LLaDA-o: An Effective and Length-Adaptive Omni Diffusion Model

[arXiv 2603.01068](https://arxiv.org/abs/2603.01068) | [GitHub](https://github.com/ML-GSAI/LLaDA-o) | Renmin University of China / Ant Group

## 概述

统一多模态扩散模型面临两个核心问题：（1）文本偏好离散掩码扩散，图像偏好连续潜空间扩散，两者联合训练易产生目标冲突和梯度干扰；（2）掩码扩散语言模型通常假设固定输出长度，不能灵活生成变长文本。

LLaDA-o 提出三项解决方案：
1. **Mixture of Diffusion（MoD）框架**：解耦两类扩散专家，通过共享注意力 backbone 实现跨模态交互，避免密集联合训练的冲突
2. **Intra-Modality Bidirectional Attention**：模态块内全量注意力 + 跨块因果注意力，固定条件的 KV 仅计算一次并缓存，推理速度 **5.9×** 快于全局双向注意力
3. **Adaptive Length Augmentation**：纯数据策略（无需改模型结构），在训练时随机扩充 [EOS] 或截断响应，激活变长推理能力

主要结果（omni 扩散模型中 SOTA）：
- DPG-Bench 图像生成：**87.04**（超过 Lumina-DiMOO 86.04、Show-o2 86.14、BAGEL 85.07）
- GenEval：0.86
- 10 项多模态理解 benchmark：扩散模型中 SOTA

## 方法

### Mixture of Diffusion（MoD）

两个专家共享同一注意力 backbone：

**理解专家（Understanding Expert）**
- 文本 + 视觉编码器 token → **masked diffusion**（离散扩散）
- 由 SigLIP 编码图像 → 2 层 MLP 投影到语言 token 空间 → LLaDA-8B-Instruct 语言模型
- 训练目标（掩码预测）：
$$\mathcal{L}_\text{und} = \int_0^1 \frac{1}{t}\mathbb{E}\left[\sum_{i: r_t^i = [\text{M}]} -\log p_\theta(r_0^i | v, p, r_t)\right] dt$$

**生成专家（Generation Expert）**
- 图像 latent token → **连续扩散**（Rectified Flow）
- 使用 Flux 的 VAE，DiT 架构与 LLaDA masked predictor 相同，时间步嵌入条件参数随机初始化
- 训练目标（flow matching 速度场）：
$$\mathcal{L}_\text{gen} = \mathbb{E}\left[\|(v_0 - \epsilon) - p_\theta(p, v_t, t)\|_2^2\right]$$

生成任务中的输入图像和文本同时也会过理解专家，参数联合训练。两个专家共享 attention backbone，确保跨模态信息交互。

### Intra-Modality Bidirectional Attention

**核心设计**：将输入序列划分为模态块（Image 1、Image 2、Prompt、Response...），块内做全量双向注意力，块间做因果（单向）注意力。

注意力模式（以理解任务为例）：
- 当前 Response block：可 attend 到所有前序 block（IMG1, IMG2, PRM, RES1...）+ 自身块内全量
- Prompt/Image 块：只能 attend 到自身及更早的固定 prefix

**推理优化**：固定条件 block（图像、prompt）的 KV 值只需计算一次，后续所有去噪步骤直接复用缓存——避免全局双向注意力每步重算所有 token 的冗余。

比较（MathVista 推理，与 LLaDA-V 的全局双向注意力相比）：
- LLaDA-o 在相同精度下实现 **5.9× 吞吐量提升**（203.9 vs ~35 tokens/s at same accuracy）
- confidence threshold=0.9：精度 65.9%，吞吐量 52.2 tokens/s

### Adaptive Length Augmentation（数据侧自适应长度）

掩码扩散模型通常需要在推理时预设固定输出长度，限制了开放性问答场景的应用。LLaDA-o 通过数据增强解决，不改模型结构：

**训练时（Algorithm 1）**：
- 以概率 $p_\text{ext}$：在响应末尾随机追加 $k$ 个 [EOS] token（暴露不同位置的终止信号）
- 以概率 $p_\text{trunc}$：将响应截断为随机前缀（鼓励从部分目标继续生成）

**推理时（block-wise generation）**：
1. 将固定条件（图像 + prompt）KV 缓存
2. 追加长度为 $L$ 的掩码块，去噪预测
3. 若 [EOS] 以高置信度（$r$）出现则停止，否则将已完成块写入 KV cache，继续下一块

块长度 $L$ 对生成长度影响小：从 32 增到 96，平均 token 数仅从 165 降到 145，精度从 63.6% 升到 66.2%。输出长度由输入内容决定，而非预设块大小。

### 三阶段训练

- Stage 1：大规模图像理解数据 + 图像生成数据（分辨率 ≤512）
- Stage 2：多模态推理数据 + Stage 1 高质量生成子集，分辨率提升至 1024
- Stage 3：开启 Adaptive Length Augmentation + 更多高质量图像生成数据

## 实验

### 文本生成图像（DPG-Bench，整体分数）

| 模型 | 参数 | Global | Entity | Attribute | Overall↑ |
|------|------|--------|--------|-----------|---------|
| Flux.1-dev | 12B | 74.35 | 90.00 | 88.96 | 83.84 |
| BAGEL | 7B | 88.94 | 90.37 | 91.29 | 85.07 |
| Show-o2 | 7B | 89.00 | 91.78 | 89.96 | 86.14 |
| Lumina-DiMOO | 8B | 81.46 | 92.08 | 88.98 | 86.04 |
| **LLaDA-o** | **8B** | **92.91** | **93.30** | **90.40** | **87.04** |

### 多模态理解（扩散模型 SOTA）

LLaDA-o（44.9/549/66.1/79.3/87.9/91.5）在 MathVista、ChartQA、DocVQA、InfoVQA 上明显超过同类扩散模型（LaViDa-O、Lumina-DiMOO、MMaDA）。

## 关键启示

- **MoD 解决模态冲突的核心是解耦而非割裂**：两个专家处理各自偏好的扩散过程，但共享 attention backbone 保证跨模态信息流通，避免密集联合训练中 state space 不一致导致的梯度干扰
- **块内双向 + 跨块因果是效率的关键设计**：固定条件（图像、prompt）只需一次前向即可缓存 KV，后续 $T$ 步去噪直接复用，本质上与 Cobra 中的 Causal Sparse Attention 思路相同——将静态条件的计算开销从 $O(T)$ 降至 $O(1)$
- **自适应长度是纯数据问题**：不需要模型架构改动，仅靠在训练时随机扩充 [EOS] 和截断，就能教会模型"知道什么时候该停"。推理时 block-wise 生成保持完整的 KV cache 复用，长度控制不以牺牲效率为代价
- **confidence threshold 提供精度-速度权衡**：threshold=0.2 时吞吐 203.9 tokens/s 但精度 57.8%，threshold=0.9 时 52.2 tokens/s 但精度 65.9%。与 AR 模型的固定生成策略不同，扩散模型可在推理时动态调节质量
