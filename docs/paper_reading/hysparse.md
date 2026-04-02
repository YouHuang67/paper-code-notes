---
tags:
  - Sparse Attention
  - LLM Inference
  - KV Cache
---

# HySparse: Hybrid Sparse Attention with Oracle Token Selection and KV Cache Sharing

- 论文：arXiv 2602.03560
- 团队：Xiaomi
- 代码：未开源

## 概述

HySparse 提出一种混合稀疏注意力架构，将每个 full attention 层与多个 sparse attention 层交替排列。核心创新在于 sparse 层的 token 选择和 KV cache 均直接来源于前一个 full attention 层，从而解决现有稀疏注意力的两个根本问题：(1) 传统方法依赖代理模块（启发式规则、近似估计、额外选择网络）来预测 token 重要性，本质上是近似的且引入额外复杂度；(2) 动态稀疏注意力虽减少计算量，但通常仍需保留完整 KV cache，内存无法受益。HySparse 用 full attention 作为精确的 oracle 来确定重要 token，并让 sparse 层复用 full attention 的 KV cache，同时实现计算和内存的双重节省。在 7B dense（1:3 比例）和 80B MoE（1:11 比例）模型上，HySparse 在通用基准和长上下文任务中均一致优于 full attention 和 hybrid SWA 基线。80B MoE 模型中 49 层仅 5 层使用 full attention，KV cache 减少近 10 倍。

## 背景与动机

### 稀疏注意力的分类

- **Training-free**：固定模式或启发式规则在推理时选择重要 token（StreamingLLM、H2O、Quest、MInference 等），无需训练改动，但存在训练-推理不匹配，长解码或多步推理中误差会累积
- **Trainable**：训练中学习 token 选择，通过轻量选择模块实现。两种方式：
  - 辅助损失（自蒸馏）对齐选择模块与原始 dense attention（SeerAttention、DSA），简单但次优
  - 端到端稀疏预训练（NSA），将压缩注意力输出注入主注意力，选择模块仅通过最终输出间接获得学习信号

### 混合注意力架构

已有工作将不同注意力机制混合使用以减少计算和 KV cache：

- 线性注意力 + softmax 注意力：MiniMax-01
- Gated DeltaNet 混合：Qwen3-Next、Kimi Linear
- Mamba + self-attention：Nemotron、Jamba
- SWA + full attention 交替：GPT-OSS、Gemma3、MiMo-V2-Flash（窗口最小至 128 tokens）

但混合 **动态稀疏注意力** 的架构尚未被充分探索，这正是 HySparse 的切入点。

### 两个关键观察

1. **跨层 salient token 稳定性**：多项研究发现注意力分数高的 token 在连续层间保持相对稳定（TidalDecode、DELTA、Kascade 等），training-free 方法已利用此特性加速推理。HySparse 将此观察提升为预训练架构设计
2. **跨层 KV cache 共享**：YOCO、CLA、Apple Foundation Model、Gemma 3n 等在架构中集成 KV 共享，SwiftKV 通过蒸馏适配已有模型，MiniCache 利用中深层 KV 的高相似性进行压缩。经验表明 KV 共享对精度影响极小

## 方法

### 整体架构

HySparse 将标准 Transformer 替换为重复的混合块，每个块由 1 个 full attention 层 + $M$ 个连续 sparse attention 层组成。核心机制：

- **Oracle token 选择**：full attention 层计算标准自注意力的同时，输出 block-wise 注意力重要性分数 $R$，通过 TopK 操作提取 block 索引 $\mathcal{I}$，供后续 $M$ 个 sparse 层复用
- **KV cache 共享**：sparse 层的 block sparse attention 分支直接复用前一个 full attention 层的 KV cache，无需额外存储
- **SWA 分支**：每个 sparse 层额外维护一个小窗口（$w=128$）的滑动窗口注意力分支，使用独立的 KV cache 增强短程建模能力
- **Sigmoid 门控融合**：两个分支的输出通过可学习的 sigmoid 门动态融合

### Full Attention 层

基于标准 scaled dot-product attention，关键改动是输出 block 级别最大注意力分数。直接物化完整注意力矩阵在内存和带宽上不可行（因为实际使用 FlashAttention），因此 HySparse 仅物化 block 级最大注意力分数。

对于每个 query token $t$，定义 block 索引 $i$ 对应的 column token 集合 $\mathcal{B}_i = \{(i-1)B+1, \ldots, \min(iB, M)\}$，block 级最大注意力分数为：

$$S_t^i = \max_{i' \in \mathcal{B}_i} \frac{\exp(q_t^\top k_{i'} / \sqrt{d})}{\sum_{j=1}^{t} \exp(q_t^\top k_j / \sqrt{d})}$$

实现方式：对 FlashAttention kernel 的微小修改。FlashAttention 在 online softmax 过程中已经计算了 attention logits 的 row-wise maximum，这个中间结果可以被存储并适当 rescale 来得到 block-wise attention scores（开销可忽略）。具体流程（假设 sparse block size $B$ = FlashAttention tile size $B_N$）：

1. 标准 FlashAttention 外循环迭代 Q blocks，内循环迭代 KV blocks
2. 内循环中：计算 $A_{ij} = Q_i K_j^\top \cdot \tau$，取 $\tilde{m}_{ij} = \text{rowmax}(A_{ij})$ 并存入 $S_{ij}$
3. 更新 running max $m_i$、output $O_i$ 和 normalization $\ell_i$（标准 FlashAttention 流程）
4. 外循环结束后，对存储的 $S_{ij}$ 进行后处理：$S_{ij} \leftarrow (S_{ij} - m_i) / \ell_i$，转化为 softmax 归一化后的 block-wise max 分数

通过 block-wise 分数 $R$，使用 TopK 操作选取 block 索引 $\mathcal{I}$。默认参数：稀疏 token 数 $n=1024$、block size $B=64$（即 TopK 选取 $1024/64=16$ 个 block）。在 GQA 下，对同一 KV group 内的 query head 取 group-wise maximum 聚合 $R$，使同组 head 共享相同稀疏索引，提升 kernel 效率。

### Sparse Attention 层

每个 sparse 层包含两个并行分支：

**Block Sparse Attention 分支**：

- 使用从前一个 full attention 层继承的 block 索引 $\mathcal{I}$ 和 KV cache
- 将选中的 K、V blocks 拼接：$\tilde{K}, \tilde{V} = \text{concat}(\{K/V[(i-1)B+1:iB]\}_{i \in \mathcal{I}})$
- 计算标准 softmax 注意力，但仅在选中的 $n$ 个 token 上：$\tilde{o}_t = \text{softmax}(q_t'^\top \tilde{K} / \sqrt{d}) \tilde{V}$

**SWA 分支**：

- 使用独立的 QKV 投影层（与 sparse 分支共享 Q 投影）
- 窗口大小 $w=128$，维护自己的小 KV cache
- 标准滑动窗口注意力：$o'_t = \text{softmax}(q_t'^\top K'_{[t-w+1:t]} / \sqrt{d}) V'_{[t-w+1:t]}$

**门控融合**：

$$\tilde{g}_t, g'_t = \sigma(W_{\tilde{g}/g'} x_t)$$
$$o_t = \tilde{g}_t \odot \tilde{o}_t + g'_t \odot o'_t$$

SWA 分支维护独立 KV cache 是必要的——SWA 主要作为局部信息通路，需要不同的表示来捕获短程连贯性，而 full attention 共享的 KV cache 针对全局检索优化，缺乏局部特征。

## 实验

### 模型配置

| 配置 | 7B Dense | 80B MoE |
|---|---|---|
| 层数 | 36 | 49 |
| Q / KV heads | 32 / 8 | 64 / 4 |
| Head dim | 128 | 128 |
| Hidden size | 4096 | 2048 |
| Hybrid ratio (Full : Sparse) | 1 : 3 | 1 : 11 |
| SWA 窗口 | 128 | 128 |
| Sparse block size | 64 | 64 |
| Sparse TopK tokens | 1024 | 1024 |
| MoE experts (activated / total) | – | 8 / 512 |

所有混合模型最后一层使用 full attention。sparse 和 SWA 层均引入 per-head learnable sink biases（参考 GPT-OSS）。

### 训练设置

- **7B**：1T tokens，seq_len=8192，AdamW（$\beta_1=0.9, \beta_2=0.95$），WSD schedule，max_lr=$8.3 \times 10^{-4}$，BF16。长上下文扩展：200B tokens，seq_len=32768，lr=$3 \times 10^{-5}$，RoPE base=640000
- **80B MoE**：500B tokens，seq_len=32768，WSD schedule，max_lr=$1 \times 10^{-3}$，RoPE base=640000

### 通用基准结果

对比三种架构：Full-Attn、Hybrid SWA、HySparse。

**7B Dense（1:3 ratio）**：

- HySparse 在知识和推理基准上超越 Full-Attn：MMLU 58.8 vs 56.9，MMLU-Redux 61.6 vs 59.6，MMLU-Pro 29.0 vs 26.8
- 数学推理一致提升：GSM8K 37.9 vs 33.3（Full-Attn）/ 35.6（Hybrid SWA），MATH 10.1 vs 9.2
- 中文理解提升：C-Eval 52.2 vs 50.6，CMMLU 54.5 vs 52.5
- 常识/阅读理解任务持平或微优

**80B MoE（1:11 ratio）**：

- HySparse 几乎在所有基准上同时超越 Full-Attn 和 Hybrid SWA
- Hybrid SWA 在此激进比例下出现明显退化（BBH 48.2、DROP 47.8、MMLU 54.9，均显著低于 Full-Attn）
- HySparse 恢复甚至超越了 Full-Attn 性能（BBH 56.3 vs 56.1，MMLU 62.2 vs 61.8，GSM8K 54.1 vs 53.8），同时 KV cache 减少 10 倍
- 仅 MMLU-Pro（32.6 vs 33.8）、DROP（56.5 vs 56.7）、ARC-C（77.6 vs 78.4）略低于 Full-Attn

核心发现：当混合比例更激进时，单纯依赖 SWA 局部窗口不够用，sparse attention 分支通过恢复对全局相关 token 的访问弥补了这一缺口。

### 长上下文基准（RULER）

在 16K 和 32K 序列长度上评估：

- **7B**：HySparse 在两个长度上整体分数均最高（16K: 94.1 vs 93.0/91.6，32K: 89.3 vs 88.2/84.2），尤其在困难子任务 CWE 上大幅领先（16K: 60.8 vs 37.1/23.7，32K: 38.8 vs 16.6/10.4）
- **80B**：Hybrid SWA 在激进比例下严重退化（16K: 72.7，32K: 69.5），HySparse 保持接近 Full-Attn 水平（16K: 90.6 vs 93.6），32K 下甚至超越 Full-Attn（87.4 vs 82.1），MK3 恢复尤为显著（98.4 vs 77.0）

### 消融实验（7B Dense）

**消融 1：是否需要 intra-layer SWA 分支？**

即使 sparse token 由 oracle 精确选择，去除 SWA 分支仍导致明显退化：DROP 46.4→52.2（+5.8），GSM8K 29.7→37.7（+8.0），BBH 48.2→52.4（+4.2）。说明专用的局部建模通路不可替代，oracle 选择的全局 token 无法完全覆盖短程连贯性需求。SWA 还可能通过提供稳定的局部路径帮助早期训练优化。

**消融 2：KV cache 共享配置**

两种方案对比：(a) SA 和 SWA 分支都共享 KV cache vs (b) 仅 SA 分支共享。

全共享方案性能大幅下降（MMLU 52.8、BBH 47.2），仅 SA 共享则恢复并提升（MMLU 58.4 +5.6、MMLU-Pro 29.0 +5.8、BBH 53.9 +6.7）。原因：SA 可安全复用 full attention 的 KV cache（两者都面向全局检索），但 SWA 需要独立的局部特征表示，强制共享会剥夺其短程局部特征能力。

## 讨论

- **能否完全避免 full attention？** 当前仍难以完全消除 $O(n^2)$ 的 full attention 组件——hybrid 模型保留显式 full attention 层，sparse 方法（SeerAttention、DSA）的选择机制本质仍是 $O(n^2)$ 的压缩形式。关键在于昂贵全局计算与廉价局部/稀疏计算的比例，以及 GPU 内存占用
- **KV cache offloading 潜力**：HySparse 天然适合系统级优化策略——将 full attention 的 KV cache offload 到外部存储并在计算前预取，GPU 上仅保留 sparse/selected KV。OmniKV 已在 post-training 场景中探索类似方案

## 关键启示

- **Oracle 优于 proxy**：用前一层 full attention 的真实注意力分数作为后续 sparse 层的 token 选择依据，比任何近似代理模块（启发式、蒸馏、额外网络）都更准确且更简单，消除了额外训练复杂度
- **KV 共享 + 稀疏的协同**：跨层 KV cache 共享和动态稀疏注意力在 HySparse 中自然结合——full attention 层已经生成了 KV cache，sparse 层直接复用即可，无需设计额外的共享机制
- **局部和全局通路不可替代**：即使有精确的全局 token 选择，专用的 SWA 局部通路仍不可或缺；且两个通路需要独立的 KV 表示（共享 KV 给 SWA 会严重降性能）
- **激进混合比例的可行性**：80B MoE 中 1:11 的比例（49 层仅 5 层 full attention）不仅可行，而且优于全 full attention 基线，说明大量层的 full attention 计算和 KV 存储可能是冗余的
