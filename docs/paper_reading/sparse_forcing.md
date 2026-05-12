---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - Diffusion Forcing
---

# Sparse Forcing: Native Trainable Sparse Attention for Real-time Autoregressive Diffusion Video Generation

[arXiv 2604.21221](https://arxiv.org/abs/2604.21221) | 代码待开源 | Meta Superintelligence Labs、UCSB

## 概述

自回归视频扩散模型（如 Self-Forcing）在 rollout 过程中面临两个核心挑战：KV cache 随历史帧线性增长（1.3B 模型生成 1 分钟视频 KV cache 达 44.9GB），以及模型以自己生成的噪声作为条件的误差累积。

Sparse Forcing 的出发点是两个经验发现：
1. **垂直集聚持久锚点**：历史帧中少数 block 持续吸引大量注意力，形成隐式时空记忆
2. **局部对角块稀疏**：即使是最近的局部窗口内，注意力也呈现结构化块稀疏模式，Top-K=25% 即可覆盖大部分注意力质量

基于此提出可训练的 Persistent Block-Sparse Attention（PBSA），维护有界持久记忆 + 局部块稀疏窗口，训练期间保持与推理一致的动态缓存和自适应局部注意力，消除 train-test mismatch。

在 Wan2.1-T2V-1.3B 上蒸馏训练 1200 步，5 秒视频 VBench +0.26，解码加速 1.11–1.17×，KV cache 峰值降低 42%。更长 rollout（20 秒/1 分钟）优势进一步扩大：VBench +0.68/+2.74，加速 1.22×/1.27×。

## 背景：自回归扩散中的误差传播视角

自回归扩散模型 (x_{1:N}) = \prod_i p(x_i | x_{<i})$ 用扩散生成器实现条件项。训练时使用 Teacher Forcing（条件为干净 ground-truth）或 Diffusion Forcing（条件为加噪 ground-truth），推理时模型需以自身过去预测为条件，存在 train-test mismatch（exposure bias）。

**稀疏性的双重作用**：稀疏不仅是计算优化，也重塑了误差传播的依赖图。密集注意力提供丰富的错误传播路径，早期错误通过密集连接放大。结构化稀疏条件机制通过限制不确定 token 影响后续帧的路径范围和强度，可以**同时**改善质量和效率。

## 核心观察：持久性涌现与局部块稀疏

在 Self-Forcing rollout 分析中发现两种注意力模式：

**垂直集聚持久锚点**：注意力质量集中在少数历史上 block 的"垂直条带"，形成持久的全局锚点（主体身份、场景布局等稳定信息）。大部分历史 block 贡献边缘。

**局部对角块稀疏**：局部窗口内注意力呈对角线块稀疏模式，反映结构化短程依赖。使用块级评分选 Top-K=25% 即可达到 0.65 ± 0.11 的 token 召回率（跨 head 和层平均）。

## 结构化记忆分解

在自回归步 $、扩散步 $，维护隐式记忆 ^k = P_t \cup L_t^k$，其中：
- $：持久去噪 block 集合（=0$ 时提取），承载长程语义锚点，跨扩散步共享（$|P_t| \leq C$）
- ^k$：滑动窗口内的最近时空连续 block，随 $ 推进更新

**PBSA 注意力计算**：拼接  = [K_P; K_L], V = [V_P; V_L]$，对持久锚点密集访问，对局部窗口块稀疏访问（mask  = [0_{N_q \times N_p} \, M_L]$），[q, \ell] = \log \mathbb{I}[\ell \in \Omega(q)]$。

**持久记忆更新**：用压缩 block representative ^c = \phi_Q(Q_t^{\mathrm{blk}})$ 和 {:t}^c = \phi_K(K_{:t}^{\mathrm{blk}})$（pooling 压缩 $ 个 token 为 1 个代表）计算块级注意力矩阵 $，聚合所有 query block 的注意力权重得每个 key block 的重要性分数 $，Top-C 保留  = \mathrm{TopC}(P_{t-1} \cup E_t; s_t)$，$ 为刚从 $ 中 evict 的候选 block。

**局部块稀疏选择**：在局部窗口内用压缩 representative 计算 $，每 query block 做 row-wise Top-K 选择得到可见 key block 集合 $\Omega(q)$。

## PBSA 定制 Kernel

用 ThunderKittens 实现，支持持久 block 携带跨步传递、动态块选择、训练和推理端到端高效执行。

**Profiling 数据**（65K KV 序列，4096 token 持久记忆，6.25% 局部块稀疏）：
- 块稀疏注意力：73.16%
- Row-wise Top-K 选择 + mask 生成：19.04%
- 块压缩 + representative 注意力 + 广播：3.12%

**Kernel 加速**（vs FlashAttention-2，H100）：Top-K=6.25%/12.5%/25% 时最高加速分别达 11.11×/7.29×/4.34×。稀疏度越高、序列越长、局部窗口越大、持久部分越紧凑 → 加速越大。

## 训练

从 Wan2.1-T2V-1.3B 双向模型蒸馏为因果自回归生成器：
- 用基模型采 16K ODE 解对初始化因果 mask
- 4 步 diffusion 采样，chunk-wise 去噪（每 chunk 3 帧）
- Distribution Matching Distillation（DMD）loss，1200 步，batch 64，AdamW
- 训练期间启用动态缓存更新和自适应局部注意力，与推理一致

**Ablation**（20s rollout）：
- 移除持久记忆 P：Dynamic Degree 从 66.39 降至 47.22，Color 从 89.47 降至 80.88
- 移除局部 BSA：Dynamic Degree 降至 50.93
- 训练不启用（training-free 即插即用）：仍比 Self-Forcing 好，但有 semantic rewrite 问题
- 完整模型在 Dynamic Degree 和 Color 等长视频最敏感指标上全面领先

## 实验

**短期（5s）**：VBench Total 84.14，超过 Self-Forcing 83.88 和 SkyReels-V2 82.67。解码延迟 0.59s vs 0.69s。

**中长期评估**（4–12× 超越训练 horizon 的 rollout）：

| 时长 | 方法 | VBench Total | Quality | Semantic | FPS↑ |
|------|------|-------------|---------|----------|------|
| 20s | Self Forcing | 82.09 | 82.48 | 80.51 | 14.4 |
| 20s | Sparse Forcing | **82.68** | 83.13 | 80.87 | **18.3** |
| 60s | Self Forcing | 78.93 | 79.48 | 76.70 | 13.9 |
| 60s | Sparse Forcing | **81.96** | 82.25 | 80.82 | **18.0** |

16 维完整 VBench 评估显示：human action（97.00 vs 80.80）、object class（95.65 vs 88.49）、multiple objects（88.14 vs 74.70）、scene（56.72 vs 44.97）等语义维度改善最大。color 在 60s rollout 上从 71.70 提升到 86.54。

## 关键启示

- **稀疏性不仅是加速手段，也是质量控制机制**：在自回归扩散中，稀疏注意力通过限制误差传播路径的结构来抑制 compounding error，这与 LLM 中稀疏注意力纯粹为效率的动机不同
- **持久记忆 + 局部窗口的组合是长视频生成的通用范式**：垂直集聚持久锚点的现象在不同 content 中普遍出现，说明视频中有稳定语义结构贯穿时间。训练期间保持与推理一致的稀疏模式对消除 mismatch 至关重要
- **ThunderKittens 是定制块稀疏 Attention kernel 的有效工具**：PBSA 支持不规则稀疏 pattern + 跨步持久 KV 携带，是对 FlashAttention 体系的重要补充
- **Training-free 即插即用可能引入 semantic rewrite**：不匹配的检索动态与记忆更新规则导致训练-推理 gap，需训练来使模型学会利用持久记忆
