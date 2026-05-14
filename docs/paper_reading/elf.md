---
tags:
    - Diffusion Model
    - Flow Matching
---

# ELF: Embedded Language Flows

- **论文**: [arXiv:2605.10938](https://arxiv.org/abs/2605.10938) (2026-05-11)
- **代码**: [github.com/lillian039/ELF](https://github.com/lillian039/ELF)
- **Checkpoint**: [huggingface.co/embedded-language-flows](https://huggingface.co/embedded-language-flows)
- **团队**: MIT (Keya Hu*, Linlu Qiu*, Yiyang Lu, Hanhong Zhao, Tianhong Li, Yoon Kim, Jacob Andreas, Kaiming He; *equal contribution, order decided by coin flip)

## 概述

ELF 提出了一类**连续扩散语言模型 (Continuous Diffusion Language Model)**，核心思路是将语言生成完全放在**连续嵌入空间**中做 Flow Matching，直到最后一步才映射回离散 token。与现有扩散语言模型 (DLM) 的最大区别在于：**整个去噪轨迹保持在连续空间**，不在中间步骤做离散化（不施加 cross-entropy loss），只在 $t=1$ 最终步用共享权重的网络做一次 decode。

传统离散 DLM（MDLM、Duo 等）性能强但需要在离散 token 空间操作，无法直接复用图像扩散模型中的成熟技术（如 CFG）；而传统连续 DLM（Diffusion-LM、CDCD 等）在中间步施加 token 级别的 cross-entropy 监督，导致去噪轨迹被词汇表约束，灵活性受限。

ELF 的核心洞察是：**Flow Matching 的最终时间步天然可以做 continuous-to-discrete 映射**，不需要额外 decoder。去噪网络和 decode 网络共享权重，训练时 80% 的 batch 做 MSE denoising，20% 做 CE decoding。这种极简设计加上 continuous-time Flow Matching 带来的 CFG 兼容性，让 ELF 在仅 45B 训练 token（比同类方法少 10 倍以上）且无需蒸馏的情况下，显著超越现有方法。

### 核心贡献

- 提出 ELF 框架：连续嵌入空间 + 连续时间 Flow Matching + 仅最终步离散化
- 证明连续 DLM 可以在极简离散化处理下达到强竞争力
- 共享权重的 denoiser-decoder，避免额外 decoder 模块
- x-prediction 参数化使 Flow Matching 能在高维嵌入（768-d per token）有效运行，且与 decode 步骤权重共享兼容
- 自然支持 CFG，训练时 CFG 不增加推理开销
- 数据效率极高：45B token 超越 500B+ token 训练的方法


## 背景与相关工作

### Flow Matching 基础

Flow Matching [Lipman et al., 2023; Liu et al., 2023; Albergo & Vanden-Eijnden, 2023] 定义了一条从噪声到数据的连续流路径。设 $x \sim p_{data}(x)$ 为数据分布，$\epsilon \sim p_{noise}(\epsilon)$ 为噪声分布（标准高斯）。通过**线性插值（rectified flow）**定义含噪隐变量：

$$z_t = t x + (1 - t) \epsilon, \quad t \in [0, 1]$$

边界条件：$z_0 = \epsilon$（纯噪声），$z_1 = x$（干净数据）。

在连续时间中，**流速 (velocity)** 定义为 $z_t$ 对时间 $t$ 的导数：

$$v(z_t, t) = \frac{dz_t}{dt} = x - \epsilon$$

Flow Matching 的目标是训练神经网络 $v_\theta(z_t, t)$ 来拟合真实流速 $v$，损失函数为：

$$\mathcal{L}_{FM} = \mathbb{E}_{t, x, \epsilon} \left\| v_\theta(z_t, t) - v \right\|^2$$

#### x-prediction 参数化

ELF 不直接预测 $v$，而是预测干净嵌入 $x$（x-prediction），原因有二：

1. 高维空间（768-d per token）上 x-prediction 更稳定（与 [Li & He, 2025] 的发现一致）
2. x-prediction 的输出自然可以直接用于最终的离散化解码步骤，与 shared-weight denoiser-decoder 设计兼容

x-prediction 和 v-prediction 的关系由线性插值约束给出：

$$v(z_t, t) = \frac{x - z_t}{1 - t}$$

代入 Flow Matching 损失即可得到 x-prediction 形式的 MSE 损失：

$$\mathcal{L}_{MSE} = \mathbb{E}_{t, x, \epsilon} \left\| v_\theta(z_t, t) - v \right\|^2 = \mathbb{E}_{t, x, \epsilon} \frac{1}{(1 - t)^2} \left\| x_\theta(z_t, t) - x \right\|^2$$

这一形式的重要性在于：当 $t \to 1$ 时，$(1-t)^{-2}$ 使损失权重趋向无穷大，自然强调接近干净数据时的精细预测。

### 扩散语言模型 (DLM) 分类

现有连续 DLM 可按设计维度分类（论文 Tab. 2 提供完整 survey）：

- **嵌入空间扩散 (Embedding-space Diffusion)**：Diffusion-LM、CDCD、DiffuSeq、SeqDiffuSeq 等，直接在 token embedding 上加高斯噪声做去噪。大部分在**中间步**施加 cross-entropy loss（训练时 per-step discretization），将中间隐状态映射回 token 做监督。
- **Simplex 扩散**：SSD-LM、TESS 等，在 softmax 概率单纯形上定义扩散过程，通过单纯形约束自然保持概率解释，同样在中间步做离散化。
- **隐空间扩散 (Latent Diffusion)**：LD4LG、PLANNER、TEncDM、Cosmos 等，在冻结的 encoder 表示上做 DDPM 去噪，然后靠**独立训练的 decoder** 恢复 token。通常使用 DDPM 噪声调度而非 Flow Matching。
- **基于 Flow 的方法**（并发工作）：DFM、CFM、FLM/FMLM、LangFlow 等也探索 continuous flow-based 语言建模，但仍在轨迹中施加 token-level CE 监督。

ELF 的定位：**连续时间 Flow Matching + 冻结预训练 encoder 嵌入空间 + 无中间离散化 + 无独立 decoder**。


## ELF 框架详解

### 从离散 Token 到连续嵌入

给定句子 $s = [s_1, \ldots, s_L] \in \mathcal{V}^L$（$\mathcal{V}$ 为词表，$L$ 为序列长度），首先通过 encoder 映射到连续嵌入：

$$x = \text{encode}(s) \in \mathbb{R}^{L \times D_{enc}}$$

ELF 默认使用**冻结的预训练 T5-small encoder**（35M 参数，$D_{enc} = 512$）获取**双向上下文嵌入**。Encoder 只在训练时使用，推理时不需要——推理时直接从高斯噪声出发迭代去噪。

之所以选双向上下文嵌入而非 non-contextual embedding（如单层 embedding lookup），是因为上下文嵌入能捕获更丰富的语言结构信息，实验证实其生成质量-多样性 trade-off 最优。

在送入 ELF 网络前，嵌入经过 bottleneck 线性投影降维到 128 维（降维有利于模型学习低维流形上的去噪），再映射回模型 hidden size 768。

**嵌入归一化**：训练前先在整个数据集上估计嵌入的均值和标准差，用这些统计量对干净嵌入做通道级归一化。

### 连续嵌入上的 Flow Matching

获得 $x$ 后，定义去噪过程。采样时间步 $t$（从 logit-normal 分布采样），构造含噪嵌入：

$$z_t = t \cdot x + (1 - t) \cdot \text{scale} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中噪声缩放因子 $\text{scale} = 2.0$（denoising branch）。网络 $x_\theta(z_t, t, \text{mode})$ 预测干净嵌入 $\hat{x}$，训练目标为 MSE：

$$\mathcal{L}_{MSE} = \mathbb{E}_{t, x, \epsilon} \frac{1}{(1 - t)^2} \| x_\theta(z_t, t, \text{"denoise"}) - x \|^2$$

**时间步采样使用 logit-normal 分布**：

$$t \sim \sigma\big(\mathcal{N}(P_{mean}, P_{std}^2)\big), \quad P_{mean} = -1.5, \; P_{std} = 0.8$$

其中 $\sigma(\cdot)$ 为 sigmoid 函数。这使采样集中在中间区域，两个极端（$t \approx 0$ 和 $t \approx 1$）采样较少但不为零。

### 回到离散 Token：最终步解码

ELF 的关键创新在于如何处理连续到离散的转换。标准做法需要独立训练的 decoder，ELF 直接复用去噪网络：

- 训练时有 denoising branch（MSE loss）和 decoding branch（CE loss）
- 两个 branch 使用**完全相同的网络权重**，仅靠一个二值 "mode" token 区分（"denoise" vs "decode"）
- 两个 branch 在一个 batch 中混合训练，不需要额外开销

Decoding branch 的特殊之处在于 $t = 1$。此时 $z_t \to x$（干净嵌入），如果直接送入网络则任务 trivial。因此引入**per-token corruption**：

$$\tilde{z} = p \cdot x + (1 - p) \cdot \text{scale}_{dec} \cdot \epsilon$$

其中 $p$ 是 per-token 的 corruption level，从另一个 logit-normal 分布采样：

$$p \sim \sigma\big(\mathcal{N}(0.8, 0.8^2)\big)$$

$\text{scale}_{dec} = 5.0$（OWT 上）。每个 token 的 $p$ 独立采样，意味着同一序列中某些 token 几乎干净、某些高度噪声。这迫使 decoder-mode 网络从上下文中恢复被破坏 token——恰好模拟了推理时 denoiser 产生的不完美嵌入。

随后通过可学习的 unembedding 矩阵 $W$ 获得 logits，计算 cross-entropy：

$$\mathcal{L}_{CE} = \mathbb{E}_{\tilde{z}} \left[ \text{CrossEnt}\big(W \cdot x_\theta(\tilde{z}, t=1, \text{"decode"}), s\big) \right]$$

训练时 80% 的 batch 分配为 denoising mode（MSE），20% 为 decoding mode（CE）。

### 为什么 x-prediction 对共享权重至关重要

如果使用 v-prediction，网络的直接输出是速度场 $v_\theta$，需要额外一步转换为干净嵌入 $x = z_t + (1-t) \cdot v$ 才能用于 CE loss。但 v-prediction 网络未直接针对干净嵌入做优化，当与 CE loss 共享权重时会出现梯度冲突。

x-prediction 则天然对齐：denoising branch 输出干净嵌入（监督 MSE），decoding branch 也输出干净嵌入（监督 CE），两个目标的输出空间一致，权重共享自然合理。

实验证实（Appendix C.1）：v-prediction 在 512-d 时还行，768-d 和 1024-d 时退化明显；$\epsilon$-prediction 在所有维度都崩塌；只有 x-prediction 在所有维度稳定。


## 训练与推理伪代码

### 训练（双分支混合）

训练流程的核心是双分支混合——denoising branch 学习连续去噪动力学，decoding branch 学习最终步离散映射。两者的区分通过二值 mode token 实现。

```
# net(z, t, mode): ELF 网络
# s: 离散 token 序列

x = encode(s)

if uniform(0, 1) < 0.8:           # denoising branch (80%)
    t = sample_t()                  # logit-normal 采样
    e = randn_like(x)
    z = t * x + (1 - t) * 2.0 * e  # 直线插值加噪
    v_target = x - e                # 真实速度
    x_pred = net(z, t, mode="denoise")
    v_pred = (x_pred - z) / (1 - t)
    loss = mse_loss(v_pred, v_target)

else:                               # decoding branch (20%)
    p = sample_p()                  # per-token logit-normal
    z = p * x + (1 - p) * 5.0 * e  # per-token corruption
    x_pred = net(z, t=1, mode="decode")
    logits = unembed(x_pred)        # W @ x_pred
    loss = ce_loss(logits, s)
```

### 推理（ODE / SDE）

推理从纯高斯噪声 $z_0 \sim \mathcal{N}(0, I)$ 开始，沿 ODE $dz_t/dt = v_\theta(z_t, t)$ 逐步去噪。时间区间 $[0, 1]$ 离散化为 $T$ 个区间（默认 logit-normal schedule）。

**ODE 采样**（确定性 Euler 积分）：

```
z = randn(shape)
x_pred = zeros_like(z)
for t, dt in time_steps:
    x_pred = net(z, t, mode="denoise")
    v = (x_pred - z) / (1 - t)     # x-prediction → velocity
    z = z + dt * v                  # Euler step
# 最终步
h = net(z, t=1, mode="decode")
tokens = argmax(unembed(h))
```

**SDE 采样**（随机性）：SDE variant 在每一步后重新注入少量高斯噪声，同时将时间变量向噪声方向回退。参数 $\gamma$ 控制噪声重注入量：$\gamma = 0$ 退化为 ODE，默认 $\gamma = 1.0$。

```
def sde_step(z, t, dt, gamma):
    e = randn_like(z)
    alpha = 1 - gamma * dt
    t_back = alpha * t
    z_back = alpha * z + (1 - alpha) * e  # 后退并重注噪声
    x_hat = net(z_back, t_back, mode="denoise")
    v = (x_hat - z) / (1 - t)             # 对原始 z 计算速度
    z = z + dt * v
    return z
```

SDE 采样的直觉：每一步重注噪声可以**校正在去噪早期累积的误差**，而非像 ODE 那样确定性放大 imperfect 轨迹。在极少步数（8-32 步）时 SDE 优势尤为显著。

**时间调度 (time schedule)**：推理时使用 logit-normal 时间调度——在 $t$ 接近 0 时分配更细的时间步，$t$ 接近 1 时步长更大。这因为噪声大的区域需要更精细的离散化，且与训练时的 logit-normal 分布匹配。

### Self-Conditioning

Self-conditioning [Chen et al., 2023] 是 ELF 的核心组件。在第 $i$ 步，模型以上一步的预测 $\hat{x}_{i-1}$ 作为额外输入：

$$\hat{x}_i = x_\theta(z_{t_i} \;|\; \hat{x}_{i-1}, t_i, \text{mode})$$

实现方式：将 $z_t$ 与 $\hat{x}_{i-1}$ 沿通道维度拼接（channel 维度翻倍），再通过一个线性层投影回原始维度。训练时以 50% 概率使用 self-conditioning（拼接前一步预测），50% 概率使用零向量（学习无条件路径）。推理时以上一步预测作为条件，**不增加额外 forward pass**。

self-conditioning 的预测 $\hat{x}'$ 也是 CFG 的 conditioning signal $c$。

### 训练时 CFG + Self-Conditioning

ELF 使用训练时 CFG [Geng et al., 2025]，避免推理时做两次 forward pass。核心思想是让网络直接建模后组合的速度场 $v^{cfg}$，而非分别建模 conditional 和 unconditional。

$$v^{target} = v + \left(1 - \frac{1}{\omega}\right) \big(v_\theta(z_t \mid c, \omega) - v_\theta(z_t \mid \emptyset, \omega)\big)$$

其中 $\omega$ 是 guidance scale。当 $\omega = 1$，$v^{target} = v$（退化为无 CFG）。

训练时对每个 sample 随机采样 $\omega \in [0.5, 5.0]$（偏向小值的 power distribution），网络中通过 in-context conditioning（prepend CFG scale token）告知 $\omega$。推理时只需改 $\omega$ token 的值即可改变引导强度，不需要额外 forward pass。

### 条件生成扩展

对于条件生成（机器翻译、摘要），将条件序列的干净嵌入 prepend 在目标序列前。条件嵌入在训练和推理中保持 uncorrupted。模型通过双向 self-attention 同时处理条件+目标。CFG 对条件嵌入做 dropout（10% 概率清零），让模型学习 conditional 和 unconditional 两条路径。


## 实验

### 实验设置

- **数据**：无条件生成用 OpenWebText (OWT, ~9B tokens)；条件生成用 WMT14 De-En 翻译和 XSum 摘要
- **评估**：生成 1000 样本，以 GPT-2 Large 计算 Gen. PPL（生成质量）+ unigram entropy（多样性）
- **模型**：基于 DiT 架构 + SwiGLU + RMSNorm + RoPE + qk-norm + in-context conditioning。三档规模：
  - ELF-B: 12 层, 768 hidden, 12 头, 105M 参数
  - ELF-M: 24 层, 1056 hidden, 16 头, 342M 参数
  - ELF-L: 32 层, 1280 hidden, 16 头, 652M 参数
- **训练**：Muon optimizer, lr=0.002, batch=512. OWT 5 epochs (~95K steps), WMT14/XSum 100 epochs. TPU v5p×64
- **推理**：默认 SDE sampler + logit-normal time schedule，self-conditioning CFG scale 可灵活调整

### 消融实验关键发现

**1. CFG scale**（Fig. 4）：增大 CFG scale → Gen. PPL 下降，熵也下降。最优方向是右下角（低 Gen. PPL + 高熵），CFG 在 3 附近最优。

**2. 嵌入选择**（Fig. 5a）：上下文嵌入（T5 encoder）> 非上下文嵌入（单层 embedding）。预训练 encoder > 从零训练的 encoder > 可学习 embedding（最差）。高斯初始化冻结 embedding 也有一定效果。

**3. 解码策略**（Fig. 5b）：共享权重 vs 两阶段独立 decoder：两者 trade-off 相近，但共享权重在低 Gen. PPL 区域延伸更远，且 pipeline 更简洁。

**4. 采样器**（Fig. 5c）：SDE 在少步时显著优于 ODE。8 步 SDE 约等于 64 步 ODE 的质量。随步数增加差距缩小，但 SDE 始终更好。

**5. x/v/$\epsilon$-prediction**（Appendix Fig. 10）：x-prediction 在 512/768/1024 维度都稳定。v-prediction 在 512-d 可接受，高维退化。$\epsilon$-prediction 在所有维度崩塌（极高 Gen. PPL 或极低熵）。

**6. Bottleneck 维度**（Fig. 11）：128-d 最佳平衡；32-d 低熵（多样性差），512-d 高 Gen. PPL（质量差）。

**7. Denoising mode probability**（Fig. 12）：0.8（80% MSE, 20% CE）最优。过低会导致去噪过程训练不足。

**8. 条件策略**（Fig. 13）：In-context conditioning（prepend 条件 token）略优于 adaLN-Zero，同时大幅减少参数（148M→105M）。

**9. 优化器**（Fig. 14）：Muon > AdamW，但两者训练的模型都超越 baseline。ELF 的优势不能仅归因于优化器。

**10. 时间调度**（Fig. 15a）：Logit-normal 在所有步数都优于 uniform，少步时优势明显。

**11. SDE $\gamma$**（Fig. 15b）：适中范围内增大 $\gamma$ 降低 Gen. PPL 但稍降熵。$\gamma = 1.0$ 最佳平衡。

### 无条件生成系统级对比（Fig. 7）

ELF-B (105M) vs 约 170M 的 baseline：

| 方法 | 采样步数 | Gen. PPL | 训练 token |
|------|--------|---------|-----------|
| ELF-B (SDE, CFG=3) | 32 | **24.1** | **45B** |
| MDLM | 1024 | ~35 | 524B |
| Duo | 1024 | ~50 | 524B |
| LangFlow | 1024 | ~40 | 524B |
| MDLM+SDTT (蒸馏) | 32 | ~38 | 551B |
| Duo+DCD (蒸馏) | 32 | ~30 | 551B |
| FMLM (蒸馏) | 32 | ~27 | 577B |

ELF-B 用 32 步达到比蒸馏模型更好的质量，训练 token 仅为其 1/12。在极少步（8-16 步）时，ELF 略弱于最佳蒸馏模型，但仍远超非蒸馏 baseline。

### 条件生成系统级对比（Tab. 1）

| 模型 | WMT14 BLEU↑ | XSum R-1↑ | XSum R-2↑ | XSum R-L↑ |
|------|------------|----------|----------|----------|
| AR (99M) | 25.2 | 30.5 | 10.2 | 24.4 |
| MDLM (99M) | 18.4 | 33.4 | 11.6 | 25.8 |
| Duo (170M) | 21.3 | 31.4 | 10.1 | 25.0 |
| E2D2 (99M) | 24.8 | 28.4 | 8.3 | 22.0 |
| **ELF-B (105M)** | **26.4** | **36.0** | **12.2** | **27.8** |

ELF-B 在翻译和摘要两个任务上全面超越所有 baseline（包括 AR 模型）。

### 缩放行为（Fig. 6）

- ELF-B (105M) → ELF-M (342M) → ELF-L (652M)：模型越大，Gen. PPL-熵 frontier 越向更优方向移动
- ELF-L CFG=3 时 Gen. PPL=23.3，熵=5.28（64 步 SDE）
- 大模型能承受更大的 CFG scale（ELF-L 在 scale=4 才退化，ELF-B 在 scale=3 就退化）
- SDE 优势在所有规模一致保持


## 训练流程全景图

ELF 的完整训练 pipeline（对应论文 Fig. 9 + Alg. 3, 4）：

```
输入 token s → T5 encoder → clean embedding x
                              │
                    ┌─────────┴──────────┐
                    │                    │
              denoising (80%)      decoding (20%)
                    │                    │
          z_t = t*x + (1-t)*ε    z̃ = p*x + (1-p)*ε
          (linear interp)        (per-token corruption)
                    │                    │
          self-cond: concat     self-cond: φ (always 0)
          [z_t, x̂_prev or 0]     concat [z̃, 0]
                    │                    │
          project to original    project to original
          dimension              dimension
                    │                    │
          prepend control tokens:
          [time_tok | cfg_tok | mode_tok | cond_emb | target_seq]
                    │                    │
          DiT network (full bidirectional self-attn)
                    │                    │
               x̂ = net(...)         x̂ = net(...)
                    │                    │
          v_pred = (x̂ - z)/(1-t)   logits = W @ x̂
          L_MSE = MSE(v_pred, v)   L_CE = CE(logits, s)
          + training-time CFG      (no CFG for decoding)
```

关键细节：
- **Self-conditioning 投影**：$[z_t \in \mathbb{R}^{L \times d}, \hat{x}_{prev} \in \mathbb{R}^{L \times d}]$ concat 为 $\mathbb{R}^{L \times 2d}$，线性层投影回 $\mathbb{R}^{L \times d}$
- **Control tokens**：4 time tokens（值 $\in [0, 1]$）+ 4 CFG tokens（值 $\in [0.5, 5]$）+ 4 mode tokens（denoise/decode）。token embedding 维度与模型 hidden size 一致，与数据序列 concat 后共同做 self-attention
- **Training-time CFG**：对每个 sample，网络做两次 forward——unconditional ($\hat{x}_{no\_sc}$) 和 conditional ($\hat{x}_{sc}$)，然后按 CFG 公式组合 velocity target

## 关键启示

1. **连续空间的极简设计可以很有效**：ELF 的核心哲学是"让连续扩散做它擅长的事"，只在最后一步接触离散空间。这打破了之前 DLM 需要在中间步加 token-level 监督的惯性思维。

2. **x-prediction 是关键**：在高维嵌入空间（512-d 到 1024-d）中，x-prediction 是唯一稳定工作的参数化方式，且天然与解码步骤兼容。这与 [Li & He, 2025] 的发现一致——"让降噪模型做降噪"。

3. **Flow Matching + 冻结 Encoder 的组合非常高效**：不需要在推理时运行额外的 encoder/decoder 模块，网络规模纯粹由 Transformer backbone 决定。数据效率的提升（12× token reduction）可能源于预训练 encoder 已经提供了丰富的语言结构信息。

4. **SDE 采样在少步时显著优于 ODE**：噪声重注入帮助修正早期去噪误差——这对高效推理很重要。配合 logit-normal time schedule，SDE 是 ODE 的近乎免费提升。

5. **CFG 在连续 DLM 中自然可用**：与离散 DLM 不同（CFG 在离散空间效果不佳），ELF 的连续设计可以直接复用图像扩散模型中成熟的 CFG 技术。Training-time CFG 进一步使推理时无需额外 forward pass。

6. **共享权重的 denoiser-decoder 设计简洁有效**：网络同时学习"去噪"和"解码"，避免了独立 decoder 的训练成本。关键超参是 denoising/decoding 的训练比例（80/20）。

7. **计算和数据的 trade-off**：ELF 用更少的训练 token（45B vs 500B+）和更少的采样步数（32 vs 1024+）实现了更好的质量，说明方法本身比单纯堆数据更关键。

8. **局限与未来方向**：当前 105M-652M 规模的实验尚未覆盖大语言模型的规模（1B+），更大规模的 scaling behavior 是重要的开放问题。推理仍需要多步 forward pass（32-64 步），与 AR 模型的一步生成相比仍有差距。条件生成目前只测试了翻译和摘要，更复杂的指令跟随/对话任务待探索。

9. **与 AR 模型的关系**：ELF 展示了一条不同于 AR 的语言生成路径。其双向 self-attention + 迭代去噪的特性适合某些需要全局一致性的任务（如翻译、摘要），但对长文本生成的质量上限还需进一步验证。
