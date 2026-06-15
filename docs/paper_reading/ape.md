---
tags:
  - Prompt Enhancer
  - Reinforcement Learning
  - Diffusion Model
  - GRPO
  - Post Training
---

# APE: Agentic Prompt Enhancer for Image Generation and Editing

- 论文：https://arxiv.org/abs/2606.00204
- 代码/项目页：https://research.nvidia.com/labs/sil/projects/ape/
- 团队：NVIDIA, University of Michigan

## 概述

APE 讨论的是图像生成和图像编辑里的同一个老问题：**视觉模型非常吃 prompt 写法**。同样的用户意图，换种措辞、补一点约束、把空间关系讲清楚，输出质量和跟随度就会明显不同。现有强 prompt enhancer 往往依赖 GPT / Gemini 这类闭源大模型，代价是成本高、延迟高、部署不可控。APE 的目标则更工程化：**能不能把小模型后训练成靠谱的 prompt enhancer，而且不改下游图像模型？**

论文给了两个答案。第一，单 agent 版本 **SAPE** 已经能成立：用一个小语言模型直接一遍改写 prompt，再用 GRPO 仅训练这个改写器本身，就能显著提升对齐和跟随。第二，更复杂的组合型约束更适合 multi-agent 版本 **MAPE**：先 router 选哪些语义字段值得增强，再让 field-specific rewriter 分别补 subject / style / lighting / composition / text rendering 等内容，最后由 composer 统一写回一个自然语言 prompt。训练时只更新增强器，图像生成器 / 编辑器冻结。

APE 的关键贡献不只是“做了多 agent”，而是把**prompt enhancement 的结构设计**和**reward-based post-training**同时系统化了。在图像生成上，MAPE 让小模型 Qwen3-1.7B / 4B 在 UniGenBench 上逼近甚至超过 Gemini-3.1-Pro 的强基线；在图像编辑上，它又把“是否需要重写”也纳入 router 决策，避免对本来就很清楚的 edit instruction 过度加工。

## 动机

- 图像生成/编辑模型对 prompt 的敏感性非常高：
  - 位置关系容易错
  - 计数容易错
  - 风格和语义容易互相干扰
  - 编辑时还可能改坏不该改的区域
- 通用大模型做 one-shot prompt enhancement 很强，但：
  - 推理成本高
  - 无法离线部署
  - 很难针对具体图像模型行为做持续优化
- prompt enhancement 没有唯一正确答案，最终好不好要看生成图像，因此比起 SFT，更适合结合下游视觉 reward 做 RL。

## 方法

### 统一视角

APE 把 prompt enhancer 看成一个策略模型：

- 输入：用户指令 `a_user`
- 输出：增强后的 prompt `a_enh`
- 下游模型：冻结的图像生成器或编辑器 `M`
- 目标：让 `M(a_enh)` 更符合用户意图

重点是：**训练目标不基于“文本像不像参考改写”，而基于“这段改写最终生成的图像好不好”。**

### SAPE：单 Agent Prompt Enhancer

SAPE 很简单：

- 一个小语言模型
- 输入原始 prompt
- 一次性输出增强后 prompt

它适合：

- 本地改写
- 风格润色
- 中等程度的属性补全

作者的发现是：仅仅把 GRPO 用在这个小改写器上，即使完全不改图像模型，也已经能获得不错收益。

### MAPE：多 Agent Prompt Enhancer

当 prompt 含有多个互相耦合的约束时，单次改写容易互相干扰，所以论文引入 MAPE。

它拆成三个角色：

1. **Router**
   - 决定哪些语义字段值得增强
   - 图像生成里字段来自 FIBO 风格的 10 个语义域，如 subject、appearance、background、composition、lighting、style、text render 等
2. **Field Rewriters**
   - 只对被选中的字段做细化重写
3. **Composer**
   - 把字段级重写合成最终自然语言 prompt

这个设计的出发点是：高质量 prompt 往往天然是 compositional 的，应该先分解再组合，而不是在一个大段落里同时解决所有问题。

### 图像编辑里的特化

在 image editing 场景，MAPE 多加了一个重要决策：**是否真的需要重写**。原因是编辑任务和生成任务不一样：

- 某些编辑指令已经很清楚，再重写反而会引入噪声
- 某些复杂编辑（Extract / Adjust / Style / 多操作组合）则需要更强的结构化解释

所以 editing router 除了选字段，还要判断 rewrite necessity。

## 训练

### RL 优化：GRPO 和 GDPO

如果 reward 是单标量，用 **GRPO**；如果是多维 reward，则用 **GDPO**。

#### GRPO

对每个用户输入，采样一组候选增强 prompt，执行到下游模型上拿到图像，再从图像上得到标量 reward。组内做 relative normalization，得到 advantage 后更新 enhancer。

适用场景：

- 单 reward，例如只优化 HPSv2.1
- 或先把多个标量合成一个总 reward

#### GDPO

如果 reward 天然是多维，比如：

- PickScore
- CLIPScore
- HPSv2.1
- Aesthetic
- ImgRwd

直接先加权再做组归一化，会让不同 reward 维度互相淹没。GDPO 的做法是：

- 先对每个 reward 维度分别做组内归一化
- 再聚合成总 advantage

这对 multi-reward 的 prompt enhancement 很关键。

### 训练协议

#### SAPE for generation

- 下游生成器固定：Z-Image-turbo
- 小模型：Qwen3-0.6B / 1.7B
- 单 reward 训练数据：Pick-a-Pic
- 评测：DrawBench

#### MAPE for generation

- enhancer：Qwen3-1.7B / 4B
- 下游生成器：Qwen-Image-2512、Z-Image-turbo、FLUX.2-klein-4B/9B
- SFT 数据：10K
- 用 Gemini-3.1-Pro 提供 scoring rubrics
- benchmark：UniGenBench
- 训练：3 epoch SFT + 30 步 GDPO
- batch size：128
- group size：8

#### MAPE for editing

- enhancer：Qwen3-VL-4B-Instruct
- 下游编辑器：FLUX.2-klein-4B、FLUX.2-klein-9B、Qwen-Image-Edit
- benchmark：ImgEdit
- reward：ImgEdit testpoints，由 GPT-4.1 评估

## 实验

### SAPE：仅训练小改写器也有效

在 Z-Image-turbo 上，单 reward 训练结果：

| Prompt Enhancer | HPSv2.1 |
|-----------------|---------|
| 无 enhancer | 0.2998 |
| Qwen3-0.6B | 0.2801 |
| **SAPE (Qwen3-0.6B)** | **0.3234** |
| Qwen3-1.7B | 0.3015 |
| **SAPE (Qwen3-1.7B)** | **0.3141** |

PickScore-only 训练也类似：

| Prompt Enhancer | PickScore |
|-----------------|-----------|
| 无 enhancer | 23.0484 |
| Qwen3-0.6B | 22.2293 |
| **SAPE (Qwen3-0.6B)** | **23.2233** |
| Qwen3-1.7B | 22.9124 |
| **SAPE (Qwen3-1.7B)** | **23.2066** |

一个很有意思的现象是：**直接拿 off-the-shelf 小模型去当 enhancer 往往会降性能，但 RL 后训练后就能拉回来并超过 baseline。**

### 多 reward SAPE

联合 PickScore + CLIPScore + HPSv2.1 训练后，SAPE 在 held-out 指标上也普遍变好，说明它不是只学会“刷一个 reward”，而是真的改善了 prompt 质量。

### MAPE：组合型 prompt 更适合多 Agent

UniGenBench 上，MAPE 的结果最能说明问题。以 Qwen3-4B 为例：

| T2I Model | Baseline Enhancer | UniGen Short | UniGen Long |
|-----------|-------------------|--------------|-------------|
| Qwen-Image-2512 | Qwen3-4B | 0.7055 | 0.7233 |
| Qwen-Image-2512 | **MAPE (Qwen3-4B)** | **0.8539** | **0.8923** |
| Z-Image-turbo | Qwen3-4B | 0.7221 | 0.8081 |
| Z-Image-turbo | **MAPE (Qwen3-4B)** | **0.8356** | **0.8512** |
| FLUX.2-klein-4B | Qwen3-4B | 0.4817 | 0.5037 |
| FLUX.2-klein-4B | **MAPE (Qwen3-4B)** | **0.8042** | **0.8539** |
| FLUX.2-klein-9B | Qwen3-4B | 0.4777 | 0.5154 |
| FLUX.2-klein-9B | **MAPE (Qwen3-4B)** | **0.8460** | **0.8641** |

这里的重点不是只赢 baseline，而是它能接近甚至匹配 Gemini-3.1-Pro (MSP)。

### MAPE 的三段收益：结构 > SFT > RL

在 Z-Image-turbo 上的消融非常清楚：

| 方法 | UniGen Short | UniGen Long |
|------|--------------|-------------|
| Qwen3-4B | 0.7221 | 0.8081 |
| Qwen3-4B (MSP) | 0.7721 | 0.8290 |
| MAPE - RL (Qwen3-4B) | 0.8233 | 0.8502 |
| **MAPE (Qwen3-4B)** | **0.8356** | **0.8512** |

说明收益来自三个层次：

- 先把 one-shot 改成 router–rewriter–composer 的结构化分解
- 再用 SFT 教模型如何按这个结构工作
- 最后再用 RL 让 enhancer 适配下游图像模型

### 图像编辑

编辑任务的结论更细：

- MSP 对复杂编辑任务通常有帮助
- 但对简单任务，过度重写可能有害
- 因此“是否重写”和“重写哪些字段”在编辑里比生成更关键

作者特别指出，强模型如 Gemini-3.1-Pro 在 editing 上 multi-agent decomposition 不一定继续受益，因为多阶段处理可能带来 error propagation；但对 Qwen3-VL-4B 这类小模型，分解通常更有帮助。

## 关键启示

- **prompt enhancement 不是界面小技巧，而是生成系统里可训练的一层。**
- 小模型直接当 enhancer 往往不够好，但**结构化分解 + reward-based post-training** 可以把它们训成实用模块。
- 对生成任务，增强更详细通常更有利；对编辑任务，**克制同样重要**，因为并不是所有指令都该被大幅改写。
- MAPE 的价值不在于“生成更长的 prompt”，而在于更好地保留和组织 compositional constraints。
- APE 和 PromptEnhancer 的共同点都是“冻结下游视觉模型，只训练 prompt 侧模块”；不同点是 APE 更强调多 agent 分解以及同时覆盖 image generation + editing。
