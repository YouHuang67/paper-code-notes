---
tags:
  - Prompt Enhancer
  - Video Generation
  - Reinforcement Learning
  - GRPO
  - Post Training
---

# ReaDe: A Reason-Then-Describe Instruction Interpreter for Controllable Video Generation

- 论文：https://openreview.net/pdf/42b7467c51a9efd514617c742225d79d66ca0b39.pdf
- 代码：论文称将发布，当前未公开
- 团队：匿名（ICLR 2026 双盲投稿）

## 概述

ReaDe 关注的问题不是“怎样把视频模型训得更强”，而是**怎样把用户原始指令解释成下游可控视频生成器真正能理解的 dense structured caption**。论文的核心观察是：训练时视频模型吃的是高质量、长描述、结构化 caption；推理时用户给的是短句、歧义句，甚至是“文本 + 参考图 + depth + pose + camera”这种跨模态混合条件。两者之间的解释鸿沟，才是 controllable generation 经常失控的直接原因。

ReaDe 因此不把自己定义成 prompt rewriter，而是一个 **instruction interpreter**。它采用 `reason-then-describe` 范式：先分析文本意图，再理解非文本条件，再做多模态对齐，最后补齐用户没说但视频生成需要的背景、风格、镜头和动作细节，输出一个 6 段式 structured caption。训练上也是两阶段：第一阶段用 CoT 数据做 reasoning initialization，第二阶段用多维 reward + GRPO 继续优化，让模型不只是“会照着数据模仿”，而是真的能泛化到新的组合条件和推理型输入。

实验上，ReaDe 在 multiple identities、depth、camera、human pose 这几类 controllable video 条件下都优于 Any2Caption 和直接短 prompt。尤其在多条件组合时优势更明显，例如 `C + D + I` 组合下，文本对齐、身份一致性、深度误差、smoothness 和 aesthetic 基本都优于基线。论文想表达的中心结论很明确：**视频可控生成要先做好“解释器”，再谈生成器本身。**

## 动机

- 用户输入往往简短、模糊，只表达了意图，没有表达生成模型需要的所有细节。
- 多条件控制里，文本、身份图、姿态、深度、镜头之间经常彼此冲突，需要一个显式对齐器。
- 直接监督改写容易过拟合训练分布，对未见条件组合和 reasoning-heavy 指令泛化差。
- 视频 caption 是开放生成任务，没有唯一标准答案，所以需要比纯 SFT 更强的反馈学习。

## 方法

### 输出形式：6 段 structured caption

ReaDe 不直接输出一个长段落，而是统一生成：

1. Dense caption
2. Action caption
3. Main Object caption
4. Background caption
5. Style caption
6. Camera caption

同时把推理过程放在 `<think>...</think>` 中，最终答案放在 `<answer>...</answer>` 中。这种格式化输出一方面便于训练，另一方面也方便把不同控制源映射到不同字段。

### Stage 1：CoT-guided reasoning initialization

这一阶段的目标，是先让模型具备“先想清楚再写 caption”的能力。论文构造了一个四步推理链：

1. **Interpretation of Textual Intent**
   - 提取文本里的核心目标
   - 区分是创建、添加还是修改某些元素
2. **Non-Textual Understanding**
   - 把参考图、身份图、depth、pose、camera 等非文本条件转成文字线索
3. **Multimodal Alignment**
   - 对齐文本意图和其他模态，解决冲突和歧义
4. **Supplementary Detail Completion**
   - 补出缺失但生成需要的环境、风格、镜头和上下文细节

最后得到 `<think>` 推理链和 `<answer>` 六段 caption。数据经去重和筛选后，SFT 集规模为 **8.4K**。

优化目标是标准 next-token loss：

$$
\mathcal{L}_{cot} = - \mathbb{E}_{(x,y)\sim D_{cot}} \sum_t \log \pi_\theta(y_t|x,y_{<t})
$$

这一步的作用，是建立一个强初始点，让模型先具备结构化多模态解释能力。

### Stage 2：多维 reward + GRPO

仅靠 SFT，模型更多是在“模仿训练数据中的解释风格”。为了让它真正学会对复杂输入做泛化，论文在第二阶段使用了 feedback-guided RL。

#### 1. 格式奖励

要求同时满足：

- 有 `<think>` 和 `<answer>` 结构
- `<answer>` 中正确输出 6 段 caption

奖励设计很简单：

- 两者都满足：`1`
- 只满足一个：`0.2`
- 都不满足：`0`

#### 2. 内容奖励

把 gold structured caption 离线拆成三类核心元素：

- `U`：用户文本中明确要求的 essential details
- `S`：来自非文本条件的 supplementary information
- `Z`：为了提升连贯性和真实性而加入的合理 imaginative details

用 Qwen3-30B judge 检查预测 caption 对这些元素的覆盖率，得到：

$$
R_{user}, \quad R_{detail}, \quad R_{supp}
$$

同时加入一个矛盾惩罚：

$$
R_{contra} = 1[\text{contradict}(\hat{y})]
$$

总 reward 为：

$$
R = \alpha R_{user} + \rho R_{detail} + \gamma R_{supp} - \lambda R_{contra}
$$

这套 reward 的含义很实用：

- `R_user` 保证别偏离用户原意
- `R_detail` 保证别丢掉其他模态条件
- `R_supp` 鼓励模型补全合理细节，而不是机械复述
- `R_contra` 防止长 caption 自己打架

#### 3. GRPO 优化

对于每条输入指令 `q`，模型采样 `G` 个候选输出，论文设置为：

- **8 rollouts per prompt**
- KL coefficient = **0.001**

组内优势为：

$$
A_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}
$$

然后用标准 GRPO 方式更新策略。

### 多模态输入怎么接入

ReaDe 以 Qwen2.5-Omni 为初始化，并额外加入一个 **camera encoder** 来补足原模型对镜头运动的理解不足。论文训练时主要覆盖四种条件类型：

- multiple identities
- camera motion
- depth map
- human pose

这让它既能做单条件解释，也能处理组合控制。

## 实验

### 设置

- 初始化骨干：Qwen2.5-Omni
- 附加模块：camera encoder
- Stage 1 数据：8.4K
- Stage 2 数据：8.3K
- Stage 1 学习率：1e-5，cosine scheduler
- Stage 2 学习率：2.5e-6
- 每个 prompt 8 次 rollout
- 评测基准：FullDiT 体系下的单条件 / 多条件控制集，以及 VBench

### 单条件结果

论文分别在 multiple identities、depth、camera、human pose 条件下比较：

- 原始短 prompt
- Any2Caption
- ReaDe

典型结果：

- **多身份控制**：ReaDe 的 `CLIP-T / DINO-I / Smoothness / Aesthetic` 都最高，说明它既更对题，也更保 identity。
- **深度控制**：相比 Ctrl-Adapter / Any2Caption，ReaDe 在文本对齐、深度误差和整体质量上更稳。
- **镜头控制**：引入 camera caption 字段后，旋转和平移误差更低。

### 多条件组合结果

多条件才是这篇论文的关键。`Camera + Depth + Identities` 组合下：

| 方法 | CLIP-T | RotErr | TransErr | DINO-I | MAE | Smoothness | Dynamic | Aesthetic |
|------|--------|--------|----------|--------|-----|------------|---------|-----------|
| FullDiT | 18.49 | 2.05 | 7.74 | 35.86 | 18.37 | 92.02 | 30.09 | 3.91 |
| Any2Caption | 19.52 | 1.57 | 7.74 | 38.74 | 17.41 | 93.03 | 32.81 | 4.99 |
| **ReaDe** | **21.24** | **1.34** | **5.28** | **39.46** | **17.03** | **95.04** | **33.47** | **5.21** |

这里可以看出，ReaDe 的收益不只是“caption 更长了”，而是**真的把多源控制信号解释成了更一致的生成约束**。

### 消融：CoT 和 reward 缺一不可

论文比较了：

- 只有 CoT 初始化
- 只有 GRPO
- CoT + GRPO

结论：

- **CoT-only** 已经能建立较强基线，因为它提供了结构化 reasoning prior
- **GRPO-only** 更不稳定，缺少初始推理能力
- **CoT + GRPO** 最好，说明 RL 更适合在“已经会解释”的基础上继续打磨，而不是从零学会解释

奖励消融还表明：

- `R_user` 和 `R_detail` 贡献最大
- `R_supp` 帮助补全合理细节
- `R_contra` 能降低内部矛盾、提升 smoothness

### 泛化分析

论文给出跨条件泛化热图：在某一类条件上训练，迁移到其他条件时，ReaDe 仍保持较高 intention accuracy。这说明模型学到的不是特定任务模板，而是更一般的“多模态意图解释”能力。

## 关键启示

- **把用户指令翻译成模型友好的 structured caption，本身就是一个独立问题。**
- 视频生成里的 prompt/interpreter 层，和图像里的 PromptEnhancer/APE 类似，都是值得单独训练的模块。
- **CoT 在这里不是为了炫推理，而是为了把多模态约束显式拆开。**
- 对开放式 caption 任务，reward 不该只看最终视频质量，还要看**是否覆盖用户约束、其他模态条件和合理补充细节**。
- 多条件 controllable generation 的瓶颈，经常不是下游生成器能力不足，而是上游解释器没有把控制条件对齐好。
