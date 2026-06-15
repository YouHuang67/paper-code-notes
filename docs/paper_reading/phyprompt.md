---
tags:
  - Video Generation
  - Reinforcement Learning
  - GRPO
  - Post Training
---

# PhyPrompt: RL-based Prompt Refinement for Physically Plausible Text-to-Video Generation

- 论文：https://arxiv.org/abs/2603.03505
- 代码：未开源
- 团队：Northwestern University, Dolby Laboratories

## 概述

PhyPrompt 的核心判断很直接：很多 T2V 模型并不是完全不会生成符合物理规律的视频，而是**用户 prompt 没把物理约束说清楚**。同一个生成器面对 “把酒倒进杯子” 这种简短描述时容易忽略液面上升、碰撞反馈、受力方向等关键细节；如果人工把这些物理过程写明，模型往往能生成明显更合理的结果。问题因此从“改视频模型”转成了“自动把用户 prompt 改写成物理感知 prompt”。

PhyPrompt 用一个两阶段训练的 LLM 改写器解决这个问题。第一阶段先用物理领域 CoT 数据做 SFT，让模型学会识别场景中的运动、受力、因果和约束；第二阶段固定视频生成器，只优化改写器本身，用 GRPO 直接根据生成视频的物理常识分数和语义一致性分数回传奖励。最关键的是它没有用固定权重把两个目标线性相加，而是用一个**动态课程 reward**：训练前期优先保语义，后期再逐步提高物理常识权重。

论文最重要的实验结论有两个。第一，动态课程不是简单折中，而是真的发现了“先搭语义骨架，再补物理细节”的更优 prompt 结构；在 VideoPhy2 上，PhyPrompt-7B 的 joint success 达到 **40.8%**，比原始 prompt 提升 **8.6 个点**，同时 SA 从 **43.4% → 47.8%**、PC 从 **55.8% → 66.8%**。第二，这个改写器是生成器无关的：只在 CogVideoX-2B 上训练，零样本迁移到 Lavie、VideoCrafter2、CogVideoX-5B 仍然稳定提升。

## 动机

- 现有 T2V 模型常见失败不是画质差，而是**违反重力、碰撞、液体累积、因果反馈**。
- 训练数据里的 caption 往往比真实用户输入详细得多，推理阶段的短 prompt 形成明显分布偏移。
- 人工补充 “force”, “trajectory”, “accumulation”, “reaction” 之类物理细节时，模型输出通常会立刻变好，说明瓶颈在 prompt 而非纯模型容量。
- 纯 few-shot prompt rewriting 不能系统优化物理合理性；只追求物理分数又会牺牲语义保真。

## 方法

### 整体流程

PhyPrompt 的训练和推理链路可以压成 5 步：

1. 输入用户原始 prompt `x`
2. 改写器 `πθ` 生成多个 physics-aware prompt `y`
3. 冻结的 T2V 生成器 `G` 根据 `y` 生成视频 `v = G(y)`
4. VideoPhy2-AutoEval 给每个视频打两个分数：语义一致性 `r_sa` 和物理常识 `r_pc`
5. 用 GRPO 更新改写器，不改动视频生成模型

这意味着它本质上是一个**外挂式 prompt optimizer**，部署时只需要放在用户和生成器之间。

### Stage 1：物理 CoT 数据 + SFT

论文先从 PhyGenBench 的 160 个物理场景出发，构造三元组：

- 原始 prompt `x`
- 物理规律或推理链 `r`
- 强化后的物理感知 prompt `y_CoT`

这些数据教模型两件事：

- 识别场景涉及的物理规律：重力、冲量、接触、液体积累、运动连续性
- 把这些规律翻译成生成模型更容易遵循的自然语言描述

SFT 目标是标准交叉熵：

$$
\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x,y)} \log \pi_\theta(y|x)
$$

这一步相当于先让模型学会“怎么写物理 prompt”。

### Stage 2：GRPO 直接优化视频效果

SFT 后的改写器继续进入 RL 阶段。对每个 prompt，模型采样 `G=4` 个候选改写，分别送入固定的 CogVideoX-2B 生成视频，再由 VideoPhy2-AutoEval 打分。

组内优势定义为：

$$
A_i^{(j)} = r_i^{(j)} - \bar{r}^{(j)}
$$

再用 clipped GRPO 更新策略，其中论文使用：

- clipping `ϵ = 0.2`
- KL 正则约束回 SFT 初始化

这一步的含义很清楚：**真正决定改写质量的不是文本看起来高级不高级，而是生成出来的视频是否更符合物理和语义要求**。

### 动态多目标奖励

这篇论文最值得记的是 reward 设计。它不用固定权重，而是令：

$$
R(t) = w_{sa}(t)\, r_{sa} + w_{pc}(t)\, r_{pc}
$$

并设：

$$
w_{sa}(t)=\exp(-\alpha t/T), \quad w_{pc}(t)=1-w_{sa}(t)
$$

训练含义：

- **前期**：`w_sa` 大，先学会不要改坏用户意图，搭出正确的对象、关系和场景框架
- **后期**：`w_pc` 大，再往这个骨架里加入物理因果和动态约束

论文的解释是，语义结构像 scaffold，物理约束像 refinement。只做其中一个目标都会负迁移：

- SA-only 会写出语义通顺但物理空洞的 prompt
- PC-only 会堆很多物理词，但可能偏离用户原意或破坏叙事结构

## 实验

### 设置

- 改写器骨干：Qwen2.5-Instruct 1.5B / 3B / 7B
- 训练时固定生成器：CogVideoX-2B
- 评测生成器：Lavie、VideoCrafter2、CogVideoX-2B、CogVideoX-5B
- 统一输出：6 秒、4 FPS、720×480
- 自动评测：VideoPhy2-AutoEval，输出 SA 和 PC 两个 1-5 Likert 分数

### 主结果：VideoPhy2 上同时提升 SA 和 PC

论文最强调 joint 指标。7B 版本在 CogVideoX-2B 上达到：

| 方法 | SA | PC | Joint SA&PC |
|------|----|----|-------------|
| 原始 Prompt | 43.4% | 55.8% | 32.2% |
| GPT-4o | 47.0% | 60.0% | 37.0% |
| DeepSeek-V3 | 48.4% | 64.6% | 38.6% |
| **PhyPrompt-7B** | **47.8%** | **66.8%** | **40.8%** |

这里最关键的不是单项最好，而是它把两个通常互相拉扯的指标一起抬高了。

### 零样本迁移

只在 CogVideoX-2B 上训练的 7B 改写器，迁移到其他生成器后 joint 指标仍然提升：

| 生成器 | 原始 Prompt | PhyPrompt-7B | 提升 |
|------|-------------|--------------|------|
| Lavie | 29.2% | 31.6% | +8.2% |
| VideoCrafter2 | 29.8% | 34.8% | +16.8% |
| CogVideoX-5B | 39.4% | 42.0% | +6.6% |

这说明它学到的是**跨模型共享的物理约束表达方式**，不是针对某个后端的 prompt hack。

### 消融：动态 reward 真正优于静态权重

论文比较了四种策略：

- SA-only
- PC-only
- 静态 0.5 / 0.5 加权
- 动态课程 reward

结论是：

- SA-only 会把 SA 拉高，但 PC 掉下来
- PC-only 反过来
- 静态加权比单目标稳，但上限不高
- **动态课程训练 reward 收敛更快，最终 plateau 也更高**

作者还给了 “hammer hits nail” 的例子：静态 reward 学到的是泛化的物理字眼，动态 reward 则更容易明确补出 “nail”、“wood plank”、“force concentrated” 这类既保语义又补物理机制的描述。

## 关键启示

- **视频物理一致性有相当一部分是 prompt 问题，不一定先动生成模型。**
- **先保语义、再补物理** 比同时硬优化两个目标更有效，因为物理细节需要依附在正确的语义骨架上。
- **视频级反馈非常重要**。如果 reward 不看最终视频，只看改写文本，很容易学成“看起来专业但对生成无效”的 prompt。
- **prompt 改写器可以跨生成器迁移**，所以它是很实用的外插模块。
- 这篇工作和 VPO 的共同点是都在 prompt 空间做 RL；不同点在于 VPO 关注 harmless/accurate/helpful，PhyPrompt 则把 reward 明确压到 physical commonsense 上。
