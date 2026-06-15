---
tags:
  - Prompt Enhancer
  - Video Generation
---

# SCMAPR: Self-Correcting Multi-Agent Prompt Refinement for Complex-Scenario Text-to-Video Generation

- 论文：https://arxiv.org/abs/2604.05489
- 代码：https://github.com/HiThink-Research/SCMAPR
- 团队：HiThink Research, East China Normal University, Guangzhou University, Tsinghua University

## 概述

SCMAPR 处理的是一个很具体、但常被混在“prompt 优化”里笼统讨论的问题：**复杂场景 prompt 不是简单写长一点就能变好**。当用户想生成抽象语义、复杂空间关系、多元素交互、长程时间依赖、镜头切换或风格混合的视频时，困难不只是“信息不够多”，而是这些约束彼此耦合，随便展开描述反而更容易漏语义、添幻觉、破坏时序一致性。

这篇论文的做法是把 prompt refinement 显式拆成一个多阶段、多代理的自纠错流程。SCMAPR 不是让一个 LLM 直接重写，而是先给 prompt 打上“复杂场景类型”标签，再按类型生成一条 prompt-specific policy，再据此重写，然后把原 prompt 原子化成 characters / objects / actions / locations / scenery 五类语义原子，对重写结果做 atom-level entailment 验证。只要发现缺失或矛盾，就触发 revision。整个框架的重点不是“多 agent 很酷”，而是它把**策略生成、改写执行、语义验收**三件事拆开了。

论文同时提出了一个新的 benchmark：**T2V-Complexity**。作者认为现有 VBench、EvalCrafter、T2V-CompBench 虽然包含复杂样本，但分布不均衡，很多复杂类别覆盖不足，因此很难系统评估 prompt refinement 对复杂场景的价值。T2V-Complexity 只保留复杂场景，共 1000 条 prompt，按 10 类复杂场景均衡采样。

实验上，SCMAPR 在 LaVie 和 Wan 两个 T2V 后端上都优于 direct prompting、Open-Sora prompt refiner 和 RAPO。在 VBench 上平均分最高达到 **88.21%**，在 EvalCrafter 上达到 **66.74**，在 T2V-CompBench 上达到 **0.523**，在只含复杂 prompt 的 T2V-Complexity 上也稳定领先。

## 动机

- 复杂场景 T2V 的难点不是单一属性补全，而是**多类约束耦合**：
  - 抽象语义要落成具体镜头
  - 多实体关系要兼顾空间和动作
  - 时间变化和镜头切换要显式规划
  - 物理、因果和风格约束可能互相影响
- 单次重写容易出现两类错误：
  - **semantic missing**：原 prompt 明说的东西在 refined prompt 里丢了
  - **semantic contradiction**：重写时加入了和原意冲突的内容
- 现有 benchmark 虽然有复杂 prompt，但类别严重不平衡，难以判断方法到底擅长哪类复杂性。

## 方法

### 复杂场景 taxonomy

SCMAPR 先定义了一个 11 标签的路由集合：

- 1 个 `non-difficult` 保底标签
- 10 个复杂场景标签：
  - Abstract Descriptions
  - Complex Spatial Relations
  - Multi-Element Scenes
  - Fine-Grained Appearance
  - Temporal Consistency
  - Stylistic Hybrids
  - Causality & Physics
  - Camera Motion
  - Object Interaction
  - Scene Transitions

这个 taxonomy 的用途，不是做 benchmark 装饰，而是作为后续 policy synthesis 的路由信号。

### 五阶段流程

#### Stage I：Scenario Routing

先用 `Scenario Router` 给用户 prompt `P_user` 分配一个场景标签 `ŷ`。它本质上回答：

- 这个 prompt 最主要的难点是什么？
- 应该按哪类复杂性去设计改写策略？

标签只取一个 dominant scenario，避免一个 prompt 同时激活太多策略。

#### Stage II：Policy Synthesis

有了标签 `ŷ` 之后，`Policy Generator` 不直接改写，而是先生成一条针对当前 prompt 的重写 policy `π`。这条 policy 主要规定三件事：

- 哪些隐含约束要显式展开
- 哪些场景能力需要重点强调
  - 如 spatial layout、temporal coherence、physical plausibility、camera motion
- 如何在补全信息时仍然保持 fidelity，不引入无依据内容

论文特别强调：policy 不是固定模板，而是“场景标签 + 当前 prompt”条件下动态生成的 prompt-specific strategy。

#### Stage III：Policy-Conditioned Prompt Refinement

`Prompt Refiner` 根据：

- 原始用户 prompt `P_user`
- 动态生成的 policy `π`

输出 refined prompt `P_rew`。这一阶段只负责执行，不负责决定“该补什么策略”，因此比让单个 LLM 直接从头兼顾所有事情更稳。

#### Stage IV：Semantic Verification

这是这篇论文真正有辨识度的部分。

1. **Atomic Extraction**
   - 从 `P_user` 中提取语义原子字典：
     - characters
     - objects
     - actions
     - locations
     - scenery
2. **Chunking**
   - 把 refined prompt 切成句级语义块
3. **Atom-Chunk Matching**
   - 用 BGE-M3 embedding 做原子和 chunk 的相似度匹配，为每个 atom 找到最相关证据句
4. **Entailment Validation**
   - 让 validator 给每个 atom 打三值标签：
     - `ET`：entailment
     - `MS`：missing
     - `CT`：contradiction

这样一来，验证不再是“整体看起来差不多”，而是能精确定位哪个原子丢了、哪个地方改歪了。

#### Stage V：Conditional Revision

如果满足：

- `p_ET = 1`
- `p_CT = 0`

则 refined prompt 通过验收；否则把缺失/矛盾报告交给 `Content Reviser`，做定向 revision。这个循环直到通过验收或达到最大 revision 次数。

作者强调，这一步的目标不是无限次打磨，而是**只在必要时修复明确的 fidelity 问题**。

## T2V-Complexity Benchmark

论文认为现有 benchmark 的复杂样本比例和类别覆盖都不理想，因此新建 T2V-Complexity：

- 总计 **1000** 个 prompt
- 全部都是 complex-scenario prompt
- 10 个类别各 **100** 个

和现有基准相比：

| Benchmark | Prompt 数 | Complex Prompt 占比 | 覆盖类别数 |
|-----------|-----------|---------------------|-----------|
| VBench | 946 | 32.9% | 9 |
| EvalCrafter | 700 | 60.3% | 9 |
| T2V-CompBench | 1400 | 69.4% | 10 |
| **T2V-Complexity** | **1000** | **100%** | **10** |

这个 benchmark 的价值在于：它专门测试 prompt refinement 在真正复杂样本上的效果，而不是让结果被大量简单 prompt 冲淡。

## 实验

### 设置

- 多代理底座：DeepSeek-V3.2
- atom-chunk matching embedding：BGE-M3
- T2V 后端：Wan2.2（文中简称 Wan）与 LaVie
- 对比方法：
  - direct prompting
  - Open-Sora prompt refiner
  - RAPO

### VBench

SCMAPR 在两个后端上都拿到最好平均分：

| 方法 | LaVie Avg | Wan Avg |
|------|-----------|---------|
| Direct Prompting | 81.89 | 86.19 |
| Open-Sora | 81.95 | 86.27 |
| RAPO | 82.80 | 87.43 |
| **SCMAPR** | **84.56** | **88.21** |

相对 direct prompting：

- LaVie：+2.67
- Wan：+2.02

### EvalCrafter

| 方法 | LaVie Avg | Wan Avg |
|------|-----------|---------|
| Direct Prompting | 62.12 | 63.46 |
| Open-Sora | 62.78 | 63.95 |
| RAPO | 63.91 | 64.54 |
| **SCMAPR** | **65.18** | **66.74** |

这里优势主要体现在：

- text-video alignment
- temporal consistency
- visual quality

说明它并不只是把 prompt 变长，而是确实让生成器更容易跟住复杂约束。

### T2V-CompBench

SCMAPR 同样是最好：

| 方法 | LaVie Avg | Wan Avg |
|------|-----------|---------|
| Direct Prompting | 0.388 | 0.454 |
| Open-Sora | 0.361 | 0.446 |
| RAPO | 0.460 | 0.495 |
| **SCMAPR** | **0.476** | **0.523** |

尤其在：

- consistent attribute
- action binding
- motion binding

这些组合推理指标上更强。

### T2V-Complexity

在只保留复杂 prompt 的基准上，Wan 后端：

| 方法 | Average Score |
|------|---------------|
| Wan | 82.95 |
| **Wan + SCMAPR** | **85.69** |

这说明方法收益并不是靠“简单 prompt 也变长了”刷出来的，而是在真正复杂场景里依然成立。

### 消融

VBench 上的消融结果：

| 版本 | Average Score |
|------|---------------|
| **SCMAPR** | **88.21%** |
| w/o Scenario Routing | 86.49% |
| w/o Policy Generation | 87.75% |
| w/o Verification & Self-Correction | 87.63% |

可见三部分都有效：

- routing 决定该用什么策略，去掉后降得最多
- policy generation 让重写不只是“统一拉长”
- verification/self-correction 负责防止语义漂移

## 关键启示

- **复杂场景 prompt refinement 的本质不是扩写，而是结构化解耦约束。**
- 如果不做显式 verification，重写器很容易把 prompt 改得更丰富，但也更偏题。
- taxonomy + policy + verification 这一套，比单个大模型“一次改写到位”更可控。
- T2V prompt benchmark 里复杂样本分布严重不平衡，专门的复杂场景 benchmark 很有必要。
- SCMAPR 更像一个“prompt compiler + validator”，而不仅仅是 prompt beautifier。
