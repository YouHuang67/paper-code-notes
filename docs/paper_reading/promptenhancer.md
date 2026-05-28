---
tags:
  - Reinforcement Learning
  - Diffusion Model
  - GRPO
  - Post Training
---

# PromptEnhancer: Enhancing Text-to-Image Models via Chain-of-Thought Prompt Rewriting

**论文**: [arXiv 2509.04545](https://arxiv.org/abs/2509.04545)
**代码**: [Hunyuan-PromptEnhancer/PromptEnhancer](https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer)
**项目页**: [hunyuan-promptenhancer.github.io](https://hunyuan-promptenhancer.github.io)
**团队**: Tencent Hunyuan

## 概述

PromptEnhancer 解决的核心问题：**用户写的 prompt 太模糊，T2I 模型理解不了**。比如用户说"一只猫坐在椅子上"，模型可能画错猫的颜色、搞混椅子的材质、甚至把猫和椅子的位置关系画反。现有方案要么需要微调 T2I 模型本身（不通用），要么用 CLIP score 这类粗粒度信号做优化（只知道"整体不像"，不知道"哪里不像"）。

PromptEnhancer 的做法是训练一个 **Chain-of-Thought 改写器**，它先像人类 prompt engineer 一样思考——分析主体是谁、属性有哪些、空间关系如何、风格是什么、用户潜在意图是什么——再输出精炼后的详细 prompt。整个过程**不改 T2I 模型一行代码**，纯粹在 prompt 层面工作，因此可以即插即用到任何 T2I 模型上。

训练这个改写器的关键是一个叫 **AlignEvaluator** 的 reward 模型。它把"图文对齐"这个概念拆成了 **24 个细粒度检查点（keypoint）**，分属 6 大类：语言理解（否定词、代词、属性一致性）、视觉属性（计数、大小、材质、表情、风格）、动作交互（全身/手部/动物动作、接触/非接触交互、持续状态）、关系结构（比较/组合/包含/相似关系、实体属性绑定、空间布局）、知识推理（知名实体、反事实）、文字排版（文字内容准确渲染、文字位置准确）。AlignEvaluator 对每张生成图在这 24 个维度上逐项判断"匹配/不匹配"，然后分组聚合为标量 reward。

训练分两阶段：**SFT 初始化**（用 Gemini-2.5-Pro 蒸馏 48.5 万条 CoT 改写数据，学会格式和改写方向）→ **GRPO 策略对齐**（冻结 HunyuanImage 2.1 生成图像，AlignEvaluator 打分，reward 反馈优化改写策略，让改写器学会"什么样的改写真正能提升图文对齐"）。

在 HunyuanImage 2.1 上验证，24 个维度平均提升 **+5.1%**，20/24 个类别有提升。其中 Similarity Relation +17.3%、Counterfactual +17.2%、Counting +15.0%、Pronoun Resolution +13.9%，这些恰是 T2I 模型最容易出错的复杂推理维度。同时开源了 **T2I-Keypoints-Align** 评测基准（6,687 条中英双语 prompt，每条标注多个 keypoint），以及 PromptEnhancer 改写器权重。


## 1. Introduction

### 1.1 问题定义：Prompt Alignment Gap

当前 T2I 扩散模型（Imagen, Stable Diffusion, SDXL, HunyuanDiT, FLUX.1, Qwen-Image 等）在图像质量上已相当成熟，但**生成准确性高度依赖用户写 prompt 的能力**。核心矛盾在于：

- **用户的输入习惯**：简短、模糊、口语化（"帮我画一只猫坐在椅子上"）
- **模型的理解需求**：需要明确的主体、属性、空间关系、风格、细节描述

这个 **prompt alignment gap** 体现在多个维度：

- **属性绑定错误（Attribute Binding）**："红色帽子和蓝色衣服的人" → 模型可能把红色画到衣服上
- **否定词忽略（Negation）**："一碗牛肉面，不加葱" → 模型还是画了葱
- **空间/逻辑关系混乱**："球在桌子上" → 球被画到了桌子下方
- **计数不准（Counting）**："四只狗" → 画了三只或五只
- **组合关系失败（Compositional）**："橙子切片做成的猫" → 橙子切片摆在猫旁边，而非猫由橙子切片构成
- **代词指代歧义（Pronoun Resolution）**："大球打破了桌子，因为它是金属做的" → 模型无法判断"它"指的是球还是桌子
- **风格/材质渲染不准**："冰雕的老鹰" → 画成了普通白色雕塑，没有冰的透明感

### 1.2 现有方法的局限

**Recaptioning 类（DALL-E 3）**：用高质量图像描述重新训练 T2I 模型。问题：需要访问和修改 T2I 模型权重，每换一个模型就要重新训练，不通用。

**On-the-fly Prompt Rewriting 类**：用 LLM 改写用户 prompt 再送入 T2I 模型。
- 迭代精炼（Idea2Img, Self-Correcting LLM）：LLM 反复 critique → revise，计算成本高
- 自动 prompt 工程（BeautifulPrompt）：学习生成"风格化 prompt"，但偏美学而非语义准确性
- Agent 系统（GenArtist, DiffusionGPT）：多模态 agent 做规划+生成+编辑，流程复杂

共性问题：依赖通用 LLM（GPT-4 等），这些 LLM 缺乏对 T2I 特定失败模式的专业理解。它们知道"怎么写一段好文字"，但不知道"什么写法能让 T2I 模型不犯错"。

**RL-based 类**：用 CLIP score、ImageReward、HPSv2 作为 reward 信号优化改写器。问题：这些是粗粒度的整体分数——告诉你"这张图总体好不好"，但无法告诉你"计数错了"还是"颜色错了"还是"空间关系错了"。缺乏诊断能力的 reward 信号导致优化方向模糊。

### 1.3 PromptEnhancer 的应对策略

三个核心设计决策：

1. **完全解耦改写器与生成器**：改写器不感知 T2I 模型内部，只关心"怎么把模糊 prompt 写清楚"。T2I 模型对它来说是一个黑盒——输入 prompt，输出图像。这带来通用性：任何 T2I 模型直接接入，零修改成本。

2. **CoT 推理驱动改写**：改写器不是"翻译"（短 prompt → 长 prompt），而是先**思考再改写**。思考过程包括：识别核心主体 → 分析属性约束 → 推理空间关系 → 判断潜在意图 → 确定合适风格 → 综合输出。这个 `<think>` 推理链是训练出来的，不是 hard-coded 的模板。

3. **细粒度 reward 替代粗粒度分数**：AlignEvaluator 的 24 个 keypoint 覆盖了 T2I 模型最常见的失败模式。每个 keypoint 是二值的（匹配/不匹配），分组聚合为最终 reward。这种结构化反馈让改写器能精确学习"哪个改写方向是对的"。

## 2. Methodology

### 2.1 整体框架（Fig. 2）

```
User Prompt → CoT Rewriter → Reprompt → Frozen T2I → Image
                                              ↓
                         AlignEvaluator ← (Image, User Prompt)
                                              ↓
                                         Scalar Reward → GRPO Update
```

三大组件 + 两阶段训练：

- **CoT Rewriter（PromptEnhancer）**：基于 Hunyuan-7B-Instruct 的策略模型 πθ，输入用户原始 prompt，输出 `<think>推理过程</think>精炼 reprompt`
- **AlignEvaluator**：预训练的细粒度 reward 模型 R，对 (Image, User Prompt) pair 做 24 维 keypoint 匹配判断，分组聚合输出标量 reward
- **Frozen T2I Model**：HunyuanImage 2.1，权重完全冻结，仅负责"接收 reprompt → 输出图像"和"提供 reward 信号的计算载体"

训练目标是优化 πθ 使得给定用户 prompt p，改写后的 reprompt p' 让 T2I 模型生成的图像 I 获得 AlignEvaluator 的最高评分 R(I, p)。

### 2.2 Stage 1: SFT 初始化改写器

**为什么需要 SFT**：Hunyuan-7B-Instruct 是一个通用指令模型，它不会"CoT 分析 prompt → 输出精炼 reprompt"这个特定格式。SFT 的作用是教会模型这个格式和基本的改写方向，为后续 GRPO 提供合理的初始化策略空间。

**训练目标**：标准 causal language modeling loss（next-token prediction）：

$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(p, c, r) \sim \mathcal{D}_{\text{SFT}}} \log \pi_\theta(c, r \mid p)$$

其中 p 是用户 prompt，c 是 CoT 推理链（`<think>...</think>`），r 是精炼 reprompt。模型学习的是给定 p，生成 c 然后 r 的条件概率。

**数据格式**（每条训练样本的结构）：
```
User Prompt: [简短的用户输入]
<think>
[CoT 推理链：核心元素分析 → 属性深化 → 构图关系 → 风格判断]
</think>
Reprompt: [结构化的精炼 prompt]
```

**关键设计**：SFT 只教格式和方向，不保证改写后的 prompt 真的提升图文对齐——这个问题留给 GRPO 解决。SFT 的定位是**强初始化**而非最终优化。

### 2.3 Stage 2: GRPO 策略对齐

**为什么需要 GRPO**：SFT 后的改写器生成的是"看起来专业"的长 prompt，但 T2I 模型用这些 prompt 生成的图像未必真的更符合用户意图。可能出现的情况：改写后的 prompt 文字很长很漂亮，但生成的图像反而更差了——这叫 **reward hacking in prompt space**。GRPO 的信号来自 AlignEvaluator 对**实际生成图像**的评分，闭环验证了"改写是否真的有效"。

**GRPO 训练循环**（每步的完整流程）：

1. **采样用户 prompt**：从 RL prompt 集（~50K 条，与 SFT 数据完全不重叠）中取一个 batch
2. **Rollout 生成**：对每条 prompt p，当前策略 πθ 生成 **N=8 个候选 reprompt** {p'_1, ..., p'_8}（带随机采样，保证多样性）
3. **图像生成**：每个 p'_i 送入冻结的 T2I 模型，生成图像 I_i
4. **Reward 计算**：AlignEvaluator 对每对 (p, I_i) 计算 reward r_i = R(I_i, p)
5. **优势估计**：在组内（8 个候选）计算相对优势 A_i = (r_i - mean(r)) / std(r)（组内归一化）
6. **策略更新**：用 GRPO 目标函数更新 πθ：
   $$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta(p'_i|p)}{\pi_{\text{old}}(p'_i|p)} A_i,\; \text{clip}\left(\frac{\pi_\theta(p'_i|p)}{\pi_{\text{old}}(p'_i|p)}, 1-\epsilon, 1+\epsilon\right) A_i\right) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$
   其中 π_old 是 rollout 时的策略（已固定），π_ref 是 SFT 模型，β=0.001 控制 KL 惩罚强度

**GRPO 的核心优势**：
- **闭环比对**：reward 基于"实际生成的图像"而非"prompt 文本本身"，防止改写器产生看似漂亮但无效的 prompt
- **组内相对比较**：组内归一化消除绝对分数的尺度漂移，训练更稳定
- **KL 约束**：防止策略偏离 SFT 初始化太远，避免 reward hacking（生成某种固定模板骗过高分）
- **无需成对偏好标注**：reward 由 AlignEvaluator 在线提供，不需要人工标注"A 比 B 好"

**GRPO 的计算成本分析**：
- 每步的主要开销不在策略更新（7B 模型的梯度计算），而在**图像生成**（HunyuanImage 2.1 前向推理 × 8）和**AlignEvaluator 评分**（24 维 × 8 次匹配判断）
- 这也是为什么 RL prompt 集只有 50K 条（而非 SFT 的 485K）：每条 prompt 要做 8 次完整的 T2I 生成

### 2.4 AlignEvaluator: 细粒度 Reward 模型的完整机制

**设计哲学**：一个好的 reward 模型不应该告诉你"这张图打 7.5 分"，而应该告诉你"计数 ✓，风格 ✓，否定词 ✗，空间关系 ✗"。只有结构化的细粒度反馈，才能指导改写器精准改进。

**Keypoint 匹配流程**（Fig. 2 下半部分）：
1. **提取 keypoint 描述**：从用户 prompt p 中自动提取每个相关 keypoint 对应的事实描述（KeyPoints Desc）
   - 例如对于"一只企鹅穿宇航服漂浮在太空，油画风格"：
     - Artistic Style → "油画风格，笔触厚重，纹理感强，颜料堆叠明显"
     - Counterfactual → "企鹅...穿宇航服...在太空中漂浮"（企鹅本不该在太空）
     - Cross-Entity Binding → "身体(白色和黄色颜料，圆形)，翅膀(黑色)，宇航服(白色，水平笔触)，头部(透明黄色玻璃舱)，围巾(鲜红色，飘动)"
     - Full-body Action → "企鹅在漂浮漫游，围巾飘动"
     - Pronoun Resolution → "它的翅膀"和"它的头"中的"它"指的是企鹅
2. **逐项匹配判断**：对每个 keypoint，判断图像是否满足对应的描述（Match or Not?）
3. **分组平均**：6 大类分别计算匹配率，再平均得到最终 reward（Group-wise Final Score = Average Score）

**24 个 Keypoint 完整分类体系**：

**Category 1: Linguistic Comprehension（语言理解）**
评估模型对核心语言结构的理解。评估准则：TIC（Text-Image Consistency）。

| # | Keypoint | 测试内容 | Prompt 示例 |
|---|----------|---------|------------|
| 1 | Negation | 正确执行否定指令 | "A bowl of beef noodles, **no scallions**" — 图中不应出现葱 |
| 2 | Attribute Consistency | 一个属性同时绑定多个实体 | "Five people **all wearing red clothes**" — 五个人必须都穿红色 |
| 3 | Pronoun Resolution | 代词指代消歧 | "The large ball broke the table because **it** was made of metal" — 正确理解为 it=ball |

**Category 2: Visual Attributes（视觉属性）**
评估模型对视觉属性的渲染能力。评估准则：TIC。

| # | Keypoint | 测试内容 | Prompt 示例 |
|---|----------|---------|------------|
| 4 | Counting | 精确计数（n ≥ 3） | "A picture with **four** dogs" — 必须是四只，不能是三只或五只 |
| 5 | Size | 相对大小关系 | "Two **large** spheres" — 球体必须明显大于参照物 |
| 6 | Material | 材质渲染 | "An **ice sculpture** of an eagle" — 必须呈现冰的透明/半透明质感 |
| 7 | Expression | 面部表情捕捉 | "A strong man with a **contemptuous expression**" — 必须是轻蔑的表情 |
| 8 | Artistic Style | 艺术风格遵循 | "Eight galloping horses in **Chinese ink wash**" — 必须是中国水墨风格 |

**Category 3: Action & Interaction（动作与交互）**
评估模型对动态行为和物理交互的描绘。评估准则：TIC&SI（部分含 Structural Integrity）。

| # | Keypoint | 测试内容 | Prompt 示例 | 准则 |
|---|----------|---------|------------|------|
| 9 | Full-body Action | 复杂全身动作 | "A girl performing a **Thomas flare**"（体操动作） | TIC&SI |
| 10 | Hand Action | 手部精细动作 | "A hand **using chopsticks** to pick up food" | TIC&SI |
| 11 | Animal Action | 动物的特征动作 | "A puppy **happily running**" | TIC&SI |
| 12 | Contact Interaction | 物理接触交互 | "A boxer **lands a punch on** a punching bag" | TIC&SI |
| 13 | Interaction w/o Contact | 非接触性交互 | "Einstein **looking at** Hawking" — 视线方向正确 | TIC |
| 14 | State | 持续状态/氛围 | "A gust of wind blows, cherry blossoms **dance in the air**" | TIC |

**Category 4: Relations & Structure（关系与结构）**
评估模型对实体间逻辑关系和空间结构的理解。评估准则：TIC。

| # | Keypoint | 测试内容 | Prompt 示例 |
|---|----------|---------|------------|
| 15 | Comparative Relation | 属性比较关系 | "Woman in red dress **taller than** woman in yellow" |
| 16 | Compositional Relation | 实体由其他实体组成 | "A cat **made of** orange slices" — 猫本身由橙片构成 |
| 17 | Containment Relation | 包含/容纳关系 | "A cup **full of** soda water" — 苏打水在杯子里 |
| 18 | Similarity Relation | 形状相似性 | "A lake **shaped like** a guitar" — 湖的轮廓是吉他形 |
| 19 | Cross-Entity Binding | 多实体+多属性绑定 | "Man (**buzz cut, blue shirt**) and woman (**long hair, yellow shirt**)" |
| 20 | Entity Layout | 实体空间位置安排 | "A race car on a city track, with a **mini-map in the top-left corner**" |

**Category 5: World Knowledge & Reasoning（知识推理）**
评估模型对常识和反事实推理的能力。评估准则：TIC。

| # | Keypoint | 测试内容 | Prompt 示例 |
|---|----------|---------|------------|
| 21 | Knowledge Application | 知名实体/人物的正确渲染 | "**The Great Wall of China**" / "**Marie Curie**" |
| 22 | Counterfactual | 超现实/反事实场景 | "A girl held onto the stem of a **huge dandelion**, **suspended above the clouds**" |

**Category 6: Scene Text & Typography（文字排版）**
评估模型对画面中文字的渲染能力。评估准则：TIC。

| # | Keypoint | 测试内容 | Prompt 示例 |
|---|----------|---------|------------|
| 23 | Text Rendering | 文字内容准确渲染 | "Poster with text **'Game of Thrones'** at the bottom" — 文字拼写正确 |
| 24 | Text Layout | 文字位置准确 | "Poster of a woman on a throne of waves, text 'Game of Thrones' **at the bottom**" — 文字必须在底部 |

**评估准则**：TIC = Text-Image Consistency（图文一致性），SI = Structural Integrity（结构完整性）。动作交互类的 keypoint 同时关注 TIC 和 SI，因为复杂动作既要求语义正确（在做什么），也要求结构合理（身体各部位关系正确）。

**AlignEvaluator 的训练**：论文提到 AlignEvaluator 是"在大规模 (reprompt, image) pair 数据集上训练，每个 pair 标注了 24 个 keypoint 的分数"，但未公开 AlignEvaluator 的具体架构和训练细节。从 Fig. 2 的使用方式来看，它接受 (image, user_prompt) 作为输入，输出 24 个 keypoint 的匹配判断和聚合 reward。推测其架构可能类似于一个 VLM（视觉语言模型），用图像+文本作为输入，输出每个 keypoint 的二分类（匹配/不匹配）。


## 3. Data Pipeline（Fig. 3）

这是整个系统最关键的工程部分。SFT 数据的质量直接决定了改写器的改写品位和方向感——如果 teacher 模型生成的 CoT/reprompt 质量差，改写器学到的是"一本正经地胡说八道"。

### 3.1 SFT 数据构造：四阶段流水线

**数据量变化链路**：
```
3.26M 图像 → 2.26M 代理 prompt → ~1M CoT+reprompt → 611,921 自动过滤 → 485,119 人工筛选（最终）
```

**Stage 1: User Prompt 模拟**

出发点是一个包含 **3.26M 图像**的多样化图像池：
- 中文场景：153 万张
- 英文场景：173 万张

用图像 captioning 模型对这些图像生成**简短、自然化的描述**，模拟真实用户写 prompt 的风格。关键设计意图：
- 生成的描述要"像用户写的"而不是"像机器写的"——简短、可能有歧义、不专业
- 这些 prompt 是 PromptEnhancer 的目标输入类型：模糊的、需要被改写的
- 产出：**2.26M 条**代理用户 prompt（proxy user prompts）

**Stage 2: CoT 与 Reprompt 生成**

Teacher 模型选择：
- 英文：**Gemini-2.5-Pro**（Google 最新旗舰模型，强推理能力）
- 中文：**DeepSeekV3**（中文理解和生成能力强）
- 分开选择的原因：不同语言场景下模型的能力差异显著，中文 prompt 的改写需要理解中国文化元素（如水墨画、古诗意境），英文 prompt 涉及更多西方艺术概念

系统 Prompt 设计（Fig. 10, Fig. 11）：论文公开了两个核心系统 prompt，设计非常精细：

**Reprompt 生成系统 Prompt（Fig. 10）**：
定义了四层描述层级 + 七条语法规则 + 九条关键约束：

**I. 四层句子结构**：
1. 开篇（General Overview）：一句话概括画面主体
2. 正文（Systematic Spatial Description）：系统化、按空间顺序组织的主体描述
3. 层级化对象描述（Hierarchical Object Description）：从整体到局部的层次化细节
4. 结语（Stylistic Identification）：识别并陈述整体风格

**II. 七条语法规则**：
1. 时态：始终使用现在时
2. 语态：主动/被动混合使用
3. 介词短语精确化：大量使用"in the background", "on the left side"等精确空间定位
4. 分词短语高效表达：用"wearing a red scarf, fluttering in the wind"压缩信息
5. 丰富的特定形容词：不用"nice"、"beautiful"等模糊词，用"metallic blue"、"thick brushstrokes"等精确描述
6. 精确与模糊语言的平衡：该精确的精确，无法确定的用"appears to be", "suggests"等保留余地
7. 复杂复合句：用从句连接相关信息，避免短句堆砌

**III. 九条关键约束**：
1. 只输出最终 caption，不使用 markdown 格式
2. 扩展后的 caption 必须遵循上述结构规则
3. 扩展后的 caption 必须忠于原文，特别是主体和主体属性（颜色、大小、空间关系等）
4. 可以用世界知识将专业术语扩展为适合图像生成的解释
5. 如原文未指定风格，默认假设为摄影风格；可根据内容推断更合适的风格
6. 直接描述场景或主体，不以"The image"、"The composition"、"The scene"等词开头
7. 除非原文明确说是照片，否则不要假设是照片
8. 如有 IP 角色，保留 IP 角色身份，在扩展 caption 中描述其背景
9. 如有需要渲染的文字，保留文字并以"rendered text"格式处理

**CoT 生成系统 Prompt（Fig. 11）**：

核心任务：不是输出最终答案，而是生成一个**推理链**，解释如何从用户 prompt（输入）推导出精炼 prompt（输出）。

关键要求：
- 必须聚焦于分析维度：核心元素、构图与相对关系、属性（大小/数量/材质/表情等）、动作（全身/局部/接触/状态等）、语法（否定/代词等）、风格、逻辑关系、潜在用户意图、背景推理、世界知识
- **禁止泄露精炼 prompt 中的信息**：CoT 是"推导过程"，不是"答案预览"
- **禁止引入用户 prompt 中没有的内容**：如果用户没提到文字/水印，CoT 中不能出现
- 输出长度严格限制在 **384 tokens** 以内

示例输出（从论文中提取）展示了 CoT 的标准格式：
```
The user wants to generate an image with the following core elements:
Person: young woman; Clothing: brown hoodie; Accessories: ski goggles;
Props: red snowboard. The action is left hand on hip, style is realistic,
background is the capital of China and the national flower of China.
The main element is a young woman, attributes include single person,
East Asian young woman, approximately 20 years old, with long brown
wavy hair, smiling at the camera...
```

**一对多生成策略**：每条 prompt 生成**多个候选 reprompt**（而非单个），目的是探索不同的改写方向和风格，后续通过人工筛选选出最优方向。这是数据质量的关键保障——单一改写方向可能不是最优的，多个候选中选优能显著提高数据上限。

初期产出约 **100 万条**实例（包含 CoT 推理链 + 多个候选 reprompt）。

**Stage 3: 自动过滤**

用 Gemini-2.5-Pro 程序化检测质量问题。三条过滤标准：
- **语义偏移或信息丢失**：reprompt 偏离了用户原意，或遗漏了关键信息元素
- **语言流畅性**：reprompt 是否自然通顺，有无语法错误或表达生硬
- **偏见或不可验证的声明**：reprompt 是否引入了不存在于用户 prompt 中的事实性断言

过滤结果：100 万 → **611,921 条**三元组（过滤率 **38.8%**）

这个过滤率相当高，说明 teacher 模型的初始生成质量波动很大。近 40% 的输出存在语义偏移、信息丢失或语言问题。这反过来证明了自动过滤的必要性——如果直接用未过滤的数据训练，改写器会学到大量低质量模式。

**Stage 4: Human-in-the-Loop 筛选**

这是质量把控的最后一道关，也是最贵的。

流程：
1. 对每个候选 reprompt，用 Hunyuan T2I 模型生成对应图像
2. **专业标注员**评估图像质量，比较多个候选 reprompt 的效果差异
3. 选择最符合用户意图、产出最高质量视觉结果的 reprompt
4. 最终筛选出 **485,119 条**高质量三元组（从 611,921 中再筛选，筛选率 79.3%）

每一行为什么必要：自动过滤能识别"明显不对的"（语义偏离、不流畅），但无法判断"改写好还是不好"——这需要人类看实际生成的图像来判断。LLM 可以判断 text-text 的一致性（user prompt vs reprompt），但需要人类来判断 text-image 的一致性（reprompt 生成的图像好不好）。

**最终数据格式**：每条三元组 = (user prompt → `<think>CoT 推理链</think>` → 最优 reprompt)

**数据集主题分布（Fig. 4）**：

五大一级类别 + 二十个子类别：

| 一级类别 | 占比 | 主要子类别 | 说明 |
|---------|------|-----------|------|
| **Design** | 27% | IP Design 3%, Logo/Icon 4%, Poster 4%, Space Design 3%, Software UI 2%, Ad/E-commerce 4%, Fashion 3%, Game 3%, Design Materials 1% | 设计类占比最高，覆盖平面/UI/广告/时尚 |
| **Art** | 23% | Graphic Art 8%, 3D Art 4%, Photography 10%, Others 1% | 摄影是最大单一子类（10%），反映真实用户需求 |
| **Film & Story** | 22% | Realistic 10%, Sci-fi 8%, Animation 3%, Others 1% | 写实和科幻相当，说明叙事性需求多 |
| **Illustration** | 18% | Content Illustration 14%, Copy Illustration 4% | 插画是内容创作的主流形式 |
| **Creative** | 10% | Imagination 7%, Others 3% | 创意/想象类场景 |

Photography（10%）、Realistic（10%）、Content Illustration（14%）是最大的三个子类。整体分布较均衡，没有单一类别超过 30%，说明数据覆盖的场景足够广泛。

### 3.2 RL Prompt 集合

用于 GRPO 阶段的 prompt 集合，与 SFT 数据严格隔离。

- 构造方法：与 SFT 相同的模拟流程（图像 → captioning → proxy prompt），但来源于**完全不重叠的图像集**
- 数量：约 **50,000 条** prompt
- 与 SFT 数据的关键区别：
  - 没有 ground-truth CoT 或 reprompt——改写器在 GRPO 中自己生成候选，AlignEvaluator 在线评分
  - 学习信号是**动态的、闭环的**（生成的图像 → 评分 → 反馈回策略）
- 主题分布：与 SFT 数据镜像一致，保持优化场景的连续性
- 不公开 RL prompt 的具体内容（仅公开了评测 benchmark），这符合 GRPO 阶段数据不包含参考改写答案的特性

### 3.3 数据构造的关键工程决策

**一对多 + 人工终选 > 一对一的 teacher forcing**：如果 teacher 直接输出单一 reprompt 用做 SFT 目标，数据质量上限就是 teacher 的一次性输出质量。一对多生成 + 人工筛选打破了这一上限，human-in-the-loop 能从多个候选中选出最优，产出"比 teacher 任何单次输出都好"的数据。

**自动过滤先行，人工筛选殿后**：经济性考量。自动过滤几乎是零成本的（API 调用），砍掉 38.8% 明显有问题的数据后，人工标注员只需要看剩下的 61.2%。这大幅降低了标注成本。

**中英文分开用不同 teacher**：Gemini-2.5-Pro 在英文创意写作上更强，DeepSeekV3 在中文场景下理解更准。这避免了单一模型的语言偏见。

**384 token 的 CoT 长度限制**：控制了推理深度与推理效率的平衡。太长会引入冗余信息甚至幻觉，太短则分析不充分。384 token 约等于中英文各 200-300 字的分析，刚好覆盖核心元素+属性+构图+风格的分析。

## 4. T2I-Keypoints-Align 评测基准

### 4.1 基准设计

- 名称：**T2I-Keypoints-Align**
- 总量：**6,687 条** prompt
  - 中文：3,687 条（55.1%）
  - 英文：3,000 条（44.9%）
- 每条 prompt 标注了**多个 keypoint 类别**，支持按 keypoint 维度的细粒度分析
- 公开地址：huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval

### 4.2 中英文 Prompt 特征差异（Fig. 7, Fig. 8）

| 特征 | 中文 | 英文 |
|------|------|------|
| 平均长度 | ~100 字符 | ~500 字符 |
| 长度分布 | 紧密集中在均值附近 | 分布较宽 |
| Keypoint 密度峰值 | 4 个 | 均匀分布 3-6 个 |
| Keypoint 共现模式 | Style ↔ World Knowledge 强相关 | 多样化的配对模式 |

**设计含义**：
- 中文子集测试：模型在**简洁 prompt 下高效捕捉语义**的能力——每个字都很关键，信息密度高
- 英文子集测试：模型解析**长文本、高组合复杂度**的能力——需要从冗长描述中准确提取和绑定多条约束
- 两者结合：覆盖真实用户从"一句话需求"到"详细技术需求"的全谱系

## 5. Experiments

### 5.1 实验设置与基础设施

- **基础 T2I 模型**：HunyuanImage 2.1（权重完全冻结，不参与任何训练或微调）
- **CoT Rewriter 初始化**：Hunyuan-7B-Instruct（Tencent 自研 7B 参数指令微调模型）
- **训练硬件**：**8 × NVIDIA H800 GPU**（80GB 显存）
- **推理硬件**：同上，改写器推理和 T2I 生成均在同一集群
- **评测基准**：T2I-Keypoints-Align（6,687 条，中英双语）
- **对比基线**：HunyuanImage 2.1 直接使用原始用户 prompt（w/o PromptEnhancer）

### 5.2 详细训练配置与参数解析

**SFT Stage**：

| 超参 | 值 | 设计理由 |
|------|-----|---------|
| Base Model | Hunyuan-7B-Instruct | 7B 规模在改写质量和推理效率之间取得平衡 |
| Learning Rate | 1.0 × 10⁻⁵ | 标准微调学习率，7B 模型在此 LR 下稳定收敛 |
| LR Scheduler | Cosine | 余弦退火平滑降低 LR，最后阶段的极低 LR 帮助精细收敛 |
| Warmup Ratio | 0.1 | 前 10% 步数线性 warmup，避免训练初期梯度不稳定 |
| Epochs | 2 | 48.5 万条数据过两遍，不过度拟合 |
| Effective Batch Size | 128 | 8 per-device × 2 grad accumulation × 8 GPUs |
| Precision | bfloat16 | 7B 模型 bf16 训练，显存和精度平衡 |

训练步数：485,119 × 2 / 128 ≈ 7,580 步

**GRPO Stage**：

| 超参 | 值 | 设计理由 |
|------|-----|---------|
| Base Model | SFT-tuned Rewriter | 从最佳 SFT checkpoint 初始化 |
| Learning Rate | 1.0 × 10⁻⁶ | 比 SFT 低一个数量级，保证策略缓慢稳定更新 |
| LR Scheduler | Constant | 不做衰减，GRPO 单 epoch 内全程恒定精细更新 |
| Epochs | 1 | 单轮 RL 优化，GRPO 的训练信号比 SFT 强得多 |
| Effective Batch Size | 64 | 全局 batch，每步处理 64 条 prompt |
| Precision | bfloat16 | 与 SFT 一致 |
| Rollout Samples (N) | 8 | 每 prompt 生成 8 个候选，组内比较 |
| KL Coefficient | 0.001 | 极小的 KL 惩罚，允许策略适度偏离但不至于 reward hacking |

GRPO 训练步数：50,000 × 1 / 64 ≈ 781 步。但每步计算量远超 SFT：
- 每步需要：64 × 8 = 512 次 CoT 改写器前向（生成候选 reprompt）
- 512 次 HunyuanImage 2.1 前向推理（T2I 生成）
- 512 次 AlignEvaluator 推理（24 维评分）
- 1 次策略梯度更新

GRPO 的瓶颈在**图像生成和 AlignEvaluator 评分**，而非策略更新本身。

### 5.3 CoT 推理过程的实际案例分析

以 Fig. 2 中的两个例子说明 CoT 改写器的工作机制：

**案例 1："A huge turtle is carrying a small city on its back"**

CoT 推理链：
```
核心关系分析：Opposing Relationship（对立关系）。
自然生命体(乌龟) vs 人造建筑(城市)的对比。
这是一种共生关系，主要对立轴是 Scale（尺度）。
实体 1：乌龟 — 必须是巨大的、古老的、岛屿级别的。
它的特征应暗示极端的年龄和规模，像一个行走的大陆。
实体 2：城市 — 必须是微型的，坐落在乌龟背上。
细节应精细但明显在规模上从属于乌龟。
```

改写后的 reprompt：
```
Epic fantasy art, a colossal, ancient sea turtle the size of
an island, swimming gracefully through a cosmic ocean filled
with shimmering nebulae. On its moss-covered, continent-like
shell, rests a miniature, intricate city, glowing with warm,
bioluminescent lights. The image should emphasize the extreme
contrast in scale between the gigantic, serene creature and
the tiny, bustling civilization it carries. Cinematic lighting,
highly detailed, trending on ArtStation.
```

改写方向分析：
- 从"龟背城"三个字 → 展开了主体属性（ancient, island-sized, moss-covered shell）、环境（cosmic ocean, nebulae）、氛围（bioluminescent, cinematic）、比例关系（extreme contrast in scale）
- CoT 发现了核心矛盾是"尺度对比"，因此改写的重点放在强调尺度差异上
- 补充了原来没有的美学描述（cinematic lighting, trending on ArtStation），增强了生成图的视觉质量

**案例 2："A cute penguin, wearing an astronaut suit and a red scarf, is floating and roaming in space. Oil painting style"**

AlignEvaluator 对这个 prompt 的 keypoint 匹配：
- **Artistic Style** ✓：图中正确呈现了厚涂油画风格
- **Counterfactual** ✓：企鹅穿宇航服在太空是一种反事实场景，图中正确呈现
- **Cross-Entity Binding** ✓：身体（白黄）、翅膀（黑）、宇航服（白色水平笔触）、头盔（透明黄色）、围巾（鲜红飘动）——各属性正确绑定到对应部位
- **Full-body Action** ✓：企鹅的"漂浮漫游"和围巾的"飘动"动态状态正确
- **Pronoun Resolution** ✓："Its wings"和"Its head"正确指代企鹅

这展示了 AlignEvaluator 的工作方式：不是打一个总分，而是在每个相关维度上做精确的匹配判断。

### 5.4 定量结果（Fig. 9）

**整体表现**：
- 24 个维度**平均提升 +5.1%**
- **20/24**（83.3%）个类别有提升
- **15 个类别**提升超过 5.0%

**按提升幅度排序的完整结果**：

| 类别 | 提升幅度 | 所属大类 | 解读 |
|------|---------|---------|------|
| Similarity Relation | **+17.3%** | Relations & Structure | 形状相似性是最需要推理的关系类型 |
| Counterfactual | **+17.2%** | Knowledge & Reasoning | 反事实场景需要想象力+逻辑一致性 |
| Counting | **+15.0%** | Visual Attributes | 计数是 T2I 众所周知的弱点 |
| Pronoun Resolution | **+13.9%** | Linguistic Comprehension | 代词消歧需要语言理解 |
| Expression | **+12.0%** | Visual Attributes | 面部表情的精细控制 |
| Cross-Entity Binding | **+11.3%** | Relations & Structure | 多实体多属性绑定的组合挑战 |
| 其他 9 个类别 | > 5.0% | 分布在各大类 | 广泛的中等幅度提升 |

**持平类别**：
- Contact Interaction：**0.0%** — baseline 已能较好处理物理接触交互（如拳击手打沙袋）
- Size：**0.0%** — 相对大小关系在简短 prompt 中已经足够明确

**轻微退化类别**：
- Text Layout：**-0.7%** — 文字位置指定的改写可能引入歧义
- Interaction w/o Contact：**-0.9%** — 非接触交互（如"爱因斯坦看着霍金"）的改写可能过度复杂化

**退化原因深入分析**：改写器偶尔会**过度指定（over-specification）**——把原本清晰的简单指令改得过于复杂，引入冗余细节反而干扰了 T2I 模型的理解。这揭示了 prompt 改写的一个固有问题：不是所有的 prompt 都需要改写，简单的、已经清晰的指令应该保持原样。如何训练改写器学会"判断什么时候不该改"，是一个重要的后续研究方向。

## 6. Related Work 与技术定位

### 6.1 Text-to-Image Generation

扩散模型技术路径：DDPM → DALL-E / Imagen / SD → SDXL / FLUX.1 → HunyuanDiT / Qwen-Image。无论如何演进，prompt alignment gap 始终存在。PromptEnhancer 不参与这条技术路径的竞争——它站在模型之上，充当"翻译层"。

### 6.2 Prompt Rewriting and Optimization

- **Recaptioning（DALL-E 3）**：改模型 → 不通用
- **On-the-fly Rewriting**：用通用 LLM → 缺乏 T2I 专业理解
- **RL-based Rewriting（RePrompt）**：粗粒度 reward → 优化方向模糊

PromptEnhancer 的差异化：专门训练的改写器 + T2I 专用的细粒度 reward 模型 = 专业工具而非通用工具。

### 6.3 CoT for Controllable Generation

两类做法：MLLM 先规划再指导扩散（RPG, LayerCraft），或 CoT 融入生成架构（Mint）。共性问题：**推理器和 T2I 模型紧耦合**。PromptEnhancer 将 CoT 纯用于 prompt 空间，实现架构解耦。

### 6.4 Fine-Grained Evaluation

从 FID/CLIP Score（整体指标）→ ImageReward/HPSv2（人类偏好）→ T2I-CompBench/EvalMuse（细粒度 benchmark）。PromptEnhancer 的贡献是**闭合评估-优化环路**：细粒度评估不仅是测量工具，更是直接驱动 GRPO 优化的训练信号。

## 7. Conclusion

- PromptEnhancer 是模型无关的通用 prompt 改写框架，不改 T2I 模型权重，即插即用
- CoT 改写器：先推理（分析主体、属性、关系、风格），再改写（输出结构化精炼 prompt）
- AlignEvaluator：24 维细粒度 keypoint 提供可诊断的反馈信号，比单一分数更有效
- 在 HunyuanImage 2.1 上全面验证，24 维度平均 +5.1%，复杂推理维度提升尤为显著

## 关键启示

1. **解耦改写与生成是实用的架构选择**：不碰 T2I 模型意味着可以独立迭代改写器和生成器。T2I 模型每月都在进化，改写器不需要跟着改。这是一个"站在巨人肩膀上"的策略——利用所有 T2I 模型的进步，同时独立优化 prompt 质量。

2. **24 维 keypoint 体系的系统性**：覆盖了 T2I 失败模式的全谱系——从最底层的语言结构（否定、代词）到中层视觉属性（计数、材质、表情）到高层推理（反事实、组合关系）。这种"检查清单"式的评估比端到端的单一分数更具可操作性和可解释性。Similarity Relation +17.3% 和 Counterfactual +17.2% 的巨大提升说明，精确诊断问题所在是有效优化的前提。

3. **GRPO 在 prompt 优化场景的自然适配**：每个用户 prompt 天然有多个合理的改写方向（详细版、简洁版、风格化版、叙事版...），GRPO 的组内比较机制允许同时探索多个方向并选出最优。KL 惩罚（0.001）很小但关键——它防止策略坍缩到某种固定模板（如总是加"trending on ArtStation"）。

4. **48.5 万条训练数据的规模效应**：对于 prompt 改写任务来说这是极大的规模。5 大类 20 子类的主题分布确保了改写器在各种场景下都有参考。更重要的是，human-in-the-loop 确保了每条数据的"改写质量"有真实的人类判断作为监督。

5. **一对多 + 人工筛选的数据构造模式值得借鉴**：不依赖 teacher 模型的一次性输出，而是让 teacher 生成多个候选，由人选出最优。这在理论上打破了数据质量的上限——产出的数据可以比 teacher 的任何单次输出都好。

6. **自动过滤砍掉 38.8% 数据的经济学启示**：即使是 Gemini-2.5-Pro 级别的模型，在 prompt 改写任务上的首次输出也有近 40% 存在质量问题。这说明了两个事实：(1) prompt 改写不是简单的"翻译"，需要专业能力；(2) 数据质量控制的投入产出比极高——用便宜的 API 调用过滤掉差数据，比用差数据训练出低质量模型再返工划算得多。

7. **过度指定（over-specification）是 prompt 改写的固有问题**：Text Layout -0.7% 和 Non-contact Interaction -0.9% 的退化揭示了改写器的一个根本困境——它学会了"添加细节"但没有学会"判断哪些细节不需要"。一个成熟的改写系统可能需要一个**改写必要性判断器**（rewrite necessity detector），先判断是否需要改写，再执行改写。

8. **跨模型泛化性是未验证的关键假设**：论文只在 HunyuanImage 2.1 上验证。虽然声称模型无关，但 AlignEvaluator 的训练数据来自 HunyuanImage 生成的图像，它学习到的"好坏判断标准"可能包含了 HunyuanImage 特有的偏差。如果换到 FLUX 或 SDXL 上，AlignEvaluator 的评分分布可能偏移，导致 GRPO 优化方向错误。这是该方法在实际部署前最需要验证的问题。

9. **AlignEvaluator 本身就是一个有价值的产品**：24 维细粒度 T2I 评估模型可以作为独立工具使用——T2I 模型开发者可以用来诊断模型在哪些维度上弱、产品团队可以用来做生成质量的自动化监控、用户可以用来理解"为什么我的图没生成好"。论文将其作为 reward 模型，但其作为评估工具的独立价值同样巨大。

10. **中文场景的特殊考量**：中文 prompt 的简短性和高信息密度（100 字符 vs 英文 500 字符）对改写器提出了不同的要求——中文改写更像"释义+展开"，英文改写更像"重组+精炼"。论文用不同 teacher 模型处理不同语言的策略说明了对这一差异的认知。
