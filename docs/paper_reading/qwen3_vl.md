---
tags:
  - VLM
---

# Qwen3-VL Technical Report

- 论文：https://arxiv.org/abs/2511.21631
- 代码：https://github.com/QwenLM/Qwen3-VL
- 团队：Qwen Team, Alibaba Group

## 概述

Qwen3-VL 是 Qwen 视觉语言系列目前最强的模型，提供 dense（2B/4B/8B/32B）和 MoE（30B-A3B / 235B-A22B）六个规模。原生支持 256K token 的交错上下文，融合文本、图像、视频。三大核心能力：(1) 纯文本理解超越同等规模 text-only backbone；(2) 256K 原生长上下文理解；(3) 多模态推理领先（MMMU、MathVista、MathVision 等）。架构上引入 Interleaved MRoPE、DeepStack 多层 ViT 特征注入、文本时间戳对齐三项改进。同时提供 non-thinking 和 thinking 两种变体。

## 模型架构

沿用 Qwen2.5-VL 的三模块设计：Vision Encoder + MLP Merger + LLM。

- **LLM**：基于 Qwen3 backbone，旗舰 235B-A22B 模型总参 235B、每 token 激活 22B
- **Vision Encoder**：SigLIP-2 架构，大模型用 SO-400M，小模型（2B/4B）用 Large-300M，支持动态输入分辨率，2D-RoPE + 插值绝对位置嵌入
- **MLP Merger**：两层 MLP 将 2×2 视觉特征压缩为单个 visual token，另有专用 merger 支持 DeepStack

### Interleaved MRoPE

Qwen2-VL 的 MRoPE 将嵌入维度划分为 temporal/height/width 三个连续子空间，导致频谱不平衡，损害长视频理解。Qwen3-VL 改为**交错分配**：将 t/h/w 分量均匀交织到嵌入维度中，使每个空间-时间轴在低频和高频波段均有均匀表示。这消除了原始的频谱偏差，显著提升长程位置建模。

### DeepStack

从 ViT 的三个不同层级提取视觉特征，经专用 merger 投影后直接加到 LLM 前三层对应的 hidden state 上。保留了从低层到高层的丰富视觉信息，不增加上下文长度。

### 视频时间戳

Qwen2.5-VL 用 MRoPE 的 temporal ID 绑定绝对时间，但存在两个问题：(1) 长视频产生过大且稀疏的 temporal ID，退化长时理解；(2) 需要大量不同帧率均匀分布的训练数据。

Qwen3-VL 改为**文本 token 编码时间**：每个视频 temporal patch 前缀格式化时间字符串（如 `<3.0 seconds>`），同时训练秒格式和时分秒格式。略增上下文长度，但时间感知更精准有效。

## 预训练

### 训练流程

| 阶段 | 目标 | 训练模块 | Token 预算 | 序列长度 |
|------|------|---------|-----------|---------|
| S0 | 视觉语言对齐 | Merger | 67B | 8,192 |
| S1 | 多模态预训练 | All | ~1T | 8,192 |
| S2 | 长上下文预训练 | All | ~1T | 32,768 |
| S3 | 超长上下文适应 | All | 100B | 262,144 |

- S0 仅训练 MLP merger，ViT 和 LLM 冻结
- S1 解冻全部参数，混合 VL 数据和纯文本数据
- S2 序列长度 4 倍扩展，增加视频和 Agent 数据比例
- S3 推至 256K，专注长视频和长文档任务

### 训练数据

**图文数据**：用微调的 Qwen2.5-VL-32B 重新生成更精细的描述，语义去重 + 视觉嵌入聚类增强稀疏概念。

**交错图文**：从中英文网页和书籍收集，领域分类过滤广告/低质内容。书籍级数据用 Qwen2.5-VL-7B 精确解析图文对齐，合并连续页面至 256K。

**知识**：十余种语义类别的实体数据，重要性采样（高知名度多采样、低知名度保覆盖），LLM 生成增强描述。

**OCR & 文档解析**：
- OCR：3000 万内部采集样本 + 新增 29 种语言约 3000 万合成样本
- 文档解析：300 万 Common Crawl PDF（10 类各 30 万）+ 400 万内部文档
- 两种格式：QwenVL-HTML（元素级 bbox）和 QwenVL-Markdown（表格用 LaTeX）
- 长文档 VQA：跨页面推理

**定位与计数**：
- Box 定位：COCO/Objects365/OpenImages/RefCOCO + 自动合成流水线（候选提取 → 定位标注 → 质量过滤）
- Point 定位：公开数据 + 合成精细指向标注
- 计数：直接计数 + box 计数 + point 计数
- 坐标系归一化到 [0, 1000]（Qwen2.5-VL 用绝对像素坐标）

**空间理解与 3D**：
- 空间关系推理（"杯子在笔记本左边"）、物体可操作性标注、动作规划查询
- 3D Grounding：单视角图像 + 自然语言 + 9-DoF 3D bounding box，统一虚拟相机坐标系

**代码**：
- 纯文本代码：复用 Qwen3/Qwen3-Coder 语料
- 多模态代码：UI 截图转 HTML/CSS、图像转 SVG、视觉编程挑战、流程图/公式转代码

**视频**：
- 短到长字幕合成 + 时间戳交织 + 事件级摘要
- 时空视频 Grounding（目标 + 动作 + 人物）
- 来源平衡（教学/电影/第一视角等）+ 长度自适应采样

**STEM**：
- 视觉感知数据：100 万点定位样本 + 200 万感知 VQA + 600 万图表描述
- 多模态推理：6000 万+ K-12 和大学级练习 + 1200 万长 CoT 推理样本
- 语言推理：复用 Qwen3 数据

**Agent**：
- GUI 感知：桌面/手机/网页跨平台数据，元素描述 + 密集定位
- 多步轨迹：自进化轨迹生成 + 人工审计 + CoT 推理标注
- Function Calling：多模态 function calling 轨迹合成
- 搜索：图像搜索 + 文本搜索工具调用轨迹

### 损失函数

从 per-sample loss 改为 **square-root-normalized per-token loss**，更好平衡文本和多模态数据的贡献。

## 后训练

三阶段流程：SFT → Strong-to-Weak Distillation → Reinforcement Learning。

### SFT

约 120 万条样本，1/3 纯文本 + 2/3 多模态。两轮训练：

1. 32K 上下文长度一轮
2. 256K 上下文长度一轮（交替长上下文和 32K 数据）

数据过滤：
- Query 过滤：丢弃不可验证/模糊查询，去除无内容网页查询
- Response 过滤：规则（重复/截断/格式错误/有害）+ 模型（Qwen2.5-VL 奖励模型多维评估）

### Long-CoT Cold Start

为 thinking 模型构建长 CoT 冷启动数据，VL 与纯文本约 1:1。

难度控制：
- 保留基线模型 pass rate 低的实例
- 多模态必要性过滤：Qwen3-30B-nothink 不看图能答对的丢弃
- 响应质量控制：去重复/语码切换/猜测无推理

### 知识蒸馏

采用 Qwen3 的 Strong-to-Weak 蒸馏：

1. Off-policy：教师模型生成响应，学生模型学习
2. On-policy：学生模型自己生成，最小化与教师 logits 的 KL 散度

**关键设计**：仅在纯文本数据上蒸馏 LLM backbone，但推理能力提升同时迁移到多模态任务。

### 强化学习

分两阶段：

**Reasoning RL**：
- 涵盖数学/编程/逻辑/视觉定位/视觉拼图等可验证任务
- 约 30K RL query，每个采样 16 响应，过滤 pass rate > 90% 的简单题
- 用 SAPO 算法
- 混合任务 batch，比例通过大量实验确定

**General RL**：
- 覆盖 VQA/字幕/OCR/文档解析/定位/时钟识别等
- 双目标：指令遵循 + 偏好对齐
- 专门构造反直觉任务纠正 SFT 阶段的错误先验
- 构造专项数据集针对语码切换/重复/格式错误进行高频惩罚
- 混合 reward：规则（可验证任务）+ 模型（Qwen2.5-VL-72B/Qwen3 judge）

### Thinking with Images

两阶段训练视觉 Agent（think → act → analyze feedback → answer）：

1. 在 Qwen2.5-VL-32B 上训练：10K 冷启动 Agent 数据 + 多轮工具集成 RL
2. 蒸馏到 Qwen3-VL：120K 多样 Agent 交互数据 + 同样的 SFT + RL

RL 三种 reward：
- 答案准确性（Qwen3-32B 评估）
- 多轮推理质量（Qwen2.5-VL-72B 评估）
- 工具调用合理性（与专家估计的调用次数比较）

## 实验结果

### 旗舰模型（235B-A22B）

**多模态推理**：

| 基准 | Gemini 2.5 Pro | GPT-5 | Claude Opus 4.1 | **Qwen3-VL-235B** (Thinking) |
|------|---------------|-------|-----------------|---------------------------|
| MMMU | 84.2 | 80.9 | 74.4 | **80.6** |
| MathVista | 81.3 | 77.7 | 50.9 | **85.8** |
| MathVision | 70.9 | 66.0 | 45.8 | **74.6** |
| MathVerse | 84.1 | 65.9 | 43.0 | **85.0** |
| DynaMath | 85.4 | 78.5 | 74.0 | **82.8** |
| VisuLogic | 28.5 | 26.9 | 27.2 | **34.4** |

Qwen3-VL-235B-A22B-Thinking 在 MathVista/MathVision/MathVerse 等多个推理基准上达到 SOTA。

**文档理解**：

- DocVQA 96.5/97.1（thinking/instruct），OCRBench 875/920
- OCRBench_v2 英文 66.8/67.1，中文 63.5/61.8
- CC-OCR 81.5/82.2，OmniDocBench 英文 0.155/0.143

**视频理解**：

- Video-MME 79.0/79.2，MLVU 83.8/84.3
- LVBench 63.6/67.7，VideoMMMU 80.0/74.7

**Agent**：

- ScreenSpot Pro 61.8/62.0
- OSWorld Grounding 68.3/66.7，AndroidWorld 62.0/63.7
- OSWorld 38.1/31.6

**定位与 3D**：

- RefCOCO-avg 92.1/91.9
- CountBench 93.7/93.0
- ARKitScenes 53.7/56.9

**多模态编码**：

- Design2Code 93.4/92.0
- ChartMimic 78.4/80.5

### 中等模型（32B / 30B-A3B）

32B-Thinking 在 MMBench/RealWorldQA 上取得最高分（89.5/79.4），已超越上代 72B。30B-A3B MoE 也有竞争力。

### 小模型（2B/4B/8B）

8B 在所有五个 VQA 基准最优。4B 在 DynaMath 和 VisuLogic 上最优。2B 也展现出强推理能力。

## 关键启示

- **Interleaved MRoPE 解决频谱偏差**：简单的交错分配策略显著改善长视频理解，说明位置编码的频谱平衡至关重要
- **DeepStack 多层特征注入**：不增加上下文长度的情况下注入多层次视觉信息，比仅使用最后一层更有效
- **文本时间戳优于位置编码时间对齐**：虽然增加少量 token，但避免了长视频中过大 temporal ID 的问题
- **Square-root loss 平衡多模态训练**：解决文本和多模态数据贡献不均的问题
- **知识蒸馏用纯文本即可提升多模态**：在纯文本上蒸馏 LLM backbone 的推理能力可以迁移到多模态任务
- **Thinking with Images 需要三重 reward**：仅答案 reward 会退化为单次工具调用，工具调用频率 reward 是关键
- **坐标归一化到 [0, 1000]**：相比 Qwen2.5-VL 的绝对像素坐标，归一化方案对分辨率和宽高比变化更鲁棒
