---
tags:
  - VLM
---

# Qwen2.5-VL Technical Report

- 论文：https://arxiv.org/abs/2502.13923
- 代码：https://github.com/QwenLM/Qwen2.5-VL
- 团队：Qwen Team, Alibaba Group

## 概述

Qwen2.5-VL 是 Qwen 视觉语言系列的旗舰模型，在视觉识别、目标定位、文档解析和长视频理解上取得显著进步。核心改进包括：(1) ViT 引入 Window Attention 降低计算开销；(2) 动态 FPS 采样将动态分辨率扩展到时间维度；(3) MRoPE 对齐绝对时间实现秒级事件定位；(4) 预训练数据从 1.2T token 扩展到 4.1T token。旗舰 72B 模型在文档理解上达到 GPT-4o 和 Claude 3.5 Sonnet 水平，7B/3B 小模型也优于同规模竞品。

## 模型架构

整体由三个模块组成：

- **Large Language Model**：基于 Qwen2.5 LLM 初始化，将 1D RoPE 修改为 Multimodal RoPE (MRoPE)
- **Vision Encoder**：重新设计的 ViT，引入 2D-RoPE 和 Window Attention
- **MLP-based Vision-Language Merger**：将相邻 4 个 patch 特征分组拼接后通过两层 MLP 压缩，对齐文本嵌入维度

三个规模的配置：

| 配置 | 3B | 7B | 72B |
|------|------|------|------|
| ViT Hidden Size | 1280 | 1280 | 1280 |
| ViT Layers | 32 | 32 | 32 |
| ViT Window Size | 112 | 112 | 112 |
| Full Attn Block | {7,15,23,31} | {7,15,23,31} | {7,15,23,31} |
| LLM Hidden Size | 2048 | 3584 | 8192 |
| LLM Layers | 36 | 28 | 80 |
| LLM KV Heads | 2 | 4 | 8 |
| 训练 Token | 4.1T | 4.1T | 4.1T |

### 高效 Vision Encoder

传统 ViT 处理不同大小图像时计算复杂度为平方级。Qwen2.5-VL 的解决方案：

- **大多数层使用 Window Attention**：最大窗口 112×112（8×8 patches），计算量与 patch 数线性相关
- **仅 4 层使用 Full Self-Attention**：第 7/15/23/31 层
- **2D RoPE 位置编码**：捕捉 2D 空间关系
- **3D Patch 分区处理视频**：相邻两帧分组，显著减少送入 LLM 的 token 数
- **架构对齐 LLM 设计**：采用 RMSNorm + SwiGLU，增强视觉和语言模块兼容性

ViT 从头训练，依次经过 CLIP 预训练 → 视觉语言对齐 → 端到端微调。训练时以原始分辨率动态采样，增强泛化能力。

### 原生动态分辨率与帧率

**空间域**：不同大小的图像动态转换为不同长度的 token 序列，直接使用实际像素尺寸表示 bounding box 和 point，模型内在学习尺度信息。

**时间域**：引入动态 FPS 训练和绝对时间编码。将 MRoPE 的 temporal ID 直接对齐到绝对时间戳，模型通过 temporal ID 之间的间隔感知时间节奏，无需额外计算开销。

### MRoPE 对齐绝对时间

MRoPE 将位置嵌入分解为三个分量：temporal、height、width。

- 文本输入：三个分量使用相同 position ID，等价于 1D RoPE
- 图像输入：temporal ID 恒定，height/width 按空间位置赋值
- 视频输入：temporal ID 按帧递增，height/width 同图像

Qwen2-VL 中 temporal ID 绑定到帧数，不反映实际时间。Qwen2.5-VL 改为对齐绝对时间，使不同 FPS 采样率的视频具有一致的时间对齐。

## 预训练

### 数据

共约 4T token，涵盖多种多模态数据：

- **交错图文数据**：四阶段评分系统（文本质量、图文相关性、信息互补性、信息密度平衡）
- **绝对坐标定位数据**：使用实际像素坐标，支持 XML/JSON 等格式，10,000+ 物体类别
- **文档全解析数据**：QwenVL HTML 格式统一表示表格、图表、公式、音乐谱、化学式等元素
- **OCR 数据**：合成引擎生成 + 大规模多语言 OCR（法/德/意/西/葡/阿/俄/日/韩/越等），100 万图表样本，600 万表格样本
- **视频数据**：动态 FPS 采样，超半小时长视频合成多帧字幕，时间戳支持秒格式和时分秒帧格式
- **Agent 数据**：手机/网页/桌面截图感知 + 统一 function call 操作空间 + 推理过程标注

### 训练流程

| 阶段 | 训练模块 | 数据 | Token | 序列长度 |
|------|---------|------|-------|---------|
| 视觉预训练 | ViT | 图文字幕、知识、OCR | 1.5T | 8192 |
| 多模态预训练 | ViT & LLM | + 纯文本、交错数据、VQA、视频、定位、Agent | 2T | 8192 |
| 长上下文预训练 | ViT & LLM | + 长视频、长 Agent、长文档 | 0.6T | 32768 |

- 阶段一：仅训练 ViT，提升与 LLM 的对齐
- 阶段二：解冻全部参数，引入复杂推理数据
- 阶段三：扩展序列长度，增强长程依赖处理

训练时按 LLM 输入序列长度动态打包数据，确保跨 GPU 计算负载均衡。

## 后训练

采用 SFT + DPO 双阶段范式，ViT 参数冻结。

### SFT

约 200 万条数据，纯文本与多模态各 50%，以中英文为主。包含：

- 通用 VQA、图文描述、数学、编程、安全
- 文档与 OCR、定位、视频分析、Agent 交互

使用 ChatML 格式结构化指令数据。

### 数据过滤

两阶段：

1. **领域分类**：Qwen2-VL-Instag 分类模型，8 大域 30 子类
2. **领域定制过滤**：
   - 规则过滤：去重复/截断/格式错误/无关/有害内容
   - 模型过滤：基于 Qwen2.5-VL 奖励模型，多维度评估正确性/完整性/清晰度/视觉信息利用

### 拒绝采样增强推理

对数学/代码/领域 VQA 等需要多步推理的任务：

- 用中间版本 Qwen2.5-VL 生成响应，仅保留与 ground truth 匹配的样本
- 过滤语码切换、过长、重复模式
- 开发规则和模型驱动的过滤策略验证 CoT 中间步骤的视觉-文本一致性

### DPO

仅在图文和纯文本数据上进行，利用偏好数据对齐人类偏好，每条样本仅处理一次。

## 实验结果

### 与 SOTA 对比（72B）

| 基准 | Claude 3.5 Sonnet | GPT-4o | InternVL2.5-78B | Qwen2-VL-72B | **Qwen2.5-VL-72B** |
|------|-------------------|--------|-----------------|-------------|-------------------|
| MMMU | 68.3 | 69.1 | 70.1 | 64.5 | **70.2** |
| MathVista | 67.7 | 63.8 | 72.3 | 70.5 | **74.8** |
| MATH-Vision | - | 30.4 | 32.2 | 25.9 | **38.1** |
| MMBench-EN | 82.6 | 83.4 | 88.3 | 86.9 | **88.6** |
| MuirBench | - | 68.0 | 63.5 | - | **70.7** |
| MMVet | 70.1 | 69.1 | 72.3 | 74.0 | **76.2** |

### 文档理解与 OCR

- DocVQA 96.4、InfoVQA 87.3、ChartQA 89.5 均大幅领先
- OCRBench 885、OCRBench_v2 英文 61.5/中文 63.7，超 Gemini 1.5-Pro 9.6%/20.6%
- CC-OCR 79.8 和 OmniDocBench 均 SOTA

### 视觉定位

- RefCOCO 系列：92.7+ 准确率，与 InternVL2.5-78B 接近
- ODinW 开放词汇检测 43.1 mAP
- CountBench 计数 93.6（"检测后计数"策略）

### 视频理解

- LVBench 47.3、MLVU 74.6：长视频理解大幅超 GPT-4o
- Charades-STA 50.9 mIoU：事件时间定位超 GPT-4o 的 35.7
- 所有评测最大帧数 768，视频 token 上限 24,576

### Agent

- ScreenSpot 87.1、ScreenSpot Pro 43.6（Qwen2-VL-72B 仅 1.6）
- AndroidWorld 35%、MobileMiniWob++ 68%：无需辅助标记即超越基线
- 可在真实动态环境中作为 Agent 运行

### 纯文本任务

保持 Qwen2.5 LLM 的语言能力：

- MATH 83.0、HumanEval 87.8、MultiPL-E 79.5
- 在多个纯文本基准上与 Qwen2.5-72B 接近甚至更优

## 关键启示

- **Window Attention 是 ViT 扩展的关键**：仅 4 层 full attention 即可保持性能，大幅降低计算开销
- **绝对时间对齐的位置编码**：简单有效地赋予模型时间感知能力，不需要额外模块
- **原生动态分辨率**：使用实际像素坐标而非归一化坐标，让模型内在学习尺度信息
- **数据质量驱动性能**：从 1.2T 到 4.1T 的扩展伴随精细的评分-过滤流水线，拒绝采样进一步增强推理
- **Agent 能力依赖定位精度**：ScreenSpot Pro 从 1.6 到 43.6 的跃升说明细粒度定位是 Agent 的基础
