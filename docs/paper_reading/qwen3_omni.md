---
tags:
  - VLM
---

# Qwen3-Omni Technical Report

- 论文：https://arxiv.org/abs/2509.17765
- 代码：https://github.com/QwenLM/Qwen3-Omni
- 团队：Qwen Team, Alibaba Group

## 概述

Qwen3-Omni 是首个在文本、图像、音频、视频四种模态上同时保持 SOTA 水平且不退化的单一多模态模型。它匹配同规模 Qwen 单模态模型的性能，在 36 个音频/视听基准中 32 个达到开源 SOTA、22 个达到总体 SOTA，超越 Gemini-2.5-Pro、Seed-ASR、GPT-4o-Transcribe。模型采用 Thinker-Talker MoE 架构，统一感知和生成，支持 119 种文本语言、19 种语音理解语言和 10 种语音生成语言，音频理解支持单条 40 分钟录音，冷启动端到端首包延迟 234ms。

**核心结论**：联合多模态训练可以实现跨模态零退化——多模态模型在每个模态上都不逊于对应的单模态模型。

## 架构

### Thinker-Talker 架构

延续 Qwen2.5-Omni 的 Thinker-Talker 设计，Qwen3-Omni 做了五项升级：

1. **Thinker 和 Talker 均采用 MoE**：提升并发和推理速度
2. **Audio Transformer (AuT) 替代 Whisper**：从头在 2000 万小时监督音频上训练，12.5 Hz token 率，block-wise window attention 实现实时 prefill 缓存
3. **多 Codebook 语音表示**：增强对多样声音、副语言线索和声学现象的建模
4. **多轨 Codec 建模**：Talker 通过 MTP 模块自回归预测多 codebook 层，Code2Wav 用轻量因果 ConvNet 替代 block-wise DiT
5. **输入输出音频码率降至 12.5 Hz**：输出 codec 支持单帧立即合成

| 模块 | 架构 | 参数 | 流式支持 |
|------|------|------|---------|
| Audio Encoder (AuT) | AuT | 650M | ✓ |
| Vision Encoder | SigLIP2-So400M | 540M | - |
| Thinker | MoE Transformer | 30B-A3B | ✓ |
| Talker | MoE Transformer | 3B-A0.3B | ✓ |
| MTP | Dense Transformer | 80M | ✓ |
| Code2Wav | ConvNet | 200M | ✓ |

### Thinker-Talker 解耦

相比 Qwen2.5-Omni，Talker 不再消费 Thinker 的高层文本表征，仅条件于音频和视觉多模态特征。理由：

- 对文本内容，离散 token 和 embedding 信息等价
- 多模态条件是语音生成协调（如保留语音翻译中的韵律/音色）所必需的
- 解耦允许外部模块（RAG/function calling/安全过滤）介入 Thinker 输出
- Thinker 和 Talker 可使用独立 system prompt

### AuT (Audio Transformer)

注意力 encoder-decoder 自回归模型：

- 训练数据：2000 万小时监督音频
- Conv2D 下采样 8 倍后接 attention layers，token 率 12.5 Hz
- 训练组成：80% 中英 ASR 伪标签 + 10% 其他语言 ASR + 10% 音频理解
- 动态 attention window（1-8 秒），平衡实时 prefill 缓存与离线任务
- 约 0.6B 参数

### 感知模块

- **文本**：Qwen tokenizer，151,643 常规 token
- **音频**：重采样 16kHz → 128 通道 mel 频谱图（25ms 窗口、10ms hop），每帧对应约 80ms 原始音频
- **视觉**：Qwen3-VL 的 SigLIP2-So400M（~543M），图像+视频联合训练

### TM-RoPE (Time-aligned Multimodal RoPE)

扩展 MRoPE，改进频率分配：temporal/height/width 交织分配 24/20/20 个 rotary angles（原始分配 16 给 temporal），改善长程外推。

各模态适配：
- 文本：三维共享 position ID，等价 1D RoPE
- 音频：共享 position ID + 绝对时间编码（每个 temporal ID 对应 80ms）
- 图像：恒定 temporal ID + 行列位置决定 height/width ID
- 视听流：音频 80ms/ID，视频帧动态调整至 80ms/ID 一致分辨率，模态间位置连续编号

与 Qwen2.5-Omni 固定 2 秒分块不同，Qwen3-Omni 直接用 temporal ID 对齐绝对时间，支持任意长度流式输入。

### 语音生成

Talker 条件于 Thinker 的历史文本 token、多模态表征和当前轮流式文本。分层预测：

1. Backbone 摄入当前帧的聚合 codebook 特征，线性头预测第 0 codebook
2. MTP 模块生成所有残差 codebook
3. 因果 ConvNet (Code2Wav) 重建波形

### 流式与并发

- **Chunked Prefilling**：Thinker 完成当前 chunk 后立即异步 prefill Talker，同时 Thinker 处理下一 chunk
- **MoE 架构优势**：相比 dense 模型，长序列 KV cache IO 消耗更低，TPS 更高
- **流式多 Codebook**：Talker 生成首 token 后 MTP 模块即预测当前帧剩余 token，Code2Wav 立即解码
- **轻量 MTP + ConvNet**：低 FLOP、支持 batch 推理、固定 KV cache 加速

首包延迟分解（单并发 Audio/Video）：

| 组件 | 延迟 |
|------|------|
| 预处理 | 72/160ms |
| Thinker TTFT | 88/160ms |
| Talker TTFT | 57/210ms |
| MTP 每 token | 14ms |
| Codec Decoder 每 code | 3ms |
| **总计** | **234/547ms** |

6 并发下总延迟增至 1172/2284ms，RTF 为 0.66（始终 < 1，保证流式不断）。

## 预训练

三阶段：

1. **Encoder 对齐 (S1)**：LLM 参数冻结，分别训练视觉和音频 encoder 的 adapter 再训练 encoder。**不再采用** encoder+adapter 联合训练同时冻结 LLM 的方案——因为这会让 encoder 补偿冻结 LLM 的局限，导致感知退化
2. **通用阶段 (S2)**：约 2T token，模态分布：文本 0.57T / 音频 0.77T / 图像 0.82T / 视频 0.05T / 视听 0.05T
3. **长上下文 (S3)**：序列长度 8192 → 32768，增加长音频和长视频比例

关键策略：**在文本预训练早期即混入单模态和跨模态数据**，这是实现多模态零退化的关键。

## 后训练

### Thinker（三阶段）

1. **轻量 SFT**：ChatML 格式指令微调
2. **Strong-to-Weak 蒸馏**：
   - Off-policy：教师模型生成响应
   - On-policy：学生自己生成 → 最小化与教师（Qwen3-32B 或 Qwen3-235B-A22B）logits 的 KL 散度
3. **GSPO**：跨文本/图像/视频/音频的通用 RL
   - 规则 reward：可验证任务（数学/编程/指令遵循）
   - 模型 reward：Qwen3 judge（通用）+ Qwen2.5-VL judge（视觉任务）

### Talker（四阶段）

1. 数亿语音数据建立多模态到语音的单调映射
2. 高质量数据 CPT：减少幻觉 + 长上下文训练
3. 多语言 DPO：偏好对优化生成稳定性
4. Speaker 微调：特定声音 + 自然度/表现力/可控性

### Captioner

微调 Qwen3-Omni-30B-A3B 得到 Captioner 模型，生成任意音频输入的详细低幻觉描述（填补缺乏通用音频描述模型的空白）。

## 实验结果

### Text → Text

| 基准 | GPT-4o-0327 | Qwen3-235B-A22B | Qwen3-30B-A3B-Instruct | **Qwen3-Omni-30B-A3B-Instruct** |
|------|-------------|-----------------|----------------------|-------------------------------|
| GPQA | 66.9 | 62.9 | 70.4 | **69.6** |
| AIME25 | 26.7 | 24.7 | 61.3 | **65.0** |
| ZebraLogic | 52.6 | 37.7 | 90.0 | **76.0** |
| WritingBench | 75.5 | 77.0 | 85.5 | **82.6** |

Qwen3-Omni-30B-A3B-Instruct 参数更小，但在多个基准上超越 Qwen3-235B-A22B Non-Thinking 和 GPT-4o-0327。

### Audio → Text

**ASR 性能**：

| 基准 | Seed-ASR | GPT-4o-Transcribe | Gemini-2.5-Pro | **Qwen3-Omni-Instruct** |
|------|----------|-------------------|---------------|------------------------|
| Librispeech clean/other | 1.58/2.84 | 1.39/3.75 | 2.89/3.56 | **1.22/2.48** |
| Fleurs-en | 3.40 | 3.32 | 2.94 | **2.72** |
| MIR-1K 歌词 | 6.45 | 11.87 | 9.85 | **5.90** |

在中英 ASR、多语 ASR、歌词 ASR 上均达 SOTA。

**音频推理**：MMAU 77.5 超 Gemini-2.5-Pro 的 77.4。

**音乐理解**：RUL-MuchoMusic 52.0、GTZAN 93.0、MTG 系列和 MagnaTagATune 全面超越 Gemini-2.5-Pro 和专家模型。

### Vision → Text

| 基准 | GPT-4o | Gemini-2.0-Flash | Qwen2.5-VL-72B | **Qwen3-Omni-Instruct** |
|------|--------|-----------------|---------------|------------------------|
| MMMU-Pro | 51.9 | 56.1 | 51.1 | **57.0** |
| MathVista | 63.8 | 71.4 | 74.8 | **75.9** |
| MATH-Vision | 30.4 | 48.6 | 38.1 | **56.3** |
| MLVU | 64.6 | 71.0 | 74.6 | **75.2** |

视觉任务与 Qwen2.5-VL-72B 可比，数学 STEM 显著更优。

### AudioVisual → Text

WorldSense 54.0（开源 SOTA），超前一代和 Gemini-2.5-Flash。Thinking 版在 DailyOmni 75.8、VideoHolmes 57.3 上领先。

### 语音生成

- SEED 零样本 TTS：test-en WER 1.39 SOTA
- 多语言：中/英/法等多语言显著优于 MiniMax 和 ElevenLabs
- 跨语言克隆：any-to-en 和 any-to-ko 优于 CosyVoice3

## 多模态零退化验证

控制实验：30B-A3B 规模下训练三个模型（text-only / vision-only / Omni），严格控制训练数据、学习率、batch size、等效 epoch。

| 基准 | Qwen3-30B-A3B-Base | Qwen3-VL-30B-A3B-Base | **Qwen3-Omni-30B-A3B-Base** |
|------|-------------------|-----------------------|---------------------------|
| MMLU | 81.24 | - | **81.69** |
| MMLU-Redux | 80.17 | - | **80.60** |
| EvalPlus | 69.70 | - | **73.96** |
| MMMU | - | 57.22 | **59.33** |
| MMStar | - | 67.2 | **69.6** |
| DocVQA | - | 95.19 | **95.27** |
| InfoVQA | - | 81.17 | **83.31** |

关键发现：
1. 预训练早期混入多模态数据可实现语言能力零退化
2. 文本模态的加入显著提升视觉和音频性能
3. 音频数据的加入一致提升 MMMU 和 OCR 相关任务
4. 视觉/音频信号对语言能力没有可观测的提升

## 关键启示

- **多模态零退化是可实现的**：关键在于预训练早期即混入单模态和跨模态数据，而非后期拼接
- **Thinker-Talker 解耦的价值**：文本表征和 embedding 信息等价，解耦后允许外部模块干预，独立控制风格
- **AuT 从头训练优于 Whisper**：2000 万小时数据 + ASR/音频理解混合任务 + 动态 window attention 产出更通用的音频表征
- **多 Codebook + ConvNet 替代 DiT**：在保证音质的同时大幅降低延迟（234ms 首包）
- **不要联合训练 encoder+adapter 同时冻结 LLM**：这会让 encoder 补偿 LLM 局限而非学到好表征
- **Thinking 模式对感知任务无益**：ASR/音乐理解中 Thinking 模型反而不如 Instruct，可能引入更多幻觉
