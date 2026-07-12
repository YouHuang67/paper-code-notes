---
tags:
  - Diffusion Model
  - Flow Matching
  - VLM
  - LLM Inference
  - KV Cache
  - Post Training
  - Unified Understanding
---

**论文**: [Lance: Unified Multimodal Modeling by Multi-Task Synergy](https://arxiv.org/abs/2605.18678)
**代码**: [github.com/bytedance/Lance](https://github.com/bytedance/Lance)
**团队**: Intelligent Creation Lab, ByteDance
**arXiv 日期**: 2026-05

## 概述

Lance 是字节跳动智创实验室提出的 **3B 激活参数**原生态统一多模态模型，在同一个小参数框架内支持图像/视频的理解（X2T）、生成（X2I/X2V）和编辑。核心主张是：多任务协同训练（multi-task synergy）不仅是能力的简单叠加，而是不同模态任务之间**相互增强**的机制。

设计基于两个原则：

1. **统一上下文建模（Unified Context Modeling）**：所有模态（文本、ViT 语义 token、VAE 隐空间 token）组织在同一个交错的（interleaved）多模态序列中，联合建模
2. **解耦能力通路（Decoupled Capability Pathways）**：理解和生成走不同的专家通路（dual-expert），避免异构优化目标在共享参数上竞争

Lance 最关键的设计选择是将**自回归语言建模（用于理解）**和**Flow Matching（用于生成）**放在同一个 backbone 中，通过 MoE 风格的 dual-stream 架构来分配专门化的专家容量。在仅 128 张 A100 GPU 的预算下从头训练，在图像生成、视频生成、图像编辑上全面超过现有开源统一模型（7B 级别），同时保持强劲的多模态理解能力。

关键贡献：
- 3B 参数的原生态统一模型，覆盖 X2T/X2I/X2V 全任务谱
- Dual-stream MoE 架构：共享上下文 + 解耦专家通路
- MaPE（Modality-aware Rotary Positional Encoding）：解决异构视觉 token 的位置编码干扰
- 分阶段多任务训练范式（PT → CT → SFT → RL）
- 在 128 GPU 预算内实现 SOTA 统一模型性能


## 1. 模型架构

### 1.1 整体框架

给定交错的文本、图像、视频输入，Lance 首先将每种模态转换为任务适配的 token 表示，然后组织成统一的多模态序列：

$$S = \cdots \oplus B_{text}(T) \oplus B_{vis}(V_{vit}) \oplus B_{vis}(V^{clean}_{vae}) \oplus B_{vis}(V^{noisy}_{vae}) \oplus B_{text}(T') \oplus \cdots$$

其中每个块有边界标记：
$$B_{text}(T) = [BOT, T, EOT], \quad B_{vis}(V) = [BOV, V, EOV]$$

**四种 token 类型**：
1. **文本 token**：通过 Qwen2.5-VL 的语言嵌入层编码
2. **ViT 语义 token**：Qwen2.5-VL ViT encoder，14× 空间 + 2× 时间 patching，再经 2×2 空间 merge，产出语言对齐的紧凑语义特征
3. **干净 VAE latent token**：Wan2.2 3D causal VAE encoder，16× 空间下采样 + 4× 时间下采样，作为生成条件
4. **噪声 VAE latent token**：同样是 VAE 隐空间，添加噪声后的生成目标

**源码对应** [lance.py L148-L312](https://github.com/bytedance/Lance/blob/main/modeling/lance/lance.py#L148-L312)：
- ViT token → `connector`（2层MLP）→ LLM hidden space
- VAE latent → `vae2llm`（nn.Linear）→ LLM hidden space + timestep embed + 3D position embed
- LLM hidden → `llm2vae`（nn.Linear）→ VAE latent space（预测 velocity）

### 1.2 编码器组件

| 组件 | 来源 | 参数量 | 作用 |
|------|------|--------|------|
| ViT Encoder | Qwen2.5-VL 3B | 冻结 | 语义视觉特征提取（理解侧） |
| VAE Encoder | Wan2.2 3D causal VAE | 冻结 | 连续隐空间编码（生成侧） |
| ViT → LLM Connector | 2层 MLP（gelu_pytorch_tanh） | 可训练 | 语义 token 维度对齐 |
| VAE → LLM Connector | 单层 Linear | 可训练 | 隐空间 patch → hidden_size |
| LLM → VAE Connector | 单层 Linear | 可训练 | hidden_size → 隐空间 patch |
| Timestep Embedder | Sinusoidal + MLP（256→hidden→hidden） | 可训练 | Flow Matching 时间步嵌入 |
| 3D Position Embedding | Sin-cos 3D positional embedding | 冻结 | VAE latent 的空间时间位置 |

### 1.3 Dual-Stream MoE 架构（核心创新）

Lance 的 LLM backbone 基于 Qwen2.5-VL 3B，但进行了关键的 MoE 改造。代码中有三种 DecoderLayer 类型：

**`Qwen2DecoderLayer`** — 标准层：共享 Q/K/V/O + MLP
- 用于非 MoE 模式或纯文本处理

**`Qwen2MoEDecoderLayer`** — MLP-only MoE：共享 Attention + 分离 MLP
- Attention 的 Q/K/V/O 共享，但 MLP 分 `self.mlp`（理解）和 `self.mlp_moe_gen`（生成）
- 理解 token → `mlp`，生成 token → `mlp_moe_gen`

**`Qwen2MoTDecoderLayer`** — Full MoE（MoAT）：分离 Attention + 分离 MLP + 分离 Norm
- **最彻底的解耦版本**
- Attention 侧：`q_proj/k_proj/v_proj/o_proj`（理解） + `q_proj_moe_gen/k_proj_moe_gen/v_proj_moe_gen/o_proj_moe_gen`（生成）
- QK-Norm 也分两组：`q_norm/k_norm` + `q_norm_moe_gen/k_norm_moe_gen`
- LayerNorm 也分两组：`input_layernorm/post_attention_layernorm` + `input_layernorm_moe_gen/post_attention_layernorm_moe_gen`
- MLP 分两组：`mlp` + `mlp_moe_gen`

**源码位置**：[qwen2_navit.py L229-L707](https://github.com/bytedance/Lance/blob/main/modeling/lance/qwen2_navit.py#L229-L707)

关键实现细节：
- 训练时：将所有 token 分为 `packed_und_token_indexes`（理解侧）和 `packed_gen_token_indexes`（生成侧），分别通过各自的 Q/K/V/O 投影后合并做统一 attention
- 推理时：通过 `mode="und"` 和 `mode="gen"` 切换使用哪组参数
- 理解侧可冻结（`freeze_und=True`）来保护理解能力不被生成训练破坏

配置通过 `config.llm_config.layer_module` 控制：
- `"Qwen2DecoderLayer"` — 全共享
- `"Qwen2MoEDecoderLayer"` — MLP MoE
- `"Qwen2MoTDecoderLayer"` — Full MoE（Attention+MLP+Norm 全部解耦）

### 1.4 统一 Attention 机制

Lance 使用 **广义 3D 因果注意（Generalized 3D Causal Attention）**：

- 序列划分为模态特定段（text / ViT visual / VAE visual）
- 每个段对前面的「干净」段（text + ViT + clean VAE）做全注意力
- 段内：文本用因果注意力，视觉用双向注意力（捕捉空间和时空结构）
- 噪声 VAE token 只看到干净内容，不看到其他噪声 token

**Attention mask 构建**：由 `create_sparse_mask` 和 PyTorch `flex_attention` 的 `create_block_mask` 实现。不同 attention mode 映射为：
- `"causal"` — 标准因果 mask
- `"full"` — 双向注意力（视觉 token 段内）
- `"noise"` / `"full_noise"` — 噪声 token，见干净内容
- `"full_noise_target"` — 噪声目标 token

推理时训练/验证模式使用 `flex_attention`（`torch.compile`），推理模式使用 `flash_attn_varlen_func`。


## 2. MaPE：Modality-Aware Rotary Positional Encoding

### 2.1 动机

统一多模态训练中，同一个序列包含三种不同来源和功能的视觉 token：
- **ViT 语义 token**：为理解提供语言对齐的视觉语义
- **干净 VAE latent**：作为视觉条件
- **噪声 VAE latent**：作为生成优化目标

标准 3D-RoPE（Qwen2.5-VL 的）按统一的时空布局分配位置，不区分 token 组来源，可能导致**位置歧义**和**跨任务对齐弱化**。

### 2.2 方法

MaPE 在时间维度上对不同 token 组施加不同的偏移步长：

$$p^{(m_i)}_{t,h,w} = [\hat{t}^{(m_i)}_{t,h,w} + i \cdot \Delta t, \ \hat{h}^{(m_i)}_{t,h,w}, \ \hat{w}^{(m_i)}_{t,h,w}]$$

其中 $m_i \in \{V^{noisy}_{vae}, V_{vit}, V^{clean}_{vae}\}$，$i \in \{0, 1, 2\}$，$\Delta t = 1000$。

**设计考量**：
- 只在**时间维度**上加偏移，空间坐标保持不变 → 保持图像/视频内在空间布局
- 组内所有 token 共享同一个时间偏移 → 视频的时序结构和相对距离完全保留
- 不同组间在全局位置空间中有明确的间隔 → 模型能更好地区分语义特征、条件和目标

### 2.3 消融结果

| Setting | GenEval ↑ | GEdit ↑ | VBench ↑ | MVBench ↑ |
|---------|-----------|---------|----------|-----------|
| w/ MaPE | 80.94 | 6.86 | 81.81 | 59.16 |
| w/o MaPE | 80.56 | 6.30 | 80.95 | 59.02 |

图像编辑的提升最明显（+0.56），因为编辑任务需要模型同时推理视觉条件和生成目标，位置区分对跨任务上下文对齐帮助最大。

## 3. 训练策略

### 3.1 训练阶段总览

| 阶段 | 步数 | LR | 调度器 | 序列长度 | Token量 | Loss权重 CE:MSE |
|------|------|-----|--------|----------|---------|-----------------|
| PT | 350k | 1e-4 | Constant | 44K-50K | 1.5T | 0.25:1 |
| CT | 80k | 1e-4 | Constant | 74K-80K | 300B | 0.5:1 |
| SFT | 15k | 2.5e-5 | Cosine | 74K-80K | 72B | 0.25:1 |
| RL | 800 | 2e-6 | Constant | 74K-80K | 0.5B | - |

所有阶段使用 **AdamW** ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-15}$)，gradient norm clip 1.0，weight decay 0.0。

### 3.2 PT（Pre-Training）：基础能力建立

**目标**：建立初步多模态对齐和基础视觉生成能力。冻结 VAE 和 ViT encoder，训练 backbone + QK-Norm + MLP connector。

**数据**：
- 图像-文本对：约 1B 样本（自然场景、人物、物体、知识、风格化内容）
- 视频-文本对：约 140M 样本（动作、事件、场景转换、长程时序过程）
- 渐进分辨率课程：192p → 360p → 480p，动态分辨率
- 图像:视频采样比约 1:4，优先视频（视频建模更难）

**任务格式**：主要组织为配对 captioning 和条件生成任务。

**数据配比**（全局）：Video-Gen : Video-Und : Image-Gen : Image-Und = 64:16:16:4

**生成子类配比**：
- T2I : I-Edit : S2I = 100:0:0（PT 阶段只有纯 T2I）
- T2V : I2V : V-Edit : S2V = 100:0:0:0（PT 阶段只有纯 T2V）

### 3.3 CT（Continual Training）：多任务空间扩展

**目标**：从基础配对监督扩展到统一多任务学习，引入更丰富的交错多模态数据和更多样化的输入-输出映射。

**数据**（新增）：
- 理解：2.73M 交错多模态样本（纯文本 41K、caption 443K、分类 142K、对话 72K、定位 200K、推理 194K、VQA 600K、OCR 120K）
- 生成：2.8M 图像编辑、2.6M 视频编辑、3.6M 主题驱动图像生成、1M 主题驱动视频生成

**CT 三阶段渐进数据策略**：

| 子阶段 | T2I : I-Edit : S2I | T2V : I2V : V-Edit : S2V |
|--------|---------------------|---------------------------|
| CT-I | 70:15:15 | 60:10:15:15 |
| CT-II | 60:20:20 | 40:20:20:20 |
| CT-III | 50:25:25 | 25:25:25:25 |

逐步增加编辑和主题驱动生成等更难任务的采样比例。

### 3.4 SFT（Supervised Fine-Tuning）：指令精调

**目标**：用高质量、任务对齐的监督信号精调指令遵循、视觉保真度、编辑精度和身份一致性。

**数据**（精选高质量）：
- 理解：190K 图像 caption、5K 视频 caption、2.73M 交错多模态理解
- 图像生成：190K 高质量 T2I、84K 高质量图像编辑
- 视频生成：5K 高质量 T2V/I2V、9K 高质量视频编辑、5.5K 高质量主题驱动视频生成

**生成配比**：T2I : I-Edit : S2I = 60:20:20，T2V : I2V : V-Edit : S2V = 60:10:15:15

### 3.5 RL（Reinforcement Learning）：奖励优化

**目标**：用 GRPO 直接优化图像生成行为，改善文本渲染精度和图文对应。

**数据**：20K 图像生成 prompt（强调细粒度文本要求），PaddleOCR 作为奖励模型评估文本-图像一致性。

### 3.6 训练数据配比总览

全局配比（所有阶段）：**Video-Gen : Video-Und : Image-Gen : Image-Und = 64:16:16:4**

视频生成占主导（64%），因为视频建模难度最大，需要更多训练信号。

任务统计（全阶段合计）：
- Text 输出：图像 caption 1B + 视频 caption 140M + 交错理解 2.73M + HQ caption 195K
- Image 输出：T2I 1B + 编辑 2.8M + 主题驱动 3.6M + HQ 生成/编辑 274K
- Video 输出：T2V/I2V 140M + 编辑 2.6M + 主题驱动 1M + HQ 生成/编辑 14K


## 4. 训练目标与推理

### 4.1 理解目标：Next-Token Prediction

理解专家 LLM_UND 处理文本 token 和 ViT 语义 token，自回归预测目标文本：

$$\mathcal{L}_{UND} = -\sum_i \log p_{\theta_{UND}}(y_i | y_{<i}, S)$$

- 通过 LM head 映射 hidden state → vocab logits
- 只计算目标文本位置（`ce_loss_indexes` 标记的部分）的交叉熵
- 超出有效 vocab 的 token（如视觉 token）设为 ignore_index=-100

### 4.2 生成目标：Flow Matching Velocity Prediction

生成专家 LLM_GEN 处理 VAE latent token，在连续隐空间中做 velocity prediction：

$$\mathcal{L}_{GEN} = \mathbb{E}_{x_0, x_1, t}\left[\|v_{\theta_{GEN}}(x_t, S, t) - (x_1 - x_0)\|^2_2\right]$$

其中：
- $x_1$：干净 VAE latent（clean）
- $x_0 \sim \mathcal{N}(0, I)$：高斯噪声
- $x_t = t x_1 + (1 - t) x_0$：插值 latent，$t \sim U(0, 1)$ 在训练时随机采样
- $v_{\theta_{GEN}}$：预测的 velocity 向量（方向从噪声指向数据，$x_1 - x_0$）
- 只计算有噪声的 token 位置（`mse_loss_indexes` 标记，即 timestep > 0 的位置）

**Timestep 处理**：
- 训练时：$t \sim U(0, 1)$ → sigmoid → timestep shift
- Timestep shift = $\frac{\alpha \cdot t}{1 + (\alpha - 1) \cdot t}$（PT 阶段 $\alpha=1.0$，后续 $\alpha=4.0$）
- 推理时：$t$ 从 1 → 0，24 步 Euler 求解

**总Loss**：$\mathcal{L} = \lambda_u \mathcal{L}_{UND} + \lambda_g \mathcal{L}_{GEN}$

### 4.3 Classifier-Free Guidance（CFG）

推理时使用 CFG 增强生成质量：

- **文本 CFG**：训练时文本条件以 10%（PT）或 5%+5%（CT/SFT）概率丢弃；推理时 CFG scale=4.0
- **视觉 CFG**：对 TI2I（text-image-to-image）等需要视觉条件的任务，额外做视觉条件无条件化
- **CFG 重归一化**（cfg_renorm）：global 或 channel 维度重新缩放，防止 CFG 导致的过大向量

$$
v_t = v_{uncond} + s_{text} \cdot (v_{cond} - v_{uncond}) + s_{vision} \cdot (v_{text\_cond} - v_{text\_vision\_cond})
$$

**源码位置**：[lance.py L590-L628](https://github.com/bytedance/Lance/blob/main/modeling/lance/lance.py#L590-L628)

三层 CFG 展开：
1. 纯无条件 → 只有文本条件（text CFG）
2. 文本条件 → 文本+视觉条件（vision CFG）
3. 最终 CFG 结果经过 norm 重缩放

### 4.4 推理流程

**生成推理**：[lance.py L315-L737](https://github.com/bytedance/Lance/blob/main/modeling/lance/lance.py#L315-L737)
1. 构建统一多模态序列（text + ViT + clean VAE）→ LLM 前向得 KV cache
2. 初始化 $x_t \sim \mathcal{N}(0, I)$（纯噪声）
3. for t in 1→0（24步 Euler 法）:
   - $x_t$ → VAE embed + timestep embed + 3D pos embed → 填入序列
   - LLM 前向（用 KV cache 获取条件上下文 + 当前噪声段的 attention）→ hidden state
   - LLM→VAE connector → velocity $v_t$
   - CFG 校正（文本 + 视觉双重 CFG + renorm）
   - $x_t = x_t - v_t \cdot dt$（Euler 步进）
4. 最终 $x_0$ → VAE decoder → 像素空间

**理解推理**：[lance.py L950-L1149](https://github.com/bytedance/Lance/blob/main/modeling/lance/lance.py#L950-L1149)
- KV cache 模式：分两步
  1. 预填充：text + ViT → LLM 前向，构建 KV cache
  2. 自回归解码：逐 token 生成，LM head → logits → argmax/sample

**多图/交错推理**：[lance.py L1151-L1346](https://github.com/bytedance/Lance/blob/main/modeling/lance/lance.py#L1151-L1346)
- 支持多张图像/视频在同一序列中交错排列
- 每张图有自己的 `<start_of_image> ... <end_of_image>` 边界

### 4.5 训练中的 Packed Sequence 打包

所有样本在训练时被打包（pack）到大序列中以最大化 GPU 利用率：
- 每个 rank 的序列长度 44K-80K token（随阶段增大）
- 多个样本拼接用 attention mask 隔离（`sample_lens` + `split_lens` + `attn_modes`）
- `create_sparse_mask` 构建块稀疏 attention mask，`flex_attention` 的 `create_block_mask` 编译执行


## 5. 实验

### 5.1 实验配置

- **初始化**：Qwen2.5-VL 3B 权重初始化 ViT encoder + LLM backbone（LLM_UND + LLM_GEN）
- **QK-Norm**：每个 attention block 配备 QK-Norm（改变原始 Qwen2.5-VL 的 Q-K 激活分布，需大幅重训）
- **VAE**：Wan2.2 3D causal VAE（图像视频统一隐空间）
- **推理分辨率**：图像 768×768，视频 480p@12fps
- **CFG**：text scale=4.0，CFG interval=[0, 1]，num_timesteps=24，timestep_shift=3.5（推理）

### 5.2 图像生成

| Benchmark | Lance 3B | 最佳统一模型 | 最佳专用模型 |
|-----------|----------|-------------|-------------|
| GenEval Overall | **0.90** | TUNA 7B 0.90 | Qwen-Image 20B 0.87 |
| DPG-Bench Overall | 84.67 | TUNA 7B 86.76 | Qwen-Image 20B 88.32 |

GenEval 各维度：
- 1-Obj: 1.00, 2-Obj: 0.94, Count: **0.84**, Colors: **0.97**, Position: **0.87**, Attr: 0.81

DPG-Bench 各维度：
- Global: 83.89, Entity: 91.07, Attribute: 89.36, Relation: **93.38**, Other: 80.80

**关键观察**：以 3B 参数匹配或超过 7B 统一模型，在计数、颜色、位置等组合性维度上尤其强。

### 5.3 图像编辑

GEdit-Bench Avg/G_O：**7.30**（统一模型中最佳）
- BC（背景变化）7.73, CA（颜色属性）7.74, MM（材质修改）7.28, MC（运动变化）**7.83**
- PB（人像美化）**7.50**, ST（风格迁移）7.03, SA（主体添加）7.64, SR（主体移除）**7.85**
- SRp（主体替换）**7.71**, TT（色调迁移）**7.57**
- 弱项：TM（文本修改）4.46 — 文本编辑仍是难点

### 5.4 视频生成

VBench Total Score：**85.11**（统一模型中最佳，超过部分专用模型如 Wan2.1-T2V 14B 的 83.69）

质量维度：Quality Score 85.14, Semantic Score **84.96**
- Motion Smooth: **99.66**, Aesthetic: 64.33, Dynamic Degree: **75.83**
- Object Class: **96.58**, Multi Objects: 68.90, Human Action: **96.40**
- Color: **85.14**, Spatial Relation: **79.05**, Scene: **58.22**

强项：运动平滑、动态程度、物体类别、人物动作、色彩；弱项：场景理解

### 5.5 视频理解

MVBench Overall：**62.0**（统一模型最佳，第二名 Show-o2 7B 55.7）

以 3B 参数超越大多数专用理解模型（通常 7B+），证明统一多任务训练没有牺牲理解能力。

### 5.6 多任务协同效应（关键消融）

**理解数据对生成的增益**（Table 9）：
| Setting | GenEval | VBench |
|---------|---------|--------|
| Gen only | 80.88 | 81.25 |
| Gen:Und = 8:2 | **81.65** | **82.91** |
| Gen:Und = 9:1 | 80.93 | 81.47 |

理解数据在适当比例下（8:2）同时提升图像和视频生成，说明理解数据为视觉合成提供了有用的语义基础。

**多任务生成数据对理解和生成的增益**：
| Setting | GenEval | VBench | MVBench |
|---------|---------|--------|---------|
| Gen:MT-Gen = 8:2 | 81.89 | 82.88 | **59.18** |
| Gen:MT-Gen = 6:4 | **82.06** | **83.05** | 58.95 |

更令人意外的是：引入多任务生成数据（编辑、主题驱动等）不仅提升生成，还**提升了视频理解**（MVBench 58.06→59.18）。这直接验证了作者的核心主张：多任务协同不是简单的叠加，而是**相互增强**。

### 5.7 训练动态分析

- 图像和视频生成在 PT 早期快速增益，随后进入慢速精调阶段
- CT 阶段虽主要引入编辑和理解数据（非纯生成数据），但仍持续改善生成质量
- 多任务整合不仅增强编辑和指令遵循，对纯生成也有正向迁移
- 从 0.5T→1T→1.5T token，模型在 prompt 对齐、视觉质量、文本渲染、时序连贯性上持续改善


## 6. 代码架构

### 6.1 目录结构

```
Lance/
├── modeling/
│   ├── lance/
│   │   ├── lance.py          # 主模型 Lance class（~1800行）
│   │   ├── qwen2_navit.py     # Backbone: DecoderLayer variants + Qwen2Model（~1300行）
│   │   └── modeling_utils.py  # TimestepEmbedder, MLPconnector, PositionEmbedding
│   ├── qwen2/                 # Qwen2 基础组件（Attention, MLP, RMSNorm, RoPE）
│   ├── qwen2_5_vl/            # Qwen2.5-VL 特有组件（VLRotaryEmbedding, 3D-MRoPE）
│   ├── vae/wan/               # Wan2.2 VAE（vae2_2.py ~ 3D causal VAE）
│   └── vit/qwen2_5_vl_vit.py # ViT encoder
├── data/                      # 数据处理
│   ├── dataset_base.py        # 基类 Dataset
│   ├── data_utils.py          # create_sparse_mask, position_ids 函数
│   ├── transforms.py          # 图像/视频变换
│   └── video/                 # 视频专用采样器和变换
├── config/
│   └── config_factory.py      # TrainingArguments 配置
├── benchmarks/                # 评估脚本（DPG, GenEval, GEdit, VBench）
├── inference_lance.py         # 推理入口
└── lance_gradio.py            # Gradio demo
```

### 6.2 关键数据流

训练 forward 一次参见 [lance.py L148-L312](https://github.com/bytedance/Lance/blob/main/modeling/lance/lance.py#L148-L312)：

1. 输入接收：packed 后的 text_ids, vit_tokens, padded_latent, position_ids, attention_modes 等
2. 文本嵌入：language_model.model.embed_tokens(packed_text_ids) -> text embedding
3. ViT 编码（if visual_und）：冻结的 vit_model 前向 -> connector（2层MLP）-> semantic tokens
4. VAE latent 处理（if visual_gen）：
   - 干净的 latent -> patchify（rearrange to patches）
   - 添加噪声：x_t = (1-t) * x_clean + t * noise
   - time_embedder(t) + latent_pos_embed(pos_ids) + vae2llm(x_t) -> VAE embedding
5. 序列组装：text emb + ViT emb + VAE emb -> packed_sequence（按 indexes 填入）
6. LLM 前向：language_model(packed_sequence, attention_mask, ...) -> last_hidden_state
   - 内部遍历所有 DecoderLayer（标准/MLP-MoE/Full-MoE）
   - 每层根据 packed_und_token_indexes / packed_gen_token_indexes 路由 token 到对应专家
7. Loss 计算：
   - CE：lm_head(h[ce_loss_indexes]) -> cross_entropy with labels
   - MSE：llm2vae(h[mse_loss_indexes]) -> MSE with velocity target (noise - x_clean)

### 6.3 QK-Norm 的作用

Qwen2.5-VL 原始模型没有 QK-Norm。Lance 在每个 attention block 添加 QK-Norm，改变了原始 Q-K 激活分布，因此不是直接复用 Qwen2.5-VL 权重，而是需要大幅重训。QK-Norm 的好处是稳定大规模统一多任务训练的 attention 计算。

### 6.4 生成推理的 CFG 实现细节

参见 [lance.py L536-L628](https://github.com/bytedance/Lance/blob/main/modeling/lance/lance.py#L536-L628)：

1. uncond_split_pro_new：从条件序列中提取无条件子序列（去掉文本描述等）
2. uncond_forward：用无条件子序列前向得到 cfg_text_v_t
3. vision CFG（可选）：进一步去掉视觉条件得到 cfg_text_vision_v_t
4. 结果合成：v_t_ = cfg_text_vision_v_t + s_text*(v_t - cfg_text_v_t) + s_vision*(cfg_text_v_t - cfg_text_vision_v_t)
5. 重归一化：scale = norm(v_t) / norm(v_t_)，clamp(min=cfg_renorm_min)

### 6.5 KV Cache 模式的 Attention Mask 管理

在推理时，序列按 attn_modes 分段的规则管理 attention：
- causal 段之间用 flash_attn_varlen_func(..., causal=True)
- full 段内（视觉 token）用 causal=False
- NaiveCache 类手动管理 key/value cache（逐层更新）

条件上下文预先计算并保存在 KV cache 中，噪声生成段每次迭代只做 query 对 cache 的 attention（不更新噪声段在 cache 中的 key/value，因为噪声在变化）。

## 7. 与现有统一模型的对比

Lance 的核心差异在于：

- **vs. Chameleon/Emu3**：Lance 不用纯自回归生成图像，而是用 Flow Matching 在连续隐空间生成，质量和效率更高
- **vs. Janus/Janus-Pro**：Lance 的解耦更彻底（不仅是视觉编码器解耦，backbone 也通过 MoE 解耦），且支持视频
- **vs. Show-o2**：类似的自回归+扩散混合，但 Lance 引入了 MoE 专家解耦和 MaPE，且任务覆盖更广
- **vs. BAGEL**：都做专家解耦，但 Lance 的 MoE 设计更系统（3 种 DecoderLayer 选项），且支持视频
- **vs. TUNA**：Lance 不用统一视觉表示，而是让理解和生成各自用最合适的表示
- **vs. InternVL-U**：Lance 是原生统一（从头联合训练），不是将单独训练的组件拼接

## 8. 关键启示

### 8.1 多任务协同 > 多任务叠加

最重要的发现：编辑和主题驱动生成等"额外"任务的加入不仅增强了对应能力，还反过来提升了基础 T2I/T2V 和理解。这验证了多任务统一训练的核心价值——不是简单的能力聚合，而是跨任务的正向迁移。

### 8.2 解耦表示 + 统一上下文 = 最优平衡

完全统一表示（如 TUNA）简化了建模但可能牺牲分别优化空间；完全解耦（如 Janus 的双 ViT）增加了复杂度但丢失了共享上下文。Lance 的方案——共享多模态序列 + 解耦专家通路——在两者间取得了好的平衡。

### 8.3 MaPE：用小设计解决大问题

仅 1000 的时间维度偏移就带来了跨任务的稳定提升。核心洞察：在统一序列中，不同类型的视觉 token 需要明确的位置区分，空间维度保持不变保留内在结构，时间维度加偏移提供 token 组感知。

### 8.4 128 GPU 预算的工程务实性

Lance 证明了不需要千卡集群也能训练有竞争力的统一多模态模型，对小团队和企业有重要参考价值。关键效率手段：packed sequence、动态分辨率、冻结 encoder、渐进分辨率课程。

### 8.5 视频理解不牺牲

大多数统一模型在加入生成能力后理解能力下降，但 Lance 在 MVBench 上以 3B 参数超越专用模型，说明多任务训练如果架构设计得当，理解能力可以保持甚至增强。

### 8.6 文本渲染仍是短板

RL 阶段用 PaddleOCR 奖励改善了文本渲染，但编辑中文本修改能力仍弱（GEdit TM: 4.46 vs 其他维度 7+），论文也承认需要专门的文本渲染数据。

### 8.7 未来的扩展方向

- 音频、语音、3D 等更多模态
- 流式多模态交互（实时感知+生成）
- 视频专用奖励模型
- 模型规模扩展和上下文长度扩展
