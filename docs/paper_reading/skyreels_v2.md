---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - Diffusion Forcing
  - Flow Matching
---
# SkyReels-V2: Infinite-Length Film Generative Model

**论文**: [arXiv 2504.13074](https://arxiv.org/abs/2504.13074) | **代码**: [GitHub](https://github.com/SkyworkAI/SkyReels-V2) | **团队**: Skywork AI

## 概述

SkyReels-V2 是 Skywork AI 提出的开源无限时长视频生成模型，目标是解决当前视频生成在镜头语言对齐、运动质量、视频时长三方面的瓶颈。

**镜头语言对齐**方面，现有方法依赖通用 MLLM（如 Qwen2.5-VL）做视频标注，但通用模型对景别、运镜、表情等电影专业概念的理解不足，导致生成结果丢失专业描述。SkyReels-V2 设计了结构化视频标注格式，将视频描述拆解为主体、景别、角度、运镜、表情等独立字段，分别训练子专家分类器，再将通用 MLLM + 子专家的知识蒸馏到一个 7B 统一模型 SkyCaptioner-V1 中。蒸馏后的 7B 模型在镜头相关字段上准确率达 76.3%，大幅超越直接使用 72B 教师模型的 58.7%。

**运动质量**方面，标准去噪损失偏重逐帧外观学习，对时序动态一致性优化不足，导致生成视频中出现大幅度变形、物理违反、局部损坏等问题。SkyReels-V2 采用 Flow-DPO 强化学习方法，但只聚焦运动质量这一个维度——偏好数据对在文本对齐和视觉质量上保持可比，仅运动质量不同，避免了多目标权重调优的困境。为解决偏好数据的标注成本问题，提出了半自动流水线：用真实视频作为 chosen 样本，通过渐进失真（V2V→I2V→T2V 三级）构造 rejected 样本，配合人工标注共 30k 对数据训练 Reward Model，再用 Reward Model 排序生成 DPO 训练数据，分 3 阶段迭代优化。

**视频时长**方面，纯 Diffusion 模型产生时序碎片，纯 AR 模型累积误差，均难以生成长视频。SkyReels-V2 采用 Diffusion Forcing 后训练策略——为每一帧分配独立噪声水平，已清晰的帧作为条件引导高噪声帧去噪。关键在于不从头训练，而是从预训练的全序列扩散模型直接微调（全序列同步扩散是 Diffusion Forcing 的特例）。引入 AR-Diffusion 的非递减噪声调度（FoPP），通过动态规划将组合空间从 $O(10^{48})$ 降至 $O(10^{32})$，显著稳定训练。推理时使用滑动窗口方式，理论上支持无限时长生成。

整体训练流程为：数据标注 → 三阶段渐进分辨率预训练（256p→360p→540p，Flow Matching + 双轴 Bucketing）→ 540p 高质量 SFT → 运动质量 RL → Diffusion Forcing 后训练 → 720p 高质量 SFT + DMD 蒸馏。在 VBench 1.0 上取得开源最高分 83.9%，SkyReels-Bench 人类评估中指令遵循显著超越 Wan2.1-14B，模型规模覆盖 1.3B/5B/14B，全系列开源。

---

## 1. 数据处理

### 1.1 数据来源

- 开源数据集：Koala-36M、HumanVid 等
- 自采集影视：280k+ 电影、800k+ 剧集，覆盖 120+ 国家，估计总时长 6.2M+ 小时
- 高质量艺术视频
- 概念平衡图像数据：$O(100M)$ 级别，用于加速早期训练

原始数据集达 $O(100M)$ 量级，不同子集用于不同训练阶段。

### 1.2 处理流水线

**预处理**：镜头边界检测（PyDetect + TransNet-V2）将原始视频切分为单镜头片段，再用 SkyCaptioner-V1 标注。

**数据过滤**：从松到严的渐进策略。质量问题分三大类：

- 基础质量：低分辨率（<720p）、低帧率（<16fps）、黑屏/白屏/静态、相机抖动、不稳定运动、随意镜头切换
- 视频类型问题：监控画面、游戏录屏、动画、无意义内容、静态视频
- 后处理痕迹：字幕、Logo、画中画、分屏、变速、特效/马赛克

过滤器包括：黑屏过滤（启发式规则）、静态屏幕过滤（光流评分）、美学过滤（美学评分模型）、去重（copy-detection 嵌入空间相似度）、OCR 过滤（文本占比检测）、马赛克/特效检测（训练的专家模型）、VQA/IQA/VTSS 质量评分。不同训练阶段设置不同阈值。

**字幕和 Logo 裁剪**：不直接丢弃带字幕/Logo 的影视数据（浪费），而是检测后裁剪。字幕检测用 CRAFT OCR 在帧边界候选区域（上 20%、下 40%、左右各 20%）做 OCR，Logo 检测用 MiniCPM-o 在四角区域（各 15%）检测。然后用单调栈算法（Algorithm A1）找最大无遮挡内接矩形，覆盖率 >80% 且宽高比接近原帧时保留裁剪结果。

**概念平衡**：后训练阶段按主体类别（captioner 提供的 subject type）做平衡。平衡前 Human 类占 84.8%，平衡后降至 63.2%，整体数据量减少约 50%。

### 1.3 Human-In-The-Loop 验证

每阶段抽样人工检查，预训练阶段抽样率 0.01%（1/10000），要求总坏例 <15%（基础质量 <3%、类型问题 <5%、后处理 <7%）。后训练阶段抽样率 0.1%（1/1000），要求总坏例 <3%（基础 <0.5%、类型 <1%、后处理 <1.5%）。超过阈值的数据批次会被丢弃或回炉，过滤参数也会根据不同数据源的特点动态调整。

---

## 2. SkyCaptioner-V1：结构化视频标注

### 2.1 结构化描述设计

为每个视频片段生成多维结构化 JSON，字段包括：

- **主体信息**：类型/子类型（如 Human→Woman、Animal→Mammal）、外观、动作、表情、画面位置、是否主体
- **镜头元数据**：景别（close-up / extreme close-up / medium / long / full shot）、角度（eye level / high / low）、位置（front / back / over-shoulder / POV / side / over-head）
- **运镜**：6DoF 参数化（平移 x/y/z + 旋转 roll/pitch/yaw），每轴离散为负/零/正三态，配合三档速度（slow <5% / medium 5-20% / fast >20% 帧位移/秒）
- **环境与光照**

T2V 模式生成密集描述，I2V 模式聚焦「主体 + 时序动作/表情 + 运镜」。每个字段有 10% 的随机丢弃率，适应用户不提供完整描述的场景。

### 2.2 子专家模型

通用 MLLM（Qwen2.5-VL-72B-Instruct）处理一般描述，以下领域由独立分类器/描述器补充：

**镜头分类器**（Shot Captioner）：覆盖景别、角度、位置三个子任务。两阶段训练——先用网图（以类别标签为关键词爬取）训练粗分类器建立 baseline，再用粗分类器从影视数据中提取平衡样本，每类 2k 条人工标注训练精分类器。精度：景别 82.2%、角度 78.7%、位置 93.1%。

**表情描述器**（Expression Captioner）：先做人脸检测和 7 类情绪分类（neutral/anger/disgust/fear/happiness/sadness/surprise），再将情绪标签 + 视频帧输入 InternVL2.5，用 Chain-of-Thought 生成详细表情描述。基于 S2D 框架训练，约 10k 内部数据集，覆盖人类和非人类角色。评测精度：情绪标签 88%、强度 95%、面部特征 85%、时序描述 93%。

**运镜描述器**（Camera Motion Captioner）：三阶段流水线。(1) 运动复杂度过滤：静止镜头二分类器（95% 精度）筛除无运动片段，不规则运动分类器（手持抖动/跟拍/突变）筛除复杂运动，剩余为标准单类型运动。(2) 单类型运动建模：6DoF 每轴 3 态 × 3 档速度 = 2187 种组合，混合人工标注 + 合成样本训练。(3) 主动学习扩展：5 轮循环（$O(10k)$ 人工标注 → 预测 100k → 平衡抽样 10k 验证 → 精调），最终得到 93k 高置信样本 + 16k 合成数据。精度：单类型 89%、手持 78%、跟拍 83%、突变 81%。

### 2.3 蒸馏为统一模型

将 72B 通用模型 + 所有子专家的标注结果蒸馏到 Qwen2.5-VL-7B-Instruct 上，训练数据为 200 万概念平衡视频（从 1000 万中筛选）。训练配置：64 × A800 GPU，batch size 512（micro batch 4 × gradient accumulation 2），AdamW lr=1e-5，2 epochs。

蒸馏效果（1000 样本测试集，人工评估各字段准确率）：

- SkyCaptioner-V1 平均 76.3%，远超 Qwen2.5-VL-72B 的 58.7% 和 Qwen2.5-VL-7B 的 51.4%
- 镜头相关字段提升尤其显著：景别 93.7%（vs 72B 的 82.5%）、角度 89.8%（vs 73.7%）、位置 83.1%（vs 32.7%）、运镜 85.3%（vs 61.2%）

---

## 3. 多阶段预训练

### 3.1 模型架构

采用 Wan2.1 的 DiT 架构，仅从头训练 DiT 部分，VAE 和文本编码器（umT5，512 维特征）沿用预训练权重。

### 3.2 训练目标：Flow Matching

给定 latent $x_1$（图像或视频），采样时间步 $t \sim \text{logit-normal}[0, 1]$，噪声 $x_0 \sim \mathcal{N}(0, I)$。

线性插值构造中间状态：

$$
x_t = t \cdot x_1 + (1-t) \cdot x_0
$$

ground-truth 速度向量：

$$
v_t = \frac{dx_t}{dt} = x_1 - x_0
$$

模型 $u_\theta(x_t, c, t)$ 预测速度场，条件 $c$ 为文本嵌入，最小化：

$$
\mathcal{L} = \mathbb{E}_{t, x_0, x_1, c} \left[ \| u_\theta(x_t, c, t) - v_t \|^2 \right]
$$

### 3.3 双轴 Bucketing + FPS 归一化

视频数据的时空异质性通过双轴分桶框架处理：

- 时间轴按时长分 $B_T$ 个桶，空间轴按宽高比分 $B_{AR}$ 个桶，形成 $B_T \times B_{AR}$ 矩阵
- 每个桶通过经验性 GPU profiling 独立设定最大 batch 容量，防止 OOM
- 训练时各节点随机采样桶，保证输入分辨率和时长的持续变化

FPS 归一化：残差感知降采样，选择余数最小的目标帧率 $f_{\text{target}} = \arg\min_{f \in \{16, 24\}} (\text{original\_fps} \bmod f)$。DiT 增加可学习频率嵌入（与 timestep embedding 相加），高质量 SFT 阶段统一为 FPS-24 后丢弃此嵌入。

### 3.4 三阶段渐进训练

**Stage 1（256p）**：图像+视频联合训练，支持多宽高比和帧长。去重 + 合成数据过滤 + 基础过滤。学习率 1e-4（收敛后降至 5e-5，weight decay 从 0 调至 1e-4）。学习低频概念，生成结果较模糊。

**Stage 2（360p）**：继续图像+视频联合训练，增加时长/运动/OCR/美学/质量过滤。学习率 2e-5。清晰度显著提升。

**Stage 3（540p）**：仅视频训练，增加来源过滤（去除 UGC 保留影视数据），更严格的运动/美学/质量过滤。学习率 2e-5。专注影视质感和人体纹理的真实感。

全程使用 AdamW 优化器。

---

## 4. 后训练

后训练是提升整体性能的关键阶段，包含四步：540p 高质量 SFT → 强化学习 → Diffusion Forcing → 720p 高质量 SFT。前三步在 540p 下完成以提高效率。

### 4.1 540p 高质量 SFT

预训练完成后、RL 之前执行的第一阶段 SFT。使用概念平衡的高质量数据，统一为 fps24 并移除 FPS embedding，为后续阶段提供良好的初始化。

### 4.2 强化学习：运动质量优化

**设计原则**：只优化运动质量，不动文本对齐和视觉质量。偏好数据对在后两者上保持可比，仅运动质量不同，避免多目标权重调优。

主要运动缺陷：过度/不足的运动幅度、主体变形、局部细节损坏、物理规律违反、不自然运动。

**人工标注偏好数据**：

1. 分析运动伪影，记录失败模式对应的 prompt，用 LLM 扩展生成多样 prompt
2. 每个 prompt 用历史 checkpoint 池生成 4 个样本，系统配对
3. 过滤：内容/质量不匹配或主体不清晰/太小/背景复杂的样本对被排除（约 80% 被丢弃）
4. 人工标注 Better/Worse/Tie，按加权运动质量评分表评估

**自动生成偏好数据**（解决人工标注成本过高、约 80% 丢弃率的问题）：

1. Ground Truth 收集：用生成 prompt 的 CLIP 特征检索语义匹配的真实视频作为 chosen 样本
2. 渐进失真创建 rejected 样本：
    - V2V：直接反转噪声 latent（最低失真）
    - I2V：首帧引导重建（中等失真）
    - T2V：纯文本重新生成（最高失真）
3. 使用不同生成模型（Wan2.1、HunyuanVideo、CogVideoX）和不同参数（如 timestep）构建多级运动质量
4. 额外技术：变帧率采样模拟过度/不足运动、Tea-Cache 参数调节模拟局部损坏、视频倒放模拟物理违反

**Reward Model**：基于 Qwen2.5-VL-7B-Instruct，共 30k 样本对。运动质量与上下文无关（context-agnostic），样本对不含 prompt。损失函数采用 Bradley-Terry with Ties (BTT)：

$$
\mathcal{L} = -\sum_{(i,j)} \left[ y_{i>j} \ln P(i>j) + y_{i<j} \ln P(i<j) + y_{i=j} \ln P(i=j) \right]
$$

**Flow-DPO 训练**：采用 VideoAlign 的 Flow-DPO 框架：

$$
\mathcal{L}_{\text{DPO}} = -\frac{1}{N} \sum_{i=1}^{N} \log \sigma \left( -\frac{\beta}{2} \left[ \Delta_{\text{model}} - \Delta_{\text{ref}} \right] \right)
$$

其中 $\Delta_{\text{model}} = L^w_{\text{model}} - L^l_{\text{model}}$，$\Delta_{\text{ref}} = L^w_{\text{ref}} - L^l_{\text{ref}}$，$L^{w/l} = \frac{1}{2}\|\hat{y}^{w/l} - y\|^2$。$\hat{y}^{w/l}$ 为模型对 chosen/rejected 样本的预测，$\beta$ 为温度系数。

训练数据构建：两类 prompt 集（概念平衡 + 运动特定），每个 prompt 生成 8 个视频，Reward Model 排序后选 best/worst 构成三元组。

分阶段迭代：当模型能轻松区分 chosen/rejected（性能趋于平台）时，刷新 reference model 为最新迭代，重新生成排序数据。每阶段 20k 数据，共 3 阶段。

### 4.3 Diffusion Forcing：无限时长生成

**核心思想**：为每一帧分配独立噪声水平，实现部分遮掩——零噪声帧作为条件引导高噪声帧去噪。全序列同步扩散是其特例（所有帧共享相同噪声），因此可从预训练的全序列扩散模型直接微调，无需从头训练。

信息流天然单向（噪声帧依赖清晰历史帧），双向注意力可替换为因果注意力，支持 KV cache 加速推理。

**FoPP 非递减噪声调度**（训练时）：采用 AR-Diffusion 的 Frame-oriented Probability Propagation：

1. 均匀采样帧索引 $f \sim U(1, F)$ 和时间步 $t \sim U(1, T)$
2. 动态规划计算以 $t_f = t$ 为条件的前后帧时间步概率
3. 转移方程（非递减约束下从帧 $i$ 时间步 $j$ 开始的有效序列计数）：$d^s_{i,j} = d^s_{i,j-1} + d^s_{i-1,j}$，边界条件 $d_{*,T} = 1$，$d_{F,*} = 1$
4. 帧 $i$ 访问时间步 $k$ 的概率：$P(t_i = k) = d^s_{i,k} / \sum_{j=K}^{T} d^s_{i,j}$
5. 逐帧按概率采样时间步

非递减约束将组合空间从 $O(10^{48})$ 降至 $O(10^{32})$，显著稳定训练。

**AD 调度器**（推理时）：相邻帧时间步差 $s$ 作为自适应变量：

$$
t_i = \begin{cases}
t_i + 1, & \text{if } i = 1 \text{ or } t_{i-1} = 0 \\
\min(t_{i-1} + s, T), & \text{if } t_{i-1} > 0
\end{cases}
$$

$s = 0$ 退化为同步扩散，$s = T$ 为完全自回归。小 $s$ 相邻帧更相似，大 $s$ 内容变化更大。

**长视频生成**：滑动窗口方式，以前 $f_{\text{prev}}$ 帧 + prompt 为条件生成后续 $f_{\text{new}}$ 帧。已生成帧注入轻微噪声（而非完全清晰）缓解误差累积。

### 4.4 720p 高质量 SFT

Diffusion Forcing 后执行第二阶段 SFT，分辨率从 540p 提升到 720p，使用更高质量 + 手工筛选的概念平衡数据，进一步提升视觉质量。

---

## 5. 基础设施

### 5.1 训练优化

- **内存优化**：Attention fp32 算子融合减少 kernel launch 开销；梯度检查点仅存 transformer block 输入，转 bf16 内存减半；选择性 activation offloading 到 CPU（需平衡，8 GPU 共享 CPU 内存，过度 offloading 反而影响吞吐）
- **训练稳定性**：自愈框架——实时检测故障节点 → 动态资源重分配（备用计算单元）→ checkpoint 恢复迁移
- **并行策略**：预计算 VAE + 文本编码器结果；FSDP 分布 DiT 权重/优化器；720p 训练使用 Sequence Parallel 解决激活内存碎片（torch.empty_cache 被频繁触发）

### 5.2 推理优化

- **FP8 量化**：Linear 层动态量化 + FP8 GEMM（1.10× 加速，RTX 4090）；Attention 用 SageAttn2-8bit（1.30× 加速）
- **多 GPU 并行**：Content Parallel + CFG Parallel + VAE Parallel，4→8 RTX 4090 延迟降低 1.8×
- **DMD 蒸馏**：去除 regression loss，用高质量视频（非纯噪声）作为学生输入加速收敛，two time-scale update（fake/student 更新比 5:1），4 步生成（原 30-50 步）。梯度：$\nabla_\theta D_{\text{KL}} \simeq \mathbb{E}_{t,x} [(s_{\text{fake}}(x,t) - s_{\text{real}}(x,t)) \frac{dG}{d\theta}]$。小学习率 + 大 batch size 对稳定蒸馏至关重要
- **单卡部署**：RTX 4090（24GB）通过 FP8 + 参数 offloading 可运行 14B 模型生成 720p 视频

---

## 6. 实验结果

### 6.1 SkyReels-Bench（人类评估）

1020 条 prompt，20 位专业评估员，1-5 分制，评估指令遵循、一致性、视觉质量、运动质量四个维度。

**T2V 结果**：

| 模型 | 平均 | 指令遵循 | 一致性 | 视觉质量 | 运动质量 |
|------|------|----------|--------|----------|----------|
| Runway-Gen3 Alpha | 2.53 | 2.19 | 2.57 | 3.23 | 2.11 |
| HunyuanVideo-13B | 2.82 | 2.64 | 2.81 | 3.20 | 2.61 |
| Kling-1.6 STD | 2.99 | 2.77 | 3.05 | 3.39 | 2.76 |
| Hailuo-01 | 3.00 | 2.80 | 3.08 | 3.29 | 2.74 |
| Wan2.1-14B | 3.12 | 2.91 | 3.31 | 3.54 | 2.71 |
| **SkyReels-V2** | **3.14** | **3.15** | **3.35** | 3.34 | 2.74 |

指令遵循（尤其镜头语言）显著领先所有基线，一致性最优，运动质量与 Wan2.1 相当。

**I2V 结果**：

| 模型 | 平均 | 指令遵循 | 一致性 | 视觉质量 | 运动质量 |
|------|------|----------|--------|----------|----------|
| HunyuanVideo-13B | 2.84 | 2.97 | 2.95 | 2.87 | 2.56 |
| Wan2.1-14B | 2.85 | 3.10 | 2.81 | 3.00 | 2.48 |
| Hailuo-01 | 3.05 | 3.31 | 2.58 | 3.55 | 2.74 |
| SkyReels-V2-DF | 3.24 | 3.64 | 3.21 | 3.18 | 2.93 |
| SkyReels-V2-I2V | 3.29 | 3.42 | 3.18 | 3.56 | 3.01 |
| Kling-1.6 Pro | 3.40 | 3.56 | 3.03 | 3.58 | 3.41 |
| Runway-Gen4 | 3.39 | 3.75 | 3.20 | 3.40 | 3.37 |

SkyReels-V2-I2V 开源最优（3.29），接近闭源 Kling-1.6 Pro（3.40）和 Runway-Gen4（3.39）。

### 6.2 VBench 1.0（自动评测）

使用 longer version prompt，50 推理步，guidance scale 6：

| 模型 | 总分 | 质量分 | 语义分 |
|------|------|--------|--------|
| CogVideoX1.5-5B | 80.3% | 80.9% | 77.9% |
| OpenSora-2.0 | 81.5% | 82.1% | 78.2% |
| HunyuanVideo-13B | 82.7% | 84.4% | 76.2% |
| Wan2.1-14B | 83.7% | 84.2% | 81.4% |
| **SkyReels-V2** | **83.9%** | **84.7%** | 80.8% |

开源最高总分和质量分。语义分略低于 Wan2.1（VBench 对镜头语义评估不足）。

---

## 7. 应用

**Story Generation**：滑动窗口方式，条件前 $f_{\text{prev}}$ 帧 + prompt 生成后续 $f_{\text{new}}$ 帧。支持单 prompt 延展（30s+ 长镜头）和多 prompt 叙事（顺序 prompt 控制动作/表情/状态变化）。

**Image-to-Video**：两种路径。(1) SkyReels-V2-I2V：全序列 T2V 模型 + 首帧注入（VAE latent 拼接噪声 latent + 4 通道 binary mask），新增卷积层和 image-to-value 投影零初始化避免微调初期波动，384 GPU × 10k iters。(2) SkyReels-V2-DF：直接用 Diffusion Forcing 的首帧条件机制，无需额外训练。

**Camera Director**：从 SFT 数据集筛选 100 万运镜平衡样本，在 I2V 模型上微调（384 GPU × 3k iters），增强运镜流畅性和多样性。

**Elements-to-Video**：基于 SkyReels-A2 框架，支持多参考图像（角色/物体/背景）+ 文本 prompt 组合生成视频。未来计划扩展支持音频和 pose 输入。

---

## 8. 关键启示

1. **结构化标注 > 通用描述**：将视频描述拆解为结构化字段 + 子专家分类器，蒸馏到 7B 小模型后在镜头字段上大幅超越 72B 教师（76.3% vs 58.7%）
2. **RL 聚焦单一维度**：只优化运动质量，通过数据构建保证其他维度不退化，避免多目标权重调优
3. **半自动偏好数据**：渐进失真（V2V→I2V→T2V）构造 rejected + 真实视频检索 chosen，大幅降低标注成本
4. **Diffusion Forcing 后训练**：从全序列模型微调而非从头训练，非递减噪声约束将搜索空间从 $O(10^{48})$ 降至 $O(10^{32})$
5. **工程务实**：单卡 4090 跑 14B 模型（FP8 + offloading），DMD 蒸馏 4 步出图，self-healing 框架保障大规模训练

开源资产：1.3B / 5B / 14B 三种规模，涵盖 diffusion-forcing、T2V、I2V、camera director、E2V 全系列模型及 SkyCaptioner-V1。
