# WanPE 方法、评测与开放状态

## 1. 定位

WanPE（[arXiv:2609.30221](https://arxiv.org/abs/2609.30221)）是接在视频生成器前的电影化 prompt enhancer。它把真实视频中的镜头、动作、摄影、光照和声音组织成文本条件，再交给 Wan3.0 渲染；公开项目页目前展示论文和预生成 demo。

## 2. 核心结论

WanPE 将真实视频中的电影化结构转化为可监督学习的文本条件，再用语义一致性强化学习约束跨镜头的要求保持。论文报告的结果表明，这一训练链条同时改善了增强文本的一致性和 Wan3.0 生成视频的人类偏好，长时长场景的收益最明显。

## 3. 主要发现

### 3.1 算法定位

WanPE 是接在视频生成器前的 prompt enhancer。论文将输出条件定义为同时满足用户约束并贴近 video-grounded caption 分布的文本，因而增强器负责电影化规划，Wan3.0 负责视频渲染。

### 3.2 训练链路

1. 约 105 万真实视频片段经筛选和多模态 caption，得到层级化、带时间戳的电影化条件。
2. GPT-5.4 按视频内容类别构造请求：2,000 条人工请求先分为十个类别的请求池；对每个视频条件，从其类别池中抽取 5 条请求作为 few-shot demonstrations，再要求模型只依据该条件重建自然用户请求。由此形成 reverse SFT 对。
3. 约 1.5 万请求进入 SC-GRPO；Qwen3.7-Max 以九维语义一致性打分，组内优势和 SFT-reference KL 驱动增强器更新。

这个方向解决的是 prompt 分布与生成器训练条件之间的错位，以及多镜头展开时的语义漂移。

WanPE 的底座是 Qwen3.5 系列：4B、9B、35B-A3B 和 397B-A17B 分别初始化四个规模的增强器。Qwen3.7-Max 是训练阶段的文本 reward evaluator，不是 WanPE 的底座；视频生成仍由 Wan3.0 完成。

### 3.3 实验结果

WanPE-397B 在 Wan3.0 上的人类专家视频 preference score 相对原始请求提升：5 秒 +16.81、10 秒 +16.28、15 秒 +18.84、30 秒 +50.86。反向构造 SFT（49.86）超过 forward rewriting（39.49）和 forward-target SFT（35.17）。论文表 3 报告的语义一致性指标从 75.5 到 97.6，并将 Wan3.0 的人类视频偏好从 42.70 提高到 49.69；该文本指标由 Gemini-3.1-Pro-Preview 评测，训练阶段的 Qwen3.7-Max 只承担 reward evaluator 角色。

跨生成器时，经过 GPT-5.4 格式适配，WanPE-397B 在 LTX-2.5-Base 和 MiniMax-H3-Base 上分别为 35.56 和 41.09，超过原生 enhancer 的 21.11 和 35.92。这个实验验证的是格式适配后的迁移，不是把同一原始字符串直接喂给所有生成器。

### 3.4 官方网页部署实际行为

官方仓库是静态项目页。`index.html` 直接内嵌论文摘要、作者、方法/结果图和表格；`script.js` 为五个 demo 读取：

- `prompt_en.txt`：原始用户请求；
- `pe_output.txt`：预先生成的 WanPE 增强条件；
- `wan30_wo_pe.mp4`：无增强视频；
- `wan30_wanpe_397B.mp4`：使用 WanPE-397B 的视频。

视频标签初始没有 `src`，IntersectionObserver 在进入视口附近才写入 `data-src` 并播放；这改善了展示页加载，但不构成推理后端。页面的 Code 链接为空，仓库没有模型权重、服务端、API schema 或启动命令。

### 3.5 复现边界

目前可复现的是：下载论文、读取网页源代码和 prompt 文本、查看官方表格和前后对比视频。当前无法仅凭公开仓库复现：

- WanPE-4B/9B/35B/397B 的推理；
- video-grounded caption、reverse request 重建和 SC-GRPO 训练；
- Wan3.0 或跨生成器的在线生成；
- 论文中的 512-GPU 训练设置和 397B 规模资源需求。

因此“网页部署”应准确理解为研究展示页，而不是可交互的 WanPE 在线服务。

## 4. 结论

论文给出的训练链条是：真实视频产生视频条件，类别化 few-shot 反向构造用户请求，SFT 学习从请求生成视频条件，SC-GRPO 再约束跨 shot 的语义保持。2,000 条人工请求为类别内的语言和具体程度提供示例，每次调用使用 5 条对应类别示例。

WanPE 是基于 Qwen3.5 初始化并经过后训练的独立增强器。公开项目页只提供论文、静态演示文本和预生成视频；截至记录的官方仓库版本，没有权重、训练代码、推理 API、许可证或明确的后续开源计划。因此当前证据支持“Qwen3.5-based WanPE research model”，不支持“WanPE 已开源”或“已有开源时间表”。

## 5. 对仓库正式网页的落地

仓库的正式论文页为 [`docs/paper_reading/wanpe.md`](../../../docs/paper_reading/wanpe.md)，论文导航已加入 `WanPE`。该页面保留算法、公式、评测数字、网页 demo 行为和公开缺口，便于后续补充权重或代码。

## 6. 残余风险

- 论文结果是作者报告的人类盲评，本文没有重新生成视频或复算 Bradley–Terry；
- 官方仓库没有 LICENSE 文件，不能据此推断网页资产或模型权重的再分发许可；
- 项目页是新发布静态站，后续可能出现代码、权重或 API，当前结论需随官方仓库更新重新核验。

## 7. 附录：原文与官方仓库核查

### A. 原文定位

- 论文 §2.2：2K 人工请求池按十类内容分组；从对应池抽取 5 条 demonstrations；GPT-5.4 重建 $x_i=f_{LM}(y_i;\mathcal{E}_i)$。
- 论文 Appendix B：视频 caption 的层级结构、类别自适应 system prompt、结构/时间戳/视频一致性检查；失败 caption 会重新生成后再保留。
- 论文 §2.3：约 15K SC-GRPO prompts 由多样人工请求和 Wan3.0 试生成后人工筛出的困难请求组成；Qwen3.7-Max 产生文本 reward。
- 论文 §3：WanPE-4B/9B/35B/397B 分别初始化自 Qwen3.5-4B、9B、35B-A3B、397B-A17B；语义一致性结果另由 Gemini-3.1-Pro-Preview 评测。
- 官方项目页与仓库：核对静态 demo、模型/代码入口和公开发布信息；记录版本为 `64602b5b036fbd601e6d66d59c42a32d98867853`。

### B. 未披露项

论文没有给出 GPT-5.4 的完整 system prompt、temperature、上下文拼接格式、示例选择随机种子、失败重试策略或 agent/tool loop。因此报告只陈述“按类别抽 5 条 few-shot”，不把它扩写成自动迭代流程。

### C. 官方开源状态

官方仓库 `Wan-PE/Wan-PE.github.io` 的记录版本为 `64602b5b036fbd601e6d66d59c42a32d98867853`。该仓库是项目展示页，包含静态 HTML/JS、示例 prompt、增强文本和预渲染视频；未发现 WanPE 权重、训练/推理代码、API、LICENSE 或 release/roadmap 声明。该状态只能证明“截至该版本未公开”，不能推断作者未来不会开源。

## 8. 证据台账

见 [`sources.json`](sources.json)。正式网页页引用论文和官方项目页；二进制 demo 没有复制进本仓库。
