# WanPE 网页部署与方法研究

## 1. 问题与范围

本报告核对 WanPE（arXiv:2609.30221）的算法、实验和官方网页部署形态，重点回答：网页是否提供可调用的 prompt enhancement 服务，公开材料能否复现训练或推理，以及论文结果如何支撑其电影化规划主张。

范围包括论文 PDF、官方项目页和官方 GitHub 仓库。没有下载或提交官方视频二进制，没有修改产品代码、日报配置或正式网页之外的站点。

## 2. 证据方法

- 论文：下载并解析 arXiv v1 PDF，核对方法章节、表 1–3、附录数据构造和结论。
- 网页：读取官方仓库 `index.html`、`script.js`、demo prompt 与 `pe_output.txt`，检查链接、资源路径、加载逻辑和结果表。
- 仓库：记录官方 main tree SHA `64602b5b036fbd601e6d66d59c42a32d98867853`。

## 3. 主要发现

### 3.1 算法定位

WanPE 是接在视频生成器前的 prompt enhancer。论文将输出条件定义为同时满足用户约束并贴近 video-grounded caption 分布的文本，因而增强器负责电影化规划，Wan3.0 负责视频渲染。

### 3.2 训练链路

1. 约 105 万真实视频片段经筛选和多模态 caption，得到层级化、带时间戳的电影化条件。
2. GPT-5.4 用约 2,000 条人工请求示例从条件反向重建自然用户请求，形成 reverse SFT 对。
3. 约 1.5 万请求进入 SC-GRPO；Qwen3.7-Max 以九维语义一致性打分，组内优势和 SFT-reference KL 驱动增强器更新。

这个方向解决的是 prompt 分布与生成器训练条件之间的错位，以及多镜头展开时的语义漂移。

### 3.3 实验结果

WanPE-397B 在 Wan3.0 上的 preference score 相对原始请求提升：5 秒 +16.81、10 秒 +16.28、15 秒 +18.84、30 秒 +50.86。反向构造 SFT（49.86）超过 forward rewriting（39.49）和 forward-target SFT（35.17）。SC-GRPO 使 397B 文本一致性从 75.5 到 97.6，并将 Wan3.0 偏好从 42.70 提高到 49.69。

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

## 4. 对仓库正式网页的落地

正式网页页为 [`docs/paper_reading/wanpe.md`](../../../docs/paper_reading/wanpe.md)，并已在 `mkdocs.yml` 的论文阅读导航中加入 `WanPE`。正文将算法、公式、评测数字、网页脚本行为、公开缺口和与 VPO/PhyPrompt/PromptEnhancer/APE 的关系分开描述，便于后续补充权重或代码。

## 5. 残余风险

- 论文结果是作者报告的人类盲评，本文没有重新生成视频或复算 Bradley–Terry；
- 官方仓库没有 LICENSE 文件，不能据此推断网页资产或模型权重的再分发许可；
- 项目页是新发布静态站，后续可能出现代码、权重或 API，当前结论需随官方仓库更新重新核验。

## 6. 证据台账

见 [`sources.json`](sources.json)。正式网页页引用论文和官方项目页；二进制 demo 没有复制进本仓库。
