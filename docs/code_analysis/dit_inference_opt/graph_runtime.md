---
tags:
  - Diffusion Model
  - CUDA
  - LLM Inference
---
# DiT 推理优化：Graph Runtime（BCG）

返回：[专题总览](overview.md)

## 抓住重点

- 将单步计算分为静态段 \(G\) 与动态段 \(E\)。BCG 固化 \(G\) 的调度图，仍逐次执行 \(E\)，故计算值不变而 launch 数下降。
- 图只对已 warmup 的输入签名生效。把 prompt 长度映射到有限 bucket，才能提高真实请求的命中率。
- 它减少的是固定开销 \(\alpha\)，对长 Attention 主导的步骤收益有限；与 Cache-DiT、`torch.compile` 的组合在当前运行时被排除。

## 1. 机制

令 \(z\) 表示一次调用的张量形状、dtype、设备及离散控制量，\(\sigma(z)\) 是其签名。将 forward 分割为

\[
f_\theta(z)=E_0\circ G_1\circ E_1\circ\cdots\circ G_m\circ E_m(z),
\]

其中 \(G_j\) 的张量地址和形状在一次服务配置内固定，\(E_j\) 包含动态 Attention、变长处理或 collective。warmup 为每个允许的 \(\sigma\) 捕获 \(G_j\)；请求只在 \(\sigma(z)\) 已捕获时 replay，否则完整走 eager。这里的“breakable”正是保留 \(E_j\) 的动态图边界。

若文本长度为 \(n\)，bucket 集合为 \(\mathcal B\)，pad 后长度为 \(b(n)=\min\{b\in\mathcal B:b\ge n\}\)。它以额外 token 计算换取更多相同 \(\sigma\)；所有实际分辨率与 \(b(n)\) 必须在 warmup 集合中。

## 2. 启用条件

启用的前提是服务输入域 \(\mathcal Z_{\rm serve}\) 被已捕获域 \(\mathcal Z_{\rm warm}\) 覆盖，或能接受未覆盖部分回退：\(\mathcal Z_{\rm serve}\subseteq\mathcal Z_{\rm warm}\cup\mathcal Z_{\rm eager}\)。当前 pin 强制提供分辨率 warmup，文本 bucket 必须为正整数；仅已验证的模型族进入捕获路径。白名单是实现覆盖范围，不是算法限制。

## 3. 何时值得开

| 场景 | 建议 |
|------|------|
| 图像 DiT、小算子多、profile 显示 launch-bound | \(\alpha\) 占比高，优先验证 BCG |
| 视频重 Attention / 已接近打满 GPU | \(\beta\,\mathrm{FLOPs}\) 主导，优先 [Kernel](kernels_fusion.md) / Attention backend / [Parallel](parallelism.md) |
| 需要 Cache-DiT 或 `torch.compile` | 当前配置路径互斥，分别测量 |

官方 deployment cookbook：BCG 是 manual opt-in；每个服务分辨率必须出现在 warmup 列表。

## 4. 与 compile / Cache / Offload

若某优化在请求间改变权重地址、控制分支或 \(\sigma\)，它会破坏 replay 的前提。故当前 CLI 令 BCG 与 Cache-DiT、`torch.compile` 互斥；层间 offload 也应单独测量。现网官方文档还要求 request-gated 融合与 BCG 分开，因为 warmup 捕获的是一个固定质量分支；该规则在本 pin 尚非硬校验，详见 [Kernels](kernels_fusion.md#3-数值契约lossless-与-high)。

## 4.1 Signature 与回退

令 \(p=\Pr[\sigma(z)\in\mathcal Z_{\rm warm}]\)。若 eager 与 replay 时间分别为 \(\tau_e,\tau_g\)，稳态平均时间是 \(p\tau_g+(1-p)\tau_e\)，不是 \(\tau_g\)。因此报告必须给出 \(p\)、warmup 成本与 bucket padding 比例；只报告“已启用”无法说明服务实际减少了多少 launch。

H3 侧断点为何落在 Attention、以及 text-only bucketing：[效率附录](../minimax_h3/06_efficiency_appendix.md)、[DiT Runtime](../minimax_h3/07_dit_runtime_and_collectives.md)。

## 附录：实现证据

| 主题 | 路径 |
|------|------|
| Diffusion BCG runner | `.../breakable_cuda_graph/runner.py` |
| SRT 原语 | `srt/.../breakable_cuda_graph/breakable_cuda_graph.py` |
| 校验 / 白名单 | `server_args.py`：`_validate_*` / `_adjust_*` / `_is_breakable_cuda_graph_supported_model` |
| 模型 padder | `.../breakable_cuda_graph/model_padders/` |

## 相关阅读

- [专题总览](overview.md) · [Kernels & Fusion](kernels_fusion.md)（静段更适合进 graph） · [Feature Cache](feature_cache.md) · [Correctness](correctness.md)

pin：`multimodal_gen/runtime/breakable_cuda_graph/runner.py`（签名、捕获与回退），`server_args.py`（warmup、白名单、互斥），`model_padders/`（模型输入规整）。模型特定断点及 H3 bucket 见 [效率附录](../minimax_h3/06_efficiency_appendix.md)。
