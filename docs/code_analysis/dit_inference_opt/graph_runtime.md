---
tags:
  - Diffusion Model
  - CUDA
  - LLM Inference
---
# DiT 推理优化：Graph Runtime（BCG）

**源码**:

- Diffusion runner：`multimodal_gen/runtime/breakable_cuda_graph/runner.py`
- 底层原语：`srt/.../breakable_cuda_graph/`
- CLI：`--enable-breakable-cuda-graph`、`--warmup-resolutions`、`--bcg-text-buckets`

返回：[专题总览](overview.md)

## 抓住重点

- BCG = Breakable CUDA Graph：把 DiT forward 里 **形状稳定** 的段捕获成可回放 graph；动态 Attention、变长、通信仍 Eager。
- 目标是压 **launch 开销**、提高 GPU busy，不改算法（output-preserving 风格）。
- **必须** `--warmup-resolutions`；仅白名单模型真正启用；与 **torch.compile / Cache-DiT 互斥**（BCG 优先）。
- 适合 launch-bound 的小算子多的图像 DiT；视频重 Attention 时收益常接近零。

## 1. 机制

`runner.py` 模块文档约定：

1. Runner 包装 `nn.Module`，属性透传。  
2. **Capture 必须显式**（warmup 驱动）；serving 路径不触发新 capture。  
3. 按 kwargs 张量 shape/dtype 等构造 signature；命中则 replay，否则 eager。  
4. 在 Attention / SP all-to-all / 动态段处「打断」，前后静段各自成 graph。

Diffusion 侧额外能力：静态 buffer、prompt-bucket padding、模型专用 padder（`model_padders/`：Ideogram / H3 / Qwen-Image / Z-Image 等）。`--bcg-text-buckets`：prompt 序列 pad 到最近 bucket，使不同长度复用同一张图；warmup 为每个 bucket capture 一次。

## 2. 启用条件

`_validate_breakable_cuda_graph`：

- 打开 BCG 时 **必须** 提供 `--warmup-resolutions`（每个服务分辨率单独 capture）。  
- `--bcg-text-buckets` 若给出，需至少一个正整数。

`_adjust_breakable_cuda_graph_support`：不在白名单则 **自动关掉** 并打 warning。当前 pin 日志列出的支持面：

- Ideogram-4  
- Lightricks/LTX-2  
- MiniMax-H3  
- Qwen/Qwen-Image、Qwen/Qwen-Image-2512  
- SANA1.5  
- Tongyi-MAI/Z-Image / Z-Image-Turbo  
- zai-org/GLM-Image  

（以 `BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS` 常量为准。）

BCG 还会推动 server warmup（见 `test_server_args.py`）。

## 3. 何时值得开

| 场景 | 建议 |
|------|------|
| 图像 DiT、小算子多、Nsight 显示 launch-bound | 值得试 BCG |
| 视频重 Attention / 已接近打满 GPU | 优先 [Kernel](kernels_fusion.md) / Attention backend / [Parallel](parallelism.md) |
| 需要 Cache-DiT 或 torch.compile | 先选一条路径对比，勿默认叠开 |

官方 deployment cookbook：BCG 是 manual opt-in；每个服务分辨率必须出现在 warmup 列表。

## 4. 与 compile / Cache / Offload

CLI help（`--enable-breakable-cuda-graph`）原文要点：

- 在 Attention 处切开；SP all-to-all / 动态 Attention 保持 eager。  
- **Mutually exclusive with `--enable-torch-compile` and Cache-DiT（BCG takes priority）**。  
- Requires `--warmup-resolutions`；warmup 时全部 capture。

现网 [Fused Kernels](https://docs.sglang.io/docs/sglang-diffusion/fused_kernels) 另写：request-gated DiT 融合（`quality=extra-high/high`）不要与 BCG 同开——warmup 捕获的是 lossless 分支。本 pin 的 `server_args` 尚未搜到这条硬拒绝。质量挂载机制见 [Kernels & Fusion §3](kernels_fusion.md#3-数值契约lossless-与-high)。

Layerwise offload 改变权重指针与异步拷贝，与 graph 捕获假设冲突风险高 → 实务上避免与 [Offload](memory_offload.md) 的 DiT layerwise 同开。

H3 侧断点为何落在 Attention、以及 text-only bucketing：[效率附录](../minimax_h3/06_efficiency_appendix.md)、[DiT Runtime](../minimax_h3/07_dit_runtime_and_collectives.md)。

## 5. 源码锚点

| 主题 | 路径 |
|------|------|
| Diffusion BCG runner | `.../breakable_cuda_graph/runner.py` |
| SRT 原语 | `srt/.../breakable_cuda_graph/breakable_cuda_graph.py` |
| 校验 / 白名单 | `server_args.py`：`_validate_*` / `_adjust_*` / `_is_breakable_cuda_graph_supported_model` |
| 模型 padder | `.../breakable_cuda_graph/model_padders/` |

## 相关阅读

- [专题总览](overview.md) · [Kernels & Fusion](kernels_fusion.md)（静段更适合进 graph） · [Feature Cache](feature_cache.md) · [Correctness](correctness.md)

## 附录：仍可加深

- 通用 DiT 上 BCG segment 自动划分规则（非 H3）  
- signature miss 时 Diffusers fallback 对评测的污染路径  
- 多分辨率 serving 下 buffer 上限（`SGLANG_DIFFUSION_IPC_A2A_MAX_BUFFERS` 等相邻约束）
