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

Breakable CUDA Graph（BCG）把 DiT forward 里 **形状稳定** 的段（Projection / Norm / RoPE / Residual / MLP 等）捕获成可回放的 graph segment；动态 Attention、变长处理、通信继续走 Eager。目标是压 launch 开销，提高 GPU busy，而不是改算法。

## 1. 机制

`runner.py` 文档约定：

- runner 包装 `nn.Module`，属性透传  
- **capture 必须显式**（warmup 驱动），serving 路径不触发新 capture  
- 按 kwargs 张量 shape/dtype 等构造 signature；命中则 replay，否则 eager  

Diffusion 侧额外拥有：静态 buffer、prompt-bucket padding、模型专用 padder（`model_padders/` 下 Ideogram / H3 / Qwen-Image / Z-Image 等）。

## 2. 启用条件

`server_args._validate_breakable_cuda_graph`：

- 打开 BCG 时 **必须** 提供 `--warmup-resolutions`（每个服务分辨率单独 capture）  
- `--bcg-text-buckets` 若给出，需至少一个正整数 bucket  

`_adjust_breakable_cuda_graph_support`：不在白名单的 pipeline/model 会 **自动关掉** BCG（日志警告）。白名单包括 Ideogram-4、LTX-2、MiniMax-H3、Qwen-Image、SANA1.5、Z-Image、GLM-Image 等（以 pin 内常量为准）。

BCG 还会推动 server warmup（见 `test_server_args.py`）。

## 3. 何时值得开

适合：**launch-bound**、小算子多、GPU busy 偏低的图像 DiT。  
不适合：已经接近打满 GPU 的重 Attention/视频形状——再补 graph 收益接近零，应回去做 GEMM/Attention/Fusion。

官方 deployment cookbook：BCG 是 manual opt-in；每个服务分辨率必须出现在 warmup 列表。

## 4. 与 compile / Cache / Offload

- `--enable-torch-compile` 是另一条图优化路径（`auto_tune` 可在 speed 模式默认打开）；与 BCG 的互斥细节以运行时与模型 padder 为准，实务上通常二选一做对比  
- Cache-DiT 改变执行路径（跳块），与静态 graph 回放难以共存——组合前需查当前校验与实践（H3/社区经验：勿与 Cache 同开）  
- Layerwise offload 改变权重指针与异步拷贝，与 graph 捕获假设冲突风险高  

H3 侧 BCG 断点为何落在 Attention、以及 text-only bucketing：见 [效率附录](../minimax_h3/06_efficiency_appendix.md)、[DiT Runtime](../minimax_h3/07_dit_runtime_and_collectives.md)。

## 5. 本轮缺口

- BCG segment 边界在通用 DiT 上的自动划分规则（非 H3）  
- `torch.compile` regional 路径与 BCG 的正式互斥表  
- serving signature miss 时 Diffusers fallback 对评测的污染
