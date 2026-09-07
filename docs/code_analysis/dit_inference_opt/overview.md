---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
  - CUDA
---
# DiT 推理优化：专题总览

**证据包**: `refs/dit-inference-opt/`  
**主实现 pin**:

- SGLang：`refs/codes/sglang` @ `db75dfe10ff7ef1f735d79178d2f2090683e3eb0`
- Cache-DiT：`refs/codes/cache-dit` @ `3db8d1e70fe4a898c85efa5fe0576d85c8396db4`

**样板切片（模型专论，专题内只交叉引用）**: [MiniMax H3 × SGLang 效率主线](../minimax_h3/05_efficiency_in_sglang.md)

本专题按工程轴展开 DiT 推理优化，不以「serving」命名。稀疏 Attention（Sage / SVG2 / VSA / SLA 等）已在本仓论文与代码分析线覆盖，此处只做索引，不新开正文。

## 1. 瓶颈怎么拆

一次请求大致经过：Text/Image Encoder → 多步 DiT denoise → VAE Decode。DiT 步数 \(N_{\mathrm{step}}\) 与单步 FLOPs 决定主成本；Encoder/VAE 在少步模型或高并发下占比会明显上升。

SGLang 官方把杠杆分成两类（见 pin 内 `docs/docs/sglang-diffusion/performance-optimization.mdx`）：

- **output-preserving**：改 residency、并行、kernel、调度，目标是保持模型行为
- **quality-tradeoff**：Cache、progressive resolution、量化等，可能改变去噪路径或数值表示

建议顺序：定模型/分辨率/帧数/步数基线 → `--performance-mode` 与显存策略 → 并行与 attention backend → profile → 再加 cache / progressive / quant。

## 2. 专题文档地图

| 文档 | 轴 | 深挖焦点 |
|------|----|----------|
| [Memory Offload](memory_offload.md) | 显存 / 常驻 | Layerwise、组件策略、与 Cache/FSDP 互斥 |
| [Kernels & Fusion](kernels_fusion.md) | 算子 | Diffusion kernel 注册、融合、中间 Tensor 消除 |
| [Graph Runtime](graph_runtime.md) | 启动开销 | Breakable CUDA Graph、warmup 签名、与 compile |
| [Feature Cache](feature_cache.md) | 步间冗余 | DBCache / TaylorSeer / SCM、TeaCache、Spectrum |
| [Parallelism](parallelism.md) | 多卡 | CFG × TP × Ulysses × Ring / KV-Gather |
| [Quantization](quantization.md) | 低精度 | ModelOpt / Online FP8 / Nunchaku、融合保收益 |
| [Encoder & VAE](encoder_vae.md) | 非 DiT 段 | Encoder parallel、VAE shard / tiled decode |
| [Progressive Resolution](progressive_resolution.md) | 轨迹分辨率 | DCT Rewind、谱上采样 |
| [Correctness](correctness.md) | 可复现 / 上线 | 精度档、组合约束、batch 与拆分部署 |

## 3. 组合约束（首轮核验）

以当前 SGLang pin 的 `server_args` 校验为准：

- **Cache-DiT × DiT layerwise offload**：互斥（cache 可能复用已被 release 的 block 权重）
- **Cache-DiT × FSDP inference**：互斥或自动关闭 FSDP
- **TeaCache × Spectrum**：互斥
- **BCG**：需 `--warmup-resolutions`；仅白名单模型/pipeline 真正启用
- **Progressive + 整模 DiT CPU offload**：文档提示会稀释加速（每步固定 PCIe 成本）

更细互斥与推荐组合见各分篇。

## 4. 与稀疏 Attention 的边界

兼容矩阵中出现的 SageAttention、SparseVideoGen2、VSA、SLA 等 → 指向既有笔记（如 [VSA](../../paper_reading/vsa.md)、[SVG2](../../paper_reading/svg2.md)、[FastVideo VSA](../fastvideo_vsa/00_overview.md)）。本专题讨论并行时只说明稀疏 backend 与 Ring/USP 的已知限制入口，不重复算法正文。

## 5. 阅读路径

1. 先读本页与 [Memory Offload](memory_offload.md)  
2. 需要压延迟：Kernel → Graph → Parallelism  
3. 可接受画质变化：Feature Cache → Progressive → Quantization  
4. DiT 已快后的长尾：Encoder & VAE → Correctness  

持续推进时，每篇以「机制 → 源码锚点 → 组合约束 → 缺口」为固定结构；缺口处欢迎在网页端标注后加厚。
