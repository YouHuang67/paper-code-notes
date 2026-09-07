---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
  - CUDA
---
# DiT 推理优化：专题总览

**证据包**: [`refs/dit-inference-opt/`](../../../refs/dit-inference-opt/README.md)  
**主实现 pin**:

- SGLang：`refs/codes/sglang` @ `db75dfe10ff7ef1f735d79178d2f2090683e3eb0`
- Cache-DiT：`refs/codes/cache-dit` @ `3db8d1e70fe4a898c85efa5fe0576d85c8396db4`

**样板切片（模型专论，专题内只交叉引用）**: [MiniMax H3 × SGLang 效率主线](../minimax_h3/05_efficiency_in_sglang.md)

## 抓住重点

- 一次请求 ≈ Encoder → \(N_{\mathrm{step}}\) 次 DiT → VAE；主成本在 DiT 步数 × 单步 FLOPs，DiT 变快后 Encoder/VAE 占比会抬头。
- 杠杆分两类：**output-preserving**（驻留 / 并行 / kernel / 图）与 **quality-tradeoff**（cache / progressive / quant）。先定基线与显存策略，再开质量换速度。
- 组合有硬互斥：最常见是 [Cache-DiT ⊥ DiT layerwise](memory_offload.md#4-组合约束必须先读)、[TeaCache ⊥ Spectrum](feature_cache.md#3-组合约束)、[BCG ⊥ torch.compile / Cache-DiT](graph_runtime.md#4-与-compile--cache--offload)、[Progressive ⊥ SP / torch.compile](progressive_resolution.md#3-组合约束)。全表见 [Correctness](correctness.md#2-已核验的互斥--自动降级)。
- 稀疏 Attention（Sage / SVG2 / VSA / SLA）本专题只索引，算法正文在论文与既有代码分析线。

## 1. 瓶颈怎么拆

SGLang 官方决策框架见 pin 内 `docs/docs/sglang-diffusion/performance-optimization.mdx`：

| 类 | 含义 | 本专题入口 |
|----|------|------------|
| output-preserving | 改 residency、并行、kernel、调度；故意不改去噪语义 | [Offload](memory_offload.md) · [Kernel](kernels_fusion.md) · [Graph](graph_runtime.md) · [Parallel](parallelism.md) · [Encoder/VAE](encoder_vae.md) |
| quality-tradeoff | 改路径 / 数值 / 分辨率日程，需质量验收 | [Feature Cache](feature_cache.md) · [Progressive](progressive_resolution.md) · [Quant](quantization.md) |

建议顺序：定模型/分辨率/帧数/步数基线 → `--performance-mode` 与显存策略 → 并行与 attention backend → profile → 再加 cache / progressive / quant。收束验收见 [Correctness](correctness.md#4-建议验收清单)。

## 2. 专题文档地图

| 文档 | 轴 | 深挖焦点 | 优先读 |
|------|----|----------|--------|
| [Memory Offload](memory_offload.md) | 显存 / 常驻 | 组件级 vs layerwise；SGLang manager vs Cache-DiT bucket | ★★★ |
| [Kernels & Fusion](kernels_fusion.md) | 算子 | AdaLN/QK-RoPE/残差门控公式、quality 挂载、删中间 Tensor | ★★★ |
| [Graph Runtime](graph_runtime.md) | 启动开销 | BCG 捕获/回放、warmup 签名、白名单 | ★★ |
| [Feature Cache](feature_cache.md) | 步间冗余 | DBCache Fn/Bn、TaylorSeer、SCM、TeaCache、Spectrum | ★★★ |
| [Parallelism](parallelism.md) | 多卡 | CFG × TP × Ulysses × Ring / KV-Gather | ★★ |
| [Quantization](quantization.md) | 低精度 | Online FP8/MXFP4、ModelOpt NVFP4、Nunchaku | ★★ |
| [Encoder & VAE](encoder_vae.md) | 非 DiT 段 | fold/dp/replicate；VAE 与组件 offload | ★ |
| [Progressive Resolution](progressive_resolution.md) | 轨迹分辨率 | dct_rewind、δ、与 SP/compile 互斥 | ★ |
| [Correctness](correctness.md) | 可复现 / 上线 | 互斥总表、验收清单、serving 索引 | ★★★ |

## 3. 组合约束速查（点进分篇看原因）

以当前 SGLang pin 的 `server_args` / `sampling_params` 校验为准：

| 组合 | 行为 | 详见 |
|------|------|------|
| Cache-DiT + DiT layerwise | **硬错误**（reuse 已 release 权重 → shape mismatch） | [Offload §4](memory_offload.md#4-组合约束必须先读) |
| Cache-DiT + FSDP | 显式开则报错，否则自动关 FSDP | 同上 |
| DiT layerwise + FSDP | 自动关 FSDP | 同上 |
| TeaCache + Spectrum | **硬错误** | [Cache §3](feature_cache.md#3-组合约束) |
| BCG 无 warmup resolutions | **硬错误** | [Graph §2](graph_runtime.md#2-启用条件) |
| BCG 非白名单模型 | 自动关掉 BCG | 同上 |
| BCG + torch.compile / Cache-DiT | CLI 声明互斥（BCG 优先） | [Graph §4](graph_runtime.md#4-与-compile--cache--offload) |
| Progressive + SP | RuntimeError | [Progressive §3](progressive_resolution.md#3-组合约束) |
| Progressive + torch.compile | 不兼容 | 同上 |
| Progressive + 整模 DiT CPU offload | 文档建议关 offload，否则冲淡加速 | [Progressive §2](progressive_resolution.md#2-收益与条件) |

## 4. 与稀疏 Attention 的边界

兼容矩阵中出现的 SageAttention、SparseVideoGen2、VSA、SLA 等 → [VSA](../../paper_reading/vsa.md)、[SVG2](../../paper_reading/svg2.md)、[FastVideo VSA](../fastvideo_vsa/00_overview.md)。并行篇只说明稀疏 backend 与 Ring/USP 的入口限制，不重复算法正文。

## 5. 阅读路径

1. 本页 → [Memory Offload](memory_offload.md)（显存先站住）  
2. 压延迟：[Kernels](kernels_fusion.md) → [Graph](graph_runtime.md) → [Parallelism](parallelism.md)  
3. 可接受画质变化：[Feature Cache](feature_cache.md) → [Progressive](progressive_resolution.md) → [Quantization](quantization.md)  
4. DiT 已快后的长尾：[Encoder & VAE](encoder_vae.md) → [Correctness](correctness.md)  

每篇结构固定为：**抓住重点 → 机制 → 源码锚点 → 组合约束 → 相关阅读**；更深实现细节放附录。
