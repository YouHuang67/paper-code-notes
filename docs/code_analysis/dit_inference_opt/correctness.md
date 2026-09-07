---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
---
# DiT 推理优化：Correctness 与部署约束

返回：[专题总览](overview.md)

优化数字只有在「硬件、模型、shape、精度、denoise、e2e、图像指标」说清楚时才有意义。本篇收束 **组合约束总表** 与验收习惯；不以 serving 命名专题，但索引部署相关能力。

## 抓住重点

- 官方说 **output-preserving**，不承诺 bit-exact：换 kernel / GPU / 精度路径仍可有微小数值差。决策边界是优化是否 **故意** 用质量换速度。
- 质量敏感路径：[Cache](feature_cache.md)、[Progressive](progressive_resolution.md)、[Quant](quantization.md)、近似 Attention backend。
- 评测若日志出现 Diffusers fallback，不能用来证明 Native Backend 速度。
- 下表是本专题可点击的互斥导航；原因与开关细节在分篇。

## 1. 输出等价边界

| 说法 | 含义 |
|------|------|
| output-preserving | 不故意改去噪语义；仍可能有实现级数值差 |
| quality-tradeoff | 故意改路径 / 数值 / 分辨率日程，必须过质量门 |

官方原文：pin 内 `performance-optimization.mdx`。H3 侧 Fast Path / admission 状态机见 [Denoise Loop](../minimax_h3/08_denoise_loop_state_machine.md)。

## 2. 已核验的互斥 / 自动降级

| 约束 | 行为 | 证据入口 | 分篇 |
|------|------|----------|------|
| Cache-DiT ⊥ DiT layerwise | ValueError（reuse released weights） | `server_args.py` layerwise conflicts | [Offload](memory_offload.md#4-组合约束必须先读) |
| Cache-DiT ⊥ FSDP | 显式开报错，否则自动关 FSDP | 同上 | [Offload](memory_offload.md) |
| DiT layerwise ⊥ FSDP | 自动关 FSDP | 同上 | [Offload](memory_offload.md) |
| TeaCache ⊥ Spectrum | ValueError | `sampling_params.py` | [Feature Cache](feature_cache.md#3-组合约束) |
| BCG 需 warmup resolutions | ValueError | `_validate_breakable_cuda_graph` | [Graph](graph_runtime.md#2-启用条件) |
| BCG 仅白名单模型 | 自动 disable + warning | `_adjust_breakable_cuda_graph_support` | [Graph](graph_runtime.md) |
| BCG ⊥ torch.compile / Cache-DiT | CLI help：互斥，BCG 优先 | `--enable-breakable-cuda-graph` help | [Graph](graph_runtime.md#4-与-compile--cache--offload) |
| BCG × request-gated DiT 融合 | 现网文档禁止（warmup 捕 lossless）；pin 未见硬拒绝 | crawl `fused_kernels` | [Kernels §9](kernels_fusion.md#9-与-graph--cache--quant--parallel) |
| Progressive ⊥ Ulysses/Ring SP | RuntimeError | `progressive_resolution.mdx` Limitations | [Progressive](progressive_resolution.md#3-组合约束) |
| Progressive ⊥ torch.compile | 文档声明不兼容 | 同上 | [Progressive](progressive_resolution.md) |
| Progressive + 整模 DiT CPU offload | 建议关 offload | 同上 Tip | [Progressive](progressive_resolution.md#2-收益与条件) |
| 量化适配器 × offload | 可能自动禁用不兼容模式 | loader adapters | [Quant](quantization.md) · [Offload](memory_offload.md) |
| KV-Gather ⊥ Ulysses/Ring 同槽 | 争 SP 槽位 | `server_args` / `parallelism.mdx` | [Parallelism](parallelism.md) |

总览速查表：[overview §3](overview.md#3-组合约束速查点进分篇看原因)。

## 3. Serving 相关能力（索引）

文档与代码中已存在、细节在分篇或待加厚：

| 能力 | 作用 | 入口 |
|------|------|------|
| Dynamic batching | 兼容 shape 的并发请求 | `dynamic_batching.mdx` |
| DP replica | 多副本吞吐 | [Parallelism](parallelism.md) `--dp-size` |
| Disaggregation / Mooncake | Encoder–Denoiser–Decoder 拆分传数据 | `disaggregation.mdx` |
| BCG warmup / text bucket | capture 成功 ≠ 真实请求必 replay | [Graph](graph_runtime.md) |
| Encoder parallel | 编码阶段吃闲置 DiT 副本 | [Encoder & VAE](encoder_vae.md) |

H3：[Denoise Loop 状态机](../minimax_h3/08_denoise_loop_state_machine.md)。

## 4. 建议验收清单

1. 固定模型、分辨率、帧数、步数、GPU、精度、随机种子协议。  
2. 先跑 output-preserving 基线（含 profile）：[Offload](memory_offload.md) / [Kernel](kernels_fusion.md) / [Graph](graph_runtime.md) / [Parallel](parallelism.md)。  
3. 再开 **单一** quality-tradeoff，看图像/视频指标与 e2e。  
4. 把互斥与自动降级写进报告（日志里「automatically disabling …」也算生效结果）。  
5. 排除 Diffusers fallback 污染后再比 Native 速度。

建议把结果拆成三层记录：

1. **路径层**：实际启用的 backend、是否发生自动 disable、BCG capture/replay 命中率。
2. **性能层**：Encoder、每步 DiT、VAE 的耗时，GPU 峰值显存与 H2D/D2H 流量。
3. **质量层**：固定 seed 下的像素/latent 差异，以及适合图像或视频的感知指标。

这样可以区分“配置未生效”“配置生效但通信/搬运占主导”和“速度提升伴随质量变化”三种结果。

## 相关阅读

- [专题总览](overview.md)（阅读路径与杠杆分类）  
- 各轴分篇见总览地图；约束原因以分篇「组合约束」节为准。

## 附录：仍可加深

- Fast Path `lossless` / `extra-high` / `high` 与 quality gate 源码表  
- Mooncake / disaggregation 角色与失败模式  
- 多步 BF16 rounding 误差累积的回归用例索引
