---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
---
# DiT 推理优化：Encoder & VAE

**文档**: `encoder_parallel.mdx`；性能总览把 `--encoder-parallel` 列为 output-preserving 杠杆。

返回：[专题总览](overview.md)

## 抓住重点

- DiT 变快之后，Text/Image Encoder 与 VAE Decode 的占比上升；少步蒸馏或高并发时更明显。
- Encoder 每请求通常一次，不像 DiT 重复 \(N_{\mathrm{step}}\) 次，但编码时 **整份 DiT 副本可能闲置** → `--encoder-parallel` 把闲置 GPU 用起来。
- Encoder/VAE 天然适合 **组件级 offload**：头尾 onload，中间把显存让给 DiT → [Memory Offload](memory_offload.md)。
- `fold` / `replicate` 相对单卡编码是 bitwise-identical（官方声明）；`dp` 用于吞吐，可能引入帧差。

## 1. Encoder Parallel

文档定位：编码阶段占请求时间可见份额、且 DiT 副本在编码时闲置时打开。

```text
--encoder-parallel {auto,fold,dp,replicate}
```

| 模式 | 行为 | 何时用 |
|------|------|--------|
| `auto` | 按 encoder 宽度与 batch 宽度为每个 encoder 选 fold/dp/replicate | 默认 |
| `fold` | 把 encoder 权重 TP-shard 到闲置的 DiT 副本 ranks | 单请求、宽 encoder（文档门控 hidden ≥ 4096） |
| `dp` | 按 batch 做数据并行编码 | 多请求/大 batch 吞吐 |
| `replicate` | 每卡完整 encoder 副本 | 需要与单卡 bit-exact，或 fold 不适用 |

`fold` 与 `dp` **按 encoder 互斥**：folding 在 load 时切权重，折叠后的 encoder 不能再走 dp。`server_args.adjust_pipeline_config` 会根据 dp/tp/sp/disagg 决定 fold world。

## 2. VAE

高分辨率图像与视频 Decode 的显存与时延问题，常见工程手段（社区/实现侧；本篇以「组件 offload + 并行索引」为主）：

- Height sharding / halo exchange  
- Parallel tiled decode  
- VAE fusion（减中间写）  

Cache-DiT 并行 YAML 也可把 `vae` / `text_encoder` 列入 `extra_parallel_modules`（`cache_dit.mdx` TE-P / VAE-P 节）——那是 Cache-DiT 自有并行扩展，与 SGLang `--encoder-parallel` 不同入口。

拆分部署（Encoder–Denoiser–Decoder）见 `disaggregation.mdx`，索引于 [Correctness](correctness.md#3-serving-相关能力索引)。

## 3. 与 Offload / Progressive / Parallel

- 组件级 CPU offload：`text_encoder_cpu_offload` / `image_encoder_cpu_offload` / `vae_cpu_offload`；也可进 `--layerwise-offload-components` 的 default 组 → [Offload](memory_offload.md)。  
- Progressive 改的是 DiT latent 分辨率日程，不替代 VAE 优化 → [Progressive](progressive_resolution.md)。  
- DiT 侧 SP/TP：[Parallelism](parallelism.md)。

## 4. 源码 / 文档锚点

| 主题 | 路径 |
|------|------|
| Encoder 并行官方说明 | `docs/.../encoder_parallel.mdx` |
| CLI | `--encoder-parallel`（`server_args.py`） |
| 组件驻留 | `component_manager.py` / `component_resident_strategies.py` |
| Disaggregation | `docs/.../disaggregation.mdx` |

## 相关阅读

- [专题总览](overview.md) · [Memory Offload](memory_offload.md) · [Parallelism](parallelism.md) · [Correctness](correctness.md)

## 附录：仍可加深

- Encoder folding 与 SP group 绑定的代码路径 walkthrough  
- VAE parallel 各策略的通信图与 CLI 开关表  
- Mooncake RDMA 下 Encoder/Denoiser/Decoder 角色与失败模式
