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

- 端到端时间 \(\tau=\tau_{\rm enc}+T\tau_{\rm dit}+\tau_{\rm vae}\)。当优化降低 \(\tau_{\rm dit}\) 或 \(T\) 时，Encoder/VAE 的相对占比严格上升。
- Encoder 每请求通常一次；编码时 DiT rank 闲置，可将其用于权重切分或数据并行。
- Encoder/VAE 天然适合 **组件级 offload**：头尾 onload，中间把显存让给 DiT → [Memory Offload](memory_offload.md)。
- `fold` / `replicate` 相对单卡编码是 bitwise-identical（官方声明）；`dp` 的目标是吞吐，质量与确定性须按实际 batch 切分验收。

## 1. Encoder Parallel

设 \(C_e\) 是单卡 encoder 计算时间、\(n\) 是并行度。fold 将权重切分，理想时间约为 \(C_e/n+A_e(n)\)，其中 \(A_e(n)\) 是 collective 与同步时间；dp 将 batch \(B\) 切成约 \(B/n\)；replicate 令每张卡保存完整权重并保持单样本计算路径。三者分别优化单请求宽模型、批吞吐和确定性路径。

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

设输出 latent 含 \(S=H W\) 个空间 token，VAE decoder 的局部卷积/上采样成本通常近似 \(O(S)\)，峰值激活近似 \(O(SC)\)，其中 \(C\) 为通道数。它不同于 DiT Attention 的 \(O(S^2)\) 项：分辨率提升后仍会成为稳定的端到端尾部，但不会以同一二次律增长。

可将空间域划分为 \(n\) 个 tile。每个 tile 需要 halo 宽度 \(r\) 才能保持边界卷积等价，因此有效工作量近似 \(n(S/n+O(r\sqrt{S/n}))\)。tile 越细，单卡峰值越低，halo 交换和边界冗余越高。由此选择 height sharding、tiled decode 或并行 decode 的依据是 \(S,C,r\) 与链路带宽，而不是仅按模型名选开关。

当前资料中的工程入口包括：

- Height sharding / halo exchange  
- Parallel tiled decode  
- VAE fusion（减中间写）  

Cache-DiT 并行 YAML 也可把 `vae` / `text_encoder` 列入 `extra_parallel_modules`（`cache_dit.mdx` TE-P / VAE-P 节）——那是 Cache-DiT 自有并行扩展，与 SGLang `--encoder-parallel` 不同入口。

拆分部署（Encoder–Denoiser–Decoder）见 `disaggregation.mdx`，索引于 [Correctness](correctness.md#3-serving-相关能力索引)。

## 3. 与 Offload / Progressive / Parallel

- Encoder/VAE 只在请求头尾活跃，适合组件级驻留调度；显存约束与搬运隐藏条件见 [Offload](memory_offload.md)。
- Progressive 改的是 DiT 的 latent 分辨率日程，最终 VAE 仍需在目标分辨率解码，故不替代 VAE 优化，见 [Progressive](progressive_resolution.md)。
- DiT 的 SP/TP 通信模型见 [Parallelism](parallelism.md)；Encoder parallel 复用闲置 rank，但不改变该模型。

## 3.1 为什么 DiT 加速后要重算占比

令 \(r=(\tau_{\rm enc}+\tau_{\rm vae})/\tau\)。若 DiT 被加速为原来的 \(\gamma\in(0,1)\)，则
\[
r'=\frac{\tau_{\rm enc}+\tau_{\rm vae}}{\tau_{\rm enc}+\gamma T\tau_{\rm dit}+\tau_{\rm vae}}>r.
\]
这解释了为什么少步蒸馏、cache 或 fusion 后必须重新 profile Encoder/VAE。

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
