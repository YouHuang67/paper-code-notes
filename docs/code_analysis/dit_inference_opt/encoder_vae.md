---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
---
# DiT 推理优化：Encoder & VAE

**文档**: `encoder_parallel.mdx`；性能总览把 `--encoder-parallel` 列为 output-preserving 杠杆。

DiT 变快之后，Text/Image Encoder 与 VAE Decode 的占比上升。Encoder 通常每请求一次，不像 DiT 重复 \(N_{\mathrm{step}}\) 次，但在少步蒸馏模型或高并发 serving 下不可再当「无所谓的前处理」。

## 1. Encoder Parallel

文档定位：当编码阶段占请求时间可见份额、且 DiT 副本在编码时闲置，打开 `--encoder-parallel`。

SGLang 侧能力（与 BBuf 叙述对应，细节待对照源码加深）：

- Encoder Parallel Folding：复用 SP Group 给 T5 等  
- Encoder DP / Replicate 模式  

## 2. VAE

高分辨率图像与视频 Decode 的显存与时延问题，常见工程手段：

- Height Sharding  
- Halo Exchange  
- Parallel Tiled Decode  
- VAE Fusion  

（具体类名与开关以 pin 内 VAE 并行实现与 CLI 为准，本轮标为加深项。）

## 3. 与 Offload

Encoder/VAE 适合作为 **组件级** offload 对象：请求头尾 onload，中间把显存让给 DiT。见 [Memory Offload](memory_offload.md)。

## 4. 本轮缺口

- Encoder folding 与 SP group 绑定的代码路径  
- VAE parallel 各策略的通信图  
- Mooncake RDMA 拆分部署中 Encoder/Denoiser/Decoder 角色（见 [Correctness](correctness.md)）
