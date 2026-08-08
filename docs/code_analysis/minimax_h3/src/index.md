---
tags:
  - Video Generation
  - Diffusion Model
  - Unified Understanding
---

# MiniMax H3 源码

**仓库**: `MiniMax-AI/MiniMax-H3` + `huggingface/diffusers` 中的 MiniMax-H3 实现

**解读**: [代码分析](../00_overview.md)

| 文件 | 说明 |
|------|------|
| [README.md](readme_md.md) | 系统三段式说明、H3-Base 架构与开源边界 |
| [transformer/config.json](transformer_config_json.md) | H3 主干配置：层数、hidden size、head 维、patch size |
| [transformer_minimax_h3.py](transformer_minimax_h3_py.md) | 单流 Omni Transformer：MM-RoPE、AdaLN、token refiner、双输出头 |
| [modular_blocks_minimax_h3.py](modular_blocks_minimax_h3_py.md) | 三种 workflow 的顶层 block 编排 |
| [before_denoise.py](before_denoise_py.md) | packed sequence 构造、rotary clock、latent 初始化 |
| [encoders.py](encoders_py.md) | Qwen3-VL presentation、文本条件编码、参考 VAE 编码 |
| [denoise.py](denoise_py.md) | 每步 transformer 前向与双 scheduler 更新 |
| [scheduling_minimax_h3.py](scheduling_minimax_h3_py.md) | MiniMax-H3 专用 rectified-flow Euler scheduler |
| [autoencoder_kl_minimax_h3.py](autoencoder_kl_minimax_h3_py.md) | 视频 VAE：时空压缩、ViT decoder |
| [autoencoder_kl_minimax_h3_audio.py](autoencoder_kl_minimax_h3_audio_py.md) | 音频 VAE：单声道编码/解码复用、双声道重组 |
