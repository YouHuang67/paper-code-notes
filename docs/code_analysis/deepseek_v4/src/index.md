---
tags:
  - LLM Inference
  - CUDA
---

# DeepSeek V4 源码

**仓库**: deepseek_v4_pro · **路径**: encoding/ + inference/ + 根目录配置 · **解读**: [代码分析](../00_overview.md)

| 文件 | 行数 | 说明 |
|------|------|------|
| [README.md](readme_md.md) | 240 | 仓库级概览、模型规模、下载与使用入口 |
| [config.json](config_json.md) | 67 | 模型超参与长上下文/量化配置 |
| [encoding_dsv4.py](encoding_dsv4_py.md) | 744 | DeepSeek-V4 对话编码与 DSML 工具协议 |
| [generate.py](generate_py.md) | 155 | 推理入口、prefill/decode 循环、交互脚本 |
| [model.py](model_py.md) | 827 | 模型主体：mHC、MLA、MoE、KV 压缩与缓存 |
| [kernel.py](kernel_py.md) | 536 | TileLang kernel：量化、GEMM、稀疏注意力、Sinkhorn |
