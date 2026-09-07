---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
---
# DiT 推理优化：Quantization

**文档**: `refs/codes/sglang/docs/docs/sglang-diffusion/quantization.mdx`  
**相关代码**: `runtime/utils/quantization_utils.py`、`configs/quantization/`、`tools/build_modelopt_nvfp4_transformer.py`

量化把权重/激活或 KV 压到更低 bit，属 **quality-tradeoff**。低 bit 不会自动变快：若前后仍在生成 BF16 大 Tensor，或反复 quant/dequant、cat、transpose，GEMM 省下的时间会被搬运吃掉。

## 1. 加载约定

常见拆分：

- `--model-path`：基座  
- `--transformer-path` / `--transformer-weights-path`：已量化 transformer 组件或权重  
- `--quantization`：对未量化模型做 online 量化  
- `--kv-cache-quant`：因果视频等场景的 KV 压缩  

## 2. 家族地图（文档摘要）

官方按 `quant_family` 组织（共享 CLI 与 loader），包括但不限于：

- Online `fp8` / `mxfp4`（激活动态量化；MXFP4 绑定 ROCm 等平台约束）  
- Offline / ModelOpt 路径（含 NVFP4 checkpoint，如 FLUX.2-dev-NVFP4）  
- Nunchaku / SVDQuant 生态（上游 `nunchux-ai/nunchaku`，本轮未 clone，仅索引）  
- 其它 INT8 / W4A16 / GGUF 等条目以 `quantization.mdx` 全表为准  

## 3. 与 Offload / Fusion

- 部分量化适配器会禁用不兼容的 DiT offload 模式（`transformer_load_utils.py`）  
- 「Token Cat + NVFP4 Quant」类融合决定量化是否真能体现在 e2e——只报 kernel 加速不够  

## 4. 本轮缺口

- 各 family 的 loader 类图与权重 layout  
- Nunchaku 在 SGLang 中的接线文件  
- Causal KV INT4/INT2 适用模型列表与正确性门禁
