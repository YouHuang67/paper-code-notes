---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
  - CUDA
---
# DiT 推理优化：Memory Offload

**源码 pin**:

- SGLang `refs/codes/sglang` @ `db75dfe…`
- Cache-DiT `refs/codes/cache-dit` @ `3db8d1e…`

Offload 解决的是「模型装不下 / 峰值显存打爆」时，如何把权重在 CPU↔GPU 间搬迁，并尽量用计算掩盖 H2D。它通常是 **output-preserving** 杠杆：不改变去噪公式，只改变权重驻留位置与拷贝时机。

## 1. 两层语义：组件级 vs 层间级

### 组件级（Component residency）

把 pipeline 拆成 text encoder、image encoder、DiT、VAE 等组件，决定哪些常驻 GPU、哪些整段 CPU offload。SGLang 用 `component_resident_strategies.py` + `auto_tune.py` 的 `--performance-mode`（`speed` / `memory` / `manual`）选择默认策略：显存紧时倾向 offload，显存够时倾向 GPU-resident。

实践习惯（与 BBuf 叙述一致）：看 **第一次请求后的 peak memory**，能留下的组件尽量留在 GPU，避免一股脑打开全部 offload 开关。

### 层间级（Layerwise）

只对 DiT（或指定组件）的 transformer blocks 做「算一层、预取下一层、释放上一层」。视频模型单层计算重，H2D 更容易藏进计算；图像模型或多卡切得很碎时，copy stream 经常藏不住，延迟上升。

## 2. SGLang：`LayerwiseOffloadManager`

路径：`python/sglang/multimodal_gen/runtime/managers/memory_managers/layerwise_offload.py`

机制要点：

- 以 `layers_attr_str`（如 `blocks`）锚定 block 列表，用正则匹配 `blocks.<idx>.*`，避免误抓嵌套 `token_refiner.blocks`
- 同 dtype 权重合并进 **pinned CPU** 大缓冲；非连续 stride 单独保存
- 专用 `copy_stream` + Event：`prefetch_layer` / `release_layer` 与 forward hook 协作
- `prefetch_size`：前瞻预取深度
- `resident_layers`：前缀若干层跨 denoise step 常驻 GPU，避免每步从头流式加载；`_residency_active` 延迟到第一次 denoise forward 再武装，避免 load 阶段就把 resident 集合钉死

配置入口包括 `--dit-layerwise-offload`、`--layerwise-offload-components`、`--dit-layerwise-resident-layers`、`--dit-offload-prefetch-size`，并由 `auto_tune` 在 memory 模式下尝试套默认组件列表。

## 3. Cache-DiT：bucket-style layerwise

路径：`src/cache_dit/offload/layerwise.py`（文档注释见文件头与 `docs/user_guide/OFFLOAD.md`）

与「严格一层进一层出」不同，它把选中 submodule 当成短流水线：

- pinned CPU mirror  
- **两套独立** onload/offload CUDA copy stream 池（onload 不被慢 D2H 堵住）  
- prefetch 窗口：目标个数上限 + 可选 `max_inflight_prefetch_bytes` 字节预算  
- `persistent_buckets` / `persistent_bins`：部分目标全程常驻，并按 bin 分散，避免热点全堆在前缀  
- 当选中目标覆盖全部参数化叶子时，可走 `keep_activations_onload_device`，减少激活设备来回

公开入口：`layerwise_offload` / `layerwise_cpu_offload`。

## 4. 组合约束（必须先读）

当前 SGLang pin 在 `server_args.py` 校验：

| 组合 | 行为 |
|------|------|
| Cache-DiT + DiT layerwise offload | **硬错误**：cache 可能复用已被 release 的 block 权重 → shape mismatch |
| Cache-DiT + FSDP inference | 显式打开则报错；否则自动关掉 FSDP |
| DiT layerwise + FSDP | 自动关掉 FSDP |
| `dit_layerwise_resident_layers` 但未启用 DiT layerwise | 警告：无效 |

注意：外部分享文案有时写「Cache-DiT 可与 Layerwise Offload 配合」。以本 pin 为准，**SGLang 集成路径下 DiT layerwise 与 Cache-DiT 不能同开**；若要用 Cache-DiT 自带的 bucket offload，应走 Cache-DiT API / 其文档路径，并单独验证与 SGLang 包装是否兼容。

量化路径也可能改写 offload：`transformer_load_utils.py` 中 ModelOpt FP8 / BitsAndBytes 适配器会 `_maybe_disable_incompatible_*_offload_modes`（见单测 `test_layerwise_offload.py`）。

Progressive resolution 官方文档建议关闭整模 `--dit-cpu-offload`，否则每步固定 PCIe 成本会冲淡分辨率带来的加速。

## 5. 何时该开

- **开 layerwise**：单卡装不下大 DiT，或 peak memory 由深层堆叠主导；视频长序列单层算得够久  
- **优先组件 offload**：Encoder/VAE 只在请求头尾用，DiT 需要满速  
- **慎开 / 别开**：已经 memory-bound 在通信上的多卡切分；计划开 Cache-DiT 加速；launch 已不是问题且 H2D 藏不住的小图像模型  

## 6. 与本仓其他笔记

- H3 效率附录讨论拓扑与 BCG，不替代本篇 offload 通论  
- 总览组合表：[overview](overview.md)

## 7. 本轮缺口（待加深）

- `ComponentManager` 状态机与请求期 onload/offload 时序图  
- Cache-DiT bucket offload 与 SGLang `LayerwiseOffloadManager` 参数对照表  
- 各 `--performance-mode` 默认组件列表的模型差异（`auto_tune` 全部分支）
