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

返回：[专题总览](overview.md)

## 抓住重点

- Offload 是 **output-preserving**：不改去噪公式，只改权重驻留与 H2D 时机。
- 先分清两层：**组件级**（Encoder/DiT/VAE 整段）与 **层间级**（transformer block 流式进出）。
- SGLang 集成路径下：**Cache-DiT 与 DiT layerwise 不能同开**（硬错误）。Cache-DiT 库自带的 bucket offload 是另一条 API，勿与 SGLang `--dit-layerwise-offload` 混谈。
- 实践：看第一次请求后的 peak memory；能常驻的组件优先留 GPU，再考虑 layerwise。

## 1. 两层语义：组件级 vs 层间级

### 组件级（Component residency）

把 pipeline 拆成 text encoder、image encoder、DiT、VAE。请求头尾用 Encoder/VAE，中间把显存让给 DiT。

SGLang 入口：

- `component_resident_strategies.py`：策略表  
- `component_manager.py` 的 `ComponentResidencyManager`：请求期内 onload / finish_use / 跨请求预热  
- `auto_tune.py`：`--performance-mode` 为 `speed` / `memory` / `manual` 时套默认驻留与 layerwise 列表  

`memory` 模式倾向 offload / layerwise；`speed` 倾向 GPU-resident（多卡且显存够时还可能推 FSDP 替换 DiT offload）。

### 层间级（Layerwise）

只对选定组件的 blocks 做「算一层 → 预取下一层 → 释放上一层」。视频模型单层算得久，H2D 更容易藏进计算；小图像模型或多卡切碎后，copy stream 经常藏不住。

## 2. SGLang：`LayerwiseOffloadManager`

路径：`python/sglang/multimodal_gen/runtime/managers/memory_managers/layerwise_offload.py`  
注释写明改编自 Skywork AI Infra diffusion optimize。

机制：

1. 用 `layers_attr_str`（如 `blocks`）锚定 block 列表；正则 `^blocks\.(?P<layer_idx>\d+)`，避免误抓 `token_refiner.blocks`。  
2. 同 dtype 权重合并进 **pinned CPU** 大缓冲；非连续 stride 单独存。  
3. 专用 `copy_stream` + Event：`prefetch_layer` / `release_layer` 与 forward hook 协作。  
4. `prefetch_size`：前瞻预取深度（至少 1，且不超过层数）。  
5. `resident_layers`：前缀若干层跨 denoise step 常驻 GPU；`_residency_active` 延迟到**第一次 denoise forward** 再武装，避免 load 阶段就把 resident 集合钉死、和组件切换抢显存。  
6. `ComponentResidencyManager` 对「带大 resident set 的 layerwise DiT」会避免跨请求常驻，防止 OOM。

CLI：`--dit-layerwise-offload`、`--layerwise-offload-components`（`dit` / `default` / `all` 或具名）、`--dit-layerwise-resident-layers`、`--dit-offload-prefetch-size`。  
`--dit-layerwise-offload` 的 help 写明：可与 `--dit-cpu-offload` 组合（权重常驻 host，仅当前步需要的层上卡，峰值最低）。

## 3. Cache-DiT：bucket-style layerwise

路径：`src/cache_dit/offload/layerwise.py`（文件头设计说明 + `docs/user_guide/OFFLOAD.md`）

与「严格一层进一层出」不同，选中 submodule 是短流水线桶：

| 设计点 | 作用 |
|--------|------|
| Pinned CPU mirror | H2D/D2H 直接对 reusable GPU storage，不经 pageable |
| 两套 copy stream 池 | onload / offload 分开；慢 D2H 不堵下一桶 H2D |
| Prefetch 双预算 | 目标个数上限（约 `min(4×transfer_buckets, 8)`）+ 可选 `max_inflight_prefetch_bytes` |
| `persistent_buckets` / `persistent_bins` | 部分目标全程常驻，并按 bin 分散，避免热点全堆前缀 |
| `keep_activations_onload_device` | 选中目标盖住全部参数化叶子时，根级 hook 一次搬输入/输出，减少激活设备来回 |

公开入口：`layerwise_offload` / `layerwise_cpu_offload`。这是 **Cache-DiT 库路径**；与 SGLang 的 `LayerwiseOffloadManager` 不是同一套开关。

## 4. 组合约束（必须先读）

`server_args.py` 校验（pin 原文语义）：

| 组合 | 行为 |
|------|------|
| `SGLANG_CACHE_DIT_ENABLED` + DiT 在 layerwise 选择中 | **ValueError**：cache 可能复用已被 release 的 block 权重 → shape mismatch |
| Cache-DiT + FSDP | 显式开 FSDP → 报错；否则自动 `use_fsdp_inference=False` |
| DiT layerwise + FSDP | 自动关 FSDP |
| `dit_layerwise_resident_layers` 但未启用 DiT layerwise | 警告：无效 |

外部分享若写「Cache-DiT 可与 Layerwise Offload 配合」，通常指 Cache-DiT **自带** bucket API，或旧行为。**SGLang 集成路径必须以本 pin 校验为准。**

量化适配器也可能关不兼容 offload（`transformer_load_utils.py` 的 `_maybe_disable_incompatible_*_offload_modes`；单测 `test_layerwise_offload.py`）。

Progressive 文档建议 `--dit-cpu-offload false`，否则每步固定 PCIe 成本冲淡分辨率加速 → [Progressive Resolution](progressive_resolution.md)。

## 5. 何时该开

- **开 layerwise**：单卡装不下大 DiT，或峰值由深层堆叠主导；视频长序列单层算得够久。  
- **优先组件 offload**：Encoder/VAE 只在头尾用，DiT 要满速 → 也见 [Encoder & VAE](encoder_vae.md)。  
- **慎开**：通信已是瓶颈的多卡切分；计划开 Cache-DiT；H2D 藏不住的小图像模型。

## 5.1 一次 denoise step 的数据流

SGLang 的 layerwise 路径把 block 权重整理到 pinned host buffer，forward hook 在当前层开始前发起下一层 H2D，在当前层结束后异步释放上一层。`prefetch_size` 决定前瞻窗口，`resident_layers` 把少量前缀层固定在 GPU 上；首次 denoise forward 才启用驻留集合，避免模型加载阶段与 Encoder/VAE 抢显存。实现见 `runtime/managers/memory_managers/layerwise_offload.py`。

因此它优化的是峰值显存和传输重叠，计算量保持不变。单层计算时间不足以覆盖 H2D 时，copy stream 会成为新的串行边界；应以 Nsight 或 profile 结果决定窗口大小。

## 6. 源码锚点

| 主题 | 路径 |
|------|------|
| Layerwise manager | `.../memory_managers/layerwise_offload.py` |
| 组件选择常量 | `.../layerwise_offload_components.py` |
| 组件驻留状态机 | `.../component_manager.py` |
| 互斥校验 | `.../server_args/server_args.py`（`validate layerwise offload conflicts`） |
| 默认策略 | `.../server_args/auto_tune.py` |
| Cache-DiT bucket | `refs/codes/cache-dit/src/cache_dit/offload/layerwise.py` |

## 相关阅读

- [专题总览](overview.md) · [Feature Cache](feature_cache.md) · [Graph Runtime](graph_runtime.md) · [Correctness](correctness.md)  
- H3 效率附录讨论拓扑与 BCG，不替代本篇 offload 通论：[效率主线](../minimax_h3/05_efficiency_in_sglang.md)

## 附录：仍可加深

- `ComponentResidencyManager` 请求期时序图（onload / finish_use / 跨请求预热）  
- Cache-DiT bucket 参数与 SGLang manager 参数逐项对照表  
- 各模型在 `auto_tune` 下的默认 `layerwise_offload_components` 分支
