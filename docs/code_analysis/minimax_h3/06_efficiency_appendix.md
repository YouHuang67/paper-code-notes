---
tags:
  - Video Generation
  - Diffusion Model
  - Unified Understanding
  - LLM Inference
---
# MiniMax H3 - 效率附录

附录收纳正文外的补充材料：

- H3 native pipeline 的关键文件地图
- breakable CUDA graph prompt padding 的具体策略
- 部署拓扑为什么会分化成 `Ulysses4` / `TP2+Ulysses2` / `offload+TP2`

denoise loop 的 row-state 细节见 [Denoise Loop 状态机](08_denoise_loop_state_machine.md)。

## 1. 文件地图：真正与性能最相关的是哪些文件

### 1.1 pipeline 层

- [`minimax_h3_pipeline.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py)
- [`configs/pipeline_configs/minimax_h3.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py)

职责：

- 把 H3 绑定成 native pipeline
- 固定 stage 链
- 固定 `native_only_components`
- 固定 `packed_varlen` attention 要求

### 1.2 热循环层

- [`runtime/models/dits/minimax_h3.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py)
- [`denoise_loop.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py)
- [`scheduling_minimax_h3_euler_ancestral.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_minimax_h3_euler_ancestral.py)

职责：

- packed DiT forward
- row-local embedding / SP 切分
- fused AdaLN / attention / output gather
- whole-loop denoise 状态更新

### 1.3 graph / capture 层

- [`model_padders/minimax_h3.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/breakable_cuda_graph/model_padders/minimax_h3.py)

职责：

- 让不同 prompt 长度在可控范围内复用 graph signature
- 又不破坏主 packed sequence 的 row partition 与数值路径

## 2. `local_embedding_layout` 为什么值钱

如果没有 `local_embedding_layout`，`_embed()` 每步都要重新做：

- `text_pos` 在本 rank 行区间里的筛选
- `img_pos` 在本 rank 行区间里的筛选
- `audio_pos` 在本 rank 行区间里的筛选

**源码位置**:

- fallback 路径: [minimax_h3.py:L1392-L1413](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1392-L1413)

而 SGLang 在 serving 正路中，会先构建好：

- `text_source_start/stop`
- `img_global_ids/img_row_ids`
- `audio_global_ids/audio_row_ids`

**源码位置**:

- [denoise_loop.py:L53-L90](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L53-L90)
- [denoise_loop.py:L209-L217](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L209-L217)
- [minimax_h3.py:L1415-L1460](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1415-L1460)

这样 `_embed()` 热路径就从“边算布局边 scatter”，变成“直接按既定布局写局部 shard”。

## 3. 为什么 H3 要专门处理 QKV checkpoint layout

H3 的 attention checkpoint 把 qkv 融在一起保存，而 TP 下三个逻辑矩阵其实必须独立分片。SGLang 因此没有直接复用普通 `ColumnParallelLinear` 的默认切法，而是显式实现了：

- grouped qkv 重排
- TP-local qkv shard loader

**源码位置**:

- [minimax_h3.py:L126-L196](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L126-L196)
- [minimax_h3.py:L532-L545](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L532-L545)
- [minimax_h3.py:L585-L614](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L585-L614)

这一步本身不是运行时热区，但它非常关键，因为：

- 它让 TP 下的 qkv 逻辑和 checkpoint 物理布局真正对齐
- 否则后面所有 fused path 都站不稳

## 4. 为什么 prompt bucket padding 只 pad text，不 pad 主 sequence

H3 的 breakable CUDA graph padder 做了一件很克制的事：

- 为了稳定 graph signature，会把 `prompt_embeds` pad 到 bucket
- 但不会把主 packed sequence 一起扩成 `media_rows + bucket`

**源码位置**:

- [padder:L88-L99](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/breakable_cuda_graph/model_padders/minimax_h3.py#L88-L99)
- [padder:L118-L157](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/breakable_cuda_graph/model_padders/minimax_h3.py#L118-L157)

它这样做是为了同时避免两个坏结果：

1. 不 pad text：prompt 长度一变就 miss capture
2. 连主 sequence 也 pad：SP row partition 改变，GEMM 形状和数值路径一起变

所以它的策略是：

- 只 pad 最需要 bucket 化的文本维
- 主 row 几何保持在既有 64-row alignment 组里

这也是为什么 H3 的 BCG 不是“盲目提高命中率”，而是优先守住主计算路径稳定性。

## 5. 为什么 topology 会分化成 H200 / H100 / 5090 三条路

### 6.1 4xH200：resident + Ulysses4

H200 的关键不是“更快的 kernel”，而是它允许 H3 把整条 BF16/FP32 pipeline 常驻，同时只靠 Ulysses4 把长序列 activation 分掉。

结果是：

- 不需要额外 TP 列切分
- 不需要 FSDP 逐 block 权重聚合
- resident 路径最纯

### 6.2 4xH100 80GB：TP2 + Ulysses2

H100 80GB 更像是一个平衡点：

- 纯 Ulysses4 对 resident 压力偏大
- 引入 TP2 能分掉部分列宽和参数驻留压力
- 再保留 Ulysses2 去处理长序列 activation

这是 H3 这类“大 block stack + packed 长序列”模型在 80GB 级别显存上的自然折中。

### 6.3 2xRTX5090：offload + TP2

消费卡路线已经不再是单纯算子问题，而是：

- 哪些层常驻
- 哪些层按 layerwise offload 预取
- offload 粒度如何和 block stack 对齐

H3 能在这类环境下仍然保持可用，依赖的正是它把最重热区集中在 DiT block stack 上，便于围绕 blocks 做驻留与传输调度。

## 6. 为什么 H3 当前的 speed path 明确不依赖默认 `torch.compile`

SGLang 对 H3 的速度路径给出了非常明确的约束：

- `speed_mode_enable_torch_compile_by_default = False`

**源码位置**:

- [pipeline config:L82-L89](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py#L82-L89)

这代表一个重要事实：

- H3 当前高效率的主贡献并不来自 compile
- 它主要来自 eager native runtime 上的 row-contract 专门化

所以分析 H3 的速度时，应该优先看：

- loop state 复用
- fused modulation / qknorm / rope
- packed varlen attention
- SP/TP late gather

而不是把焦点放在 compile。

## 7. 一张总表：核心收益分别来自哪里

| 层次 | 关键机制 | 主要收益 |
|------|----------|----------|
| 模型结构 | packed single-stream audio-video DiT | 把系统压成一条主热区 |
| 训练/发布 | CFG-distilled 单分支 | 每个 denoise step 只跑一次主干 |
| loop 状态 | persistent row buffer | 避免每步全量重建 packed input |
| request-static | refined text / rope / sigmas / local layout | 减少热循环重复工作 |
| 算子 | indexed AdaLN / fused qknorm+rope / fused silu*mul | 降低 launch 和中间访存 |
| attention | packed varlen + Ulysses/Ring | 降 activation 压力，保持长序列可扩展 |
| 通信 | batched AdaLN gather + late gather | 压低 collective payload |
| graph | 窄边界 BCG + text-only bucketing | 兼顾动态图与 graph 复用 |
