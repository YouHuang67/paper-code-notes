---
tags:
  - Video Generation
  - Diffusion Model
  - Unified Understanding
  - LLM Inference
---
# MiniMax H3 - Denoise Loop 状态机：static branch state、timestep plan、target-row 增量更新

**源码仓库**:

- [sgl-project/sglang](https://github.com/sgl-project/sglang)

**核心文件**:

- [denoise_loop.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py)
- [stages/denoising.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py)

## 1. 这层状态机负责什么

H3 的 denoise loop 不只是“按 scheduler 反复调模型”。它负责把 request 级静态信息和 step 级动态信息拆开，再把动态性收缩到 target rows 的数值更新。

对应分工是：

- `stages/denoising.py` 负责组装 full-loop 上下文
- `MiniMaxH3DenoiseBranch` 负责保存每个 branch 的静态布局与固定 forward kwargs
- `minimax_h3_denoise_loop()` 负责按 step 驱动主干前向与 Euler 更新

与 [MiniMax H3 在 SGLang 中的效率主线](05_efficiency_in_sglang.md) 的关系是：

- `05` 讲为什么 persistent row buffer 和 request-static 预计算重要
- 本文讲这些机制在代码里如何落地成一个稳定状态机

## 2. `stages/denoising.py` 先把 full-loop 上下文组装完整

`MiniMaxH3DenoisingStage._run_full_loop()` 在进入主循环之前，先完成：

1. 解析 `batch.extra` 中的文本、sigma schedule 和 denoise state
2. 组装 visual/audio condition rows
3. 构建 packed layout 与 `token_tags`
4. 预计算 refined prompt embeddings
5. 预计算 RoPE cache
6. 展开初始 video/audio rows
7. 把这些对象交给 `minimax_h3_denoise_loop()`

**源码位置**:

- [denoising.py:L503-L612](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py#L503-L612)
- [denoising.py:L328-L361](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py#L328-L361)
- [denoising.py:L364-L380](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py#L364-L380)

这一步之后，热循环里不再做：

- 文本 refinement
- RoPE cache 构造
- packed layout 求解
- condition row 组装

## 3. `MiniMaxH3DenoiseBranch` 固化了 branch 的静态布局

### 3.1 构造时就完成 row 角色划分

`MiniMaxH3DenoiseBranch.__init__()` 会把 packed layout 里的 row 角色先拆开：

- `img_cond_seq_idx`
- `img_target_seq_idx`
- `audio_target_seq_idx`
- `audio_ref_seq_idx`
- `cond_row_idx`
- `audio_ref_row_idx`

**源码位置**:

- [denoise_loop.py:L92-L230](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L92-L230)

这里最关键的是两个边界：

- `video_target_start = int((~self.update_mask).sum())`
- `audio_target_start = int((~self.audio_update_mask).sum())`

这说明 packed layout 明确把静态锚点放在前缀，把需要随 step 更新的 rows 放在后缀。

### 3.2 构造时就准备好 persistent buffer

branch 会持有两块持久缓冲区：

- `x_buffer`: `[1, seq_len, 96]`
- `audio_x_buffer`: `[1, seq_len, 32]`

**源码位置**:

- [denoise_loop.py:L163-L173](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L163-L173)

这两块 buffer 生命周期覆盖整个 denoise loop，而不是每个 step 重新分配。

### 3.3 构造时就把 rank-local 布局算完

branch 还会一次性求出：

- `local_row_slice`
- `block_token_tags`
- `local_embedding_layout`
- `packed_seq_params`
- `refiner_packed_seq_params`

**源码位置**:

- [denoise_loop.py:L174-L230](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L174-L230)
- [_build_local_embedding_layout:L53-L90](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L53-L90)

这样 `minimax_h3.py::_embed()` 后面就能直接按固定布局 scatter 到本 rank 的 local row shard。

## 4. `forward_kwargs()` 把“全量写一次、之后只写 target rows”固化下来

`forward_kwargs()` 的策略分成两段：

- 首步：把全部 `img_pos` / `audio_pos` rows 写进 persistent buffer
- 后续：只重写 `img_target_seq_idx` / `audio_target_seq_idx` 对应的 target suffix

**源码位置**:

- [denoise_loop.py:L232-L260](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L232-L260)

这一步的前提是：

1. condition/reference rows 在整轮生成期间是 pinned anchor
2. target rows 的物理位置在 packed layout 里已经固定

所以每个 step 变的不是输入几何，而只是 target rows 当前值。

## 5. `prepare_timestep_plan()` 把每步的 timestep 语义先展开

### 5.1 H3 每步并不只有一个 timestep

同一步里，不同 row 的 timestep 语义不同：

- 文本、padding、视频 target rows 继承当前 `t_video`
- visual condition rows 固定到 `max(t_video, imgvid_cond_noise_aug)`
- 音频 target rows 走当前 `t_audio`
- 音频 reference rows 固定到 `max(t_audio, audio_ref_cond_noise_aug)`

**源码位置**:

- [denoise_loop.py:L262-L315](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L262-L315)

因此 H3 的每步条件控制不是一个标量 timestep，而是一份 packed-row timestep layout。

### 5.2 `unique_timesteps` / `inverse_indices` / `block_combined_indices`

为避免每步构造整条 `[seq_len]` timestep tensor 再在 device 上 `torch.unique`，代码只对至多 4 个候选值做 host 侧去重，然后恢复出：

- `unique_timesteps`: 本步真正出现的 distinct timestep 集合
- `inverse_indices`: 每个 global row 映射到哪个 distinct timestep
- `block_combined_indices`: 本 rank local rows 的 `(modality, timestep-bucket)` 组合索引

**源码位置**:

- [denoise_loop.py:L262-L315](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L262-L315)

其中：

- `inverse_indices` 供全局 row timestep 恢复使用
- `block_combined_indices` 直接喂给 block 内的 indexed AdaLN kernel

这就是 H3 把“row 级 timestep 条件”压成紧凑索引语义的关键一步。

### 5.3 pattern cache 避免重复生成索引

`prepare_timestep_plan()` 里维护了两个 cache：

- `inverse_indices_by_pattern`
- `block_combined_by_pattern`

**源码位置**:

- [denoise_loop.py:L317-L353](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L317-L353)

当 step 间的 row->timestep pattern 相同，只需复用已有索引张量，不必重新分配和填充。

## 6. 主循环只保留三类动态工作

`minimax_h3_denoise_loop()` 的主循环里，真正每步重复发生的动态工作只剩三类：

1. 把当前 target rows 写回 persistent buffer
2. 调一次 DiT 主干前向
3. 对视频 target rows 和音频 target rows 各做一次 Euler 更新

**源码位置**:

- [denoise_loop.py:L356-L510](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L356-L510)

循环前还会先做两件固定工作：

- 把 visual condition rows 写成 `cond_anchor`
- 把 audio reference rows 写成 `audio_anchor`

**源码位置**:

- [denoise_loop.py:L402-L429](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L402-L429)

这两类锚点一旦写入，后续步骤不再变化。

## 7. 更新公式只作用在 target suffix

每步主干输出后，更新逻辑只落在：

- `video_rows[video_target_slice]`
- `audio_rows[audio_target_slice]`

**源码位置**:

- [denoise_loop.py:L30-L50](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L30-L50)
- [denoise_loop.py:L474-L505](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L474-L505)

`_minimax_h3_update_target_rows_()` 做的是 Euler-eta0 形式的 target-row 原位更新：

```text
denoised = state + sigma_t * velocity
state = sigma_ratio * state + (1 - sigma_ratio) * denoised
```

condition/reference rows 完全不参与这个更新。

## 8. 为什么这套状态机适合 native H3 runtime

这套状态机刚好满足 `runtime/models/dits/minimax_h3.py` 的高效路径要求：

- 输入几何固定
- local row ownership 固定
- `prompt_embeds` 固定
- `rope_cache` 固定
- `packed_seq_params` 固定
- 只有 `x/audio_x` 里的 target rows 每步变化

于是 `minimax_h3.py` 那条热路径就能把主要精力放在：

- `_embed()` 的 row-local scatter
- block 内 fused AdaLN / attention / MLP
- 输出侧的 SP/TP late gather

对应 runtime 热路径见 [DiT Runtime 与 Collectives](07_dit_runtime_and_collectives.md)。

## 9. 结论

H3 的 denoise loop 高效，不是因为 scheduler 数学本身特殊，而是因为它把整轮推理变成了一个很窄的状态机：

- 静态部分在 loop 外一次性求好
- 锚点 rows 在首步固定
- packed buffer 在整轮循环中复用
- timestep 条件被压成紧凑索引
- 动态更新只作用在 target suffix

这就是 `persistent row buffer + request-static precompute + indexed AdaLN` 能在同一条 H3 执行链里咬合起来的原因。
