---
tags:
  - Video Generation
  - Diffusion Model
  - Unified Understanding
  - LLM Inference
---
# MiniMax H3 - DiT Runtime 与 Collectives

**源码仓库**:

- [sgl-project/sglang](https://github.com/sgl-project/sglang)

**核心文件**:

- [runtime/models/dits/minimax_h3.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py)
- [denoise_loop.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py)

前面的正文已经把结论说清楚了：H3 在 SGLang 里之所以快，核心在于 packed single-stream DiT、persistent row buffer、fused AdaLN/QKNorm/RoPE、packed varlen attention、Ulysses/Ring 和 late gather。

但如果想像读 FlashAttention 一样，把“这些结论到底在代码里如何串成一条热路径”看透，还得单独盯住 `runtime/models/dits/minimax_h3.py`。

这篇就只做一件事：**按执行顺序，把 H3 的 native DiT runtime 热路径梳理清楚**。

建议先读：

- [如何嵌入 SGLang 体系](04_sglang_integration.md)
- [SGLang 中的效率主线](05_efficiency_in_sglang.md)

## 1. 先抓住这个文件的职责边界

`minimax_h3.py` 并不是一个“普通模型定义文件”。它同时承担了 4 层职责：

1. 定义 H3 native DiT 的 forward contract
2. 定义 row-local embedding 的构造方式
3. 定义 block stack 内部的 fused dataflow
4. 定义 SP/TP/Ring/Ulysses 下的 collectives 边界

这意味着它不像 diffusers 那种“高层语义很清楚，具体性能看 backend”；这里的高层语义和性能路径是绑在一起的。

如果把整个执行图抽象成一条主链，可以写成：

```text
packed rows
  -> request-static text refinement / rope cache
  -> rank-local embedding materialization
  -> 50 x [AdaLN -> Attention -> AdaLN -> MLP]
  -> row gather across SP
  -> late gather across TP
  -> video/audio logits
```

这条链里最贵的是中间的 50 层 block stack，但真正把它跑快的，是前后的数据组织和后面的 gather 时机。

## 2. Forward contract：它吃的不是“普通 DiT 输入”

H3 的 native DiT `forward()` 一上来先拒绝任何不在白名单里的 kwarg。

**源码位置**:

- [minimax_h3.py:L97-L123](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L97-L123)
- [minimax_h3.py:L1465-L1480](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1465-L1480)

它要求的输入，不是典型的：

- `hidden_states`
- `encoder_hidden_states`
- `timestep`

而是 H3 自己的 packed contract：

- `x`
- `audio_x`
- `img_position_ids`
- `prompt_embeds`
- `packed_seq_params`
- `refiner_packed_seq_params`
- `local_embedding_layout`
- `img_pos_info/audio_pos_info/text_pos_info`
- `inverse_indices`
- `block_token_tags`

这套 contract 本质上说明了一件事：

- **H3 的运行时核心对象不是张量 batch，而是一条被严格标注过的 packed row sequence**

一旦这个对象成立，后面几乎所有优化才会自然成立。

## 3. 第一步不是进 block，而是把 row-local embedding 摆好

### 3.1 `_embed()` 是真正的输入 staging 区

H3 的 `_embed()` 被 `@eager_on_graph(True)` 包起来，专门负责把当前 rank 该拥有的 row shard 变成 block stack 可直接消费的 `decoder_input`。

**源码位置**:

- [minimax_h3.py:L1314-L1463](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1314-L1463)

可以把它的语义写成：

```text
text rows
video rows
audio rows
  -> scatter into one local [S_local, H] buffer
  -> return embeddings + distinct timestep embeddings
```

这一步的重要性远大于“简单做个线性投影”，因为它决定了 block stack 是否能只处理：

- 本 rank 的连续 row shard
- 统一 width 的残差流

### 3.2 它优先使用 `local_embedding_layout`，不是临时扫描位置

如果 `local_embedding_layout` 存在，`_embed()` 直接按预先算好的：

- `text_source_start/text_source_stop`
- `img_global_ids/img_row_ids`
- `audio_global_ids/audio_row_ids`

来写本地 buffer。

**源码位置**:

- [minimax_h3.py:L1374-L1428](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1374-L1428)
- [minimax_h3.py:L1436-L1460](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1436-L1460)

而不是每一步再通过：

- `nonzero`
- `pos >= row_start`
- `pos < row_stop`

临时筛选本 rank 拥有哪些 row。

这就是一个典型的“把 shape-dependent 控制流前移”的优化：

- serving 阶段先把 row ownership 解出来
- block stack 热路径只负责读和写

### 3.3 视频 / 音频 patch projection 仍然保留 fp32

`_embed()` 里，视频和音频 row 在进入 `video_patch_proj` / `audio_patch_proj` 之前都会先转到 fp32，再投影，再 cast 回 bf16。

**源码位置**:

- [minimax_h3.py:L1436-L1460](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1436-L1460)

这一步的直觉很简单：

- patch projection 是所有视觉 / 音频 row 进入主残差流的唯一入口
- 如果这里一开始就把误差放大，50 层 block 都会继承这份误差

所以它宁可把真正重的 block stack 压 bf16，也把入口投影保成 fp32 island。

## 4. Request-static 条件是怎么被塞进热路径外的

### 4.1 文本 refinement 先做掉

如果 `refined_prompt_embeds_length` 已经存在，`_embed()` 直接使用 refined text，不再重复跑 token refiner；否则才 fallback 到 `refine_prompt_embeds()`。

**源码位置**:

- [minimax_h3.py:L1241-L1269](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1241-L1269)
- [minimax_h3.py:L1342-L1372](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1342-L1372)

这意味着 serving 正路已经把 text refinement 明确视作：

- request-static
- not step-static

而不是每步都重算的动态条件。

### 4.2 RoPE cache 也是按 rank row shard 先做好

`build_rope_cache()` 会根据 `ring_rank` 和 `ulysses_rank`，只为本 rank 的 local row shard 构建 cos/sin cache。

**源码位置**:

- [minimax_h3.py:L1271-L1312](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1271-L1312)

这里最值得注意的不是“缓存了 RoPE”，而是：

- row split 的定义必须和 `forward()` 里的 sequence-parallel row split 完全同构

否则 cache 就会对错行。

所以 H3 的 request-static cache 不是独立小优化，而是和 SP row geometry 绑死的。

## 5. Block 内部 dataflow：为什么说 H3 的块很像“薄控制层 + 厚共享算子”

### 5.1 Block 结构本身并不复杂

单个 `MiniMaxH3DiTBlock` 的顺序非常干净：

```text
norm1
-> indexed scale/shift
-> attention
-> indexed gate residual

norm2
-> indexed scale/shift
-> MLP
-> indexed gate residual
```

**源码位置**:

- [minimax_h3.py:L875-L950](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L875-L950)

这和 H3 open-source diffusers 版的语义是一致的，但 native runtime 在这里做了两层关键变化：

- modulation 走 indexed fused kernel
- attention 走 packed varlen + SP-aware backend

### 5.2 AdaLN 不是“每个 row 一套小网络”，而是 table-like broadcast

`MiniMaxH3AdalnProj` 的输入是 `adaln_input = silu(t_emb)`，输出是：

- `shift_msa`
- `scale_msa`
- `gate_msa`
- `shift_mlp`
- `scale_mlp`
- `gate_mlp`

它们的组织方式本质上是一张 `(timestep bucket, modality)` 表，然后按 `combined_indices` 选给每个 row。

**源码位置**:

- [minimax_h3.py:L739-L791](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L739-L791)

因此它在运行时的真实角色更像：

- 少量 distinct timestep 的 modulation table 生成器

而不是：

- 对每个 token 都独立前向的条件网络

### 5.3 为什么 indexed modulation 值得专门写 kernel

块里真正高频出现的是：

- `_modulate_scale_shift()`
- `_modulate_gate()`

它们在满足 CUDA bf16 contiguous 条件时会落到：

- `indexed_scale_shift_bf16_`
- `indexed_gate_bf16_`

**源码位置**:

- [minimax_h3.py:L211-L255](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L211-L255)

如果不这么做，块内每一步都会退化成：

1. 先按 indices gather AdaLN 参数
2. 再做逐元素 scale/shift 或 gated residual
3. 再写回

这会把本来很规整的 block 热路径变成“很多小 gather + 很多 pointwise launch”。所以 H3 的 block 想快，AdaLN 这里必须专门化。

## 6. Attention 前端：QKV loader、QKNorm、RoPE 是一套连续的前处理链

### 6.1 `MergedColumnParallelLinear` 先把 qkv 逻辑矩阵与 TP 对齐

H3 的 attention 先用 `MergedColumnParallelLinear` 做 fused qkv projection，但权重加载时会额外处理 checkpoint 的 grouped-qkv 排布。

**源码位置**:

- [minimax_h3.py:L126-L196](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L126-L196)
- [minimax_h3.py:L532-L545](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L532-L545)
- [minimax_h3.py:L585-L614](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L585-L614)

这说明 H3 的 TP 不是“先写个线性层再让框架自动切”，而是从 checkpoint layout 起就主动保证：

- q / k / v 的逻辑矩阵分片方式正确

### 6.2 QKNorm 和 RoPE 被尽量压成一段 fused 前处理

Attention forward 里，如果 `rope_cache` 存在且平台支持，会直接走：

- `fused_inplace_qknorm_rope`

否则至少也会走：

- `fused_inplace_qknorm`

**源码位置**:

- [minimax_h3.py:L271-L296](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L271-L296)
- [minimax_h3.py:L644-L676](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L644-L676)

这一段在性能上的真实意义是：

- qkv projection 后立刻进入就地前处理
- 尽量少制造中间 q/k buffer 的往返读写

而且因为 H3 的 row shard 已经在 `_embed()` 阶段稳定下来，这段前处理天然是 rank-local 的，很适合被压成一段连续热区。

## 7. Attention 核心：为什么 BCG 断点只放在这里

### 7.1 `_minimax_h3_attention_core_impl` 是整条热路径最动态的地方

SGLang 明确把 attention core 定义成：

- 动态 varlen attention
- Ulysses input/output all-to-all
- Ring attention varlen

这段函数再通过 `eager_on_graph(True)` 包成 narrow graph break。

**源码位置**:

- [minimax_h3.py:L437-L504](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L437-L504)

它为什么是唯一值得断开的地方？

因为真正随请求 shape、row layout、SP topology 变化最大的就是：

- `cu_seqlens`
- `max_seqlen`
- all-to-all / ring collectives

而：

- qkv projection
- qk norm
- RoPE
- AdaLN
- MLP

这些反而更接近固定形状的 dense 计算。

### 7.2 Ulysses 在这里做“sequence for heads”交换

如果 `ulysses_active`，attention core 会先执行：

- `_usp_input_all_to_all_packed_qkv`

把 row-sharded q/k/v 变成 head-sharded attention 输入；attention 结束后再通过：

- `_usp_output_all_to_all`

换回 row shard。

**源码位置**:

- [minimax_h3.py:L456-L500](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L456-L500)

这一步非常像一个“attention 内部的局部 layout transform”。外部 block stack 始终认为自己处理的是 row shard；只有 attention 核心内部，短暂把 sequence partition 换成 head partition。

### 7.3 Ring 再叠一层 row 外切分

如果 `ring_active`，attention core 不走普通 varlen backend，而是走：

- `_ring_attention_varlen`

**源码位置**:

- [minimax_h3.py:L473-L489](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L473-L489)

这里最关键的不是 Ring “支持跨节点”，而是它保持了 H3 的统一语义：

- 外部仍然是一条 packed sequence
- `max_seqlen` 仍然对应该请求真实 `used` row count
- ring 只是把这条 sequence 的 KV 外层切开，再用 online softmax 合并

因此 Ring 是 row geometry 的扩展，不是模型语义的重写。

## 8. Sequence Parallel 几何：为什么 forward 里要显式推导 `row_start/row_stop`

`forward()` 里有一段非常重要的代码，专门推导：

- `sp_ws`
- `local_seq_len`
- `ring_chunk_len`
- `row_start`
- `row_stop`

**源码位置**:

- [minimax_h3.py:L1554-L1581](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1554-L1581)

这段代码决定了两个事实：

1. block stack 外层的并行单位是连续 row shard
2. ring 是外层维，ulysses 是内层维

因此整条 runtime 都围绕一个隐含不变量展开：

- **每个 rank 永远处理 packed sequence 的一个连续子区间**

只要这一点成立：

- `_embed()` 可以直接做 local scatter
- `build_rope_cache()` 可以直接 slice local rows
- final layer 也能先 row-local 出 logits 再 gather

这就是 H3 runtime 能维持“前后几何一致”的基础。

## 9. Block stack 外的两个 gather：为什么都故意放晚

### 9.1 先做 SP row gather

所有 block 跑完以后，如果 `sp_ws > 1`，先在 SP group 上做：

- row-wise `all_gather`

得到完整 row 维度上的 TP-local logits。

**源码位置**:

- [minimax_h3.py:L1677-L1688](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1677-L1688)

注意这里 gather 的只是：

- 当前 TP shard 的输出宽度

所以它不是“最大张量的最重 gather”，而是一个已经被列切分过的 gather。

### 9.2 再裁掉 dead rows，最后才做 TP gather

随后 H3 会先做：

- `video_logits = index_select(infer_out_pos)`
- `audio_logits = index_select(audio_pos)`

把不需要对外导出的 rows 去掉，再在 TP 维度上做最终 `all_gather`。

**源码位置**:

- [minimax_h3.py:L1690-L1698](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1690-L1698)

这就是 H3 整个通信路径里最漂亮的一点：

- **row 先聚全**
- **dead rows 先裁掉**
- **列最后才聚全**

它避免了很多实现里常见的“图省事，先 full gather 再裁剪”的浪费。

## 10. 回到 denoise loop：这个 runtime 为什么刚好适合 H3 的状态机

前面这条 runtime 热路径之所以能稳定发挥作用，是因为 `denoise_loop.py` 给它喂的状态本来就长成它喜欢的样子：

- row layout 静态
- text refinement 静态
- rope cache 静态
- `packed_seq_params` 静态
- 只有 target rows 的 `x/audio_x` 每步变化

**源码位置**:

- [denoise_loop.py:L193-L230](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L193-L230)
- [denoise_loop.py:L232-L260](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L232-L260)

所以从系统角度看，H3/SGLang 不是“有个快模型，再有个快 loop”，而是：

- loop 把动态性收缩到 target row values
- runtime 把静态几何和静态条件最大化复用

这两者正好咬合。

## 11. 用一段伪代码把整条热路径收起来

```python
# request-static
text = refine_prompt_embeds(prompt_embeds)
rope_cache = build_rope_cache(img_position_ids_for_local_rows)
layout = local_embedding_layout

for step in denoise_steps:
    # only target rows changed
    x_buffer[target_img_rows] = current_video_rows
    audio_x_buffer[target_audio_rows] = current_audio_rows

    hidden = embed_local_rows(
        x_buffer, audio_x_buffer, text, layout, unique_timesteps
    )

    for block in blocks:
        hidden = block(
            indexed_adaln(hidden),
            packed_varlen_attention(hidden, cu_seqlens, ulysses, ring),
            fused_mlp(hidden),
        )

    video_logits, audio_logits = final_layer(hidden)
    video_logits = sp_gather(video_logits)
    audio_logits = sp_gather(audio_logits)
    video_logits = select_live_video_rows(video_logits)
    audio_logits = select_live_audio_rows(audio_logits)
    video_logits = tp_gather(video_logits)
    audio_logits = tp_gather(audio_logits)
```

这段伪代码的重点不在语法，而在你能一眼看到 H3 的核心策略：

- 把动态性缩到 target row values
- 把几何和条件做成静态输入
- 把最重计算压到一套高度专门化的 block stack 上
- 把 gather 尽量往后推

## 12. 最后的判断

如果只盯住 `minimax_h3.py` 这一份 runtime 文件，H3 的高效率可以被更具体地描述成：

- **它不是“一个大 DiT + 几个优化”**
- **而是一套围绕 packed row geometry 精心组织的执行路径**

其中真正最关键的不是某一个 kernel，而是 3 个层次同时成立：

1. `_embed()` 把 rank-local row shard 整理成 block stack 最喜欢的连续形状
2. block stack 内部把 modulation、attention 和 MLP 压成高度规整的热区
3. collectives 在最晚时机才发生，而且只搬仍然有价值的 rows / columns

这也是为什么 H3 的性能分析不能只说“它用了 FlashAttention”。

更准确的说法应该是：

- **H3 把多模态生成改写成一条 packed single-stream dataflow**
- **SGLang 再把这条 dataflow 实现成一条几乎处处围绕 row geometry 优化的 native runtime**

