---
tags:
  - Video Generation
  - Diffusion Model
  - Unified Understanding
  - LLM Inference
---
# MiniMax H3 - SGLang 中的效率主线

**源码仓库**:

- [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)
- [sgl-project/sglang](https://github.com/sgl-project/sglang)

这篇只做一件事：把 H3 在 SGLang 中真正决定性能上限的因素梳理成一条清晰主线。

我会刻意不在正文堆太多枝节，因为 H3 的关键并不在“某个零散 trick”，而在几层优化刚好互相咬合：

1. **H3 先把问题改写成 packed single-stream 音视频 DiT**
2. **SGLang 再围绕这份 packed contract 专门做 kernel、通信和缓存设计**

所以最准确的问题不是“它用了哪个算子很快”，而是：

- 为什么这种计算形态本身就适合跑快
- 为什么 SGLang 的优化可以稳定落在这份形态上

实现补充、部署形态和长代码路径请看 [效率附录](06_efficiency_appendix.md)。

## 1. 第一原则：先把系统压成一条主计算链

H3 高效率的第一贡献，不来自 SGLang，而来自 H3 自己对问题的改写方式。

它没有把系统拆成：

- 一个视频主干
- 一个音频主干
- 若干 cross-attention / conditioner 分支

而是把：

- 文本
- 视觉条件
- 音频条件
- 目标视频 latent
- 目标音频 latent

全部压成一条 packed row sequence，然后交给一个单流 DiT 主干处理。

**相关背景**:

- [模型结构](01_model_architecture.md)
- [推理流程](02_inference_pipeline.md)

这一改写直接带来三件事：

- 没有双主干之间的显式特征交换
- 没有“视频 step / 音频 step 各跑一次大模型”的结构性浪费
- 整个系统只有一条真正昂贵的计算热区：**50 层 block stack**

这决定了后面的所有优化都可以非常聚焦。

## 2. 第二原则：每个 denoise step 只跑一次大模型

H3 的公开 checkpoint 是 CFG-distilled 单正分支，这一点在 SGLang pipeline config 中被直接固化。

**源码位置**:

- [pipeline config:L66-L89](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py#L66-L89)

这意味着它天生就绕开了扩散模型最常见的一类成本：

- negative branch
- CFG 双次主干前向
- CFG parallel 的额外同步和拼接

从“总 FLOPs”这个最粗粒度指标看，这就是 H3 能够快起来的第一大削减项。

如果主干每步还要跑两次，再多 kernel 融合也很难追平。

## 3. 第三原则：把热循环外能静态化的东西全部静态化

一旦整条生成过程围绕同一个 block stack 展开，最值钱的优化就不是再造一个 scheduler，而是尽可能缩小每个 denoise step 的“新增工作”。

H3 在 SGLang 中最重要的运行时思想，是把 request-static 状态尽量移出热循环：

- refined text embeddings
- RoPE cache
- sigma schedules
- local row layout
- packed-sequence metadata

### 3.1 文本 refinement 只做一次

SGLang 在 denoise 之前会先把 text encoder 输出进一步经过 token refiner，得到 request-static 的 refined prompt rows。

**源码位置**:

- [minimax_h3.py:L1251-L1269](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1251-L1269)
- [stages/denoising.py 中 `_precompute_refined_prompt_embeds`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py)

如果这一步放在 50-step loop 里，每步都会重复同一段文本投影和两层 refiner block；H3/SGLang 明确把它提了出去。

### 3.2 RoPE cache 只构一次

RoPE 也不在每步重建，而是针对本 rank 的 row shard 预先构好 cache。

**源码位置**:

- [minimax_h3.py:L1271-L1312](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1271-L1312)

这一步的收益不在数学复杂度，而在于：

- 让热循环里只保留真正随 timestep 改变的计算
- 让后续 attention 侧更接近纯 tensor core / packed attention 热区

### 3.3 scheduler 也先变成静态 schedule

H3 的 scheduler adapter 数学很薄，真正重要的是 SGLang 先生成整条：

- `video sigmas`
- `audio sigmas`

然后每步只取当前/下一步做更新。

**源码位置**:

- [timestep preparation stage](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/timestep_preparation.py)
- [scheduler adapter](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_minimax_h3_euler_ancestral.py)

这说明 H3 的性能中心从来不在 scheduler，而在 block stack 本身。

## 4. 第四原则：persistent row buffer，而不是每步重建整条输入

这是 H3 在 SGLang 中最核心、也最容易被忽略的优化之一。

`denoise_loop.py` 里的 `MiniMaxH3DenoiseBranch` 会持有 persistent packed buffers：

- `x_buffer`
- `audio_x_buffer`

第一次 forward 时把全量 row 写进去；之后每一步只重写 target rows，对 condition/reference rows 不再重复搬运。

**源码位置**:

- [denoise_loop.py:L163-L173](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L163-L173)
- [denoise_loop.py:L241-L258](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py#L241-L258)

它等价于把每个 step 的输入构造从：

- “重建整条 packed sequence”

变成：

- “增量更新 target suffix”

这带来的收益非常直接：

- 更少 host 端与 device 端的索引组织
- 更少全量 `index_copy_`
- 更少无变化 condition rows 的重复流动

H3 之所以能吃到这份收益，根本原因还是前面那条：它的 row layout 已经把静态锚点和动态 target 明确分开了。

## 5. 第五原则：算子不是泛泛 fused，而是围绕 H3 的 row contract 做融合

H3 在 SGLang 里的 kernel 设计不是“把常见算子都 fusion 一遍”，而是围绕最热、最重复、最符合 row-wise 语义的部分做融合。

### 5.1 QK norm + RoPE 直接融到 attention 前端

H3 attention 侧同时用了：

- `fused_inplace_qknorm`
- `fused_inplace_qknorm_rope`

**源码位置**:

- [minimax_h3.py:L271-L296](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L271-L296)
- [minimax_h3.py:L653-L676](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L653-L676)

这个融合的价值不是“函数调用少一点”，而是把：

- q/k RMSNorm
- rotary application

从多次读写中间 tensor，压成更接近原地的前处理路径。对于 50 层 attention 来说，这种节省会被成倍放大。

### 5.2 AdaLN modulation 走 indexed Triton kernel

H3 的 AdaLN 不是 batch 维统一一个 scale/shift，而是行级别按 `combined_indices` 选择不同 modulation 参数。因此 SGLang 没走普通 `index_select + elementwise` 路，而是用了：

- `indexed_scale_shift_bf16_`
- `indexed_gate_bf16_`

**源码位置**:

- [minimax_h3.py:L211-L255](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L211-L255)

这非常关键，因为 H3 的 block stack 每层都要做两轮 AdaLN 调制；如果这里退回通用 gather + pointwise 路线，显存带宽和 launch 开销都会迅速放大。

### 5.3 MLP activation 也做成 fused hot path

MLP 的 `SiLU(gate) * up` 被做成了专门 fused path：

- `silu_and_mul_with_activation_rounding_`

**源码位置**:

- [minimax_h3.py:L258-L268](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L258-L268)
- [minimax_h3.py:L732-L735](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L732-L735)

单层收益不夸张，但 50 层累加后依然可观。

## 6. 第六原则：attention 的关键不是“用了 FA”，而是 packed varlen + Ulysses/Ring 共设计

H3 在 SGLang 中的 attention 设计，最关键的不只是 FlashAttention 后端，而是它和 packed sequence、sequence parallel 的共设计。

### 6.1 attention 原生吃 packed varlen sequence

H3 要求 attention backend 支持：

- `packed_varlen=True`

**源码位置**:

- [pipeline config:L192-L197](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py#L192-L197)
- [attention core:L464-L498](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L464-L498)

这意味着 attention 直接工作在：

- `cu_seqlens`
- `max_seqlen`
- packed multimodal row layout

而不是先把不同模态 pad 成更大的 dense tensor。这个选择直接减少了无效 attention 计算和无效搬运。

### 6.2 Ulysses SP 用来换掉长序列 activation 压力

H3 packed sequence 很长，真正的瓶颈之一是 block stack 内部 activation 占用。SGLang 的做法不是只靠 TP，而是让 attention 内部支持 Ulysses：

- 行切分在外
- attention 内部 sequence/head all-to-all

**源码位置**:

- [attention core:L456-L500](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L456-L500)
- [forward row split:L1554-L1581](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1554-L1581)

它的意义不是“多卡支持更丰富”，而是：

- 把长序列负担分掉
- 同时避免每层都走更重的 TP 通信模式

### 6.3 Ring SP 补的是更大 world size 下的 row 外层切分

H3 还允许 ring degree 叠在 Ulysses 外面。Ring 不分 heads，只分 rows，用 online softmax 方式聚合部分结果。

**源码位置**:

- [attention core:L473-L489](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L473-L489)
- [sequence-parallel 校验:L1069-L1101](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1069-L1101)

这让 H3 的 row contract 能继续扩展到更大拓扑，而不必完全重写注意力语义。

## 7. 第七原则：通信优化的重点不是“少通信”，而是“晚 gather、窄 gather”

H3/SGLang 的通信优化非常克制。它没有神奇地把 collectives 变没，而是尽量把通信放到更晚、更窄的时候做。

### 7.1 block AdaLN 先批处理，再一次 gather

如果满足条件，SGLang 会把所有 block 的 AdaLN local projection 先堆叠起来，再做一次 TP all-gather。

**源码位置**:

- [minimax_h3.py:L1030-L1038](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1030-L1038)
- [minimax_h3.py:L1642-L1650](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1642-L1650)

这等价于把“50 层很多次小 gather”压成“一次更规整的大 gather”，减少 collective launch 和调度碎片。

### 7.2 final logits 先裁 dead rows，再做 TP gather

final layer 先算全 row 输出，但真正的 TP gather 发生在：

- 只保留 live video rows
- 只保留 live audio rows

之后。

**源码位置**:

- [final layer 注释:L1003-L1008](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1003-L1008)
- [forward gather 路径:L1690-L1698](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1690-L1698)

这一步之所以重要，是因为它避免在：

- text rows
- padding rows
- 不需要导出的中间 rows

上浪费 TP 通信 payload。

H3 在通信层最核心的思路，其实就是这一句：

- **先把对外无用的 rows 裁掉，再做列聚合**

## 8. 第八原则：breakable CUDA graph 只在最动态的地方断开

H3 在 SGLang 中不是“彻底 eager”，也不是“全量强 capture”。它把 breakable CUDA graph 的断点收得很窄：

- projection、AdaLN、MLP 尽量保持图内
- 真正动态的 packed attention core 和 SP collectives 放在 eager break

**源码位置**:

- [attention core 注释:L449-L454](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L449-L454)
- [eager_on_graph 包装:L504](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L504)

这非常符合 H3 的结构特征：

- attention 的 shape / row partition / collective 语义最动态
- 其余 block 路径则相对稳定

于是它既保留了动态图弹性，也尽量不把整层 block 都踢出图外。

## 9. 第九原则：精度策略不是全低精度，而是把关键路径留在 fp32 island

H3 在 SGLang 中有一组非常明确的 fp32 island：

- patch projections
- timestep embedder
- final output heads
- `rope.inv_freq`

**源码位置**:

- [fp32 参数定义:L73-L88](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L73-L88)
- [post-load 校验:L1190-L1200 左右](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py)
- [模块定义:L1125-L1159](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1125-L1159)

它说明一件事：H3 的性能路径不是“粗暴全部压 BF16/FP8”，而是：

- 大部分 block stack 走 BF16 + fused path
- 少数数值敏感点保 fp32

这让它既能快，也更容易维持稳定输出契约。

## 10. 把主线收束成一句话

如果把前面所有内容压缩成最核心的判断，H3 在 SGLang 中之所以高效，关键不是某一个 kernel，而是以下 5 件事同时成立：

1. **模型层先做对了问题改写**：把联合音视频生成折叠成 packed single-stream DiT。
2. **主干层避免结构性浪费**：CFG-distilled 单分支让每步只跑一次大模型。
3. **运行时层最大化 request-static 复用**：text / rope / schedule / row layout 都尽量移出热循环。
4. **内核与通信围绕 row contract 专门化**：indexed AdaLN、fused qknorm+rope、packed varlen attention、Ulysses/Ring SP、late gather。
5. **loop 层避免全量重建输入**：persistent row buffer 只重写 target suffix，而不是每步重拼整条序列。

这五条里，前两条决定“它本来就比很多系统更适合高性能执行”，后三条决定“这种潜力真的能在 SGLang runtime 里被兑现出来”。

实现补充、代码路径和部署形态见 [效率附录](06_efficiency_appendix.md)。
