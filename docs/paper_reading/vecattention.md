---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
  - Triton
---

# VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference

- 论文：https://arxiv.org/abs/2603.29494
- 代码：https://github.com/anminliu/VecAttention
- 团队：Peking University, Fudan University, Alibaba Group, Shanghai Jiao Tong University

## 概述

VecAttention 面向长上下文视频理解和视频生成。它的核心不是继续做 block sparse，而是把注意力图中的**垂直向量**作为稀疏单元：对一段相邻 query，共享一组重要 key columns。

直观地说，视频 token 在时间和空间上相邻时，多个 query 往往会关注同一批关键帧、关键区域或全局语义 token。因此 attention map 里真正重要的区域常表现为短而碎的 vertical line segments，而不是完整竖线、斜线或大矩形 block。VecAttention 用 query pooling 得到每个 query group 的代表向量，再为每个 query group 动态选择重要 key vectors，最后只对这些 key/value rows 做 attention。

工程上，VecAttention 分成两段：

1. **Important-vector selection**：Triton kernel 在 pooled-Q 与 K 的 tile GEMM 内直接做 minS 过滤，输出每个 query block 的 selected key indices，避免把估计 attention map 写到 HBM。
2. **Vector sparse attention**：modified `vllm_flash_attn.sparse_attn_func` 消费 `col_count/col_idx`，对每个 Q tile gather-load 被选中的 K/V rows，并用 FlashAttention 风格 online softmax 累积输出。

论文在 VLM 和 DiT 上都评测。视频生成部分使用 Wan2.1-T2V-14B 和 HunyuanVideo-T2V-13B，50 inference steps、6% warm-up。Wan 720p 约 76K tokens 下，VecAttention 在 52.3% sparsity 取得 PSNR 19.7 / SSIM 0.668 / LPIPS 0.339；HunyuanVideo 约 119K tokens 下，在 62.1% sparsity 取得 PSNR 22.8 / SSIM 0.779 / LPIPS 0.330。整体摘要结果是 attention computation 最高 2.65× 加速，相对已有 sparse attention 最高 1.83×。

## 稀疏思路

### Vertical-Vector Pattern

Full attention 是：

$$
S = \frac{QK^\top}{\sqrt D}, \quad A=\text{softmax}(S), \quad O=AV
$$

已有稀疏模式常见两类：

- **Line / slash pattern**：按整条竖线、斜线或规则 stride 保留。
- **Block pattern**：按矩形块保留。

VecAttention 的观察是：视频 attention 的 oracle sparse pattern 往往更细碎。完整竖线太粗，会保留大量不重要 query-key 交互；矩形 block 也容易因为 block 内混有无关 token 而浪费计算。更合适的粒度是：

$$
A[iP_q:(i+1)P_q, j]
$$

即一个 query block 对某个 key column 的垂直向量。这里 $P_q$ 是 vector size / query pooling size。它比 block sparse 细，因为 key 维不是一整个 $B_k$ 矩形块；也比整条 vertical line 细，因为只让局部 query group 共享 key columns。

### Query Pooling

为了估计一个 query group 应该选哪些 key columns，VecAttention 对 query 做平均池化：

$$
Q_p[i] = \frac{1}{P_q}\sum_{t=iP_q}^{(i+1)P_q-1} Q[t]
$$

然后计算估计分数：

$$
S_p = Q_pK^\top/\sqrt D
$$

其中 $S_p[i,j]$ 表示第 $i$ 个 query group 对 key $j$ 这条 vertical vector 的重要性。这样估计矩阵规模从 $N \times N$ 降为 $(N/P_q) \times N$，但如果直接 materialize，长视频下仍然很贵，所以需要 TilingSelect。

### minS：排序-free 选择

TopK / TopP 都需要排序或近似排序，GPU 上有分支和非连续访存。VecAttention 改用 minS 过滤：

$$
M_i(j) = S_p(i,j) \ge m_i - \alpha,\quad m_i=\max_j S_p(i,j)
$$

也就是保留距离该 query group 最大 logit 不超过 $\alpha$ 的 key columns。代码里用户传入的是 `threshold`，会转换成：

```python
gap = -log(threshold + 1e-9)
```

Triton kernel 里判断 `(qk + gap) >= qk_max`。`threshold` 越接近 1，`gap` 越小，保留越少；`threshold` 越小，保留越多。论文显示，在同等 sparsity 下 minS 的 recall 接近 TopK/TopP，但 selection latency 明显更低。

### Per-Head Threshold Tuning

不同 layer/head 的可稀疏程度不同。VLM demo 可以直接用统一 threshold；DiT 论文结果使用 per-head DP tuning：

1. 用少量 prompt dump 每层 Q/K。
2. 对每个 head 扫描候选 threshold，计算 recall 和 sparsity。
3. 用动态规划在目标平均 sparsity 下选择每个 head 的 threshold，最大化平均 recall。

开源脚本中 `spattn/threshold/dump_qk_layers.sh` 生成 QK cache，`model_tuner.py` 用 `seg_pr_perHead` 评估每个 threshold，再用 DP 回溯输出 threshold JSON。`run_t2v_eval.py` 的 `vecattention` 模式加载该 JSON；`vecattention_wo_DP` 则使用统一 threshold。

## 代码实现

### 实现结构

VecAttention 的代码路径可以概括为：

1. **模型接入**：`customize_prefill_attention` 根据 `FastPrefillConfig.metric` 在 full / xattn / flex / anchor / vecattention 之间 dispatch。DiT 评测通过 Wan/Hunyuan attention patch 把 self-attention 替换成这个统一入口。
2. **Q pooling**：`average_vector` 用 Triton `bnhd_pool_kernel` 将 `[B,H,S,D]` 的 Q 按 `q_pooling_size` 聚合成 `[B,H,S/P_q,D]`。
3. **重要向量选择**：`fuse_qk_softmax_minp_wo_causal` 调 Triton kernel，对 pooled Q 和 full K 做 tile GEMM + minS，输出 `column_count` 和 `column_index`。
4. **稀疏 attention 执行**：`sparse_attn_func` 来自 modified `vllm-flash-attention` 子模块，接收固定 local block、selected columns 两套索引，执行 vector-sparse attention。

开源仓库可见的核心是 selection kernel；最终 CUDA attention kernel 在 `vllm_flash_attn` 扩展中暴露为 Python API。笔记里把两者分开，是因为加速来自二者配合：selection 必须足够便宜，执行 kernel 必须能直接消费离散 key indices。

### Q Pooling Kernel

`bnhd_pool_kernel` 的输入 layout 是 `[B,N,H,D]` 风格，grid 为：

$$
(B,\ \lceil N/P_q\rceil,\ \lceil H/BLOCK\_H\rceil)
$$

每个 Triton program 负责一个 batch、一个 query pooling block 和一组 heads。它加载：

$$
[BLOCK\_SIZE\_N,\ BLOCK\_SIZE\_H,\ BLOCK\_SIZE\_D]
$$

然后沿 N 维求平均。`BLOCK_SIZE_N` 就是 `q_pooling_size`，实际只允许 64 或 128；`BLOCK_SIZE_D` 取 head dim 的 next power-of-two。这个 kernel 的任务很简单，但很重要：如果 Q pooling 走 PyTorch reshape/mean，在长序列、多层、多 step 下会造成额外 kernel launch 和中间张量开销。

### TilingSelect 的 Grid 与数据流

`fuse_qk_softmax_minp_wo_causal` 会先 pad Q blocks 和 K length，然后分配：

- `column_count`: `[B,H,padded_q_num_blocks]`
- `column_index`: `[B,H,padded_q_num_blocks,padded_k_len]`

Triton selection kernel 的 grid 是：

$$
(\lceil N_q^p / BLOCK\_SIZE\_Q\rceil,\ \lceil N_k/(k\_local\_size \cdot group\_k\_block)\rceil,\ B \cdot H)
$$

三个 axis 分别是 query-block tile、key-block group、batch-head。一个 program 处理：

- `BLOCK_SIZE_Q` 个 pooled query rows。
- `group_k_block` 个 key blocks，每个 key block 大小为 `k_local_size`。
- 一个 batch-head。

在 kernel 内，`qblock` 是 `[BLOCK_SIZE_Q,D]`，`kblock` 是 `[D,BLOCK_SIZE_K]`。每次循环计算：

$$
qk = qblock \cdot kblock
$$

并维护 `qk_max[BLOCK_SIZE_Q,1]`。这个 running max 是 cross-tile 的：同一个 program 处理多个 K tiles 时，后续 tile 可以用前面 tile 的最大值做更接近全局的 minS 过滤。

### minS 输出为什么用 Atomic

每个 program 对当前 key tile 得到布尔 `qk_mask[BLOCK_SIZE_Q,BLOCK_SIZE_K]`。对每个 pooled query row：

1. `row_counts = sum(qk_mask)` 得到当前 tile 选中多少 key columns。
2. 把未选中的位置替换成 `k_length`，再对 `idx` 排序，使有效 key indices 排在前面。
3. 对 `column_count` 做 `tl.atomic_add`，拿到该 row 在 `column_index` 中的写入 offset。
4. 把有效 indices 写到 `column_index[row, offset:offset+row_counts]`。

这里 atomic add 是必要的，因为同一个 query row 的 selected columns 来自不同 key-block groups，多个 programs 会并发向同一行追加 indices。atomic 只发生在每个 row 每个 K tile 的 append 级别，而不是每个 selected element 都做一次全局协调，开销可控。

这个设计解释了论文里 selection overhead 低的原因：kernel 不把 $S_p$ 或 softmax 后的 estimated map 写回 HBM，而是在 QK tile 仍在寄存器/SRAM 中时直接过滤，只输出稀疏 indices。论文给出的 64K context、0.9 sparsity 分析中，TilingSelect+minS 的 HBM 访问从 naive minS 的 18.3GB 降到 1.8GB。

### Causal 与 Non-Causal 差异

VecAttention 同时服务 VLM prefill 和 DiT self-attention：

- VLM 是 causal attention，需要强制保留起始 sink block 和当前 local/diagonal block。
- DiT 视频生成是 non-causal attention，所有 query 可以看全序列。

`VecAttention_prefill` 里 causal 模式会预构造 `blk_count/blk_idx`：

- `SPATTN_BLOCK_SIZE_K = 64`
- 对每个 query block，保留初始 blocks 和当前 query block 附近的 local blocks
- `wo_initial=causal` 传给 selection kernel，避免重复选择 initial block

non-causal 模式下 `blk_count` 初始为 0，只依赖 minS 选出的 `col_count/col_idx`。这也是视频生成场景下 VecAttention 更像纯 vertical-vector sparse attention 的原因。

### Sparse Attention 执行 Kernel

可见 Python 调用是：

```python
sparse_attn_func(
    q_chunk.transpose(1, 2).contiguous(),
    k_chunk.transpose(1, 2).contiguous(),
    v_chunk.transpose(1, 2).contiguous(),
    q_pooling_size,
    blk_count_chunk,
    blk_idx_chunk,
    col_count,
    col_idx,
    return_softmax_lse=False,
    causal=causal,
)
```

`blk_count/blk_idx` 表示固定保留的 local blocks，`col_count/col_idx` 表示动态选出的 vertical vectors。论文 Algorithm 2 描述的 CUDA 设计是：一个 thread block 处理一个 Q tile，Q tile 大小等于 $P_q$；同一个 Q tile 内所有 query rows 共享 selected key indices。kernel 按 `col_idx` gather-load K/V rows，计算 QK，并用 FlashAttention 的 running max / running normalizer 累积：

- `m_i`：当前处理过的 selected K 中每个 query row 的最大 logit。
- `l_i`：softmax denominator。
- `acc`：weighted V accumulator。

这样既避免 materialize sparse attention matrix，也避免把未选中的 key rows 加载进片上内存。代价是 K/V 访存从连续 tile 变成 gather-load，因此 VecAttention 的收益依赖较高 sparsity；如果保留 columns 太多，离散 gather 的带宽成本会抵消细粒度稀疏的优势。

### Chunking 与长序列控制

`VecAttention_prefill` 按 `chunk_size` 切 Q，默认要求 `chunk_size % q_pooling_size == 0`。每个 chunk 只为当前 Q chunk 做 pooling 和 selection：

```python
for i in range(0, seq_len, chunk_size):
    q_chunk = query_states[:, :, i:i + chunk_size, :]
    avg_q_chunk = average_vector(q_chunk, q_pooling_size)
    col_count, col_idx = fuse_qk_softmax_minp_wo_causal(...)
    sparse_attn_func(...)
```

causal VLM 下 K/V 只取 `:i+chunk_size`；non-causal DiT 下 K/V 是全序列。DiT 评测默认 `chunk_size=32K`，`q_pooling_size=64`，`k_local_size=16`，`group_k_block=8192`。`group_k_block` 很大是因为生成模型序列本来很长，Q 维并行度已经足够，减少 K 维切分可以降低 cross-program append 和调度开销。

### DP Threshold 在 DiT 中的接入

`run_t2v_eval.py` 对 VecAttention 默认配置是：

```text
block_size_q = 64
block_size_k = 16
group_k_block = 8192
chunk_size = 32K
```

如果使用 `vecattention` 而不是 `vecattention_wo_DP`，脚本会加载 threshold JSON，并作为 tensor 传入 `FastPrefillConfig.threshold`。`customize_prefill_attention` 根据当前 `layer_idx` 取出该层所有 heads 的 threshold；selection kernel 的 per-head variant `_causal_fuse_qk_cutoff_wo_causal_perHead_kernel` 再按 `head_id` 读取对应 `gap`。

这与 SVOO 的 CSV schedule 类似，都是把离线 profile 信息变成在线 head 级预算；区别是 VecAttention 的预算控制变量是 minS gap/threshold，而不是 key cluster 保留比例。

## 实验结果

视频生成主表：

| 模型 | 方法 | Sparsity | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: |
| Wan2.1-T2V-14B 720P | XAttention | 54.6% | 19.7 | 0.658 | 0.348 |
| Wan2.1-T2V-14B 720P | SVG | 52.2% | 18.7 | 0.639 | 0.381 |
| Wan2.1-T2V-14B 720P | VecAttention | 52.3% | 19.7 | 0.668 | 0.339 |
| HunyuanVideo-T2V-13B 720P | XAttention | 60.8% | 21.2 | 0.734 | 0.348 |
| HunyuanVideo-T2V-13B 720P | SVG | 60.1% | 21.8 | 0.769 | 0.326 |
| HunyuanVideo-T2V-13B 720P | VecAttention | 62.1% | 22.8 | 0.779 | 0.330 |

视频理解上，VecAttention 在 Qwen2.5-VL-7B 约 26K tokens 下用 78.5% 平均 sparsity 基本匹配 full attention 平均准确率；InternVL-3.5-8B 约 17K tokens 下也比 FlexPrefill、XAttention、AnchorAttention 保持更好精度。

延迟上，论文报告 attention computation 最高 2.65× 加速、端到端 TTFT 1.17×。注意这个端到端数字来自 VLM TTFT 微基准；视频生成表主要报告质量/sparsity，不像 SVOO 那样给出完整端到端 generation latency 表。

## 关键启示

- **更细粒度不等于更慢**：如果 selection 不 materialize 估计 attention map，vertical-vector 这种细粒度稀疏可以比 block sparse 更接近 oracle。
- **稀疏单元要匹配注意力形状**：视频 attention 的重要区域常是碎片化 vertical segments，用大 block 会引入块内冗余。
- **selection kernel 是成败关键**：TilingSelect 把 QK tile 计算、running max、minS filter 和 index append 融在一个 Triton kernel 里，避免 selection overhead 吃掉 sparse attention 收益。
- **执行 kernel 要支持离散 gather**：VecAttention 的动态 columns 不能只靠普通 block sparse kernel，需要 sparse attention kernel 直接消费 `col_count/col_idx` 并做 gather-load K/V。
- **生成模型需要 per-head threshold**：DiT 上论文结果依赖 DP-tuned thresholds；统一 threshold 能跑通，但通常不是最佳质量-稀疏折中。
