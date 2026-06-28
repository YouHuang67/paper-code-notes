---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
  - Triton
---

# Rectified SpaAttn: Revisiting Attention Sparsity for Efficient Video Generation

- 论文：https://arxiv.org/abs/2511.19835
- 代码：https://github.com/BienLuky/Rectified-SpaAttn
- 团队：Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences

## 概述

Rectified SpaAttn 关注的问题不是“如何找到更稀疏的 mask”本身，而是指出高稀疏率下 sparse attention 会系统性偏离 full attention：

- **Critical token 权重被放大**：sparse softmax 只在被保留的 key blocks 上归一化，原本 full attention 中属于 critical blocks 的概率质量被重新放大。
- **Non-critical token 权重被完全丢失**：未被选中的 blocks 直接不参与计算，它们在 full attention 中残留的概率质量变成 0。

因此，普通 block sparse attention 即使选中了“最重要”的 blocks，也不等于接近 full attention。Rectified SpaAttn 的做法是：用 pooled QK 低成本估计一个 block-level 的 implicit full attention，再用它修正 sparse attention 的分配偏置。

方法包含两个核心模块：

1. **IPAR（Isolated-Pooling Attention Reallocation）**：对视觉 token 做 block pooling，但把 text tokens 隔离出来，避免文本被平均池化破坏；随后重分配 pooled video/text 权重，得到更接近 full attention 的 block-level 概率。
2. **GAPR（Gain-Aware Pooling Rectification）**：对被 sparse mask 丢弃的 non-critical blocks，只在估计收益大于 pooling 误差时才补偿，避免错误的 pooled attention 反而伤害质量。

工程上，Rectified SpaAttn 是一个 diffusers attention processor 替换模块：模型侧取 Q/K/V 后，先用 pooled Q/K 生成 one-hot block mask 和 rectification 项，再用自写 Triton block-sparse attention kernel 计算 selected blocks，最后执行：

$$
O' = O_{\text{sparse}} \cdot R + O_{\text{noncritical}}
$$

其中 $R$ 修正 critical blocks 的权重放大，$O_{\text{noncritical}}$ 用 pooled value 恢复部分被丢弃的 non-critical 信息。

实验上，Rectified SpaAttn 在 H100 PCIe 上对 HunyuanVideo、Wan2.1 T2V/I2V、Flux.1-dev 做评测。HunyuanVideo 128 frames 720p 在 88.95% sparsity 下从 2425s 降到 729s，端到端 **3.33×** 加速且 VBench 82.57；Wan2.1-I2V 在 83.91% sparsity 下 **2.08×** 加速。结合 TeaCache 后，HunyuanVideo / Wan2.1-I2V / Flux 分别达到 5.24× / 8.97× / 4.15×。

## 稀疏思路

### Sparse Softmax 的两个偏置

Full attention 对所有 keys 归一化：

$$
A_{n,m} = \frac{\exp(S_{n,m})}{\sum_j \exp(S_{n,j})}
$$

Sparse attention 只对 critical blocks $M_c$ 内的 keys 归一化：

$$
A^{spa}_{n,m} = \frac{\exp(S_{n,m}) \cdot M_{n,m}}{\sum_j \exp(S_{n,j}) \cdot M_{n,j}}
$$

如果 block 被保留，分母变小，所以 $A^{spa}_{n,m} > A_{n,m}$，critical blocks 被放大；如果 block 被剪掉，则 $A^{spa}_{n,m}=0$，non-critical blocks 的概率质量被完全丢弃。稀疏率越高，这两个偏置越严重。

Rectified SpaAttn 的关键是把 sparse attention 输出拆成两部分处理：

- 对 selected critical blocks：不要让 sparse softmax 的概率质量占满 1，而是乘以它们在 full attention 中应该占的总质量 $R$。
- 对 dropped non-critical blocks：用 pooled attention 和 pooled value 近似恢复其中一部分概率质量。

### Implicit Full Attention

显式 full attention 太贵，因此论文用 pooled Q/K 估计 block-level full attention。给定 query block $B^q_n$、key block $B^k_m$：

$$
Q^{pool}_n = \text{mean}_{i \in B^q_n}(Q_i), \quad
K^{pool}_m = \text{mean}_{j \in B^k_m}(K_j)
$$

然后计算：

$$
A^{pool} = \text{softmax}(Q^{pool}(K^{pool})^\top/\sqrt d)
$$

它不是 attention 的最终输出，而是作为 full attention 分布的低成本代理。代码中这一步对应 `Q_blocks.mean(dim=-2)`、`K_blocks.mean(dim=-2)` 和 `torch.bmm(q_bmm, k_bmm)`。

### IPAR：文本隔离和权重重分配

视频 token 在局部 block 内有较强同质性，mean pooling 近似还可以；text tokens 则不同，一个 block 内不同词的语义可能完全不一样。直接把文本也按 block 平均，会破坏 text attention sink。

IPAR 因此使用 mixed granularity：

- 视觉 tokens：按 block pool 成 $K^{pool}_v$。
- 文本 tokens：保持 token-level $K_t$，不做池化。
- 计算 $Q^{pool}$ 对 `[K^{pool}_v, K_t]` 的 softmax。
- 再把 video block 权重乘以 block size 后重归一化，使 video block-level 概率和 text token-level 概率处于一致粒度。

代码中，Hunyuan/Flux/CogVideo 路径会构造：

```python
key_pool_normal = K_blocks.mean(dim=-2)
key_pool = torch.cat((key_pool_normal, key_text), dim=-2)
```

Wan 路径的 self-attention 实现较简化，主要在纯视觉序列上做 block pooling；I2V 的 image condition 另走 dense SDPA 后和主输出相加。

### Critical Rectification

对 critical blocks，论文推导出修正因子：

$$
R_n = \sum_{m \in M_c(n)} A^{pool}_{n,m}
$$

也就是 full attention 中 selected blocks 原本应该占据的概率质量。代码里：

```python
attn_pool = probs.masked_fill(~one_hot_output_partical, 0.0)
attn_pool_sum = torch.sum(attn_pool, dim=-1)
rectified_factor_R = attn_pool_sum.repeat_interleave(block_size_M, dim=-1)
```

Triton kernel 计算得到的 `output_normal` 是 sparse softmax 后的结果；最后乘以 `rectified_factor_R`，把“被 sparse softmax 放大到总和 1 的 critical 输出”缩回到 full attention 中 critical 部分应占的质量。

### GAPR：Non-Critical 补偿的安全阀

如果直接把所有被剪掉的 blocks 都用 pooled attention 补回来，就会引入 pooling 误差。GAPR 估计两件事：

- **Gain**：补偿该 block 能恢复多少被丢失的注意力质量。
- **Error**：用 pooled Q/K 近似真实 token-level attention 会带来多少误差。

只有当 gain 大于 error 时，才给 non-critical block 做补偿。代码中的 `estimate_pr_gain` 计算：

```python
delta_q = Q_blocks - q_pools[..., None, :]
delta_k = K_blocks - k_pools[..., None, :]
err_score = err_q_sum + err_k_sum
Gain_score = IQ * JK * attention_scores.abs()
gapr_mask = Gain_score > err_score
return ~gapr_mask
```

返回的是 `~gapr_mask`，命名为 `nogapr_mask`。在主路径里它会并入 `one_hot_output_partical`，用于区分哪些 blocks 不应该进入 non-critical compensation，避免误补偿。

Non-critical 补偿项为：

```python
attn_pool_novalid = probs.masked_fill(one_hot_output_partical, 0.0)
value_pool = value.reshape(..., block_size_N, head_dim).mean(dim=-2)
rectified_noncriattention = torch.matmul(attn_pool_novalid, value_pool)
```

它本质是 block-level $A^{pool}V^{pool}$，再 repeat 到 query block 内所有 query rows。

## 代码实现

### 实现结构

Rectified SpaAttn 的实现可以分为五层：

1. **模型 processor**：`RectifiedWanT2VSpaAttnProcessor2_0`、`RectifiedWanI2VSpaAttnProcessor2_0`、`RectifiedHunyuanVideoSpaAttnProcessor2_0` 等替换 diffusers attention processor，负责 Q/K/V projection、RoPE、输出投影和 warm-up 判断。
2. **空间重排层**：脚本里使用 Jenga/Gilbert space-filling curve 将 3D latent token 重排，使局部时空邻域更容易落在相邻 block；同时生成 `block_neighbor_list`，用于强制保留物理邻居 blocks。
3. **mask 与 rectification 元数据层**：`_build_block_index_with_importance_optimized` 用 pooled QK 得到 `probs`、top blocks one-hot mask、GAPR mask。
4. **稀疏 attention kernel**：`_triton_block_sparse_attention_onehot` 按 one-hot block mask 执行 block sparse attention。
5. **量化加速层**：可选 `--use_sage`，使用 SageAttention 风格 per-block INT8 Q/K 量化和 `spasageattn_fwd_triton.py` 的 INT8 sparse kernel。

这个结构和 SVOO/VecAttention 的差别在于：它不仅把 mask 送进 sparse kernel，还在 kernel 输出后做显式 reweight + compensation。因此笔记里的“代码实现”不能只看 sparse attention kernel，必须把 pooled mask 生成和输出修正一起看。

### Mask 生成与 IPAR/GAPR 张量流

以 Wan2.1 路径为例，`_build_block_index_with_importance_optimized` 输入 `[B,H,S,D]` 的 query/key：

1. 把 token reshape 成 blocks：

```python
Q_blocks = query.reshape((B, H, -1, block_size_M, D))
K_blocks = key.reshape((B, H, -1, block_size_N, D))
```

2. 对 block 内 token 求均值：

```python
query_pool = Q_blocks.mean(dim=-2)
key_pool = K_blocks.mean(dim=-2)
```

3. 展平 batch/head 后用 `torch.bmm` 算 block-level scores：

```python
q_bmm = query_pool.reshape(B * H, NQ, D)
k_bmm = key_pool.reshape(B * H, NK, D).transpose(1, 2)
attention_scores = torch.bmm(q_bmm, k_bmm).reshape(B, H, NQ, NK) * D**-0.5
probs = torch.softmax(attention_scores, dim=-1)
```

4. 调 `estimate_pr_gain` 得到 GAPR mask。
5. 对 `probs` 排序并累积，保留到 `p_remain_rates`，同时至少保留 `top_k = select_block_num` 个 blocks。
6. 用高级索引一次性写 `one_hot_output[B,H,NQ,NK]`。
7. 把 `block_neighbor_list` 和 first-frame blocks union 进去。

`select_block_num` 来自脚本：

```python
select_block_num = int((1 - sa_drop_rate) * img_block_num)
```

也就是 sparse drop rate 越高，每个 query block 至少保留的 key blocks 越少。`p_remain_rates` 则是按 pooled probability 累计保留的阈值，默认 0.3。

### Jenga/Gilbert 重排

脚本中的 `build_multi_curve(latent_time, latent_height, latent_width, axis_order_list)` 用 Gilbert space-filling curve 生成：

- `hilbert_order`：把原始 latent tokens 重排到空间填充曲线顺序。
- `linear_to_hilbert`：输出后逆变换回原始 token 顺序。
- `block_neighbor_list`：每个 block 的 3D 邻域 one-hot mask。

在 Wan/Hunyuan forward 中，attention 前执行：

```python
hidden_states = hidden_states[:, self.hilbert_order]
rotary_emb = rotary_emb[:, :, self.hilbert_order]
```

attention 后再：

```python
hidden_states = hidden_states[:, self.linear_to_hilbert]
```

这一步不是 Rectified SpaAttn 理论的核心，但对 block sparse 性能和质量很重要：重排让连续 block 更接近真实 3D 时空邻域，`block_neighbor_list` 强制保留邻域 blocks，避免 top-k pooled attention 漏掉局部运动连续性。

### Triton Block Sparse Kernel

`_triton_block_sparse_attn_fwd_kernel_onehot` 是基础稀疏 attention kernel。输入 Q/K/V 已 reshape 成 `[B*H,S,D]`，`block_mask` 是 `[B*H,NUM_QUERY_BLOCKS,NUM_BLOCKS]`。

kernel grid：

$$
(\text{num\_query\_blocks},\ B \times H,\ 1)
$$

每个 Triton program 负责一个 `(query block, batch-head)`：

- `start_m = tl.program_id(0)`：当前 query block。
- `off_hz = tl.program_id(1)`：batch-head。
- `BLOCK_M/BLOCK_N` 通常为 128。
- `BLOCK_DMODEL` 支持 16/32/64/128。

kernel 先加载当前 query tile：

$$
Q_{tile} \in \mathbb{R}^{BLOCK_M \times D}
$$

然后在 Python 编译期循环 `for block_idx in range(NUM_BLOCKS)`，每次从 `block_mask` 读取该 key block 是否有效：

```python
is_valid_block = tl.load(mask_ptr + block_idx * stride_bn)
if is_valid_block:
    load K/V block
    qk = dot(q, k)
    online softmax update
```

online softmax 状态包括：

- `m_i`：当前 row max。
- `l_i`：softmax denominator。
- `acc`：未归一化 output accumulator。

每处理一个 selected K block，就用：

```python
m_i_new = max(m_i, max(qk))
alpha = exp2(m_i - m_i_new)
p = exp2(qk - m_i_new)
acc = acc * alpha + dot(p, v)
l_i = l_i * alpha + sum(p)
m_i = m_i_new
```

最后 `acc /= l_i` 写回。这个实现和 FlashAttention 的稳定 softmax 逻辑一致，只是 key block 遍历由 `block_mask` 控制。

这里有一个性能取舍：mask 是 one-hot bool，kernel 每个 query block 会扫描 `NUM_BLOCKS` 并跳过 false blocks。优点是接口简单，和 Python 侧 one-hot mask 构造直接对齐；缺点是当 blocks 很多且 sparsity 很高时，扫描 false blocks 本身也有开销。论文强调“based on FlashAttention2 with Triton”，开源代码实现更像一个清晰的 one-hot block sparse kernel，而不是 CSR/BSR 压缩索引 kernel。

### 输出修正的位置

Triton kernel 只负责计算 selected blocks 上的 sparse attention：

```python
output_normal = _triton_block_sparse_attention_onehot(...)
```

真正的 Rectified SpaAttn 在 kernel 后完成：

```python
output_normal = output_normal * rectified_factor_R.unsqueeze(-1) + rectified_noncriattention
```

这是实现理解的关键。如果只看 kernel，会误以为它就是普通 block sparse attention；Rectified SpaAttn 的质量提升来自 kernel 外的概率质量校正和 non-critical compensation。

### SageAttention / INT8 路径

仓库 2026/01 后加入 `--use_sage`。这条路径先调用 `per_block_int8`：

- `quant_per_block_int8_kernel` grid 为：

$$
(\lceil seq\_len / BLK\rceil,\ H,\ B)
$$

- 对 Q 使用 `BLKQ=block_size_M`，对 K 使用 `BLKK=block_size_N`。
- 每个 block 计算 abs max scale，把 Q/K 量化成 int8，并保存 per-block scale。

`_triton_block_sparse_sageattn_fwd_kernel_onehot` 与 fp16/bf16 稀疏 kernel 的 grid 相同：

$$
(\text{num\_query\_blocks},\ B \times H,\ 1)
$$

区别是 `tl.dot(q_int8, k_int8)` 后乘 `q_scale * k_scale` 还原分数，概率 `p` 转 fp16 后与 V 做累积。这样减少 QK matmul 带宽和算力开销。README 中给出的新版结果显示，结合 INT8 attention quantization 后，HunyuanVideo 6.1×、Wan2.1-T2V 5.3×、Flux 4.5×；论文主表的核心结果仍是非量化 Rectified SpaAttn 与 TeaCache 组合。

### Warm-Up 与模型差异

不同 processor 的稀疏启用条件不同：

- Wan2.1 T2V：`processor_id >= 2 and current_step >= 10` 才启用 sparse，相当于 warm up 前两层和早期 denoising steps。
- Wan2.1 I2V：`processor_id >= 2` 后启用 sparse。
- Hunyuan：processor 里没有同样的 10-step Wan T2V 条件，但脚本层可以通过参数控制模式和 TeaCache。
- I2V 的 image condition 分支会先对 `encoder_hidden_states_img` 单独走 dense SDPA，再和 sparse 主输出相加。

这说明 Rectified SpaAttn 不是一刀切地替换所有 attention。早期 step/layer 对生成质量更敏感，代码保留 dense 或 Sage dense 路径做 warm-up。

## 实验结果

### HunyuanVideo-T2V

| 方法 | VBench | Sparsity | FLOPs | Latency | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 83.16 | 0% | 612.38 PFLOPs | 2425s | 1.00× |
| SVG | 81.94 | 78.91% | 320.38 PFLOPs | 1010s | 2.43× |
| SVG2 | 81.76 | 79.26% | 314.27 PFLOPs | 986s | 2.46× |
| Jenga | 82.43 | 74.64% | 335.38 PFLOPs | 1050s | 2.31× |
| Ours | 83.13 | 79.68% | 310.27 PFLOPs | 970s | 2.50× |
| Ours | 82.57 | 88.95% | 278.11 PFLOPs | 729s | 3.33× |
| Ours+Tea | 82.53 | 78.36% | 180.91 PFLOPs | 463s | 5.24× |

### Wan2.1

| 设置 | 方法 | VBench | Sparsity | Latency | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| T2V | Dense | 84.15 | 0% | 2731s | 1.00× |
| T2V | Ours | 83.72 | 74.88% | 1624s | 1.68× |
| T2V | Ours | 83.47 | 79.71% | 1515s | 1.80× |
| T2V | Ours+Tea | 83.17 | 74.82% | 592s | 4.61× |
| I2V | Dense | 83.53 | 0% | 2754s | 1.00× |
| I2V | Ours | 83.89 | 74.69% | 1522s | 1.81× |
| I2V | Ours | 83.43 | 83.91% | 1327s | 2.08× |
| I2V | Ours+Tea | 83.87 | 74.64% | 307s | 8.97× |

### Ablation

HunyuanVideo 高稀疏设置下，Jenga baseline VR 只有 0.0585 / VBench 81.15。直接用 pooled attention 做 rectification 反而降到 VR 0.0435 / VBench 79.85，说明不处理 text pooling 和 non-critical 误差会误修正。加入 IPAR 后提升到 VR 0.0805 / VBench 81.92；再加入 GAPR 后达到 VR 0.0890 / VBench 82.57，latency 仅从 726s 到 729s，额外开销很小。

## 关键启示

- **高稀疏率的主要问题不只是漏选 block**：sparse softmax 会重分配概率质量，导致 selected blocks 被系统性放大。
- **Pooled QK 可以不只用于选块**：Rectified SpaAttn 把它作为 implicit full attention，用来修正 attention allocation。
- **文本 token 不能粗暴 block pooling**：IPAR 的 text isolation 是多模态 DiT 里很实际的细节，否则 pooled attention 会误导修正。
- **补偿 non-critical blocks 需要安全阀**：GAPR 只在 gain 大于 pooling error 时补偿，避免“补得越多越差”。
- **kernel 输出之后还有算法逻辑**：这篇的实现重点不是单纯 block sparse kernel，而是 `sparse kernel + rectified_factor_R + pooled V compensation` 的组合。
- **one-hot mask 简洁但不是最压缩的执行表达**：开源 Triton kernel 易接入多模型，但会扫描 key blocks；更极致的性能可能需要 CSR/BSR 化 mask 或更深的 FlashAttention kernel 集成。
