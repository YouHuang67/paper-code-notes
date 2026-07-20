# FastVideo VSA：框架接入、Tile Metadata 与门控

本文只解释 VSA 在 FastVideo 框架里的接入路径，不展开 Triton/CUDA kernel 细节。核心问题有三个：

- 论文里的 `(4,4,4)` cube 在代码里怎么变成真实张量布局；
- 为什么 VSA 需要 `tile_partition_indices / variable_block_sizes / non_pad_index` 这一整套 metadata；
- 论文里的 coarse/fine/gate，当前实现具体保留了哪些、省掉了哪些。

核心源码：

- [`wanvideo.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/models/dits/wanvideo.py#L456-L585)
- [`layer.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/attention/layer.py#L171-L230)
- [`video_sparse_attn.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/attention/backends/video_sparse_attn.py#L25-L324)

## 1. 模型层不是只投影 Q/K/V，还额外投影 `gate_compress`

在 `WanTransformerBlock_VSA` 里，VSA 版 self-attention 比普通 block 多了一条线性层：

- `to_q`
- `to_k`
- `to_v`
- `to_gate_compress`

**源码位置**: [`WanTransformerBlock_VSA.__init__`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/models/dits/wanvideo.py#L472-L478)

forward 时它和 QKV 一起从 `norm_hidden_states` 上投影出来：

```python
query, _ = self.to_q(norm_hidden_states)
key, _ = self.to_k(norm_hidden_states)
value, _ = self.to_v(norm_hidden_states)
gate_compress, _ = self.to_gate_compress(norm_hidden_states)
```

然后全部 reshape 成 `[B, S, H, D]`：

```python
query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
gate_compress = gate_compress.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
```

这一步已经揭示了当前实现对论文的一个保守化处理：

- 代码里只有 `gate_compress`
- 没有单独的 `gate_fine`

所以 backend 最终融合的是：

$$
O = O_s + G_c \odot O_c
$$

而不是论文通式里的 `O_c \odot G_c + O_f \odot G_f`。

## 2. `DistributedAttention_VSA` 会把 `Q/K/V/G` 一起做序列并行重排

VSA 在 `DistributedAttention_VSA.forward()` 中把四个张量沿 batch 维拼起来：

```python
qkvg = torch.cat([q, k, v, gate_compress], dim=0)
```

**源码位置**: [`DistributedAttention_VSA.forward`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/attention/layer.py#L178-L229)

这样做有两个目的：

- sequence parallel 时，Q/K/V/G 共享一套 all-to-all 和 tile 逻辑；
- tile/pad 这类预处理只做一次，不必对四个张量分别调度。

接着执行：

```python
qkvg = sequence_model_parallel_all_to_all_4D(qkvg, scatter_dim=2, gather_dim=1)
qkvg = qkvg[:, :original_seq_len, :, :]
qkvg = self.attn_impl.preprocess_qkv(qkvg, ctx_attn_metadata)
q, k, v, gate_compress = qkvg.chunk(4, dim=0)
```

这意味着 VSA 的 tile 操作不发生在单个 `q/k/v` 上，而发生在已经做完 sequence parallel 汇聚后的联合 `qkvg` 张量上。

## 3. Tile metadata 是 VSA 真正的“输入格式”

论文里说“按 `(4,4,4)` cube 划分视频 latent”，落到代码里对应的是一组 metadata：

- `tile_partition_indices`
- `reverse_tile_partition_indices`
- `variable_block_sizes`
- `non_pad_index`
- `untile_combined_index`
- `tile_buf`

### 3.1 `tile_partition_indices`：从 raster order 到 tile order

`get_tile_partition_indices()` 本质是在 `(T,H,W)` 网格上按 tile 顺序遍历，然后把每个 tile 内部 token flatten 后拼起来。

**源码位置**: [`get_tile_partition_indices`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/attention/backends/video_sparse_attn.py#L31-L47)

它构造的是一个排列索引，不做数值计算。意义在于：

- 原始 raster order token 可能把一个时空邻域打散；
- tile order 会把一个 cube 内 token 变成连续片段；
- 这样 64-token 或 256-token block 才有几何意义，也才适合 block kernel。

### 3.2 `variable_block_sizes`：边界 tile 不是“全 64”

`construct_variable_block_sizes()` 根据实际 `dit_seq_shape` 和 tile 数，逐维算最后一个 tile 的真实长度，再做广播乘法：

$$
|B_{t,h,w}| = t\_size[t] \cdot h\_size[h] \cdot w\_size[w]
$$

**源码位置**: [`construct_variable_block_sizes`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/attention/backends/video_sparse_attn.py#L59-L98)

这一步解释了为什么 VSA 不能简单假设所有 block 大小都等于 `64`：

- 边界 frame 不满；
- 边界高宽不满；
- pad 只用于让 kernel 输入规则，不能参与真实均值和 softmax。

### 3.3 `non_pad_index`：规则 padded layout 与真实 token 的桥

`get_non_pad_index()` 假定每个 block 占 `max_block_size` 个槽位，然后只返回真实 token 的位置。

**源码位置**: [`get_non_pad_index`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/attention/backends/video_sparse_attn.py#L101-L111)

于是 VSA 可以同时拥有：

- 对 kernel 友好的规则 padded layout
- 对语义正确的真实 token 数

这是 VSA 能支持 variable block size 的关键工程点。

### 3.4 `untile_combined_index`：省掉两次 fancy indexing

metadata builder 里有一行很关键：

```python
untile_combined_index = non_pad_index[reverse_tile_partition_indices]
```

**源码位置**: [`VideoSparseAttentionMetadataBuilder.build`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/attention/backends/video_sparse_attn.py#L174-L209)

它把下面两步：

```python
x[:, non_pad_index][:, reverse_tile_partition_indices]
```

预先融合成一次索引，避免每层都物化中间张量。这类小优化在单层看不起眼，但 VSA 会出现在很多层、很多 denoising step 中，累计开销并不小。

## 4. `tile()` 不是 reshape，而是“按索引写入 padded buffer”

`VideoSparseAttentionImpl.tile()` 的核心逻辑是：

```python
buf[:, attn_metadata.non_pad_index] = x[:, attn_metadata.tile_partition_indices]
```

**源码位置**: [`VideoSparseAttentionImpl.tile`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/attention/backends/video_sparse_attn.py#L228-L263)

这说明 VSA 的 tile 操作不是纯 view/rearrange，而是：

1. 先按照 tile 顺序抽出真实 token；
2. 再把它们写入一个按 `max_block_size` 规则铺开的 padded buffer；
3. pad 位置保留为零。

训练和推理在这里还有一个重要差异：

- 推理时可以复用 `tile_buf`
- 训练时为了让 activation checkpointing 释放显存，`cache_tile_buf=False`

所以 VSA metadata 里把 `tile_buf` 做成了“每个 denoising step 共享，但可关闭复用”的资源。

## 5. `VSA_sparsity` 直到 backend 才变成 `topk`

框架对外暴露的是稀疏率 `VSA_sparsity`，不是直接暴露 `topk`。实际转换发生在：

```python
cur_topk = math.ceil((1 - attn_metadata.VSA_sparsity) * num_kv_blocks)
```

**源码位置**: [`_compute_cur_topk`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/attention/backends/video_sparse_attn.py#L160-L163)

这有两个工程含义：

- 同样的稀疏率，在不同分辨率/不同 block 数下会得到不同 `topk`
- 上层配置更像“预算比例”，下层再把它转成 block 个数

这比直接把 `topk` 写死在模型配置里更稳，因为 VSA 现在已经要兼容多种视频分辨率和 tile volume。

## 6. `forward()` 的真正职责是布局桥接

`VideoSparseAttentionImpl.forward()` 干的不是注意力数学，而是 **布局路由**：

- 若 `block_elements == 256` 且 `video_sparse_attn_bshd` 可用：
  - 直接走 BSHD fastpath
- 否则：
  - 转成 BHSD，再走旧路径

**源码位置**: [`VideoSparseAttentionImpl.forward`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/attention/backends/video_sparse_attn.py#L287-L324)

这说明当前实现把两个问题分得很清楚：

- **框架层负责 layout 和 metadata**
- **kernel 层只关心已经准备好的规则 block 输入**

这种分层的好处是：

- sparse kernel 不必知道原始视频 `(T,H,W)` 拓扑；
- 64 路径与 256 路径可以完全独立演化；
- `gate_compress`、`tile_buf`、sequence parallel 等系统细节不会污染 kernel。

## 7. 这一层最关键的实现判断

### 7.1 先 tile 再 sparse，而不是直接对 raster token 稀疏化

这不是纯粹的访存优化，而是方法正确性的要求。否则 coarse stage 的 block mean 就没有稳定几何含义。

### 7.2 Variable block size 被提升成一等公民

FastVideo 没有把边界 pad 当作“无所谓的小误差”，而是让：

- coarse mean
- fine mask
- output restore

都显式依赖 `variable_block_sizes`。

### 7.3 当前开源实现选择了更保守、更工程化的 gate 方案

只保留 coarse gate，等价于把 fine 分支固定成主通路。这比论文双 gate 更稳，也更接近 sparse adaptation 的实际使用方式。

## 8. 小结

框架层最重要的不是“调用了哪个 kernel”，而是它把论文中的 cube partition 变成了一份严格的数据契约：

- 输入必须先变成 tile-contiguous、block-padded layout；
- 每个 block 的真实长度必须被保留下来；
- kernel 只消费 `q/k/v/gate + variable_block_sizes + topk`；
- 输出再通过 `untile_combined_index` 回到原始 token 顺序。

理解这一层之后，再看 Triton selector 和 sparse backend，很多实现选择就会变得非常自然。
