# FastVideo VSA：Triton Coarse Selector

这一页专门解释 VSA 的 coarse stage 是如何被写成 Triton fused kernel 的。重点不是“它会均值池化和 top-k”，而是：

- 为什么要 fused；
- fused 到了什么程度；
- 哪些地方刻意没有再继续 fuse；
- 这些决定如何映射回论文 §2.4 的 kernel 讨论。

核心源码：

- [`ops.video_sparse_attn`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/ops.py#L69-L143)
- [`fused_compress_topk.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/fused_compress_topk.py#L1-L334)
- [`test_fused_compress_topk.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/tests/test_fused_compress_topk.py#L1-L460)

## 1. `ops.video_sparse_attn` 明确把 coarse branch 与 sparse branch 分开

入口非常直接：

```python
q_c = fused_block_mean(q, q_variable_block_sizes, block_elements)
k_c = fused_block_mean(k, variable_block_sizes, block_elements)
v_c = fused_block_mean(v, variable_block_sizes, block_elements)

scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (dim ** 0.5)
attn = torch.softmax(scores, dim=-1)
out_c = torch.matmul(attn, v_c)

mask = fused_topk_mask(scores, topk)
```

**源码位置**: [`video_sparse_attn`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/ops.py#L122-L139)

这里有一个容易误解的点：

- **coarse branch 没有把 `q_c @ k_c^T` 和 softmax 也写进 Triton kernel**
- 真正 fused 的只有：
  - block mean
  - top-k mask 构造

原因和论文正文一致：

- `q_c @ k_c^T` 的规模已经缩小了 `64x`
- 这部分 FLOPs 很便宜
- 真正值得优化的是大量小 kernel、临时张量、`torch.topk` 和 `scatter_` 的 launch/访存开销

也就是说，FastVideo 选择了“**融合最贵的杂碎环节**”，而不是为了追求形式上的“全流程单 kernel”去重写整个 coarse attention。

## 2. `fused_block_mean`：一块一 program 的规整 reduction

### 2.1 数学对应

它实现的正是论文 coarse pooling：

$$
q_c^{(i)}=\frac{1}{|B_i|}\sum_{u\in B_i} q_u
$$

`k_c` 与 `v_c` 同理。

### 2.2 程序布局

`_fused_block_mean_kernel` 的 grid 是：

```text
program_id(0) = block_idx
program_id(1) = bh_idx
```

也就是一个 Triton program 负责：

- 一个 `(batch, head)`；
- 一个 block；
- 对这个 block 的 `[BLOCK_ELEMENTS, HEAD_DIM]` 子矩阵做 reduction。

**源码位置**: [`_fused_block_mean_kernel`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/fused_compress_topk.py#L22-L59)

核心逻辑非常直接：

```python
block_data = tl.load(...).to(tl.float32)
acc = tl.sum(block_data, axis=0) / vbs
tl.store(out_base, acc.to(OUTPUT_DTYPE))
```

它的设计重点不在“算法聪明”，而在三个 GPU-aware 选择：

- **bf16 读，fp32 累加**：保证块均值稳定；
- **一 program 一块**：负载极规整；
- **2D 连续 load/store**：规避 Python eager 中间 view/sum/div 的多次 launch。

### 2.3 backward 也被显式实现

`_fused_block_mean_bwd_kernel` 不是依赖 PyTorch 自动拆解，而是手写了反向：

```python
grad_val = grad_out / vbs
grad_2d = broadcast(grad_val, [BLOCK_ELEMENTS, HEAD_DIM])
tl.store(...)
```

**源码位置**: [`_fused_block_mean_bwd_kernel`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/fused_compress_topk.py#L62-L99)

数学上就是：

$$
\frac{\partial \ell}{\partial x_{u}}
=
\frac{1}{|B_i|}
\frac{\partial \ell}{\partial x_c^{(i)}}
$$

把每个块级梯度平均广播回块内 token。

这说明开源实现不是只为 inference 临时拼个 kernel，而是把 coarse pooling 当成训练图中的正式算子来维护。

## 3. `fused_topk_mask`：不是排序网络，而是阈值求解器

### 3.1 为什么不直接 `torch.topk`

原始 eager 写法大概是：

```python
topk_idx = torch.topk(scores, topk, dim=-1).indices
mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)
```

这有两个问题：

- 会产生额外中间张量；
- 每层每步都要跑 `topk + scatter`，launch 很碎。

所以 FastVideo 把它换成了行级 Triton kernel。

### 3.2 一行一个 program

`_fused_topk_mask_kernel` 的 grid：

```text
program_id(0) = q_idx
program_id(1) = bh_idx
```

一个 program 处理一个 `(batch, head, q_block)` 的整行 `kv_blocks` 分数。

**源码位置**: [`_fused_topk_mask_kernel`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/fused_compress_topk.py#L203-L269)

### 3.3 它不是完整排序，而是二分阈值

核心思想是找一个阈值 `T`，满足：

- `count(scores > T) <= topk`
- `count(scores >= T) >= topk`

实现上用 32 次 fp32 bisection：

```python
for _i in range(32):
    mid = (lo + hi) * 0.5
    count_ge = ...
    lo = tl.where(count_ge >= topk, mid, lo)
    hi = tl.where(count_ge >= topk, hi, mid)
```

然后再用：

- `above_threshold`
- `at_threshold`
- `cumsum`

精确处理 ties，确保每行 **恰好** 选出 `topk` 个 block。

这套逻辑的好处是：

- 不必构造完整排序网络；
- 对 VSA 这种 `scores=q_c@k_c^T/sqrt(d)`、数值范围有限的分数矩阵足够稳定；
- ties 可以被明确控制，而不是依赖库内部不透明行为。

### 3.4 ties 被当成一等问题处理

这不是理论上的小心谨慎，而是实际 reviewer case 驱动的工程问题。测试里专门覆盖了：

- 所有分数相等；
- 边界 ties；
- 一行中多个相同最大值但 `topk=1`。

**源码位置**:

- [`TestFusedTopkMaskTies`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/tests/test_fused_compress_topk.py#L16-L75)

这说明 `fused_topk_mask` 的设计目标不是“近似对就行”，而是 **保证 row-count 精确正确**。

## 4. `MAX_KV_BLOCK_SIZE = 4096` 是很典型的 GPU 边界

源码里明确写了：

```python
MAX_KV_BLOCK_SIZE = 4096
```

超过这个长度时直接 fallback 到 `torch.topk`。

**源码位置**: [`MAX_KV_BLOCK_SIZE` 与 fallback](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/fused_compress_topk.py#L272-L334)

原因很现实：

- 这个 kernel 需要把整行 `kv_blocks` 分数装进寄存器逻辑；
- 同时还要维护 `valid_mask`、`above_threshold`、`at_threshold`、`cumsum` 等数组；
- 超过一定长度后会寄存器溢出或严重 spill。

所以这不是“算法上不能继续做”，而是：

- 从硬件角度不值得；
- 先承认边界，再在边界内做最好。

这是 VSA 实现里非常典型的系统风格。

## 5. coarse branch 的输出如何回到 token 维

coarse 分支算完后：

```python
out_c = out_c.view(batch, heads, q_num_blocks, 1, dim)
out_c = out_c.repeat(1, 1, 1, block_elements, 1).view(batch, heads, q_seq_len, dim)
```

**源码位置**: [`out_c` 广播](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/ops.py#L127-L131)

这和论文 coarse output broadcasting 完全一致：每个 query block 的 `O_c^{(i)}` 被广播回 block 内的所有 token。

也因此 `gate_compress` 的形状必须是 token 级 `[B,H,S,D]`，而不是 block 级。模型侧的 `to_gate_compress` 也是按 token 投影的。

## 6. 为什么官方没有把 coarse attention 全部写成一版 FlashAttention 变体

论文里已经给了原则：coarse stage 的 FLOPs 和显存占比很小，真正显著的是 Top-K 与 index conversion 的 overhead。

从实现上看，这个判断被完整继承了：

- **均值池化** fused 了；
- **Top-K mask 构造** fused 了；
- `q_c @ k_c^T` 和 `softmax` 保持 PyTorch matmul；
- 没有为了“更炫技”去重写一版支持 in-kernel Top-K 的 FlashAttention。

这是一个很成熟的工程判断：

- 大头优化掉；
- 小头不做过度内核化；
- 代码复杂度、可维护性和收益之间取平衡。

## 7. 这一层与论文 §2.4 的精确对照

论文 §2.4 的说法可以压缩成两句：

- coarse stage 不能直接套 FA，因为要 materialize 行级 Top-K；
- 但 coarse stage 足够小，没必要为此大改 FA 内核。

FastVideo 当前实现就是这个结论的直接代码化：

- 用 Triton fused kernel 清理 coarse selector 的碎算子开销；
- 把真正的 heavy lifting 留给 sparse fine kernel。

## 8. 小结

VSA 的 coarse selector 值得深入看的原因，不是它用了 Triton，而是它非常清楚地回答了一个问题：

“哪些环节值得专门写 kernel，哪些环节不值得？”

FastVideo 的答案是：

- `block mean` 值得写 Triton
- `topk mask` 值得写 Triton
- `q_c @ k_c^T` 没必要硬写

这恰好体现了 VSA 作为一篇系统论文最强的地方：它不只是提出了方法，也知道真正的 runtime 瓶颈在哪里。
