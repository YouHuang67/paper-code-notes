---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
  - Triton
---

# VSA: Faster Video Diffusion with Trainable Sparse Attention

[arXiv 2505.13389](https://arxiv.org/abs/2505.13389) | [代码分析](../code_analysis/fastvideo_vsa/00_overview.md) | 当前官方代码入口：`refs/codes/FastVideo`

## 结论先行

VSA 是一篇面向视频扩散模型的 **trainable sparse attention** 工作。它不是把 dense attention 后处理成稀疏，而是把 attention 拆成：

- **压缩分支**：对时空 tile 做 coarse attention，覆盖全局长程信息；
- **稀疏分支**：在压缩分支给出的 block 级候选上做精细 Top-K sparse attention；
- **融合门控**：把 coarse output 与 sparse output 重新组合，让模型在训练中适应这套稀疏访问机制。

这篇工作的关键点不只是“保留 Top-K block”，而是 **让视频模型从训练时就学会依赖 coarse-to-sparse 两级记忆访问**。因此它和 `PISA/PASA/SVG2` 这类 training-free 推理稀疏化有本质不同。

从 `FastVideo` 当前实现看，VSA 已经不是孤立 demo，而是被做成了框架内正式 attention backend：

- Python 框架层：负责 token tile 重排、metadata 构建、训练/推理接入；
- `fastvideo-kernel` 层：负责压缩 + Top-K mask 的 Triton fused kernel，以及 64/256 tile volume 下的 block-sparse backend 路由；
- 后端层：支持 `Triton`、`ThunderKittens(sm_90)`、`FA4 CuTe DSL(sm_100, VSA-256)` 三条执行路径。

这也解释了为什么 FastVideo 官方文档已经把 `VSA finetune` 列为正式训练方法，而不是实验特性。

## 1. 动机

视频扩散模型的 attention 成本随 token 数快速爆炸。视频 token 比图像多出一个时间维，而且高分辨率、长时长、更多帧数都会把序列长度推高。标准 full attention 有两个直接问题：

- **计算量是二次的**：`O(N^2 d)`；
- **真正重要的远程交互并不均匀**：大量时空 token 只需要粗粒度全局信息，没必要都做 token 级精确注意力。

VSA 的核心判断是：

- 远距离全局依赖可以先在 **压缩块空间** 中大致定位；
- 真正需要精算的只是少数 block；
- 但模型必须在训练中适应这种稀疏访问，不然 inference-only 稀疏替换会带来明显质量退化。

所以 VSA 不是单纯为了推理补 kernel，而是直接把 **coarse global retrieval + fine sparse refinement** 做成可训练注意力结构。

## 2. 方法主线

### 2.1 时空 tile 化

VSA 先把视频 latent token 组织成规则时空 tile。`FastVideo` 当前默认 tile size 是：

$$
(t_s, h_s, w_s) = (4, 4, 4)
$$

因此一个 tile 的 token 数是：

$$
B = 4 \times 4 \times 4 = 64
$$

如果换成更大的 tile，例如 `(4, 8, 8)`，则 block volume 变成：

$$
B = 256
$$

这个 tile volume 不只是论文超参，而是直接决定后端执行路径：

- `B=64`：走现有 64-token block sparse 路径；
- `B=256`：走 VSA-256 路径，可路由到 Triton route-A 或 Blackwell 上的 FA4 CuTe block-sparse fastpath。

### 2.2 压缩分支

对每个 tile 内 token 做 block mean，得到压缩后的 block 表示：

$$
q_c^{(i)} = \frac{1}{|B_i^q|} \sum_{u \in B_i^q} q_u,\quad
k_c^{(j)} = \frac{1}{|B_j^k|} \sum_{v \in B_j^k} k_v,\quad
v_c^{(j)} = \frac{1}{|B_j^k|} \sum_{v \in B_j^k} v_v
$$

这里特别要注意，视频边界 tile 可能是不完整块，所以实际除数不是固定 `64` 或 `256`，而是 **variable block size**。

压缩分支在 block 空间执行 dense attention：

$$
S_{ij} = \frac{q_c^{(i)} {k_c^{(j)}}^\top}{\sqrt d}
$$

$$
A_{ij} = \operatorname{Softmax}(S_{ij})
$$

$$
o_c^{(i)} = \sum_j A_{ij} v_c^{(j)}
$$

之后再把 `o_c^{(i)}` 广播回这个 query block 内的所有 token。

这条分支的作用不是高精度建模，而是：

- 提供全局 coarse 上下文；
- 同时提供 block-level score，用于后续 Top-K sparse selection。

### 2.3 稀疏分支

压缩分支得到 `S_{ij}` 后，对每个 query block 选 Top-K key block，构造 block 稀疏图：

$$
\mathcal{N}(i) = \operatorname{TopK}_j \; S_{ij}
$$

然后只对这些候选 block 执行 token 级精确 sparse attention：

$$
o_s^{(u)} =
\operatorname{Softmax}
\left(
\frac{q_u K_{\mathcal{N}(i)}^\top}{\sqrt d}
\right)
V_{\mathcal{N}(i)},
\quad u \in B_i^q
$$

因此，VSA 的精确计算预算不是按 token 逐个挑 key，而是先在 block 空间检索，再在选中的 block 内做 full token interaction。

### 2.4 融合门控

FastVideo 当前实现里，最终输出不是只用 sparse 分支，而是：

$$
o = o_s + g \odot o_c
$$

或等价地，把 `o_c` 乘一个 `compress_attn_weight / gate_compress` 再加回 sparse 输出。

这一步很关键。它说明 VSA 不是“压缩分支只做 selector”，而是把压缩分支当作真正的低频全局信息通道。模型训练过程中会学会如何利用这条分支补充 sparse branch 的长程信息损失。

## 3. 数学与复杂度直觉

设 token 数为 `N`，block size 为 `B`，block 数为 `T=N/B`，每个 query block 只访问 `K` 个 key block。

那么：

- 压缩分支复杂度：`O(T^2 d) = O((N/B)^2 d)`
- 稀疏分支复杂度：`O(N K B d)`

当 `B` 足够大、`K << T` 时，总复杂度显著低于 full attention 的 `O(N^2 d)`。

VSA 和简单 block-sparse 方法的区别在于：它不是直接在 block-level score 上截断后只保留 sparse branch，而是保留了一个全局 coarse path。这让它在高稀疏率下更稳。

## 4. FastVideo 实现结构

当前官方实现不是把 VSA 写成一两个函数，而是四层结构：

### 4.1 框架层：`VideoSparseAttentionBackend`

入口在 `fastvideo/attention/backends/video_sparse_attn.py`。

这一层做的事情是：

- 根据视频 latent shape 构建 tile metadata；
- 把 raster order token 重排成 tile-contiguous layout；
- 计算 `variable_block_sizes`、`non_pad_index`、`untile_combined_index`；
- 在 forward 中调用 `fastvideo_kernel.video_sparse_attn` 或 `video_sparse_attn_bshd`；
- 再把输出从 tile layout 还原回原始 token 顺序。

这层的重点不是算 attention，而是把 **视频时空 token 拓扑** 转成 kernel 喜欢的 padded block layout。

### 4.2 算子封装层：`fastvideo_kernel.ops.video_sparse_attn`

入口在 `fastvideo-kernel/python/fastvideo_kernel/ops.py`。

这一层把 VSA 明确拆成三段：

1. `fused_block_mean(q/k/v)` 生成压缩表示；
2. `scores = q_c @ k_c^T / sqrt(d)`，再用 `fused_topk_mask(scores, topk)` 构造 block mask；
3. 对 mask 调 `block_sparse_attn` 或 `block_sparse_attn_256` 做精细 sparse branch。

最后把 `out_c` 和 `out_s` 用门控重新相加。

### 4.3 稀疏分支后端层

`block_sparse_attn.py` 负责 64-token block 路径：

- 默认优先 `sm_90` 上的 `ThunderKittens` C++ kernel；
- 否则回退到 Triton index-native sparse attention；
- 支持 autograd forward/backward。

`block_sparse_attn_256.py` 负责 256-token block 路径：

- 默认仍是 Triton route-A；
- 在 `FASTVIDEO_VSA_CUTEDSL=1` 且依赖齐全时，走 `FlashAttention-4 CuTe DSL` block sparse fastpath；
- 256 logical block 会展开到 128-token 或 64-token 的物理表示。

### 4.4 Metadata 工具层

`vsa_utils.py` / backend builder 负责：

- `tile_partition_indices`
- `reverse_tile_partition_indices`
- `variable_block_sizes`
- `non_pad_index`

这部分是 VSA 真正落地到视频 token 的关键。没有这层，论文里的“3D 时空 tile”就只是概念，无法变成内核可执行的连续 block。

## 5. 关键实现细节

### 5.1 先 tile，再 pad，再 sparse

VSA 不是直接对 raster order token 做 block sparse。FastVideo 先把 `(T,H,W)` 网格上的相邻 token 重排到连续 tile 中，再对每个 tile 进行 padding。

这样做的原因很直接：

- block 的语义变成真实的时空局部块；
- block mean 才有明确几何含义；
- 稀疏分支的 block 访问也更符合视频局部性。

如果不先 tile，而直接对光栅顺序 block 化，那么一个 block 可能跨多个空间区域甚至跨帧，压缩分支的 coarse score 会失真。

### 5.2 Variable block size 是一等公民

视频边界 tile 往往是不满的，例如最后几个 frame 或空间边界不足 `4x4x4`。FastVideo 没有粗暴把这些 pad token 当成真实 token 参与平均，而是显式维护：

$$
\text{variable\_block\_sizes}[j] = |B_j|
$$

这同时影响：

- 压缩分支 block mean 的除数；
- sparse branch 中每个 block 的真实有效长度；
- 从 padded layout 映射回真实 token 的 `non_pad_index`。

这一步虽然看似工程细节，但对质量和数值稳定性都重要。

### 5.3 压缩和 Top-K 不是两串 PyTorch 小算子，而是 fused Triton

`fused_compress_topk.py` 做了两件事：

- `fused_block_mean`
- `fused_topk_mask`

其目标非常明确：避免 Python 层 `.view() -> sum() -> div()` 和 `torch.topk() -> scatter_()` 带来的多次 launch 与中间张量物化。

对于视频扩散，attention 层很多，denoising step 也多。如果 coarse selector 每层都由一串小 op 组成，最后 overhead 会相当显著。

### 5.4 64 与 256 两条路不是同一个 kernel 换个超参

当前实现把 block volume 分成两个 regime：

- `64 = 4x4x4`
- `256 = 4x8x8`

`64` 更像当前稳态默认路径，后端成熟：

- Triton fallback
- Hopper `sm_90a` 上可走 TK C++ kernel

`256` 更像为更大 tile / Blackwell fastpath 准备的分支：

- 默认用 Triton route-A，把 logical 256-block 展开成多个 64-block；
- 如果启用 CuTe DSL，则把 logical 256-block 改写成 FA4 兼容的物理 block sparse 表示。

因此 `256` 的重点不只是“大 block 更省选择开销”，而是 **为新硬件上的更强 block sparse kernel 适配物理布局**。

## 6. Kernel 设计要点

### 6.1 `fused_block_mean` 的程序布局

`_fused_block_mean_kernel` 是一个很典型的 Triton 二维 grid：

- `program_id(0) = block_idx`
- `program_id(1) = bh_idx`

也就是一个 program 负责：

- 某个 `(batch, head)` 对；
- 某个 query 或 key/value block；
- 对这个 block 的 `BLOCK_ELEMENTS x HEAD_DIM` 子矩阵做 reduction。

它的核心设计很清晰：

- 沿 token 维一次性加载整个 block；
- 在寄存器 / fp32 accumulator 中 `tl.sum(axis=0)`；
- 再除以 `variable_block_size`；
- 输出一个 `[HEAD_DIM]` 向量。

这不是 Tensor Core kernel，而是一个典型的 reduction kernel，所以性能关键不在 MMA，而在：

- 每个 program 的寄存器占用是否可控；
- `BLOCK_ELEMENTS x HEAD_DIM` 读取是否连续；
- `B*H*num_blocks` 是否足够大，能提供足够 program 数填满 SM。

它的好处是负载非常规整。每个 `(block, bh)` 的工作量几乎一致，因此几乎没有 row imbalance。

### 6.2 `fused_topk_mask` 的程序布局

`_fused_topk_mask_kernel` 的 grid 同样是：

- `program_id(0) = q_idx`
- `program_id(1) = bh_idx`

也就是一个 program 处理一个 `(batch, head, q_block)` 行。

这个 program 会：

1. 把这一整行 `kv_blocks` 分数加载到寄存器；
2. 用二分搜索近似找出第 `k` 大阈值；
3. 用 `>` 和 `==` + `cumsum` 处理 ties；
4. 直接写出 bool mask。

这里最关键的不是“Top-K 算法多高级”，而是它有一个非常现实的 GPU 约束：

- 一整行 `kv_blocks` 都要被 program 放进寄存器逻辑里处理；
- 寄存器数组过长会导致 spilling；
- 所以实现里明确设了 `MAX_KV_BLOCK_SIZE = 4096`，再大就回退到 PyTorch `topk`。

这就是非常典型的 kernel-aware 设计：算法上完全可以继续写，但硬件上不值得。

### 6.3 64-block sparse branch 的后端选择

`block_sparse_attn.py` 的 public API 先把 bool block map 压缩成 index-native 表示：

- `q2k_idx`
- `q2k_num`

然后根据硬件与环境变量选择后端：

- `FASTVIDEO_VSA_TRITON=1`：强制 Triton
- `FASTVIDEO_VSA_TK=1`：若 `sm_90` 且扩展可用，则优先 TK
- 默认：`sm_90` 有 TK 就用 TK，否则 Triton

这个路由非常重要。说明官方并没有把某个后端绝对化，而是明确承认：

- Hopper 上可以吃更强的专用 C++ kernel；
- 其他 GPU 则靠 Triton 保底；
- 同一套 VSA 上层逻辑不变，底层 sparse backend 可替换。

### 6.4 256-block path 的 route-A 思想

`block_sparse_attn_256.py` 默认并不是直接写一个完整的 256-token Triton kernel，而是：

- 把 logical 256-block 拆成多个 64-token 物理块；
- 把 logical block map 也沿 Q/K 维重复展开；
- 然后复用现有 64-block Triton sparse kernel。

这条 route-A 的核心价值不是最优，而是 **最小化新 kernel 开发量**：

- 上层算法已经支持更大 tile；
- 后端没有现成 256 sparse kernel 时，仍能运行；
- 真正高性能的 256 fastpath 交给可选的 FA4 CuTe block-sparse 实现。

这是很典型的工程取舍：先让算法路径可用，再用新硬件专用 kernel 吃性能红利。

## 7. 训练与框架定位

FastVideo 文档里已经把 `VSA finetune` 列为正式训练方法，说明这篇论文在官方项目中的角色不是“一个推理 benchmark”，而是：

- 可用于 finetune / post-training；
- 可作为框架 attention backend；
- 与推理优化、量化、蒸馏等其他系统组件组合使用。

这很符合 VSA 的论文定位：它强调的是 **trainable sparse attention**，不是 inference-only patch。

## 8. 实现与实验现象的对应关系

从实现链路看，VSA 的优势主要来自三点：

1. **全局信息不完全丢失**：
   coarse branch 仍保留了压缩全局上下文，因此高稀疏率下比纯 Top-K block sparse 稳定。
2. **真实 token 级计算只花在候选块上**：
   expensive attention 被限制在少数 block 内。
3. **selector overhead 被专门优化过**：
   `fused_block_mean + fused_topk_mask` 避免 coarse selector 自己变成瓶颈。

而它的限制也同样直接：

1. coarse selector 仍然是 block-level dense attention，因此 block 数过大时会有额外成本；
2. 目前高性能后端强依赖硬件条件，尤其 `TK sm_90` 与 `CuTe sm_100`；
3. 256 path 的默认 Triton route-A 更像兼容路径，不一定是最终最优实现。

## 9. 关键启示

- **训练型 sparse attention 和 training-free sparse patch 是两种不同范式**：VSA 的价值在于模型本身学会了依赖 coarse+sparse 双通道，而不是把 dense 模型硬切 sparse。
- **视频 sparse attention 真正落地必须绑定 3D token 组织**：tile partition、variable block size、untile index 都是算法的一部分。
- **selector 不是免费午餐**：如果 coarse block scoring 与 Top-K 构造没有 fused kernel，稀疏 attention 本身省下的算力很容易被 selector 吃回去。
- **后端分层是工业实现的关键**：同一篇论文方法，在 `FastVideo` 里对应的是 Python backend + Triton fused selector + sparse kernel dispatch + optional TK/CuTe fastpath，而不是单一手写 CUDA 文件。

## 10. 下一步阅读

- [FastVideo VSA 代码分析：总览](../code_analysis/fastvideo_vsa/00_overview.md)
- [Sparse Forcing](sparse_forcing.md)
- [PISA](pisa.md)
- [训练型 Sparse Attention 与哈希 Top-K 调研](sparse_attention_training_hash_survey.md)
