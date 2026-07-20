---
tags:
  - Sparse Attention
  - Video Generation
  - CUDA
  - Triton
  - CUTLASS
---

# FastVideo VSA 代码分析：总览

对应论文：[VSA: Faster Video Diffusion with Trainable Sparse Attention](../../paper_reading/vsa.md)

**源码仓库**: `refs/codes/FastVideo`

**分析基线提交**: `970409962f358afd529b969a378174c849665837`

**额外核对**: 我在 `2026-07-20` 检查了该提交到 `origin/main` 的 VSA 相关文件，`video_sparse_attn.py`、`ops.py`、`fused_compress_topk.py`、`block_sparse_attn.py`、`block_sparse_attn_256.py`、`block_sparse_attn_cute_fwd.py`、`block_sparse_h100.cu` 的主逻辑没有发生改变，因此本分析与当前官方实现仍对应。

## 1. 先建立正确心智模型

FastVideo 里的 VSA 不是一个“单独的 VSA kernel”，而是一整条从模型层到 GPU backend 的执行链：

```text
Wan / FastVideo transformer block
  -> to_q / to_k / to_v / to_gate_compress
  -> DistributedAttention_VSA
  -> tile + pad + metadata
  -> video_sparse_attn(...)
       -> fused_block_mean(q/k/v)
       -> q_c @ k_c^T
       -> fused_topk_mask(scores, topk)
       -> 64-path: Triton or ThunderKittens CUDA
          256-path: Triton route-A or CuTe DSL
       -> out_s + gate_compress * out_c
  -> untile + reverse sequence parallel
```

如果只盯一个 `.cu`、`.py` 或 Triton kernel，很容易把 VSA 看扁。它真正的工程价值在于四件事被统一起来了：

- 视频 token 的 **3D tile 重排**
- 边界 tile 的 **variable block size**
- coarse selector 的 **Triton fused kernel**
- sparse branch 的 **多后端路由**

## 2. 代码结构

建议按下面四层读。

### 2.1 框架层

- `fastvideo/models/dits/wanvideo.py`
- `fastvideo/attention/layer.py`
- `fastvideo/attention/backends/video_sparse_attn.py`

职责：

- 生成 `Q/K/V/gate_compress`
- 做 sequence parallel all-to-all
- 构造 tile / untile metadata
- 把 `VSA_sparsity` 转成每步 `topk`
- 调 `fastvideo_kernel.video_sparse_attn`

### 2.2 算子封装层

- `fastvideo-kernel/python/fastvideo_kernel/ops.py`

职责：

- 统一 VSA 的 coarse branch 与 sparse branch
- 按 `block_elements in {64,256}` 分流
- 把 coarse 输出与 sparse 输出融合

### 2.3 Triton selector 层

- `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/fused_compress_topk.py`

职责：

- `fused_block_mean`
- `fused_topk_mask`

这是 VSA 的 coarse stage 在 GPU 上真正“跑得值得”的原因。

### 2.4 Sparse backend 层

- `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py`
- `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py`
- `fastvideo-kernel/csrc/attention/block_sparse_h100.cu`
- `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py`
- `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_cute_fwd.py`

职责：

- 64-token 稀疏注意力的 Triton 路径
- Hopper `sm_90a` 的 ThunderKittens CUDA 路径
- 256-token 逻辑块到 64/128 物理块的转换
- 可选 CuTe DSL / FA4 block-sparse fastpath

## 3. 这条路径里到底有哪些 DSL

VSA 相关实现里真正涉及的“非 PyTorch DSL / 低层实现”是：

- **Triton**
  - `fused_block_mean`
  - `fused_topk_mask`
  - 64-token sparse attention forward/backward
  - 256-token 的 Triton route-A fallback
- **CUDA / ThunderKittens**
  - `block_sparse_h100.cu`
  - 使用 Hopper `TMA + WGMMA`
- **CuTe DSL / FlashAttention-4 block sparsity**
  - `block_sparse_attn_cute_fwd.py`
  - 仅 256-token 路径，且是 opt-in

需要特别指出：

- **这里没有 TileLang**
- TileLang 出现在 FastVideo 其他组件或别的项目里，但 **不在 VSA 这条实现链里**

因此如果用户关心“非 PyTorch DSL”，VSA 的重点应当放在：

- Triton selector 和 Triton sparse kernel
- Hopper CUDA/TK kernel
- CuTe block-sparse tensor 描述与 `mask_mod`

而不是去找并不存在的 TileLang 版本。

## 4. 论文方法到当前代码的四个关键映射

### 4.1 论文的 cube partition 对应 metadata builder

**源码位置**:

- [`get_tile_partition_indices`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/attention/backends/video_sparse_attn.py#L31-L47)
- [`construct_variable_block_sizes`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/attention/backends/video_sparse_attn.py#L59-L98)

这两段代码把论文里的 `(4,4,4)` cube 公式落成了：

- tile-contiguous 顺序索引
- 每个 tile 的真实有效 token 数

### 4.2 论文的 coarse stage 对应 `ops.video_sparse_attn`

**源码位置**:

- [`video_sparse_attn`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/ops.py#L69-L143)

这里直接实现了：

- `q_c / k_c / v_c`
- `scores = q_c @ k_c^T / sqrt(d)`
- `out_c = softmax(scores) @ v_c`
- `mask = fused_topk_mask(scores, topk)`

### 4.3 论文的 fine stage 对应 sparse backend 分发

**源码位置**:

- [`block_sparse_attn`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py#L372-L424)
- [`block_sparse_attn_256`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py#L118-L170)

64-token 和 256-token 路径分开维护，不是一个 kernel 换超参。

### 4.4 论文的双 gate 在代码里被保守化成单 gate

**源码位置**:

- [`to_gate_compress` 与 VSA self-attn 调用](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo/models/dits/wanvideo.py#L472-L585)
- [`out_c * compress_attn_weight + out_s`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/ops.py#L141-L143)

也就是说，开源实现里只有 coarse 分支显式 gate，fine 分支系数固定为 `1`。这与论文 sparse adaptation 阶段的 `G_f = 1` 是一致的。

## 5. 推荐阅读顺序

建议按下面顺序看：

1. [框架接入、tile metadata 与门控](01_framework_and_metadata.md)
2. [Triton coarse selector：fused block mean 与 Top-K mask](02_fused_coarse_selector.md)
3. [Sparse backends：Triton / ThunderKittens CUDA / CuTe DSL](03_sparse_backends.md)

这样读的原因很简单：

- 先搞清楚数据怎么被重排、padding、还原；
- 再看 coarse selector 如何构造 block 图；
- 最后看不同后端怎么吃这张 block 图。

## 6. 先给结论

- FastVideo 的 VSA 实现已经明显偏“系统工程”，而不是论文 demo。
- 论文里的核心思想在代码里基本没有走样：tile 化、coarse dense、Top-K block sparse、coarse 残差融合都还在。
- 真正决定性能的不是某个单独 kernel，而是 **数据布局 + selector + 稀疏后端** 三者的一致性。
- 如果只想看“非 PyTorch DSL”，重点看 `fused_compress_topk.py`、`block_sparse_attn_triton.py`、`block_sparse_h100.cu`、`block_sparse_attn_cute_fwd.py`。
