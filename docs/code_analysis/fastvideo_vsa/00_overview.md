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

当前分析基于 `refs/codes/FastVideo` 提交：
`970409962f358afd529b969a378174c849665837`。

## 1. 不要把它理解成“一个 VSA kernel”

FastVideo 里的 VSA 不是一个孤立 kernel，而是一条完整执行链：

1. **框架层**：把视频 token 从 raster order 重排成 tile-contiguous layout；
2. **metadata 层**：构造 `tile_partition_indices / variable_block_sizes / non_pad_index`；
3. **coarse selector 层**：用 fused Triton kernel 做 block mean 和 Top-K mask；
4. **sparse branch 层**：按 block volume 64/256 分流，再按硬件选择 `Triton / ThunderKittens / FA4 CuTe`；
5. **输出恢复层**：把 padded tile layout 还原回原始 token 顺序。

所以如果只盯着某个 `.cu` 或 `.py` 文件，很容易把实现看扁。VSA 的真正工程价值，在于它把**视频 token 拓扑、稀疏模式构造和硬件后端路由**统一起来了。

## 2. 代码结构

建议按下面四层读：

### 2.1 FastVideo 框架接入

- `fastvideo/attention/backends/video_sparse_attn.py`

职责：

- 构造 VSA metadata；
- 做 tile / untile；
- 计算当前步 `cur_topk`；
- 调 `fastvideo_kernel.video_sparse_attn` 或 `video_sparse_attn_bshd`。

### 2.2 Kernel API 封装

- `fastvideo-kernel/python/fastvideo_kernel/ops.py`

职责：

- coarse branch：`fused_block_mean`
- Top-K mask：`fused_topk_mask`
- sparse branch：`block_sparse_attn` / `block_sparse_attn_256`
- 融合输出：`out = out_s + gate * out_c`

### 2.3 稀疏 attention 后端

- `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py`
- `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py`
- `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_cute_fwd.py`
- `fastvideo-kernel/csrc/attention/block_sparse_h100.cu`

职责：

- 64-token block sparse 的 Triton / TK 路由；
- 256-token logical block 的 Triton route-A / CuTe route；
- autograd forward/backward。

### 2.4 Metadata 与测试 / benchmark

- `fastvideo-kernel/python/fastvideo_kernel/vsa_utils.py`
- `fastvideo-kernel/tests/test_vsa*.py`
- `fastvideo-kernel/tests/test_fused_compress_topk.py`
- `fastvideo-kernel/benchmarks/bench_vsa.py`

职责：

- standalone metadata 工具；
- 数值正确性验证；
- wrapper 级性能测试。

## 3. 执行主线

### 3.1 tile 重排

VSA 先把 `(T,H,W)` latent token 划分成 `(4,4,4)` 或 `(4,8,8)` 时空 tile。

`get_tile_partition_indices()` 的本质是：

- 构造原始 `T*H*W` raster order index；
- 按 tile 遍历顺序把同一个 tile 内的 token flatten 后拼接；
- 输出一个“tile 顺序排列”的索引。

这让每个 block 的 token 在内存上变成连续片段，后续 block mean 和 sparse block 访问才有局部性。

### 3.2 pad 与 variable block size

视频边界 tile 可能不完整。FastVideo 不把 pad token 混入真实统计，而是显式构造：

- `variable_block_sizes`
- `non_pad_index`

所以 padded layout 只是 kernel 视角下的规则化表示；真实有效 token 数在 metadata 里始终被保留。

### 3.3 coarse selector

`ops.video_sparse_attn()` 先对 `q/k/v` 做：

- `fused_block_mean(q)`
- `fused_block_mean(k)`
- `fused_block_mean(v)`

得到 `q_c/k_c/v_c` 后，在 block 空间做 dense attention，得到 `scores`。

随后：

- `fused_topk_mask(scores, topk)` 直接构造 block sparse mask；
- 压缩分支输出 `out_c` 被广播回 token 维。

这一步把“全局粗定位”和“全局粗上下文”统一起来了。

### 3.4 sparse branch

mask 构造好后：

- `block_elements == 64`：走 `block_sparse_attn`
- `block_elements == 256`：走 `block_sparse_attn_256`

再由后端解析是否走：

- Triton
- Hopper TK (`sm_90a`)
- Blackwell FA4 CuTe (`sm_100+`, opt-in)

### 3.5 输出恢复

输出仍在 tile-padded layout 中，需要：

- 先按 `untile_combined_index` 抽出真实 token；
- 再还原回原始 raster order。

FastVideo 预先把 `non_pad_index + reverse_tile_partition_indices` 融合成一个 fancy index，就是为了避免每层都额外分配中间张量。

## 4. `fused_block_mean` 的核心设计

`_fused_block_mean_kernel` 的 grid 是：

- `program_id(0) = block_idx`
- `program_id(1) = bh_idx`

也就是一个 Triton program 负责一个 `(batch*head, block)`。

这个设计有三个好处：

1. **负载规则**：每个 program 工作量基本相同，没有 row imbalance；
2. **访存简单**：读取一个 `BLOCK_ELEMENTS x HEAD_DIM` 的矩形 tile；
3. **数值稳定**：bf16 读入，fp32 accumulate，再除以 `vbs`，最后 cast 回输出 dtype。

它不是 Tensor Core GEMM，而是 reduction kernel，因此性能关键在：

- program 数能否填满 SM；
- 寄存器占用是否可控；
- `HEAD_DIM` 与 `BLOCK_ELEMENTS` 是否让 2D load 足够规整。

## 5. `fused_topk_mask` 的核心设计

`_fused_topk_mask_kernel` 也是一行一个 program：

- `program_id(0) = q_block`
- `program_id(1) = bh_idx`

每个 program 会把这一整行 `kv_blocks` 分数加载进来，然后做阈值搜索而不是完整排序。

关键点有两个：

### 5.1 它不是写一个排序网络，而是写一个阈值求解器

代码里用二分逼近找到阈值 `T`，满足：

- `count(scores > T) <= topk`
- `count(scores >= T) >= topk`

然后再用：

- `above_threshold`
- `at_threshold`
- `cumsum`

来精确控制 ties，保证每行恰好选出 `topk` 个 block。

### 5.2 它明确受寄存器容量约束

实现里有：

```python
MAX_KV_BLOCK_SIZE = 4096
```

原因不是算法不会写，而是：

- 这一整行分数都要放在 program 的寄存器逻辑里；
- `scores_f32 / masks / cumsum` 都会消耗寄存器；
- 超过阈值后会 spill 到 local memory，性能迅速恶化。

这正是 VSA selector 的一个很典型的 GPU-aware 设计：**先承认硬件边界，再在边界内优化。**

## 6. 64-block sparse branch：为什么先转成 index-native

`block_sparse_attn()` 的 bool mask 只是兼容接口。真正进入 kernel 前，会先变成：

- `q2k_idx`
- `q2k_num`

原因很简单：

- bool mask 适合表达语义，不适合实际 sparse kernel 执行；
- kernel 真正需要的是“每个 query block 该访问哪些 key block”；
- backward 还需要通过 `invert_indices()` 生成 `k2q_idx / k2q_num`。

也就是说，FastVideo 这里已经把 sparse attention 看成 **CSR-like row index execution**，而不是 dense mask + masked matmul。

## 7. 64-block 后端路由：Triton vs ThunderKittens

`block_sparse_attn_from_indices()` 的策略很直接：

- `FASTVIDEO_VSA_TRITON=1`：总是 Triton
- `FASTVIDEO_VSA_TK=1`：若 `sm_90` 且扩展存在，则强制 TK
- 默认：`sm_90` 能用 TK 就用，否则 Triton

这反映的是很现实的工程判断：

- Triton 提供跨设备保底实现；
- Hopper 上如果能用专门的 TK C++ kernel，吞吐和调度会更强；
- 上层 API 保持不变，后端可以按架构切换。

## 8. 256-block 路径：route-A 为什么重要

`block_sparse_attn_256.py` 默认不要求你有新 kernel。它先把：

- logical 256-token block map
- logical 256-token variable sizes

展开成更小物理 block，再复用已有后端。

### Triton route-A

对 logical `256x256` block：

- Q 维 repeat 4 次
- KV 维 repeat 4 次

于是 logical block graph 被展开为 64-token 物理 block graph，再喂给 64-path Triton sparse kernel。

### CuTe route

如果启用 `FASTVIDEO_VSA_CUTEDSL=1`，则会：

- 把 logical 256-token KV block 拆成两个 128-token 物理子块；
- 调用 `flash_attn.cute` block-sparse 前向。

这说明 256 路径真正的设计目的是：

- 上层保持更大 tile 语义；
- 下层根据可用 kernel 重写物理布局；
- 把新硬件专用 fastpath 包在统一接口里。

## 9. GPU 利用率视角下的优缺点

### 9.1 规则负载的部分

- `fused_block_mean`：一 program 一 block，规则且易填满
- 压缩分支 dense matmul：直接复用标准 GEMM 路径
- `out_c` 广播：规则写回

这些部分的 GPU 利用率主要取决于总 block 数、head 数和 batch 数，而不是稀疏模式本身。

### 9.2 不规则负载的部分

- sparse branch 的每行 Top-K block 选择
- backward 中 `k2q` 反向稀疏访问

不过 VSA 有一个重要优势：**每个 query block 的 Top-K 数是固定的**。这意味着：

- 每行非零块数相同；
- CTA / program 的循环次数更接近；
- 比“每行非零数变化很大”的稀疏模式更容易保持负载均衡。

这是 VSA 比很多动态不规则 sparse pattern 更容易写出高吞吐 kernel 的根本原因之一。

## 10. 测试与 benchmark 在说明什么

`tests/test_vsa.py` 和相关测试并不只是做 forward parity，它们还验证：

- variable block size
- q_len != kv_len
- backward correctness
- 256-path Triton / CuTe parity

这说明当前实现的关注点不是单一 demo case，而是把 VSA 当成正式 attention backend 维护。

`bench_vsa.py` 则专门 benchmark wrapper 层：

- 计时不只含 kernel，也包含 map-to-index 与 dispatch overhead；
- 这很合理，因为真实系统里你最终支付的是整条 VSA wrapper 的代价，而不是某个裸 kernel 的理想值。

## 11. 最值得记住的实现判断

VSA 在 FastVideo 里最值得记住的不是某个 kernel 名字，而是这四个判断：

1. **3D tile metadata 是算法的一部分**，不是预处理杂务；
2. **selector 必须 fused**，否则 coarse path 自己就会吃掉收益；
3. **固定 Top-K 行宽让 sparse branch 更容易高利用率**；
4. **256 block path 的关键是物理布局重写，而不是重新发明所有 kernel。**

## 12. 继续阅读

- [VSA 论文笔记](../../paper_reading/vsa.md)
- [PISA 代码实现：总览](../pisa/00_overview.md)
- [Native Sparse Attention 代码实现：总览](../native_sparse_attention/00_overview.md)
