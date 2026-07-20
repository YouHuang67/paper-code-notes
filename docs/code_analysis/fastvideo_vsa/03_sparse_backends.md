# FastVideo VSA：Sparse Backends

这一页专门讲 VSA 最有“低层味”的部分，也就是非 PyTorch 实现的 sparse backend。重点覆盖：

- 64-token 路径的 Triton sparse attention
- Hopper `sm_90a` 上的 ThunderKittens CUDA kernel
- 256-token 路径的 Triton route-A
- 256-token 路径的 CuTe DSL / FA4 fastpath

这里实际出现的 DSL 只有三类：

- Triton
- ThunderKittens/CUDA C++
- CuTe DSL

VSA 这条开源路径里没有 TileLang 内核。

核心源码：

- [`block_sparse_attn.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py#L1-L424)
- [`block_sparse_attn_triton.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py#L1-L850)
- [`block_sparse_h100.cu`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/csrc/attention/block_sparse_h100.cu)
- [`block_sparse_attn_256.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py#L1-L170)
- [`block_sparse_attn_cute_fwd.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_cute_fwd.py#L1-L266)

## 1. 先分清三层抽象

VSA 的 sparse backend 不是一个文件，而是三层抽象：

1. **逻辑稀疏图**
   - 每个 query block 连接哪些 key block
2. **索引表示**
   - `q2k_idx`
   - `q2k_num`
3. **具体后端**
   - Triton
   - ThunderKittens CUDA
   - CuTe DSL

这三层分开之后，FastVideo 才能做到：

- 上层统一输出 bool block map；
- 中间层统一压成 index-native 格式；
- 底层根据硬件和 block 大小自动切换实现。

## 2. 64-token 路径：先把 bool map 压成 index-native 表示

`block_sparse_attn()` 只是兼容接口，实际第一步会调用 `_map_to_index(block_map)`：

```python
q2k_idx, q2k_num = _map_to_index(block_map)
return block_sparse_attn_from_indices(...)
```

**源码位置**: [`block_sparse_attn`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py#L413-L424)

这一步的含义非常明确：

- bool mask 适合表达语义；
- kernel 真正需要的是“每行访问哪些块”的压缩索引；
- backward 还需要反向索引 `k2q_idx / k2q_num`。

因此 VSA 在内核执行层已经完全不是“dense mask + masked matmul”的思路，而是 **CSR-like row execution**。

更具体地说：

- `q2k_num[b,h,q_blk]` 是这一行的非零块数；
- `q2k_idx[b,h,q_blk,:]` 是这一行实际访问的 KV block 编号；
- 最后一维虽然是定长 `max_kv_blks`，但只有前 `q2k_num` 个位置有效。

这相当于把 block mask 从布尔邻接矩阵压成“每行一段索引表”。forward 直接按 `q` 行走，backward 再把它翻成按 `kv` 行走的 `k2q` 表。

## 3. 64-token 后端路由：Triton 与 TK 的角色分工

`block_sparse_attn_from_indices()` 的决策逻辑是：

- `FASTVIDEO_VSA_TRITON=1`：强制 Triton
- `FASTVIDEO_VSA_TK=1`：若 `sm_90` 且扩展存在，优先 TK
- 默认：`sm_90` 有 TK 就用 TK，否则 Triton

**源码位置**: [`block_sparse_attn_from_indices`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py#L372-L410)

这反映的不是“谁更先进”，而是一个很现实的工程策略：

- Triton 是跨设备保底实现；
- Hopper 上如果能用 ThunderKittens CUDA，就吃专用 kernel 红利；
- 上层 API 保持不变。

## 4. Triton 64-token 前向：在线 softmax 的 block-sparse 版本

### 4.1 一个 program 处理一个 Q block

`_attn_fwd_sparse` 的 program mapping：

```text
wrapper launch grid = (Tq / 64, B * H, 1)
program_id(0) = q_blk
program_id(1) = off_hz = batch * head
```

**源码位置**: [`_attn_fwd_sparse`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py#L32-L160)

每个 program 会：

- 先从 `q2k_num` 读出本行有多少个有效 KV block；
- 再从 `q2k_index` 读这行对应的 block 列表；
- 对这些 block 逐个执行 `Q_i @ K_j^T`、softmax 累积、`P @ V_j`。

这里的 `q_blk` 永远对应一个 64-token query tile。`off_hz` 再把 `(batch, head)` 融成一个轴，所以 grid 上一个 program 就是一块标准的输出 tile：`[64, D]`。

### 4.2 仍然是 FlashAttention 风格 online softmax

程序内部维护：

- `m_i`
- `l_i`
- `acc`

并按 FA 风格做增量归一化：

```python
m_ij = maximum(m_i, max(qk))
p = exp2(qk * scale - m_ij)
l_i = l_i * alpha + l_ij
acc = acc * alpha + p @ v
```

所以 VSA 的 fine sparse branch 不是“先把 sparse logits 全部攒出来再 softmax”，而是继续保留了 FlashAttention/online softmax 的数值稳定与 IO 友好性。

### 4.3 Variable block size 是在列维 mask 的

对每个选中的 KV block，代码都从 `variable_block_sizes[kv_idx]` 读真实长度：

```python
block_size = tl.load(variable_block_sizes + kv_idx)
mask = tl.arange(0, BLOCK_N) < block_size
qk = tl.where(mask[None, :], qk, -float("inf"))
```

这说明：

- 稀疏图的边描述的是“逻辑块之间有连接”
- 真正参与 softmax 的仍然只是该块里真实存在的 token

所以边界块的 pad token 永远不会混进 sparse softmax。

### 4.4 前向 grid 为什么基本均衡

这一层的 program 数量固定是 `B * H * (Tq / 64)`。单个 program 的主要工作量由它要遍历多少个 `kv` block 决定，而 VSA 的 coarse selector 会把每个 query block 的 `q2k_num` 固定到同一个 `topk`，只在 `topk > kv_blocks` 时整体截断到 `kv_blocks`。

这意味着前向的行稀疏度几乎是常数：

- 每个 `q_blk` 都做同样数量的 sparse block 迭代；
- 每次迭代的 tile 形状固定都是 `64 x 64`；
- `variable_block_sizes` 只减少最后一个边界块中的有效列数，不改变 CTA/program 的 tile 形状。

因此 Triton 前向几乎没有结构性失衡，最多只有边界块因 `block_size < 64` 少做少量有效列运算。

## 5. Triton 64-token 反向：`k2q` 反索引是关键

### 5.1 backward 不能只靠 `q2k`

为了算 `dK/dV`，需要知道“某个 KV block 被哪些 Q block 用到了”，所以在 Python wrapper 里先生成：

```python
k2q_idx, k2q_num = _invert_indices_for_backward(...)
```

**源码位置**:

- [`_invert_indices_for_backward`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py#L87-L93)
- [`block_sparse_attn_backward_triton`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py#L158-L197)

这是一个很重要的实现思想：

- forward 是按 Q 行展开
- backward 中 `dK/dV` 更适合按 KV 块展开

也就是说，执行方向会在反向里翻转一次。

### 5.2 dK/dV 与 dQ 被拆成两类 kernel

在 Triton 实现中：

- `_attn_bwd_dkdv_kernel`
  - `grid_kv = (Tkv / 64, 1, B * H)`
  - 一个 program 固定拥有一个 KV block
- `_attn_bwd_dq_kernel`
  - `grid_q = (Tq / 64, 1, B * H)`
  - 一个 program 固定拥有一个 Q block

**源码位置**: [`triton_block_sparse_attn_backward`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py#L746-L850)

这比“单个超级 backward kernel”更自然，因为：

- `dK/dV` 复用同一个 `K/V` tile
- `dQ` 复用同一个 `Q` tile

对应的稀疏邻接方向也不同。

grid 这样拆开还有一个直接后果：

- `dQ` kernel 继续沿着 `q2k` 方向工作，负载形态和 forward 很接近；
- `dK/dV` kernel 改成沿着 `k2q` 方向工作，负载不再由固定 `topk` 决定，而由每个 KV block 被多少个 query block 指向决定。

### 5.3 64-token block 被拆成两个 32-token half

反向里经常出现：

- `BLOCK_M1 = 32, BLOCK_N1 = 64`
- `BLOCK_M2 = 64, BLOCK_N2 = 32`
- `kv_blocks * 2`

原因是 backward 内部把 `64` token block 再拆成两个 `32` token half-block 来配合矩阵形状和流水方式。这个选择既影响 mask 写法，也影响 `variable_block_sizes` 的使用。

更具体地说：

- `dK/dV` 里 `q_blocks * 2` 表示每个 64-token Q block 被拆成两个 32-token 子块；
- `dQ` 里 `kv_blocks * 2` 表示每个 64-token KV block 也拆成两个 32-token 子块；
- `offs_in_block = half * 32 + arange(32)` 负责把 `variable_block_sizes[kv_idx]` 映射到 half-block 内的真实有效列。

这样做之后，反向内层 GEMM 的形状变成 `32x64` 或 `64x32`，更适合该实现里的寄存器布局与流水。

### 5.4 Triton 反向里真正出现不均衡的位置

`dQ` 侧仍然相对规整，因为每个 Q block 还是沿着自己的 `q2k` 邻接表走，`q2k_num` 基本等于固定 `topk`。

`dK/dV` 侧则不同。某个 KV block 被多少个 Q block 命中，取决于 coarse top-k 在全局图上的汇聚情况：

- 稀疏图中心位置的 KV block，`k2q_num` 可能很大；
- 冷门 KV block，`k2q_num` 可能很小，甚至为 0；
- kernel 又把每个命中的 Q block 再拆成两个 32-token half-block，所以真实循环次数是 `2 * k2q_num`。

因此 Triton 反向的主要失衡点不是 tile 大小，而是 `k2q_num` 的行间离散程度。

## 6. Hopper ThunderKittens CUDA：这才是 64-path 的高性能专用核

### 6.1 只在 `sm_90a` 编译真实 kernel

`block_sparse_h100.cu` 开头就用宏把真实 kernel 限定在 Hopper `sm_90a`：

```cpp
#if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define FASTVIDEO_TK_HOPPER 1
#else
#define FASTVIDEO_TK_HOPPER 0
#endif
```

**源码位置**: [`block_sparse_h100.cu` 文件开头](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/csrc/attention/block_sparse_h100.cu#L1-L18)

意思很明确：

- host pass 或 sm_90a device pass：编译真实 body
- 其他架构：只留空壳 stub

因为内核内部用到了 Hopper 特有的 `TMA + WGMMA`。

### 6.2 前向 kernel 的执行模型

`fwd_attend_ker` 每个 CTA 处理一个 query block，核心流程是：

1. 用 TMA 预取 `Q` block；
2. 预取第一个 `K/V` block；
3. 逐个访问 `q2k_block_sparse_index` 中列出的 KV block；
4. 用 `warpgroup::mm_ABt` 做 `QK^T`；
5. 在寄存器中维护 `max_vec / norm_vec / o_reg`；
6. 再用 `warpgroup::mma_AB` 做 `P @ V`；
7. 最后写回输出和 `l`。

**源码位置**: [`fwd_attend_ker`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/csrc/attention/block_sparse_h100.cu#L61-L280)

这其实是一个 Hopper 版 block-sparse FlashAttention：

- 稀疏性来自 `q2k_block_sparse_index`
- 数值稳定来自在线 softmax
- 访存效率来自 TMA
- GEMM 吞吐来自 WGMMA / warpgroup MMA

host 侧的 launch 也写得很直接：

```text
dim3 grid(q_seq_len / 64, qo_heads, batch)
threads = 128
dynamic_smem = 54000
```

也就是说，一个 CTA 固定对应：

- `blockIdx.x`: 一个 64-token Q block
- `blockIdx.y`: 一个 query/output head
- `blockIdx.z`: 一个 batch

而 `kv_head_idx = blockIdx.y / hr`，其中 `hr = qo_heads / kv_heads`。这就是 GQA/MQA 的 head 映射方式：多个 query head 可以共享同一个 KV head，但每个 CTA 仍然只负责一个 `(q_blk, qo_head, batch)` 输出 tile。

### 6.3 CTA 内部线程组织

forward kernel 用 `128` 线程，即 `4` 个 warp，也就是刚好 `1` 个 warpgroup。对应关系很清楚：

- 线程块级别只服务一个输出 tile；
- warpgroup 级别负责 `QK^T` 和 `PV` 的 WGMMA；
- `threadIdx.x == 0` 负责发起 TMA load/store 和 semaphore 协调；
- 在线 softmax 状态 `max_vec`、`norm_vec`、`o_reg` 常驻寄存器。

这也是这个 kernel 能把 “一块 Q 对若干稀疏 KV block 的扫描” 压缩成单 CTA 流水的关键。它没有把一个 Q block 再切成多个 CTA，所以不会有 CTA 间归约；代价是 CTA 的运行时长完全由这行稀疏邻接长度决定。

### 6.4 前向 CTA 负载为什么基本均衡

forward CUDA 路径和 Triton 路径一样，都是按 `q2k` 行展开。每个 CTA 读取：

- 一个固定大小的 `Q` tile；
- 一条 `q2k_block_sparse_index` 邻接表；
- 固定数量的稀疏迭代，通常就是同一个 `topk`。

因此 CTA 之间最大的差异只来自：

- 某些边界 KV block 的 `block_size < 64`；
- 极少数 `topk` 被全局截断到 `kv_blocks` 的情况。

在正常 VSA 设置下，forward CTA 的循环深度近似常数，所以 `grid.x * grid.y * grid.z` 上的算力分布是比较平的。

### 6.5 为什么 `block_size` 作为单独数组传入

前向里有：

```cpp
right_fill(att_block, att_block, g.block_size[q2k_block_sparse_index_ptr[kv_idx]], -inf)
```

这和 Triton 版的 `variable_block_sizes` 完全同构：每个 KV block 的真实有效列数不同，所以最后一个边界块必须按真实宽度做裁剪。

### 6.6 backward 同样走 `k2q` 邻接

`bwd_attend_ker` 中读取的是：

- `k2q_block_sparse_index`
- `k2q_block_sparse_num`

也就是说 CUDA 后端和 Triton 后端在抽象层上保持一致：

- forward 用 `q2k`
- backward 的 `dK/dV` 用 `k2q`

这说明上层 index-native 抽象设计得足够干净，换后端并不需要改方法层。

host 侧先用 `bwd_attend_prep_ker` 生成 `D = sum(O * dO)`，然后主 kernel 用：

```text
dim3 grid_bwd_2(kv_seq_len / 64, qo_heads, batch)
threads = 128
dynamic_smem = 72000
```

这时 CTA 的拥有者从 Q block 换成了 KV block：

- `blockIdx.x`: 一个 64-token KV block
- `blockIdx.y`: 一个 query/output head
- `blockIdx.z`: 一个 batch

CTA 一进来先读 `qo_blocks = *k2q_block_sparse_num_ptr`，如果 `qo_blocks <= 0` 直接返回。之后整条主循环都是沿 `k2q_block_sparse_index_ptr[qo_idx]` 扫描所有指向这个 KV block 的 Q block。

### 6.7 CUDA 反向里 CTA 不均衡出现在哪里

这个 CTA 设计避免了对 `dK/dV` 的跨 CTA 归约：一个 CTA 独占一个 KV block 的 `dK/dV` 累积，最后直接写回或 `store_add_async`。但代价也很直接：

- 热门 KV block 的 CTA 要扫描很多 `qo_blocks`；
- 冷门 KV block 的 CTA 很快结束；
- 没有命中的 KV block 会在入口直接返回。

因此 CUDA 反向最主要的失衡来源就是 `k2q_block_sparse_num` 的离散分布，而不是线程块形状本身。CTA 形状始终一样，变的是每个 CTA 要走多长的邻接表。

## 7. 256-token 路径：默认不是 CuTe，而是 Triton route-A

`block_sparse_attn_256.py` 的设计非常值得注意。它不是“256 block 上也有一套成熟自研 kernel”，而是：

- 默认仍走 Triton
- 把逻辑 256 block 展开成多个物理 64 block
- 复用现有 64-token Triton sparse kernel

### 7.1 route-A：256 逻辑块扩成 64 物理块

核心函数：

- `_expand_mask_and_sizes_256_to_64`
- `_triton_via_route_a`

做法是：

- Q 维 repeat 4 次
- KV 维 repeat 4 次
- `variable_block_sizes` 按 `[0,64,128,192]` 四个 offset 裁成四个子块长度

**源码位置**: [`_expand_mask_and_sizes_256_to_64`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py#L78-L115)

这条路线的价值不是最优，而是：

- 最小化新 kernel 开发量；
- 先保证算法路径可运行；
- 在没有可选依赖时仍然有默认实现。

这个展开不是单纯把一个逻辑边复制四次，而是复制成 `4 x 4` 的物理边子图。假设逻辑上 `Q_i -> K_j` 连通，那么 route-A 会生成：

- `Q_{i,0..3} -> K_{j,0..3}`

因此一条 256 级逻辑边会变成 16 条 64 级物理边。对应影响有两个：

- 稀疏图在物理层面变稠，`q2k_num` 会按 4 倍 Q 展开和 4 倍 KV 展开同步放大；
- 但每条物理边仍然交给已经成熟的 64x64 Triton kernel 处理，不需要再新写一套 256 block 专用稀疏 attention。

最后一个逻辑 KV block 如果真实长度不足 256，则会被拆成最多四个 `[0,64]` 的子块长度。边界处理没有丢到 kernel 之外，而是直接体现在新的 `sizes_64` 上。

## 8. CuTe DSL / FA4 fastpath：256 路径真正的专用实现

### 8.1 CuTe fastpath 是 opt-in，不是默认

只有当：

```text
FASTVIDEO_VSA_CUTEDSL=1
```

时，256 路径才会走 CuTe DSL。

**源码位置**: [`_resolve_backend`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py#L38-L50)

默认仍是 Triton route-A。

### 8.2 逻辑 256 block 先变成物理 128 KV block

CuTe 路径不是直接吃 256-token KV block，而是先拆成两个 128-token child：

```python
child0 = clamp(size, 0, 128)
child1 = clamp(size - 128, 0, 128)
```

**源码位置**: [`_expand_mask_and_sizes_256_to_128`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py#L53-L75)

这是因为当前接入的 FA4 block-sparse 前向走的是 128-token KV 稀疏块。

### 8.3 CuTe wrapper 不是写 kernel body，而是做 block-sparse tensor 适配

`block_sparse_attn_cute_fwd.py` 的真正工作是把 VSA 的：

- `block_map`
- `variable_block_sizes`

翻译成 `flash_attn.cute` 所需的：

- `BlockSparseTensorsTorch`
- `mask_mod`
- `aux_tensors`

FastVideo 在这一层并不直接暴露 CTA grid。它暴露的是：

- 稀疏 block 索引；
- full block / partial block 的划分；
- 基于 `variable_block_sizes` 的运行时谓词。

真正的 tile 调度、CTA 划分和流水策略都在 FA4/CuTe 内核内部。

#### full block 与 partial block 分开描述

它先把 KV block 分成：

- `full_map`
- `mask_map`

完整块可以直接走 full block sparse；不满块则交给 `mask_mod` 在 kernel 内裁剪。

**源码位置**: [`_cute_forward`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_cute_fwd.py#L144-L196)

#### `mask_mod` 把 `variable_block_sizes` 变成运行时谓词

`_build_vbs_mask_mod()` 用 `cute.jit` 生成一个运行时 mask：

```python
kv_blk = n_idx // block_size
kv_off = n_idx % block_size
valid = kv_sizes[kv_blk]
return (valid > 0) & (kv_off < valid)
```

**源码位置**: [`_build_vbs_mask_mod`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_cute_fwd.py#L108-L141)

这一步非常关键，因为它说明 CuTe 路径并不是“假设全块满长”，而是同样严肃支持边界块。

#### Q 侧稀疏粒度还会按硬件放大

`_choose_q_sparse_block_size()` 会在 `sm_100+` 且 `q_len > 128` 时把 Q 侧稀疏块增大到 `256`。

这表明 CuTe 路径不仅在“算 attention”，还在主动适配 Blackwell 上更合适的稀疏粒度。

## 9. 测试怎么说明这些路径是认真的

64-token 路径的前向测试不仅比较数值，还显式覆盖了：

- `S_q == S_kv`
- `S_q != S_kv`
- query 侧和 KV 侧各自独立的 `variable_block_sizes`

这说明 `q`/`kv` 长度不对称和边界块长度不一致是被当成标准场景维护的，不是偶然支持。

256 路径则有专门的三方对照测试：

- torch reference
- CuTe
- Triton route-A

**源码位置**: [`test_vsa256_forward_cross.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/tests/test_vsa256_forward_cross.py#L1-L136)

64-token 前向对照见：

- [`test_vsa_forward.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/tests/test_vsa_forward.py#L1-L220)

这意味着 256 路径不是实验性死代码，而是有明确的数值对照基线。

此外，`test_vsa.py` 也专门测了：

- variable block size
- `q_seq_len != kv_seq_len`
- forward/backward 与 PyTorch reference 的一致性

这进一步说明 VSA backend 的“非 PyTorch实现”并不是黑盒 benchmark 代码，而是被当成正式训练算子维护。

## 10. 小结

这几条 backend 的分工可以直接按执行模型来记：

1. **Triton sparse kernel**
   - 用 `q2k/k2q` 索引表驱动 64x64 block-sparse attention
   - forward 负载规则，反向主要在 `k2q_num` 上出现离散
2. **ThunderKittens CUDA**
   - 用 `(q_blk, qo_head, batch)` 或 `(kv_blk, qo_head, batch)` 的 CTA 映射承载 Hopper `TMA + WGMMA`
   - forward CTA 深度近似常数，backward CTA 深度由 `qo_blocks` 决定
3. **CuTe DSL**
   - FastVideo 侧只负责 block-sparse tensor、`mask_mod` 与 `variable_block_sizes` 的契约转换
   - 物理 CTA/tile 调度留在 FA4/CuTe 内核内部

256 route-A 则单独说明了一件事：逻辑 256 稀疏图可以通过 `4 x 4` 物理子图展开，复用已有 64-token backend，而不必同步维护一套新的 256-token 稀疏 kernel。
