# FastVideo VSA：Sparse Backends

这一页专门讲 VSA 最有“低层味”的部分，也就是非 PyTorch 实现的 sparse backend。重点覆盖：

- 64-token 路径的 Triton sparse attention
- Hopper `sm_90a` 上的 ThunderKittens CUDA kernel
- 256-token 路径的 Triton route-A
- 256-token 路径的 CuTe DSL / FA4 fastpath

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
program_id(0) = q_blk
program_id(1) = off_hz = batch * head
```

**源码位置**: [`_attn_fwd_sparse`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py#L32-L160)

每个 program 会：

- 先从 `q2k_num` 读出本行有多少个有效 KV block；
- 再从 `q2k_index` 读这行对应的 block 列表；
- 对这些 block 逐个执行 `Q_i @ K_j^T`、softmax 累积、`P @ V_j`。

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
  - grid over KV blocks
- `_attn_bwd_dq_kernel`
  - grid over Q blocks

**源码位置**: [`triton_block_sparse_attn_backward`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py#L746-L850)

这比“单个超级 backward kernel”更自然，因为：

- `dK/dV` 复用同一个 `K/V` tile
- `dQ` 复用同一个 `Q` tile

对应的稀疏邻接方向也不同。

### 5.3 64-token block 被拆成两个 32-token half

反向里经常出现：

- `BLOCK_M1 = 32, BLOCK_N1 = 64`
- `BLOCK_M2 = 64, BLOCK_N2 = 32`
- `kv_blocks * 2`

原因是 backward 内部把 `64` token block 再拆成两个 `32` token half-block 来配合矩阵形状和流水方式。这个选择既影响 mask 写法，也影响 `variable_block_sizes` 的使用。

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

### 6.3 为什么 `block_size` 作为单独数组传入

前向里有：

```cpp
right_fill(att_block, att_block, g.block_size[q2k_block_sparse_index_ptr[kv_idx]], -inf)
```

这和 Triton 版的 `variable_block_sizes` 完全同构：每个 KV block 的真实有效列数不同，所以最后一个边界块必须按真实宽度做裁剪。

### 6.4 backward 同样走 `k2q` 邻接

`bwd_attend_ker` 中读取的是：

- `k2q_block_sparse_index`
- `k2q_block_sparse_num`

也就是说 CUDA 后端和 Triton 后端在抽象层上保持一致：

- forward 用 `q2k`
- backward 的 `dK/dV` 用 `k2q`

这说明上层 index-native 抽象设计得足够干净，换后端并不需要改方法层。

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

256 路径有专门的三方对照测试：

- torch reference
- CuTe
- Triton route-A

**源码位置**: [`test_vsa256_forward_cross.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/tests/test_vsa256_forward_cross.py#L1-L136)

这意味着 256 路径不是实验性死代码，而是有明确的数值对照基线。

此外，`test_vsa.py` 也专门测了：

- variable block size
- `q_seq_len != kv_seq_len`
- forward/backward 与 PyTorch reference 的一致性

这进一步说明 VSA backend 的“非 PyTorch实现”并不是黑盒 benchmark 代码，而是被当成正式训练算子维护。

## 10. 小结

如果把 VSA 的非 PyTorch实现按价值排序，我会这样看：

1. **Triton sparse kernel**
   - 是通用保底路径
   - 也是理解索引执行模型的最好入口
2. **ThunderKittens CUDA**
   - 是 Hopper 上真正的高性能专用路径
   - 展示了 VSA 如何把 block 稀疏映射到 TMA/WGMMA
3. **CuTe DSL**
   - 主要服务 256-token 路径
   - 核心看点是 block-sparse tensor 表示与 `mask_mod`，而不是手写 kernel body

而 256 route-A 则体现了另一个同样重要的工程原则：

- 在没有专用 fastpath 时，先把算法跑通；
- 再用新硬件和可选依赖逐步替换默认实现。
