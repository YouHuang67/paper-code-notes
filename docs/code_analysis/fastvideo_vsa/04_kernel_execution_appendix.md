---
tags:
  - Sparse Attention
  - Video Generation
  - CUDA
  - Triton
  - CUTLASS
---

# FastVideo VSA：Kernel Execution Appendix

这一页不再介绍“模块之间怎么调用”，而是直接按源码执行顺序展开 VSA 的关键 kernel。重点是三件事：

- 张量在进入 kernel 之前到底被整理成了什么形状；
- 每个 backend 实际按什么 grid / CTA / 邻接表运行；
- 数值状态、shared memory、寄存器和写回结果分别落在哪。

核心源码：

- [`ops.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/ops.py#L69-L170)
- [`block_sparse_attn.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py#L1-L424)
- [`index.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/index.py#L1-L260)
- [`block_sparse_attn_triton.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py#L1-L850)
- [`block_sparse_h100.cu`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/csrc/attention/block_sparse_h100.cu)
- [`block_sparse_attn_256.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py#L1-L170)
- [`block_sparse_attn_cute_fwd.py`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_cute_fwd.py#L1-L266)

## 1. 加速根源先压成一个执行式

对 `block_elements = B`、`q_num_blocks = T_q`、`kv_num_blocks = T_k`、每个 query block 选 `K` 个 key block 的 VSA，一次前向的执行式可以写成：

$$
\text{VSA}(Q,K,V)
=
\underbrace{\text{Broadcast}\left(
\operatorname{Softmax}\left(\frac{Q_cK_c^\top}{\sqrt d}\right)V_c
\right)}_{\text{coarse}}
+
\underbrace{\text{BlockSparseAttn}(Q,K,V;\mathcal{N})}_{\text{fine}}
$$

其中：

$$
Q_c \in \mathbb{R}^{T_q \times d},\quad
K_c,V_c \in \mathbb{R}^{T_k \times d},\quad
\mathcal{N}(i)\subseteq \{1,\dots,T_k\},\ |\mathcal{N}(i)|=K
$$

对应到实现，主要开销被拆成四类：

1. `fused_block_mean`
   - 把 token 级 `Q/K/V` 压到 block 级 `Q_c/K_c/V_c`
2. `fused_topk_mask`
   - 由 `scores = Q_cK_c^T / sqrt(d)` 生成 block 级稀疏图
3. `map_to_index / invert_indices`
   - 把布尔图压成 `q2k` 行索引，再翻成 `k2q`
4. 稀疏 attention backend
   - 64-token Triton
   - Hopper TK/CUDA
   - 256-token route-A 或 CuTe DSL

这条链里真正把 wall-clock 拉下来的不是某一个算子，而是：

- coarse selector 把稀疏图构造成固定 `topk` 的 block 行；
- fine backend 直接按这张图执行，不再 materialize dense logits；
- 64-token 路径把 tile 形状固定在稀疏 kernel 最熟悉的 `64 x 64` 上。

## 2. 进入 backend 之前，张量契约已经完全变了

`ops.video_sparse_attn()` 接收的是：

- `q: [B, H, S_q, D]`
- `k: [B, H, S_kv, D]`
- `v: [B, H, S_kv, D]`
- `variable_block_sizes: [T_k]`
- `q_variable_block_sizes: [T_q]`

**源码位置**: [`video_sparse_attn`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/ops.py#L69-L143)

它先检查：

- `S_q % B == 0`
- `S_kv % B == 0`
- `len(variable_block_sizes) == T_k`
- `len(q_variable_block_sizes) == T_q`

随后：

```python
q_c = fused_block_mean(q, q_variable_block_sizes, block_elements)
k_c = fused_block_mean(k, variable_block_sizes, block_elements)
v_c = fused_block_mean(v, variable_block_sizes, block_elements)
scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (dim ** 0.5)
mask = fused_topk_mask(scores, topk)
```

此时张量语义已经从 token 邻接变成了 block 邻接：

- `q_c: [B, H, T_q, D]`
- `k_c: [B, H, T_k, D]`
- `scores: [B, H, T_q, T_k]`
- `mask: [B, H, T_q, T_k]`

`mask[b,h,i,j] = True` 表示第 `i` 个 query block 需要访问第 `j` 个 key block。后续所有 sparse backend 都只消费这个 block 图，不再关心 coarse attention 的数学来源。

## 3. 布尔 block 图如何压成 index-native 格式

### 3.1 `q2k` 的物理含义

`block_sparse_attn()` 的第一步是：

```python
q2k_idx, q2k_num = _map_to_index(block_map)
```

**源码位置**: [`block_sparse_attn`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py#L413-L424)

`index.py` 里的 `map_to_index_kernel` 是一个非常直接的压缩过程：

- 一个 Triton program 处理一个 `(b, h, q_blk)`；
- 顺序扫描 `num_kv_blocks` 个布尔位；
- 见到 `True` 就把 `kv_idx` 写进 `index_ptr_base + num * stride`；
- 最后把计数写进 `index_num_ptr`。

**源码位置**: [`map_to_index_kernel`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/index.py#L31-L63)

因此：

- `q2k_idx[b,h,q_blk,:]` 是这一行实际访问的 KV block 编号；
- `q2k_num[b,h,q_blk]` 是这一行有效编号的长度；
- `q2k_idx` 的最后一维虽然定长，但只前 `q2k_num` 个元素有效。

### 3.2 `k2q` 为什么必须单独构造

反向需要从 KV block 视角知道“谁访问了我”，所以 `block_sparse_attn_backward_triton()` 和 `block_sparse_attn_backward_sm90()` 都会先做：

```python
k2q_idx, k2q_num = _invert_indices_for_backward(q2k_idx, q2k_num, num_kv_blocks)
```

**源码位置**:

- [`block_sparse_attn_backward_triton`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py#L163-L197)
- [`block_sparse_attn_backward_sm90`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py#L293-L327)

`invert_indices` 的写法是典型的 GPU 原子转置：

- 一个 program 处理一个 `(b,h,q_blk)`；
- 遍历这行的 `q2k_idx`；
- 对目标 `kv_blk` 的 `k2q_num` 做 `atomic_add` 占位；
- 把当前 `q_blk` 写入 `k2q_idx[..., kv_blk, pos]`。

**源码位置**: [`_invert_indices_kernel`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/index.py#L159-L260)

因此 `k2q_num` 的分布完全由全局图决定，而不再受固定 `topk` 约束。这就是反向 load imbalance 的根。

## 4. Triton 64-token 前向的实际执行状态

### 4.1 host wrapper 交给 kernel 的不是“稀疏矩阵”，而是三张表

`triton_block_sparse_attn_forward()` 把这些对象传进 `_attn_fwd_sparse`：

- `q, k, v`
- `q2k_index`
- `q2k_num`
- `variable_block_sizes`
- 输出 `M, o`

**源码位置**: [`triton_block_sparse_attn_forward`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py#L692-L743)

launch grid 是：

```text
grid = (Tq / 64, B * H, 1)
```

这意味着一个 program 永远只负责：

- 一个 `64` token 的 Q block
- 一个 `(batch, head)`

### 4.2 program 内部持有的状态

`_attn_fwd_sparse` 在进入 sparse loop 前会构造：

- `Q_ptr`: 当前 `64 x D` query tile 的 block pointer
- `K_base`: 所有 KV tile 的“列视图”基址
- `V_base`: 所有 KV tile 的“行视图”基址
- `m_i: [64]`
- `l_i: [64]`
- `acc: [64, D]`

**源码位置**: [`_attn_fwd_sparse`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py#L34-L160)

这里最关键的一点是：`acc` 永远只对应当前这 64 个 query token 的输出寄存器块，不会展开成整行 dense attention。

### 4.3 每次稀疏迭代到底做了什么

对第 `i` 个命中的 KV block，程序执行：

1. `kv_idx = q2k_index[row, i]`
2. `block_size = variable_block_sizes[kv_idx]`
3. `k = load(K[:, kv_idx*64:(kv_idx+1)*64])`
4. `qk = q @ k`
5. `mask` 掉 `block_size` 之外的列
6. 以 FA2 方式更新 `m_i, l_i, acc`
7. `v = load(V[kv_idx*64:(kv_idx+1)*64, :])`
8. `acc += p @ v`

数值上相当于：

$$
S_{ij} = Q_i K_j^\top,\quad
P_{ij} = \exp(S_{ij} - m_i),\quad
O_i \leftarrow O_i + P_{ij}V_j
$$

只是 `m_i,l_i` 用 online softmax 维护，所以不会保存全体 `S_{ij}`。

### 4.4 `M` 存的不是普通 max，而是 base-2 LSE

forward 末尾有两句非常关键：

```python
m_i += tl.math.log2(l_i)
tl.store(M + off_hz * N_CTX_Q + offs_m, m_i)
```

`M` 不是裸 `row_max`，而是：

$$
M_i = m_i + \log_2 l_i
$$

也就是以 `log2` 域表示的 row-wise log-sum-exp。反向再配合 `exp2` 复原概率。这样做是为了把整条实现统一到 base-2 指数路径上，减少缩放常数和数值误差。

## 5. Triton 64-token 反向为什么拆成三段

### 5.1 预处理 kernel：先算 `D = \sum (O \odot dO)`

`_attn_bwd_preprocess` 对每个 `64` token Q block 做：

$$
\Delta_i = \sum_{d=1}^{D} O_{i,d}\, dO_{i,d}
$$

**源码位置**: [`_attn_bwd_preprocess`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py#L166-L188)

这个量在 softmax 反向里反复使用，所以单独预处理比在主 kernel 里重复计算更合适。

### 5.2 为什么 `K` 先乘 `sm_scale / ln 2`

wrapper 里有：

```python
RCP_LN2 = 1.4426950408889634
arg_k = k * (sm_scale * RCP_LN2)
```

**源码位置**: [`triton_block_sparse_attn_backward`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py#L746-L850)

原因是前向和反向都改写成了 `exp2` 形式：

$$
e^x = 2^{x / \ln 2}
$$

把 `K` 预先乘上 `sm_scale / \ln 2` 之后，kernel 内部的 `qk` 可以直接喂给 `exp2`。对应地：

- `dK` 写回前乘 `sm_scale`
- `dQ` 写回前乘 `\ln 2`

这是把链式法则分摊到不同位置的实现写法，不是额外的近似。

### 5.3 `dK/dV` 与 `dQ` 为什么分开

`_attn_bwd_dkdv_kernel` 的 grid 是：

```text
grid_kv = (Tkv / 64, 1, B * H)
```

`_attn_bwd_dq_kernel` 的 grid 是：

```text
grid_q = (Tq / 64, 1, B * H)
```

这样拆后：

- `dK/dV` kernel 把一个 `64 x D` 的 `K/V` tile 常驻寄存器或 SRAM；
- `dQ` kernel 把一个 `64 x D` 的 `Q` tile 常驻寄存器；
- 两边各自沿不同的邻接表展开。

这比单个“大而全”的 backward kernel 更贴合数据复用方向。

### 5.4 为什么要拆成 32-token half-block

在 `_attn_bwd_dkdv` 和 `_attn_bwd_dq` 里，64-token block 都被拆成两个 half：

- `BLOCK_M1 = 32, BLOCK_N1 = 64`
- `BLOCK_M2 = 64, BLOCK_N2 = 32`

对应代码：

- `for blk_idx in range(q_blocks * 2)` in `dK/dV`
- `for blk_idx in range(kv_blocks * 2)` in `dQ`

这不是语义需要，而是 kernel 形状需要。拆成 `32x64` 或 `64x32` 后：

- `dV += P^T dO`
- `dK += dS^T Q`
- `dQ += dS K`

这些 GEMM 的寄存器布局更规整；同时边界块的 `variable_block_sizes[kv_idx]` 还能通过：

```python
offs_in_block = half * 32 + tl.arange(0, 32)
mask = offs_in_block < block_size
```

精确切掉尾部无效列。

## 6. Hopper ThunderKittens 前向：CTA 内部到底在做什么

### 6.1 host wrapper 先把 head 关系编码进 `hr`

前向 wrapper 会检查：

- `qo_heads >= kv_heads`
- `qo_heads % kv_heads == 0`

然后得到：

```cpp
auto hr = qo_heads / kv_heads;
dim3 grid(q_seq_len / 64, qo_heads, batch);
```

**源码位置**: [`block_sparse_attention_forward`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/csrc/attention/block_sparse_h100.cu#L687-L877)

device kernel 再用：

```cpp
int kv_head_idx = blockIdx.y / g.hr;
```

把多个 query head 映射到同一个 KV head。这就是 GQA/MQA 在 kernel 层的具体落点。

### 6.2 前向 CTA 的全局对象和共享内存

`fwd_globals<D>` 里带着：

- 全局 `q/k/v/o/l`
- `N`
- `hr`
- `max_kv_blocks_per_q`
- `q2k_block_sparse_index`
- `q2k_block_sparse_num`
- `block_size`

**源码位置**: [`fwd_globals`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/csrc/attention/block_sparse_h100.cu#L37-L63)

CTA 进入后会在 shared memory 上分配：

- `q_smem`
- `k_smem`
- `v_smem`
- `l_smem`
- `o_smem`

再由 `threadIdx.x == 0` 发起：

- `Q` block 的 TMA load
- 第一个 `K/V` block 的 TMA load
- 对应 semaphore 初始化

这意味着 forward CTA 不是“所有线程一起做全局 load”，而是单线程发起 TMA，warpgroup 负责 compute。

### 6.3 128 线程为什么刚好对应一个 warpgroup

kernel 声明是：

```cpp
__launch_bounds__(128, 4)
```

也就是一个 CTA 恰好 `4` 个 warp，正好是一个 Hopper warpgroup。随后：

- `warpgroup::mm_ABt` 做 `QK^T`
- `warpgroup::mma_AB` 做 `PV`

在线 softmax 状态：

- `max_vec`
- `norm_vec`
- `o_reg`

全都常驻寄存器。共享内存只保存 TMA 进来的 tile 和最终的输出暂存。

### 6.4 稀疏扫描的流水方式

forward 主循环有两个特点：

1. 当前 `kv_idx` 正在计算 `QK^T` / `PV`
2. 下一个 `kv_idx + 1` 的 `K/V` tile 由 `threadIdx.x == 0` 提前发起 TMA load

对应代码模式是：

```cpp
wait(k_smem_arrived, kv_idx % 2);
warpgroup::mm_ABt(...);
if (threadIdx.x == 0) tma::load_async(next_k);
...
wait(v_smem_arrived, kv_idx % 2);
warpgroup::mma_AB(...);
if (threadIdx.x == 0) tma::load_async(next_v);
```

这就是典型的“当前块计算 + 下一块预取”的双缓冲稀疏扫描。

### 6.5 `right_fill` 是边界块真实宽度的落点

每次算出 `att_block = QK^T` 之后，都会执行：

```cpp
right_fill(att_block, att_block, g.block_size[q2k_block_sparse_index_ptr[kv_idx]], -inf)
```

它把第 `block_size` 列之后的元素置成 `-inf`，于是：

- row max 不会看到 padding；
- `exp2` 后 padding 列概率为 `0`；
- `PV` 也不会把无效列带进输出。

因此 variable block size 在 TK 路径里不是 wrapper 层概念，而是 CTA 内 softmax 的一部分。

## 7. Hopper ThunderKittens 反向：`qg/kg/vg` 是怎么累积出来的

### 7.1 预处理 kernel 先生成 `D`

`bwd_attend_prep_ker` 读取：

- `og`
- `o`

并计算：

$$
D_i = \sum_d O_{i,d}\, dO_{i,d}
$$

写入 `d_vec`。

**源码位置**: [`bwd_attend_prep_ker`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/csrc/attention/block_sparse_h100.cu#L274-L340)

### 7.2 主 backward CTA 拥有一个 KV block

host wrapper launch：

```text
grid_bwd_2 = (kv_seq_len / 64, qo_heads, batch)
threads = 128
dynamic_smem = 72000
```

**源码位置**: [`block_sparse_attention_backward`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/csrc/attention/block_sparse_h100.cu#L881-L1102)

device 里第一件事就是：

```cpp
const int kv_head_idx = (blockIdx.y) / hr;
const int qo_blocks = *k2q_block_sparse_num_ptr;
if (qo_blocks <= 0) return;
```

也就是说这个 CTA 负责的是：

- 一个 KV block
- 一个 output/query head
- 一个 batch

并沿着 `k2q` 邻接表扫描所有访问它的 Q block。

### 7.3 CTA 内部真正积累的三个梯度块

在 `bwd_attend_ker` 中：

- `vg_reg` 是当前 KV block 的 `dV`
- `kg_reg` 是当前 KV block 的 `dK`
- `qg_reg` 是当前扫描到的某个 Q block 的 `dQ` 增量

对应计算是：

1. `dp_block_t = V dO^T`
2. `s_block_t = K Q^T - l`
3. `p_block_t = exp2(s_block_t)`
4. `ds_block_t = p_block_t * (dp_block_t - D)`
5. `vg_reg += P^T dO`
6. `kg_reg += dS^T Q`
7. `qg_reg = dS K`

其中：

- `vg_reg` 和 `kg_reg` 在整个 CTA 生命周期内持续累加；
- `qg_reg` 对每个被命中的 Q block 都要单独写回一次。

### 7.4 为什么 `dQ` 要用 `store_add_async`

同一个 Q block 可能会被多个 KV block 引用，所以 `dQ` 不能由某个 CTA 独占。代码里：

```cpp
tma::store_add_async(g.qg, qg_smem, tile_idx);
```

表示这个 CTA 只贡献“当前 KV block 对某个 Q block 的那一份 `dQ`”，由全局加法累积成最终结果。

而 `dK/dV` 的拥有者是当前 CTA 本身，所以它们在 CTA 尾部直接写回：

```cpp
tma::store_add_async(g.kg, kg_smem[0], tile_idx);
tma::store_add_async(g.vg, vg_smem[0], tile_idx);
```

这里虽然也是 add 语义，但目标 tile 只对应当前 `(kv_blk, kv_head, batch)`，不存在像 `dQ` 那样跨所有 KV block 的全局共享。

### 7.5 反向不均衡为什么主要集中在这里

forward 的工作量近似是固定 `topk` 次稀疏迭代；backward 则是固定 `kv_blk`，扫描可变长度的 `qo_blocks`：

- 热门 KV block：`qo_blocks` 很大
- 冷门 KV block：`qo_blocks` 很小
- 未命中 KV block：直接 return

因此 CTA 数量虽然规则，但 CTA 运行时间不是规则的。这个不均衡不是来自 block shape，而是来自图的入度分布。

## 8. 256-token 路径：真正的细节在“图和尺寸怎么变”

### 8.1 route-A 不是复制一份 mask，而是展开成 16 条物理边

`_expand_mask_and_sizes_256_to_64()` 做的是：

```python
expanded_mask = logical_mask_256.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
expanded_sizes = clamp(sizes[:, None] - [0,64,128,192], 0, 64).reshape(-1)
```

**源码位置**: [`_expand_mask_and_sizes_256_to_64`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py#L78-L115)

如果逻辑边是：

$$
Q_i \rightarrow K_j
$$

物理层会变成：

$$
Q_{i,a} \rightarrow K_{j,b},\qquad a,b\in\{0,1,2,3\}
$$

也就是 `4 x 4 = 16` 条 64-token 物理边。这样 256-token 路径就能直接复用已有的 64-token Triton backend。

### 8.2 CuTe 路径不是改 kernel，而是改稀疏张量描述

`block_sparse_attn_cute_fwd.py` 做的事可以压成三步：

1. 把逻辑 256 KV block 拆成两个 128-token child
2. 按 `variable_block_sizes` 分成 `full_map` 和 `mask_map`
3. 生成 `mask_mod`，把 partial block 的真实有效列传进 FA4/CuTe

**源码位置**:

- [`_expand_mask_and_sizes_256_to_128`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py#L53-L75)
- [`_cute_forward`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_cute_fwd.py#L144-L196)
- [`_build_vbs_mask_mod`](https://github.com/hao-ai-lab/FastVideo/blob/970409962f358afd529b969a378174c849665837/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_cute_fwd.py#L108-L141)

也就是说，FastVideo 侧对 CuTe 的责任到这里就结束了：

- 给出 block-sparse tensor
- 给出 partial block 的运行时谓词
- 给出 `aux_tensors=[variable_block_sizes]`

真正的 CTA grid、tile schedule、SM100+ 上的细节都被封装在 FA4/CuTe 内核内部。

## 9. 这一页的阅读方式

如果后面要对源码做逐段检查，推荐按下面顺序对着看：

1. `ops.py`
   - 看 block 级粗筛是怎样把 `scores` 和 `mask` 产出来的
2. `index.py`
   - 看布尔图怎样压成 `q2k`，再翻成 `k2q`
3. `block_sparse_attn_triton.py`
   - 看 forward 的 `Q tile -> sparse KV scan -> M/O`
   - 看 backward 的 `preprocess -> dK/dV -> dQ`
4. `block_sparse_h100.cu`
   - 看 host wrapper 的 grid
   - 看 CTA 内部 TMA / warpgroup 流水
   - 看 backward 里 `qo_blocks` 如何决定实际循环长度
5. `block_sparse_attn_256.py` + `block_sparse_attn_cute_fwd.py`
   - 看 256 路径到底是在扩图，还是在描述 partial block

这样读，代码里的 `grid`、`q2k_num`、`k2q_num`、`block_size`、`qg/kg/vg` 这些名字就不会再是抽象符号，而能对应到具体执行状态。
