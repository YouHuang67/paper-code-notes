---
tags:
  - CUTLASS
  - Flash Attention
  - Online Softmax
---
# CUTLASS Memory Efficient Attention 前向 Kernel

**源码仓库**: [pytorch/pytorch](https://github.com/pytorch/pytorch) — `aten/src/ATen/native/transformers/cuda/mem_eff_attention/`

本文分析 PyTorch ATen 内置的 CUTLASS Memory Efficient Attention 前向 kernel。该 kernel 是 xformers `cutlass.py` 后端通过 `torch.ops.aten._efficient_attention_forward` 调用的实际 CUDA 实现，覆盖 SM50（Maxwell）到 SM90（Hopper）全架构，支持 f32/f16/bf16、任意 head_dim（最大 65536）、additive bias、因果/局部 mask、dropout，兼有前向与反向。

### 在 xformers 后端体系中的位置

xformers `memory_efficient_attention` API 根据输入特征自动分发到不同后端（完整后端列表见 [Triton Split-K 文档的后端分发全景](../xformers_memory_efficient_attention/00_overview.md#后端分发全景)）。本文的 CUTLASS kernel 与 [Triton Split-K](../xformers_memory_efficient_attention/00_overview.md) 是其中两个核心后端，两者实现**完全相同的 Online Softmax 算法**（[数学公式](../xformers_memory_efficient_attention/00_overview.md#12-online-softmaxflashattention)），但面向不同场景：

| | CUTLASS（本文） | Triton Split-K |
|--|----------------|----------------|
| 适用场景 | prefill / 训练（Q 较多） | decode 推理（Q 极少、KV 极长） |
| 并行策略 | grid = `(Q_blocks, H, B)`，串行遍历 KV | grid 增加 `split_k` 维度，KV 分 chunk 并行 |
| 反向传播 | 支持（kernel_backward.h） | 不支持 |
| 量化/分页 | 不支持 | INT4/FP8 + PagedAttention |
| GEMM 实现 | CUTLASS MMA 模板，显式流水线 | `tl.dot`，Triton 编译器自动选 MMA |
| head_dim | 最大 65536（K-dim tiling） | 受寄存器限制 |

dispatch 的核心判据：`get_split_k()` 当 `Mq > 1` 且 `B×G×H > 64` 时返回 `split_k=1`（即走 CUTLASS/Flash），否则计算 `split_k > 1` 走 Triton Split-K。

涉及的 CUDA / CUTLASS / CuTe 公共知识参见 [CUDA 基础：导读](../cuda_foundations/00_overview.md)。

## 1. 调用链

```
xformers/ops/fmha/cutlass.py  FwOp.apply_bmhk()
  → torch.ops.aten._efficient_attention_forward
```

**源码位置**: [attention.cu#L1384-L1872](https://github.com/pytorch/pytorch/blob/24be3ec/aten/src/ATen/native/transformers/cuda/attention.cu#L1384-L1872)

ATen 入口做参数校验、分配输出/LSE 张量、填充 `Params` 结构体，然后通过 `dispatch_cutlassF` 分发到具体 kernel 变体：

```cpp
DISPATCH_TYPES(query, ([&]() {
    dispatch_cutlassF<scalar_t>(launchKernel, computeCapability);
}));
```

**源码位置**: [cutlassF.h](https://github.com/pytorch/pytorch/blob/24be3ec/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassF.h)

`dispatch_cutlassF` 按 `(dtype, compute_capability)` 选择预编译的 kernel 变体，每个变体是 `AttentionKernel` 模板的一个特化实例。`launchKernel` 回调会逐个尝试变体，选择第一个兼容当前输入的 kernel 启动。

## 2. 模板参数与 Kernel 变体

**源码位置**: [kernel_forward.h#L75-L131](https://github.com/pytorch/pytorch/blob/24be3ec/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L75-L131)

```cpp
template <
    typename scalar_t_,       // 数据类型: float / half_t / bfloat16_t
    typename ArchTag,         // 目标架构: Sm50 / Sm70 / Sm75 / Sm80
    bool isAligned_,          // Q/K/V 是否满足对齐要求
    int kQueriesPerBlock_,    // 每个 block 处理的 Q token 数
    int kKeysPerBlock_,       // 每次迭代处理的 K token 数
    int kMaxK_,               // head_dim 上界（编译期常量）
    bool kSupportsDropout_,
    bool kSupportsBias_>
struct AttentionKernel { ... };
```

编译期推导的关键分支：

- `kSingleValueIteration = kMaxK <= kKeysPerBlock`：head_dim 足够小时为 true，V 矩阵一次迭代处理完，输出留寄存器（`kKeepOutputInRF=true`），避免 GMEM 中间缓冲
- `kPreloadV = Sm80+ && f16`：MM0 完成后立即预取 V 到 SMEM，与 iterative_softmax 计算重叠

```cpp
static constexpr bool kSingleValueIteration = kMaxK <= kKeysPerBlock;
static constexpr bool kKeepOutputInRF = kSingleValueIteration;
static constexpr bool kPreloadV = ArchTag::kMinComputeCapability >= 80 && kIsHalf;
```

实际预编译的变体（以 Sm80 + f16 aligned 为例）：

| 变体名 | QueriesPerBlock | KeysPerBlock | MaxK | 特点 |
|--------|:-:|:-:|:-:|------|
| `64x64_rf` | 64 | 64 | 64 | 小 head_dim，输出留寄存器 |
| `64x128_rf` | 64 | 128 | 128 | 中 head_dim，输出留寄存器 |
| `32x128_gmem` | 32 | 128 | 65536 | 大 head_dim，输出经 GMEM 累加 |

`launchKernel` 回调按顺序尝试，选中第一个满足 `kMaxK >= value.size(3)` 且对齐/SMEM 大小符合的变体。

## 3. 数据布局与 Grid 配置

**源码位置**: [kernel_forward.h#L132-L345](https://github.com/pytorch/pytorch/blob/24be3ec/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L132-L345)

输入 Q/K/V 的布局：`[batch, seq_len, num_heads, head_dim]`，stride 灵活（非 contiguous 也可）。

Grid 配置：

```cpp
dim3 blocks(
    ceil_div(num_queries, kQueriesPerBlock),  // Q 序列分块
    num_heads,                                 // head 维度
    num_batches                                // batch 维度
);
dim3 threads(kWarpSize, kNumWarpsPerBlock, 1); // 如 (32, 4, 1) = 128 threads
```

`advance_to_block()` 在 kernel 入口处根据 `blockIdx.{x,y,z}` 计算当前 block 处理的 Q/K/V 起始指针，同时处理：

- **变长序列**：通过 `seqstart_q_ptr` / `seqstart_k_ptr`（cu_seqlens）索引每个 batch 的实际序列长度
- **因果 mask 裁剪**：`num_keys` 收缩到 `query_start + causal_diagonal_offset + kQueriesPerBlock`，跳过不可能有非零 attention 的 K 范围
- **MQA/GQA 优化**：当 `num_queries==1` 且 K/V 的 `strideH==0`（head 维度广播）时，将多个 head 映射到 Q 的序列维度：`q_strideM = q_strideH`，`num_queries = num_heads`——避免 Tensor Core 利用率低（1 query 只用到 1/kQueriesPerBlock 的计算）

## 4. Shared Memory 布局

**源码位置**: [kernel_forward.h#L514-L570](https://github.com/pytorch/pytorch/blob/24be3ec/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L514-L570)

SMEM 通过 [union 复用](../cuda_foundations/01_cuda_execution_model_and_memory.md#14-shared-memory)三个阶段的存储：

```
ScalingCoefs (常驻):
├─ m_prime[kQueriesPerBlock]    // 历史 max（对应 Online Softmax m^(j-1)）
├─ s_prime[kQueriesPerBlock]    // 历史 sum（对应 ℓ^(j-1)）
├─ mi[kQueriesPerBlock]         // 当前迭代 max（对应 m^(j)）
├─ out_rescale[kQueriesPerBlock]// rescale 系数 α = exp(m^(j-1) - m^(j))
└─ addition_storage[...]        // warp 间 reduction 暂存

union (分时复用):
├─ mm0: MM0::Mma::SharedStorage          // 阶段 1: QK^T GEMM
├─ after_mm0:
│   ├─ bias / si: AccumulatorSharedStorage // 阶段 2: bias tile 或 softmax 概率 P
│   └─ mm1: MM1::Mma::SharedStorage       // 阶段 3: PV GEMM
└─ epilogue: SharedStorage                 // 阶段 4: 输出写回
```

## 5. 主循环

**源码位置**: [kernel_forward.h#L624-L1117](https://github.com/pytorch/pytorch/blob/24be3ec/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L624-L1117)

主循环的整体结构与 [Triton Split-K 前向 kernel](../xformers_memory_efficient_attention/00_overview.md#2-前向-kernel) 完全对应——都是遍历 KV blocks，每次做 QK^T → softmax → PV，区别在于 Triton 版一个 chunk 内串行多个 block 然后跨 chunk 归约，CUTLASS 版则在一个 threadblock 内串行遍历所有 KV blocks。

对照 Triton Split-K 的关键差异：

- **MM0 (Q@K^T)**：Triton 中 Q 常驻寄存器、K 流式加载，一次 `tl.dot` 完成；CUTLASS 中 Q/K 均通过 IteratorA/B 从 GMEM→SMEM→寄存器，沿 head_dim 分 `gemm_k_iterations` 次迭代（[K-dim tiling](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md#mma-流水线)，支持 head_dim 最大 65536）
- **Scale/Bias/Mask**：Triton 用 `qk *= qk_scale` + `tl.where`；CUTLASS 通过 `AccumLambdaIterator` 逐元素操作寄存器 fragment
- **P 矩阵传递**：Triton 中 P 留在寄存器直接 `tl.dot`；CUTLASS 中 P 先写入 SMEM（`accumToSmem`），MM1 通过 `MmaFromSharedMemory` 从 SMEM 读取——这是 CUTLASS MMA 模板要求操作数经固定 Iterator 接口加载的结果
- **MM1 (P@V)**：`kKeepOutputInRF=true`（小 head_dim）时 `accum_o` 常驻寄存器，循环结束后一次性写回；否则每次迭代经 Epilogue 写回 GMEM buffer，下次迭代读回做 rescale 累加

```cpp
// ===== 初始化：Online Softmax 状态 (SMEM) + 输出累加器 (寄存器) =====
if (thread_id() < kQueriesPerBlock) {
    s_prime[thread_id()] = 0;           // ℓ^(0) = 0
    m_prime[thread_id()] = -inf;        // m^(0) = -∞
    mi[thread_id()] = -inf;
    out_rescale[thread_id()] = 1.0;
}
typename MM1::Mma::FragmentC accum_o;
accum_o.clear();                        // a^(0) = 0

// ===== KV block 主循环 =====
for (int32_t iter_key_start = 0; iter_key_start < p.num_keys;
     iter_key_start += kKeysPerBlock) {
    // 窗口注意力优化：KV block 完全在窗口外时 continue 跳过

    // --- MM0: Q @ K^T ---
    // Q/K 通过 IteratorA/B 从 GMEM 流式加载到 SMEM，再到寄存器执行 MMA
    // gemm_k_iterations = ceil(head_dim / Shape::kK)，沿 head_dim 分次迭代
    typename MM0::Mma mma(shared_storage.mm0, thread_id(), my_warp_id, my_lane_id);
    typename MM0::Mma::FragmentC accum;
    accum.clear();
    auto gemm_k_iterations =
        (head_dim + MM0::Mma::Shape::kK - 1) / MM0::Mma::Shape::kK;
    mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
    __syncthreads();

    // Sm80+f16: MM0 完成后立即预取 V 到 SMEM (prologueV)，与 softmax 计算重叠

    // --- Scale + Bias + Mask ---
    // AccumLambdaIterator 逐元素操作寄存器中的 accum fragment
    accum *= p.scale;                                          // QK^T × scale
    accum[idx] += bias_tensor_ref.at({accum_m, accum_n});      // additive bias (可选)
    if (accum_n > query_start + accum_m + causal_diagonal_offset - iter_key_start)
        accum[idx] = -inf;                                     // 因果 mask
    if (accum_n <= accum_m + offset)
        accum[idx] = -inf;                                     // 窗口 mask

    // --- iterative_softmax: 更新 mi/m_prime/s_prime/out_rescale，accum → P ---
    // 详见 §6
    iterative_softmax<...>(accum_o, accum, mi, m_prime, s_prime, out_rescale, ...);

    // --- P 写入 SMEM + Dropout ---
    // P 从寄存器写入 SMEM (AccumulatorSharedStorage)，作为 MM1 的 A 操作数
    // 可选 dropout: cuRAND Philox 生成器在 SMEM 上逐元素随机 mask
    MM0::B2bGemm::accumToSmem(shared_storage.after_mm0.si, accum, ...);
    __syncthreads();

    // --- MM1: P @ V ---
    // P 从 SMEM 读取 (MmaFromSharedMemory)，V 从 GMEM 流式加载
    for (int blockN = 0; blockN < nBlockN; ++blockN) {
        typename MM1::Mma mma_pv(
            shared_storage.after_mm0.si.accum_ref(),       // A: P (SMEM)
            shared_storage.after_mm0.mm1.operand_B_ref(),  // B staging area
            thread_id(), my_warp_id, my_lane_id);
        mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o);

        if (!kKeepOutputInRF) {
            // Epilogue: rescale + 写回 GMEM buffer (详见 §7)
        }
    }
}

// ===== 最终输出 =====
// kKeepOutputInRF 路径：全部 KV 迭代结束后调用一次 Epilogue
// 执行 output = accum_o / s_prime
if (kKeepOutputInRF) {
    EpilogueOutputOp rescale(s_prime, out_rescale);
    epilogue(rescale, dest_iter, accum_o);
}

// ===== LogSumExp =====
// 对照 Triton 完全相同：lse = m_i / log2e + log(l_i)
if (p.logsumexp_ptr && thread_id() < kQueriesPerBlock) {
    p.logsumexp_ptr[thread_id()] =
        mi[thread_id()] / kLog2e + cutlass::fast_log(s_prime[thread_id()]);
}
```

## 6. iterative_softmax 详解

**源码位置**: [kernel_forward.h#L1181-L1327](https://github.com/pytorch/pytorch/blob/24be3ec/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L1181-L1327)

这是 Online Softmax 的核心，在每次 KV block 迭代后被调用，更新 SMEM 中的统计量并将 QK^T 结果转为 softmax 概率。与 [Triton Split-K 的公式](../xformers_memory_efficient_attention/00_overview.md#12-online-softmaxflashattention) 完全对应。

逐变量对照（CUTLASS ↔ Triton Split-K）：

| Online Softmax 状态 | CUTLASS（SMEM 数组） | Triton（寄存器向量） |
|-------|------|------|
| 历史 max $m^{(j-1)}$ | `m_prime[kQueriesPerBlock]` | `m_i [BLOCK_M]` |
| 当前 max $m^{(j)}$ | `mi[kQueriesPerBlock]` | `m_i_new [BLOCK_M]` |
| 累计 sum $\ell$ | `s_prime[kQueriesPerBlock]` | `l_i [BLOCK_M]` |
| rescale $\alpha$ | `out_rescale[kQueriesPerBlock]` | `alpha [BLOCK_M]` |
| 累计输出 $\mathbf{a}$ | `accum_o`（寄存器 fragment） | `acc [BLOCK_M, D_PER_GROUP]` |

核心区别：Triton 中所有状态都在**寄存器**（每个 program 独立），CUTLASS 中 `mi/m_prime/s_prime/out_rescale` 在 **Shared Memory**（block 内跨 warp 共享），需要 `__syncthreads()` 和 `atomicMaxFloat` 协调。

对照数学公式（[§1.2](../xformers_memory_efficient_attention/00_overview.md#12-online-softmaxflashattention)）：

$$m^{(j)} = \max(m^{(j-1)},\; \max_i s_i)$$

$$\alpha^{(j)} = e^{m^{(j-1)} - m^{(j)}}$$

$$p_i = e^{s_i - m^{(j)}}$$

$$\ell^{(j)} = \alpha \cdot \ell^{(j-1)} + \sum p_i$$

后续全部使用 `exp2f` 替代 `expf`（[相关说明](../cuda_foundations/01_cuda_execution_model_and_memory.md#17-exp2f-vs-expf)），预乘 `kLog2e = 1.4426950408889634`，利用 $e^x = 2^{x \cdot \log_2 e}$。

```cpp
// ===== 预乘 log2(e)：后续全部使用 exp2f =====
constexpr float kLog2e = 1.4426950408889634074;
frag = cutlass::multiplies<Fragment>()(scaling * kLog2e, frag);

// ===== 更新 mi（行最大值）→ m^(j) =====
// 每个 warp 内各线程持有 accum 的不同列，先 warp 内求局部 max
// 再用 atomicMaxFloat 跨 warp 归约到 SMEM mi[row]
LambdaIterator::iterateRows(lane_offset,
    [&](int accum_m) { max = -inf; },
    [&](int accum_m, int accum_n, int idx) {
        if (accum_n < max_col) max = cutlass::fast_max(max, frag[idx]);
    },
    [&](int accum_m) { atomicMaxFloat(&mi[accum_m], max); });
__syncthreads();

// ===== 计算 rescale 系数 → α^(j) = exp2(m^(j-1) - m^(j)) =====
// 只需 kLinesPerWarp 个线程处理（每个 warp 负责几行）
if (lane_id < kLinesPerWarp) {
    int id = warp_id * kLinesPerWarp + lane_id;
    bool changed = m_prime[id] < mi[id];
    if (changed) {
        out_rescale[id] = exp2f(m_prime[id] - mi[id]);  // α = exp(m^(j-1) - m^(j))
        s_prime[id] *= out_rescale[id];                   // ℓ × α
    } else {
        out_rescale[id] = 1.0f;
    }
}
__syncthreads();

// ===== rescale 历史输出 → a^(j-1) × α^(j) =====
// 仅在 kKeepOutputInRF 且非首次迭代时执行（否则 rescale 在 Epilogue 中处理）
if (kKeepOutputInRF && !is_first) {
    frag_o[idx] = frag_o[idx] * out_rescale[accum_m];
}

// ===== 计算 softmax 概率 → p_i = exp2(s_i - m^(j)) =====
// 超出 max_col 的位置设为 0（边界处理）
frag[idx] = (accum_n < max_col) ? exp2f(frag[idx] - mi_row) : 0.0f;

// ===== 更新 s_prime → ℓ^(j) = α · ℓ^(j-1) + Σp_i =====
// 先 warp 内 reduce 求 total_row（同一行各列的 p 之和）
// 存入 addition_storage，然后跨 warp 汇总
LambdaIterator::iterateRows(...,
    [&](int accum_m, int accum_n, int idx) { total_row += frag[idx]; },
    [&](int accum_m) {
        if (LambdaIterator::reduceSameRow(lane_id, total_row, add))
            addition_storage[accum_m + kQueriesPerBlock * tile_offset.column()]
                = total_row;
    });
__syncthreads();

// 跨 warp 汇总到 s_prime，同时推进 m_prime ← mi
if (lane_id < kLinesPerWarp) {
    accum_t total_row = s_prime[id];
    for (int i = 0; i < MM0::MmaCore::WarpCount::kN; ++i)
        total_row += addition_storage[id + kQueriesPerBlock * i];
    s_prime[id] = total_row;
    m_prime[id] = mi[id];              // m^(j-1) ← m^(j)
}
```

## 7. Epilogue 输出归一化

**源码位置**: [epilogue_rescale_output.h](https://github.com/pytorch/pytorch/blob/24be3ec/aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_rescale_output.h)

`MemoryEfficientAttentionNormalize` 作为 CUTLASS [Epilogue](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md#epilogue) 的 OutputOp，在 MM1 结果写回 GMEM 时执行 rescale 和归一化。

对照 Triton Split-K：Triton 的归一化非常直接——主循环结束后 `attn_out = acc / l_i`，一步在寄存器内完成。CUTLASS 需要通过 Epilogue 框架处理，因为大 head_dim 时输出需要分次写回 GMEM：

$$
\text{output} = \alpha \cdot \text{accum} + \beta \cdot \text{source}
$$

其中：

- $\alpha = \begin{cases} 1/\ell^{(j)} & \text{isLast} \\ 1 & \text{otherwise} \end{cases}$
- $\beta = \alpha \cdot \text{out\_rescale}$（即 $\alpha \cdot e^{m^{(j-1)} - m^{(j)}}$）
- `source` 是上一次迭代写入的中间结果
- `accum` 是本次 MM1 的结果

三种场景：

| 场景 | isFirst | isLast | 行为 |
|-----|---------|--------|------|
| 首次且唯一 | true | true | `output = accum / s_prime` |
| 首次但非唯一 | true | false | `output = accum`（不归一化，写入 accum buffer） |
| 中间迭代 | false | false | `output = accum + out_rescale * source` |
| 最后一次 | false | true | `output = (accum + out_rescale * source) / s_prime` |

当 `kKeepOutputInRF=true` 时，全程在寄存器中累加，只在最后调用一次 Epilogue（isFirst=true, isLast=true），执行 `output = accum_o / s_prime`。

## 8. 与 Triton Split-K 的对比

两个后端实现相同的 Online Softmax 数学公式（见 [§1.2](../xformers_memory_efficient_attention/00_overview.md#12-online-softmaxflashattention)），核心差异来自**编程模型**和**目标场景**：

- **Triton Split-K** 的每个 program 独立处理一个 Q block × 一个 KV chunk，所有状态（`m_i`, `l_i`, `acc`）在寄存器内，program 之间通过 Split-K reduce kernel 归约。适合 decode（Q 少）——通过增加 `split_k` 维度弥补并行度不足
- **CUTLASS** 的每个 threadblock 处理一个 Q block × **全部** KV，block 内多个 warp 协作，通过 SMEM 共享 `mi/s_prime` 等状态并用 `__syncthreads__` 同步。适合 prefill（Q 多）——单 block 足以打满 Tensor Core

| 维度 | Triton Split-K | CUTLASS kernel_forward.h |
|-----|----------------|--------------------------|
| 并行粒度 | `(Q_blocks, B×G×H, split_k)` | `(Q_blocks, H, B)` |
| KV 遍历 | chunk 内串行，chunk 间并行 + reduce | 完全串行遍历 |
| GEMM 实现 | `tl.dot`（编译器自动选 MMA） | CUTLASS MmaMultistage 显式流水线 |
| P 矩阵存储 | 寄存器（`exp2` 后直接 `tl.dot`） | SMEM（accumToSmem → MM1 从 SMEM 读） |
| head_dim 处理 | 一次加载到寄存器（受限） | K-dim tiling（支持到 65536） |
| 输出累加 | 始终在寄存器 | 小 head_dim 寄存器，大的经 GMEM |
| 量化/分页 | INT4/FP8 + PagedAttention | 不支持 |
| 反向传播 | 无 | 有（kernel_backward.h） |
| 适用场景 | decode 推理（Q 少 KV 长） | 通用 prefill + 训练 |
