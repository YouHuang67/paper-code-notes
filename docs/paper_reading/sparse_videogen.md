---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
  - Triton
---

# Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity

- 论文：https://arxiv.org/abs/2502.01776
- 代码：https://github.com/svg-project/Sparse-VideoGen
- 团队：UC Berkeley, MIT, NVIDIA, Tsinghua University

## 概述

Sparse VideoGen（SVG）针对视频 DiT 中 3D full attention 的高成本问题，提出 training-free 的稀疏注意力框架。它的关键观察是：视频 attention head 并不是统一稀疏，而是大致分成两类功能模式：

- **Spatial Head**：关注同一帧或邻近帧内的空间局部 token，用于保持单帧空间结构
- **Temporal Head**：关注不同帧中同一空间位置附近的 token，形成 stride 为每帧 token 数 $L$ 的 slash pattern，用于保持时间一致性

单独使用 spatial-only 或 temporal-only 都会损伤质量，因为不同 head、不同 denoising step、不同 prompt 下最合适的模式会变。SVG 的做法是在线抽样少量 query row，对 spatial / temporal 两种稀疏输出分别和 full attention 输出比较 MSE，为每个 attention head 动态选择更接近 full attention 的模式。

工程上，SVG 的重点不是简单生成一个 mask，而是让 temporal slash pattern 变得硬件友好：通过 layout transformation 把原本 stride-$L$ 的跨帧同位置 token 重排成连续布局，使 sparse attention 能接近理论加速。代码实现中，SVG1 主路径是：在线 sample MSE 选 head 类型 → Triton placement kernel 对 temporal head 做 token-major 重排 → FlexAttention 或 FlashInfer BSR 执行统一 block sparse attention → Triton inverse placement 写回每个 head 的输出。同时用自定义 CUDA/Triton kernel 优化 QK-norm、RoPE 和重排，减少 attention 之外的系统瓶颈。

实验上，SVG 在 CogVideoX-v1.5 和 HunyuanVideo 上保持接近 dense 的视频质量，并显著加速：CogVideoX-v1.5 T2V 2.28×，CogVideoX-v1.5 I2V 2.23×，HunyuanVideo T2V 1.92×，结合 FP8 后 HunyuanVideo 达 2.33×。

## 稀疏性的两个模式

### Spatial Head

Spatial head 的注意力集中在同一帧及邻近帧，attention map 呈 block-wise layout。原因是视频 latent 通常按 frame-major 排列：一帧内的 $L$ 个空间 token 连续存储，因此同帧 / 邻近帧注意力在序列维度上自然形成连续块。

对 $N$ 个 frame、每帧 $L$ 个 token，若 spatial head 只关注 $c_s$ 个邻近 frame，则计算量从 full attention 的：

$$
O(N^2 L^2 H)
$$

降为：

$$
O(N c_s L^2 H)
$$

对应稀疏比例约为 $c_s / N$。这类模式本身是硬件友好的，因为保留区域是连续 block，容易映射到 block sparse attention。

### Temporal Head

Temporal head 关注跨帧同一空间位置的 token。若每帧有 $L$ 个 token，那么相同空间位置在序列中相隔 $L$，attention map 呈 slash / striped pattern。

如果每个 token 只关注 $c_t$ 个空间位置邻域，则计算量约为：

$$
O(L c_t N^2 H)
$$

稀疏比例约为 $c_t / L$。理论上也能省计算，但直接在原 frame-major layout 上计算很低效：要访问的 token 以 stride $L$ 分散在内存中，无法形成 Tensor Core 喜欢的连续 tile。

这就是 SVG 和许多 naive sparse mask 的关键差别：**不是发现稀疏就能加速，稀疏模式必须变成连续可计算的 layout**。

### Prompt 和首帧作为 attention sink

论文还观察到 text prompt 和 first frame 对 spatial / temporal head 都重要。因此 SVG 的 mask 不只是纯 spatial / temporal 局部区域，还会保留 prompt token 或第一帧 token 作为全局 sink，避免语义和初始视觉条件丢失。

## Online Profiling

直接用 full attention 判断每个 head 属于 spatial 还是 temporal 没有加速意义。SVG 采用在线抽样 profiling：

1. 从序列中随机采样少量 query row，论文实验中 1% token 已足够
2. 对采样 query 分别计算：
   - full attention 输出 $O_{\text{full}}$
   - spatial mask 输出 $O_{\text{spatial}}$
   - temporal mask 输出 $O_{\text{temporal}}$
3. 计算每个 head 的 MSE：

$$
\text{MSE}_s = \|O_{\text{full}} - O_{\text{spatial}}\|^2, \quad
\text{MSE}_t = \|O_{\text{full}} - O_{\text{temporal}}\|^2
$$

4. 每个 head 选择 MSE 更小的稀疏模式

代码里的 SVG1 processor 对应 `sample_mse` 逻辑：随机取 `num_sampled_rows`，对 sampled Q 先算一次 dense attention 作为 golden output，再对两个预生成 mask 分别 masked softmax，最后得到 `[cfg, num_heads]` 的 `best_mask_idx`。

这个 profiling 的关键是粒度选择：它不是给每个 token 动态选 mask，而是给每个 head 选模式。这样只需很小抽样量，就能覆盖 head 功能差异，同时避免 per-token 动态路由带来的额外 kernel 和调度成本。

## 代码实现

当前开源仓库同时包含 SVG1 和 SVG2。这里分析 SVG1：`WanAttn_SVGAttn_Processor2_0` / `Hunyuan_SVGAttn_Processor2_0` 这条路径；仓库里的 `SAPAttn`、k-means、semantic-aware permutation 属于 SVG2，已在 Sparse VideoGen2 笔记中单独分析。

### 实现结构

SVG1 的实现可以概括为五步：

1. 从模型 attention block 中取 Q/K/V，并做 QK norm、RoPE
2. 采样少量 query row，计算 spatial / temporal 两种 mask 与 dense 的 MSE，得到每个 head 的模式 `best_mask_idx`
3. 根据 `best_mask_idx` 将 temporal head 的 Q/K/V 从 frame-major 重排成 token-major；spatial head 保持原 layout
4. 在统一 layout 上执行 block sparse attention
5. 将每个 head 的输出按原模式逆变换回原序列位置，再接输出投影

这个结构的核心不是“为 spatial 和 temporal 分别跑两套 attention”，而是先通过 placement 将不同 head 变成统一的可执行布局，再用一个 block sparse attention backend 计算。

### 为什么 temporal head 必须重排

原始 video token layout 是 frame-major：

$$
[(f_0,p_0),(f_0,p_1),...,(f_0,p_{L-1}), (f_1,p_0),...]
$$

在这个 layout 下，相同空间位置 $p_i$ 跨 frame 的 token 是：

$$
(f_0,p_i), (f_1,p_i), ..., (f_{N-1},p_i)
$$

它们在内存中的 stride 是 $L$。Temporal head 需要的 slash pattern 在逻辑上很稀疏，但 CUDA kernel 会面对不连续访存和碎片化 block，Tensor Core 无法有效利用。

SVG 的 layout transformation 将 video token 重排为 token-major：

$$
[(p_0,f_0),(p_0,f_1),...,(p_0,f_{N-1}), (p_1,f_0),...]
$$

这样 temporal head 的跨帧同位置邻域变成连续区间，原本的 slash pattern 变成对角局部 block。重排不改变 attention 的数学结果，因为对 Q/K/V 做相同置换，再对输出做逆置换，等价于在原序列上计算同一个稀疏关系。

### Triton placement kernel

Wan 路径的 `wan_sparse_head_placement_kernel` 是实现这个重排的关键。它的 grid 是：

$$
\text{grid} = (\text{cfg}, \text{num\_heads}, \lceil \text{seq\_len} / 128 \rceil)
$$

每个 Triton program 处理一个 `(cfg, head, token block)`，`BLOCK_SIZE=128` 个 token，沿 `head_dim` 维度向量化 load/store。kernel 读取 `best_mask_idx[cfg, head]`：

- 如果该 head 是 spatial：Q/K/V 原样拷贝到输出 buffer
- 如果该 head 是 temporal：对 video token 执行

$$
\text{frame\_id} = \lfloor \text{token} / L \rfloor,\quad
\text{patch\_id} = \text{token} - \text{frame\_id}\cdot L
$$

然后写到：

$$
\text{store\_token} = \text{patch\_id}\cdot N + \text{frame\_id}
$$

也就是 frame-major 到 token-major。text / context token 不参与 video 重排，保持固定区域。

这个 kernel 的意义有两点：

- **按 head 动态分流**：同一个 batch 中不同 head 可以走不同 layout，不需要拆成多个 Python tensor 切片
- **重排融合 Q/K/V**：一次 traversal 同时搬运 query/key/value，避免对三个 tensor 分别做昂贵的 `torch.gather`

输出端还有 `hidden_states_placement`，根据 `best_mask_idx` 将 sparse attention 的结果写回原 head 和 token 顺序。对于 temporal head，相当于 token-major 到 frame-major 的逆置换。

### Block Sparse Attention 执行

SVG1 有两类 attention backend：

**FlexAttention 路径**

Wan 当前 processor 默认走 `flex_attention(query_out, key_out, value_out, block_mask=block_mask)`。`prepare_flexattention` 用 `generate_temporal_head_mask_mod` 构造 mask function，再通过 `create_block_mask` 编译成 PyTorch FlexAttention 的 block mask。

由于 placement 已经把 temporal head 重排成 token-major，spatial 和 temporal 都能用类似“对角带 + first-frame sink”的 block mask 表达。也就是说，FlexAttention 看到的是连续 block mask，而不是原始 slash pattern。

**FlashInfer BSR 路径**

老接口和部分 util 中使用 FlashInfer `BlockSparseAttentionWrapper`。空间和时间 mask 都被转成 BSR：

- spatial mask：row/column block 以 frame 为单位，block size 是 `num_tokens_per_frame x num_tokens_per_frame`
- temporal mask：重排后按较小 token block 切分，block size 由 `get_factor(num_frames, num_tokens_per_frame)` 选一个小于 256 的因子，保证 video length 能整除
- `row_indices` / `column_indices` 描述每个 block row 访问哪些 block column
- 额外 padding 256 个 column index，避免 FlashInfer kernel 中不规则访问带来的边界问题

FlashInfer 执行时，video token 走 BSR sparse attention；text prompt / first-frame sink 可以单独作为 dense term，再用 FlashInfer 的 LSE merge 和 sparse video 输出合并。这一点和 LVSA 的 text/video 分离思想类似：不要为了适配 block sparse kernel 把非视频 token 硬 padding 进 block grid。

### CUDA / Triton 辅助 kernel

SVG 的加速不只来自 sparse attention。论文和代码都显示 QK-norm、RoPE、layout transform 也是明显瓶颈，因此仓库提供了自定义 kernel。

**RMSNorm / LayerNorm CUDA kernel**

`narrow_rms_norm.cuh` 和 `narrow_layer_norm.cuh` 针对 head_dim 很小的场景设计，例如 32/64/128/256。kernel 的布局是：

- 输入看作 `[m, n]`，其中 `n=head_dim`
- 每个 CTA 处理 `bdy` 行，每行由一个 sub-warp 负责
- `bdx = head_dim / (sizeof(float4)/sizeof(T))`
- `bdy = 32 / bdx`
- 每个 thread 用 `float4` vectorized load 读一段 hidden dim
- sub-warp 内用 `__shfl_xor_sync` 做 sum / variance reduction
- 归一化后原地写回

这类 kernel 的核心思路是：head_dim 小时，PyTorch 通用 norm kernel 的并行度和调度开销不划算；SVG 用 sub-warp 把一整行归一化限制在一个 warp 内完成，不需要 shared memory 或跨 warp reduction。论文表 2 报告 QK-norm 平均 7.4× 加速。

**RoPE CUDA kernel**

`rope_enc.cuh` / `rope_enc_txtlast.cuh` / `rope_enc_complex.cuh` 对 Q/K 原地应用 RoPE。kernel grid 近似为：

$$
\text{grid} = (\text{batch}, \lceil \text{valid\_seq\_len}/bdy \rceil)
$$

block 维度是 `(bdx, bdy)`：

- `bdx = head_dim / vec_size`，负责一个 head 向量的 hidden 维切片
- `bdy = num_threads / bdx`，一个 CTA 同时处理多个 sequence position
- 每个 thread vectorized load 一段 Q/K、cos、sin
- 对所有 Q heads 和 KV heads 循环执行同一个位置的 RoPE
- text prompt 可以通过 `skip_seq_len` 或 txt-last 版本跳过，不对文本 token 做 video RoPE

这种设计把 RoPE 从大量小 tensor 操作变成连续原地 kernel，论文表 2 报告 RoPE 平均 15.5× 加速。

**Triton permute / inverse permute**

仓库后续 SVG2 也使用 `permute.py`，但它体现了同一类系统思想：将序列维重排变成二维 tile copy。Triton grid 是 `(B*H, ceil(S/BLOCK_S))`，每个 program 处理 `BLOCK_S` 个 token 和完整 `D` 维，生成 `[BLOCK_S, D]` 的 pointer matrix 做 coalesced load/store。SVG1 的 placement kernel则更特化，直接根据 spatial/temporal head 类型计算 store offset。

### FlashInfer patch 的意义

仓库还包含对 FlashInfer `VariableBlockSparseAttentionWrapper.plan` 的 patch：原实现中 variable block sparse 需要用 `torch.repeat_interleave` 展开 variable-length column block 到 token-level `kv_indices`，并把 `kv_indices` 拷回 CPU 做 host assert。SVG 用 Triton `_kvidx_kernel` 在 GPU 上生成连续 index：

$$
\text{kv\_idx} = \text{base} + \text{offset}
$$

并跳过 `kv_indices_host` 的 CPU copy。这属于降低 plan 阶段开销的系统优化，尤其对动态 block sparse / SVG2 更重要，但也反映了 SVG 系列的共同原则：稀疏 attention 的收益不能只看 kernel FLOPs，mask/indices 构造和 host-device 同步也必须压低。

## 实验效果

### 主结果

| 模型/任务 | Dense 延迟 | SVG 延迟 | 加速 | 质量 |
|-----------|------------|----------|------|------|
| CogVideoX-v1.5 I2V 720p 80 frames | 528s | 237s | 2.23× | PSNR 28.165 |
| CogVideoX-v1.5 T2V 720p 80 frames | 528s | 232s | 2.28× | PSNR 29.989 |
| HunyuanVideo T2V 720p 128 frames | 2253s | 1171s | 1.92× | PSNR 29.546 |
| HunyuanVideo T2V + FP8 | 2253s | 968s | 2.33× | PSNR 29.452 |

相比 spatial-only、temporal-only、MInference、PAB，SVG 同时保持更高 PSNR / SSIM / LPIPS 和更低延迟。原因是它不是强行套一种稀疏模式，而是按 head 动态选择 spatial 或 temporal。

### profiling 开销

论文中 1% profiling ratio 已经接近 oracle：CogVideoX-v1.5-I2V 上 PSNR 约 31.1，额外运行开销约 3%。这说明 head 类型可以通过少量 query row 稳定估计，不需要完整 full attention 来决定 mask。

### kernel 贡献

HunyuanVideo 的 runtime breakdown 显示，系统优化是叠加收益：

- sparse attention 是最大贡献，单独带来约 1.81×
- QK-norm / RoPE 自定义 kernel 减少 attention 之外的固定开销
- FP8 attention 对 HunyuanVideo 进一步带来约 1.3×，最终 2.33×

block sparse attention 的 kernel-level benchmark 也说明 layout transformation 是必要的：没有 layout transform 的 naive temporal sparse 实现无法接近理论速度；转换后在不同 sparsity 下更接近 theoretical latency，论文报告额外约 1.7×。

## 关键启示

- **视频 attention 的稀疏性是 head-specific 的**：spatial 和 temporal head 承担不同功能，统一 mask 会明显掉质量
- **动态识别不必 per-token**：按 head 用少量 query row 做 MSE profiling，能在很小开销下接近 oracle
- **稀疏模式必须重排成硬件友好 layout**：temporal slash pattern 理论稀疏但原始 layout 不连续，只有 token-major transformation 后才能真正利用 block sparse kernel
- **端到端加速需要清理非 attention 瓶颈**：QK-norm、RoPE、placement、mask/indices 构造都会变成长视频推理中的显著开销，必须用 CUDA/Triton kernel 处理
- **SVG1 到 SVG2 的演进方向很清楚**：SVG1 解决“head 选 spatial/temporal 模式”和“temporal layout 不连续”；SVG2 进一步解决“基于位置的块划分不够语义准确”和“关键 token 分散导致计算浪费”
