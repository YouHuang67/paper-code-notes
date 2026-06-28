---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
---

# LVSA: Training-Free Sparse Attention for Long Video Diffusion

- 论文：https://arxiv.org/abs/2605.31057
- 代码：https://github.com/JiusiServe/LongVideoSparseAttention
- 团队：Huawei Paris Research Center, Huawei Technologies

## 概述

LVSA 解决的是长视频扩散推理里的两个耦合问题：一是 3D self-attention 随帧数平方增长，长视频显存和延迟不可接受；二是视频长度超过训练 horizon 后，dense attention 反而容易生成近似静止或循环的视频。论文的核心判断是：长视频推理不一定需要所有帧两两互看，关键是给每个 query frame 保留稳定的全局锚点和足够的局部时序上下文。

方法上，LVSA 采用 training-free 的 frame-level block sparse attention。每个 query frame 只关注两类 key frame：

- **全局锚点帧**：开头若干帧 + 周期性 keyframe，承载全局场景和主体信息
- **局部窗口帧**：query frame 周围的滑动窗口，承载短程运动连续性

关键改进是 **rotating global anchors**：周期性 keyframe 的网格在每个 denoising step 平移，使不同帧轮流成为全局锚点。这样避免固定 keyframe 网格带来的长程偏置，也缓解 dense attention 在超长 rollout 中的 frozen / looping 失败模式。

代码实现上，LVSA 不是动态 token 选择，而是先把 frame 级稀疏模式编译成 `LVSAMetadata`：包括每个 query frame 的窗口上下文、全局帧索引、FlashInfer block-sparse CSR、compact KV copy plan。推理时通过 diffusers attention processor 替换模型原始 attention；短序列走 per-frame SDPA，长序列走 FlashInfer `BlockSparseAttentionWrapper`，将被访问的 K/V frame 压入 compact buffer 后一次 block-sparse kernel 完成。

实验上，LVSA 在 Wan 2.1 和 HunyuanVideo 1.5 上保持或提升长视频质量，同时显著加速：Wan 2.1 1.3B 在 6× horizon 下 LVSA-FI 3.17× 加速，Wan 2.1 14B 2.98×，HunyuanVideo 1.5 在 1.5× horizon 3.33×，并能生成 dense attention 在 80GB GPU 上 OOM 的 2× horizon 视频。

## 问题设定

视频 DiT 将 latent video patchify 成长度为 $N = T \cdot P$ 的 token 序列：

- $T$：latent temporal frames
- $P = H_p \cdot W_p$：每个 latent frame 内的空间 patch 数
- dense self-attention 对每个 query token 访问所有 $T \cdot P$ 个 key token

长视频推理时，计算量和 attention activation 都随 $T^2$ 增长。Wan 2.1 14B、HunyuanVideo 1.5 这类模型在 80GB 单卡上已经接近显存边界，继续拉长帧数时很容易 OOM。

论文还指出 dense attention 的质量问题：超过训练 horizon 后，视频会趋向静态或循环。VBench-Long 的 subject/background consistency 会奖励这种“静止一致性”，所以论文引入 VQeval，显式惩罚 dynamic 和 loop failure。

## 稀疏模式

### 基础形式：Global Anchors + Local Window

对 query frame $t$，LVSA 的可见 key frame 集合为：

$$
A(t) = G \cup W(t)
$$

其中：

- $G$：全局锚点帧集合，包括开头固定帧和周期性 keyframe
- $W(t)$：以 $t$ 为中心的局部窗口，半宽为 $W$

如果直接用窗口 $[t-W, t+W]$，边界位置会因为 clipping 得到更小 attention budget。代码里的 `adaptive_window_bounds` 先把边界窗口整体平移，使每个 frame 尽量看到固定数量的窗口帧。

### Expanded Window：补偿全局帧重叠

局部窗口可能和全局锚点重叠。如果重叠帧已经在 $G$ 中，再把它算进窗口预算会浪费可见帧数。LVSA 因此使用 expanded window：先计算基础 adaptive window，再向左右扩展，直到窗口内非全局帧数量达到目标。

代码对应 `lvsa/sparse_attention.py`：

- `adaptive_window_bounds(f, W, T)`：边界处平移窗口，保持宽度
- `expanded_window_bounds(f, W, T, global_set, global_count)`：统计窗口内 non-global frame，不够就向外扩展
- `get_window_bounds(...)`：根据 `expand_window` 开关选择 expanded / adaptive

这个设计让每个 query frame 的总可见帧数更稳定，避免全局锚点密集区域浪费 attention budget。

### Rotating Global Anchors

固定周期 keyframe 会产生固定网格偏置：始终成为 global anchor 的帧获得过强长程传播，中间帧只能通过局部窗口被访问。LVSA 在 denoising step $s$ 使用 offset：

$$
G_s = \{0,\ldots,n_{\text{first}}-1\} \cup \{(s + i \cdot \text{kfi}) \bmod T\}
$$

这样每个 denoising step 的 global keyframe 网格都平移一格。经过多个 denoising step 后，所有帧都有机会成为全局锚点。

代码实现：

- `compute_global_indices(..., offset)` 用 modular wrapping 构造旋转后的周期 keyframe
- `DistributedLVSAProcessor.set_step(step_idx)` 计算 `offset = step_idx % key_frame_interval`
- offset 变化时调用 `_rebuild_for_current_params(offset)`，重建 `LVSAMetadata`、CSR、copy plan 和 FlashInfer plan 状态

### Auto Keyframe Scheduler

LVSA 希望每个 query frame 的可见帧数大致接近模型训练时的 latent frame 数。代码中的 `compute_auto_kfi` 使用：

- `reference_frames`：模型原生训练 horizon 的 latent frame 数，如 Wan 81 raw frames 对应 21 latent frames
- `sparsity_scale`：预算缩放，小于 1 更稀疏，大于 1 更保守
- `window_size` 和 `n_first_frames`：决定局部窗口与固定开头帧预算

当 $T \leq$ 预算时，`kfi=1`，所有帧都成为 global，相当于 dense。超过训练长度后，scheduler 选择尽量大的 keyframe interval，在保证目标 global 数量的同时保持稀疏。

## 代码实现

LVSA 的实现不是“写一个新的 attention kernel 然后强行接模型”，而是三层解耦：

1. **模型接入层**：从不同 DiT 的 attention block 中取出 Q/K/V、应用 RoPE、做输出投影，并把原 attention processor 替换为 LVSA processor。
2. **稀疏模式编译层**：把论文里的 `global anchors + local window + rotating offset` 编译成 frame-block 级 CSR、compact KV copy plan 和窗口边界。
3. **执行层**：短序列或 NPU 走逐帧 SDPA fallback；CUDA 长序列走 FlashInfer block-sparse attention。

需要先明确一点：开源 LVSA 仓库本身没有自写 CUDA/Triton kernel。论文里所谓 LVSA-FI 的高性能路径，是把 LVSA 的稀疏图组织成 FlashInfer `BlockSparseAttentionWrapper` 接口，由 FlashInfer 内部 CUDA kernel 执行。LVSA 代码真正关键的工程贡献，是如何把视频 attention 映射成 FlashInfer 能高效吃下的 block-sparse 问题，而不是在 Python 层循环省 FLOPs。

### 模型接入：只抽象 attention 语义，不污染稀疏核心

不同视频 DiT 的 attention 接口差异很大：

- Wan 是 single-stream，视频 token 和 text condition 进入同一套 attention 逻辑
- HunyuanVideo 是 dual-stream，文本/视觉 encoder token 和视频 token 有不同投影
- Cosmos 3.0 甚至是 separate-stream，需要保留 understanding token 的 causal 分支，只替换 generation token 的 full attention

LVSA 通过 adapter / processor-swap 两种方式接入这些模型。adapter 只负责模型私有语义：如何算每帧 token 数 $P$、raw frame 如何转 latent frame、Q/K/V 如何投影、RoPE 怎么切片、输出怎么投影回模型 hidden state。稀疏核心只接收统一形状：

$$
Q,K,V \in \mathbb{R}^{B \times (T\cdot P) \times H \times D}
$$

这样做的好处是，稀疏策略完全在 frame geometry 上定义。只要某个模型能给出 $T$、$P$ 和 Q/K/V，LVSA 就能复用同一套窗口、CSR 和 FlashInfer 执行逻辑。

### 从 dense attention 到 frame-block sparse attention

Dense video attention 可以看成一个 $T \times T$ 的 frame-pair block grid。每个 grid cell 是一个 dense token block：

$$
Q_t \in \mathbb{R}^{P \times H \times D}, \quad K_\tau,V_\tau \in \mathbb{R}^{P \times H \times D}
$$

如果全连接，每个 query frame $t$ 要访问全部 $T$ 个 key frame，计算规模近似：

$$
O(T^2 \cdot P^2 \cdot H \cdot D)
$$

LVSA 把 block grid 的每一行裁成：

$$
A(t)=G_s \cup W(t)
$$

其中 $|A(t)| \approx C$，$C$ 被 auto-keyframe scheduler 控制在接近训练 horizon 的 latent frame 数。于是计算变成：

$$
O(T \cdot C \cdot P^2 \cdot H \cdot D)
$$

这就是加速的根本来源：不是降低一个被选中 frame-pair 内部的空间 attention 复杂度，而是跳过大多数 frame-pair block。对于 Wan 2.1 1.3B 的 6× horizon，$T$ 远大于训练参考长度，$C$ 近似保持有界，因此速度随视频长度拉开。

### 稀疏图编译：把 `G ∪ W(t)` 变成 CSR

运行时不能每层每步都在 Python 里判断“第 t 帧看哪些帧”，否则 kernel 省下来的时间会被调度开销吃掉。LVSA 的做法是每当稀疏 pattern 改变时，一次性构造 metadata。

关键是 `_build_flashinfer_csr` 的逻辑：

- 遍历本 rank 的 query frame block row
- 对每一行，根据当前 rotating offset 得到 global set，再加 expanded window
- 收集该行实际访问的 key frame
- 汇总所有被访问 key frame，压成一个 compact key-frame 空间
- 用 CSR 记录每个 query frame row 访问 compact 空间中的哪些 column block

CSR 的语义是 frame block 级：

- `indptr` 长度为 `MB + 1`，`MB` 是本 rank 的 query frame block 数
- `indices` 保存每个 query frame 要访问的 compact key-frame block id
- 一个 row block 对应一个 query latent frame
- 一个 column block 对应一个 key latent frame
- FlashInfer plan 中设置 `R = C = P`，表示逻辑 block 的高和宽都是一帧内的 spatial token 数

这里的 `R = C = P` 不是说一个 CUDA CTA 直接粗暴计算 $P \times P$ 的完整大矩阵。它是传给 FlashInfer 的**逻辑块大小**：LVSA 在 block-sparse 图上把“一帧对一帧”的可见关系交给 FlashInfer；FlashInfer 内部再按自己的 tiled attention kernel 切分 token tile、做 online softmax 和 value accumulation。LVSA 能控制的是 block grid 稀疏度和 K/V 内存布局，不直接控制 FlashInfer 内部 CTA 形状。

因此，CUDA 层面的工作量从 dense 的 $T \times T$ 个 logical frame-pair blocks，变为 $T \times C$ 个 nonzero frame-pair blocks。FlashInfer 内部只为 CSR 中存在的 block column 发起计算，不为被跳过的 frame-pair 生成 score tile，也不构造完整 $N \times N$ mask。

### compact KV：让非零 block 变成连续内存

只给 kernel 一个 CSR 还不够。如果 K/V 仍然散落在原始 `[T*P]` 序列中，访问全局帧和窗口帧会产生大量不连续 gather，吞掉带宽收益。LVSA 因此在 CSR 构造时同步生成 compact KV layout：

1. `compact_frames = sorted(all_attended)` 收集本次 attention 调用会被访问的所有 key frame
2. `frame_to_compact` 把原始 frame id 映射到 compact column id
3. `fi_global_copies` 描述 global K/V frame 从 `k_global` 复制到 compact buffer 的位置
4. `fi_local_copies` 描述 local non-global K/V frame 从原始 K/V 复制到 compact buffer 的位置

运行时 `_compute_lvsa_flashinfer` 先填充：

$$
K_{\text{compact}}, V_{\text{compact}} \in \mathbb{R}^{B \times (N_c\cdot P) \times H_{kv} \times D}
$$

其中 $N_c$ 是本 rank 实际被访问的 compact frame 数。之后 FlashInfer kernel 看到的是一个连续的 K/V 空间，CSR 里的 column id 也都是 compact id。这样做有两个直接收益：

- 避免在 CUDA kernel 内部按原始 frame id 做散乱索引
- 减少 K/V resident buffer 宽度，只保留本次 sparse pattern 真正会访问的 frame

这个设计解释了为什么 LVSA-FI 比 SDPA 版本快。SDPA 版本虽然也减少了可见帧，但它逐 query frame 拼接 `k_ctx/v_ctx` 并多次 dispatch；FlashInfer 版本把整层 attention 变成一次 block-sparse kernel 调用，减少 Python 循环、kernel launch 和临时拼接开销。

### FlashInfer plan 复用：把调度开销摊到多个层

FlashInfer block-sparse kernel 需要先 `plan(indptr, indices, M, N, R, C, heads, head_dim, dtype)`。plan 的输入本质上定义了 CUDA kernel 要执行的 block-sparse grid：

- `M = MB * P`：query token 数按 frame block 对齐
- `N = compact_n * P`：compact K/V token 数
- `R = C = P`：logical block 是 frame-to-frame
- `num_qo_heads` / `num_kv_heads`：分别传 query heads 和 KV heads，原生支持 GQA，避免提前 repeat KV
- `head_dim` 和 dtype 决定底层 attention kernel 模板

plan 不是每层都重建。代码用 `_FIState` 缓存 wrapper、128MB workspace、compact K/V buffer 和 padded Q buffer；vLLM-Omni / Cosmos 的 runner 进一步做 process-wide singleton，让所有层共享同一个 runner。因为同一次 denoising step 内，各 transformer layer 的 attention geometry 相同，CSR 和 compact buffer shape 也相同，复用 plan 可以避免每层重复规划和重复申请大块 workspace。

旋转 keyframe 会改变 `indices`，所以 `set_step(step_idx)` 在 offset 变化时重建 `LVSAMetadata`，并重置 FlashInfer plan。这个开销只发生在 denoising step 粒度，而不是每个 attention layer 内重新推导稀疏图。

### text / encoder token 的 LSE merge：保持 softmax 正确

很多视频 DiT 的 attention 不只有 video token，还会有 text 或 understanding token。如果简单把这些 token 塞进 frame-block CSR，需要把 text length padding 到 $P$ 的整数倍；padding 出来的 zero key 会进入 softmax denominator，导致每个 query 的注意力被 phantom key 稀释。

LVSA 的实现没有这么做。FlashInfer 路径把 key set 拆成两个互不相交的部分：

- video K/V：走 LVSA block-sparse CSR
- text / encoder K/V：走 dense `single_prefill_with_kv_cache`

两边都返回 output 和 log-sum-exp。最终用 online softmax 的合并公式精确合并：

$$
O = \frac{e^{l_v}O_v + e^{l_t}O_t}{e^{l_v}+e^{l_t}}
$$

FlashInfer 返回的 LSE 是 log2 标度，所以代码里用 `exp2` 做权重。这个细节非常关键：它保证“video token 稀疏、text token 全保留”仍然等价于在两个 disjoint key set 上做一次统一 softmax，而不是近似相加。

### SDPA fallback 为什么不是主要加速路径

SDPA fallback 的实现更直观：每个 query frame 单独取 `q_chunk`，拼接 global K/V 和 window K/V，然后调用 PyTorch / diffusers attention。这个路径的价值是稳定、易移植、可跑 CUDA/NPU/CPU，但它有明显开销：

- 每个 frame 一次 attention dispatch
- 每次要拼接上下文 K/V
- 无法像 FlashInfer 一样把整层 attention 表达为一个 block-sparse grid

所以论文主表中长 horizon 的最大收益来自 LVSA-FI，而不是 SDPA 版本。SDPA 更像 correctness fallback 和短序列路径；真正接近 CUDA 级加速的是 CSR + compact KV + FlashInfer plan/run 这条路径。

### 多 GPU路径的核心约束

LVSA 的 sparse pattern 是 sequence/frame 维度上的，而不是 head 维度上的。多 GPU 时要保证每个 rank 计算本地 Q 时，能看到自己需要的 K/V frame：

- `custom` 模式切 sequence shard，本地 Q 只算本 rank token；global K/V 复制到所有 rank，rank 边界附近加入 boundary guard frame，避免 local window 被 shard 边界切断
- `ulysses` 模式先 all-to-all，把完整 sequence grid 汇聚到每个 head shard 上，再执行单机同构 LVSA，最后 scatter 回去
- `ring` 模式让 K/V shard 在 rank 间环形传递；每到一个 K/V shard，就用当前 Q 对这个 shard 中符合 `G ∪ W(t)` 的 frame-pair 做 attention，并通过 LSE merge 合并多个 shard 的结果

这些路径说明 LVSA 的工程抽象不是“mask 掉一部分 attention”这么简单，而是要保证稀疏图、K/V 数据分布和 softmax 归一化在单卡、多卡、dual-stream 模型里都一致。

## 实验效果

### 延迟与显存

论文在单 80GB GPU 上测试 HunyuanVideo 1.5、Wan 2.1 1.3B、Wan 2.1 14B。

| 模型 | Horizon | Frames | Dense | LVSA-FI | 加速 |
|------|---------|--------|-------|---------|------|
| HunyuanVideo 1.5 | 1.5× | 193 | 79.7 min | 23.9 min | 3.33× |
| HunyuanVideo 1.5 | 2× | 257 | OOM | 54.9 min | dense 不可运行 |
| Wan 2.1 1.3B | 2× | 161 | 7.3 min | 5.1 min | 1.42× |
| Wan 2.1 1.3B | 4× | 321 | 24.0 min | 10.3 min | 2.32× |
| Wan 2.1 1.3B | 6× | 481 | 50.8 min | 16.0 min | 3.17× |
| Wan 2.1 14B | 6× | 481 | 237.9 min | 79.8 min | 2.98× |

趋势很清楚：训练 horizon 附近，LVSA 可能只有持平甚至略慢，因为稀疏调度和 CSR/compact buffer overhead 还没有被长序列摊薄；序列越长，dense attention 的二次复杂度越明显，LVSA 的收益越大。

### 质量

在 1× training horizon，LVSA 基本质量中性，VQeval composite 和 dense 差距在 1 分以内。

超过训练 horizon 后，LVSA 的优势反而扩大：

- Wan 2.1 1.3B：LVSA-FI 相对 dense 的 VQeval 提升从 2× 的 +4.7 增至 6× 的 +12.1
- Wan 2.1 14B：2× +3.8，4× +9.7，6× +12.2
- HunyuanVideo 1.5：2× horizon 下 dense OOM，LVSA 仍能生成视频

论文特别强调 VBench-Long 和 VQeval 的分歧：dense video 在超长 horizon 变静止后，VBench 的 subject/background consistency 会升高，但 VQeval 会惩罚 lack of motion 和 looping。LVSA 的 sliding window + rotating anchor 更能保持长视频运动。

## 与其他稀疏注意力方法的区别

- **不是 token 级 top-k**：LVSA 不根据当前 QK 分数动态选择 token，而是使用 frame block 级结构模式，overhead 更低
- **不是固定局部窗口**：保留全局锚点，避免长程语义断裂
- **不是固定 keyframe grid**：旋转 keyframe 让每帧都有机会成为 global anchor，减少固定网格偏置
- **不是训练型稀疏**：不改权重，不需要蒸馏或 LoRA，可作为 attention processor 插入现有 DiT
- **实现偏系统工程**：核心收益依赖 CSR、compact KV、FlashInfer plan 复用和 processor/adapter 接入，而不是仅靠数学 mask

## 关键启示

- **长视频稀疏可以从 frame 级结构入手**：视频 token 天然有 frame block 结构，按 frame 组织稀疏模式比逐 token 选择更容易映射到高效 kernel
- **全局锚点必须动态化**：固定 keyframe 会把长程信息传播责任压到少数帧上，旋转锚点用 denoising step 维度摊平这个偏置
- **稀疏模式要和 kernel 表达一致**：LVSA 的真正加速来自把 `G ∪ W(t)` 编译成 block CSR 和 compact KV，而不是构造 dense mask 后交给普通 attention
- **文本/视频 token 混合要谨慎处理**：代码中将 encoder/text tokens 独立 dense attention 后用 LSE merge，而不是 padding 到 frame block，说明工程细节会直接影响 softmax 正确性
- **质量评估要识别“静止一致性”陷阱**：长视频里静止不等于好，评估指标必须惩罚 frozen / loop failure，否则会误判 dense attention 的长程退化
