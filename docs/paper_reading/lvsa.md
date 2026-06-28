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

### 1. Adapter 把不同模型接入统一 LVSA engine

仓库把模型相关逻辑和稀疏 attention 逻辑分开：

- `lvsa/adapters/base.py` 定义 `ModelAdapter`
- `lvsa/adapters/wan.py`、`hunyuan_video.py`、`cogvideox.py` 分别适配不同模型
- `lvsa/lvsa_processor.py` 只处理稀疏模式、KV 收集、并行通信和 backend dispatch
- `lvsa/sparse_attention.py` 保存无状态的窗口/CSR/attention primitive

以 Wan 为例，`WanAdapter` 负责：

- `patches_per_frame`：根据 VAE spatial scale 和 patch size 算每帧 token 数 $P$
- `latent_frames`：用 VAE temporal factor 将 raw frame 数转成 latent frame 数
- `reference_latent_frames`：默认 21，即 81 raw frames 的训练 horizon
- `extract_qkv`：调用 diffusers 的 `_get_qkv_projections`，做 QK norm 并 reshape 成 `[B, seq, H, D]`
- `apply_rotary`：对 context parallel rank 切片后的 RoPE 做旋转
- `install_processor`：把 `block.attn1.processor` 替换成 LVSA processor

这个适配层的意义是：LVSA 的稀疏模式只依赖 frame geometry，不依赖某个模型的 QKV 投影细节。

### 2. LVSAMetadata 预计算所有索引结构

`LVSAMetadata.build(...)` 是核心入口。它从 `T, P, W, n_first_frames, kfi, rank, world, offset` 推导出三类结构：

**窗口结构**

- `global_indices` / `global_set`：当前 denoising step 的全局帧
- `local_frames`：当前 rank 拥有的 query frame 和 token range
- `window_ctx`：每个 local query frame 对应哪些本地 K/V token range
- `window_bounds`：每个 frame 的窗口边界

**SDPA / indexed kernel 结构**

- `attended_indices`：每个 local frame 的可见 frame 列表，padding 到统一长度 `C`
- `global_src_idx` / `global_dst_idx`：把 global K/V 放入统一位置的 copy plan
- `local_src_idx` / `local_dst_idx`：把 local non-global K/V 放入统一位置的 copy plan

**FlashInfer 结构**

- `fi_indptr` / `fi_indices`：block-sparse CSR，block 行是 query frame，block 列是 compact key frame
- `fi_M = MB * P`：padded query token 数，按 frame block 对齐
- `fi_N = compact_n * P`：实际被访问的 compact K/V token 数
- `fi_global_copies` / `fi_local_copies`：将 global 或 local K/V frame 复制到 compact buffer 的指令

因此，运行时不需要为每个 query token 重新做重要性判断，也不需要构造完整 $N \times N$ mask。稀疏模式是 frame block 级的静态结构，旋转 keyframe 时才重建。

### 3. SDPA backend：逐 frame 拼接上下文

`lvsa_sdpa` 是简单稳定的 fallback，适合较短序列、NPU 或没有 FlashInfer 的环境。

对每个 local query frame：

1. 取出该 frame 的 query chunk
2. 将 `k_global/v_global` 和窗口内本地 K/V token range 拼接成 `k_ctx/v_ctx`
3. 调用 diffusers `dispatch_attention_fn` 或 PyTorch `scaled_dot_product_attention`
4. 将输出写回原 query token 区间

这个路径实现简单，但每个 frame 都会单独 dispatch，并且上下文拼接有额外开销。长序列下真正的加速主要来自 FlashInfer backend。

### 4. FlashInfer backend：CSR + compact KV

FlashInfer 路径的关键在 `_build_flashinfer_csr` 和 `_compute_lvsa_flashinfer`。

`_build_flashinfer_csr` 先遍历每个 query frame：

- 收集该 frame 可见的 global frame 和 local window frame
- 汇总所有被访问的 key frame，形成 compact frame layout
- 为每个 query frame 写入 CSR row：它要访问 compact layout 中哪些 frame block
- 生成 copy list，把原始 K/V 中的 frame block 搬到 compact K/V buffer

`_ensure_flashinfer_planned` 使用这些 CSR 调用：

- `flashinfer.BlockSparseAttentionWrapper`
- block size `R = C = P`，即一个 attention block 对应一个 latent frame 的所有 spatial tokens
- `num_qo_heads` 和 `num_kv_heads` 分开传入，支持 GQA，不需要预先 repeat KV
- 128MB workspace 复用，pattern 不变时 plan 也复用

`_compute_lvsa_flashinfer` 每次 attention 调用时：

1. 按 `fi_global_copies` 将 global video K/V 复制到 compact buffer
2. 按 `fi_local_copies` 将 local non-global K/V 复制到 compact buffer
3. query 长度不足 frame block 整数倍时 pad 到 `M = MB * P`
4. 对 video token 调用 FlashInfer block-sparse kernel
5. 如果模型有 text / encoder tokens，不把它们硬塞进 block CSR，而是单独跑 dense attention
6. 用 log-sum-exp merge 将 video sparse 输出和 text dense 输出精确合并

第 5 点很重要：早期实现如果把 text token padding 成 frame block，会引入 phantom zero key，稀释 softmax。当前代码把 encoder/text K/V 作为单独 dense term，再用 FlashInfer 返回的 LSE 做精确合并，避免 padding 伪影。

### 5. Context Parallel 支持

LVSA 还实现了多 GPU 路径：

- `custom`：sequence shard，每个 rank 负责本地 Q；global K/V 通过 all-reduce / gather 复制；rank 边界附近用 boundary guard frame 避免窗口断裂
- `ulysses`：all-to-all 把完整 frame grid 重建到每个 rank 的 head shard 上，执行单机同构 LVSA，再 scatter 回去
- `ring`：K/V 在 rank 间环形旋转，每个 block-pair 用 mask 或 FlashInfer CSR 计算，再用 online softmax / LSE merge 合并

论文主结果主要强调单 80GB GPU，但代码已经把 LVSA 设计成可服务化的稀疏 attention engine，并提供 vLLM-Omni 插件。

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
