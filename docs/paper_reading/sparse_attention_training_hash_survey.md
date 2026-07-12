---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - LLM Inference
  - KV Cache
---

# 训练型 Sparse Attention 与哈希 Top-K Attention 初步调研

[更新日期 2026-07-12]

## 调研范围

这页不是单篇论文精读，而是给后续选题做一轮**先宽后深**的候选筛查。范围分两部分：

1. **带训练的 sparse attention**：优先视频生成 / DiT，再看图像 DiT、MLLM、LLM；
2. **从哈希角度快速找 Top-K token 做 attention**：优先视频生成，其次 MLLM/LLM。

这里的“带训练”包括三类：

- 稀疏模式本身就是训练内生的一部分，训练和推理使用一致或近似一致的 sparse attention；
- 用蒸馏 / 微调让模型适应稀疏 mask；
- 用可学习 router / selector / hash module 决定 sparse 访问。

这里的“哈希 Top-K”则特指：

- 用 LSH / hash bucket / hash-aware routing 降低候选检索开销；
- 或把 Top-K attention 选择改写成 hash / ANN / MIPS 风格索引问题。

## 先看结论

当前检索到的结果很明确：

- **视频生成里，训练型 sparse attention 已经开始形成一条独立路线**，代表候选包括 `VSA`、`Sparse Forcing`、`Bidirectional Sparse Attention`、`SpargeAttention2`，以及更宽泛但相邻的 `LLSA` / `SparseDiT`。
- **视频生成里，以“哈希”作为主机制的 sparse attention 目前并不多见**。这条线更集中在长上下文 LLM/MLLM 推理，代表候选是 `HashAttention`、`HATA`、`MagicPIG`、`Reformer`，外加 `Quest`/asymmetric indexing 这类近似检索系方法。
- **本仓库已经覆盖的内容主要是 training-free 视频稀疏注意力**，例如 `PISA`、`PASA`、`SVG2`、`SVOO`、`LVSA`、`AdaCluster`，以及长上下文参考 `MInference`。如果接下来要扩展“训练型”与“哈希型”，优先级应当另外开线，不要混在同一类里。

## 本仓库已有覆盖

### 已有视频 sparse attention 笔记

- [PISA](pisa.md)
- [PASA](pasa.md)
- [Sparse VideoGen2 / SVG2](svg2.md)
- [SVOO](svoo.md)
- [LVSA](lvsa.md)
- [AdaCluster](adacluster.md)
- [Sparse Forcing](sparse_forcing.md)

### 已有长上下文 / 训练型参考

- [MInference](minference.md)：training-free，作为模式发现和 kernel-aware 标定参考
- [LLSA](llsa.md)：训练型、log-linear sparse attention，偏图像 DiT / pixel-space DiT
- [HySparse](hysparse.md)：LLM 训练型 / 架构型 sparse attention
- [Native Sparse Attention 代码分析](../code_analysis/native_sparse_attention/00_overview.md)：训练型 LLM sparse attention 的实现参考
- [DeepSeek-V4](deepseek_v4.md)：不是哈希 Top-K 主论文，但包含压缩记忆 top-k 检索与 hash routing 参考

## 一、训练型 Sparse Attention

### 1.1 视频生成 / 视频扩散优先候选

#### VSA: Faster Video Diffusion with Trainable Sparse Attention

- arXiv: `2505.13389`
- 状态：已确认论文存在；代码仓库需要进一步核到官方主页
- 相关性：**高**

这是目前最值得补的候选之一。它的定位非常清楚：不是 training-free patch，而是**让视频扩散模型直接学会在 sparse attention 下工作**。从标题和后续引用关系看，它大概率是视频 DiT 训练型 sparse attention 的代表工作之一。

从方法脉络推测，VSA 处在 `SpargeAttn / PISA` 这类 inference-only 稀疏化与 `Sparse Forcing` 这类 native trainable 稀疏化之间：目标是让稀疏访问模式本身成为模型可适应的训练条件，而不是把 dense 模型硬插 sparse kernel。

后续深挖时应重点核三件事：

- 稀疏模式是否固定族内学习，还是输入自适应；
- 训练时的 sparse pattern 与推理时是否完全一致；
- 是否有针对视频时序一致性的专门约束。

#### Sparse Forcing: Native Trainable Sparse Attention for Real-time Autoregressive Diffusion Video Generation

- arXiv: `2604.21221`
- 状态：本仓库已做笔记；官方代码暂未见稳定公开
- 相关性：**极高**

这篇已经在仓库里，但在本轮 survey 里仍应作为“训练型视频 sparse attention”主轴。它和 PISA/PASA 的最大区别是：

- `PISA/PASA` 是 **training-free inference acceleration**；
- `Sparse Forcing` 是 **native trainable sparse attention**，并把稀疏依赖结构本身当作控制长视频误差传播的手段。

它的 Persistent Block-Sparse Attention（PBSA）把历史记忆拆成：

- 持久锚点 `P_t`：有界、长期保留的高价值历史块；
- 局部块稀疏窗口 `L_t^k`：当前 rollout 附近的短程结构依赖。

这条路线很重要，因为它说明视频生成里“训练型 sparse attention”的价值不只是推理加速，还包括**控制误差传播图**。

#### Bidirectional Sparse Attention for Faster Video Diffusion Training

- arXiv: `2509.01085`
- 状态：已确认论文存在；代码待核
- 相关性：**高**

这篇从标题看不是只做推理，而是明确切到 **video diffusion training**。如果内容属实，它补的是当前仓库相对缺的一块：很多视频 sparse attention 论文强调推理加速，但训练阶段 attention 成本同样巨大，尤其是双向 denoising Transformer。

这类方法通常要回答两个难点：

- 训练时怎么保持双向 attention 里的信息充分性，避免稀疏化后梯度质量掉得太快；
- 稀疏模式怎样与 batch / timestep / frame count 的变化兼容。

如果后续核到代码，应优先补这篇。

#### SpargeAttention2: Trainable Sparse Attention via Hybrid Top-k+Top-p Masking and Distillation Fine-Tuning

- arXiv: `2602.13515`
- 状态：已确认论文存在；代码待核
- 相关性：**高**

从标题就能看出它是直接承接 `SpargeAttention` 路线的二代工作。最值得注意的是两点：

- 它不是单纯 Top-K，而是 **Top-k + Top-p 混合稀疏掩码**；
- 它显式引入 **distillation fine-tuning**，这说明作者把“稀疏模式设计”和“模型适应稀疏模式”分开处理。

这类方法很适合放到“训练型 sparse attention”子类中的 **稀疏掩码蒸馏适配** 路线：

- 第一阶段：设计一个近似 dense attention 的 sparse selector；
- 第二阶段：用蒸馏或短程微调，让原模型适应这个 selector。

对视频生成很有现实意义，因为很多工业模型不会从头重训，但可以接受轻量蒸馏。

### 1.2 邻近的图像 DiT / 通用 Diffusion 候选

#### LLSA: Trainable Log-linear Sparse Attention for Efficient Diffusion Transformers

- arXiv: `2512.16615`
- 状态：本仓库已做笔记
- 相关性：**中高**

虽然它不是视频生成，但它很值得作为训练型 sparse attention 的算法参考。它补的是另一类瓶颈：**Top-K 选择阶段本身还是 O(N^2)**。LLSA 用分层 coarse-to-fine Top-K，把选择复杂度降到 `O(NK)`，总体做到 `O(N log N)`。

这条线的重要性在于，它不是围绕视频时序结构设计，而是围绕**可训练的稀疏选择复杂度**重写算法。后续若把视频 token 组织和 hierarchical search 结合起来，潜力很大。

#### SparseDiT: Token Sparsification for Efficient Diffusion Transformer

- arXiv: `2412.06028`
- 状态：已确认论文存在；代码待核
- 相关性：**中**

这篇更偏 token sparsification，而不一定是标准 attention sparsification，但仍应列为候选，因为很多训练型 DiT 加速会把“少算 token”与“少算 attention”合并考虑。

如果其核心是动态 token pruning / token routing，那么它更适合作为“trainable token sparsity”而不是纯 sparse attention。但对视频生成的可迁移意义仍然存在，尤其是在高分辨率 latent token 数爆炸的场景。

### 1.3 LLM / 长上下文参考

#### Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

- arXiv: `2502.11089`
- 代码：`fla-org/native-sparse-attention`
- 相关性：**高，但偏 LLM**

这篇在仓库已有代码分析。它很重要，因为它代表的是**训练期就对硬件友好的 sparse attention 结构建模**，不是先发明稀疏模式，再想办法写 kernel。

它的三路分解：

- compression branch
- selection branch
- sliding window branch

本质上是在算法层就对 GPU kernel 友好做约束。对于视频生成后续方法，这条路线的参考价值非常高，尤其是在你要求的“CTA 负载均衡、GPU 利用率”这个视角下。

#### HySparse / IndexCache / DeepSeek-V4

这几篇更像补充参照系：

- `HySparse`：训练型混合 sparse/full attention 架构；
- `IndexCache`：跨层复用 indexer，偏工程加速；
- `DeepSeek-V4`：压缩记忆 + top-k retrieval + hash routing，不是纯 sparse attention 论文，但很适合参考“压缩记忆检索”这条线。

如果后续要从视频 sparse attention 继续拓到 MLLM/LLM，这组可作为第二梯队。

## 二、哈希角度的 Top-K Attention

### 2.1 视频生成结论先说清楚

这轮检索里，**没有看到已经形成主流影响力、并且明确把“哈希找 Top-K token”作为核心机制的视频生成 sparse attention 方法**。至少在当前公开论文线索中，视频生成主流还是：

- 聚类 / co-clustering：`SVG2`、`AdaCluster`
- block sparse + 近似：`PISA`、`PASA`
- 时空模式 / anchor / local-global 结构：`LVSA`、`Sparse Forcing`

也就是说，视频生成里大家更多是在利用**空间-时间拓扑**和**attention map 结构先验**，而不是像 LLM 那样走哈希/ANN 检索。

这不是坏消息，反而说明后续有空白方向：如果能把视频 token 的时空局部性和哈希近邻检索结合起来，可能会形成一条新路线。

### 2.2 LLM / MLLM 中较强的哈希候选

#### HashAttention: Semantic Sparsity for Faster Inference

- arXiv: `2412.14468`
- 状态：已确认论文存在；代码待核
- 相关性：**高**

从标题看，这是最直接命中“hash + sparse attention + faster inference”的方法。它值得优先深挖，因为它很可能不是经典 Reformer 那种 LSH bucket attention，而是更现代的**semantic-aware hashing / retrieval sparsity**。

如果后续核到代码，需要重点看：

- hash 是建在 key 空间、query-key 联合空间，还是压缩语义表征空间；
- Top-K 是 hash bucket 内再精排，还是 hash 本身直接给候选；
- kernel 实现是按 bucket 分组、还是转回 block-sparse 结构执行。

#### HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference

- arXiv: `2506.02572`
- 状态：已确认论文存在；代码待核
- 相关性：**极高**

这是本轮检索里最贴近你问题表述的一篇。它的标题几乎就是：

- trainable
- hash-aware
- top-k attention
- hardware-efficient

如果它有官方代码，优先级应该很高。因为它把三件事绑在一起了：

- 如何近似找 Top-K；
- 如何让近似过程可训练；
- 如何把最终稀疏结构映射到高效 kernel。

这篇很可能是连接“哈希检索算法”和“GPU 友好 sparse attention 实现”的关键节点。

#### MagicPIG: LSH Sampling for Efficient LLM Generation

- arXiv: `2410.16179`
- 状态：已确认论文存在；代码待核
- 相关性：**高**

MagicPIG 明确用 `LSH sampling` 做 LLM generation 加速。它不一定是标准 sparse attention paper，但从“哈希快速筛候选”的角度，它非常值得纳入。

它更像把“访问哪些历史 token / KV”改写成近似近邻采样问题。对于视频生成的迁移意义在于：

- 如果视频 token 有稳定语义簇，LSH 可以成为候选检索器；
- 之后再用 block / cluster / local window 做精排。

#### Reformer: The Efficient Transformer

- 经典基线
- 相关性：**中**

Reformer 不是最近工作，但任何“哈希 attention”调研都绕不过去。它用 LSH 把相似 query/key 放进同 bucket，只在桶内做 attention。虽然它和现代生成模型的 kernel/系统实践差距很大，但它提供了最经典的 conceptual baseline：

- 哈希不是为了近似 softmax 值本身；
- 哈希是为了先缩小候选集合，再在小集合里做精确 attention。

对于写 survey，这是必须提到的祖先方法。

### 2.3 哈希相邻但不完全等价的检索路线

#### Quest

Quest 更准确说是**上界打分 / 近似候选选择**，不一定是哈希。但它在视频稀疏注意力中已经产生了明显影响，例如 `AdaCluster` 里的 `TensorQuest` 就是把 Quest 风格簇打分重写成 Tensor Core 友好的矩阵乘。

因此，Quest 应该放在“哈希相邻路线”里：不是 hash bucket，但也是在做**比 dense 全对打分更便宜的 Top-K 候选筛选**。

#### Asymmetric Indexing / ANN / MIPS 路线

这条线目前更多出现在长上下文 LLM 检索式 attention 工作中。它们的共同点是：

- 不直接对所有 key 做完整打分；
- 先构造某种 index；
- query 时用低成本 index 检索少量候选，再做精排。

这类方法未必使用显式哈希，但和“从哈希角度快速找 Top-K token”在工程目的上是同一类问题。后续若继续深挖，建议把它们和 `HashAttention / HATA / MagicPIG` 一起比较。

## 三、按优先级给出下一步候选

### A 档：最值得下一步补正式笔记

- `VSA (2505.13389)`：视频生成，训练型 sparse attention，领域相关性最高
- `HATA (2506.02572)`：hash-aware top-k attention，问题贴合度最高
- `HashAttention (2412.14468)`：哈希语义稀疏推理，适合建立 hash 路线主线
- `Bidirectional Sparse Attention (2509.01085)`：补“视频扩散训练加速”这块空白

### B 档：很值得补，但作用偏补充

- `SpargeAttention2 (2602.13515)`：蒸馏式训练适配 sparse mask
- `MagicPIG (2410.16179)`：LSH 检索型 generation 加速
- `SparseDiT (2412.06028)`：更偏 token sparsification，但可迁移
- `Native Sparse Attention (2502.11089)`：LLM 训练型 sparse attention 的硬件实现参考

### C 档：作为对照和背景

- `Reformer`：经典 LSH attention 基线
- `Quest`：上界筛选 / 候选检索参考
- `DeepSeek-V4`：压缩记忆 top-k retrieval + hash routing 的系统参考

## 四、当前空白与判断

### 训练型视频 sparse attention 的主矛盾

和 training-free 方法不同，训练型方法必须同时解决三件事：

- 训练时稀疏图是否稳定，梯度是否可学；
- 推理时 sparse pattern 是否和训练一致；
- 稀疏模式是否足够硬件友好，值得真正写 kernel。

也因此，视频训练型 sparse attention 的论文数量还不算多，但一旦做得好，价值会明显高于 inference-only patch。

### 哈希路线在视频生成里的空白

当前公开视频生成 sparse attention 论文更喜欢用：

- 聚类
- block 重要性
- local-global 模式
- 持久锚点

而不是 hash / LSH / ANN。一个合理判断是：

- 视频 token 的自然结构太强，空间-时间先验已经足够好用；
- 而哈希更适合无明显局部拓扑的长上下文文本 KV 检索。

但这也说明，**面向视频 token 的 hash-aware Top-K attention 仍然是潜在研究空白**。

## 五、建议的后续顺序

如果按你当前仓库的组织方式继续推进，建议顺序是：

1. 先补 `VSA`：把训练型视频 sparse attention 的主线立起来；
2. 再补 `HATA`：把哈希 Top-K attention 的主线立起来；
3. 然后补 `HashAttention` 与 `MagicPIG`：把哈希检索的两种不同实现路线串起来；
4. 最后再看 `SpargeAttention2` 和 `Bidirectional Sparse Attention`，补齐“蒸馏适配”和“训练加速”两个分支。

这样仓库里的 sparse attention 结构会更完整：

- training-free video sparse attention
- trainable video sparse attention
- long-context sparse attention
- hash / ANN / retrieval-style sparse attention

## 六、当前核验状态说明

本页基于以下已确认信息撰写：

- `VSA`、`Sparse Forcing`、`Bidirectional Sparse Attention`、`SpargeAttention2`、`SparseDiT`
  的 arXiv 条目已检到；
- `HashAttention`、`HATA`、`MagicPIG`、`Reformer`
  的 arXiv 条目已检到；
- 本仓库已有内容已做本地交叉检查，避免与现有 `PISA / PASA / SVG2 / SVOO / LVSA / AdaCluster / MInference / LLSA / Sparse Forcing / Native Sparse Attention` 重复。

仍待下一轮精查的部分：

- `VSA / SpargeAttention2 / Bidirectional Sparse Attention / HashAttention / HATA / MagicPIG`
  的官方代码仓库链接；
- 各方法是否已有 Triton / CUDA / FlashInfer / ThunderKittens 等实现；
- 视频生成方向是否有尚未被本轮检索命中的 hash-aware Top-K 新工作。
