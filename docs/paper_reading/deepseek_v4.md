---
tags:
  - Sparse Attention
  - LLM Inference
  - KV Cache
---

# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

- 论文：`refs/papers/DeepSeek_V4.pdf`（本地 PDF）
- 代码：https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference
- 模型：https://huggingface.co/collections/deepseek-ai/deepseek-v4
- 团队：DeepSeek-AI

## 概述

DeepSeek-V4 解决的不是“模型再大一点”的问题，而是 **reasoning model 的 test-time scaling 如何突破超长上下文的算力与缓存墙**。标准 Transformer 的 attention 在序列极长时会同时遭遇三类瓶颈：

- **计算瓶颈**：每个 query 都要面对过长的历史 key 集合
- **存储瓶颈**：所有历史 token 的 KV 都要完整保留
- **训练与部署瓶颈**：一旦为了省算力引入更复杂的稀疏/压缩/路由结构，稳定训练、可复现实验、可部署推理就会迅速变难

DeepSeek-V4 的整体思路，是把模型重构成一个**分层压缩记忆系统**：

- 在残差路径上，用 **mHC (Manifold-Constrained Hyper-Connections)** 代替普通 residual，使深层信号传播更稳定
- 在注意力路径上，用 **CSA (Compressed Sparse Attention)** 与 **HCA (Heavily Compressed Attention)** 交错，把“历史序列”改写成“压缩记忆 + 稀疏检索 + 局部精修”
- 在优化路径上，用 **Muon** 处理大部分矩阵参数，使收敛更快、数值更稳
- 在系统路径上，把 KV cache、FP4、contextual parallelism、deterministic kernel、磁盘前缀复用等都作为架构的一部分来协同设计

最终结果是：DeepSeek-V4-Pro（1.6T 参数，49B activated）与 DeepSeek-V4-Flash（284B 参数，13B activated）都原生支持 1M token context。论文最关键的效率结论是：在 1M context 场景下，**DeepSeek-V4-Pro 的单 token inference FLOPs 只有 DeepSeek-V3.2 的 27%，KV cache 只有 10%；DeepSeek-V4-Flash 进一步降到 10% FLOPs 与 7% KV cache**。

如果用一句话概括本文的技术主线：**DeepSeek-V4 不是单纯做 sparse attention，而是把 Transformer 从“逐 token 全量交互”改造成“多时间尺度的压缩记忆访问系统”。**

## 行文说明

为了控制详略，本文采用“**正文讲总体概念与数学建模，附录承接工程细节与配置补充**”的组织方式。

- 正文重点：
  - [第 1 节：问题定义与设计目标](#sec-1-problem)
  - [第 2 节：总体架构视角](#sec-2-overview)
  - [第 3 节：mHC 数学建模](#sec-3-mhc)
  - [第 4 节：混合注意力的统一建模](#sec-4-hybrid-attn)
  - [第 5 节：Muon 与训练稳定性](#sec-5-muon)
  - [第 6 节：将各模块拼成一个完整系统](#sec-6-synthesis)
- 附录补充：
  - [附录 A：符号表](#appendix-a-symbols)
  - [附录 B：模型配置与参数规模](#appendix-b-configs)
  - [附录 C：预训练与稳定性细节](#appendix-c-training)
  - [附录 D：系统与基础设施细节](#appendix-d-systems)
  - [附录 E：后训练与 agent 化能力](#appendix-e-posttraining)
  - [附录 F：评测补充](#appendix-f-eval)
  - [附录 G：后续代码解析路线图](#appendix-g-code-roadmap)

正文里所有“完整超参”“系统实现”“后训练流水线”“更细实验数据”，都会显式交叉引用到相应附录。

## 1. 论文要解决什么问题 {#sec-1-problem}

### 1.1 从 test-time scaling 看长上下文瓶颈

reasoning model 的一个基本趋势是：**推理时给更多上下文、更多思考 token、更多中间工具调用，通常能显著提高性能**。但这条路线有一个硬上限：标准 attention 的成本会随上下文长度迅速膨胀。

如果把标准自回归 attention 简化看成“每个 query token 都要和前面全部 key token 交互”，那么当上下文从 32K 拉到 1M 时，模型面临的不是线性困难，而是一个会牵扯所有子系统的连锁问题：

- attention score 的生成更贵
- KV cache 更大
- 训练阶段的长序列并行与重算策略更复杂
- 推理阶段的 prefix reuse、更长链 agent 工作流、共享前缀磁盘缓存都变得更敏感

因此，V4 的目标并不是“在 benchmark 上把 long-context 分再抬一点”，而是要让 **百万长度上下文成为稳定训练、稳定推理、稳定后训练、可日常部署** 的模型能力。

### 1.2 DeepSeek-V4 的四个设计目标

从论文结构倒推，DeepSeek-V4 追求的是四个同时成立的目标：

1. **长上下文效率**：1M token 必须在 FLOPs 和 KV cache 上大幅低于 V3.2
2. **表达能力不显著受损**：压缩和稀疏不能把模型变成只会做粗摘要的低精度系统
3. **训练稳定性**：引入复杂 routing / 压缩 / 多分支结构后，trillion-parameter MoE 训练仍需可控
4. **端到端可落地**：架构上的收益必须能被 kernel、并行、cache layout、rollout 服务真正兑现

这四点决定了 V4 的设计不会是“一个漂亮公式 + 一个简单 kernel”那么单薄。它必须同时回答：

- 记忆怎么表示？
- 相关信息怎么检索？
- 当前局部细节怎么保真？
- 深层信号怎么稳定传播？
- 这些东西如何在真实系统中被训练与部署？

## 2. 总体架构：把 Transformer 改造成分层压缩记忆系统 {#sec-2-overview}

### 2.1 从 Figure 2 读 V4 block

按论文 Figure 2，DeepSeek-V4 沿用了 Transformer + MoE + MTP 的基本大框架，但 block 内部的关键路径已经明显不同于传统 LLM：

- 输入 token 先过 Embedding
- 进入 block 后，先经过 **mHC 的 pre-block mixing**
- 然后进入 attention 子层：这里不是普通 MHA，而是 **CSA 或 HCA**
- attention 输出再经过 **mHC 的 post-block mixing**
- 接着进入下一次 **mHC 的 pre-block mixing**
- 再进入 FFN 子层：FFN 继续采用 **DeepSeekMoE**
- 最后再经历 **mHC 的 post-block mixing**
- 顶部仍保留 Prediction Head 与 MTP 模块

因此，相对 DeepSeek-V3/V3.2，V4 的核心改动集中在三件事：

- **残差路径变了**：普通 residual 改成 mHC
- **注意力路径变了**：普通或旧式 sparse attention 改成 CSA/HCA 混合体系
- **优化路径变了**：大部分矩阵参数改由 Muon 更新

### 2.2 一个更统一的抽象：三种状态同时存在

如果只从源码模块名去看，很容易把 V4 读成“很多独立技巧拼起来”。但从概念上，更好的读法是：**V4 的每一层都同时维护三类不同时间尺度的状态**。

1. **残差通道状态**：由 mHC 维护
   - 不再是一条单 residual stream
   - 而是多条 residual 通道的受约束混合系统
2. **压缩历史记忆**：由 CSA/HCA 维护
   - 不是逐 token KV
   - 而是压缩后的共享 KV memory blocks
3. **局部精细状态**：由 SWA 分支维护
   - 负责保留最近 token 的未压缩细节
   - 修补压缩路径在局部因果依赖上的精度损失

这三类状态共同定义了 V4 的“记忆访问语义”：

- 远距离历史：主要通过压缩后的 memory blocks 访问
- 真正重要的远程部分：通过 CSA 的 top-k 检索精确取回
- 最近局部上下文：通过 SWA 直接高精度访问
- 深层表征传播：通过 mHC 稳定控制

**下面这句是理解全文最重要的抓手之一**：DeepSeek-V4 不是把 attention 变 sparse，而是把“历史序列”改造成了“多时间尺度记忆层级”。

### 2.3 为什么 MoE 主体保留，而 attention 大改

V4 没有推翻 DeepSeekMoE，本身就说明论文作者对“容量从哪里来”与“瓶颈在哪里”有很清晰的判断：

- **容量与专家分工**：MoE 主干已经足够强，继续沿用 DeepSeekMoE 很自然
- **真正的长上下文瓶颈**：不是 FFN，而是 attention 的计算和 KV cache

因此，V4 选择“**保留 MoE 主干，重写记忆访问系统**”。这也是为什么你在做后续代码解析时，应该优先从 attention / KV cache / mHC 三块下手，而不是先去看 expert router。

完整模型规模与层级配置见[附录 B](#appendix-b-configs)。

## 3. mHC：把 residual 从单通道加法，改造成受约束的多通道混合 {#sec-3-mhc}

### 3.1 从普通 residual 到 Hyper-Connections

普通 residual 可以写成：

$$
x_{l+1} = x_l + F_l(x_l)
$$

它的隐含假设是：**一层只有一条 residual 通道**，并且旧状态与新计算结果之间的交互只有“直接相加”这一种方式。

Hyper-Connections (HC) 试图打破这个限制。它把 residual stream 扩展成 $n_{hc}$ 条并行通道：

$$
X_l = [x_{l,1}; \dots; x_{l,n_{hc}}]^T \in \mathbb{R}^{n_{hc} \times d}
$$

然后用三组映射控制：

- 怎样从这 $n_{hc}$ 条通道里混出本层真正要看的输入
- 怎样在通道之间传播旧 residual
- 怎样把当前 block 的新输出重新写回多条 residual 通道

公式写成：

$$
X_{l+1} = B_l X_l + C_l F_l(A_l X_l)
$$

其中：

- $A_l \in \mathbb{R}^{1 \times n_{hc}}$：输入混合矩阵
- $B_l \in \mathbb{R}^{n_{hc} \times n_{hc}}$：残差传播矩阵
- $C_l \in \mathbb{R}^{n_{hc} \times 1}$：输出写回矩阵

这时可以把 HC 理解成：**把 residual 本身也变成了一个可学习的状态空间系统**。

### 3.2 naive HC 的问题：表达能力增加了，但深层稳定性更差

HC 的优势是显然的：

- residual 不再只能“原样保留”
- 模型获得一条新的缩放轴：不改 hidden size，也能增加 residual dynamics 的表达能力

但论文指出，naive HC 在深层堆叠时经常会数值不稳定。直观地说，如果 $B_l$ 学成了扩张型线性变换，那么：

- 某些 residual 通道会被不断放大
- 多层连乘后，梯度和信号都会在特定方向上失控
- 动态生成的 $A_l, B_l, C_l$ 又会进一步放大这种不稳定

因此，V4 不是简单沿用 HC，而是引入 **mHC (Manifold-Constrained Hyper-Connections)**。

### 3.3 核心约束：把 residual 传播矩阵限制在 doubly stochastic manifold 上

mHC 的关键是约束残差传播矩阵 $B_l$：

$$
B_l \in \mathcal{M} = \{M \in \mathbb{R}^{n \times n} \mid M\mathbf{1}=\mathbf{1}, \mathbf{1}^TM=\mathbf{1}^T, M \ge 0\}
$$

也就是把 $B_l$ 投到 **doubly stochastic matrices** 的流形上。

这个约束的效果可以从三个层面理解：

- **非负性**：避免不同 residual 通道通过大幅正负抵消制造数值不稳定
- **行和列都归一**：每条通道既不会无界放大，也不会单方面吞噬总量
- **谱范数受控**：论文直接指出 $\|B_l\|_2 \le 1$，因此残差传播是 non-expansive 的

更重要的是，这个集合对乘法封闭。也就是说，如果很多层的 residual propagation 都在这个集合里，那么深层连乘后，系统仍然停留在“稳定的变换家族”中。

这正是 mHC 相对 naive HC 的核心差异：**不是放弃残差混合自由度，而是把这种自由度约束在一个深层可堆叠的稳定几何里。**

### 3.4 动态参数化：不是固定 mixing，而是输入相关的 mixing

mHC 仍然保留 HC 的一个关键优点：混合矩阵不是固定常数，而是随当前 residual state 动态生成。

首先把当前 residual state flatten 并 RMSNorm：

$$
\hat X_l = \mathrm{RMSNorm}(\mathrm{vec}(X_l)) \in \mathbb{R}^{1 \times n_{hc} d}
$$

然后生成未约束参数：

$$
\tilde A_l = \alpha_l^{pre}(\hat X_l W_l^{pre}) + S_l^{pre}
$$

$$
\tilde B_l = \alpha_l^{res}\, \mathrm{Mat}(\hat X_l W_l^{res}) + S_l^{res}
$$

$$
\tilde C_l = \alpha_l^{post}(\hat X_l W_l^{post})^T + S_l^{post}
$$

这里：

- $W_l^{pre}, W_l^{res}, W_l^{post}$ 生成动态项
- $S_l^{pre}, S_l^{res}, S_l^{post}$ 是静态偏置项
- $\alpha_l^{pre}, \alpha_l^{res}, \alpha_l^{post}$ 是小 gate

论文特别说明这些 $\alpha$ 都初始化为较小值。这意味着：

- 训练初期，系统更接近保守、近似静态的 residual mixing
- 随训练推进，才逐步释放输入依赖的动态混合能力

这是一种很典型的“先稳定，再逐步增加自由度”的设计。

### 3.5 如何把原始参数变成真正可用的 mHC 参数

生成完原始参数后，mHC 再施加约束：

$$
A_l = \sigma(\tilde A_l)
$$

$$
C_l = 2\sigma(\tilde C_l)
$$

对 $\tilde B_l$，先取指数得到正矩阵，再用 Sinkhorn-Knopp 迭代做行列归一化：

$$
M^{(t)} = T_r(T_c(M^{(t-1)}))
$$

最终：

$$
B_l = M^{(t_{max})}, \qquad t_{max}=20
$$

这条链条非常值得在代码里重点验证，因为它说明 mHC 不是一个简单 residual add，而是一个真正的“**动态生成 + 流形投影 + 多通道传播**”子系统。

### 3.6 mHC 在 V4 中到底扮演什么角色

从整体架构看，mHC 的职责不是增加长上下文本身的 memory 容量，而是：

- 为更复杂的 CSA/HCA 注意力提供更稳的深层信号通路
- 减轻超深 MoE + 压缩注意力组合时的优化不稳定性
- 提供一种比普通 residual 更强的层间状态调度机制

所以在 V4 里，mHC 更像是一个**稳定器与状态调度器**，而不是单独追求性能的 flashy 模块。

mHC 的实现与训练开销补充见[附录 C](#appendix-c-training)和[附录 D](#appendix-d-systems)。

## 4. 混合注意力：把长上下文改写为压缩记忆检索 {#sec-4-hybrid-attn}

### 4.1 统一视角：V4 不在原始 token 序列上做注意力，而在压缩记忆上做访问

标准自回归 attention 可以粗略理解为：

- 所有历史 token 都保留自己的 KV
- 每个 query 直接在全部历史 KV 上打分、归一化、聚合

V4 的想法是把“历史序列”替换成“压缩记忆”来访问：

1. **先压缩**：把很多 token 聚合成一个 compressed KV block
2. **再访问**：query 不再面对全部历史 token，而是面对压缩块
3. **再补局部**：最近局部 token 仍保留未压缩窗口，保证细节与因果

于是，V4 的 attention 可以抽象成三路记忆：

- **CSA**：在压缩块空间里做 top-k 检索，是细粒度 retrieval memory
- **HCA**：在极重压缩空间里 dense read，是全局 summary memory
- **SWA**：保留最近未压缩 token，是 local exact memory

这是全文最重要的统一模型：**DeepSeek-V4 的 attention = 压缩记忆检索 + 全局粗摘要 + 局部精细补偿。**

### 4.2 CSA：轻压缩 + 可学习检索

#### 4.2.1 第一步：双分支重叠压缩

设输入隐藏状态为 $H \in \mathbb{R}^{n \times d}$。CSA 先产生两组 KV 分支和两组压缩权重：

$$
C^a = H W_a^{KV}, \qquad C^b = H W_b^{KV}
$$

$$
Z^a = H W_a^Z, \qquad Z^b = H W_b^Z
$$

然后第 $i$ 个压缩块并不是只聚合当前区间 $[mi, m(i+1)-1]$，而是同时吸收：

- 当前块的 $a$ 分支
- 前一块的 $b$ 分支

权重由 learnable positional bias 修正后的 softmax 给出：

$$
[S^a_{mi:m(i+1)-1}; S^b_{m(i-1):mi-1}] = \mathrm{Softmax}_{row}([Z^a + B_a; Z^b + B_b])
$$

对应的 compressed KV 为：

$$
C_i^{Comp} = \sum_{j=mi}^{m(i+1)-1} S_j^a \odot C_j^a + \sum_{j=m(i-1)}^{mi-1} S_j^b \odot C_j^b
$$

这里的关键洞见不是“用了两套分支”本身，而是：**相邻压缩块之间通过重叠来源发生软耦合**。

如果只做普通不重叠 block pooling，那么每个压缩块只代表自己的 $m$ 个 token，块与块之间边界是硬切的。CSA 的双分支设计则让相邻块共享部分原始 token 的不同表示分支，从而弱化块边界断裂。

以 $m=4$ 为例：

- 第 $i-1$ 个块会读取区间 $[4(i-1), 4i-1]$ 的 $a$ 分支
- 第 $i$ 个块又会再读取同一区间的 $b$ 分支

因此，相邻压缩块不是彼此独立，而是带有跨块连续性。

#### 4.2.2 为什么论文说它“实际上压到 $1/m$”

第一次看公式时很容易误以为，每个压缩块用了 $2m$ 个位置，是不是总块数也要翻倍。不是。

正确理解是：

- 压缩块数量仍是 $n/m$
- 只是每个块的感受野不再是硬切开的 $m$ 个 token，而是一个带重叠竞争的 $2m$ 位置池

所以“长度压缩到 $1/m$”与“每个块有更柔和的边界”并不冲突。

#### 4.2.3 第二步：Lightning Indexer 在压缩块空间里做检索

得到 compressed KV 后，CSA 不直接在全部压缩块上做主 attention，而是先做 sparse selection。

先把 query token 的隐藏状态 $h_t$ 压到一个低秩 latent：

$$
c_t^Q = h_t W_D^Q
$$

再用同一个 latent 生成所有 indexer query heads：

$$
[q_{t,1}^I; \dots; q_{t,n_h^I}^I] = c_t^Q W_U^{IQ}
$$

同时，从 $h_t$ 生成每个 indexer head 的权重：

$$
[w_{t,1}^I; \dots; w_{t,n_h^I}^I] = h_t W^w
$$

于是，query token 与压缩块 $s$ 的索引分数为：

$$
I_{t,s} = \sum_{h=1}^{n_h^I} w_{t,h}^I \cdot \mathrm{ReLU}(q_{t,h}^I \cdot K_s^{IComp})
$$

这个分数结构很有意思：

- $q_{t,h}^I \cdot K_s^{IComp}$ 表示第 $h$ 个检索视角下的匹配度
- $\mathrm{ReLU}$ 抑制负相关匹配
- $w_{t,h}^I$ 让当前 token 自适应地决定“该更信哪类检索视角”

最后取 top-k：

$$
C_t^{SprsComp} = \{C_s^{Comp} \mid I_{t,s} \in \mathrm{Top\text{-}k}(I_{t,:})\}
$$

再结合因果条件 $s < \lfloor t/m \rfloor$，CSA 的语义就很清楚了：

- query 只能看已经闭合的历史压缩块
- 只把少量最相关的压缩块送入主 attention
- 因此真正昂贵的 attention 计算对象，从“全部历史 token”变成“少量相关压缩块”

#### 4.2.4 第三步：shared-KV MQA 真正读取内容

检索解决的是“去哪里找”，而不是“如何读出内容”。主 attention 仍要执行。

主 attention query 同样从 $c_t^Q$ 生成：

$$
[q_{t,1}; \dots; q_{t,n_h}] = c_t^Q W_U^Q
$$

随后对每个 query head 做：

$$
o_{t,i} = \mathrm{CoreAttn}(\text{query}=q_{t,i},\; \text{key}=C_t^{SprsComp},\; \text{value}=C_t^{SprsComp})
$$

这条式子里最重要的信息是：**同一压缩向量同时充当 key 和 value，且所有 query heads 共享同一套 compressed KV**。

因此，CSA 不是标准 MHA，而是一个更极端的 shared-KV MQA：

- query 是多头的
- KV 是压缩后的单共享记忆
- 读取时所有 query heads 共同访问同一 memory bank

这正是 V4 在 KV cache 上能大幅下降的根本原因之一。

### 4.3 HCA：极重压缩的全局摘要记忆

HCA 的角色与 CSA 明显不同。它不做 sparse selection，而是追求**极小的记忆长度**。

设输入仍为 $H \in \mathbb{R}^{n \times d}$，HCA 先生成：

$$
C = H W^{KV}, \qquad Z = H W^Z
$$

然后每 $m'$ 个 token 压成一个块：

$$
S_{m'i:m'(i+1)-1} = \mathrm{Softmax}_{row}(Z_{m'i:m'(i+1)-1}+B)
$$

$$
C_i^{Comp} = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j
$$

于是长度直接变成 $n/m'$。

对于 1M token：

- CSA 用 $m=4$，压成约 250K 个块
- HCA 用 $m'=128$，压成约 7813 个块

这时即使对 HCA 不做 sparse selection，直接 dense attention 成本也已经很低了。

因此 HCA 的职责不是“精准检索”，而是提供一个**极低成本、全局可见的粗摘要通路**。

### 4.4 为什么必须是 CSA 和 HCA 交错，而不是只保留其中一个

这正是 V4 的设计精髓：

- **只保留 HCA**：全局很便宜，但表达过粗，容易失去远距精细检索能力
- **只保留 CSA**：检索精度高，但压缩率没那么激进，整体成本仍然更高
- **两者交错**：
  - CSA 层负责“在长历史中精确找相关块”
  - HCA 层负责“始终保留一个全局粗摘要视角”

**下面这句是我对论文架构的核心概括**：

- CSA 是 retrieval memory
- HCA 是 summary memory
- SWA 是 exact local memory

三者交错后，模型在不同层上看到的是不同粒度的历史表征，这比单一路径稀疏化更像一个真正的 memory hierarchy。

### 4.5 输出路径为什么还要专门降本：grouped output projection

如果每个 query head 的维度固定为 $c=512$，那么拼接后的主 attention 输出维度会非常大：

- Flash：$512 \times 64 = 32768$
- Pro：$512 \times 128 = 65536$

若直接投回 hidden size：

- Flash：$32768 \rightarrow 4096$
- Pro：$65536 \rightarrow 7168$

这个投影本身就会很贵。于是 V4 采用 grouped output projection：

- 先把 $n_h$ 个 query heads 分成 $g$ 组
- 每组先投到中间维 $d_g$
- 再把所有组的结果拼起来，投回 hidden size

这在两个模型里都保持了一个非常规整的工程单元：

- 每组固定 8 个 heads
- 每组输入固定 4096 维
- 每组都先压到 1024 维

这说明 grouped projection 不只是数学上的 low-rank factorization，更是**为 kernel 友好性刻意固定出来的块状结构**。

### 4.6 另外三块关键补丁：RMSNorm、partial RoPE、SWA、attention sink

#### 4.6.1 Q / KV head 级 RMSNorm

在 core attention 前，CSA/HCA 都会对：

- 每个 query head
- 唯一那条 compressed KV head

额外做 RMSNorm。这一设计的作用非常直接：**防止超长上下文下 attention logits 爆炸**。它也是后文 “Muon 不需要 QK-Clip” 的一个重要前提。

#### 4.6.2 partial RoPE 只放最后 64 维

V4 并不是把完整 head 维都拿来做 RoPE，而是只对 query、compressed KV、以及 attention 输出的最后 64 维施加 RoPE。

更细一点说：

- query 与 compressed KV 先做常规 RoPE
- 由于 compressed KV 同时充当 key 和 value，输出 $o_{t,i}$ 会携带绝对位置信号
- 作者再对输出最后 64 维施加位置 $-i$ 的 RoPE

这一步的直观作用是：把 value 里残留的绝对相位“转回去”，让最终输出更像保留相对位置信息。

#### 4.6.3 SWA：修补压缩分支的局部精度缺口

CSA/HCA 都要求 query 只能访问**之前已经闭合的压缩块**。于是会出现一个局部问题：当前 token 所在块里那些更早 token，在 compressed branch 中反而不可见。

例如 $m=4$ 时，token 15 所在块为 $[12,13,14,15]$。按 $s < \lfloor t/m \rfloor$ 的规则，token 15 无法通过 compressed branch 直接看见 12/13/14。这会损坏语言建模中最重要的局部因果依赖。

因此，V4 在 CSA/HCA 之外再增加一个滑窗分支：

- 每个 query 额外直接看到最近 $n_{win}=128$ 个未压缩 token 的 KV
- 主 attention 读入的真实内容 = 选中的 compressed KV + 最近滑窗 KV

所以 SWA 的角色非常明确：**不是提供全局视野，而是修补压缩路径丢掉的局部精度。**

#### 4.6.4 attention sink：允许 head 暂时“不看”真实记忆

V4 在 softmax 分母里加入 learnable sink logit：

$$
s_{h,i,j} = \frac{\exp(z_{h,i,j})}{\sum_k \exp(z_{h,i,k}) + \exp(z_h')}
$$

这意味着一个 head 的注意力质量不必全部分配给真实 token / 压缩块，也可以有一部分“流进 sink”。

对于压缩注意力，这很合理，因为：

- 并不是每个 query 都一定能在当前可见的压缩块里找到高质量信息
- 如果强迫总质量一定归一到 1，就可能把噪声压缩块的权重抬高

### 4.7 用 1M context 超参直观看 V4 在“读什么”

把配置代入 1M context：

- CSA 压缩率 $m=4$：压缩后约 **250,000** 个块
- Flash 的 top-k=512：每个 query 只读约 **0.20%** 的压缩块
- Pro 的 top-k=1024：每个 query 只读约 **0.41%** 的压缩块
- HCA 压缩率 $m'=128$：压缩后约 **7813** 个块
- SWA：额外保留最近 **128** 个未压缩 token

所以从访问模式看，V4 对一个 query 的真实读取近似是：

- 一小撮高相关远程压缩块（CSA）
- 一条便宜但全局可见的粗摘要通路（HCA）
- 一段最近的高精度局部上下文（SWA）

这三路叠加起来，才是 V4 真正意义上的 attention。

完整系统实现细节（KV cache 组织、contextual parallelism、磁盘缓存）见[附录 D](#appendix-d-systems)。

## 5. Muon 与训练稳定性：让复杂架构真正可训练 {#sec-5-muon}

### 5.1 为什么 V4 选择 Muon

V4 的主体参数里有大量“大矩阵”：

- attention 的各种投影矩阵
- MoE expert 的 up / down / gate projection
- mHC 的动态参数生成矩阵

对这类参数，逐元素自适应优化器未必是最合适的建模方式。论文选用 Muon，是因为它把更新方向当作**矩阵几何对象**来处理，而不是只做 element-wise scaling。

### 5.2 哪些参数仍然保留 AdamW

V4 并不是把所有参数都交给 Muon。保留 AdamW 的模块包括：

- embedding
- prediction head
- mHC 的 static biases 与 gating factors
- 所有 RMSNorm 权重

这说明论文作者的判断很明确：**结构化大矩阵适合 Muon，小尺度标量/向量参数仍更适合 AdamW。**

### 5.3 Muon 的更新公式怎么理解

对每个逻辑独立矩阵 $W \in \mathbb{R}^{n \times m}$：

先计算梯度：

$$
G_t = \nabla_W \mathcal{L}_t(W_{t-1})
$$

再做动量累计：

$$
M_t = \mu M_{t-1} + G_t
$$

然后对 $\mu M_t + G_t$ 做 Hybrid Newton-Schulz 正交化，得到近似正交的更新方向；再按 RMS 重新缩放，最后做 weight decay 与学习率更新。

从优化视角看，Muon 最本质的区别不在“有动量”“有 weight decay”，而在于：**更新方向先被投成一个近似正交矩阵。**

### 5.4 Hybrid Newton-Schulz：分两阶段把奇异值推向 1

论文给出的迭代形式为：

$$
M_k = aM_{k-1} + b(M_{k-1}M_{k-1}^T)M_{k-1} + c(M_{k-1}M_{k-1}^T)^2M_{k-1}
$$

V4 使用 10 步两阶段版本：

- 前 8 步：$(a,b,c)=(3.4445,-4.7750,2.0315)$，目的是快速把奇异值推近 1
- 后 2 步：$(a,b,c)=(2,-1.5,0.5)$，目的是让奇异值更稳定地收敛到 1 附近

这可以理解成一种“**先快速靠近，再稳定钉住**”的策略。

### 5.5 为什么论文专门强调“不需要 QK-Clip”

V4 指出，由于 CSA/HCA 在 core attention 前已经对 query 与 compressed KV 做了 RMSNorm，因此 attention logits 本身不容易爆炸，所以 Muon 不再需要额外的 QK-Clip。

这说明 V4 的稳定性并不是靠一个单点 trick，而是多个环节形成闭环：

- mHC 控制 residual propagation
- Q/KV RMSNorm 控制 attention logits
- Muon 提供更稳的矩阵更新
- 训练层面再用 Anticipatory Routing 与 SwiGLU Clamping 兜底

这四者是配套系统，不应拆开孤立理解。

完整训练细节见[附录 C](#appendix-c-training)。

## 6. 把各模块拼起来看：DeepSeek-V4 到底是怎样工作的 {#sec-6-synthesis}

如果把上面几部分组合起来，V4 的一层可以这样理解：

1. **mHC 先决定当前层怎样从多通道 residual 中读状态**
2. **attention 子层从三类记忆读取信息**：
   - CSA：从长历史中检索少量高相关压缩块
   - HCA：读取一个极便宜的全局粗摘要
   - SWA：补上最近局部未压缩细节
3. **grouped output projection 把高维多头输出压回主干维度**
4. **mHC 再把当前层新产生的信息稳定写回 residual state**
5. **MoE 继续负责容量与专家分工**
6. **Muon 让这些大矩阵子系统的联合训练保持稳定**

从这个角度看，V4 真正的创新不是“某一个公式”，而是把下面这些问题同时解决了：

- 如何让远程历史不必逐 token 存着
- 如何在压缩后仍保留远距精细检索能力
- 如何不牺牲局部精度
- 如何不让复杂的 residual / routing / compression 结构破坏训练稳定性
- 如何让架构收益最终兑现到真实系统效率

因此，V4 是一篇非常典型的 **模型-系统协同设计论文**：数学建模决定记忆访问形态，而系统设计保证这种形态能真正落地。

## 7. 核心结果：V4 证明这套记忆系统确实值回票价 {#sec-7-results}

### 7.1 最关键的效率结论

论文最值得记住的效率结果只有一组：在 **1M token context** 下，相对 DeepSeek-V3.2：

- **DeepSeek-V4-Pro**：单 token inference FLOPs = **27%**，KV cache = **10%**
- **DeepSeek-V4-Flash**：单 token inference FLOPs = **10%**，KV cache = **7%**

这说明 V4 的收益不是“某个 benchmark 上快一点”，而是把长上下文系统的主成本项都压下来了。

### 7.2 base model 结果说明：结构优化真的换来了能力

V4-Flash-Base 虽然 activated params 只有 13B，但已经在大量基准上超过 37B activated 的 V3.2-Base；V4-Pro-Base 则进一步在知识、推理、代码、长上下文上全方位拉开差距。

这说明两件事：

- V4 的注意力与训练体系不是单纯“省算力”，而是保持甚至提升了基础能力
- 结构优化和高质量长文档训练数据，足以显著提高参数效率

更完整对比见[附录 F](#appendix-f-eval)。

### 7.3 post-trained model 结果说明：open model 的新上限

DeepSeek-V4-Pro-Max 的定位很清楚：

- **知识类**：显著强于现有 open models，但仍落后最强闭源模型
- **代码与数学推理**：已经非常接近甚至局部超过闭源前沿
- **1M context**：在 MRCR / CorpusQA 上强于 Gemini-3.1-Pro，但仍低于 Claude Opus 4.6
- **agent**：open model 里很强，但离最强闭源 coding / tool agent 还有一小段距离

因此，V4 不是“全面碾压所有闭源模型”，但它非常清楚地证明了：**通过压缩记忆架构 + 系统级基础设施，open long-context reasoning model 可以被推进到一个新的效率-能力边界。**

## 8. 关键启示 {#sec-8-takeaways}

- **压缩优先于稀疏**：V4 不是直接在原始 token 序列上做 sparse attention，而是先把历史改写成压缩记忆，再在压缩空间中做检索与聚合。
- **长上下文的核心不是“看得更远”，而是“如何表示远方”**：CSA/HCA 本质上是在重写 long-range memory representation。
- **局部精度必须单独保底**：SWA 的存在说明压缩路径永远无法完全代替最近局部未压缩信息。
- **稳定性必须写进架构本身**：mHC、Q/KV RMSNorm、Muon、Anticipatory Routing、SwiGLU Clamping 共同构成稳定闭环。
- **真正的百万上下文能力一定是模型-系统协同结果**：如果没有 KV cache layout、contextual parallelism、FP4、磁盘前缀复用、rollout/sandbox 基础设施，V4 的架构收益不会完全兑现。
- **后续做代码解析时，最重要的是沿“数学对象 → 内存布局 → kernel 实现”的主线去看**，而不是把每个模块孤立当成一个 op。

---

## 附录 A：主要符号表 {#appendix-a-symbols}

下面只列正文中最常用的核心符号；与系统实现强相关的变量放在相应附录节里解释。

- $n$：序列长度
- $d$：hidden size
- $c$：compressed KV entry 的维度
- $d_c$：query latent 的压缩维度
- $n_h$：主 attention 的 query head 数
- $n_h^I$：indexer query head 数
- $c^I$：indexer head 维度
- $m$：CSA 的压缩率
- $m'$：HCA 的压缩率，且 $m' \gg m$
- $k$：CSA 在压缩块空间的 top-k
- $n_{win}$：SWA 保留的未压缩局部窗口长度
- $g$：grouped output projection 的分组数
- $d_g$：每组的中间投影维度
- $n_{hc}$：mHC 的 residual 通道扩展倍数
- $A_l, B_l, C_l$：mHC 的输入混合、残差传播、输出写回矩阵
- $C^{Comp}$：压缩后的 KV memory blocks
- $K^{IComp}$：压缩后的 indexer keys
- $c_t^Q$：query 的低秩 latent，供 indexer 与主 attention 共用

正文中涉及这些符号的核心讨论分别见[第 3 节](#sec-3-mhc)与[第 4 节](#sec-4-hybrid-attn)。

## 附录 B：模型配置与参数规模 {#appendix-b-configs}

### B.1 DeepSeek-V4-Flash

- 层数：43
- hidden dim：4096
- 总参数：284B
- activated params：13B
- 前两层注意力：纯 SWA
- 后续层：CSA / HCA 交错
- CSA：
  - 压缩率 $m=4$
  - top-k = 512
  - indexer query heads = 64
  - indexer head dim = 128
- HCA：
  - 压缩率 $m'=128$
- 主 attention：
  - query heads = 64
  - head dim = 512
  - query compression dim = 1024
- 输出投影：
  - groups = 8
  - 每组中间维 $d_g=1024$
- SWA window：128
- MoE：
  - 1 shared expert + 256 routed experts
  - 每个 token 激活 6 个 routed experts
  - 每个 expert 中间维 2048
- 前 3 个 MoE 层采用 Hash routing
- MTP depth = 1
- mHC：
  - $n_{hc}=4$
  - Sinkhorn 迭代 20 次

### B.2 DeepSeek-V4-Pro

- 层数：61
- hidden dim：7168
- 总参数：1.6T
- activated params：49B
- 前两层注意力：纯 HCA
- 后续层：CSA / HCA 交错
- CSA：
  - 压缩率 $m=4$
  - top-k = 1024
  - indexer query heads = 64
  - indexer head dim = 128
- HCA：
  - 压缩率 $m'=128$
- 主 attention：
  - query heads = 128
  - head dim = 512
  - query compression dim = 1536
- 输出投影：
  - groups = 16
  - 每组中间维 $d_g=1024$
- SWA window：128
- MoE：
  - 1 shared expert + 384 routed experts
  - 每个 token 激活 6 个 routed experts
  - 每个 expert 中间维 3072
- 前 3 个 MoE 层采用 Hash routing
- MTP depth = 1
- mHC：
  - $n_{hc}=4$
  - Sinkhorn 迭代 20 次

### B.3 从这些超参能读出的设计信号

- Flash 与 Pro 共享同一套长上下文记忆机制：$m=4$、$m'=128$、$n_{win}=128$ 完全一致
- Pro 相比 Flash 主要增强的是：
  - 检索带宽（top-k 512 → 1024）
  - head 数（64 → 128）
  - 模型容量（更深、更宽、专家更多）
- grouped output projection 的工程单元基本固定：每组恒为 8 个 heads、每组输入 4096 维、每组输出 1024 维
- attention 容量主要通过增加 head 数和层数来扩张，而不是继续抬高单 head 宽度
- 激活专家数始终固定为 6，说明扩容主要靠专家池变大，而不是让每个 token 同时访问更多 experts

## 附录 C：预训练与稳定性细节 {#appendix-c-training}

### C.1 数据构建

V4 在 V3 预训练数据之上继续扩展和清洗，重点包括：

- web 数据里更强地过滤批量自动生成与模板化内容，降低 model collapse 风险
- 数学和代码仍是核心语料
- 中期训练引入 agentic data，强化 coding 与 agent 场景
- 扩大 multilingual 数据，补全长尾文化知识
- 重点增加长文档数据，如论文、技术报告等

总语料规模超过 **32T tokens**。此外：

- tokenizer 在 V3 基础上只增添少量 special tokens，词表仍为 128K
- 保留 token-splitting 与 FIM
- 采用 sample-level attention masking

### C.2 训练配置：Flash

- 训练 token：32T
- 最大 batch：75.5M tokens
- AdamW：
  - $\beta_1=0.9$
  - $\beta_2=0.95$
  - $\epsilon=10^{-20}$
  - weight decay = 0.1
- Muon：
  - momentum = 0.95
  - weight decay = 0.1
  - update RMS 重标定到 0.18
- 学习率：
  - 前 2000 step warmup
  - 峰值 $2.7\times10^{-4}$
  - 末尾 cosine decay 到 $2.7\times10^{-5}$
- 序列长度：4K → 16K → 64K → 1M
- 前 1T tokens 先用 dense attention
- 从 64K 开始引入 sparse attention
- 先短暂 warmup lightning indexer，再长期 sparse training

### C.3 训练配置：Pro

- 训练 token：33T
- 最大 batch：94.4M tokens
- AdamW 与 Muon 超参同 Flash
- 学习率：
  - 峰值 $2.0\times10^{-4}$
  - 末尾 $2.0\times10^{-5}$
- 序列长度：4K → 16K → 64K → 1M
- dense attention warmup 阶段比 Flash 更长

### C.4 共同训练细节

- auxiliary-loss-free load balancing 的 bias update speed = 0.001
- sequence-wise balance loss 权重 = 0.0001
- MTP loss 权重：
  - 大部分训练阶段为 0.3
  - 学习率衰减开始后降到 0.1

### C.5 训练稳定性：为什么还需要额外技巧

论文明确说，trillion-parameter MoE 的训练里遇到过显著 loss spike，且异常与 MoE outlier 高度相关，routing 机制会放大这种问题。

因此在 mHC + RMSNorm + Muon 之外，又引入两条实用稳定化策略。

### C.6 Anticipatory Routing

核心思想：**在 step $t$，feature 用当前参数 $\theta_t$ 计算，但 routing index 用历史参数 $\theta_{t-\Delta t}$ 计算。**

实现上，为了避免双次加载参数，把 step $t$ 用到的 routing index 提前在 step $t-\Delta t$ 预计算并缓存。

它的作用可以理解为：

- 打断 backbone 更新与 routing 更新的同步强耦合
- 减轻 routing 对异常激活的即时正反馈放大

工程代价：

- 额外墙钟开销约 20%
- 但系统只在检测到 loss spike 时短暂启用，之后再恢复普通训练，因此总体额外成本较小

### C.7 SwiGLU Clamping

- linear 分支 clamp 到 [-10, 10]
- gate 分支只截上界 10

这是一个看似简单但很有力的 outlier 抑制手段。论文强调，它能显著提升稳定性而不明显伤性能。

## 附录 D：系统与基础设施细节 {#appendix-d-systems}

这一附录服务于正文[第 4 节](#sec-4-hybrid-attn)与[第 6 节](#sec-6-synthesis)：V4 的模型收益只有在系统层被兑现，才真正成立。

### D.1 Expert Parallel：细粒度通信-计算重叠

V4 把一个 MoE 层拆成四段：

- Dispatch（通信）
- Linear-1（计算）
- Combine（通信）
- Linear-2（计算）

profiling 发现，单层内部总通信时间小于总计算时间。因此论文把 experts 再切成多个 waves：

- 每个 wave 只包含一小部分 experts
- 一个 wave 的通信一完成就立刻启动该 wave 的计算
- 当前 wave 计算、下一 wave token transfer、上一 wave 结果发送同时并行

效果：

- 通用 inference workload 提速约 1.50～1.73×
- RL rollout / latency-sensitive agent serving 场景最高 1.96×

该 mega-kernel 以 `MegaMoE` 名义开源进 DeepGEMM。

### D.2 TileLang：让复杂 kernel 变得可快速迭代

论文强调 TileLang 的三类价值：

- **Host Codegen**：把原本 Python 侧的 contract check、shape/stride 验证与参数打包下沉到生成的 host launcher，中断 CPU 侧固定开销
- **Z3 驱动的整数分析**：支持复杂 tensor index 的 layout inference、hazard detection、bound analysis 与 vectorization
- **数值精度与 bitwise reproducibility**：默认不走激进 fast-math，并尽量与 CUDA 工具链对齐

对 V4 来说，TileLang 的意义不是“写 kernel 更方便”，而是 **CSA/HCA/mHC/FP4/可复现实验** 这些复杂算子都能被快速迭代。

### D.3 Batch-Invariant 与 Deterministic Kernel

V4 非常强调可复现性。

**Batch invariance**：同一 token 的输出不应因它在 batch 中的位置不同而发生 bitwise 差异。

- attention 侧不能直接依赖 split-KV；V4 为 decoding 设计 dual-kernel，以同时兼顾一致性与尾 wave 延迟
- GEMM 侧不能只依赖传统 cuBLAS，因此端到端替换为 DeepGEMM

**Determinism**：重点解决 backward 中的非确定性累加。

- sparse attention backward：为每个 SM 分配单独 buffer，再做全局确定性求和
- MoE backward：用 token order 预处理与 buffer 隔离固定写入顺序
- mHC 的小维 GEMM：split-k 结果先分开，再单独确定性 reduce

### D.4 FP4 QAT

V4 在后训练阶段引入 QAT，并对两类对象做 FP4：

- MoE expert weights
- CSA indexer 的 QK path

核心过程：

- optimizer 保留 FP32 master weights
- forward 前先量化到 FP4，再无损反量化回 FP8 参与训练
- backward 直接对 forward 用过的 FP8 权重求梯度，相当于在量化上使用 STE

额外优化：

- index score 从 FP32 压到 BF16
- top-k selector 获得 2× 加速
- KV entry recall 仍有 99.7%

在 rollout / inference 时则直接使用“真实 FP4 权重”，从而保证采样行为与部署一致。

### D.5 训练框架：Muon 的 ZeRO 兼容、mHC 重算、contextual parallelism

**Muon + ZeRO**

- dense 参数：限制 ZeRO 并行度，用 knapsack 负载均衡
- MoE 参数：按大量专家矩阵 flatten 后分发
- 同形状矩阵可自动 merge，便于 batched Newton-Schulz
- MoE 梯度同步时随机舍入到 BF16，通信量减半

**mHC 低成本实现**

- 训练/推理都写 fused kernel
- selective recomputation 只重算相对便宜的中间量
- 调整 DualPipe 1F1B overlap
- 最终把 mHC 墙钟额外开销压到 overlapped 1F1B pipeline stage 的 6.7%

**Contextual Parallelism for compressed attention**

由于 CSA/HCA 的压缩长度跨 rank 不再天然对齐，V4 采用两级通信：

1. rank $i$ 把最后 $m$ 个未压缩 KV 发给 rank $i+1$
2. 相邻 rank 用“收到的尾巴 + 本地 token”一起压缩
3. 再 all-gather 压缩结果
4. 用 fused select-and-pad 重组全局 compressed KV

这套设计本质上是让“跨 rank 的压缩边界”也被系统层显式建模，而不是隐式交给通用 context parallel。

### D.6 推理框架：异构 KV cache 结构

V4 的 KV cache 管理非常关键，因为存在多类异构状态：

- CSA/HCA 的 compressed KV
- CSA indexer 的额外 KV
- SWA 的未压缩局部 KV
- 尚未凑够压缩块的 tail tokens 状态

论文把它们拆成两大类：

- **classical KV cache**：存 CSA/HCA 的压缩后 KV
- **state cache**：存 SWA 与未完成压缩的尾状态

这样做的原因是：PagedAttention 之类的统一假设不再适用，V4 的不同层具有不同 cache policy 与对齐需求。

### D.7 磁盘 KV cache 与共享前缀复用

对于共享前缀请求：

- CSA/HCA 的 compressed KV 可以直接落盘，并在命中前缀时读取到最后一个完整压缩块
- 尾部未满块 token 需要重算，因为未压缩 KV 不落盘

SWA 更麻烦，因为它是未压缩、且每层都存在，容量约是 compressed KV 的 8 倍。论文给出三种策略：

- **Full SWA Caching**：全存，零重算，但 SSD 访问模式差
- **Periodic Checkpointing**：每隔 $p$ 个 token 存最近 $n_{win}$ token 的 SWA 状态
- **Zero SWA Caching**：完全不存，仅靠 CSA/HCA cache 重算最后 $n_{win}\cdot L$ 个 token 恢复

这三者本质上是在存储空间、I/O 压力与计算冗余之间做折中。

## 附录 E：后训练与 agent 化能力 {#appendix-e-posttraining}

### E.1 两阶段后训练范式

V4 的 post-training 主线是：

1. **specialist training**：按数学、代码、agent、instruction following 等领域分别做 SFT + RL
2. **multi-teacher OPD merge**：把多个 specialist 的能力蒸馏进统一 student model

相对 V3.2，一个关键变化是：**mixed RL 阶段被完全替换成 On-Policy Distillation (OPD)**。

### E.2 三种 reasoning effort mode

V4-Pro 与 V4-Flash 都支持三档推理强度：

- **Non-think**：快速、直觉式、低成本推理
- **Think High**：显式推理、更慢但更稳
- **Think Max**：拉满 reasoning budget，并注入更强 system prompt，鼓励极端彻底的思考

论文强调，Think Max 不只是“给更多 token”，而是把：

- 更长的 context window
- 更低的 length penalty
- 更强的 reasoning instruction

一起作为一个 mode 来训练和评测。

### E.3 Generative Reward Model (GRM)

对 hard-to-verify tasks，V4 不再依赖传统 scalar reward model，而是让模型本身承担评价者角色：

- 基于 rubric-guided RL data 学习轨迹评价
- actor network 原生兼作 GRM
- 把“会做题”和“会判题”统一进同一模型

这样做的好处是：模型的内在推理能力直接参与评分过程，评价更稳健，且对大规模人标的依赖更低。

### E.4 Tool schema、interleaved thinking 与 Quick Instruction

**Tool schema**

- 使用 `|DSML|` special token
- 用 XML 格式表示 tool call
- 论文认为这样更稳健，escaping 错误更少

**Interleaved thinking**

- tool-calling 场景：完整保留 reasoning trace，哪怕跨 user turn 也保留
- 普通对话场景：新 user message 到来时丢弃旧 reasoning，保持上下文紧凑

**Quick Instruction**

为了避免额外调用一个小模型去做 search trigger / title / authority / domain 等辅助任务，V4 直接往输入后面附 special tokens，例如：

- `<|action|>`：是否要 web search
- `<|title|>`：生成会话标题
- `<|query|>`：生成搜索 query
- `<|authority|>`：判断权威性需求
- `<|domain|>`：识别问题领域

这些任务可直接复用已有 KV cache，避免额外 prefill。

### E.5 OPD 的目标函数

设 teacher 集合为 $\{\pi_{E_1}, \dots, \pi_{E_N}\}$，V4 的 OPD 目标为：

$$
L_{OPD}(\theta) = \sum_{i=1}^{N} w_i \cdot D_{KL}(\pi_\theta \parallel \pi_{E_i})
$$

其关键特征：

- student 在**自己生成的轨迹上**学习，因此是 on-policy
- 不采用 token-level KL 近似，而采用 **full-vocabulary logit distillation**

论文强调，full-vocabulary OPD 的梯度更稳、知识保留更完整。

### E.6 RL / OPD 基础设施

为了让 million-token RL 与 full-vocabulary OPD 真能跑起来，论文配了一整套基础设施：

- rollout / teacher forward 全面支持 FP4
- teacher 权重放在中心化分布式存储中按需加载
- 只缓存 teacher 最后一层 hidden states，再在线恢复 logits
- 为每个生成请求维护 token 级 Write-Ahead Log (WAL)
- 通过共享内存 loader 减轻 million-token rollout data 的 CPU/GPU 压力
- 构建 DSec sandbox 平台，统一支持 Function Call / Container / microVM / fullVM 四种 substrate

这说明在 V4 里，agent 能力不是 post-hoc 外接出来的，而是从后训练基础设施层面被当成一等公民来设计。

## 附录 F：评测补充 {#appendix-f-eval}

### F.1 Base model 的代表性结果

下面只保留最能说明 V4 架构价值的几项 base 结果：

| Benchmark | DeepSeek-V3.2-Base | DeepSeek-V4-Flash-Base | DeepSeek-V4-Pro-Base |
| --- | ---: | ---: | ---: |
| MMLU-Pro | 65.5 | 68.3 | 73.5 |
| Simple-QA verified | 28.3 | 30.1 | 55.2 |
| FACTS Parametric | 27.1 | 33.9 | 62.6 |
| HumanEval | 62.8 | 69.5 | 76.8 |
| LongBench-V2 | 40.2 | 44.7 | 51.5 |

解读：

- Flash 在更低 activated params 下已超过 V3.2-Base 的多数指标
- Pro 则进一步在知识、代码、长上下文上明显拉开差距
- 这说明“压缩记忆 + 更稳训练”并不只是节省成本，而是换来了更强 base capability

### F.2 Post-trained model 的关键对比

保留最有代表性的几项结果：

| Benchmark | Opus-4.6-Max | GPT-5.4-xHigh | Gemini-3.1-Pro-High | DeepSeek-V4-Pro-Max |
| --- | ---: | ---: | ---: | ---: |
| SimpleQA Verified | 46.2 | 45.3 | 75.6 | 57.9 |
| LiveCodeBench | 88.8 | - | 91.7 | 93.5 |
| Codeforces Rating | - | 3168 | 3052 | 3206 |
| MRCR 1M | 92.9 | - | 76.3 | 83.5 |
| CorpusQA 1M | 71.7 | - | 53.8 | 62.0 |
| Terminal Bench 2.0 | 65.4 | 75.1 | 68.5 | 67.9 |
| Toolathlon | 47.2 | 54.6 | 48.8 | 51.8 |

解读：

- 知识问答仍落后 Gemini-3.1-Pro，但已显著超过当前 open baselines
- 代码与数学推理已经非常接近甚至局部超过闭源模型
- 1M context 能力强于 Gemini-3.1-Pro，但仍弱于 Opus-4.6
- agent 能力很强，但离最强闭源 coding/tool agent 还有一小段距离

### F.3 系列内部不同 reasoning mode 的趋势

V4-Flash / V4-Pro 的 Non-think、High、Max 三档结果显示：

- reasoning budget 增加后，难题性能显著提升
- Flash 在知识量上弱于 Pro，但在高推理预算下，很多 reasoning 任务能接近 Pro
- Max mode 在 hardest tasks 上普遍优于 High mode，说明更长 context + 更低 length penalty 的 RL 是有效的

### F.4 真实任务

论文还专门给了内部真实任务评测：

- 中文写作：V4-Pro 对 Gemini-3.1-Pro 具明显优势
- Search：agentic search 明显优于普通 RAG
- White-collar tasks：对 Opus-4.6-Max 有较高 non-loss rate
- 内部 R&D coding benchmark：V4-Pro-Max 已非常接近强闭源 coding agent

这进一步证明 V4 的价值不只是 benchmark 提分，而是长链工作流中的综合表现提升。

## 附录 G：后续代码解析路线图 {#appendix-g-code-roadmap}

既然后面还会继续展开代码解析，我建议直接按“数学对象 → 内存布局 → kernel 实现”的路径去拆。

### G.1 最优先的 5 个实现热点

1. **mHC 的 pre/post mixing kernel**
   - 动态参数 $A/B/C$ 在哪里生成
   - Sinkhorn 投影是单独 kernel 还是融合到前后处理
   - residual state 是否真实以 $n_{hc} \times d$ 形式组织
2. **CSA 的双分支重叠压缩**
   - $a/b$ 双分支怎样缓存
   - block 边界怎样保留“前一块”的 token/state
   - `Softmaxrow` 是逐通道做还是融合实现
3. **Lightning Indexer 的低秩 query 复用**
   - $c_t^Q$ 是否真正只算一次，同时服务 indexer 与主 attention
   - indexer key 的缓存与量化路径如何落地
   - FP4 QK path 的量化/反量化发生在哪一层
4. **compressed KV 与 SWA KV 的拼接**
   - causal / visibility mask 如何同时作用在两类 memory 上
   - partial RoPE 与输出位置 $-i$ 的逆向 RoPE 放在哪个阶段
   - attention sink 如何并入 softmax 分母
5. **grouped output projection 的固定 8-head 分组**
   - 这是否直接决定了 kernel tile 形状
   - Flash/Pro 只是组数不同，单组实现是否完全复用

### G.2 推荐代码解析顺序

- 第一步：mHC
- 第二步：CSA compressor
- 第三步：Lightning Indexer
- 第四步：CSA/HCA core attention + SWA branch
- 第五步：grouped output projection
- 第六步：KV cache layout / inference path
- 第七步：FP4、contextual parallelism 与 rollout/inference 基础设施

如果这条主线打通，那么 DeepSeek-V4 的“模型设计 → 数学对象 → 内存结构 → 真实系统实现”基本就会非常清楚。
