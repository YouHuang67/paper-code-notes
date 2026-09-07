---
tags:
  - Diffusion Model
  - Video Generation
  - CUDA
---
# DiT 推理优化：Parallelism

返回：[专题总览](overview.md)

## 抓住重点

设 CFG、张量并行和序列并行度分别为 \(g,p,s\)，则参与一次 DiT step 的设备数为 \(N=gps\)。并行化只在通信代价低于被分摊的计算时加速；它保持去噪数学不变。序列并行内部有两种互斥的数据交换组织：重排序列与 head 的 Ulysses，或保留本地 query、交换 K/V 的 Ring/KV-Gather。

## 1. 三种切分的数学对象

设输入 \(X\in\mathbb R^{S\times D}\)，头数为 \(H\)，每头维度为 \(d=D/H\)。CFG 将条件/无条件两次函数评价分配给不同 rank，最后计算 \(y=y_u+w(y_c-y_u)\)。TP 将权重行列切分并在 GEMM 后 collective，约束是 \(H\bmod p=0\)。SP 将 token 轴切成 \(s\) 份，使每个 rank 只持有 \(S/s\) 行。

Ulysses 通过 all-to-all 把 token 份重排为完整的本地 heads；Ring 保持本地 heads，轮换 K/V 并在线合并 softmax。KV-Gather 直接执行
\[
K=\operatorname{AllGather}(K_r),\quad V=\operatorname{AllGather}(V_r),\quad O_r=\operatorname{Attn}(Q_r,K,V),
\]
与 Ring/Ulysses 争用同一 SP 维度，当前配置禁止叠加。

## 2. 形状与拓扑约束

必须满足 \(H/p\in\mathbb N\)，并使 packed token 数能被 \(s\) 整除；Ulysses 还要求本地 head 数能被其 head 重排度整除。Ring 需要 attention backend 支持邻居 K/V 旋转。Ulysses 的 all-to-all 通常应映射到连续 NVLink rank，Ring 的邻居序列应贴合高速链路；错误映射保持正确性但增加通信时间。

## 3. 成本模型与选择

令单卡计算时间随切分为 \(C(N)\)，通信时间为 \(A(N)\)，则
\[
\tau(N)=C(N)+A(N),\qquad \mathrm{speedup}=\tau(1)/\tau(N).
\]
当 \(A(N)\) 接近 \(C(1)-C(N)\) 时继续扩卡没有收益。长序列优先测 Ulysses；跨节点且 all-to-all 昂贵时测 Ring 或 KV-Gather；TP 需确认每层 GEMM 足以覆盖 collective。Encoder 并行是独立轴，见 [Encoder & VAE](encoder_vae.md)。

## 4. 组合约束

Progressive Resolution 改变每步 token 数，当前与 Ulysses/Ring SP 互斥；Cache-DiT 在 SP/TP 下必须对相似度判据做组内归约，否则不同 rank 会进入不同分支。详见 [Feature Cache](feature_cache.md) 与 [Progressive](progressive_resolution.md)。

## 附录：实现证据

本页依据 SGLang `parallelism.mdx`、`ring_sp_performance.mdx`、`encoder_parallel.mdx` 及 `server_args` 的并行校验；H3 的 packed QKV 与 collective 次序见 [DiT Runtime](../minimax_h3/07_dit_runtime_and_collectives.md)。

## 相关阅读

- [专题总览](overview.md) · [Feature Cache](feature_cache.md) · [Graph Runtime](graph_runtime.md) · [Correctness](correctness.md)
