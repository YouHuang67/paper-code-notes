---
tags:
  - Diffusion Model
  - Video Generation
  - CUDA
---
# DiT 推理优化：Memory Offload

返回：[专题总览](overview.md)

## 抓住重点

- Offload 保持去噪函数 \(f_\theta\) 不变，改变参数在 CPU/GPU 间的驻留函数 \(R(t)\)，以时间换取显存。
- 组件级调度处理 Encoder、DiT、VAE 的使用区间；层间级调度将 DiT 的 \(L\) 个 block 变成带前瞻的传输流水线。
- 可行性同时取决于峰值显存和传输能否被计算隐藏。模型“能跑”并不等于速度可接受。
- 当前 SGLang 集成禁止 Cache-DiT 与 DiT layerwise 同开；Cache-DiT 自己的 bucket offload 是另一套语义。

## 1. 优化对象：驻留约束与时间代价

设组件 \(c\) 的参数大小为 \(w_c\)，瞬时激活与工作区为 \(a(t)\)、\(b(t)\)，\(R_c(t)\in\{0,1\}\) 表示其是否驻留 GPU。任何合法调度都必须满足

\[
\sum_c R_c(t)w_c+a(t)+b(t)\le M_{\rm GPU}.
\]

转移 \(R_c:0\to1\) 产生 H2D 代价 \(h_c\)，反向转移产生 D2H 代价。Offload 的目标是降低峰值左侧，同时把可见传输时间压到最小；它不减少 DiT 的 FLOPs。

## 2. 两个层次

### 组件级：按使用区间释放

一次生成的时间轴是 Encoder \(\to\) \(T\) 次 DiT \(\to\) VAE。Encoder 和 VAE 各被调用一次，DiT 被调用 \(T\) 次。若显存不足以让三者重叠，组件级策略在前者完成后释放其参数，将预算交给 DiT；VAE 开始前再加载。DiT 的重复使用使其常驻通常更划算。

### 层间级：用计算覆盖搬运

令第 \(\ell\) 层的计算时间、H2D 时间为 \(c_\ell,h_\ell\)。预取下一层、计算当前层、释放上一层后，一步的理想时间近似为

\[
\tau_{\rm step}\approx h_1+\sum_{\ell=1}^{L}\max(c_\ell,h_{\ell+1})+h_{\rm tail}.
\]

只有 \(c_\ell\gtrsim h_{\ell+1}\) 时，搬运大多被隐藏。长视频常因单层计算长而受益；小图像模型容易暴露传输为新瓶颈。

## 3. SGLang 的层间流水线

当前 pin 将权重放入 pinned host buffer，以独立 copy stream 和 event 建立“预取 \(\ell+k\) / 计算 \(\ell\) / 释放 \(\ell-1\)”的偏序。设前瞻深度为 \(k\)，跨步常驻层集合为 \(P\)，额外显存约为

\[
M_{\rm extra}\simeq\sum_{j\in P}w_j+\sum_{i=1}^{k}w_{\ell+i}.
\]

所以 \(k\) 同时影响重叠和峰值显存；常驻集合在第一次 denoise forward 后启用，避免加载阶段与其他组件争预算。

## 4. Cache-DiT 的 bucket 版本

Cache-DiT 独立库把选中模块聚成短流水线桶。令桶 \(j\) 的大小为 \(u_j\)，未来已发起的预取集合为 \(Q\)，其调度满足

\[
|Q|\le K,\qquad\sum_{j\in Q}u_j\le B,
\]

其中 \(K\) 是前瞻目标数，\(B\) 是在途和已就绪桶的字节预算。两条 copy stream 分别处理加载和卸载，避免慢 D2H 串行阻塞未来 H2D。它能保留部分 persistent bucket，但接口与 SGLang 逐层 manager 不同。

## 5. 组合约束

缓存需要在未来步访问对应 block 的状态；层间策略则按单次前向生命周期释放该 block。当前 SGLang 认为这个生命周期冲突会导致已释放权重被复用，因此 Cache-DiT 与 DiT layerwise 是硬错误。FSDP 对权重分片/生命周期也有冲突，配置会报错或自动降级。

Progressive Resolution 每步改变工作量；若整模 DiT 又反复从 CPU 传入，固定传输代价会吞掉分辨率降低带来的收益，故官方文档建议关闭整模 CPU offload。

## 6. 决策与验收

当常驻约束无法满足、且大多数 \(c_\ell\) 可以覆盖 \(h_{\ell+1}\) 时，layerwise 值得开启。若仅 Encoder/VAE 造成阶段性峰值，优先组件级调度，见 [Encoder & VAE](encoder_vae.md)。若通信已主导、层计算很短，或计划启用 Cache-DiT，应先测量传输与同步边界。

验收至少记录峰值显存、H2D/D2H 字节数、每层未隐藏时间 \(\max(0,h_{\ell+1}-c_\ell)\)，以及端到端时间 \(\tau\)。

## 附录：实现证据

本页结论对应 SGLang pin `runtime/managers/memory_managers/layerwise_offload.py`、`component_manager.py`、`server_args.py`，以及 Cache-DiT pin `offload/layerwise.py`。组件策略入口、参数名与模型默认值属于实现证据，不能替代显存与 overlap 测量。

## 相关阅读

- [专题总览](overview.md) · [Feature Cache](feature_cache.md) · [Progressive Resolution](progressive_resolution.md) · [Correctness](correctness.md)
