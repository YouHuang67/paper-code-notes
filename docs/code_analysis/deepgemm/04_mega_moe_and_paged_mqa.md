---
tags:
  - CUDA
  - CUTLASS
  - LLM Inference
---
# DeepGEMM 代码分析：Mega MoE 与 paged MQA

[03](03_grouped_gemm_moe_contract.md) 把标准 MoE 接到 grouped GEMM：EP 与 SwiGLU 在核外。本节记录同一库里的两条专用路径：先把不规则工作收成规则任务流，再交给 TMA / UMMA。官方说明见 README 的 [Mega MoE](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/README.md#mega-moe) 与 [V3.2 MQA kernels for the indexer](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/README.md#v32-mqa-kernels-for-the-indexer)。

## 1. Mega MoE：共享 token 池先于 expert GEMM

[`layout/mega_moe.cuh`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/layout/mega_moe.cuh) 把容量建立在「本 rank 最坏能收到多少 routed token」上。记 \(N_{\mathrm{rank}}\) 为参与进程数，\(T_{\max}\) 为每 rank 最大 token 数，\(E_{\mathrm{rank}}\) 为每 rank 的 expert 数。`BLOCK_M` 候选为 \(\{8,16,32,64,96,128,192\}\)，LCM 对齐 384。池上界为

\[
\mathrm{align}\bigl(N_{\mathrm{rank}}\cdot T_{\max}\cdot \min(k_{\mathrm{top}}, E_{\mathrm{rank}}) + E_{\mathrm{rank}}\cdot(192-1),\; 384\bigr)
\]

**源码位置**: [`get_num_max_pool_tokens`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/layout/mega_moe.cuh#L17-L25) · [mega_moe.cuh:L17-L25](src/mega_moe_cuh.md#__codelineno-0-17)

token 进入共享池，局部 expert 从池中消费。`Workspace::get_num_bytes` 从一开始就编进 combine 反向所需对象：grid/NVLink barrier、expert send/recv count、recv count sum、L1 arrival、L2 block arrival mask、dispatch 源 token-topk、以及每池 token 一份 `TokenSrcMetadata`（`rank_idx, token_idx, topk_idx`）。布局是路由控制面。

**源码位置**: [`Workspace::get_num_bytes`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/layout/mega_moe.cuh#L33-L96) · [mega_moe.cuh:L33-L96](src/mega_moe_cuh.md#__codelineno-0-33)

[`sm100_fp8_fp4_mega_moe_impl`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh) 在同一 kernel 持有 L1/L2 两层 MLP 的 act / weight / SF TMA 描述符，并划分 dispatch 线程、MMA 非 epilogue 线程、epilogue/combine 线程。L1 的 CD 为 FP8（2 个 TMA store stage，SwiGLU 后宽为 `BLOCK_N/2`）；L2 的 CD 为 BF16（单 stage，由 epilogue 写回）。通信与 tensor core 在 NVLink 对称内存上重叠。Mega MoE 在多进程对称内存上把 EP 与两层线性收进一次 launch。

**源码位置**: [`sm100_fp8_fp4_mega_moe_impl`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L51-L69) · [站内 L51](src/sm100_fp8_fp4_mega_moe_cuh.md#__codelineno-0-51)；[`SharedStorage`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L185-L217) · [站内 L185](src/sm100_fp8_fp4_mega_moe_cuh.md#__codelineno-0-185)

## 2. paged MQA：元数据把「每个 query 看多少 KV」变成任务流

Lightning indexer 的 logits 是 token-to-token 点积再 ReLU 加权求和（[README](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/README.md#v32-mqa-kernels-for-the-indexer)）。paged 版本的输入含不规则 page table 与变长 context。DeepGEMM 先跑 `smxx_paged_mqa_logits_metadata`：对每个 query（或 varlen atom）把 \(\lceil L_{\mathrm{ctx}} / \mathrm{SPLIT\_KV}\rceil\) 做 warp 前缀和，其中 \(L_{\mathrm{ctx}}\) 为该 query 的 context 长度，`SPLIT_KV` 为每个 KV split 覆盖的 token 数。由此得到该 query 的 segment 数，再把总 segment 均分到 \(N_{\mathrm{SM}}\) 个 SM。每个 SM 得到 `(q_atom_idx, kv_split_idx)` 起点，写入 `schedule_metadata[sm*2], schedule_metadata[sm*2+1]`。

**源码位置**: [`smxx_paged_mqa_logits_metadata`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.cuh#L47-L94) · [paged_mqa_logits.cuh:L47-L94](src/paged_mqa_logits_cuh.md#__codelineno-0-47)

varlen 路径允许相邻 token 共享同一 `indices` 时合成一个 atom（成对 token），把 packed 输入折成更规则的 atom 序列。

**源码位置**: [paged_mqa_logits.cuh:L26-L40](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.cuh#L26-L40) · [站内 L26](src/paged_mqa_logits_cuh.md#__codelineno-0-26)

主 kernel [`sm100_fp4_paged_mqa_logits`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh) 把 CTA 划成四类角色：TMA 装 Q、TMA 装 KV、UMMA、math warpgroup 做 reduce。scheduler 的 `fetch_next_task` 给出本轮 q atom、KV split 与 block table 行；主循环只消费已离散化的任务。地址不规则性由 page table 承担，工作不规则性由 schedule metadata 承担。

**源码位置**: [Q TMA warp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh#L181-L211) · [站内 L181](src/sm100_fp4_paged_mqa_logits_cuh.md#__codelineno-0-181)；[KV TMA warp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh#L212-L221) · [站内 L212](src/sm100_fp4_paged_mqa_logits_cuh.md#__codelineno-0-212)

这与 grouped GEMM 的 scheduler 同构：persistent 领取任务流，任务坐标为 `(q_atom, kv_split)`。

## 3. 任务几何

三条路径共用 01/02 的 TMA / UMMA 执行语言，差别在任务坐标：

- 标准 MoE（[03](03_grouped_gemm_moe_contract.md)）：任务是 M-grouped tile；permute、SwiGLU、finalize 在核外。
- Mega MoE：任务是共享池中的 token 块；EP 与两层线性在同一 launch 内。
- paged MQA：任务是 `(q_atom, kv_split)`；page table 与 metadata kernel 先把变长 context 收成这条流。
