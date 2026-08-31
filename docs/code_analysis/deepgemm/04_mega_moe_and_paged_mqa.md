---
tags:
  - CUDA
  - CUTLASS
  - LLM Inference
---
# DeepGEMM 代码分析：Mega MoE 与 paged MQA

[03](03_grouped_gemm_moe_contract.md) 把标准 MoE 接到 grouped GEMM：EP 与 SwiGLU 在核外。本节记录同一库里的两条专用路径：先把不规则工作收成规则任务流，再交给 TMA / UMMA。

## 1. Mega MoE：共享 token 池先于 expert GEMM

`layout/mega_moe.cuh` 把容量建立在「本 rank 最坏能收到多少 routed token」上。`BLOCK_M` 候选为 \(\{8,16,32,64,96,128,192\}\)，LCM 对齐 384。池上界为

\[
\mathrm{align}\bigl(N_{\mathrm{rank}}\cdot T_{\max}\cdot \min(k_{\mathrm{top}}, E_{\mathrm{rank}}) + E_{\mathrm{rank}}\cdot(192-1),\; 384\bigr)
\]

**源码位置**: [`get_num_max_pool_tokens`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/layout/mega_moe.cuh#L17-L25)

token 进入共享池，局部 expert 从池中消费。`Workspace` 同时编码 combine 反向所需的 `TokenSrcMetadata`（`rank_idx, token_idx, topk_idx`）、arrival mask、expert send/recv count。布局是路由系统的控制面，不只是算子输入格式。[`mega_moe.cuh:L33-L80`](src/mega_moe_cuh.md#__codelineno-0-33)

`sm100_fp8_fp4_mega_moe_impl` 在同一 kernel 持有 L1/L2 两层 MLP 的 act / weight / SF TMA 描述符，并划分 dispatch 线程、MMA 非 epilogue 线程、epilogue/combine 线程。L1 输出经 SwiGLU 后以 FP8 进 L2；通信与 tensor core 在 NVLink 对称内存上重叠。这与标准路径「两次 grouped GEMM + 核外激活」是不同产品边界：标准路径可组合 DeepEP；Mega MoE 要求多进程对称内存，把 EP dispatch/combine 与两层线性收进一次 launch。

## 2. paged MQA：元数据把「每个 query 看多少 KV」变成任务流

Lightning indexer 的 logits 是 token-to-token 点积再 ReLU 加权求和（见仓库 README）。paged 版本面对不规则 page table 与变长 context。DeepGEMM 先在 `smxx_paged_mqa_logits_metadata` 里生成 `schedule_metadata`：对每个 query（或 varlen atom）把 `ceil_div(context_len, SPLIT_KV)` 做 warp 前缀和，再把总 segment 均分到 \(N_{\mathrm{SM}}\) 个 SM。每个 SM 得到 `(q_atom_idx, kv_split_idx)` 起点。[`paged_mqa_logits.cuh:L47-L94`](src/paged_mqa_logits_cuh.md#__codelineno-0-47)

varlen 路径允许相邻 token 共享同一 `indices` 时合成一个 atom（成对 token），把 packed 输入折成更规则的 atom 序列。[`paged_mqa_logits.cuh:L26-L40`](src/paged_mqa_logits_cuh.md#__codelineno-0-26)

主 kernel `sm100_fp4_paged_mqa_logits` 把 CTA 划成四类角色：TMA 装 Q、TMA 装 KV、UMMA、math warpgroup 做 reduce。scheduler 的 `fetch_next_task` 给出本轮 q atom、KV split 与 block table 行；主循环只消费已离散化的任务。地址不规则性由 page table 承担，工作不规则性由 schedule metadata 承担。

这与 grouped GEMM 的 scheduler 同构：都是 persistent 领取任务流。差别是任务坐标从 `(m_block, n_block)` 换成 `(q_atom, kv_split)`。

## 3. 和标准 MoE 的关系

- 标准 MoE + grouped GEMM：两次 M-grouped GEMM；DeepEP 或 `ep_scatter` / `src2dst` 在核外；框架做 permute、SwiGLU、finalize。
- Mega MoE：单核 L1+SwiGLU+L2；kernel 内 NVLink 与 token 池；框架提供对称内存与权重 transform。
- paged MQA：logits GEMM 族；page table 与 metadata kernel；框架提供 `context_lens` / `indices`。

标准 MoE 适配走 [03](03_grouped_gemm_moe_contract.md) 的契约即可。Mega MoE 与 paged MQA 说明同一分层（scheduler + TMA + MMA + epilogue）可以换任务几何，而不必换执行语言。
