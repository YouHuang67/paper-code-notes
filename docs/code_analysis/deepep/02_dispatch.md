---
tags:
  - CUDA
  - CUTLASS
---

# DeepEP：Dispatch 内核

本文详细拆解 Dispatch 的完整流程。Dispatch 负责将每个 token 路由到其 top-k 专家所在的 GPU。

**源码**: [dispatch.cuh](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/include/deep_ep/impls/dispatch.cuh)、[hybrid_dispatch.cuh](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh)、[dispatch_copy_epilogue.cuh](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/include/deep_ep/impls/dispatch_copy_epilogue.cuh)、[dispatch.hpp](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/elastic/dispatch.hpp)

## 函数签名与模板参数

[dispatch.cuh](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/include/deep_ep/impls/dispatch.cuh)

```cpp
template <bool kIsScaleupNVLink,       // scaleup 域是否全 NVLink（决定用 TMA store 还是 RDMA put）
          bool kDoCPUSync,             // 是否等待 CPU 端获知精确的 receive count
          bool kReuseSlotIndices,      // cached mode：复用上次 dispatch 的 slot 分配
          int kNumSMs,                 // 使用的 SM 数
          int kNumNotifyWarps,         // Notify warp 数（4）
          int kNumDispatchWarps,       // 数据搬运 warp 数
          int kNumRanks,               // 总 rank 数
          int kNumHiddenBytes,         // hidden * elem_size
          int kNumSFPacks,             // scale factor pack 数（FP8 模式），BF16 时为 0
          int kNumMaxTokensPerRank,    // 每 rank 最大 token 数（预分配 buffer slot 数）
          int kNumExperts, kNumTopk,   // 专家总数、top-k 选择数
          int kExpertAlignment,        // 每专家接收 token 数对齐粒度
          int kNumQPs,                 // RDMA QP 数
          int64_t kNumTimeoutCycles>   // GPU 超时周期数
__global__ void dispatch_impl(
    void* x, sf_pack_t* sf,           // 输入数据 + scale factor
    topk_idx_t* topk_idx,             // [num_tokens, num_topk] 专家索引
    float* topk_weights,              // [num_tokens, num_topk] 门控权重
    topk_idx_t* copied_topk_idx,      // 输出：拷贝的 topk_idx（供 handle 保存）
    int* cumulative_local_expert_recv_stats,  // 输出：每专家累计 token 数（负载监控）
    int* psum_num_recv_tokens_per_scaleup_rank,  // 输出：每 rank 的接收 token 前缀和
    int* psum_num_recv_tokens_per_expert,        // 输出：每专家的接收 token 前缀和
    int* dst_buffer_slot_idx,         // 输出：每个 token 的目标 buffer slot
    const int num_tokens,
    const int sf_token_stride, sf_hidden_stride,
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
    void* buffer,                     // 预分配的通信 buffer
    void* workspace, void* mapped_host_workspace,  // workspace + CPU 可见 workspace
    const int rank_idx)
```

## 总体流程

```
Dispatch 内核分为三个 warp 角色：

Notify Warps (warp 0-3):
  遍历 tokens → atomicAdd 统计 per-rank/per-expert 计数
  → 跨 SM reduce → 通知远程 GPU → 等待远程通知
  → 本地对齐/prefix sum → 写入 CPU workspace

Dispatch Warps (warp 4+):
  for each token (interleaved):
    1. TMA load token data → smem TMA buffer
    2. cp.async load scale factors → smem (FP8)
    3. Load topk_idx / topk_weights
    4. 去重 + atomicAdd 分配 buffer slot
    5. [NVLink] TMA store → 远程 GPU recv buffer
       [RDMA]  TMA store → 本地 send buffer → Gin put → 远程
    6. mbarrier 同步

Barrier:
  确保所有远程数据到达

Copy Epilogue (独立 kernel):
  遍历 recv buffer → TMA load → TMA store 到输出 tensor
  → 填充 recv_src_metadata（路由元数据）
```

## Notify Warps：Token 统计

### 共享内存初始化与本地统计

```cpp
// smem 布局为 rank_expert_count[0..kNumRanks-1]  rank_expert_count[kNumRanks..kNumRanks+kNumExperts-1]
int *rank_count = rank_expert_count, *expert_count = rank_expert_count + kNumRanks;

// 清零
for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
    rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
named_barrier(kNotifyBarrierIndex);

// 遍历 tokens：每个 notify warp 处理 interleaved 的 token 子集
// global_warp_idx = sm_idx * kNumNotifyWarps + warp_idx
for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
    // 读取专家索引（lane 0..kNumTopk-1 各负责一个 topk 选择）
    dst_expert_idx = lane_idx < kNumTopk ?
        static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
    if (dst_expert_idx >= 0)
        atomicAdd_block(expert_count + dst_expert_idx, 1);  // 按专家计数

    // 按 rank 去重计数：同一 token 的多个 topk 选择可能落在同一 rank
    dst_rank_idx = dst_expert_idx >= 0 ?
        dst_expert_idx / kNumExpertsPerRank : -1;
    if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
        atomicAdd_block(rank_count + dst_rank_idx, 1);      // 每 rank 只计一次
}
named_barrier(kNotifyBarrierIndex);
```

去重逻辑（`ptx::deduplicate`）：当 warp 内多个 lane 有相同的 `dst_rank_idx` 时，只保留 lane ID 最小的那个执行后续操作。

### 跨 SM Reduce

```cpp
// 所有 SM 将自己的统计写入 global workspace 做 reduce
for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
    int64_t counter = (1ll << 32ll) | rank_expert_count[i];
    // 高 32 位 = 参与 SM 数，低 32 位 = 累计 count
    ptx::red_add(workspace.get_notify_reduction_workspace_ptr() + i, counter);
}

// SM 0 的 notify warps 等待所有 SM 完成并聚合结果
if (sm_idx == 0) {
    for (int i = thread_idx; ...) {
        timeout_while([=]() {
            status = ld_volatile<int64_t>(workspace.get_notify_reduction_workspace_ptr() + i);
            if ((status >> 32) == kNumSMs) {     // 所有 SM 已到达
                count = status & 0xffffffff;
                encoded = encode_decode_positive(count);  // 编码为就绪标记
                rank_expert_count[i] = encoded;           // 存回 smem
                if (!kIsScaleupNVLink)
                    workspace.get_scaleup_rank_expert_count_ptr<true>()[i] = encoded;  // RDMA 模式写 send buffer
                workspace.get_notify_reduction_workspace_ptr()[i] = 0;  // 清理供下次使用
                return true;
            }
            ...
        });
    }
```

`red_add` 是 NVLink 硬件加速的远程原子加。高 32 位累积 SM 数，低 32 位累积 token 数。

### 跨 Rank 通知与接收

SM 0 的 notify warps 使用 Gin 将计数通知到其他 rank：

```cpp
// 1. 发送 rank 计数到目标 rank
for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
    gin.put_value<team_t>(
        workspace.get_scaleup_rank_count_ptr<false>() + rank_idx,  // 源地址（本 rank 的 slot）
        static_cast<int64_t>(rank_count[i]),                         // 值
        i,                                                           // 目标 rank
        ncclGinOptFlagsAggregateRequests);                           // 合并请求
}

// 2. 发送 expert 计数
if constexpr (kIsScaleupNVLink) {
    // NVLink: 逐元素 put_value
    for (int i = thread_idx; i < kNumExperts; i += kNumNotifyThreads)
        gin.put_value<team_t>(...);
} else {
    // RDMA: 批量 put（一次性传输整个 expert 段）
    for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads)
        gin.put<team_t>(dst_ptr, src_ptr, kNumExpertsPerRank * sizeof(int64_t), i);
}

// 3. 等待所有 rank 的计数就绪
for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
    timeout_while([=]() {
        count = ld_volatile<int64_t>(recv_count_ptr[i]);
        decoded = encode_decode_positive(count);
        if (is_decoded_positive_ready(decoded)) {
            rank_expert_count[i] = decoded;  // 存回 smem
            recv_count_ptr[i] = 0;            // 清理
            return true;
        }
        ...
    });
}
```

### 后处理：对齐、CPU 同步、Prefix Sum

```cpp
// 对每个 local expert 的 token 数做 alignment
for (int i = thread_idx; i < kNumExpertsPerRank; i += kNumNotifyThreads) {
    int sum = 0;
    for (int j = 0; j < kNumRanks; ++ j)
        sum += expert_count[j * kNumExpertsPerRank + i];  // 跨 rank 求和
    expert_count[i] = math::align(sum, kExpertAlignment);
    // 更新负载统计
    if (cumulative_local_expert_recv_stats != nullptr)
        atomicAdd(cumulative_local_expert_recv_stats + i, sum);
}

// CPU 同步：将计数写入 mapped_host_workspace（CPU 可轮询）
if constexpr (kDoCPUSync) {
    for (int i = thread_idx; i < kNumRanks + kNumExpertsPerRank; i += kNumNotifyThreads)
        host_workspace.get_scaleup_rank_expert_count_ptr<false>()[i] =
            encode_decode_positive(rank_expert_count[i]);
}

// Warp 级 prefix sum（用 shfl 实现，避免 cub::BlockScan 的依赖）
auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
    int psum = 0;
    for (int i = 0; i < ceil_div(n + is_exclusive, 32); ++ i) {
        int value = (0 <= lane_idx - is_exclusive < n) ? count[lane_idx - is_exclusive] : 0;
        int sum = psum + warp_inclusive_sum(value);
        if (lane_idx < n + is_exclusive) out[lane_idx] = sum;
        psum = shfl(sum, 31);  // 用最后一个 lane 的值更新 psum
    }
};
// warp 0 做 rank 维度的 inclusive prefix sum
if (warp_idx == 0) do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumRanks, 0);
// warp 1 做 expert 维度的 exclusive prefix sum（供 expand 模式使用）
if (warp_idx == 1) do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
```

用 shfl 做 prefix sum 而非 `cub::BlockScan`，因为数据量小（kNumRanks ≤ 32），warp 级 shfl 更快且无额外依赖。

## Dispatch Warps：数据搬运

### Shared Memory 管理

```cpp
const int dispatch_warp_idx = warp_idx - kNumNotifyWarps;

// 每个 dispatch warp 有独立的 TMA buffer（在 smem 中）
const auto token_layout = TokenLayout(
    kNumHiddenBytes,                       // hidden 数据
    kNumSFPacks * sizeof(sf_pack_t),       // scale factor（可选）
    kNumTopk, true);                       // topk 元数据

// smem 布局：先分配给 notify warps 的 rank_expert_count，
// 然后 dispatch warps 各自独立 TMA buffer（不重叠）
const auto tma_buffer = BufferLayout<true>(token_layout, kNumDispatchWarps, 1,
    smem + kNumSmemBytesForNotify)
    .get_rank_buffer(dispatch_warp_idx).get_token_buffer(0);

// 初始化 mbarrier
if (elect_one_sync()) mbarrier_init_with_fence(mbarrier_ptr, 1);
```

### Token 处理循环

```cpp
for (int token_idx = token_start; token_idx < num_tokens; token_idx += token_stride) {
    // ========== 阶段 1: 加载 ==========

    // TMA load: hidden data → smem TMA buffer
    if (elect_one_sync())
        tma_load_1d(tma_buffer.get_hidden_ptr(),
                     x + token_idx * kNumHiddenBytes / sizeof(x[0]),
                     mbarrier_ptr, kNumHiddenBytes);

    // cp.async: scale factors → smem（FP8 模式）
    // 分块拷贝，每 lane 负责 1 个 sf_pack
    if constexpr (kNumSFPacks > 0) {
        for (int k = 0; k < full_iters; ++ k)
            cp_async_ca(src_sf + (k * 32 + lane_idx) * sf_hidden_stride,
                        dst_sf + k * 32 + lane_idx);
        cp_async_mbarrier_arrive(mbarrier_ptr);  // SF 拷贝完成后 arrive mbarrier
    }

    // Load topk 索引和权重到 smem
    if (lane_idx < kNumTopk) {
        tma_buffer.get_topk_idx_ptr()[lane_idx] = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
        if (topk_weights != nullptr)
            tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
        if (copied_topk_idx != nullptr)
            copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
    }

    // 写入源 metadata：rank 索引 + token 全局编号
    if (elect_one_sync())
        *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
    tma_store_fence();  // 确保 metadata 写入在 TMA store 开始前可见
```

TMA load 是异步的，`cp.async` 也是异步的。两者通过 mbarrier 协调：`cp.async` 完成后 `cp_async_mbarrier_arrive`，TMA load 完成后 `mbarrier_arrive_and_set_tx` 设置 expected transaction count，然后 `mbarrier_wait_and_flip_phase` 等待两者都完成。

```cpp
    // ========== 阶段 2: 分配目标 slot ==========

    // 去重 + atomicAdd 分配
    if constexpr (kReuseSlotIndices) {
        // Cached mode: 直接用上次的 slot 索引
        if (lane_idx < kNumTopk)
            stored_dst_slot_idx = __ldg(dst_buffer_slot_idx + token_idx * kNumTopk + lane_idx);
        stored_dst_slot_idx = stored_dst_slot_idx >= 0 ?
            (stored_dst_slot_idx - rank_idx * kNumMaxTokensPerRank) : -1;
    } else {
        // 正常模式：去重后 atomicAdd 分配
        if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)
            stored_dst_slot_idx = atomicAdd(workspace.get_scaleup_atomic_sender_counter() + stored_dst_rank_idx, 1);
        if (lane_idx < kNumTopk) {
            dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] =
                stored_dst_slot_idx >= 0 ? rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx : -1;
        }
    }

    // ========== 阶段 3: 等待 + 发送 ==========

    // 等待 TMA load 和 cp.async 全部完成
    if (elect_one_sync()) {
        mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);  // 设定预期 transaction
        mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);          // 等待完成并翻转 phase
    }

    // RDMA 模式：TMA store 到本地 send buffer
    if constexpr (not kIsScaleupNVLink) {
        if (elect_one_sync())
            tma_store_1d(send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                         tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
        tma_store_commit();
    }
```

TMA store 分为两条路径：

```cpp
    // NVLink 路径：TMA store 直接写入远程 GPU 的 recv buffer
    const auto dst_ptr = stored_dst_slot_idx >= 0 ?
        gin.get_sym_ptr<team_t>(recv_buffer.get_rank_buffer(rank_idx)
                                 .get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                stored_dst_rank_idx) : nullptr;
    if (dst_ptr != nullptr)
        tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
    tma_store_commit();

    // RDMA 路径：等待 send buffer 的 TMA store 完成，然后 RDMA put
    if constexpr (not kIsScaleupNVLink) {
        tma_store_wait<1>();  // 等待本地 send buffer 的 TMA store
        if (stored_dst_slot_idx >= 0 and dst_ptr == nullptr) {
            // dst_ptr == nullptr 表示目标不在 NVLink 可达域，走 RDMA
            gin.put<team_t>(
                recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),  // 远程目标
                send_buffer_ptr,                                                    // 本地源
                tma_buffer.get_num_bytes<false>(),                                  // 字节数
                stored_dst_rank_idx);                                               // 目标 rank
        }
    }
}
```

关键判断逻辑：
- `dst_ptr != nullptr`：目标在 NVLink 可达域 → TMA store 直接写入远程
- `dst_ptr == nullptr`：目标需要 RDMA → TMA store 到 send buffer → Gin put
- `kIsScaleupNVLink` 为 true 时不需要 send buffer 和 RDMA 路径

## Copy Epilogue

[dispatch_copy_epilogue.cuh](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/include/deep_ep/impls/dispatch_copy_epilogue.cuh)

主 kernel 通过 `cudaTriggerProgrammaticLaunchCompletion()` 触发 PDL 链中的下一个 kernel。Copy epilogue 负责：

```cpp
// 等待主 kernel 完成（PDL 保证）
cudaGridDependencySynchronize();

// 如果不做 CPU sync，则从 GPU tensor 读取实际 receive count
if (num_recv_tokens == kNumMaxTokensPerRank * kNumRanks)
    num_recv_tokens = psum_num_recv_tokens_per_scaleup_rank[kNumScaleupRanks - 1];

// 遍历所有收到的 token
for (int i = global_warp_idx; i < num_recv_tokens; i += kNumWarps * kNumSMs) {
    // 1. 找到 token 属于哪个 rank（通过 prefix sum 做二分）
    while (i >= current_rank_end) {
        current_rank_idx += 1;
        // 用 shfl 从 prefix sum 数组读取对应 rank 的 range
        current_rank_end = shfl(psum_num_recv_tokens_per_scaleup_rank[current_rank_idx + lane_idx],
                                current_rank_idx % 32);
    }

    // 2. TMA load token 从 recv buffer → smem
    if (elect_one_sync())
        tma_load_1d(tma_buffer.get_base_ptr(), buffer_token.get_base_ptr(),
                     mbarrier_ptr, tma_buffer.get_num_bytes<false>());

    // 3. 验证专家索引是否属于本地（range check）
    // expert_start_idx = num_experts / num_ranks * rank_idx
    in_range = expert_start_idx <= dst_expert_idx < expert_end_idx;
    dst_expert_idx = in_range ? dst_expert_idx - expert_start_idx : -1;

    // 4. 计算目标 tensor 中的位置
    if (not kDoExpand) {
        dst_tensor_idx = i;  // 直接映射
    } else if (dst_expert_idx >= 0) {
        dst_tensor_idx = atomicAdd(psum_num_recv_tokens_per_expert + dst_expert_idx, 1);  // expand: 原子分配
    }

    // 5. 等待 TMA load 完成 → TMA store 到输出 tensor
    if (elect_one_sync()) mbarrier_wait_and_flip_phase(...);
    tma_store_1d(recv_x + dst_tensor_idx * kNumHiddenBytes, tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
    tma_store_commit();

    // 6. 存储 scale factor (cp.async 从 smem 到 gmem)
    // 7. 存储 topk_idx / topk_weights
    // 8. 填充 recv_src_metadata（供 combine 使用）
    recv_src_metadata[i * kMetadataStride + 0] = src_token_global_idx;  // 源全局 token 索引
    recv_src_metadata[i * kMetadataStride + 1] = current_rank_idx * kNumTopk + master_src_topk_idx;  // 源 rank + topk
    // expand 模式下额外存储每个 topk 选择的 slot 索引
    if (kDoExpand and lane_idx < kNumTopk)
        recv_src_metadata[i * kMetadataStride + 2 + lane_idx] = dst_tensor_idx;
}
```

recv_src_metadata 的格式：
```
每 token 一行: [src_token_global_idx, src_rank_topk_idx, topk_slot_0, ..., topk_slot_{kNumTopk-1}]
共 2 + kNumTopk 个 int
```

## Hybrid Dispatch

[hybrid_dispatch.cuh](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh)

Hybrid 模式有三类 warp：

### Notify Warps

与 direct 模式类似，但通知流程分两级：
1. Scaleout 维度：聚合所有 scaleup rank 的计数 → `gin.put<ncclTeamTagRail>` 发送给所有 scaleout peer
2. Scaleup 维度：等待所有 scaleout peer 的计数 → reduce → `gin.put_value<ncclTeamTagLsa>` 发给本地 scaleup peer

### Scaleout Warps

```cpp
// 每个 channel 独立运行
const int channel_idx = sm_idx * kNumChannelsPerSM + scaleout_warp_idx;

// 预加载第一个 token（TMA + cp.async 异步）
preload_next_token(channel_idx);

for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += kNumChannels) {
    // 1. Load topk 信息
    // 2. 去重 → stored_dst_slot_idx（在 scaleout_recv_buffer 中）
    stored_dst_slot_idx = stored_old_slot_idx;  // tail 即为 slot

    // 3. 更新 scale-out tail
    //    bitmask 统计哪些 rank 需要数据，tail += popcount(mask)
    stored_scaleout_tail += (scaleout_rank_mask >> lane_idx) & 1;

    // 4. 等待 TMA 到达 → TMA store 到 send buffer（仅当有非本地 rank 需要数据时）
    mbarrier_wait_and_flip_phase(...);
    if (scaleout_rank_mask ^ (1 << scaleout_rank_idx))  // 跳过纯本地 rank
        tma_store_1d(send_buffer_ptr, tma_buffer);

    // 5. 本地 bypass：TMA store 直接写入本地 recv buffer
    if (stored_dst_scaleout_rank_idx == scaleout_rank_idx)
        tma_store_1d(local_recv_buffer, tma_buffer);

    // 6. RDMA put：非本地 rank
    if (stored_dst_scaleout_rank_idx != scaleout_rank_idx)
        gin.put<ncclTeamTagRail>(remote_recv_buffer, send_buffer_ptr, ..., dst_rank);

    // 7. 通知 forward warps（更新 tail）
    update_scaleout_tail();  // red_add + signaled_tail 到 workspace
}
```

`update_scaleout_tail` 的实现：

```cpp
// 每 kScaleoutUpdateInterval=3 个 slot 更新一次 tail（减少 red_add 开销）
if (stored_scaleout_tail >= stored_old_scaleout_tail + kScaleoutUpdateInterval or finish_flag) {
    signaled_tail = pack2(finish_flag, stored_scaleout_tail);     // 高 32: finish, 低 32: tail
    old_signaled = pack2(0, stored_old_scaleout_tail);
    gin.red_add_rel<ncclTeamTagRail>(tail_ptr, signaled_tail - old_signaled, lane_idx);
    // lane_idx = dst_rank: 只通知目标 scaleout rank 的对应 lane
}
```

### Forward Warps

```cpp
// 轮询所有 scaleout rank 的 tail，round-robin 消费数据
while (wip_mask = gather(stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or finish_flag == 0)) {
    // 1. 选择下一个有数据的 scaleout rank（round-robin + ffs）
    recv_scaleout_rank_idx = ffs(wip_mask);

    // 2. 等待该 rank 的 tail 更新
    timeout_while(ld_acquire_sys(tail_ptr) 未到达);

    // 3. 消费一个 chunk（最多 kNumSlotsPerForwardChunk=3 个 slot）
    for (slot_idx = start; slot_idx < end; ++slot_idx) {
        // TMA load token 从 scaleout_recv_buffer
        // 读取 topk 索引 → 确定目标 scaleup rank
        // 去重 → atomicAdd 分配 scaleup slot
        // TMA store 直接写入 scaleup buffer（NVLink 路径）
        // 记录 token_metadata_at_forward（供 combine 使用）
        // 更新 channel_linked_list（供 combine 使用）
    }
}

// 写入 linked list 的尾部标记（-1）
// 写入 channel_scaleup_tail（供 combine forward warps 使用）
```

Forward warp 的核心职责是"拆包重路由"：从 scaleout 域收到的是按 `(channel, scaleout_rank)` 分组的数据，需要重新按 `(scaleup_rank, expert)` 分发到 scaleup recv buffer。

### Hybrid 模式下的 Copy Epilogue

与 direct 模式相同，但额外维护 `channel_linked_list`：
- 按 channel 建立 token → 全局索引的链表
- `channel_linked_list[channel][idx][scaleup_rank]` 指向该 channel 的第 idx 个 token 在 scaleup buffer 中的位置
- 链表尾部写入 `-1`

## 关键启示

- **Notify warp 的"通知-等待"两阶段协议**是关键同步模式：发送方写入 local count → remote 轮询等待 → 清理并返回。这种模式避免了全局 barrier 的性能开销
- **deduplicate + atomicAdd** 是高效的 slot 分配策略。deduplicate 保证每个 rank 只被计数一次，atomicAdd 保证 slot 分配的原子性
- **TMA store + RDMA put 的双路径设计**最大化 NVLink 带宽：NVLink 可达的 rank 用 TMA store 直接写入（零拷贝），不可达的走 send buffer → RDMA put
- **Hybrid 的流水线设计**：scaleout warps 不断接收并发起 RDMA、forward warps 不断消费并转发到 scaleup，两者通过 tail 指针做生产者-消费者同步
- **PDL（Programmatic Dependent Launch）**让 dispatch kernel 和 copy epilogue kernel 串行化而不需要 CPU 同步，`cudaGridDependencySynchronize` 等待前一个 kernel 的 grid 完成
