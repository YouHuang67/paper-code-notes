---
tags:
  - CUDA
  - CUTLASS
---

# DeepEP：Combine 内核

本文详细拆解 Combine 的完整流程。Combine 负责将各 GPU 上专家计算的结果按 token 归约回原始 GPU。

**源码**: [combine.cuh](../../../refs/DeepEP/deep_ep/include/deep_ep/impls/combine.cuh)、[hybrid_combine.cuh](../../../refs/DeepEP/deep_ep/include/deep_ep/impls/hybrid_combine.cuh)、[combine_reduce_epilogue.cuh](../../../refs/DeepEP/deep_ep/include/deep_ep/impls/combine_reduce_epilogue.cuh)、[combine_utils.cuh](../../../refs/DeepEP/deep_ep/include/deep_ep/impls/combine_utils.cuh)、[combine.hpp](../../../refs/DeepEP/csrc/kernels/elastic/combine.hpp)

## 函数签名与模板参数

[combine.cuh](../../../refs/DeepEP/deep_ep/include/deep_ep/impls/combine.cuh)

```cpp
template <bool kIsScaleupNVLink,            // scaleup 域是否全 NVLink
          bool kUseExpandedLayout,          // expand 模式：token 在输出中展开为每 expert 一个 slot
          bool kAllowMultipleReduction,     // 允许本地多 token reduce
          int kNumSMs, int kNumWarps,       // SM 和 warp 数
          int kNumRanks,                    // 总 rank 数
          int kHidden,                      // hidden 维度
          int kNumMaxTokensPerRank,         // 每 rank 最大 token 数
          int kNumExperts, int kNumTopk,    // 专家总数、topk
          int kNumQPs, int64_t kNumTimeoutCycles>
__global__ void combine_impl(
    nv_bfloat16* x,                        // [num_tokens, hidden] 本地专家输出
    float* topk_weights,                   // [num_tokens, num_topk] 门控权重
    int* src_metadata,                     // [num_reduced_tokens, 2+kNumTopk] 路由元数据
    int* psum_num_recv_tokens_per_scaleup_rank,  // 每 rank 接收 token 前缀和
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
    void* buffer, void* workspace,
    const int rank_idx,
    int num_reduced_tokens)                // 需要 reduce 的 token 数
```

## Buffer 布局

Combine 的 buffer 布局与 dispatch 是一对镜像（同一块 buffer 复用）：

```cpp
// Direct 模式：
const auto recv_buffer = BufferLayout<false>(
    token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, buffer);
const auto send_buffer = BufferLayout<false>(
    token_layout, kNumRanks,
    kNumMaxTokensPerRank * (kDoExpandedSend ? kNumTopk : 1),  // expand no-reduce 时需要 kNumTopk 倍空间
    recv_buffer.get_buffer_end_ptr());
```

`kNumTokensInLayout` 由 `use_rank_layout` 和 `get_num_tokens_in_layout` 在编译期决定：

```cpp
// combine_utils.cuh
// kAllowMultipleReduction: true 时可能按 rank 排布（使同一 rank 的多个 slot 在 smem 中连续）
// kNumRanks <= kNumTopk: 当 rank 数 ≤ topk 数时用 rank layout，否则用 topk layout
constexpr bool use_rank_layout = kAllowMultipleReduction && kNumRanks <= kNumTopk;

// kUseRankLayout: recv buffer 有 kNumRanks 个 slot 组
// !kUseRankLayout: recv buffer 有 kNumTopk 个 slot 组
constexpr int get_num_tokens_in_layout = kAllowMultipleReduction
    ? min(kNumRanks, kNumTopk) : kNumTopk;
```

## 总体流程

```
Combine 分为两个 kernel：

Combine 主 kernel:
  1. Barrier → 确保 dispatch 的远程 buffer 可写入
  2. for each reduced token:
     a. 从 src_metadata 读取源 rank 和 topk 信息
     b. 判断本地 reduce 模式（no-reduce / reduce / expanded-send）
     c. TMA load token → smem
        - no-reduce: 直接从 x 加载
        - reduce: 从 x 的多个 slot 加载 + 本地加权求和
     d. NVLink 路径：TMA store → 远程 recv buffer
        RDMA 路径：TMA store → send buffer → Gin put
     e. 写入 topk_weights
  3. Barrier → 确保所有远程数据到达

Reduce Epilogue（独立 kernel）:
  1. 从 recv buffer TMA load 数据
  2. 按 topk 做 weighted reduce / 直接拷贝
  3. 加 bias
  4. 写出 combined_x
```

## Combine 主 Kernel

### 初始化与 Barrier

```cpp
// warp 索引旋转（避免所有 rank 的 warp 0 竞争同一 QP）
const auto warp_idx = (get_warp_idx() + rank_idx) % kNumWarps;

// smem TMA buffer 初始化
// combine 的 TokenLayout 不含 SF（SF 只在 dispatch 时传输）
const auto token_layout = TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);

const auto tma_buffer = BufferLayout<true>(token_layout, kNumWarps, 1, smem)
    .get_rank_buffer(warp_idx).get_token_buffer(0);

// 初始化 mbarrier
if (elect_one_sync()) mbarrier_init_with_fence(mbarrier_ptr, 1);

// 获取 Gin handle（每个 warp 一个 QP/channel）
const auto [qp_idx, sharing_mode] =
    comm::get_qp_mode<kNumSMs, kNumQPs, kNumWarps>(sm_idx, warp_idx);
const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

// Barrier：确保 dispatch 阶段的数据已全部被消费，远程 buffer 可安全写入
comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks, ...>(gin, ..., comm::kCombineTag0, ...);
```

[combine.cuh](../../../refs/DeepEP/deep_ep/include/deep_ep/impls/combine.cuh)

### 主循环：逐 Token 处理

```cpp
int num_tokens_per_warp = ceil_div(num_reduced_tokens, kNumSMs * kNumWarps);
for (int i = token_start_idx; i < token_end_idx; ++ i) {
    // ========== 阶段 1: 读取元数据 ==========
    constexpr int kMetadataStride = 2 + kNumTopk;
    const int src_token_idx = __ldg(src_metadata + i * kMetadataStride) % kNumMaxTokensPerRank;
    const int src_rank_topk_idx = __ldg(src_metadata + i * kMetadataStride + 1);
    const int src_rank_idx = src_rank_topk_idx / kNumTopk;
    const int src_topk_idx = src_rank_topk_idx % kNumTopk;
    // 元数据由 dispatch copy epilogue 填充

    // 判断目标 buffer
    const bool nvlink_bypass = gin.is_nvlink_accessible<team_t>(src_rank_idx);
    auto master_token_buffer = [=]() {
        if (nvlink_bypass) {
            // NVLink 路径：直接定位到远程 recv buffer 的目标地址
            auto token_buffer = recv_buffer
                .get_rank_buffer(kUseRankLayout ? rank_idx : src_topk_idx)
                .get_token_buffer(src_token_idx);
            token_buffer.set_base_ptr(gin.get_sym_ptr<team_t>(token_buffer.get_base_ptr(), src_rank_idx));
            return token_buffer;
        }
        // RDMA 路径：先写到本地 send buffer
        return send_buffer.get_rank_buffer(src_rank_idx).get_token_buffer(src_token_idx);
    }();
```

用 `gin.get_sym_ptr` 将本地地址转换为远程 LSA 地址，后续 TMA store 直接写入远程 GPU。

### 阶段 2: 三种数据写入模式

**模式 A: no-reduce（非 expand 或 expand + 单 topk 命中）**

```cpp
auto no_local_reduce = not kUseExpandedLayout
    or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);

if (no_local_reduce) {
    // 确定源 tensor 中的索引
    int token_idx_in_tensor = i;
    if constexpr (kUseExpandedLayout)
        token_idx_in_tensor = shfl(stored_topk_slot_idx, master_lane_idx(reduce_valid_mask));

    // TMA load → smem → TMA store
    if (elect_one_sync()) {
        tma_load_1d(tma_buffer, x + token_idx_in_tensor * kNumHiddenBytes, mbarrier, kNumHiddenBytes);
        mbarrier_arrive_and_set_tx(mbarrier, kNumHiddenBytes);
        mbarrier_wait_and_flip_phase(mbarrier, phase);
        tma_store_1d(master_token_buffer, tma_buffer, kNumHiddenBytes);
        tma_store_commit();
    }
```

**模式 B: local reduce（expand + multiple_reduction）**

```cpp
else if constexpr (kAllowMultipleReduction) {
    // 对同一 token 的多个 expand slot 做本地 reduce

    // 1. 按 valid mask 排序 topk slot 索引到数组前部
    int topk_slot_idx[kNumTopk];
    compute_topk_slots(topk_slot_idx, reduce_valid_mask,
        [=](int idx) { return shfl(stored_topk_slot_idx, idx); });

    // 2. 从 x 的多个 slot 加载数据到 smem，逐元素求和
    combine_reduce<kHiddenVec, kUnrollFactor, ceil_div(kNumTopk, kNumRanks)>(
        lane_idx, topk_slot_idx, tma_buffer,
        // 源地址回调：按 slot_idx 定位 x 中的 token
        [=](int slot_idx) { return x + slot_idx * kNumHiddenBytes / sizeof(vec_t); },
        // 等待 buffer 释放回调：TMA store 完成后才能重用 smem
        [=]() { tma_store_wait(); }
    );

    // 3. TMA store reduce 结果到目标 buffer
    if (elect_one_sync()) {
        tma_store_1d(master_token_buffer, tma_buffer, kNumHiddenBytes);
        tma_store_commit();
    }
}
```

**模式 C: expanded send（expand + 无 multiple_reduction）**

```cpp
else {
    // 不做本地 reduce，所有 expand 的 slot 都直接发送
    for (int k = 0; k < kNumTopk; ++ k) {
        slot_idx = shfl(stored_topk_slot_idx, k);
        if (slot_idx >= 0) {
            // TMA load → TMA store 到对应 topk 维度的 buffer slot
            if (nvlink_bypass) {
                tma_store_1d(gin.get_sym_ptr(token_buffer[k].get_base_ptr(), src_rank_idx), ...);
            } else {
                // 先 TMA store 到 send buffer → Gin put
                tma_store_1d(send_buffer[src_rank_idx][token * kNumTopk + k], ...);
                tma_store_wait();
                gin.put<team_t>(recv_buffer[k][token], send_slot, kNumHiddenBytes, src_rank_idx);
            }
        }
    }
}
```

### 阶段 3: Topk Weights 写入

```cpp
// 在 master_token_buffer 中写入 topk weights（供后续 reduce epilogue 使用）
if (not kUseExpandedLayout and topk_weights != nullptr and lane_idx < kNumTopk) {
    master_token_buffer.get_topk_weights_ptr()[lane_idx] =
        __ldg(topk_weights + (i * kNumTopk + lane_idx));
}
```

### 阶段 4: RDMA 发送 + Barrier

```cpp
// 非 expand-send 且非 NVLink bypass 时，发起 RDMA put
if (not kDoExpandedSend and not nvlink_bypass and elect_one_sync()) {
    tma_store_wait();
    gin.put<team_t>(
        recv_buffer[rank_idx][src_token_idx].get_base_ptr(),  // 远程目标
        master_token_buffer.get_base_ptr(),                     // 本地源
        master_token_buffer.get_num_bytes<false>(),             // 大小
        src_rank_idx);                                          // 目标 rank
}

// 最终 barrier
comm::gpu_barrier<..., comm::kCombineTag1, true, true, false>(...);
```

## Combine Reduce Epilogue

[combine_reduce_epilogue.cuh](../../../refs/DeepEP/deep_ep/include/deep_ep/impls/combine_reduce_epilogue.cuh)

主 kernel 将数据写入 recv buffer 后，reduce epilogue kernel 负责归约回输出 tensor：

```cpp
// Recv buffer 中每 rank 预留了 kNumTopk 个 slot 组（对应不同的 topk 选择）
// 也可能使用 rank layout（按 rank 而非 topk 分组）
const auto recv_buffer = BufferLayout<false>(
    token_layout,  // 不含 SF 和 metadata
    kNumTokensInLayout, kNumMaxTokensPerRank, reduce_buffer);

extern __shared__ __align__(kNumTMAAlignBytes) int8_t smem[];
const auto tma_buffer = BufferLayout<true>(token_layout, kNumWarps, 1, smem)
    .get_rank_buffer(warp_idx).get_token_buffer(0);

// 遍历需要 combine 的 token
for (int i = global_warp_idx; i < num_combined_tokens; i += kNumWarps * kNumSMs) {
    // 读取该 token 的 topk 路由信息
    int src_rank = combined_topk_idx[i * kNumTopk + lane_idx];  // 每个 lane 读一个 topk 选择

    if (kUseExpandedLayout) {
        // Expand 模式：无需 local reduce，直接从对应 slot 拷贝
        // recv buffer 已存储了来自 remote rank 的 expand 数据
        tma_load_1d(tma_buffer, recv_buffer[slot_idx].get_base_ptr(), ...);
        tma_store_1d(combined_x + i * kNumHiddenBytes, tma_buffer, kNumHiddenBytes);
    } else {
        // 非 expand 模式：需要从 kNumTokInLayout 个 topk slot 中 reduce
        // 每个 topk 选择对应 recv buffer 中的一个 slot 组

        // 确定需要 reduce 的 slot
        int topk_slot_idx[kNumTopk];
        // Slot 索引来自 combined_topk_idx
        // ...

        // 从 recv buffer 的多个 slot 加载 + weighted sum + bias
        combine_reduce<kHiddenVec, kUnrollFactor, ...>(
            lane_idx, topk_slot_idx, tma_buffer,
            [=](int slot) { return recv_buffer[slot + offset]; },
            [=]() { tma_store_wait(); }
        );

        // 应用 bias
        if (bias_0 != nullptr)  combined_x[i] += bias_0[i];
        if (bias_1 != nullptr)  combined_x[i] += bias_1[i];
    }
}
```

recv buffer 的排布取决于 `kUseRankLayout` 和 `kUseExpandedLayout`：

- `kUseRankLayout = true`：按 `(src_rank, token)` 排布。同一 rank 的所有 topk 选择数据连续存储在 `recv_buffer[rank_idx][token * kTopk + topk_idx]`
- `kUseRankLayout = false`：按 `(topk_idx, token)` 排布。同一 topk 索引的所有数据连续存储
- `kUseExpandedLayout = true`：每个 expand slot 直接对应一个输出 token，不需要 reduce，直接拷贝

## Hybrid Combine

[hybrid_combine.cuh](../../../refs/DeepEP/deep_ep/include/deep_ep/impls/hybrid_combine.cuh)

Hybrid combine 有两类 warp，与 hybrid dispatch 形成对称：

### Scaleup Warps

```cpp
// 职责：从 scaleup buffer 读取数据，TMA store 直接写入远程 scaleup GPU 的 recv buffer
// 同时更新 channel scaleup tail（通知 forward warps）

if (warp_idx < kNumScaleupWarps) {
    const auto channel_idx = sm_idx * kNumChannelsPerSM + warp_idx;

    // 调整寄存器分配：scaleup warp 少用寄存器，forward warp 多用
    if constexpr (kAdjustRegisters) warpgroup_reg_dealloc<kNumRegistersForScaleupWarps>();

    // 遍历 channel_linked_list 中的 token
    while (true) {
        // 从 linked_list 读取下一批 token 索引
        // linked_list[channel][idx][scaleup_rank] = token 在 scaleup buffer 中的索引
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
            stored_token_idx[i] = __ldg(channel_linked_list +
                channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumScaleupRanks +
                stored_ll_idx[i] * kNumScaleupRanks + (i * 32 + lane_idx));
        }
        // 全部为 -1 表示已完成
        if (all(stored_token_idx < 0)) break;

        // Round-robin 消费所有 rank 的 token
        while (wip_mask) {
            // 选择下一个有数据的 rank
            dst_rank = ffs(wip_mask);

            // 获取该 token 的源 metadata
            src_global_token_idx = __ldg(src_metadata + token_idx * kMetadataStride);

            // 确定目标 buffer 地址
            auto token_buffer = scaleup_buffer
                .get_rank_buffer(...)
                .get_token_buffer(src_slot_idx);
            token_buffer.set_base_ptr(gin.get_sym_ptr<ncclTeamTagLsa>(
                token_buffer.get_base_ptr(), dst_scaleup_rank_idx));

            // 三种模式：no-reduce / reduce / expanded-send
            // （与 direct combine 相同）

            // TMA store 到远程 scaleup buffer
            tma_store_1d(token_buffer, tma_buffer, ...);
            tma_store_commit();

            // 更新发送计数
            stored_num_tokens_sent[dst_rank % 32] += 1;
        }

        // 通知 scaleout forward warps：更新 tail
        update_tails();  // st_release_sys 写入 remote tail
    }
    update_tails(/*finish=*/true);
}
```

### Forward Warps

```cpp
else {
    // 职责：重放 dispatch 的 forward metadata，从 scaleup buffer 拷贝数据到 scaleout buffer
    // 然后发起 RDMA put 或直接 NVLink 写

    const auto channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;

    // 调整寄存器分配：forward warp 需要更多寄存器做 reduce
    if constexpr (kAdjustRegisters) warpgroup_reg_alloc<kNumRegistersForForwardWarps>();

    // 重放 dispatch 的 metadata
    for (int i = 0; ; ++ i) {
        src_token_global_idx = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims);
        if (src_token_global_idx < 0) break;  // 结束标记

        // 读取该 token 在 dispatch 时的路由信息
        // token_metadata_at_forward[i][0] = src_token_global_idx
        // token_metadata_at_forward[i][1] = is_last_in_chunk
        // token_metadata_at_forward[i][2:2+kNumTopk] = scaleup_rank_indices
        // token_metadata_at_forward[i][2+kNumTopk:2+2*kNumTopk] = slot_indices

        // 等待 scaleup tail 到达（确保 scaleup warp 已完成写入）
        timeout_while(all tail >= expected) { ... }

        // 从 scaleup buffer 读取数据：
        if (kAllowMultipleReduction) {
            // Reduce 模式：从多个 scaleup rank 的数据做加权求和
            compute_topk_slots(topk_slot, reduce_valid_mask, ...);
            combine_reduce(...);
        }

        // 写入 scaleout send buffer 或直接写本地 recv buffer
        if (src_scaleout_rank_idx == scaleout_rank_idx) {
            // 本地 bypass
            tma_store_1d(recv_buffer, tma_buffer, ...);
        } else {
            // 先写 send buffer
            tma_store_1d(send_buffer, tma_buffer, ...);
            tma_store_wait();
            // 延迟 RDMA put（与下一轮 TMA store 重叠）
            // flush_last_tma_and_issue_rdma 在下一轮迭代前发起
        }
    }

    // 最后 flush 剩余的 RDMA 请求
    flush_last_tma_and_issue_rdma();

    // 清理 channel tails（供下一轮 dispatch 使用）
    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
        *workspace.get_channel_scaleup_tail_ptr(channel_idx, j) = 0;

    // 通知所有 scaleout peer 的 forward warp：本 channel 的数据已全部转发
    gin.red_add_rel<ncclTeamTagRail>(tail_ptr, pack2(1, 0), lane_idx);

    // 等待所有 scaleout peer 完成
    timeout_while(ld_acquire_sys(tail_ptr) != pack2(1, 0));
    *tail_ptr = 0;  // 清理
}
```

### Hybrid Reduce Epilogue

与 direct 模式相同：recv buffer 作为输入，按 topk 做 weighted reduce 或直接拷贝，加 bias，写出 combined_x。

## 关键启示

- **Combine 是 dispatch 的对称逆操作**，但增加了本地 reduce 逻辑。两者的 buffer 布局、warp 角色、barrier 模式都是一一对应的
- **三种 reduce 模式的选择**由编译期参数决定：`kUseExpandedLayout` 和 `kAllowMultipleReduction` 的组合覆盖了训练前向/反向、推理 prefill/decoding 的所有场景
- **TMA store 与 RDMA put 的重叠**是 hybrid combine 的关键优化。Forward warp 先 TMA store 到 send buffer，然后延迟到下一轮迭代再发起 `gin.put`，使 TMA 和 IBGDA 流水线化
- **寄存器压力管理**：hybrid combine 中 scaleup warp 显式 dealloc 寄存器（`warpgroup_reg_dealloc`），将空闲寄存器让给 forward warp。这是 Hopper 架构 `setmaxnreg` 的实际应用
- **Linked list + tail 的生产者-消费者模型**使 scaleup warps 和 forward warps 完全解耦：scaleup warp 按 linked list 顺序消费，forward warp 通过 tail 轮询消费，两者不需要全局 barrier
