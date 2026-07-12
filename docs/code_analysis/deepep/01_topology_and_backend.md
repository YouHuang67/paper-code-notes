---
tags:
  - CUDA
  - CUTLASS
---

# DeepEP：通信拓扑与 NCCL Gin 后端

本文详述 DeepEP 的通信拓扑抽象、NCCL Gin backend 的初始化流程、对称内存分配、QP 管理以及 Barrier 实现。

**源码**: [nccl.cu](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/backend/nccl.cu)、[comm.cuh](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/include/deep_ep/common/comm.cuh)、[layout.cuh](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/include/deep_ep/common/layout.cuh)

## 通信拓扑抽象

### 物理域与逻辑域

DeepEP 根据物理拓扑自动推导两个维度：

- **物理维度**：NVLink 连接数 = `ncclTeamLsa(comm).nRanks`（LSA = Local Shared Address，即 NVLink 可达的 GPU 集合）
- **RDMA 维度**：总 rank 数 / NVLink rank 数

由物理维度推导出逻辑域（`get_logical_domain_size`）：

```cpp
// nccl.cu
if (allow_hybrid_mode) {
    num_scaleout_ranks = num_rdma_ranks;  // 跨节点数
    num_scaleup_ranks = num_nvl_ranks;    // 每节点 GPU 数
} else {
    num_scaleout_ranks = 1;               // 无跨节点层级
    num_scaleup_ranks = num_ranks;        // 所有 rank 扁平为一个 scaleup 域
}
```

`is_scaleup_nvlink` 判断 scaleup 域是否全部通过 NVLink 可达：当 `num_scaleup_ranks == num_nvl_ranks` 时为 true，用 `ncclTeamTagLsa` 做 NVLink 通信；否则为 false，用 `ncclTeamTagWorld` 走 RDMA。

### Direct vs Hybrid 模式

- **Direct 模式**（`num_scaleout_ranks == 1`）：单级通信，所有 GPU 在一个扁平的 scaleup 域中
  - 纯 NVLink 环境：TMA store 直接跨 GPU 写入
  - 纯 RDMA 环境：TMA store 到本地 send buffer → Gin put 到远程
- **Hybrid 模式**（`num_scaleout_ranks > 1`）：两级层次化通信
  - Scaleout warps 负责跨节点 RDMA 收发
  - Forward warps 负责将数据从 scaleout 域转发到 scaleup 域
  - 两级 barrier：scaleup barrier（NVLink）和 scaleout barrier（Gin signal）

## NCCL Gin Backend 初始化

### NCCLSymmetricMemoryContext

`NCCLSymmetricMemoryContext`（[nccl.cu](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/backend/nccl.cu)）封装了整个通信上下文：

```
初始化流程：
1. ncclDevCommCreate(comm, &reqs, &dev_comm)
   - reqs.ginContextCount = num_allocated_qps  （每个 QP 对应一个 Gin context）
   - reqs.ginQueueDepth = 1024
   - reqs.ginTrafficClass = sl_idx             （RDMA Service Level，用于流量隔离）
   - reqs.ginSignalCount = num_ranks + 4       （barrier 信号量）
   - reqs.ginConnectionType = hybrid ? RAIL : FULL
2. 推导物理/逻辑拓扑（nvlink_ranks, rdma_ranks, scaleout/scaleup）
3. symmetric::alloc() 分配对称内存（所有 rank 等大、等偏移）
4. ncclCommWindowRegister() 注册 memory window
5. ncclGetLsaDevicePointer() 获取所有 NVLink peer 的 LSA 指针
```

LSA 地址映射是核心：`get_sym_ptr(ptr, dst_rank)` 通过偏移量计算远程指针：

```cpp
void* get_sym_ptr(void* ptr, const int& dst_rank_idx) const {
    const auto offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(mapped_window_ptr);
    return static_cast<uint8_t*>(nvl_window_ptrs[dst_rank_idx]) + offset;
}
```

只要双方注册了相同的 window 且 offset 一致，就能直接用 TMA store 写入对方地址。

### 对称内存布局

```
Buffer 内存布局（从低地址到高地址）：
┌──────────────┬─────────────────┬─────────────────┐
│  Workspace   │   GPU Buffer    │   CPU Buffer    │
│  (2MB对齐)   │  (dispatch/     │  (Engram 存储)  │
│              │   combine用)    │                 │
└──────────────┴─────────────────┴─────────────────┘
 sym.num_bytes = workspace + gpu_buffer + cpu_buffer
```

Workspace（`WorkspaceLayout`）的详细布局固定为：

```
  offset 0:  barrier counter (16 bytes)
  offset 16: notify reduction workspace [(kNumMaxRanks + kNumMaxExperts) * 8]
            scaleup rank count send [kNumMaxRanks * 8]
            scaleup rank count recv [kNumMaxRanks * 8]
            scaleup expert count send [kNumMaxExperts * 8]
            scaleup expert count recv [kNumMaxExperts * 8]
            scaleup atomic sender counter [kNumMaxRanks * 4]
            scaleout rank count send [kNumMaxRanks * 4]
            scaleout rank count recv [kNumMaxRanks * 4]
            scaleout expert count send [kNumMaxExperts * 4]
            scaleout expert count recv [kNumMaxExperts * 4]
            scaleout channel signaled tail [kNumMaxRanks * kNumMaxChannels * 8]
            channel scaleup tail [kNumMaxRanks * kNumMaxChannels * 4]
            PP counters [2 * 2 * 8]
            AGRS signals [(kNumMaxInflightAGRS + 1) * kNumMaxRanks * 4]
```

最大支持 EP1024 / 2048 专家的 workspace 预分配约 3.5 MB。

## QP（Queue Pair）分配策略

[comm.cuh](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/include/deep_ep/common/comm.cuh) 中 `get_qp_mode` 决定每个 warp 使用哪个 QP：

```cpp
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps>
__device__ auto get_qp_mode(sm_idx, channel_in_sm_idx, is_notify_warp) {
    // Notify warp 固定用 QP 0（独立，不与其他 channel 共享）
    if (is_notify_warp) return {0, CTA};

    constexpr int kQPStartIdx = kWithNotifyWarps ? 1 : 0;
    constexpr int kNumAvailableQPs = kNumQPs - kQPStartIdx;

    if constexpr (kNumSMs <= kNumAvailableQPs) {
        // 每 SM 独占一部分 QP，channel 在 SM 内轮转
        // 例：3 SMs, 10 QPs → SM0: {0,3,6,9}, SM1: {1,4,7}, SM2: {2,5,8}
        num_qps_in_sm = (kNumAvailableQPs / kNumSMs) + (sm_idx < (kNumAvailableQPs % kNumSMs));
        return {kQPStartIdx + sm_idx + (channel_in_sm_idx % num_qps_in_sm) * kNumSMs, CTA};
    } else {
        // QP 不够时，所有 SM 共享所有 QP
        global_channel_idx = sm_idx * kNumChannelsPerSM + channel_in_sm_idx;
        return {kQPStartIdx + (global_channel_idx % kNumAvailableQPs), GPU};
    }
}
```

核心原则：
- Notify warp 独占 QP 0（不与数据 channel 混用）
- 数据 channel：SM 数 ≤ QP 数时 CTA 独占（避免 flush 开销），否则 GPU 共享（所有 SM 竞争）

QP 数量由 `num_allocated_qps` 决定：
- Hybrid 模式：65（fast RDMA atomic）或 129（slow atomic），因为多个 channel × 多 rail 需要更多 QP
- Direct 模式：17

## Barrier 机制

DeepEP 有两级 barrier，根据拓扑自适应选择：

### Scaleup Barrier（NVLink）

```cpp
template <int kNumRanks, ...>
__device__ void nvlink_barrier_wo_local_sync(gin, workspace, rank_idx, sm_idx, thread_idx) {
    // 只在 SM 0 上执行
    if (sm_idx > 0) return;

    // 读取当前 phase（counter 低 2 位：phase=0/1, sign=+/−）
    int phase = (*counter) & 1, sign = (*counter >> 1);

    // 每个 thread 负责一个 rank：remote atomicAdd +1 或 −1
    if (thread_idx < kNumRanks)
        gin.get_sym_ptr<ncclTeamTagLsa>(signal_ptr[phase], thread_idx)
           .red_add_rel_sys(sign ? -1 : 1);

    // Thread 0 翻转 phase counter
    if (thread_idx == 0) atomicAdd(counter, 1);

    // 等待所有 rank 的信号到达 target（sign=0→kNumRanks, sign=1→0）
    timeout_while(signal != target);
}
```

关键设计：
- 用 1 个 64-bit counter 的两个 bit 做 phase 切换，支持 $2^{62}$ 次 barrier（约 $5\times10^{11}$ 年不溢出）
- 通过 `red_add_rel_sys` 保证 release 语义，`ld_acquire_sys` 保证 acquire 语义
- 仅 SM 0 参与，其他 SM 通过 `grid.sync()` 对齐

### Scaleout Barrier（Gin Signal）

```cpp
template <int kNumRanks, ..., typename team_t>
__device__ void gin_barrier_wo_local_sync(...) {
    // 1. Flush 所有 QP（确保之前的 RDMA 操作完成）
    for (int i = global_warp_idx; i < num_qps; i += kNumSMs * kNumWarps)
        ncclGin(dev_comm, i, CTA).flush(coopWarp);
    grid.sync();

    // 2. SM 0 执行 barrier
    if (sm_idx == 0) {
        // signal: 每个 rank 向所有其他 rank 发送递增信号
        for (i = thread_idx; i < kNumRanks; i += kNumThreads)
            gin.signal(team, i, ncclGin_SignalInc{rank_idx});

        // wait: 轮询 signal shadow counter，等待所有 rank 到达
        for (i = thread_idx; i < kNumRanks; i += kNumThreads) {
            target = ++(*shadow_ptr);
            timeout_while(ld_acquire_sys(signal_ptr) < target);
        }
    }
}
```

关键设计：
- Flush 所有 QP 保证 release 语义，这对 RDMA 的数据可见性至关重要
- `ncclGin_SignalInc` 是硬件加速的递增信号（比 atomicAdd 快得多）
- Shadow counter 在本地维护预期值，避免每次读远程 signal

### GPU Barrier 总控

```cpp
template <bool kIsScaleupNVLink, int kNumScaleoutRanks, int kNumScaleupRanks, ...>
__device__ void gpu_barrier(gin, workspace, ...) {
    // 1. Flush TMA stores（release 语义）
    tma_store_commit(); tma_store_wait();

    // 2. 全 grid 对齐
    grid.sync();

    // 3. 根据拓扑选择 barrier 类型
    if (do_scaleup && do_scaleout) {
        // 并行执行：SM 0 做 scaleup barrier，SM 1..N 做 scaleout barrier
        if (sm_idx == 0) scaleup_barrier(...);
        else scaleout_barrier(...);
    } else if (do_scaleup) {
        // 仅 scaleup：NVLink barrier 或 Gin world barrier
        scaleup_barrier(...);
    } else if (do_scaleout) {
        // 仅 scaleout：Gin rail barrier
        scaleout_barrier(...);
    }

    // 4. 全 grid 对齐
    grid.sync();
}
```

Hybrid 模式下 scaleup + scaleout barrier 并行执行是关键优化：SM 0 处理 NVLink barrier，其余 SM 处理 RDMA Gin barrier，两者不串行等待。

## Workspace 同步原语

### Notify 计数器

Dispatch 中 notify warps 统计 token 分布后，通过 workspace 的计数器通知其他 rank：

```
发送侧（notify warp, SM 0）:
  workspace.get_scaleup_rank_count_ptr<false>()[rank_idx] ← count
  gin.put_value<team_t>(dst_counter, count, dst_rank)

接收侧（notify warp, SM 0）:
  timeout_while(ld_volatile(dst_counter) 未就绪)
  读取 count → encode_decode_positive → rank_expert_count[smem]

编码方式（encode_decode_positive）：
  正数 → 原值     （低 31 位存值，第 32 位 = 0）
  待定 → INT_MAX  （第 32 位 = 1，表示"未就绪"）
  就绪判断：math::is_decoded_positive_ready(x) ≡ (x != INT_MAX)
```

这种编码方式避免额外的 flag 字段：一个 int 既存值又表示状态。

### Channel Tail 同步

Hybrid 模式中 scaleout warps 和 forward warps 之间通过 channel tail 做生产者-消费者同步：

```
Scaleout warp（生产者）:
  收到 token → 写入 scaleout_recv_buffer[slot_idx]
  update_scaleout_tail():
    gin.red_add_rel<ncclTeamTagRail>(tail_ptr[channel][scaleout_rank], signaled_tail, lane_idx)
    用 red_add 原子递增 tail 计数器（lane_idx = dst_rank）

Forward warp（消费者）:
  轮询 ld_acquire_sys(tail_ptr[channel][lane_idx])
  发现 tail > old_tail → TMA load 新到达的 token
  处理完后 old_tail = tail
```

tail 用 `math::pack2` 打包：高 32 位 = finish flag，低 32 位 = tail 位置。finish 时写入 `pack2(1, tail)`。

## 关键启示

- **NCCL Gin 的 signal 机制是高效 GPU 端 barrier 的基础**。`ncclGin_SignalInc` 比 atomicAdd 快得多，配合 shadow counter 消除远程读取
- **QP 与 SM 的映射策略直接影响性能**。SM 独占 QP 时用 CTA 共享模式免去跨 SM flush，SM 共享 QP 时用 GPU 共享模式保证公平
- **LSA pointer 的 offset 计算**让 TMA store 可以像访问本地内存一样访问远程 GPU，这是 NVLink 路径零拷贝的关键
- **Workspace 的固定布局设计**使 buffer 可以在 dispatch/combine/barrier 之间复用，所有计数器的位置在编译期确定
