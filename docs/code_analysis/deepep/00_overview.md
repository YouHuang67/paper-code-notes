---
tags:
  - CUDA
  - CUTLASS
---

# DeepEP 代码分析：总览

**源码仓库**: [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)

**团队**: DeepSeek（Chenggang Zhao, Shangyan Zhou, Liyue Zhang 等）

**分析范围**: V2 版本的 elastic EP 实现，聚焦 `csrc/kernels/elastic/` 和 `deep_ep/include/deep_ep/impls/`

## 核心思想

DeepEP 是面向 MoE 的 expert-parallel 通信库。核心任务是完成 **all-to-all 通信**：

- **Dispatch**：将每个 token 按 top-k 门控结果路由到对应专家所在的 GPU
- **Combine**：将各 GPU 上专家计算后的结果按 token 归约回原始 GPU

V2 版本通过三个关键设计实现极致性能：

- **TMA（Tensor Memory Accelerator）**：SM90 硬件异步拷贝引擎，kernel 直接发起跨 GPU 的 TMA store，无需 CPU 参与
- **NCCL Gin（GPUDirect Async Kernel Initiated）**：kernel 内直接发起 RDMA put/get/signal，配合 flush 和 barrier 实现零 CPU 通信
- **Warp 级 Channel 并行**：每个 warp 作为一个独立 channel 处理 token，映射到独立 QP（Queue Pair），所有 warp/SM 并发通信

与 V1 的核心区别：V2 用 NCCL Gin 替代 NVSHMEM，Dispatch/Combine 统一为 ElasticBuffer 接口，SM 用量从 24 降到 4-6。

## 通信拓扑

```
EP 8x2 拓扑（8 个 scaleup GPU × 2 个 scaleout 节点）：

  Node 0 (RDMA rank 0)              Node 1 (RDMA rank 1)
  ┌─────────────────────┐           ┌─────────────────────┐
  │ GPU0 GPU1 ... GPU7  │  RDMA    │ GPU8 GPU9 ... GPU15 │
  │  ├─────NVLink────┤  │<═══════>│  ├─────NVLink────┤  │
  │  scaleup rank 0..7  │           │  scaleup rank 0..7  │
  └─────────────────────┘           └─────────────────────┘
       scaleout rank 0                   scaleout rank 1

- Scaleup 域：NVLink 连接的同节点 GPU（LSA team）
- Scaleout 域：RDMA 连接的跨节点 GPU（Rail team）
- Hybrid 模式：两级层次化通信
- Direct 模式：单级 flat 通信（纯 NVLink 或纯 RDMA）
```

## 代码架构

```
deep_ep/
├── __init__.py                    # Python 入口：NCCL 检查、JIT 初始化
├── buffers/
│   ├── elastic.py                 # ElasticBuffer (V2)：dispatch/combine/engram/PP/AGRS
│   └── legacy.py                  # Buffer (V1)：NVSHMEM 后端
├── include/deep_ep/
│   ├── common/                    # 公共工具（device/host 共用）
│   │   ├── comm.cuh               # Barrier、QP 分配、超时等待
│   │   ├── layout.cuh             # WorkspaceLayout、TokenLayout、BufferLayout
│   │   ├── ptx.cuh                # PTX 指令封装：TMA、mbarrier、cp.async、shfl
│   │   ├── math.cuh               # 数学工具：align、ceil_div、prefix sum
│   │   ├── handle.cuh             # NCCL Gin handle 封装
│   │   └── compiled.cuh           # 编译期宏：数据类型、SM90 特性开关
│   └── impls/                     # GPU kernel 实现（JIT 编译的模板）
│       ├── dispatch.cuh           # Direct dispatch（单级 NVLink）
│       ├── hybrid_dispatch.cuh    # Hybrid dispatch（两级 RDMA+NVLink）
│       ├── dispatch_copy_epilogue.cuh  # Dispatch 后的 buffer→tensor 拷贝
│       ├── combine.cuh            # Direct combine（单级 NVLink）
│       ├── hybrid_combine.cuh     # Hybrid combine（两级 RDMA+NVLink）
│       ├── combine_reduce_epilogue.cuh # Combine 后的 reduce + bias
│       ├── combine_utils.cuh      # Combine 共用工具：topk slot 排布、reduce
│       ├── barrier.cuh            # GPU barrier 实现
│       ├── engram_fetch.cuh       # Engram 远程 KV cache fetch
│       └── pp_send_recv.cuh       # Pipeline parallel send/recv
└── csrc/
    ├── python_api.cpp             # PyBind11：注册 Buffer/ElasticBuffer/JIT API
    ├── elastic/
    │   ├── buffer.hpp             # ElasticBuffer C++ 实现（1900+ 行）
    │   └── utils.hpp              # Elastic 工具：stream wait、shape 提取
    ├── kernels/
    │   ├── elastic/
    │   │   ├── api.hpp            # 汇总 include
    │   │   ├── dispatch.hpp       # Dispatch launcher：JIT 编译 + launch
    │   │   ├── combine.hpp        # Combine launcher：JIT 编译 + launch
    │   │   ├── barrier.hpp        # Barrier launcher
    │   │   ├── engram.hpp         # Engram fetch launcher
    │   │   └── pp_send_recv.hpp   # PP send/recv launcher
    │   └── backend/
    │       ├── nccl.cu            # NCCL backend：symmetric memory、window、device comm
    │       ├── nvshmem.cu         # NVSHMEM backend（V1 legacy）
    │       └── cuda_driver.cu     # CUDA driver：batched write+wait
    └── jit/                       # JIT 编译系统
        ├── api.hpp                # 注册 Python API
        ├── compiler.hpp           # NVRTC 编译 + 缓存
        ├── launch_runtime.hpp     # Kernel launch 封装
        └── device_runtime.hpp     # 获取 GPU 属性（SM 数、smem 大小、clock rate）
```

### 关键文件依赖

```
buffer.hpp ──┬── nccl.cu (NCCLSymmetricMemoryContext)
             ├── dispatch.hpp ── dispatch.cuh / hybrid_dispatch.cuh / dispatch_copy_epilogue.cuh
             ├── combine.hpp  ── combine.cuh  / hybrid_combine.cuh  / combine_reduce_epilogue.cuh
             ├── barrier.hpp  ── barrier.cuh
             └── jit/compiler.hpp

dispatch.cuh ──┬── comm.cuh (barrier, QP mode)
               ├── layout.cuh (workspace, token, buffer)
               ├── ptx.cuh (TMA, mbarrier, cp.async, shfl)
               └── handle.cuh (NCCL Gin handle)
```

## 通信流程总览

### Dispatch（Token → Expert）

```
对于每个 rank 的本地 tokens：
  1. Notify Warps（4 warps）：
     - 遍历 tokens，atomicAdd 统计每个 rank/expert 应接收的 token 数量
     - 跨 SM reduce → 写入 workspace 通知远程 GPU
     - 等待所有远程 GPU 的通知到达 → 计算 prefix sum
  2. Dispatch Warps（每个 warp 一个 channel）：
     - TMA load token → shared memory TMA buffer
     - cp.async 加载 scale factor（FP8 模式）
     - 加载 topk_idx / topk_weights
     - 去重 → atomicAdd 分配目标 slot
     - NVLink 路径：TMA store 直接写远程 GPU recv buffer
     - RDMA 路径：TMA store 到本地 send buffer → RDMA put 到远程
  3. GPU Barrier：确保所有远程写入完成
  4. Copy Epilogue：
     - PDL（Programmatic Dependent Launch）等待主 kernel 完成
     - 遍历 recv buffer，TMA load → TMA store 到输出 tensor
     - 记录路由元数据（recv_src_metadata、channel_linked_list）
```

### Combine（Expert → Token）

```
对于每个本地 rank 已计算的 expert 输出：
  1. GPU Barrier：确保 dispatch 的远程 buffer 已释放
  2. Combine Warps：
     - 从 src_metadata 读取每个 token 的源 rank 和 topk 信息
     - TMA load token 到 shared memory TMA buffer
     - 可选本地 reduce（expand 模式 + multiple_reduction）
     - NVLink 路径：TMA store 直接写远程 GPU recv buffer
     - RDMA 路径：TMA store 到本地 send buffer → RDMA put 到远程
  3. GPU Barrier：确保所有远程写入完成
  4. Reduce Epilogue：
     - 从 recv buffer TMA load 数据
     - 按 topk 做 weighted reduce / 直接拷贝 + bias
     - 写出 combined_x
```

## 文档导航

| 文档 | 内容 |
|------|------|
| [01 通信拓扑与 NCCL Gin 后端](01_topology_and_backend.md) | Scaleup/Scaleout 域、Direct/Hybrid 模式、对称内存、Workspace 布局、QP 分配、Barrier 机制 |
| [02 Dispatch 内核](02_dispatch.md) | Notify Warps 计数与通知、Dispatch Warps 数据搬运、RDMA/NVLink 双路径、Copy Epilogue、Hybrid Dispatch 两级转发 |
| [03 Combine 内核](03_combine.md) | Combine Warps 数据写入、本地 Reduce、RDMA 归约、Reduce Epilogue、Hybrid Combine 流水线、TMA 与 IBGDA 重叠 |

## 关键启示

- **TMA + NCCL Gin 是 Hopper 架构上全 GPU 端 all-to-all 的核心组合**。TMA 提供异步硬件拷贝引擎，Gin 让 kernel 内直接发起 RDMA，CPU 完全不参与通信路径
- **Warp 即 Channel + QP 映射**是充分利用多 QP 带宽的关键。每个 warp 独立处理 token、独占 QP，所有 SM 的所有 warp 并发通信
- **Hybrid 两级通信**将 token 从 scaleout 域（RDMA）转发到 scaleup 域（NVLink），通过 workspace 中的 tail 指针做生产者-消费者同步，用 TMA 做跨层级的数据搬运
- **JIT 编译**按需实例化：hidden 大小、topk 数、SM 数、QP 数全是模板参数，运行时 NVRTC 编译后缓存，避免组合爆炸
- **Buffer 内存复用的关键**：dispatch/combine 共享同一个 buffer，workspace 中所有计数器在每次调用后清零，不依赖动态分配
