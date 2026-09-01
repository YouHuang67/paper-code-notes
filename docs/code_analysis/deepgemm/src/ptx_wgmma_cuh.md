---
tags:
  - CUDA
  - CUTLASS
  - LLM Inference
---
# ptx/wgmma.cuh

**原始文件**: `deep_gemm/include/deep_gemm/ptx/wgmma.cuh`

**仓库**: [deep_gemm/include/deep_gemm/ptx/wgmma.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/ptx/wgmma.cuh)

```cpp linenums="1"
#pragma once

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::ptx {

CUTLASS_DEVICE void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

CUTLASS_DEVICE void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

CUTLASS_DEVICE void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

template <int N>
CUTLASS_DEVICE void warpgroup_wait() {
    DG_STATIC_ASSERT(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

} // namespace deep_gemm::ptx
```
