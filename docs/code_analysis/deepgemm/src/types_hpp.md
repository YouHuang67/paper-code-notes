---
tags:
  - CUDA
  - CUTLASS
  - LLM Inference
---
# types.hpp

**原始文件**: `deep_gemm/include/deep_gemm/common/types.hpp`

**仓库**: [deep_gemm/include/deep_gemm/common/types.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/types.hpp)

```cpp linenums="1"
#pragma once

namespace deep_gemm {

enum class MmaKind {
    BF16        = 0,
    MXFP8FP4    = 1,
};

constexpr __host__ __device__ int get_element_size(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return 2;
        case MmaKind::MXFP8FP4: return 1;
        default: return 0;
    }
}

enum class GemmType {
    Normal                              = 0,
    MGroupedContiguous                  = 1,
    MGroupedMasked                      = 2,
    KGroupedContiguous                  = 3,
    Batched                             = 4,
    MGroupedContiguousWithPsumLayout    = 5,
};

constexpr __host__ __device__ bool is_m_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::MGroupedContiguous:                  return true;
        case GemmType::MGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

enum class KernelType {
    Kernel1D1D = 0,
    Kernel1D2D = 1,
    KernelNoSF = 2
};

} // namespace deep_gemm
```
