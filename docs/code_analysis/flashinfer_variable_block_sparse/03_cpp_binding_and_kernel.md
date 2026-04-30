---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---

# C++ Binding 与 Kernel

**核心文件**:

- [`data/csrc/batch_prefill_jit_binding.cu`](src/batch_prefill_jit_binding_cu.md)
- [`data/csrc/batch_prefill.cu`](src/batch_prefill_cu.md)
- [`data/csrc/batch_prefill_customize_config.jinja`](src/batch_prefill_customize_config_jinja.md)
- [`data/csrc/batch_prefill_paged_kernel_inst.jinja`](src/batch_prefill_paged_kernel_inst_jinja.md)
- [`data/csrc/batch_prefill_ragged_kernel_inst.jinja`](src/batch_prefill_ragged_kernel_inst_jinja.md)
- [`data/csrc/tvm_ffi_utils.h`](src/tvm_ffi_utils_h.md)

## 先看 binding 层：导出的其实只有三个符号

`batch_prefill_jit_binding.cu` 本身非常短：

```cpp
Array<int64_t> BatchPrefillWithKVCachePlan(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView qo_indptr, TensorView kv_indptr,
    TensorView kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, int64_t window_left, int64_t fixed_split_size,
    bool disable_split_kv, int64_t num_colocated_ctas);

void BatchPrefillWithRaggedKVCacheRun(TensorView float_workspace_buffer,
                                      TensorView int_workspace_buffer, Array<int64_t> plan_info_vec,
                                      TensorView q, TensorView k, TensorView v,
                                      TensorView qo_indptr, TensorView kv_indptr, TensorView o,
                                      Optional<TensorView> maybe_lse, int64_t mask_mode_code,
                                      int64_t layout, int64_t window_left,
                                      bool enable_pdl ADDITIONAL_FUNC_PARAMS);

void BatchPrefillWithPagedKVCacheRun(TensorView float_workspace_buffer,
                                     TensorView int_workspace_buffer, Array<int64_t> plan_info_vec,
                                     TensorView q, TensorView paged_k_cache,
                                     TensorView paged_v_cache, TensorView qo_indptr,
                                     TensorView paged_kv_indptr, TensorView paged_kv_indices,
                                     TensorView paged_kv_last_page_len, TensorView o,
                                     Optional<TensorView> maybe_lse, int64_t mask_mode_code,
                                     int64_t layout, int64_t window_left,
                                     bool enable_pdl ADDITIONAL_FUNC_PARAMS);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, BatchPrefillWithKVCachePlan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged_run, BatchPrefillWithRaggedKVCacheRun);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run, BatchPrefillWithPagedKVCacheRun);
```

这层做的事情极少，但很重要：

- 把 C++ 函数签名和 Python 侧调用协议固定下来
- 用 TVM FFI 宏导出 `plan / ragged_run / paged_run`
- 让 JIT 编译出来的模块在 Python 侧可直接拿到 `.plan` 与 `.paged_run`

对于 variable block 路径，真正会走的是：

- `plan`
- `paged_run`

`ragged_run` 只是跟随 batch prefill 基础设施一并保留。

## `plan()` 在 C++ 侧到底干了什么

`batch_prefill.cu` 里的 `BatchPrefillWithKVCachePlan` 是 Python `module.plan(...)` 的真正落点：

```cpp
Array<int64_t> BatchPrefillWithKVCachePlan(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView qo_indptr, TensorView kv_indptr,
    TensorView kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, int64_t window_left, int64_t fixed_split_size,
    bool disable_split_kv, int64_t num_colocated_ctas = 0) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * get_element_size(int_workspace_buffer);

  PrefillPlanInfo plan_info;

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());
  cudaError_t status = PrefillPlan<IdType>(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(qo_indptr.data_ptr()),
      static_cast<IdType*>(kv_indptr.data_ptr()), total_num_rows, batch_size, num_qo_heads,
      num_kv_heads, head_dim_qk, head_dim_vo, page_size, enable_cuda_graph,
      /*sizeof_dtype_o=*/2, window_left, fixed_split_size, disable_split_kv, num_colocated_ctas,
      stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Failed to plan prefill with error: " << cudaGetErrorString(status);

  return Array(plan_info.ToVector());
}
```

这一步最该注意的不是模板细节，而是它的输入和输出语义：

- 输入：
  - workspace
  - `qo_indptr`
  - `kv_indptr`
  - 行长度、head dim、page size 等规划参数
- 输出：
  - `PrefillPlanInfo` 被压平成 `Array<int64_t>`

也就是说，Python 侧拿到的 `self._plan_info` 并不是抽象对象，而是一串编码后的调度结果。之后 `paged_run()` 会再次把它恢复回来。

## `paged_run()` 的核心：恢复 plan_info，装配 PagedParams，再 dispatch

最值得看的部分是 `BatchPrefillWithPagedKVCacheRun`：

```cpp
void BatchPrefillWithPagedKVCacheRun(TensorView float_workspace_buffer,
                                     TensorView int_workspace_buffer, Array<int64_t> plan_info_vec,
                                     TensorView q, TensorView paged_k_cache,
                                     TensorView paged_v_cache, TensorView qo_indptr,
                                     TensorView paged_kv_indptr, TensorView paged_kv_indices,
                                     TensorView paged_kv_last_page_len, TensorView o,
                                     Optional<TensorView> maybe_lse, int64_t mask_mode_code,
                                     int64_t layout, int64_t window_left,
                                     bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  int64_t batch_size = paged_kv_indptr.size(0) - 1;
  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;
  uint32_t head_dim_qk = q.size(2);
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  ...

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        PagedParams params;

        params.q = static_cast<DTypeQ*>(q.data_ptr());
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
            static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
            static_cast<IdType*>(paged_kv_indices.data_ptr()),
            static_cast<IdType*>(paged_kv_indptr.data_ptr()),
            static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
        params.paged_kv = paged_kv;
        params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());

        params.lse = maybe_lse ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
        params.num_qo_heads = num_qo_heads;
        params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.window_left = window_left;

        ...

        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          params.merge_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
          ...
        }

        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
          status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
              CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
              PagedParams>(params, tmp_v, tmp_s, enable_pdl, stream);
        });
      });
}
```

这段代码非常值得慢读，因为它把整套闭环串起来了。

### 第一步：把 Python 侧的 `plan_info_vec` 恢复成结构体

```cpp
PrefillPlanInfo plan_info;
plan_info.FromVector(...)
```

这说明 Python 侧保存的 `self._plan_info` 只是序列化后的计划结果；真正运行前要恢复为结构化调度信息。

### 第二步：构造 `paged_kv_t`

```cpp
paged_kv_t<DTypeKV, IdType> paged_kv(
    num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
    ...,
    static_cast<IdType*>(paged_kv_indices.data_ptr()),
    static_cast<IdType*>(paged_kv_indptr.data_ptr()),
    static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
```

这一步最关键：

- `paged_kv_indices`
- `paged_kv_indptr`
- `paged_kv_last_page_len`

正是前面 Python metadata 翻译层的最终产物。

也就是说，variable block 语义在这里第一次完全“消失”了，底层只看到一个普通 paged KV 结构。

### 第三步：从 workspace 偏移恢复 plan 阶段产出的调度数组

```cpp
params.request_indices = GetPtrFromBaseOffset(...)
params.qo_tile_indices = GetPtrFromBaseOffset(...)
params.kv_tile_indices = GetPtrFromBaseOffset(...)
params.o_indptr = GetPtrFromBaseOffset(...)
params.kv_chunk_size_ptr = GetPtrFromBaseOffset(...)
```

这表示 `plan()` 做的重活，并不是某种“逻辑计划”的抽象描述，而是已经把运行需要的 request/tile/merge 索引全部算好并布局进 workspace 了。

运行时只是：

- 按偏移取出这些数组
- 挂到 `PagedParams`
- 交给 kernel

### 第四步：如果 split-kv 被启用，还要恢复 merge 所需的中间缓冲

```cpp
if (plan_info.split_kv) {
  params.merge_indptr = ...
  tmp_v = ...
  tmp_s = ...
}
```

这说明 float workspace 的存在并不是装饰，它要给 split-k 的中间结果与 merge 服务。

### 第五步：按编译期实例和运行时 `cta_tile_q` 做最终 dispatch

```cpp
DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
  status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<...>(...);
});
```

这里的 dispatch 是典型的“编译期模板实例 + 运行期选择”模式。要理解它，可以配合：

- [CUDA 基础：CUTLASS/CuTe 编程模型](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- [Flash Attention V2：Dispatch 与实例化](../flash_attention_v2/05_dispatch_and_instantiation.md)

一起看。

## Jinja 模板的作用：不是写逻辑，而是铺实例化矩阵

`batch_prefill_customize_config.jinja`、`batch_prefill_paged_kernel_inst.jinja`、`batch_prefill_ragged_kernel_inst.jinja` 这几份模板不要读成“业务逻辑文件”，它们是实例化骨架。

例如 paged kernel 模板非常短：

```cpp
#include "batch_prefill_config.inc"

namespace flashinfer {

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<
    CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
    USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant, Params>(
    Params params, typename Params::DTypeO* tmp_v, float* tmp_s, bool enable_pdl,
    cudaStream_t stream);

}
```

它真正的意义是：

- 把配置文件里的宏和类型别名代入
- 形成当前规格下的模板实例化单元
- 让编译器只为需要的那一组配置产出符号

这也是为什么 JIT 层要先写 `batch_prefill_config.inc`，再生成这些 `.cu`。

## 这一层的最终结论

把 [`batch_prefill_jit_binding.cu`](src/batch_prefill_jit_binding_cu.md) 和 [`batch_prefill.cu`](src/batch_prefill_cu.md) 串起来看，可以得到一个很清晰的判断：

1. Python `plan()` 的产物是序列化后的 `plan_info` 与 GPU/CPU 元数据
2. C++ `plan` 会把调度细节计算完并编码回 `plan_info`
3. `paged_run` 在运行时恢复调度信息、拼装 `PagedParams`
4. kernel 看到的只是标准 paged prefill 输入

因此，底层并不存在一个“专门为 variable block 写的新 kernel”。真正发生的事情是：

> variable block 语义在 Python metadata 层被翻译完毕，C++/CUDA 层执行的是成熟的 FA2 paged prefill 内核与调度体系。

