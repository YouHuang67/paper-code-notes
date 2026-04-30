---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---

# C++ Binding 与 Kernel

**源码**:

- [data/csrc/batch_prefill_jit_binding.cu:L22-L50](src/batch_prefill_jit_binding_cu.md#__codelineno-0-22)
- [data/csrc/batch_prefill.cu:L47-L75](src/batch_prefill_cu.md#__codelineno-0-47)
- [data/csrc/batch_prefill.cu:L203-L334](src/batch_prefill_cu.md#__codelineno-0-203)
- [data/csrc/batch_prefill_customize_config.jinja](src/batch_prefill_customize_config_jinja.md)
- [data/csrc/batch_prefill_paged_kernel_inst.jinja](src/batch_prefill_paged_kernel_inst_jinja.md)
- [data/csrc/batch_prefill_ragged_kernel_inst.jinja](src/batch_prefill_ragged_kernel_inst_jinja.md)

到了这一层，variable block 语义已经基本消失。C++/CUDA 代码看到的输入是 `qo_indptr / paged_kv_indptr / paged_kv_indices / last_page_len`，也就是标准 paged prefill 认识的 metadata。

## binding 层：导出的只有 `plan / ragged_run / paged_run`

[`batch_prefill_jit_binding.cu:L22-L50`](src/batch_prefill_jit_binding_cu.md#__codelineno-0-22) 非常短，但它把 Python 侧协议完全钉死了：

```cpp
Array<int64_t> BatchPrefillWithKVCachePlan(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView qo_indptr, TensorView kv_indptr,
    TensorView kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, int64_t window_left, int64_t fixed_split_size,
    bool disable_split_kv, int64_t num_colocated_ctas);

void BatchPrefillWithRaggedKVCacheRun(...);
void BatchPrefillWithPagedKVCacheRun(...);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, BatchPrefillWithKVCachePlan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged_run, BatchPrefillWithRaggedKVCacheRun);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run, BatchPrefillWithPagedKVCacheRun);
```

对 variable block 路径来说，真正会走的是：

- `plan`
- `paged_run`

`ragged_run` 只是跟随 batch prefill 通用基础设施一起保留下来。

## `plan()`：把调度结果编码成 `Array<int64_t>`

Python 里 `self._cached_module.plan(...)` 的真正落点是 [`batch_prefill.cu:L47-L75`](src/batch_prefill_cu.md#__codelineno-0-47)：

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

这一步最重要的不是 `PrefillPlan` 的模板细节，而是输入输出语义：

- 输入是 workspace、`qo_indptr`、`kv_indptr`、row/head/page 等调度参数
- 输出不是 C++ 对象，而是 `plan_info.ToVector()` 之后的 `Array<int64_t>`

所以 Python 侧保存的 `self._plan_info` 本质上是“序列化后的调度结果”，不是某个可直接操作的高层对象。

## `paged_run()` 第一段：恢复 `plan_info`，构造 `paged_kv_t`

运行阶段的主线从 [`batch_prefill.cu:L203-L267`](src/batch_prefill_cu.md#__codelineno-0-203) 开始：

```cpp
void BatchPrefillWithPagedKVCacheRun(... ) {
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

  DISPATCH_context(..., [&] {
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
```

这段代码把前面几篇文档的两条主线接上了：

- `plan_info.FromVector(...)` 对应 Python `plan()` 阶段缓存下来的序列化调度信息
- `paged_kv_t(...)` 对应 Python metadata 层构造的 `paged_kv_indptr / paged_kv_indices / last_page_len`

从这里开始，variable block 已经完全不再以“块”的形式出现了。底层只看到一个普通 paged KV 结构。

## `paged_run()` 第二段：从 workspace 偏移恢复调度数组

真正体现 `plan()` 价值的是 [`batch_prefill.cu:L295-L318`](src/batch_prefill_cu.md#__codelineno-0-295)：

```cpp
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
  if (plan_info.enable_cuda_graph) {
    params.block_valid_mask =
        GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
  }
}
params.padded_batch_size = plan_info.padded_batch_size;
params.max_total_num_rows = plan_info.total_num_rows;
if (plan_info.enable_cuda_graph) {
  params.total_num_rows =
      GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
}
```

这段代码很关键，因为它说明：

- `plan()` 并不是只算出几个标量参数
- 它已经把运行期真正需要的 request/tile/merge 索引组织进 workspace
- `paged_run()` 只是根据 `plan_info` 记录的 offset 再把这些数组挂回 `PagedParams`

如果 `split_kv` 开启，还会同时恢复：

- `merge_indptr`
- `tmp_v`
- `tmp_s`

这也是为什么 wrapper 构造函数一开始就必须持有 float/int 两类 workspace。

## 最终 dispatch：编译期模板实例 + 运行期 `cta_tile_q`

最后的落点很短，但非常典型，见 [`batch_prefill.cu:L321-L328`](src/batch_prefill_cu.md#__codelineno-0-321)：

```cpp
cudaError_t status = cudaSuccess;

DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
  status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
      CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
      PagedParams>(params, tmp_v, tmp_s, enable_pdl, stream);
});
```

这就是典型的“编译期模板实例化 + 运行期选择”的两阶段 dispatch：

- `HEAD_DIM_QK`、`MASK_MODE`、`POS_ENCODING_MODE`、`AttentionVariant` 等由 JIT 规格决定，编译期固定
- `cta_tile_q` 由 `plan()` 输出，运行期再做一层选择

要理解这种模式，可以一起看：

- [CUDA 基础：CUTLASS/CuTe 编程模型](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- [Flash Attention V2：调度与实例化](../flash_attention_v2/05_dispatch_and_instantiation.md)

## Jinja 模板：不是业务逻辑，而是实例化骨架

这一层还会看到三份模板：

- [batch_prefill_customize_config.jinja](src/batch_prefill_customize_config_jinja.md)
- [batch_prefill_paged_kernel_inst.jinja](src/batch_prefill_paged_kernel_inst_jinja.md)
- [batch_prefill_ragged_kernel_inst.jinja](src/batch_prefill_ragged_kernel_inst_jinja.md)

它们的作用不是承载算法逻辑，而是：

- 把 JIT 规格写进 `batch_prefill_config.inc`
- 为不同 `mask_mode` 生成实例化单元
- 让编译器只为当前规格产出必要符号

这也是上一节里 URI 和 `gen_customize_batch_prefill_module()` 必须存在的原因。

## 这一层的结论

把 binding 和 kernel 主线串起来，可以得到一个非常明确的工程判断：

1. Python `plan()` 先把 variable block 语义翻译成 paged prefill metadata
2. C++ `plan` 再把调度索引编码进 `PrefillPlanInfo` 和 workspace
3. `paged_run()` 恢复 `plan_info`、拼装 `PagedParams`
4. 最终进入标准 FA2 paged prefill kernel

因此，底层并不存在“专门为 variable block 写的新 kernel”。真正的新东西只在 metadata 翻译层；C++/CUDA 层执行的是成熟的 batch prefill 调度与 kernel 体系。
