---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---

# run 阶段：进入 C++ 与 CUDA

**源码**:

- [data/csrc/batch_prefill_jit_binding.cu:L22-L50](src/batch_prefill_jit_binding_cu.md#__codelineno-0-22)
- [data/csrc/batch_prefill.cu:L47-L75](src/batch_prefill_cu.md#__codelineno-0-47)
- [data/csrc/batch_prefill.cu:L203-L334](src/batch_prefill_cu.md#__codelineno-0-203)
- [data/csrc/batch_prefill_customize_config.jinja](src/batch_prefill_customize_config_jinja.md)
- [data/csrc/batch_prefill_paged_kernel_inst.jinja](src/batch_prefill_paged_kernel_inst_jinja.md)
- [data/csrc/batch_prefill_ragged_kernel_inst.jinja](src/batch_prefill_ragged_kernel_inst_jinja.md)

这一章只讲真正执行 attention 的后半段：

1. Python `run()` 怎样重排 q/k/v
2. `module.paged_run(...)` 怎样落到 C++
3. C++ 怎样恢复 `plan_info`、拼装 `PagedParams`
4. 最终怎样 dispatch 到 FA2 paged prefill kernel

也就是说，前一章负责“准备怎么跑”，这一章负责“真正开始跑”。

到了这一层，variable block 语义已经基本消失。C++/CUDA 代码看到的输入是 `qo_indptr / paged_kv_indptr / paged_kv_indices / last_page_len`，也就是标准 paged prefill 认识的 metadata。

## `run()`：先把张量改成底层期望布局

Python 侧真正的执行入口在 [`wrapper.py:L284-L394`](src/wrapper_py.md#__codelineno-0-284)：

```python
if enable_pdl is None:
    enable_pdl = device_support_pdl(q.device)

if logits_soft_cap is None:
    logits_soft_cap = 0.0
if sm_scale is None:
    sm_scale = 1.0 / math.sqrt(q.size(-1))
if rope_scale is None:
    rope_scale = 1.0
if rope_theta is None:
    rope_theta = 1e4

'''
Q 改成 batch prefill 期望的 grouped-query 视图
K/V 改成 page size = 1 的 paged cache 视图
'''
q = einops.rearrange(
    q,
    "(num_kv_heads gqa_group_size) qo_len head_dim -> (num_kv_heads qo_len) gqa_group_size head_dim",
    num_kv_heads=self._num_kv_heads,
).contiguous()
k = einops.rearrange(
    k,
    "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
).contiguous()
v = einops.rearrange(
    v,
    "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
).contiguous()

'''
把 plan() 阶段缓存的 metadata 与 plan_info 全部交给 paged_run
'''
self._cached_module.paged_run(
    self._float_workspace_buffer,
    self._int_workspace_buffer,
    self._plan_info,
    q,
    k,
    v,
    self._qo_indptr,
    self._paged_kv_indptr_buf,
    self._paged_kv_indices_buf,
    self._paged_kv_last_page_len,
    out,
    lse,
    self._mask_mode,
    TensorLayout[self._kv_layout].value,
    -1,
    enable_pdl,
    ...,
)
```

这一层的职责很明确：

- `q` 被重排成 grouped-query 视图
- `k/v` 被重排成 `page_size=1` 的 paged cache 视图
- 前两章准备好的 metadata 和 `plan_info` 被一次性交给 `paged_run`

因此 `run()` 本身不是主要算法阶段，它更像“执行前的最后一层 Python 适配”。

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

按执行顺序看，这一章做了四件事：

1. Python `run()` 先把 q/k/v 重排成 paged prefill 期望布局。
2. `paged_run` 进入 C++，恢复前一章产出的 `plan_info`。
3. C++ 侧拼装 `PagedParams`，从 workspace 偏移恢复调度数组。
4. 最终 dispatch 到标准 FA2 paged prefill kernel。

因此，真正执行阶段并没有额外的 variable block 专属 kernel；这一层做的仍然是成熟的 paged prefill 执行路径。
