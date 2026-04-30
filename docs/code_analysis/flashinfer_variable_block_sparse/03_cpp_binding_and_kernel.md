---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---

# run 阶段：进入 C++ 与 CUDA

**源码**:

- [wrapper.py:L284-L395](src/wrapper_py.md#__codelineno-0-284)
- [data/csrc/batch_prefill_jit_binding.cu:L22-L50](src/batch_prefill_jit_binding_cu.md#__codelineno-0-22)
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

## 1. 这一章里各个张量的 shape

进入 `run()` 执行阶段时，最关键的几个张量 shape 如下：

- 外部 `q`: `[H_qo, qo_len, D]`
- 外部 `k/v`: `[H_kv, kv_len, D]`
- `run()` 重排后的 `q`: `[H_kv * qo_len, G, D]`
- `run()` 重排后的 `k/v`: `[H_kv * kv_len, 1, 1, D]`
- `self._qo_indptr`: `[H_kv * R + 1]`
- `self._paged_kv_indptr_buf`: `[H_kv * R + 1]`
- `self._paged_kv_indices_buf`: `[nnz_tokens]`
- `self._paged_kv_last_page_len`: `[H_kv * R]`

这里最关键的视角切换是：

- 对 Python 外部调用者，`q/k/v` 还是普通的 `[heads, seqlen, dim]`
- 对底层 paged prefill runtime，`q/k/v` 已经被改写成“逻辑 request + page”的布局

并且因为当前本地实现固定采用：

```text
page_size = 1
num_kv_heads_per_logical_request = 1
```

所以底层 C++/CUDA 看到的是非常标准的 paged prefill 输入形态。

## 2. `run()`：先把 q/k/v 改成底层期望布局

Python 侧真正的执行入口在 [`wrapper.py:L284-L395`](src/wrapper_py.md#__codelineno-0-284)：

```python
if enable_pdl is None:
    enable_pdl = device_support_pdl(q.device)                        # bool，是否启用 programmatic dependent launch

pos_encoding_mode = self._pos_encoding_mode
logits_soft_cap = self._logits_soft_cap
sm_scale = self._sm_scale
rope_scale = self._rope_scale
rope_theta = self._rope_theta
_check_pos_encoding_mode(pos_encoding_mode)
if logits_soft_cap is None:
    logits_soft_cap = 0.0
if sm_scale is None:
    sm_scale = 1.0 / math.sqrt(q.size(-1))                           # scalar，默认 1 / sqrt(D)
if rope_scale is None:
    rope_scale = 1.0
if rope_theta is None:
    rope_theta = 1e4

'''
第一段：把外部 Q/K/V 改成 paged prefill runtime 期望的布局
- q: [H_qo, qo_len, D] -> [H_kv * qo_len, G, D]
- k: [H_kv, kv_len, D] -> [H_kv * kv_len, 1, 1, D]
- v: [H_kv, kv_len, D] -> [H_kv * kv_len, 1, 1, D]
'''
q = einops.rearrange(
    q,
    "(num_kv_heads gqa_group_size) qo_len head_dim -> (num_kv_heads qo_len) gqa_group_size head_dim",
    num_kv_heads=self._num_kv_heads,
).contiguous()                                                       # [H_kv * qo_len, G, D]
k = einops.rearrange(
    k,
    "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
).contiguous()                                                       # [H_kv * kv_len, 1, 1, D]
v = einops.rearrange(
    v,
    "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
).contiguous()                                                       # [H_kv * kv_len, 1, 1, D]

'''
第二段：准备输出张量
'''
if return_lse:
    if lse is None:
        lse = torch.empty(
            (q.size(0), q.size(1)),
            dtype=torch.float32,
            device=q.device,
        )                                                            # [H_kv * qo_len, G]
    else:
        check_shape_dtype_device(
            lse,
            (q.size(0), q.size(1)),
            torch.float32,
            q.device,
            "lse",
        )

if out is None:
    out = torch.empty_like(q, dtype=self._o_dtype)                   # [H_kv * qo_len, G, D]
else:
    check_shape_dtype_device(out, q.shape, self._o_dtype, q.device, "out")

'''
第三段：把 plan() 阶段缓存的 metadata 与 plan_info 一次性交给 paged_run
'''
self._cached_module.paged_run(
    self._float_workspace_buffer,                                    # [workspace_bytes]
    self._int_workspace_buffer,                                      # [8MiB]
    self._plan_info,                                                 # [plan_info_words]
    q,                                                               # [H_kv * qo_len, G, D]
    k,                                                               # [H_kv * kv_len, 1, 1, D]
    v,                                                               # [H_kv * kv_len, 1, 1, D]
    self._qo_indptr,                                                 # [H_kv * R + 1]
    self._paged_kv_indptr_buf,                                       # [H_kv * R + 1]
    self._paged_kv_indices_buf,                                      # [nnz_tokens]
    self._paged_kv_last_page_len,                                    # [H_kv * R]
    out,                                                             # [H_kv * qo_len, G, D]
    lse,                                                             # [H_kv * qo_len, G] or None
    self._mask_mode,                                                 # scalar
    TensorLayout[self._kv_layout].value,                             # scalar，当前是 NHD
    -1,                                                              # scalar，window_left
    enable_pdl,                                                      # scalar
    None,
    None,
    None,
    None,
    None,
    None,
    logits_soft_cap,
    sm_scale,
    None,
    None,
    None,
    rope_scale,
    rope_theta,
    0,
    self._workspace_size,
)

'''
第四段：把输出从 runtime 视角 reshape 回外部视角
'''
out = einops.rearrange(
    out,
    "(num_kv_heads qo_len) gqa_group_size head_dim -> (num_kv_heads gqa_group_size) qo_len head_dim",
    num_kv_heads=self._num_kv_heads,
).contiguous()                                                       # [H_qo, qo_len, D]

if return_lse:
    lse = einops.rearrange(
        lse,
        "(num_kv_heads qo_len) gqa_group_size -> (num_kv_heads gqa_group_size) qo_len",
        num_kv_heads=self._num_kv_heads,
    ).contiguous()                                                   # [H_qo, qo_len]
```

这一段代码的职责很明确：

- `run()` 本身不重新做 metadata 翻译
- 它只负责最后一层“布局适配 + 参数透传”
- 真正的执行已经完全交给 `paged_run`

也就是说，到了这一章，variable block 语义在 Python 侧基本已经结束了。

## 3. binding 层：导出的其实只有三个符号

[`batch_prefill_jit_binding.cu:L22-L50`](src/batch_prefill_jit_binding_cu.md#__codelineno-0-22) 很短，但它把 Python 和 C++ 的协议完全钉死了：

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

## 4. `BatchPrefillWithPagedKVCacheRun()`：先恢复 `plan_info`，再解释输入 shape

真正执行阶段的主线从 [`batch_prefill.cu:L203-L267`](src/batch_prefill_cu.md#__codelineno-0-203) 开始：

```cpp
void BatchPrefillWithPagedKVCacheRun(
    TensorView float_workspace_buffer,
    TensorView int_workspace_buffer,
    Array<int64_t> plan_info_vec,
    TensorView q,
    TensorView paged_k_cache,
    TensorView paged_v_cache,
    TensorView qo_indptr,
    TensorView paged_kv_indptr,
    TensorView paged_kv_indices,
    TensorView paged_kv_last_page_len,
    TensorView o,
    Optional<TensorView> maybe_lse,
    int64_t mask_mode_code,
    int64_t layout,
    int64_t window_left,
    bool enable_pdl ADDITIONAL_FUNC_PARAMS
) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));  // 从 Python 序列化结果恢复

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  int64_t batch_size = paged_kv_indptr.size(0) - 1;                                         // = H_kv * R
  int64_t num_qo_heads = q.size(1);                                                         // = G
  int64_t num_kv_heads, page_size;
  uint32_t head_dim_qk = q.size(2);                                                         // = D

  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);                                                      // 当前会读到 1
    num_kv_heads = paged_k_cache.size(2);                                                   // 当前会读到 1
  }

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TVM_FFI_ICHECK_EQ(lse.size(0), q.size(0));
    TVM_FFI_ICHECK_EQ(lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer.data_ptr());
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  const auto q_stride_n = q.stride(0);
  const auto q_stride_h = q.stride(1);
  const int64_t* kv_cache_strides = paged_k_cache.strides().data();
```

这段代码把前面两章准备好的东西正式接起来了：

- `plan_info_vec` 是上一章 C++ `plan` 返回给 Python 的序列化结果
- `q / paged_k_cache / paged_v_cache` 是这一章 `run()` 刚刚重排好的 layout
- `paged_kv_indptr / paged_kv_indices / last_page_len` 是第一章生成好的 metadata

这里最关键的事实是：

- Python 已经把 `k/v` 改造成 `[num_pages, page_size, num_kv_heads, D]`
- 当前本地实现里 `page_size = 1`
- 当前逻辑 request 视角里 `num_kv_heads = 1`

因此对 C++ kernel 来说，它处理的就是“单 token page、单 KV head request”的普通 paged prefill 输入。

## 5. `PagedParams`：真正交给 kernel 的执行参数长什么样

上面那段代码接下来会继续构造 `PagedParams`：

```cpp
DISPATCH_context(
    DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
    USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
    RaggedParams, PagedParams, [&] {
      PagedParams params;

      params.q = static_cast<DTypeQ*>(q.data_ptr());                                           // [H_kv * qo_len, G, D]
      paged_kv_t<DTypeKV, IdType> paged_kv(
          num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
          static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
          static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
          static_cast<IdType*>(paged_kv_indices.data_ptr()),
          static_cast<IdType*>(paged_kv_indptr.data_ptr()),
          static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
      params.paged_kv = paged_kv;                                                             // page_size=1, num_heads=1 的 paged KV
      params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());                           // [H_kv * R + 1]
      params.o = static_cast<DTypeO*>(o.data_ptr());                                          // [H_kv * qo_len, G, D]

      params.lse = maybe_lse ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;  // [H_kv * qo_len, G] or nullptr
      params.num_qo_heads = num_qo_heads;                                                     // = G
      params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);                    // 这里通常就是 G / 1 = G
      params.q_stride_n = q_stride_n;
      params.q_stride_h = q_stride_h;
      params.window_left = window_left;

      params.request_indices = nullptr;
      params.qo_tile_indices = nullptr;
      params.kv_tile_indices = nullptr;
      params.merge_indptr = nullptr;
      params.o_indptr = nullptr;
      params.kv_chunk_size_ptr = nullptr;
      params.block_valid_mask = nullptr;
      params.total_num_rows = nullptr;
      params.max_total_num_rows = 0;
      params.padded_batch_size = 0;
      params.partition_kv = false;
```

如果只抓最重要的字段，可以这样理解：

- `params.q`
  - query 主体数据
  - shape 视角：`[H_kv * qo_len, G, D]`

- `params.paged_kv`
  - paged KV 主体数据 + page table + row indptr + last_page_len
  - 这是 variable block metadata 被完全吸收之后的统一执行接口

- `params.q_indptr`
  - query 侧 CSR 分段
  - 告诉 kernel 每个逻辑 request 对应哪一段 query token

- `params.o / params.lse`
  - 输出和可选的 LSE 缓冲

后面那些先置空的字段，是 planner 生成的调度数组位置；下一段会真正填进去。

## 6. 从 workspace 偏移恢复 planner 生成的调度数组

真正体现 `plan()` 价值的是 [`batch_prefill.cu:L295-L318`](src/batch_prefill_cu.md#__codelineno-0-295)：

```cpp
params.request_indices =
    GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
params.qo_tile_indices =
    GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
params.kv_tile_indices =
    GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
params.o_indptr =
    GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
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

这段代码说明：

- `plan()` 并不是只算出几个标量参数
- planner 已经把运行期真正需要的 request/tile/merge 索引组织进 workspace
- `run()` 阶段只是根据 `plan_info` 里记录的 offset，再把这些数组找出来挂回 `PagedParams`

这也是为什么第一章的 `float_workspace_buffer` / `_int_workspace_buffer` 不能简单理解成“一个临时 Tensor”：

- `int workspace` 里会放 request/tile/merge 相关的调度数组
- `float workspace` 在 `split_kv` 打开时会放 `tmp_v / tmp_s` 等中间结果

所以前面 `__init__()` 里那几个成员，真正的用途在这里才完全落地。

如果把这里恢复出来的几组数组再翻译成人话：

- `request_indices`
  - 运行期真正要处理哪些逻辑 request
  - 可以理解成“CTA 要按什么顺序取 request”

- `qo_tile_indices`
  - 每个 CTA / tile 对应哪一段 query tile

- `kv_tile_indices`
  - 每个 CTA / tile 对应哪一段 KV tile

- `o_indptr`
  - 输出缓冲区里各段结果的边界

- `kv_chunk_size_ptr`
  - 每个 split-k / tile 分块到底覆盖多少个 KV token

- `merge_indptr`
  - 如果 `split_kv` 打开，后续 merge 阶段该怎样把中间结果拼回去

## 7. 最终 dispatch：编译期模板实例 + 运行期 `cta_tile_q`

最后的落点很短，但非常关键，见 [`batch_prefill.cu:L321-L328`](src/batch_prefill_cu.md#__codelineno-0-321)：

```cpp
cudaError_t status = cudaSuccess;

DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
  status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
      CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
      PagedParams>(params, tmp_v, tmp_s, enable_pdl, stream);
});
```

这一步是典型的“两阶段选择”：

- 编译期已经由 JIT 规格固定了：
  - `HEAD_DIM_QK`
  - `MASK_MODE`
  - `POS_ENCODING_MODE`
  - `AttentionVariant`

- 运行期再由 planner 输出的 `cta_tile_q` 做最后一层 dispatch

所以如果按执行顺序理解：

- 第一章把 variable block 语义翻译成 token 级 metadata
- 第二章把这组 metadata 变成 planner 能理解的调度问题
- 第三章只是把“已经决定好的执行方案”真正交给底层 kernel

## 8. Jinja 模板在这一章的角色

这一章还能看到三份模板文件：

- [batch_prefill_customize_config.jinja](src/batch_prefill_customize_config_jinja.md)
- [batch_prefill_paged_kernel_inst.jinja](src/batch_prefill_paged_kernel_inst_jinja.md)
- [batch_prefill_ragged_kernel_inst.jinja](src/batch_prefill_ragged_kernel_inst_jinja.md)

它们并不承载算法逻辑，而是承载：

- 配置宏展开
- 模板实例化骨架
- 最终符号生成

所以它们更像“执行载体的生成模板”，而不是“variable block 算法本体”。

## 小结

按执行顺序看，这一章做了四件事：

1. Python `run()` 先把 q/k/v 重排成 paged prefill 期望布局。
2. `paged_run` 进入 C++，恢复前一章产出的 `plan_info`。
3. C++ 侧拼装 `PagedParams`，从 workspace 偏移恢复调度数组。
4. 最终 dispatch 到标准 FA2 paged prefill kernel。

因此，真正执行阶段并没有额外的 variable block 专属 kernel；这一层做的仍然是成熟的 paged prefill 执行路径。
