import argparse
import importlib

import torch
import triton.testing as tt


def summarize_times(times_ms):
    times = torch.tensor(times_ms, dtype=torch.float64)
    return {
        "mean_ms": float(times.mean()),
        "median_ms": float(times.median()),
        "p20_ms": float(torch.quantile(times, 0.2)),
        "p80_ms": float(torch.quantile(times, 0.8)),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "std_ms": float(times.std(unbiased=False)),
        "n": int(times.numel()),
    }


def make_mask(num_kv_heads, num_blocks_row, num_blocks_col, density):
    mask = torch.zeros(
        (num_kv_heads, num_blocks_row, num_blocks_col),
        dtype=torch.bool,
    )
    threshold = int(round(density * 1000))
    for h in range(num_kv_heads):
        for r in range(num_blocks_row):
            for c in range(num_blocks_col):
                score = (h * 193 + r * 97 + c * 57 + r * c * 13) % 1000
                if score < threshold:
                    mask[h, r, c] = True
            if not mask[h, r].any():
                mask[h, r, (r + h) % num_blocks_col] = True
            mask[h, r, (r * 3 + h) % num_blocks_col] = True
    return mask


def make_inputs(
    seq_len,
    row_blocks,
    col_blocks,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    seed,
):
    assert sum(row_blocks) == seq_len
    assert sum(col_blocks) == seq_len

    torch.manual_seed(seed)
    block_row_sz = torch.tensor([row_blocks] * num_kv_heads, dtype=torch.int32)
    block_col_sz = torch.tensor([col_blocks] * num_kv_heads, dtype=torch.int32)
    q = torch.randn((num_qo_heads, seq_len, head_dim), dtype=torch.float16)
    k = torch.randn((num_kv_heads, seq_len, head_dim), dtype=torch.float16)
    v = torch.randn((num_kv_heads, seq_len, head_dim), dtype=torch.float16)
    return block_row_sz, block_col_sz, q, k, v


SCENARIOS = [
    {
        "name": "s128_uniform16_d50",
        "seq_len": 128,
        "row_blocks": [16] * 8,
        "col_blocks": [16] * 8,
        "density": 0.50,
        "seed": 101,
    },
    {
        "name": "s256_mildvar_d45",
        "seq_len": 256,
        "row_blocks": [8, 24, 40, 56, 64, 32, 16, 16],
        "col_blocks": [32, 16, 64, 24, 40, 8, 56, 16],
        "density": 0.45,
        "seed": 102,
    },
    {
        "name": "s512_uniform32_d30",
        "seq_len": 512,
        "row_blocks": [32] * 16,
        "col_blocks": [32] * 16,
        "density": 0.30,
        "seed": 103,
    },
    {
        "name": "s512_highvar_d25",
        "seq_len": 512,
        "row_blocks": [8, 8, 16, 32, 64, 96, 128, 96, 32, 16, 8, 8],
        "col_blocks": [8, 8, 16, 32, 96, 128, 96, 64, 32, 16, 8, 8],
        "density": 0.25,
        "seed": 104,
    },
    {
        "name": "s1024_uniform64_d20",
        "seq_len": 1024,
        "row_blocks": [64] * 16,
        "col_blocks": [64] * 16,
        "density": 0.20,
        "seed": 105,
    },
    {
        "name": "s1024_spiky_d18",
        "seq_len": 1024,
        "row_blocks": [8, 8, 16, 32, 64, 128, 256, 256, 128, 64, 32, 16, 8, 8],
        "col_blocks": [8, 8, 16, 32, 64, 128, 256, 256, 128, 64, 32, 16, 8, 8][
            ::-1
        ],
        "density": 0.18,
        "seed": 106,
    },
    {
        "name": "s2048_uniform128_d12",
        "seq_len": 2048,
        "row_blocks": [128] * 16,
        "col_blocks": [128] * 16,
        "density": 0.12,
        "seed": 107,
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=400)
    args = parser.parse_args()

    package = importlib.import_module(args.package)

    device = torch.device("cuda")
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 128

    all_results = {}
    for scenario in SCENARIOS:
        scenario_results = {}
        block_mask_map = make_mask(
            num_kv_heads,
            len(scenario["row_blocks"]),
            len(scenario["col_blocks"]),
            scenario["density"],
        )
        block_row_sz, block_col_sz, q, k, v = make_inputs(
            scenario["seq_len"],
            scenario["row_blocks"],
            scenario["col_blocks"],
            num_qo_heads,
            num_kv_heads,
            head_dim,
            scenario["seed"],
        )
        block_mask_map = block_mask_map.to(device)
        block_row_sz = block_row_sz.to(device)
        block_col_sz = block_col_sz.to(device)
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)

        for causal in (False, True):
            wrapper = package.VariableBlockSparseAttentionWrapper(
                workspace,
                backend="auto",
            )
            wrapper.plan(
                block_mask_map,
                block_row_sz,
                block_col_sz,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                causal=causal,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
            )
            out, lse = wrapper.run(q, k, v, return_lse=True)

            bench_wrapper = package.VariableBlockSparseAttentionWrapper(
                workspace,
                backend="auto",
            )
            bench_wrapper.plan(
                block_mask_map,
                block_row_sz,
                block_col_sz,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                causal=causal,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
            )
            plan_times = tt.do_bench(
                lambda: bench_wrapper.plan(
                    block_mask_map,
                    block_row_sz,
                    block_col_sz,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    causal=causal,
                    q_data_type=q.dtype,
                    kv_data_type=k.dtype,
                ),
                warmup=args.warmup_ms,
                rep=args.rep_ms,
                return_mode="all",
            )
            run_times = tt.do_bench(
                lambda: wrapper.run(q, k, v),
                warmup=args.warmup_ms,
                rep=args.rep_ms,
                return_mode="all",
            )

            scenario_results[str(causal)] = {
                "backend": wrapper._backend,
                "seq_len": scenario["seq_len"],
                "num_row_blocks": len(scenario["row_blocks"]),
                "num_col_blocks": len(scenario["col_blocks"]),
                "density": scenario["density"],
                "row_blocks": scenario["row_blocks"],
                "col_blocks": scenario["col_blocks"],
                "qo_indptr": wrapper._qo_indptr.detach().cpu(),
                "kv_indptr": wrapper._paged_kv_indptr_buf.detach().cpu(),
                "kv_indices": wrapper._paged_kv_indices_buf.detach().cpu(),
                "last_page_len": wrapper._paged_kv_last_page_len.detach().cpu(),
                "out": out.detach().cpu(),
                "lse": lse.detach().cpu(),
                "plan_bench": summarize_times(plan_times),
                "run_bench": summarize_times(run_times),
            }

        all_results[scenario["name"]] = scenario_results

    torch.save(all_results, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
