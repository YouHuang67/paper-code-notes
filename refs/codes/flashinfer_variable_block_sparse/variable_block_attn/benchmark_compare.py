import argparse

import torch


def fmt(x):
    return f"{x:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--local", required=True)
    args = parser.parse_args()

    up = torch.load(args.upstream, map_location="cpu")
    local = torch.load(args.local, map_location="cpu")

    for scenario_name in up:
        print(f"scenario {scenario_name}")
        for causal in ("False", "True"):
            u = up[scenario_name][causal]
            l = local[scenario_name][causal]
            out_diff = (u["out"].float() - l["out"].float()).abs()
            lse_diff = (u["lse"].float() - l["lse"].float()).abs()
            plan_ratio = l["plan_bench"]["median_ms"] / u["plan_bench"]["median_ms"]
            run_ratio = l["run_bench"]["median_ms"] / u["run_bench"]["median_ms"]
            print(
                "  causal=",
                causal,
                "backend=",
                u["backend"],
                l["backend"],
                "meta=",
                torch.equal(u["qo_indptr"], l["qo_indptr"])
                and torch.equal(u["kv_indptr"], l["kv_indptr"])
                and torch.equal(u["kv_indices"], l["kv_indices"])
                and torch.equal(u["last_page_len"], l["last_page_len"]),
            )
            print(
                "    out_max_abs=",
                float(out_diff.max()),
                "lse_max_abs=",
                float(lse_diff.max()),
            )
            print(
                "    upstream_plan_ms median/p20/p80=",
                fmt(u["plan_bench"]["median_ms"]),
                fmt(u["plan_bench"]["p20_ms"]),
                fmt(u["plan_bench"]["p80_ms"]),
            )
            print(
                "    local_plan_ms    median/p20/p80=",
                fmt(l["plan_bench"]["median_ms"]),
                fmt(l["plan_bench"]["p20_ms"]),
                fmt(l["plan_bench"]["p80_ms"]),
                "ratio=",
                fmt(plan_ratio),
            )
            print(
                "    upstream_run_ms  median/p20/p80=",
                fmt(u["run_bench"]["median_ms"]),
                fmt(u["run_bench"]["p20_ms"]),
                fmt(u["run_bench"]["p80_ms"]),
            )
            print(
                "    local_run_ms     median/p20/p80=",
                fmt(l["run_bench"]["median_ms"]),
                fmt(l["run_bench"]["p20_ms"]),
                fmt(l["run_bench"]["p80_ms"]),
                "ratio=",
                fmt(run_ratio),
            )


if __name__ == "__main__":
    main()
