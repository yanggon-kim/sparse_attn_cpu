#!/usr/bin/env python3
"""Cross-sample aggregation of the LongBench collection (Results 1/2/3 with error bars).

Usage: aggregate_longbench.py [out_dir]
Reads  runs/lb_*_q2/analysis/{metrics_run_summary,moe_metrics_run_summary,hotset_coverage}.json
       + prompts/sample.json (task attribution)
Writes <out_dir>/longbench_aggregate.json (default analysis_longbench/) and prints tables.

Statistics: per task and pooled, mean +/- sample std (ddof=1) and a 95% bootstrap CI over samples
(reuses locality_lib.bootstrap_ci). n is small (12/task) so bootstrap is the honest choice.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locality_lib import bootstrap_ci

EXP = "<WORKDIR>/experiment"
LAGS = [1, 2, 4, 8, 16, 32, 64]
out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(EXP, "analysis_longbench")
os.makedirs(out_dir, exist_ok=True)


def load_runs():
    rows = []
    for rd in sorted(glob.glob(os.path.join(EXP, "runs", "lb_*_q2"))):
        try:
            sample = json.load(open(os.path.join(rd, "prompts", "sample.json")))
            kv = json.load(open(os.path.join(rd, "analysis", "metrics_run_summary.json")))
            moe = json.load(open(os.path.join(rd, "analysis", "moe_metrics_run_summary.json")))
            hot = json.load(open(os.path.join(rd, "analysis", "hotset_coverage.json")))
        except Exception:
            continue  # incomplete run
        rows.append({
            "run": os.path.basename(rd),
            "task": sample["task_subtype"],
            "prompt_tokens": sample["context_length_target"],
            "decode_steps": kv.get("n_decode_steps"),
            # R1 KV
            "kv_adj": kv["overall_adjacent_overlap_mean"],
            "kv_lift": kv["overall_locality_lift_mean"],
            "kv_recency": kv.get("overall_recency_overlap_mean"),
            "kv_cand": kv.get("mean_n_candidates"),
            "kv_retention": {str(l): kv["overall_retention"][f"lag_{l}"] for l in LAGS},
            # R2 MoE
            "moe_adj": moe["learned"]["adjacent_overlap_mean"],
            "moe_lift": moe["learned"]["locality_lift_mean"],
            "moe_hash_adj": moe["hash"]["adjacent_overlap_mean"],
            # R3 hot-set
            "hot_A99": hot["A99_pct_mean"],
            "hot_B99": hot["B99_pct_mean"],
            "hot_cov": {p: hot["coverage_by_pool_pct"][p]["mean"] for p in hot.get("coverage_by_pool_pct", {})},
        })
    return rows


def stat(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    lo, hi = bootstrap_ci(vals)
    return {"n": n, "mean": mean, "std": std, "ci95": [lo, hi]}


def group(rows, keys):
    out = {}
    for k in keys:
        out[k] = stat([r[k] for r in rows])
    out["kv_retention"] = {str(l): stat([r["kv_retention"][str(l)] for r in rows]) for l in LAGS}
    pool_pcts = sorted({p for r in rows for p in r["hot_cov"]}, key=lambda x: float(x))
    out["hot_coverage_by_pool_pct"] = {p: stat([r["hot_cov"].get(p) for r in rows]) for p in pool_pcts}
    return out


def main():
    rows = load_runs()
    if not rows:
        print("no analyzed lb_* runs found")
        return
    scalar_keys = ["prompt_tokens", "decode_steps", "kv_adj", "kv_lift", "kv_recency", "kv_cand",
                   "moe_adj", "moe_lift", "moe_hash_adj", "hot_A99", "hot_B99"]
    tasks = sorted({r["task"] for r in rows})
    agg = {"n_runs": len(rows), "per_task": {}, "pooled": group(rows, scalar_keys),
           "runs": rows}
    for t in tasks:
        agg["per_task"][t] = group([r for r in rows if r["task"] == t], scalar_keys)

    json.dump(agg, open(os.path.join(out_dir, "longbench_aggregate.json"), "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else o)

    def fmt(s, nd=3):
        return f"{s['mean']:.{nd}f}±{s['std']:.{nd}f}" if s else "-"

    print(f"\n=== LongBench aggregate ({len(rows)} runs) ===")
    hdr = f"{'task':<12} {'n':>2} {'ptok':>6} {'steps':>5} | {'KV adj':>11} {'KV lift':>11} {'recency':>11} | {'MoE adj':>11} {'MoE hash':>11} | {'A@99%':>10} {'B@99%':>10}"
    print(hdr)
    for t in tasks + ["POOLED"]:
        g = agg["per_task"][t] if t != "POOLED" else agg["pooled"]
        print(f"{t:<12} {g['kv_adj']['n']:>2} {g['prompt_tokens']['mean']:>6.0f} {g['decode_steps']['mean']:>5.0f} | "
              f"{fmt(g['kv_adj']):>11} {fmt(g['kv_lift'],2):>11} {fmt(g['kv_recency']):>11} | "
              f"{fmt(g['moe_adj']):>11} {fmt(g['moe_hash_adj']):>11} | "
              f"{fmt(g['hot_A99'],1):>10} {fmt(g['hot_B99'],1):>10}")
    print("\nKV retention (pooled):",
          {l: round(agg["pooled"]["kv_retention"][str(l)]["mean"], 3) for l in LAGS})


if __name__ == "__main__":
    main()
