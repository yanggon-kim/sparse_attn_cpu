#!/usr/bin/env python3
"""Aggregate per-run analysis JSONs of a model's ladder into the exp1 §6 / exp2 §5 tables.
Usage: aggregate_gpu_sweep.py --runs-root <WORKDIR>/runs --model-tag v32 --model-name DeepSeek-V3.2
                              --out docs/gpu_sweep [--variant all_layers|computing_only] [--suffix ""]
Reads per run: analysis/metrics_run_summary.json (R1), analysis/extended_retention.json (long lags),
  analysis/moe_metrics_run_summary.json (R2), analysis/hotset_coverage.json (R3), run_manifest.json, req.json.
Writes: R1_kv_locality.csv, R2_moe_locality.csv, R3_hotset_coverage.csv (per run rows + per-rung aggregate rows
  with mean, std, 95% bootstrap CI over runs via locality_lib.bootstrap_ci), accuracy_by_source.csv,
  and sweep_<model-tag>.json (everything, for the summary md / plots). Run ids are kept on every row.
"""
import argparse, csv, glob, json, os, statistics, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from locality_lib import bootstrap_ci  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs-root", required=True)
ap.add_argument("--model-tag", required=True)
ap.add_argument("--model-name", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--variant", default="all_layers")
ap.add_argument("--suffix", default="")
ap.add_argument("--exclude", nargs="*", default=["_smoke", "_smoke2", "_solo", "_b", "smoke0_capital_s0"])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
CPU_V4 = {4096: (0.868, 1.72), 8192: (0.790, 2.92), 16384: (0.718, 5.72), 32768: (0.672, 10.53), 65536: (0.668, 21.37)}


def jl(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def agg(vals):
    v = [x for x in vals if x is not None and x == x]
    if not v:
        return {"n": 0, "mean": None, "std": None, "ci_lo": None, "ci_hi": None}
    ci = bootstrap_ci(v) if len(v) > 1 else (v[0], v[0])
    return {"n": len(v), "mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0,
            "ci_lo": ci[0], "ci_hi": ci[1]}


rows = []
for rd in sorted(glob.glob(os.path.join(a.runs_root, f"{a.model_tag}_*"))):
    rid = os.path.basename(rd)
    base = rid[:-2] if rid.endswith("_b") else rid
    if any(base.endswith(x) for x in a.exclude if x != "_b"):
        continue
    if a.variant == "computing_only" and not rid.endswith("_b"):
        continue
    if a.variant == "all_layers" and rid.endswith("_b"):
        continue
    man = jl(os.path.join(rd, "run_manifest.json"))
    m = jl(os.path.join(rd, "analysis", "metrics_run_summary.json"))
    if not man or not m:
        continue
    req = jl(os.path.join(rd, "req.json")) or jl(os.path.join(rd.rstrip("_b"), "req.json")) or {}
    er = jl(os.path.join(rd, "analysis", "extended_retention.json")) or {}
    moe = jl(os.path.join(rd, "analysis", "moe_metrics_run_summary.json")) or {}
    hs = jl(os.path.join(rd, "analysis", "hotset_coverage.json")) or {}
    cov = hs.get("coverage_by_pool_pct", {})
    ret = m.get("overall_retention", {})
    eret = er.get("retention", er.get("overall_retention", {})) if isinstance(er, dict) else {}
    eret = {("lag_" + k if not str(k).startswith("lag_") else k): v for k, v in eret.items()} if isinstance(eret, dict) else {}
    covm = lambda k: (cov.get(k, {}).get("mean") if isinstance(cov.get(k), dict) else cov.get(k))
    ws = m.get("overall_working_set_ratio", {})
    learned = moe.get("learned", {})
    rows.append({
        "run_id": rid, "model": a.model_name, "variant": a.variant, "rung": man.get("context_length_target"),
        "kind": man.get("kind"), "source": man.get("benchmark_name"), "task": man.get("task_subset"),
        "prompt_tokens": man.get("context_length_actual_tokens"), "decode_steps": m.get("n_decode_steps"),
        "n_csa_layers": m.get("n_csa_layers"), "mean_n_candidates": m.get("mean_n_candidates"),
        "adjacent_overlap": m.get("overall_adjacent_overlap_mean"), "adjacent_jaccard": m.get("overall_adjacent_jaccard_mean"),
        "weighted_overlap": m.get("overall_weighted_overlap_mean"), "lift_vs_random": m.get("overall_locality_lift_mean"),
        "recency_baseline_overlap": m.get("overall_recency_overlap_mean"),
        "retention_lag1": ret.get("lag_1"), "retention_lag8": ret.get("lag_8"), "retention_lag64": ret.get("lag_64"),
        "retention_lag512": (eret.get("lag_512") if isinstance(eret, dict) else None),
        "retention_lag1024": (eret.get("lag_1024") if isinstance(eret, dict) else None),
        "retention_lag2048": (eret.get("lag_2048") if isinstance(eret, dict) else None),
        "working_set_w64_ratio": ws.get("w64"),
        "moe_adjacent_overlap": learned.get("adjacent_overlap_mean"), "moe_lift": learned.get("locality_lift_mean"),
        "moe_random_baseline": moe.get("random_baseline_overlap"), "moe_n_layers": learned.get("n_layers"),
        "hotset_pool_N": hs.get("pool_N_mean"), "hotset_A99_pct": hs.get("A99_pct_mean"),
        "hotset_A99_range": json.dumps(hs.get("A99_pct_range")) if hs.get("A99_pct_range") else None,
        "hotset_B99_pct": hs.get("B99_pct_mean"), "hotset_top10pct_coverage": covm("10"),
        "hotset_top1pct_coverage": covm("1"), "hotset_top5pct_coverage": covm("5"),
        "is_correct": req.get("is_correct"), "score": req.get("score"), "finish_reason": req.get("finish_reason"),
        "gen_tokens": len(req.get("generated_token_ids", [])) if req else None,
    })

rungs = sorted({r["rung"] for r in rows})
R1_COLS = ["adjacent_overlap", "adjacent_jaccard", "weighted_overlap", "lift_vs_random", "recency_baseline_overlap",
           "retention_lag1", "retention_lag8", "retention_lag64", "retention_lag512", "retention_lag1024", "retention_lag2048", "working_set_w64_ratio"]
R2_COLS = ["moe_adjacent_overlap", "moe_lift"]
R3_COLS = ["hotset_A99_pct", "hotset_B99_pct", "hotset_top10pct_coverage", "hotset_top5pct_coverage", "hotset_top1pct_coverage", "hotset_pool_N"]


def write_table(name, cols, rows_):
    p = os.path.join(a.out, name)
    per_run = [{k: r[k] for k in ["run_id", "model", "variant", "rung", "kind", "source", "task", "prompt_tokens", "decode_steps"] + cols} for r in rows_]
    aggs = []
    for rung in rungs:
        for kind in ("bf", "ld", "all"):
            sel = [r for r in rows_ if r["rung"] == rung and (kind == "all" or r["kind"] == kind)]
            if not sel:
                continue
            row = {"run_id": f"AGG:{a.model_tag}:{rung}:{kind}", "model": a.model_name, "variant": a.variant, "rung": rung,
                   "kind": kind, "source": "aggregate", "task": f"n={len(sel)}", "prompt_tokens": statistics.mean(x["prompt_tokens"] for x in sel),
                   "decode_steps": statistics.mean(x["decode_steps"] for x in sel if x["decode_steps"])}
            for c in cols:
                g = agg([x[c] for x in sel])
                row[c] = g["mean"]
                row[c + "_std"] = g["std"]
                row[c + "_ci_lo"] = g["ci_lo"]
                row[c + "_ci_hi"] = g["ci_hi"]
                row[c + "_n"] = g["n"]
            if name.startswith("R1") and kind == "all" and rung in CPU_V4:
                row["cpu_v4_adjacent_overlap"], row["cpu_v4_lift"] = CPU_V4[rung]
            aggs.append(row)
    keys = list(per_run[0].keys()) if per_run else []
    for r in aggs:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in per_run + aggs:
            w.writerow(r)
    print(f"wrote {p}: {len(per_run)} runs + {len(aggs)} aggregates")
    return aggs


sfx = a.suffix
r1 = write_table(f"R1_kv_locality{sfx}.csv", R1_COLS, rows)
r2 = write_table(f"R2_moe_locality{sfx}.csv", R2_COLS, [r for r in rows if r["moe_adjacent_overlap"] is not None])
r3 = write_table(f"R3_hotset_coverage{sfx}.csv", R3_COLS, [r for r in rows if r["hotset_A99_pct"] is not None])
# accuracy sanity by (rung, source)
acc = []
for rung in rungs:
    for src in sorted({r["source"] for r in rows}):
        sel = [r for r in rows if r["rung"] == rung and r["source"] == src and r["kind"] == "bf"]
        if not sel:
            continue
        corr = [r["is_correct"] for r in sel if r["is_correct"] is not None]
        acc.append({"model": a.model_name, "rung": rung, "source": src, "n": len(sel),
                    "accuracy": (sum(1 for c in corr if c) / len(corr)) if corr else None,
                    "mean_score": statistics.mean(r["score"] for r in sel if r["score"] is not None) if any(r["score"] is not None for r in sel) else None,
                    "run_ids": ";".join(r["run_id"] for r in sel)})
with open(os.path.join(a.out, f"accuracy_by_source{sfx}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "rung", "source", "n", "accuracy", "mean_score", "run_ids"])
    w.writeheader()
    for r in acc:
        w.writerow(r)
json.dump({"model": a.model_name, "model_tag": a.model_tag, "variant": a.variant, "runs": rows, "R1": r1, "R2": r2, "R3": r3,
           "accuracy": acc, "cpu_v4": CPU_V4}, open(os.path.join(a.out, f"sweep_{a.model_tag}{sfx}.json"), "w"), indent=1)
for r in r1:
    if r["kind"] == "all":
        print(f"rung {r['rung']:>6}: n={r['task']} adj_overlap={r['adjacent_overlap']:.3f} lift={r['lift_vs_random']:.2f} "
              f"recency={r['recency_baseline_overlap'] if r['recency_baseline_overlap'] is None else round(r['recency_baseline_overlap'],3)}")
