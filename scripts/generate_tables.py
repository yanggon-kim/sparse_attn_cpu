#!/usr/bin/env python3
"""Generate summary tables (CSV + Markdown) across runs (doc §28).
Usage: generate_tables.py <runs_root> <out_dir>
"""
import json, os, sys
import pandas as pd

runs_root, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
LAGS = [1, 2, 4, 8, 16, 32, 64]
WINDOWS = [1, 4, 16, 64]


def load_runs():
    runs = []
    for rid in sorted(os.listdir(runs_root)):
        rd = os.path.join(runs_root, rid)
        sl_p = os.path.join(rd, "analysis", "metrics_sample_layer.parquet")
        sm_p = os.path.join(rd, "analysis", "metrics_run_summary.json")
        if os.path.exists(sl_p) and os.path.exists(sm_p):
            runs.append({"run_id": rid, "sl": pd.read_parquet(sl_p),
                         "summary": json.load(open(sm_p)),
                         "manifest": json.load(open(os.path.join(rd, "run_manifest.json"))),
                         "gen": json.loads(open(os.path.join(rd, "outputs", "generations.jsonl")).read().splitlines()[0])})
    return runs


def emit(df, name, title):
    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    with open(os.path.join(out_dir, f"{name}.md"), "w") as f:
        f.write(f"### {title}\n\n{df.to_markdown(index=False)}\n")
    print(f"  {name}: {len(df)} rows")


def main():
    runs = load_runs()
    if not runs:
        print("no completed runs yet"); return
    # 1. run configuration summary
    cfg = pd.DataFrame([{
        "run_id": r["run_id"], "context_target": r["summary"]["context_length"],
        "context_actual_tok": r["manifest"].get("context_length_actual_tokens"),
        "quant": r["manifest"]["quantization"][:18], "top_k": r["summary"]["top_k"],
        "csa_layers": r["summary"]["n_csa_layers"], "decode_steps": r["summary"]["n_decode_steps"],
        "prefill_tok_s": (r["manifest"].get("timing", {}).get("prefill_tok_per_s")
                          or r["manifest"].get("timing", {}).get("derived_prefill_tok_per_s")),
        "wall_clock": r["manifest"].get("timing", {}).get("wall_clock"),
        "peak_rss_gb": round((r["manifest"].get("timing", {}).get("peak_rss_bytes") or 0) / 1e9, 1),
        "correct": r["summary"]["is_correct"]} for r in runs])
    emit(cfg, "01_run_config", "Run configuration summary")

    # 2. sample counts
    emit(pd.DataFrame([{"benchmark": "RULER", "task": "niah_single_2", "context": r["summary"]["context_length"],
                        "samples": 1, "correct": r["summary"]["is_correct"]} for r in runs]),
         "02_sample_counts", "Benchmark/task sample counts")

    # 3/4. per-layer adjacent overlap + churn (one column per run)
    def per_layer(col, name, title):
        base = None
        for r in runs:
            s = r["sl"][["layer_id", col]].rename(columns={col: f"{r['summary']['context_length']}"})
            base = s if base is None else base.merge(s, on="layer_id", how="outer")
        emit(base.sort_values("layer_id"), name, title)
    per_layer("adjacent_overlap_mean", "03_per_layer_adjacent_overlap", "Per-layer adjacent overlap (by context length)")
    per_layer("churn_mean", "04_per_layer_churn", "Per-layer churn (by context length)")

    # 5. retention lags (run x lag)
    emit(pd.DataFrame([{"context": r["summary"]["context_length"],
                        **{f"lag_{l}": r["summary"]["overall_retention"][f"lag_{l}"] for l in LAGS}} for r in runs]),
         "05_retention_lags", "Retention at lags 1/2/4/8/16/32/64 (overall mean)")

    # 6. working set ratio
    emit(pd.DataFrame([{"context": r["summary"]["context_length"],
                        **{f"ws_ratio_w{w}": r["summary"]["overall_working_set_ratio"][f"w{w}"] for w in WINDOWS}} for r in runs]),
         "06_working_set", "Normalized working-set ratio at windows 1/4/16/64")

    # 7. reuse-distance percentiles (mean over layers)
    emit(pd.DataFrame([{"context": r["summary"]["context_length"],
                        "cold_fraction": r["sl"].reuse_cold_fraction.mean(),
                        "reuse_p50": r["sl"].reuse_p50.mean(), "reuse_p90": r["sl"].reuse_p90.mean(),
                        "reuse_p99": r["sl"].reuse_p99.mean(),
                        "persistence_mean_run": r["sl"].persistence_mean_run.mean()} for r in runs]),
         "07_reuse_distance", "Reuse-distance percentiles (mean over CSA layers)")

    # 8. access-age percentiles
    emit(pd.DataFrame([{"context": r["summary"]["context_length"],
                        "age_mean": r["sl"].age_mean.mean(), "age_median": r["sl"].age_median.mean(),
                        "age_p90": r["sl"].age_p90.mean(), "age_p99": r["sl"].age_p99.mean(),
                        "frac_recent": r["sl"].frac_recent.mean(), "frac_middle": r["sl"].frac_middle.mean(),
                        "frac_old": r["sl"].frac_old.mean()} for r in runs]),
         "08_access_age", "Access-age percentiles + recent/middle/old fractions")

    # 9. correct vs incorrect
    emit(pd.DataFrame([{"group": "correct" if r["summary"]["is_correct"] else "incorrect",
                        "context": r["summary"]["context_length"],
                        "adjacent_overlap": r["summary"]["overall_adjacent_overlap_mean"],
                        "locality_lift": r["summary"]["overall_locality_lift_mean"]} for r in runs]),
         "09_correct_vs_incorrect", "Correct vs incorrect comparison")

    # 10. context-length comparison
    emit(pd.DataFrame([{"context": r["summary"]["context_length"],
                        "adjacent_overlap": r["summary"]["overall_adjacent_overlap_mean"],
                        "adjacent_jaccard": r["summary"]["overall_adjacent_jaccard_mean"],
                        "weighted_overlap": r["summary"]["overall_weighted_overlap_mean"],
                        "locality_lift": r["summary"]["overall_locality_lift_mean"],
                        "recency_overlap": r["summary"]["overall_recency_overlap_mean"],
                        "mean_n_candidates": r["summary"]["mean_n_candidates"]} for r in runs]),
         "10_context_length", "Context-length comparison")

    # 11/12 placeholders (full-spec: deferred axes)
    emit(pd.DataFrame([{"axis": "quantization", "status": "Q2 only this pass; Q4 deferred to expansion"},
                       {"axis": "CPU_vs_GPU", "status": "CPU only; GPU adapter documented, run deferred"}]),
         "11_12_deferred_axes", "Quantization & CPU-vs-GPU axes (deferred)")
    print(f"tables -> {out_dir}")


if __name__ == "__main__":
    main()
