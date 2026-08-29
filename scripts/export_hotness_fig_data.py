#!/usr/bin/env python3
"""Export the paper fig:hotness data (coverage vs hot-region size, retention vs lag) to CSV.

Sources (V4-Flash IQ2_XXS, ds4 CPU, k = 512 blocks of 4 tokens, 21 CSA layers):
  runs/niah_single_2_{16384,32768,65536}_moe_q2/analysis/hotset_coverage.json  (coverage_by_pool_pct)
  runs/niah_single_2_65536_moe_q2/analysis/metrics_run_summary.json            (overall_retention, overall_working_set_ratio)
  runs/longform_p16k_g4k_q2/analysis/extended_retention.json                    (retention, working_set_ratio, lags 1..2048)
Output: <sparse_attn_cpu>/docs/00_doc/data/hotness_coverage_{64k,32k,16k}.csv, hotness_retention.csv
Usage: python3 export_hotness_fig_data.py [out_dir]
"""
import csv, json, sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / "01_github/sparse_attn_cpu/docs/00_doc/data"
OUT.mkdir(parents=True, exist_ok=True)

for L, tag in ((65536, "64k"), (32768, "32k"), (16384, "16k")):
    d = json.load(open(RUNS / f"niah_single_2_{L}_moe_q2/analysis/hotset_coverage.json"))
    cov = d["coverage_by_pool_pct"]
    with open(OUT / f"hotness_coverage_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool_pct", "mean", "min", "max"])
        for p in sorted(cov, key=float):
            w.writerow([p] + [f"{cov[p][k]:.6f}" for k in ("mean", "min", "max")])
    print(tag, d["run_id"], "pool_N_mean", d["pool_N_mean"], "A99_pct_mean", round(d["A99_pct_mean"], 2), "n_layers", d["n_csa_layers"])

s = json.load(open(RUNS / "niah_single_2_65536_moe_q2/analysis/metrics_run_summary.json"))
e = json.load(open(RUNS / "longform_p16k_g4k_q2/analysis/extended_retention.json"))
ret64 = {int(k.split("_")[1]): v for k, v in s["overall_retention"].items()}
ws64 = {int(k[1:]): v for k, v in s["overall_working_set_ratio"].items()}
retL = {int(k): v for k, v in e["retention"].items()}
wsL = {int(k): v for k, v in e["working_set_ratio"].items()}
lags = sorted(set(ret64) | set(retL))
fmt = lambda x: "" if x is None else f"{x:.6f}"
with open(OUT / "hotness_retention.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["lag", "retention_ruler64k", "working_set_ratio_ruler64k", "retention_longform", "working_set_ratio_longform"])
    for lag in lags:
        w.writerow([lag, fmt(ret64.get(lag)), fmt(ws64.get(lag)), fmt(retL.get(lag)), fmt(wsL.get(lag))])
print("ruler64k steps", s["n_decode_steps"], "longform steps", e["n_steps"], "layers", e["n_layers"])
