#!/usr/bin/env python3
"""Generate KV lightning-indexer locality plots (PNG) for the ds4 CPU report.

Usage: generate_kv_plots.py <out_dir> <run_dir> [<run_dir> ...]
Each run_dir must have analysis/metrics_run_summary.json and analysis/metrics_sample_layer.parquet
(produced by analyze_locality.py). Writes combined PNGs (across the given context lengths) into <out_dir>:
  kv_01_context_scaling.png, kv_02_retention.png, kv_03_adjacent_overlap_by_layer.png
"""
import json, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAGS = [1, 2, 4, 8, 16, 32, 64]
PALETTE = ["#1f6feb", "#8250df", "#c55a11", "#1a7f37", "#b54708"]

out_dir = sys.argv[1]
run_dirs = sys.argv[2:]
os.makedirs(out_dir, exist_ok=True)

runs = []
for rd in run_dirs:
    try:
        summ = json.load(open(os.path.join(rd, "analysis", "metrics_run_summary.json")))
        sl = pd.read_parquet(os.path.join(rd, "analysis", "metrics_sample_layer.parquet"))
        runs.append((summ.get("context_length") or os.path.basename(rd), summ, sl))
    except Exception as e:
        print(f"skip {rd}: {e}", file=sys.stderr)
runs.sort(key=lambda x: (x[0] if isinstance(x[0], (int, float)) else 1e9))
lab = lambda c: f"{int(c)//1024}K" if isinstance(c, (int, float)) else str(c)
col = {c: PALETTE[i % len(PALETTE)] for i, (c, _, _) in enumerate(runs)}

# 1) context scaling: adjacent overlap + lift
fig, ax = plt.subplots(figsize=(7, 4.2))
xs = [lab(c) for c, _, _ in runs]
ov = [s["overall_adjacent_overlap_mean"] for _, s, _ in runs]
lift = [s["overall_locality_lift_mean"] for _, s, _ in runs]
ax.plot(xs, ov, "-o", color="#1f6feb", label="adjacent overlap")
ax.set(xlabel="context length", ylabel="adjacent overlap", title="KV indexer locality vs context length (ds4 CPU)")
ax.set_ylim(0, 1); ax.grid(alpha=.3)
ax2 = ax.twinx(); ax2.plot(xs, lift, "-s", color="#1a7f37", label="locality lift ×")
ax2.set_ylabel("locality lift (× random)")
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "kv_01_context_scaling.png"), dpi=120); plt.close(fig)

# 2) retention per context
fig, ax = plt.subplots(figsize=(7, 4.2))
for c, s, _ in runs:
    ret = s["overall_retention"]
    ax.plot(LAGS, [ret[f"lag_{l}"] for l in LAGS], "-o", ms=4, color=col[c], label=lab(c))
ax.set(xscale="log", xlabel="decode lag (steps)", ylabel="retained fraction",
       title="KV indexer retention vs lag (ds4 CPU)")
ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "kv_02_retention.png"), dpi=120); plt.close(fig)

# 3) per-layer adjacent overlap
fig, ax = plt.subplots(figsize=(8, 4.2))
for c, _, sl in runs:
    d = sl.sort_values("layer_id")
    ax.plot(d.layer_id, d.adjacent_overlap_mean, "-o", ms=3, color=col[c], label=lab(c))
ax.set(xlabel="CSA layer index", ylabel="adjacent-step KV overlap",
       title="KV indexer adjacent overlap by layer (ds4 CPU)")
ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "kv_03_adjacent_overlap_by_layer.png"), dpi=120); plt.close(fig)

print(f"wrote 3 KV plots to {out_dir} for contexts: {[lab(c) for c,_,_ in runs]}")
