#!/usr/bin/env python3
"""Generate MoE expert-selection locality plots (PNG) for the report.

Usage: generate_moe_plots.py <out_dir> <run_dir> [<run_dir> ...]
Each run_dir must have analysis/moe_metrics_sample_layer.parquet and
analysis/moe_metrics_run_summary.json (produced by analyze_moe_locality.py).
Writes combined PNGs (across the given context lengths) into <out_dir>.
"""
import json, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAGS = [1, 2, 4, 8, 16, 32, 64]
WINDOWS = [1, 2, 4, 8, 16, 32, 64]
BLUE, ORANGE, GREEN, GREY = "#1f6feb", "#c55a11", "#1a7f37", "#647083"
COLORS = ["#1f6feb", "#8250df", "#c55a11", "#1a7f37", "#b54708"]

out_dir = sys.argv[1]
run_dirs = sys.argv[2:]
os.makedirs(out_dir, exist_ok=True)


def load(run_dir):
    sl = pd.read_parquet(os.path.join(run_dir, "analysis", "moe_metrics_sample_layer.parquet"))
    summ = json.load(open(os.path.join(run_dir, "analysis", "moe_metrics_run_summary.json")))
    return sl, summ


runs = []
for rd in run_dirs:
    try:
        sl, summ = load(rd)
        runs.append((summ.get("context_length") or os.path.basename(rd), sl, summ))
    except Exception as e:
        print(f"skip {rd}: {e}", file=sys.stderr)
runs.sort(key=lambda x: (x[0] if isinstance(x[0], (int, float)) else 1e9))


def ctx_label(c):
    return f"{int(c)//1024}K" if isinstance(c, (int, float)) else str(c)


# 1) adjacent overlap by layer (learned solid, hash markers) — one line per context
fig, ax = plt.subplots(figsize=(8, 4.2))
rand = None
for i, (c, sl, summ) in enumerate(runs):
    rand = summ.get("random_baseline_overlap", 0.0234)
    lrn = sl[~sl.is_hash_layer].sort_values("layer_id")
    hsh = sl[sl.is_hash_layer].sort_values("layer_id")
    ax.plot(lrn.layer_id, lrn.adjacent_overlap_mean, "-o", ms=3, color=COLORS[i % len(COLORS)],
            label=f"{ctx_label(c)} (learned)")
    ax.plot(hsh.layer_id, hsh.adjacent_overlap_mean, "x", ms=7, color=COLORS[i % len(COLORS)])
if rand is not None:
    ax.axhline(rand, ls="--", color=GREY, lw=1, label=f"random (6/256={rand:.3f})")
ax.set(xlabel="layer index", ylabel="adjacent-step expert overlap",
       title="MoE routed-expert selection: adjacent-step overlap by layer")
ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "moe_01_adjacent_overlap_by_layer.png"), dpi=120)
plt.close(fig)

# 2) retention curve (learned) — one line per context
fig, ax = plt.subplots(figsize=(7, 4.2))
for i, (c, sl, summ) in enumerate(runs):
    ret = summ.get("learned", {}).get("retention", {})
    ys = [ret.get(f"lag_{l}") for l in LAGS]
    ax.plot(LAGS, ys, "-o", ms=4, color=COLORS[i % len(COLORS)], label=ctx_label(c))
if runs:
    ax.axhline(runs[0][2].get("random_baseline_overlap", 0.0234), ls="--", color=GREY, lw=1, label="random")
ax.set(xscale="log", xlabel="decode lag (steps)", ylabel="retained expert fraction",
       title="MoE expert-selection retention vs lag (learned layers)")
ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "moe_02_retention.png"), dpi=120)
plt.close(fig)

# 3) working-set growth (learned)
fig, ax = plt.subplots(figsize=(7, 4.2))
for i, (c, sl, summ) in enumerate(runs):
    ws = summ.get("learned", {}).get("working_set_ratio", {})
    ys = [ws.get(f"w{w}") for w in WINDOWS]
    ax.plot(WINDOWS, ys, "-s", ms=4, color=COLORS[i % len(COLORS)], label=ctx_label(c))
ax.set(xscale="log", xlabel="decode window (steps)", ylabel="unique experts / (w·6)",
       title="MoE working-set growth (learned layers)")
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "moe_03_working_set.png"), dpi=120)
plt.close(fig)

# 4) hash vs learned adjacent overlap, per context
fig, ax = plt.subplots(figsize=(7, 4.2))
labels = [ctx_label(c) for c, _, _ in runs]
lrn = [s.get("learned", {}).get("adjacent_overlap_mean", float("nan")) for _, _, s in runs]
hsh = [s.get("hash", {}).get("adjacent_overlap_mean", float("nan")) for _, _, s in runs]
x = range(len(labels)); w = 0.38
ax.bar([xi - w / 2 for xi in x], lrn, w, color=BLUE, label="learned top-k")
ax.bar([xi + w / 2 for xi in x], hsh, w, color=ORANGE, label="hash-routed (first 3)")
if runs:
    ax.axhline(runs[0][2].get("random_baseline_overlap", 0.0234), ls="--", color=GREY, lw=1, label="random 6/256")
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set(ylabel="adjacent-step expert overlap",
       title="Learned routing is temporally local; hash routing is not")
ax.set_ylim(0, 1); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "moe_04_hash_vs_learned.png"), dpi=120)
plt.close(fig)

# 5) context scaling: learned adjacent overlap + lift
fig, ax = plt.subplots(figsize=(7, 4.2))
xs = [ctx_label(c) for c, _, _ in runs]
ov = [s.get("learned", {}).get("adjacent_overlap_mean", float("nan")) for _, _, s in runs]
lift = [s.get("learned", {}).get("locality_lift_mean", float("nan")) for _, _, s in runs]
ax.plot(xs, ov, "-o", color=BLUE, label="adjacent overlap (learned)")
ax.set(xlabel="context length", ylabel="adjacent overlap", title="MoE locality vs context length")
ax.set_ylim(0, 1); ax.grid(alpha=.3)
ax2 = ax.twinx()
ax2.plot(xs, lift, "-s", color=GREEN, label="locality lift ×")
ax2.set_ylabel("locality lift (× random)")
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "moe_05_context_scaling.png"), dpi=120)
plt.close(fig)

print(f"wrote 5 MoE plots to {out_dir} for contexts: {[ctx_label(c) for c,_,_ in runs]}")
