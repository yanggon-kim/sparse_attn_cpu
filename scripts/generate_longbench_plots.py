#!/usr/bin/env python3
"""Generate LongBench-collection figures (PNG) for the reports.

Usage: generate_longbench_plots.py <out_dir>
Reads  runs/lb_*_q2/analysis/*.json (per-sample) and the RULER sweep summaries
       runs/niah_single_2_*_moe_q2/analysis/metrics_run_summary.json (overlay).
Writes lb_01_kv_vs_context.png, lb_02_moe_by_task.png, lb_03_retention_by_task.png
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "<WORKDIR>/experiment"
LAGS = [1, 2, 4, 8, 16, 32, 64]
TASK_COLOR = {"multi_news": "#1f6feb", "gov_report": "#8250df", "qmsum": "#c55a11"}
GREY, GREEN = "#647083", "#1a7f37"

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(EXP, "analysis_longbench", "plots")
os.makedirs(out_dir, exist_ok=True)

# ---- load LongBench per-sample rows ----
rows = []
for rd in sorted(glob.glob(os.path.join(EXP, "runs", "lb_*_q2"))):
    try:
        s = json.load(open(os.path.join(rd, "prompts", "sample.json")))
        kv = json.load(open(os.path.join(rd, "analysis", "metrics_run_summary.json")))
        moe = json.load(open(os.path.join(rd, "analysis", "moe_metrics_run_summary.json")))
        rows.append({"task": s["task_subtype"], "tok": s["context_length_target"],
                     "adj": kv["overall_adjacent_overlap_mean"],
                     "lift": kv["overall_locality_lift_mean"],
                     "ret": {str(l): kv["overall_retention"][f"lag_{l}"] for l in LAGS},
                     "moe": moe["learned"]["adjacent_overlap_mean"],
                     "moe_hash": moe["hash"]["adjacent_overlap_mean"]})
    except Exception:
        pass

# ---- RULER overlay points ----
ruler = []
for rd in sorted(glob.glob(os.path.join(EXP, "runs", "niah_single_2_*_moe_q2"))):
    try:
        kv = json.load(open(os.path.join(rd, "analysis", "metrics_run_summary.json")))
        g = json.load(open(os.path.join(rd, "outputs", "generations.jsonl")))if False else json.loads(open(os.path.join(rd, "outputs", "generations.jsonl")).read().splitlines()[0])
        ruler.append({"tok": g["prompt_token_count"] or kv["context_length"],
                      "adj": kv["overall_adjacent_overlap_mean"],
                      "lift": kv["overall_locality_lift_mean"],
                      "moe": None})
    except Exception:
        pass
ruler.sort(key=lambda r: r["tok"])

# ---- fig 1: KV adj overlap + lift vs prompt tokens (LongBench scatter + RULER curve) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, key, ylab, title in [
        (axes[0], "adj", "adjacent-step overlap", "KV selection locality vs context (real tasks on the RULER curve)"),
        (axes[1], "lift", "locality lift (× random)", "Locality lift vs context")]:
    for t, c in TASK_COLOR.items():
        xs = [r["tok"] for r in rows if r["task"] == t]
        ys = [r[key] for r in rows if r["task"] == t]
        ax.scatter(xs, ys, s=26, color=c, alpha=0.85, label=f"LongBench {t} (n={len(xs)})", zorder=3)
    ax.plot([r["tok"] for r in ruler], [r[key] for r in ruler], "-o", color=GREY, ms=5,
            label="RULER sweep (n=1/len)", zorder=2)
    ax.set(xscale="log", xlabel="prompt tokens", ylabel=ylab, title=title)
    ax.grid(alpha=.3)
axes[0].set_ylim(0, 1.02)
axes[0].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "lb_01_kv_vs_context.png"), dpi=120); plt.close(fig)

# ---- fig 2: MoE learned adj by task (points + mean±std) vs RULER band ----
fig, ax = plt.subplots(figsize=(7.5, 4.4))
tasks = ["multi_news", "gov_report", "qmsum"]
for i, t in enumerate(tasks):
    ys = [r["moe"] for r in rows if r["task"] == t]
    ax.scatter([i + (j - len(ys)/2) * 0.02 for j in range(len(ys))], ys, s=22, color=TASK_COLOR[t], alpha=0.8)
    m, sd = np.mean(ys), np.std(ys, ddof=1)
    ax.errorbar([i], [m], yerr=[sd], fmt="s", color="#172033", capsize=5, ms=7, zorder=4)
hs = [r["moe_hash"] for r in rows]
ax.axhspan(0.348, 0.376, color=GREY, alpha=0.15, label="RULER sweep range (4K–64K)")
ax.axhline(np.mean(hs), color="#c55a11", ls=":", lw=1.2, label=f"hash layers ({np.mean(hs):.3f})")
ax.axhline(6/256, color=GREY, ls="--", lw=1, label="random 6/256")
ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks)
ax.set(ylabel="MoE learned adjacent overlap", title="MoE routing locality is task- and context-independent")
ax.set_ylim(0, 0.5); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "lb_02_moe_by_task.png"), dpi=120); plt.close(fig)

# ---- fig 3: retention curves per task (mean ± std band) ----
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for t, c in TASK_COLOR.items():
    sub = [r for r in rows if r["task"] == t]
    means = [np.mean([r["ret"][str(l)] for r in sub]) for l in LAGS]
    stds = [np.std([r["ret"][str(l)] for r in sub], ddof=1) for l in LAGS]
    ax.plot(LAGS, means, "-o", ms=4, color=c, label=f"{t} (n={len(sub)})")
    ax.fill_between(LAGS, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)],
                    color=c, alpha=0.15)
ax.set(xscale="log", xlabel="decode lag (steps)", ylabel="retained fraction",
       title="KV retention vs lag — LongBench tasks (mean ± std)")
ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "lb_03_retention_by_task.png"), dpi=120); plt.close(fig)

print(f"wrote 3 plots to {out_dir} from {len(rows)} LongBench + {len(ruler)} RULER runs")
