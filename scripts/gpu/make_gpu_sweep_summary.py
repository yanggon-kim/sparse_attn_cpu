#!/usr/bin/env python3
"""Write docs/<sweep>/<name>_summary.md + context-scaling plots from sweep_<tag>.json files (exp1 §6 / exp2 §5).
Usage: make_gpu_sweep_summary.py --out docs/gpu_sweep --title "..." --sweeps docs/gpu_sweep/sweep_v32.json [more...]
       [--labels "DeepSeek-V3.2" ...] [--md gpu_sweep_summary.md] [--png-prefix gpu]
Plots: <prefix>_01_context_scaling.png (adjacent overlap + lift vs context, one line per sweep, CPU V4 dotted),
       <prefix>_02_retention.png (retention vs lag per rung, first sweep), <prefix>_03_hotset.png (A@99 and top-10 %
       coverage vs context), <prefix>_04_moe.png (MoE adjacent overlap / lift vs context).
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--title", required=True)
ap.add_argument("--sweeps", nargs="+", required=True)
ap.add_argument("--labels", nargs="*", default=None)
ap.add_argument("--md", default="gpu_sweep_summary.md")
ap.add_argument("--png-prefix", default="gpu")
a = ap.parse_args()
sweeps = [json.load(open(p)) for p in a.sweeps]
labels = a.labels or [s["model"] + ("" if s.get("variant") == "all_layers" else f" ({s.get('variant')})") for s in sweeps]
CPU = sweeps[0].get("cpu_v4", {})
COLORS = ["#1f6feb", "#c55a11", "#1a7f37", "#8250df", "#b54708"]


def agg_rows(s, table, kind="all"):
    return sorted([r for r in s[table] if r["kind"] == kind], key=lambda r: r["rung"])


def fmt(v, d=3):
    return "—" if v is None or v != v else f"{v:.{d}f}"


# ---- plot 1: context scaling ----
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for i, (s, lab) in enumerate(zip(sweeps, labels)):
    rows = agg_rows(s, "R1")
    x = [r["rung"] for r in rows]
    for ax, key, ttl in ((axs[0], "adjacent_overlap", "adjacent-step overlap"), (axs[1], "lift_vs_random", "locality lift over random")):
        y = [r[key] for r in rows]
        lo = [r[key] - (r.get(key + "_ci_lo") or r[key]) for r in rows]
        hi = [(r.get(key + "_ci_hi") or r[key]) - r[key] for r in rows]
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", color=COLORS[i % len(COLORS)], label=lab, capsize=3)
        ax.set_title(ttl); ax.set_xscale("log", base=2); ax.set_xlabel("context (tokens)")
if CPU:
    xs = sorted(int(k) for k in CPU)
    axs[0].plot(xs, [CPU[str(k)][0] if str(k) in CPU else CPU[k][0] for k in xs], "k:", label="DeepSeek-V4-Flash CPU (k=512)")
    axs[1].plot(xs, [CPU[str(k)][1] if str(k) in CPU else CPU[k][1] for k in xs], "k:", label="DeepSeek-V4-Flash CPU (k=512)")
axs[1].set_yscale("log")
for ax in axs:
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle(a.title); fig.tight_layout(); fig.savefig(os.path.join(a.out, f"{a.png_prefix}_01_context_scaling.png"), dpi=150)

# ---- plot 2: retention vs lag (first sweep, ld runs) ----
fig, ax = plt.subplots(figsize=(6, 4))
for i, r in enumerate(agg_rows(sweeps[0], "R1", "ld")):
    lags = [1, 8, 64, 512, 1024]
    vals = [r.get("retention_lag1"), r.get("retention_lag8"), r.get("retention_lag64"), r.get("retention_lag512"), r.get("retention_lag1024")]
    pts = [(l, v) for l, v in zip(lags, vals) if v is not None and v == v]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=COLORS[i % len(COLORS)], label=f"{r['rung']//1024}K (ld, {r['task']})")
ax.set_xscale("log", base=2); ax.set_xlabel("lag (decode steps)"); ax.set_ylabel("retention"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title(f"{labels[0]}: retention vs lag (2048-step decodes)"); fig.tight_layout(); fig.savefig(os.path.join(a.out, f"{a.png_prefix}_02_retention.png"), dpi=150)

# ---- plot 3: hot-set ----
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for i, (s, lab) in enumerate(zip(sweeps, labels)):
    rows = agg_rows(s, "R3")
    x = [r["rung"] for r in rows]
    axs[0].plot(x, [r["hotset_A99_pct"] for r in rows], marker="o", color=COLORS[i % len(COLORS)], label=lab)
    axs[1].plot(x, [r["hotset_top10pct_coverage"] for r in rows], marker="o", color=COLORS[i % len(COLORS)], label=lab)
CPU_A99 = {4096: 79.5, 8192: 65.7, 16384: 47.5, 32768: 31.4, 65536: 20.4}
axs[0].plot(sorted(CPU_A99), [CPU_A99[k] for k in sorted(CPU_A99)], "k:", label="V4-Flash CPU")
axs[0].set_title("hot set A@99 (% of pool)"); axs[1].set_title("coverage of the hottest 10 % of the pool")
for ax in axs:
    ax.set_xscale("log", base=2); ax.set_xlabel("context (tokens)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(a.out, f"{a.png_prefix}_03_hotset.png"), dpi=150)

# ---- plot 4: MoE ----
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for i, (s, lab) in enumerate(zip(sweeps, labels)):
    rows = agg_rows(s, "R2")
    x = [r["rung"] for r in rows]
    axs[0].plot(x, [r["moe_adjacent_overlap"] for r in rows], marker="o", color=COLORS[i % len(COLORS)], label=lab)
    axs[1].plot(x, [r["moe_lift"] for r in rows], marker="o", color=COLORS[i % len(COLORS)], label=lab)
axs[0].set_title("MoE routed-expert adjacent overlap (8 of 256)"); axs[1].set_title("MoE lift over random (8/256)")
axs[1].set_ylim(bottom=0)
for ax in axs:
    ax.set_xscale("log", base=2); ax.set_xlabel("context (tokens)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(a.out, f"{a.png_prefix}_04_moe.png"), dpi=150)

# ---- summary md ----
L = [f"# {a.title}", "", f"Generated by `scripts/gpu/make_gpu_sweep_summary.py` from {', '.join(os.path.basename(p) for p in a.sweeps)}.",
     "Every number is the mean over runs of the rung (bf + ld unless noted), with 95 % bootstrap CI over runs; run ids are in the CSVs.", ""]
for s, lab in zip(sweeps, labels):
    L += [f"## {lab}", "", "### R1 — KV top-k locality (all runs of the rung)", "",
          "| context | n | adj. overlap [CI] | lift vs random [CI] | recency baseline | ret@8 | ret@64 | ret@512 (ld) | ret@1024 (ld) | CPU V4 overlap / lift |",
          "|---:|---:|---|---|---:|---:|---:|---:|---:|---|"]
    ld = {r["rung"]: r for r in agg_rows(s, "R1", "ld")}
    for r in agg_rows(s, "R1"):
        l_ = ld.get(r["rung"], {})
        cpu = f"{r['cpu_v4_adjacent_overlap']:.3f} / {r['cpu_v4_lift']:.1f}×" if r.get("cpu_v4_adjacent_overlap") else "—"
        L.append(f"| {r['rung']//1024}K | {r['task'][2:]} | {fmt(r['adjacent_overlap'])} [{fmt(r.get('adjacent_overlap_ci_lo'))}, {fmt(r.get('adjacent_overlap_ci_hi'))}] "
                 f"| {fmt(r['lift_vs_random'],2)}× [{fmt(r.get('lift_vs_random_ci_lo'),2)}, {fmt(r.get('lift_vs_random_ci_hi'),2)}] | {fmt(r['recency_baseline_overlap'])} "
                 f"| {fmt(r['retention_lag8'])} | {fmt(r['retention_lag64'])} | {fmt(l_.get('retention_lag512'))} | {fmt(l_.get('retention_lag1024'))} | {cpu} |")
    L += ["", "### R2 — MoE routed-expert locality", "", "| context | n | adj. overlap [CI] | lift vs 8/256 [CI] |", "|---:|---:|---|---|"]
    for r in agg_rows(s, "R2"):
        L.append(f"| {r['rung']//1024}K | {r['task'][2:]} | {fmt(r['moe_adjacent_overlap'])} [{fmt(r.get('moe_adjacent_overlap_ci_lo'))}, {fmt(r.get('moe_adjacent_overlap_ci_hi'))}] "
                 f"| {fmt(r['moe_lift'],1)}× [{fmt(r.get('moe_lift_ci_lo'),1)}, {fmt(r.get('moe_lift_ci_hi'),1)}] |")
    L += ["", "### R3 — hot-set coverage", "", "| context | n | pool N | A@99 (% of pool) [CI] | B@99 | top-1 % cov | top-5 % cov | top-10 % cov (MEASURED_TOP10) |", "|---:|---:|---:|---|---:|---:|---:|---:|"]
    for r in agg_rows(s, "R3"):
        L.append(f"| {r['rung']//1024}K | {r['task'][2:]} | {fmt(r['hotset_pool_N'],0)} | {fmt(r['hotset_A99_pct'],1)} [{fmt(r.get('hotset_A99_pct_ci_lo'),1)}, {fmt(r.get('hotset_A99_pct_ci_hi'),1)}] "
                 f"| {fmt(r['hotset_B99_pct'],1)} | {fmt(r['hotset_top1pct_coverage'])} | {fmt(r['hotset_top5pct_coverage'])} | {fmt(r['hotset_top10pct_coverage'])} |")
    L += ["", "### Accuracy sanity (bf runs, our scorer; no official numbers for these sub-samples)", "", "| context | source | n | accuracy | mean score |", "|---:|---|---:|---:|---:|"]
    for r in s["accuracy"]:
        L.append(f"| {r['rung']//1024}K | {r['source']} | {r['n']} | {fmt(r['accuracy'],2)} | {fmt(r['mean_score'],3)} |")
    L.append("")
L += [f"![context scaling]({a.png_prefix}_01_context_scaling.png)", f"![retention]({a.png_prefix}_02_retention.png)",
      f"![hot set]({a.png_prefix}_03_hotset.png)", f"![moe]({a.png_prefix}_04_moe.png)", ""]
open(os.path.join(a.out, a.md), "w").write("\n".join(L))
print("wrote", os.path.join(a.out, a.md), "and 4 plots")
