#!/usr/bin/env python3
"""Generate plots (doc §29). Usage: generate_plots.py <runs_root> <out_dir>
Per-run plots + combined context-length scaling. matplotlib Agg backend.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

runs_root, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
LAGS = [1, 2, 4, 8, 16, 32, 64]
WINDOWS = [1, 2, 4, 8, 16, 32, 64]


def runs():
    for rid in sorted(os.listdir(runs_root)):
        rd = os.path.join(runs_root, rid)
        if os.path.exists(os.path.join(rd, "analysis", "metrics_sample_layer.parquet")):
            yield rid, rd


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, name), dpi=110); plt.close(fig)
    print(f"  {name}")


def per_run_plots(rid, rd):
    sl = pd.read_parquet(os.path.join(rd, "analysis", "metrics_sample_layer.parquet"))
    summ = json.load(open(os.path.join(rd, "analysis", "metrics_run_summary.json")))
    sel = pd.read_parquet(os.path.join(rd, "traces", "selected_kv.parquet"))
    cl = pd.read_parquet(os.path.join(rd, "analysis", "cross_layer_jaccard.parquet"))
    tag = f"{summ['context_length']}"

    # 1 adjacent overlap by layer
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(sl.layer_id.astype(str), sl.adjacent_overlap_mean)
    ax.set(title=f"Adjacent overlap by CSA layer ({tag})", xlabel="layer", ylabel="overlap")
    ax.tick_params(axis="x", rotation=90); save(fig, f"{rid}_01_adjacent_overlap.png")

    # 2 churn by layer
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(sl.layer_id.astype(str), sl.churn_mean, color="tomato")
    ax.set(title=f"Churn by CSA layer ({tag})", xlabel="layer", ylabel="churn"); ax.tick_params(axis="x", rotation=90)
    save(fig, f"{rid}_02_churn.png")

    # 3 retention curve by layer
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, row in sl.iterrows():
        ax.plot(LAGS, [row[f"retention_lag_{l}"] for l in LAGS], alpha=0.4)
    ax.plot(LAGS, [summ["overall_retention"][f"lag_{l}"] for l in LAGS], "k-o", lw=2, label="mean")
    ax.set(title=f"Retention vs lag ({tag})", xlabel="decode lag", ylabel="retained fraction", xscale="log")
    ax.legend(); save(fig, f"{rid}_03_retention.png")

    # 4 working-set growth by layer (mean curve)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(WINDOWS, [summ["overall_working_set_ratio"][f"w{w}"] for w in WINDOWS], "b-o")
    ax.set(title=f"Normalized working-set ratio vs window ({tag})", xlabel="window", ylabel="ws/(w*top_k)", xscale="log")
    save(fig, f"{rid}_04_working_set.png")

    # 5 reuse-distance CDF (approx from per-layer p50/p90/p99 means)
    fig, ax = plt.subplots(figsize=(6, 4))
    pts = [(0.5, sl.reuse_p50.mean()), (0.9, sl.reuse_p90.mean()), (0.99, sl.reuse_p99.mean())]
    ax.plot([p[1] for p in pts], [p[0] for p in pts], "g-o")
    ax.set(title=f"Reuse-distance CDF (approx, {tag})", xlabel="unique reuse distance", ylabel="CDF")
    save(fig, f"{rid}_05_reuse_cdf.png")

    # 6 access-age distribution (recent/middle/old)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["recent", "middle", "old"], [sl.frac_recent.mean(), sl.frac_middle.mean(), sl.frac_old.mean()],
           color=["seagreen", "goldenrod", "slategray"])
    ax.set(title=f"Access-age fractions ({tag})", ylabel="fraction of selections"); save(fig, f"{rid}_06_access_age.png")

    # 7 cross-layer similarity heatmap
    piv = cl.pivot(index="layer_i", columns="layer_j", values="cross_jaccard_mean")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set(title=f"Cross-layer Jaccard ({tag})", xlabel="layer j", ylabel="layer i")
    fig.colorbar(im, ax=ax); save(fig, f"{rid}_07_cross_layer.png")

    # 8 access raster for representative shallow/mid/deep layers
    layers = sorted(sel.layer_id.unique())
    reps = [layers[0], layers[len(layers) // 2], layers[-1]] if layers else []
    fig, axes = plt.subplots(1, len(reps), figsize=(4 * len(reps), 4), squeeze=False)
    for k, lid in enumerate(reps):
        g = sel[sel.layer_id == lid]
        axes[0][k].scatter(g.decode_step, g.compressed_kv_index, s=1, alpha=0.3)
        axes[0][k].set(title=f"L{lid} access raster", xlabel="decode step", ylabel="compressed idx")
    fig.suptitle(f"Decode-step × compressed-index raster ({tag})"); save(fig, f"{rid}_08_raster.png")

    # 9 decode-step × layer overlap heatmap
    tl = pd.read_parquet(os.path.join(rd, "analysis", "metrics_token_layer.parquet"))
    if "adjacent_overlap" in tl:
        piv2 = tl.pivot_table(index="layer_id", columns="decode_step", values="adjacent_overlap")
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(piv2.values, cmap="magma", aspect="auto", vmin=0, vmax=1)
        ax.set(title=f"Adjacent overlap: layer × decode step ({tag})", xlabel="decode step", ylabel="CSA layer idx")
        ax.set_yticks(range(len(piv2.index))); ax.set_yticklabels(piv2.index)
        fig.colorbar(im, ax=ax); save(fig, f"{rid}_09_step_layer_overlap.png")

    # 10 boundary-margin distribution
    if "boundary_margin" in tl and tl.boundary_margin.notna().any():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(tl.boundary_margin.dropna(), bins=40, color="purple")
        ax.set(title=f"Boundary margin distribution ({tag})", xlabel="rank_k - rank_k+1 score", ylabel="count")
        save(fig, f"{rid}_10_boundary_margin.png")


def combined_plots(rs):
    data = []
    for rid, rd in rs:
        s = json.load(open(os.path.join(rd, "analysis", "metrics_run_summary.json")))
        data.append(s)
    data.sort(key=lambda s: s["context_length"])
    if not data:
        return
    cx = [d["context_length"] for d in data]
    # 11 context-length scaling
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cx, [d["overall_adjacent_overlap_mean"] for d in data], "-o", label="adjacent overlap")
    ax.plot(cx, [d["overall_recency_overlap_mean"] for d in data], "-s", label="recency-baseline overlap")
    ax.set(title="Adjacent overlap vs context length", xlabel="context length (tokens)", ylabel="overlap")
    ax.legend(); save(fig, "combined_11_context_scaling.png")
    # 11b locality lift
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cx, [d["overall_locality_lift_mean"] for d in data], "-o", color="darkgreen")
    ax.axhline(1.0, ls="--", c="gray"); ax.set(title="Locality lift vs random vs context length",
                                               xlabel="context length", ylabel="observed / random overlap")
    save(fig, "combined_11b_locality_lift.png")
    # 12 correct vs incorrect
    fig, ax = plt.subplots(figsize=(6, 4))
    cols = ["seagreen" if d["is_correct"] else "tomato" for d in data]
    ax.bar([str(c) for c in cx], [d["overall_adjacent_overlap_mean"] for d in data], color=cols)
    ax.set(title="Adjacent overlap by context (green=correct)", xlabel="context", ylabel="overlap")
    save(fig, "combined_12_correct_vs_incorrect.png")


def main():
    rs = list(runs())
    if not rs:
        print("no completed runs yet"); return
    for rid, rd in rs:
        per_run_plots(rid, rd)
    combined_plots(rs)
    # 13 task comparison placeholder (single task this pass)
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.text(0.5, 0.5, "Task comparison: only niah_single_2 this pass\n(vt / cwe deferred to expansion)",
            ha="center", va="center"); ax.axis("off")
    save(fig, "combined_13_task_comparison.png")
    print(f"plots -> {out_dir}")


if __name__ == "__main__":
    main()
