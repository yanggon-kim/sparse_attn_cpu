#!/usr/bin/env python3
"""Hot-set coverage of the CSA lightning-indexer top-k KV selection.

Usage: analyze_hotset_coverage.py <out_dir> <run_dir> [<run_dir> ...]
Reads  run_dir/traces/selected_kv.parquet (cols layer_id, decode_step, compressed_kv_index)
       and run_dir/traces/score_summaries.parquet (n_candidates_visible per step x layer).
Writes run_dir/analysis/hotset_coverage.json (per-layer + per-run summary), and two combined
plots into <out_dir>: hotset_01_coverage_curve.png, hotset_02_size_by_context.png.

Question: of all candidate compressed-KV positions, how many must be held "hot" to cover a target
fraction of the top-k selection over a decode? Two definitions, per CSA layer, then averaged:

  Metric A (access-weighted / mean hit-rate): rank positions by selection frequency; H_a = smallest
    prefix whose summed frequency >= a * (total selection events). "A hot set of the H_a most-selected
    positions serves fraction a of all selection accesses." Reported for a in {.90,.95,.99,.999}.
  Metric B (per-step tail containment): for the freq-ranked hot set of size n, per-step containment
    c_t(n) = |topk_t ∩ hot_n| / |topk_t|; smallest n such that the 1st-percentile step (>=99% of steps)
    has c_t >= 0.99. "Cache misses >1% of a step's selection on at most 1% of steps." B >= A.

Denominator N (the "whole candidates") = max n_candidates_visible for the layer (final pool size).
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHAS = [0.90, 0.95, 0.99, 0.999]
POOL_PCT_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
PALETTE = ["#1f6feb", "#8250df", "#c55a11", "#1a7f37", "#b54708"]


def per_layer_coverage(sel_layer, pool_N):
    """sel_layer: DataFrame rows for one layer (decode_step, compressed_kv_index). Returns metrics dict."""
    steps = sel_layer.groupby("decode_step")["compressed_kv_index"].apply(lambda s: s.to_numpy())
    step_sets = [set(a.tolist()) for a in steps.to_numpy()]
    n_steps = len(step_sets)
    # selection frequency per position
    freq = sel_layer["compressed_kv_index"].value_counts()          # position -> #steps selected
    counts = np.sort(freq.to_numpy())[::-1].astype(np.int64)         # desc
    T = int(counts.sum())                                           # total selection events
    order = freq.sort_values(ascending=False).index.to_numpy()      # positions, most-frequent first
    cum = np.cumsum(counts) / T
    # Metric A: smallest prefix reaching alpha coverage
    A = {}
    for a in ALPHAS:
        H = int(np.searchsorted(cum, a) + 1)                        # 1-based count
        A[a] = {"H": H, "pct_of_pool": 100.0 * H / pool_N}
    distinct = len(counts)                                          # H at 100% coverage
    # Metric B: per-step p1 containment >= 0.99 as hot set grows (freq-ranked)
    # Precompute, for each step, the rank (in freq order) of each selected position; then containment at
    # size n = fraction of the step's positions whose rank < n. p1 over steps must reach 0.99.
    rank_of = {p: i for i, p in enumerate(order.tolist())}          # position -> 0-based freq rank
    # for each step: sorted ranks of its selected positions
    step_ranks = [np.sort([rank_of[p] for p in s]) for s in step_sets]
    step_k = np.array([len(s) for s in step_sets])
    def p1_containment(n):
        # containment_t = (#selected positions with rank < n) / k_t  ; return 1st percentile over steps
        c = np.array([np.searchsorted(sr, n) / k for sr, k in zip(step_ranks, step_k)])
        return np.percentile(c, 1)                                  # 99% of steps do at least this well
    # binary search smallest n in [1, distinct] with p1_containment(n) >= 0.99
    lo, hi = 1, distinct
    if p1_containment(distinct) < 0.99:
        B_n = distinct                                             # even full union can't (shouldn't happen)
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if p1_containment(mid) >= 0.99:
                hi = mid
            else:
                lo = mid + 1
        B_n = lo
    # Forward sweep: coverage achieved by a hot set of a fixed % of the pool
    cov_by_pool_pct = {}
    for p in POOL_PCT_GRID:
        n = int(round(p / 100.0 * pool_N))
        if n < 1:
            cov_by_pool_pct[p] = 0.0
        elif n >= distinct:
            cov_by_pool_pct[p] = 1.0
        else:
            cov_by_pool_pct[p] = float(cum[n - 1])
    return {
        "n_steps": n_steps, "pool_N": int(pool_N), "distinct_selected": distinct,
        "distinct_pct_of_pool": 100.0 * distinct / pool_N,
        "A": A, "B_n": int(B_n), "B_pct_of_pool": 100.0 * B_n / pool_N,
        "cov_by_pool_pct": cov_by_pool_pct,
        # coverage curve sampled at fixed % grid for plotting (x = % of pool, y = coverage)
        "curve_x_pct": [100.0 * (i + 1) / pool_N for i in range(0, distinct, max(1, distinct // 200))],
        "curve_y_cov": [float(cum[i]) for i in range(0, distinct, max(1, distinct // 200))],
    }


def analyze_run(run_dir):
    TR = os.path.join(run_dir, "traces")
    sel = pd.read_parquet(os.path.join(TR, "selected_kv.parquet"),
                          columns=["layer_id", "decode_step", "compressed_kv_index"])
    ss = pd.read_parquet(os.path.join(TR, "score_summaries.parquet"),
                         columns=["layer_id", "decode_step", "n_candidates_visible"])
    pool = ss.groupby("layer_id")["n_candidates_visible"].max()
    ctx = None
    try:
        ctx = int(json.load(open(os.path.join(run_dir, "run_manifest.json")))["context_length_target"])
    except Exception:
        pass
    per_layer = {}
    for lid, g in sel.groupby("layer_id"):
        per_layer[int(lid)] = per_layer_coverage(g, int(pool.loc[lid]))
    layers = sorted(per_layer)
    def mean_pct(path):
        vals = [ _dig(per_layer[l], path) for l in layers ]
        return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))
    summ = {
        "run_id": os.path.basename(run_dir), "context_length": ctx, "n_csa_layers": len(layers),
        "pool_N_mean": float(np.mean([per_layer[l]["pool_N"] for l in layers])),
        "A99_pct_mean": mean_pct(("A", 0.99, "pct_of_pool"))[0],
        "A_pct_by_alpha": {str(a): mean_pct(("A", a, "pct_of_pool"))[0] for a in ALPHAS},
        "A99_pct_range": mean_pct(("A", 0.99, "pct_of_pool"))[1:],
        "B99_pct_mean": mean_pct(("B_pct_of_pool",))[0],
        "B99_pct_range": mean_pct(("B_pct_of_pool",))[1:],
        "distinct_pct_mean": mean_pct(("distinct_pct_of_pool",))[0],
        "coverage_by_pool_pct": {
            str(p): {"mean": mean_pct(("cov_by_pool_pct", p))[0],
                     "min": mean_pct(("cov_by_pool_pct", p))[1],
                     "max": mean_pct(("cov_by_pool_pct", p))[2]}
            for p in POOL_PCT_GRID
        },
        "per_layer": per_layer,
    }
    json.dump(summ, open(os.path.join(run_dir, "analysis", "hotset_coverage.json"), "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else o)
    return summ


def _dig(d, path):
    for k in path:
        d = d[k]
    return d


def main():
    out_dir = sys.argv[1]
    runs = sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for rd in runs:
        s = analyze_run(rd)
        results.append(s)
        print(f"{s['run_id']}: ctx={s['context_length']} pool~{s['pool_N_mean']:.0f}  "
              f"A@99={s['A99_pct_mean']:.1f}% (range {s['A99_pct_range'][0]:.1f}-{s['A99_pct_range'][1]:.1f})  "
              f"A@99.9={s['A_pct_by_alpha']['0.999']:.1f}%  B(p99/99%)={s['B99_pct_mean']:.1f}%  "
              f"distinct={s['distinct_pct_mean']:.1f}%")
    results.sort(key=lambda s: s["context_length"] or 0)
    lab = lambda c: f"{c//1024}K" if c else "?"

    # Combined forward-sweep table: coverage (% of top-k accesses) at each hot-set pool-% budget
    cols = [lab(s["context_length"]) for s in results]
    print("\n=== coverage (% of top-k accesses) vs hot-set budget (% of candidate pool), mean over CSA layers ===")
    print("pool% | " + " | ".join(f"{c:>6}" for c in cols))
    for p in POOL_PCT_GRID:
        row = [100.0 * s["coverage_by_pool_pct"][str(p)]["mean"] for s in results]
        print(f"{p:>4}% | " + " | ".join(f"{v:6.1f}" for v in row))

    # Plot 1: coverage curve per context (use the median-layer curve via a representative layer = middle CSA)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, s in enumerate(results):
        layers = sorted(int(x) for x in s["per_layer"])
        rep = layers[len(layers) // 2]                              # middle CSA layer as representative
        pl = s["per_layer"][str(rep)] if str(rep) in s["per_layer"] else s["per_layer"][rep]
        ax.plot(pl["curve_x_pct"], pl["curve_y_cov"], "-", color=PALETTE[i % len(PALETTE)],
                label=f"{lab(s['context_length'])} (A@99={s['A99_pct_mean']:.0f}%)")
    ax.axhline(0.99, ls="--", color="#647083", lw=1, label="99% coverage")
    ax.set(xlabel="hot set size (% of candidate pool)", ylabel="cumulative top-k coverage (access-weighted)",
           title="Hot-set coverage of the top-512 KV selection (ds4 CPU)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "hotset_01_coverage_curve.png"), dpi=120); plt.close(fig)

    # Plot 2: hot-set size (% of pool) by context — Metric A@99, A@99.9, Metric B
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs = np.arange(len(results)); w = 0.26
    a99 = [s["A99_pct_mean"] for s in results]
    a999 = [s["A_pct_by_alpha"]["0.999"] for s in results]
    b = [s["B99_pct_mean"] for s in results]
    ax.bar(xs - w, a99, w, color="#1f6feb", label="A: 99% of accesses")
    ax.bar(xs, a999, w, color="#8250df", label="A: 99.9% of accesses")
    ax.bar(xs + w, b, w, color="#c55a11", label="B: 99% contained on 99% of steps")
    ax.set_xticks(xs); ax.set_xticklabels([lab(s["context_length"]) for s in results])
    ax.set(ylabel="hot set size (% of candidate pool)",
           title="How big a hot set to cover 99% of the top-k selection")
    ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
    for xi, v in zip(xs - w, a99): ax.text(xi, v + 0.4, f"{v:.0f}", ha="center", fontsize=7)
    for xi, v in zip(xs + w, b): ax.text(xi, v + 0.4, f"{v:.0f}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "hotset_02_size_by_context.png"), dpi=120); plt.close(fig)
    print(f"wrote 2 plots to {out_dir}")


if __name__ == "__main__":
    main()
