#!/usr/bin/env python3
"""Vectorized drop-in for scripts/extended_retention.py (same inputs, same extended_retention.json / .png).
Usage: extended_retention_fast.py <run_dir>
Definitions kept identical: retention[lag] = mean over (layer, step) of |S_t ∩ S_{t-lag}| / |S_t|;
working_set_ratio[w] = mean over (layer, step) of |∪ window| / (w · max|S| in window), windows truncated at the start.
The original is O(steps · w · k) per window and stalls at 2048 steps; this one uses per-layer membership matrices.
"""
import json, os, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run_dir = sys.argv[1]
LAGS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
WINS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
sel = pd.read_parquet(os.path.join(run_dir, "traces", "selected_kv.parquet"), columns=["layer_id", "decode_step", "compressed_kv_index"])
layers = sorted(sel.layer_id.unique().tolist())
steps = sorted(sel.decode_step.unique().tolist())
step_pos = {s: i for i, s in enumerate(steps)}
n = len(steps)
ret_sum = {lag: [0.0, 0] for lag in LAGS}
ws_sum = {w: [0.0, 0] for w in WINS}
lid = sel.layer_id.values; st = sel.decode_step.values; ix = sel.compressed_kv_index.values
for l in layers:
    m = lid == l
    s_l, i_l = st[m], ix[m]
    pool, inv = np.unique(i_l, return_inverse=True)
    rows = np.array([step_pos[s] for s in s_l]) if len(s_l) else np.zeros(0, dtype=int)
    M = np.zeros((n, len(pool)), dtype=bool)
    M[rows, inv] = True
    sizes = M.sum(1)
    present = sizes > 0
    # retention: step t vs t-lag (only when both exist; steps are contiguous decode steps here)
    for lag in LAGS:
        if lag >= n:
            continue
        a, b = M[lag:], M[:-lag]
        ok = present[lag:] & present[:-lag]
        if ok.any():
            inter = (a[ok] & b[ok]).sum(1)
            ret_sum[lag][0] += float((inter / sizes[lag:][ok]).sum())
            ret_sum[lag][1] += int(ok.sum())
    # working set: distinct columns seen in the last w rows (truncated at start) = columns whose last-seen row >= i-w+1
    last = np.full(len(pool), -1, dtype=np.int32)
    last_hist = np.empty((n, len(pool)), dtype=np.int32)
    for i in range(n):
        last[M[i]] = i
        last_hist[i] = last
    for w in WINS:
        cnt = np.zeros(n, dtype=np.int64)
        for i in range(n):
            cnt[i] = int((last_hist[i] >= max(0, i - w + 1)).sum())
        # k = max set size in the window (original: max over the window's sets)
        kmax = np.array([sizes[max(0, i - w + 1): i + 1].max() for i in range(n)])
        ok = kmax > 0
        ws_sum[w][0] += float((cnt[ok] / (w * kmax[ok])).sum())
        ws_sum[w][1] += int(ok.sum())
retention = {lag: (ret_sum[lag][0] / ret_sum[lag][1]) if ret_sum[lag][1] else float("nan") for lag in LAGS}
ws = {w: (ws_sum[w][0] / ws_sum[w][1]) if ws_sum[w][1] else float("nan") for w in WINS}
out = {"n_steps": n, "n_layers": len(layers), "retention": retention, "working_set_ratio": ws}
os.makedirs(os.path.join(run_dir, "analysis"), exist_ok=True)
json.dump(out, open(os.path.join(run_dir, "analysis", "extended_retention.json"), "w"), indent=2)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(LAGS, [retention[l] for l in LAGS], "o-", color="#2f5496")
ax[0].set(xscale="log", xlabel="decode lag (steps)", ylabel="retained fraction", title=f"Long-horizon retention ({n} steps)")
ax[0].grid(alpha=.3)
ax[1].plot(WINS, [ws[w] for w in WINS], "s-", color="#c55a11")
ax[1].set(xscale="log", xlabel="decode window (steps)", ylabel="working set / (w·top_k)", title="Working-set growth")
ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(run_dir, "analysis", "extended_retention.png"), dpi=120)
print("retention:", {l: round(retention[l], 3) for l in LAGS})
print("working_set:", {w: round(ws[w], 3) for w in WINS})
