#!/usr/bin/env python3
"""Vectorized drop-in for scripts/analyze_locality.py (same inputs, same outputs, same metric definitions):
metrics_token_layer.parquet, metrics_sample_layer.parquet, cross_layer_jaccard.parquet, metrics_run_summary.json.
Usage: analyze_locality_fast.py <run_dir>
Needed because the original's reuse-distance loop is O(stream x window) (hours for 2K-step GPU runs).
Verified equal (to float tolerance) with the original on the exp1 smoke runs — see docs/00_doc/reports/exp1_smoke.
Notes: score-based fields (boundary_margin, score_pearson, mean_abs_dscore) are NaN when no scores were traced;
weighted overlap uses the rank order (= ascending index when scores are absent, as in ranks_from_scores fallback).
"""
import json, os, statistics, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from locality_lib import percentile, random_expected_overlap_fraction, representative_original_pos  # noqa: E402

LAGS = [1, 2, 4, 8, 16, 32, 64]
WINDOWS = [1, 2, 4, 8, 16, 32, 64]
run_dir = sys.argv[1]
TR = os.path.join(run_dir, "traces")
AN = os.path.join(run_dir, "analysis")
os.makedirs(AN, exist_ok=True)


def weighted_overlap_np(cur_order, prev_order):
    """locality_lib.weighted_overlap: sum over shared indices of min(w_cur, w_prev) / sum(w_cur), with
    w(rank) = rank_weight(rank) (= 1/log2(rank+2)); ranks are positions in the ranked lists."""
    if len(cur_order) == 0:
        return float("nan")
    wc = 1.0 / np.log2(np.arange(len(cur_order)) + 2.0)
    wp = 1.0 / np.log2(np.arange(len(prev_order)) + 2.0)
    # position of each cur index inside prev_order (prev_order may be any order: use a sort index)
    ps = np.argsort(prev_order, kind="stable")
    prev_sorted = prev_order[ps]
    loc = np.searchsorted(prev_sorted, cur_order)
    loc_c = np.minimum(loc, len(prev_sorted) - 1)
    shared = prev_sorted[loc_c] == cur_order
    prev_rank = ps[loc_c[shared]]
    num = np.minimum(wc[shared], wp[prev_rank]).sum()
    return float(num / wc.sum())


def _reuse_py(stream_arr):
    n = len(stream_arr)
    tree = np.zeros(n + 1, dtype=np.int64)
    out = np.empty(n, dtype=np.int64)
    m = 0
    cold = 0
    last = {}
    for pos in range(n):
        blk = int(stream_arr[pos])
        lp = last.get(blk, -1)
        if lp < 0:
            cold += 1
        else:
            # prefix(pos) - prefix(lp+1)
            s1 = 0; i = pos
            while i > 0:
                s1 += tree[i]; i -= i & (-i)
            s2 = 0; i = lp + 1
            while i > 0:
                s2 += tree[i]; i -= i & (-i)
            out[m] = s1 - s2; m += 1
            i = lp + 1
            while i <= n:
                tree[i] -= 1; i += i & (-i)
        i = pos + 1
        while i <= n:
            tree[i] += 1; i += i & (-i)
        last[blk] = pos
    return out[:m], cold


try:
    import numba

    @numba.njit(cache=True)
    def _reuse_nb(stream_arr, last_arr):
        n = stream_arr.shape[0]
        tree = np.zeros(n + 1, dtype=np.int64)
        out = np.empty(n, dtype=np.int64)
        m = 0
        cold = 0
        for pos in range(n):
            blk = stream_arr[pos]
            lp = last_arr[blk]
            if lp < 0:
                cold += 1
            else:
                s1 = 0; i = pos
                while i > 0:
                    s1 += tree[i]; i -= i & (-i)
                s2 = 0; i = lp + 1
                while i > 0:
                    s2 += tree[i]; i -= i & (-i)
                out[m] = s1 - s2; m += 1
                i = lp + 1
                while i <= n:
                    tree[i] -= 1; i += i & (-i)
            i = pos + 1
            while i <= n:
                tree[i] += 1; i += i & (-i)
            last_arr[blk] = pos
        return out[:m], cold
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


def unique_reuse_distances(stream):
    """For each access with a previous occurrence: number of DISTINCT blocks accessed strictly between the
    previous occurrence and now (Mattson stack distance). O(N log N) with a Fenwick tree over last-access times;
    numba-compiled when available (pure-Python fallback is ~50x slower)."""
    arr = np.asarray(stream, dtype=np.int64)
    if len(arr) == 0:
        return np.zeros(0, dtype=np.int64), 0
    if _HAVE_NUMBA:
        last_arr = np.full(int(arr.max()) + 1, -1, dtype=np.int64)
        out, cold = _reuse_nb(arr, last_arr)
        return out, int(cold)
    out, cold = _reuse_py(arr)
    return out, int(cold)


def main():
    sel = pd.read_parquet(os.path.join(TR, "selected_kv.parquet"),
                          columns=["layer_id", "decode_step", "absolute_position", "selected_rank",
                                   "compressed_kv_index", "index_score", "compression_ratio"])
    ss = pd.read_parquet(os.path.join(TR, "score_summaries.parquet"))
    gen = json.loads(open(os.path.join(run_dir, "outputs", "generations.jsonl")).read().splitlines()[0])
    manifest = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    mc = json.load(open(os.path.join(run_dir, "model_config.json")))
    top_k = mc["sparse_top_k"]
    ctx_len = manifest.get("context_length_target")
    is_correct = gen.get("is_correct")

    sel = sel.sort_values(["layer_id", "decode_step", "selected_rank"], kind="stable")
    layers = sorted(sel.layer_id.unique().tolist())
    lid_arr = sel.layer_id.values
    step_arr = sel.decode_step.values
    idx_arr = sel.compressed_kv_index.values.astype(np.int64)
    abs_arr = sel.absolute_position.values
    ratio_arr = sel.compression_ratio.values
    has_scores = sel.index_score.notna().any()
    score_arr = sel.index_score.values if has_scores else None
    ss_idx = {(int(r.layer_id), int(r.decode_step)): r for r in ss.itertuples()}

    token_layer_rows, sample_layer_rows = [], []
    layer_data = {}  # l -> dict(step -> (order, set_sorted, n_comp, abs_pos, ratio))
    # group boundaries per layer
    for l in layers:
        m = lid_arr == l
        st, ix, ab, ra = step_arr[m], idx_arr[m], abs_arr[m], ratio_arr[m]
        sc = score_arr[m] if has_scores else None
        # split by step (steps are contiguous after the sort)
        bounds = np.flatnonzero(np.diff(st)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(st)]])
        steps = [int(st[s]) for s in starts]
        data = {}
        for s, e, t in zip(starts, ends, steps):
            order = ix[s:e]  # rank order (ascending index if no scores)
            row = ss_idx.get((l, t))
            n_comp = int(row.n_candidates_visible) if row is not None else int(order.max()) + 1
            data[t] = {"order": order, "sorted": np.sort(order), "n_comp": n_comp, "abs_pos": int(ab[s]),
                       "ratio": int(ra[s]), "sbi": (dict(zip(order.tolist(), sc[s:e].tolist())) if has_scores else {})}
        layer_data[l] = (steps, data)

        ov, jac, chn, wov, bnd, scorr, dscore, lift, rec_ov = [], [], [], [], [], [], [], [], []
        for i, t in enumerate(steps):
            cur = data[t]
            row = {"layer_id": l, "decode_step": t, "n_candidates": cur["n_comp"], "selected_count": len(cur["order"])}
            if i > 0 and steps[i - 1] == t - 1:
                p = data[t - 1]
                inter = np.intersect1d(cur["sorted"], p["sorted"], assume_unique=True)
                o = len(inter) / len(cur["sorted"]) if len(cur["sorted"]) else float("nan")
                union = len(cur["sorted"]) + len(p["sorted"]) - len(inter)
                jj = (len(inter) / union) if union else 1.0
                ov.append(o); jac.append(jj); chn.append(1 - o)
                wo = weighted_overlap_np(cur["order"], p["order"]); wov.append(wo)
                ne = (len(cur["sorted"]) - len(inter), len(p["sorted"]) - len(inter))
                if has_scores and len(inter) >= 2:
                    xs = np.array([cur["sbi"][c] for c in inter.tolist() if c in cur["sbi"] and c in p["sbi"]])
                    ys = np.array([p["sbi"][c] for c in inter.tolist() if c in cur["sbi"] and c in p["sbi"]])
                    if len(xs) >= 2 and xs.std() > 0 and ys.std() > 0:
                        scorr.append(float(np.corrcoef(xs, ys)[0, 1])); dscore.append(float(np.abs(xs - ys).mean()))
                exp = random_expected_overlap_fraction(top_k, cur["n_comp"])
                if exp and exp == exp and exp > 0:
                    lift.append(o / exp)
                row.update({"adjacent_overlap": o, "adjacent_jaccard": jj, "churn": 1 - o, "weighted_overlap": wo,
                            "new_entries": ne[0], "evicted_entries": ne[1]})
            srow = ss_idx.get((l, t))
            if srow is not None and srow.boundary_margin is not None and srow.boundary_margin == srow.boundary_margin:
                row["boundary_margin"] = srow.boundary_margin
                bnd.append(srow.boundary_margin)
            token_layer_rows.append(row)
            nc = cur["n_comp"]
            lo = max(0, nc - top_k)
            rec_ov.append(float(((cur["sorted"] >= lo) & (cur["sorted"] < nc)).sum() / len(cur["sorted"])) if len(cur["sorted"]) else float("nan"))

        retention = {}
        for lag in LAGS:
            vals = [len(np.intersect1d(data[t]["sorted"], data[t - lag]["sorted"], assume_unique=True)) / len(data[t]["sorted"])
                    for t in steps if (t - lag) in data and len(data[t]["sorted"])]
            retention[f"retention_lag_{lag}"] = (sum(vals) / len(vals)) if vals else float("nan")

        ws_stats = {}
        for w in WINDOWS:
            sizes, ratios = [], []
            for i in range(len(steps)):
                window = [data[steps[j]]["sorted"] for j in range(max(0, i - w + 1), i + 1)]
                u = len(np.unique(np.concatenate(window))) if window else 0
                sizes.append(u); ratios.append(u / (w * top_k))
            ws_stats[f"working_set_w{w}_mean"] = sum(sizes) / len(sizes) if sizes else float("nan")
            ws_stats[f"working_set_ratio_w{w}_mean"] = sum(ratios) / len(ratios) if ratios else float("nan")

        stream = np.concatenate([data[t]["order"] for t in steps]) if steps else np.zeros(0, dtype=np.int64)
        reuse_d, cold = unique_reuse_distances(stream)
        cold_frac = cold / len(stream) if len(stream) else float("nan")
        rd_sorted = np.sort(reuse_d).tolist()

        # persistence run lengths: membership matrix steps x pool (pool = distinct blocks)
        pool = np.unique(stream)
        runs = []
        if len(pool) and steps:
            M = np.zeros((len(steps), len(pool)), dtype=bool)
            for i, t in enumerate(steps):
                M[i, np.searchsorted(pool, data[t]["sorted"])] = True
            # run lengths along axis 0 for every column
            padded = np.vstack([np.zeros((1, len(pool)), bool), M, np.zeros((1, len(pool)), bool)]).astype(np.int8)
            d = np.diff(padded, axis=0)
            starts_r, cols_s = np.nonzero(d == 1)
            ends_r, cols_e = np.nonzero(d == -1)
            # nonzero returns row-major order; sort by column then row to pair starts/ends
            o1 = np.lexsort((starts_r, cols_s)); o2 = np.lexsort((ends_r, cols_e))
            runs = (ends_r[o2] - starts_r[o1]).tolist()

        ages_list = []
        for t in steps:
            ap_, r_ = data[t]["abs_pos"], data[t]["ratio"]
            if r_ == 1:
                ages_list.append(ap_ - data[t]["order"])
            else:
                ages_list.append(ap_ - np.array([representative_original_pos(int(c), r_) for c in data[t]["order"]]))
        ages = np.concatenate(ages_list) if ages_list else np.zeros(0)
        ages_sorted = np.sort(ages).tolist()
        n_ages = len(ages)
        frac_recent = float((ages <= ctx_len * 0.01).sum() / n_ages) if n_ages else float("nan")
        frac_old = float((ages >= ctx_len * 0.50).sum() / n_ages) if n_ages else float("nan")
        frac_mid = 1 - frac_recent - frac_old if n_ages else float("nan")

        mean = lambda v: (sum(v) / len(v)) if v else float("nan")
        sample_layer_rows.append({
            "layer_id": l, "n_decode_steps": len(steps),
            "adjacent_overlap_mean": mean(ov), "adjacent_jaccard_mean": mean(jac), "churn_mean": mean(chn),
            "weighted_overlap_mean": mean(wov), "boundary_margin_mean": mean(bnd), "score_pearson_mean": mean(scorr),
            "mean_abs_dscore": mean(dscore), "locality_lift_mean": mean(lift), "recency_overlap_mean": mean(rec_ov),
            **retention, **ws_stats,
            "reuse_cold_fraction": cold_frac,
            "reuse_p50": percentile(rd_sorted, .5), "reuse_p90": percentile(rd_sorted, .9), "reuse_p99": percentile(rd_sorted, .99),
            "persistence_mean_run": mean(runs), "persistence_max_run": max(runs) if runs else 0,
            "age_mean": float(ages.mean()) if n_ages else float("nan"),
            "age_median": percentile(ages_sorted, .5), "age_p90": percentile(ages_sorted, .9), "age_p99": percentile(ages_sorted, .99),
            "frac_recent": frac_recent, "frac_middle": frac_mid, "frac_old": frac_old,
            "mean_n_candidates": statistics.mean([data[t]["n_comp"] for t in steps]),
        })

    tl = pd.DataFrame(token_layer_rows)
    sl = pd.DataFrame(sample_layer_rows)
    tl.to_parquet(os.path.join(AN, "metrics_token_layer.parquet"), index=False)
    sl.to_parquet(os.path.join(AN, "metrics_sample_layer.parquet"), index=False)

    # cross-layer jaccard over common steps via membership-matrix products
    common = sorted(set.intersection(*[set(layer_data[l][0]) for l in layers])) if layers else []
    L = len(layers)
    inter_sum = np.zeros((L, L)); jac_sum = np.zeros((L, L))
    for t in common:
        sets = [layer_data[l][1][t]["sorted"] for l in layers]
        pool = np.unique(np.concatenate(sets))
        M = np.zeros((L, len(pool)), dtype=np.float32)
        for i, s in enumerate(sets):
            M[i, np.searchsorted(pool, s)] = 1.0
        inter = M @ M.T
        sizes = M.sum(1)
        union = sizes[:, None] + sizes[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            j = np.where(union > 0, inter / union, 1.0)
        jac_sum += j
    cross_rows = []
    for i, li in enumerate(layers):
        for k, lj in enumerate(layers):
            cross_rows.append({"layer_i": li, "layer_j": lj, "cross_jaccard_mean": float(jac_sum[i, k] / len(common)) if common else float("nan")})
    pd.DataFrame(cross_rows).to_parquet(os.path.join(AN, "cross_layer_jaccard.parquet"), index=False)

    n = len(layers)
    groups = {"shallow": layers[:n // 3], "middle": layers[n // 3:2 * n // 3], "deep": layers[2 * n // 3:]}
    grp = {}
    for gname, gl in groups.items():
        sub = sl[sl.layer_id.isin(gl)]
        grp[gname] = {"layers": gl, "adjacent_overlap_mean": float(sub.adjacent_overlap_mean.mean()),
                      "retention_lag_8_mean": float(sub.retention_lag_8.mean()),
                      "working_set_ratio_w64_mean": float(sub.working_set_ratio_w64_mean.mean()),
                      "locality_lift_mean": float(sub.locality_lift_mean.mean()),
                      "recency_overlap_mean": float(sub.recency_overlap_mean.mean())}
    summary = {
        "run_id": os.path.basename(run_dir), "context_length": ctx_len, "is_correct": is_correct, "top_k": top_k,
        "n_csa_layers": len(layers), "n_decode_steps": int(sl.n_decode_steps.max()) if len(sl) else 0,
        "overall_adjacent_overlap_mean": float(sl.adjacent_overlap_mean.mean()),
        "overall_adjacent_jaccard_mean": float(sl.adjacent_jaccard_mean.mean()),
        "overall_churn_mean": float(sl.churn_mean.mean()),
        "overall_weighted_overlap_mean": float(sl.weighted_overlap_mean.mean()),
        "overall_locality_lift_mean": float(sl.locality_lift_mean.mean()),
        "overall_recency_overlap_mean": float(sl.recency_overlap_mean.mean()),
        "overall_retention": {f"lag_{lag}": float(sl[f"retention_lag_{lag}"].mean()) for lag in LAGS},
        "overall_working_set_ratio": {f"w{w}": float(sl[f"working_set_ratio_w{w}_mean"].mean()) for w in WINDOWS},
        "mean_n_candidates": float(sl.mean_n_candidates.mean()),
        "layer_groups": grp,
        "caveats": ["vLLM FP8 GPU runtime (native weights)", "logical KV reuse (not physical cache)",
                    "per-request attributed batched decode", "analysis by analyze_locality_fast.py (vectorized twin)"],
    }
    json.dump(summary, open(os.path.join(AN, "metrics_run_summary.json"), "w"), indent=2)
    print(f"analyzed(fast) {os.path.basename(run_dir)}: csa_layers={len(layers)} steps={summary['n_decode_steps']} "
          f"adj_overlap={summary['overall_adjacent_overlap_mean']:.3f} lift={summary['overall_locality_lift_mean']:.2f}")


if __name__ == "__main__":
    main()
