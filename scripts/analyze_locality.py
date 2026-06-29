#!/usr/bin/env python3
"""Compute sparse-attention KV temporal-locality metrics (doc Part V §20-27).
Usage: analyze_locality.py <run_dir>
Reads run_dir/traces/*.parquet; writes run_dir/analysis/metrics_*.{parquet,json}.
All metrics from the immutable trace alone (reproducible).
"""
import json, os, sys, statistics
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locality_lib import (overlap_fraction, jaccard, new_evicted, weighted_overlap,
                          ranks_from_scores, random_expected_overlap_fraction,
                          percentile, pearson, spearman, representative_original_pos)

LAGS = [1, 2, 4, 8, 16, 32, 64]
WINDOWS = [1, 2, 4, 8, 16, 32, 64]
run_dir = sys.argv[1]
TR = os.path.join(run_dir, "traces")
AN = os.path.join(run_dir, "analysis")
os.makedirs(AN, exist_ok=True)


def pct_stats(vals):
    v = sorted(x for x in vals if x == x)
    if not v:
        return {k: float("nan") for k in ("mean", "median", "p10", "p25", "p75", "p90", "p99", "n")}
    return {"mean": sum(v) / len(v), "median": percentile(v, .5), "p10": percentile(v, .1),
            "p25": percentile(v, .25), "p75": percentile(v, .75), "p90": percentile(v, .9),
            "p99": percentile(v, .99), "n": len(v)}


def main():
    sel = pd.read_parquet(os.path.join(TR, "selected_kv.parquet"))
    ss = pd.read_parquet(os.path.join(TR, "score_summaries.parquet"))
    gen = json.loads(open(os.path.join(run_dir, "outputs", "generations.jsonl")).read().splitlines()[0])
    manifest = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    ctx_len = manifest.get("context_length_target")
    top_k = int(ss.configured_top_k.iloc[0]) if len(ss) else 512
    is_correct = gen.get("is_correct")

    # Per (layer, step): ordered sel, scores, n_comp
    layers = sorted(sel.layer_id.unique().tolist())
    data = {l: {} for l in layers}  # layer -> step -> dict
    for (lid, dstep), g in sel.groupby(["layer_id", "decode_step"]):
        g = g.sort_values("selected_rank")
        idx = g.compressed_kv_index.tolist()
        sc = g.index_score.tolist()
        sbi = {c: s for c, s in zip(idx, sc)}
        ratio = int(g.compression_ratio.iloc[0])
        nrow = ss[(ss.layer_id == lid) & (ss.decode_step == dstep)]
        n_comp = int(nrow.n_candidates_visible.iloc[0]) if len(nrow) else len(idx)
        data[lid][dstep] = {"order": idx, "set": set(idx), "sbi": sbi,
                            "n_comp": n_comp, "ratio": ratio,
                            "abs_pos": int(g.absolute_position.iloc[0])}

    token_layer_rows = []
    sample_layer_rows = []
    layer_steps = {l: sorted(data[l].keys()) for l in layers}

    for l in layers:
        steps = layer_steps[l]
        # ---- per-step adjacent metrics ----
        ov, jac, chn, wov, bnd, scorr, dscore, lift = [], [], [], [], [], [], [], []
        for i, t in enumerate(steps):
            cur = data[l][t]
            row = {"layer_id": l, "decode_step": t, "n_candidates": cur["n_comp"],
                   "selected_count": len(cur["order"])}
            if i > 0 and steps[i - 1] == t - 1:
                p = data[l][t - 1]
                o = overlap_fraction(cur["set"], p["set"]); ov.append(o)
                jj = jaccard(cur["set"], p["set"]); jac.append(jj)
                chn.append(1 - o)
                wo = weighted_overlap(cur["order"], p["order"]); wov.append(wo)
                ne = new_evicted(cur["set"], p["set"])
                # score stability over shared
                shared = cur["set"] & p["set"]
                if len(shared) >= 2:
                    xs = [cur["sbi"][c] for c in shared if c in cur["sbi"] and c in p["sbi"]]
                    ys = [p["sbi"][c] for c in shared if c in cur["sbi"] and c in p["sbi"]]
                    if len(xs) >= 2:
                        pc = pearson(xs, ys); scorr.append(pc)
                        md = sum(abs(a - b) for a, b in zip(xs, ys)) / len(xs); dscore.append(md)
                exp = random_expected_overlap_fraction(top_k, cur["n_comp"])
                if exp and exp == exp and exp > 0:
                    lift.append(o / exp)
                row.update({"adjacent_overlap": o, "adjacent_jaccard": jj, "churn": 1 - o,
                            "weighted_overlap": wo, "new_entries": ne[0], "evicted_entries": ne[1]})
            nrow = ss[(ss.layer_id == l) & (ss.decode_step == t)]
            if len(nrow) and nrow.boundary_margin.iloc[0] is not None:
                bm = nrow.boundary_margin.iloc[0]
                row["boundary_margin"] = bm
                if bm == bm:
                    bnd.append(bm)
            token_layer_rows.append(row)

        # ---- retention at lags ----
        retention = {}
        for lag in LAGS:
            vals = []
            for t in steps:
                if (t - lag) in data[l]:
                    vals.append(overlap_fraction(data[l][t]["set"], data[l][t - lag]["set"]))
            retention[f"retention_lag_{lag}"] = (sum(vals) / len(vals)) if vals else float("nan")

        # ---- working set ----
        ws_stats = {}
        for w in WINDOWS:
            sizes, ratios = [], []
            for i, t in enumerate(steps):
                window = [data[l][steps[j]]["set"] for j in range(max(0, i - w + 1), i + 1)]
                if len(window) < min(w, len(steps)):
                    pass
                u = set().union(*window) if window else set()
                sizes.append(len(u))
                ratios.append(len(u) / (w * top_k))
            ws_stats[f"working_set_w{w}_mean"] = sum(sizes) / len(sizes) if sizes else float("nan")
            ws_stats[f"working_set_ratio_w{w}_mean"] = sum(ratios) / len(ratios) if ratios else float("nan")

        # ---- reuse distance (logical access stream in rank order) ----
        stream = []
        for t in steps:
            stream.extend(data[l][t]["order"])
        last_pos = {}
        seen = set()
        reuse_d, cold = [], 0
        for pos_i, blk in enumerate(stream):
            if blk in last_pos:
                # unique reuse distance = #distinct blocks since last use
                between = stream[last_pos[blk] + 1:pos_i]
                reuse_d.append(len(set(between)))
            else:
                cold += 1
            last_pos[blk] = pos_i
            seen.add(blk)
        cold_frac = cold / len(stream) if stream else float("nan")
        rd_sorted = sorted(reuse_d)
        # persistence run-length: consecutive steps an index stays selected
        runs = []
        for blk in set(stream):
            run = 0
            for t in steps:
                if blk in data[l][t]["set"]:
                    run += 1
                else:
                    if run:
                        runs.append(run)
                    run = 0
            if run:
                runs.append(run)

        # ---- access age ----
        ages = []
        for t in steps:
            ap = data[l][t]["abs_pos"]
            ratio = data[l][t]["ratio"]
            for c in data[l][t]["order"]:
                ages.append(ap - representative_original_pos(c, ratio))
        ages_sorted = sorted(ages)
        recent_cut = ctx_len * 0.99
        mid_cut = ctx_len * 0.50
        frac_recent = sum(1 for a in ages if a <= ctx_len * 0.01) / len(ages) if ages else float("nan")
        frac_old = sum(1 for a in ages if a >= ctx_len * 0.50) / len(ages) if ages else float("nan")
        frac_mid = 1 - frac_recent - frac_old if ages else float("nan")

        # ---- recency baseline overlap (most-recent top_k compressed) ----
        rec_ov = []
        for t in steps:
            nc = data[l][t]["n_comp"]
            rec_set = set(range(max(0, nc - top_k), nc))
            rec_ov.append(overlap_fraction(data[l][t]["set"], rec_set))

        sample_layer_rows.append({
            "layer_id": l, "n_decode_steps": len(steps),
            "adjacent_overlap_mean": (sum(ov) / len(ov)) if ov else float("nan"),
            "adjacent_jaccard_mean": (sum(jac) / len(jac)) if jac else float("nan"),
            "churn_mean": (sum(chn) / len(chn)) if chn else float("nan"),
            "weighted_overlap_mean": (sum(wov) / len(wov)) if wov else float("nan"),
            "boundary_margin_mean": (sum(bnd) / len(bnd)) if bnd else float("nan"),
            "score_pearson_mean": (sum(scorr) / len(scorr)) if scorr else float("nan"),
            "mean_abs_dscore": (sum(dscore) / len(dscore)) if dscore else float("nan"),
            "locality_lift_mean": (sum(lift) / len(lift)) if lift else float("nan"),
            "recency_overlap_mean": (sum(rec_ov) / len(rec_ov)) if rec_ov else float("nan"),
            **retention, **ws_stats,
            "reuse_cold_fraction": cold_frac,
            "reuse_p50": percentile(rd_sorted, .5), "reuse_p90": percentile(rd_sorted, .9),
            "reuse_p99": percentile(rd_sorted, .99),
            "persistence_mean_run": (sum(runs) / len(runs)) if runs else float("nan"),
            "persistence_max_run": max(runs) if runs else 0,
            "age_mean": (sum(ages) / len(ages)) if ages else float("nan"),
            "age_median": percentile(ages_sorted, .5), "age_p90": percentile(ages_sorted, .9),
            "age_p99": percentile(ages_sorted, .99),
            "frac_recent": frac_recent, "frac_middle": frac_mid, "frac_old": frac_old,
            "mean_n_candidates": statistics.mean([data[l][t]["n_comp"] for t in steps]),
        })

    tl = pd.DataFrame(token_layer_rows)
    sl = pd.DataFrame(sample_layer_rows)
    tl.to_parquet(os.path.join(AN, "metrics_token_layer.parquet"), index=False)
    sl.to_parquet(os.path.join(AN, "metrics_sample_layer.parquet"), index=False)

    # ---- cross-layer jaccard (same decode step) ----
    cross_rows = []
    common_steps = set.intersection(*[set(layer_steps[l]) for l in layers]) if layers else set()
    for li in layers:
        for lj in layers:
            vals = [jaccard(data[li][t]["set"], data[lj][t]["set"]) for t in common_steps]
            cross_rows.append({"layer_i": li, "layer_j": lj,
                               "cross_jaccard_mean": (sum(vals) / len(vals)) if vals else float("nan")})
    pd.DataFrame(cross_rows).to_parquet(os.path.join(AN, "cross_layer_jaccard.parquet"), index=False)

    # ---- layer-group summaries (shallow/mid/deep over CSA layers) ----
    n = len(layers)
    groups = {"shallow": layers[:n // 3], "middle": layers[n // 3:2 * n // 3], "deep": layers[2 * n // 3:]}
    grp = {}
    for gname, gl in groups.items():
        sub = sl[sl.layer_id.isin(gl)]
        grp[gname] = {
            "layers": gl,
            "adjacent_overlap_mean": float(sub.adjacent_overlap_mean.mean()),
            "retention_lag_8_mean": float(sub.retention_lag_8.mean()),
            "working_set_ratio_w64_mean": float(sub.working_set_ratio_w64_mean.mean()),
            "locality_lift_mean": float(sub.locality_lift_mean.mean()),
            "recency_overlap_mean": float(sub.recency_overlap_mean.mean()),
        }

    summary = {
        "run_id": os.path.basename(run_dir),
        "context_length": ctx_len,
        "is_correct": is_correct,
        "top_k": top_k,
        "n_csa_layers": len(layers),
        "n_decode_steps": int(sl.n_decode_steps.max()) if len(sl) else 0,
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
        "caveats": ["Q2 quantized runtime (not full precision)", "CPU reference path (not GPU)",
                    "logical KV reuse (not physical cache)", "n=1 sample per context length (no cross-sample CIs)"],
    }
    json.dump(summary, open(os.path.join(AN, "metrics_run_summary.json"), "w"), indent=2)
    print(f"analyzed {os.path.basename(run_dir)}: csa_layers={len(layers)} steps={summary['n_decode_steps']} "
          f"adj_overlap={summary['overall_adjacent_overlap_mean']:.3f} "
          f"lift={summary['overall_locality_lift_mean']:.2f}")


if __name__ == "__main__":
    main()
