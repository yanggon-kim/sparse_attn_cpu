#!/usr/bin/env python3
"""Compute MoE routed-expert-selection temporal-locality metrics.

Usage: analyze_moe_locality.py <run_dir>
Reads  run_dir/traces/selected_experts.parquet
Writes run_dir/analysis/moe_metrics_{sample_layer.parquet, cross_layer_jaccard.parquet,
       run_summary.json}

Reuses the id-set-agnostic helpers in locality_lib.py. The MoE expert pool is
FIXED (n_expert, default 256) with n_used chosen per token (default 6), so unlike
the KV indexer the candidate pool does NOT grow with context; the random-baseline
adjacent overlap is n_used/n_expert (= 6/256 = 0.0234). The first n_hash layers
use deterministic token-id hash routing and are reported SEPARATELY from the
learned biased-top-k layers (they must not inflate "routing locality").
"""
import json, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locality_lib import (overlap_fraction, jaccard, new_evicted, weighted_overlap,
                          random_expected_overlap_fraction, percentile)

LAGS = [1, 2, 4, 8, 16, 32, 64]
WINDOWS = [1, 2, 4, 8, 16, 32, 64]
run_dir = sys.argv[1]
TR = os.path.join(run_dir, "traces")
AN = os.path.join(run_dir, "analysis")
os.makedirs(AN, exist_ok=True)


def per_layer_metrics(data, layers, layer_steps, n_expert, k):
    rows = []
    for l in layers:
        steps = layer_steps[l]
        ov, jac, chn, wov, lift = [], [], [], [], []
        for i, t in enumerate(steps):
            cur = data[l][t]
            if i > 0 and steps[i - 1] == t - 1:
                p = data[l][t - 1]
                o = overlap_fraction(cur["set"], p["set"]); ov.append(o)
                jac.append(jaccard(cur["set"], p["set"]))
                chn.append(1 - o)
                wov.append(weighted_overlap(cur["order"], p["order"]))
                exp = random_expected_overlap_fraction(k, n_expert)
                if exp and exp == exp and exp > 0:
                    lift.append(o / exp)
        # retention at lags
        retention = {}
        for lag in LAGS:
            vals = [overlap_fraction(data[l][t]["set"], data[l][t - lag]["set"])
                    for t in steps if (t - lag) in data[l]]
            retention[f"retention_lag_{lag}"] = (sum(vals) / len(vals)) if vals else float("nan")
        # working set (normalizer w*k)
        ws = {}
        for w in WINDOWS:
            ratios = []
            for i, t in enumerate(steps):
                window = [data[l][steps[j]]["set"] for j in range(max(0, i - w + 1), i + 1)]
                u = set().union(*window) if window else set()
                ratios.append(len(u) / (w * k))
            ws[f"working_set_ratio_w{w}_mean"] = (sum(ratios) / len(ratios)) if ratios else float("nan")
        # reuse distance + persistence over the ranked access stream
        stream = []
        for t in steps:
            stream.extend(data[l][t]["order"])
        last_pos, reuse_d, cold = {}, [], 0
        for pos_i, e in enumerate(stream):
            if e in last_pos:
                reuse_d.append(len(set(stream[last_pos[e] + 1:pos_i])))
            else:
                cold += 1
            last_pos[e] = pos_i
        runs = []
        for e in set(stream):
            run = 0
            for t in steps:
                if e in data[l][t]["set"]:
                    run += 1
                elif run:
                    runs.append(run); run = 0
            if run:
                runs.append(run)
        rd = sorted(reuse_d)
        rows.append({
            "layer_id": l, "is_hash_layer": data[l][steps[0]]["is_hash"],
            "n_decode_steps": len(steps),
            "adjacent_overlap_mean": (sum(ov) / len(ov)) if ov else float("nan"),
            "adjacent_jaccard_mean": (sum(jac) / len(jac)) if jac else float("nan"),
            "churn_mean": (sum(chn) / len(chn)) if chn else float("nan"),
            "weighted_overlap_mean": (sum(wov) / len(wov)) if wov else float("nan"),
            "locality_lift_mean": (sum(lift) / len(lift)) if lift else float("nan"),
            **retention, **ws,
            "reuse_cold_fraction": (cold / len(stream)) if stream else float("nan"),
            "reuse_p50": percentile(rd, .5), "reuse_p90": percentile(rd, .9),
            "n_distinct_experts": len(set(stream)),
            "persistence_mean_run": (sum(runs) / len(runs)) if runs else float("nan"),
            "persistence_max_run": max(runs) if runs else 0,
        })
    return rows


def group_summary(df):
    if not len(df):
        return {}
    return {
        "n_layers": int(len(df)),
        "adjacent_overlap_mean": float(df.adjacent_overlap_mean.mean()),
        "adjacent_jaccard_mean": float(df.adjacent_jaccard_mean.mean()),
        "churn_mean": float(df.churn_mean.mean()),
        "weighted_overlap_mean": float(df.weighted_overlap_mean.mean()),
        "locality_lift_mean": float(df.locality_lift_mean.mean()),
        "retention": {f"lag_{lag}": float(df[f"retention_lag_{lag}"].mean()) for lag in LAGS},
        "working_set_ratio": {f"w{w}": float(df[f"working_set_ratio_w{w}_mean"].mean()) for w in WINDOWS},
        "reuse_cold_fraction_mean": float(df.reuse_cold_fraction.mean()),
        "persistence_mean_run_mean": float(df.persistence_mean_run.mean()),
        "n_distinct_experts_mean": float(df.n_distinct_experts.mean()),
    }


def main():
    sel = pd.read_parquet(os.path.join(TR, "selected_experts.parquet"))
    if not len(sel):
        print("no MoE rows; skipping", file=sys.stderr)
        return
    n_expert = int(sel.n_expert.dropna().iloc[0]) if sel.n_expert.notna().any() else 256
    k = int(sel.n_used.dropna().iloc[0]) if sel.n_used.notna().any() else 6
    ctx_len = sel.context_length.iloc[0] if "context_length" in sel else None
    if ctx_len is not None and ctx_len == ctx_len:
        try:
            ctx_len = int(ctx_len)
        except (TypeError, ValueError):
            ctx_len = None

    layers = sorted(sel.layer_id.unique().tolist())
    data = {l: {} for l in layers}
    for (lid, dstep), g in sel.groupby(["layer_id", "decode_step"]):
        g = g.sort_values("selected_rank")
        idx = g.expert_id.tolist()
        data[lid][dstep] = {"order": idx, "set": set(idx),
                            "is_hash": bool(g.is_hash_layer.iloc[0])}
    layer_steps = {l: sorted(data[l].keys()) for l in layers}

    rows = per_layer_metrics(data, layers, layer_steps, n_expert, k)
    sl = pd.DataFrame(rows)
    sl.to_parquet(os.path.join(AN, "moe_metrics_sample_layer.parquet"), index=False)

    learned = sl[~sl.is_hash_layer]
    hashed = sl[sl.is_hash_layer]

    # cross-layer jaccard over learned layers, same decode step
    ll = learned.layer_id.tolist()
    common = set.intersection(*[set(layer_steps[l]) for l in ll]) if ll else set()
    cross_rows = []
    for li in ll:
        for lj in ll:
            vals = [jaccard(data[li][t]["set"], data[lj][t]["set"]) for t in common]
            cross_rows.append({"layer_i": li, "layer_j": lj,
                               "cross_jaccard_mean": (sum(vals) / len(vals)) if vals else float("nan")})
    pd.DataFrame(cross_rows).to_parquet(os.path.join(AN, "moe_cross_layer_jaccard.parquet"), index=False)

    # shallow/mid/deep over LEARNED layers
    n = len(ll)
    groups = {"shallow": ll[:n // 3], "middle": ll[n // 3:2 * n // 3], "deep": ll[2 * n // 3:]}
    grp = {}
    for gname, gl in groups.items():
        sub = learned[learned.layer_id.isin(gl)]
        grp[gname] = {"layers": gl, **group_summary(sub)} if len(sub) else {"layers": gl}

    summary = {
        "run_id": os.path.basename(run_dir),
        "context_length": ctx_len,
        "n_expert": n_expert, "n_used": k,
        "random_baseline_overlap": random_expected_overlap_fraction(k, n_expert),
        "n_layers_total": len(layers),
        "n_hash_layers": int(len(hashed)),
        "n_learned_layers": int(len(learned)),
        "n_decode_steps": int(sl.n_decode_steps.max()) if len(sl) else 0,
        "learned": group_summary(learned),
        "hash": group_summary(hashed),
        "layer_groups_learned": grp,
        "caveats": ["IQ2 quantized runtime (not full precision)", "CPU reference path (not GPU)",
                    "n=1 sample per context length (no cross-sample CIs)",
                    "expert pool fixed at n_expert; context length varies token content/position, "
                    "not the sparsity regime (unlike the KV indexer)",
                    "first n_hash layers are deterministic token-id hash routing (reported separately)"],
    }
    json.dump(summary, open(os.path.join(AN, "moe_metrics_run_summary.json"), "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    lo = summary["learned"].get("adjacent_overlap_mean", float("nan"))
    ll_ = summary["learned"].get("locality_lift_mean", float("nan"))
    print(f"analyzed MoE {os.path.basename(run_dir)}: learned_layers={len(learned)} "
          f"hash_layers={len(hashed)} steps={summary['n_decode_steps']} "
          f"learned_adj_overlap={lo:.3f} lift={ll_:.1f}x")


if __name__ == "__main__":
    main()
