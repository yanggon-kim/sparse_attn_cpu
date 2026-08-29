#!/usr/bin/env python3
"""Vectorized drop-in for scripts/ingest_trace.py (same inputs, same parquet files/columns), for GPU-scale
traces (61 layers x 2048 selections x 2048 steps = 256 M selected rows per long-decode run).
Usage: ingest_trace_fast.py <run_dir>
Differences vs ingest_trace.py (verified with compare_ingest.py on the exp1 smoke run):
  - string constant columns (sample_id, benchmark, task_type, layer_type) are pandas categoricals;
  - index_score is float64 NaN where ingest_trace.py stores None (no per-index scores in GPU traces);
  - ranks: sel is sorted ascending and scores absent -> selected_rank = position (ranks_from_scores fallback).
"""
import hashlib, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from locality_lib import original_token_range, representative_original_pos  # noqa: E402

run_dir = sys.argv[1]
TR = os.path.join(run_dir, "traces")


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    gen = load_jsonl(os.path.join(run_dir, "outputs", "generations.jsonl"))[0]
    manifest = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    mc = json.load(open(os.path.join(run_dir, "model_config.json")))
    sample_id = gen["sample_id"]
    benchmark = manifest.get("benchmark_name", "RULER")
    task_type = "retrieval"
    ctx_len = manifest.get("context_length_target")
    gen_ids = gen["generated_token_ids"]

    recs = [r for r in load_jsonl(os.path.join(TR, "indexer_trace.jsonl")) if r.get("phase") == 1]
    recs.sort(key=lambda r: (r["pos"], r["layer"]))
    base_pos = min((r["pos"] for r in recs), default=0)
    layers = sorted({r["layer"] for r in recs})
    max_n = max((r["n_comp"] for r in recs), default=0) + 1

    # per-layer state
    prev_sets = {l: np.zeros(0, dtype=np.int64) for l in layers}
    first_seen = {l: np.full(max_n, -1, dtype=np.int64) for l in layers}

    cols = {k: [] for k in ("decode_step", "absolute_position", "layer_id", "selected_rank", "compressed_kv_index",
                            "compression_ratio", "original_token_start", "original_token_end",
                            "query_to_entry_distance_tokens", "previous_step_selected", "first_seen_decode_step")}
    ss_rows, le_rows = [], []
    for r in recs:
        layer, ratio = r["layer"], r.get("ratio", 4)
        ds, abs_pos = r["pos"] - base_pos, r["pos"]
        sel = np.asarray(r.get("sel", []), dtype=np.int64)
        sel = np.sort(sel)  # ranks_from_scores fallback = ascending index
        n = len(sel)
        scores = r.get("scores")
        sc = list(scores) if scores else []
        ss = {"sample_id": sample_id, "benchmark": benchmark, "task_type": task_type, "context_length": ctx_len,
              "decode_step": ds, "absolute_position": abs_pos, "layer_id": layer, "layer_type": "CSA",
              "n_candidates_total": r["n_comp"], "n_candidates_visible": r["n_comp"], "configured_top_k": r["top_k"],
              "valid_selected_count": r.get("valid_k", n),
              "selected_score_min": min(sc) if sc else None, "selected_score_max": max(sc) if sc else None,
              "selected_score_mean": (sum(sc) / len(sc)) if sc else None,
              "selected_score_std": (float(np.std(sc)) if len(sc) > 1 else 0.0) if sc else None,
              "rank_k_score": r.get("rank_k_score"), "rank_k_plus_1_score": r.get("rank_kp1_score"),
              "boundary_margin": (r["rank_k_score"] - r["rank_kp1_score"]) if ("rank_k_score" in r and "rank_kp1_score" in r) else None}
        ss_rows.append(ss)
        le_rows.append({**ss, "is_sparse_layer": True, "compression_ratio": ratio})
        if n:
            prev = prev_sets[layer]
            fs = first_seen[layer]
            fsv = fs[sel]
            newmask = fsv < 0
            fs[sel[newmask]] = ds
            fsv = np.where(newmask, ds, fsv)
            if ratio == 1:
                ots, ote, rep = sel, sel, sel
            else:
                ots = sel * ratio
                ote = ots + ratio - 1
                rep = np.array([representative_original_pos(int(c), ratio) for c in sel], dtype=np.int64)
            cols["decode_step"].append(np.full(n, ds, dtype=np.int64))
            cols["absolute_position"].append(np.full(n, abs_pos, dtype=np.int64))
            cols["layer_id"].append(np.full(n, layer, dtype=np.int64))
            cols["selected_rank"].append(np.arange(n, dtype=np.int64))
            cols["compressed_kv_index"].append(sel)
            cols["compression_ratio"].append(np.full(n, ratio, dtype=np.int64))
            cols["original_token_start"].append(ots)
            cols["original_token_end"].append(ote)
            cols["query_to_entry_distance_tokens"].append(abs_pos - rep)
            cols["previous_step_selected"].append(np.isin(sel, prev, assume_unique=True))
            cols["first_seen_decode_step"].append(fsv)
        prev_sets[layer] = sel

    N = sum(len(x) for x in cols["decode_step"])
    cat = lambda v: pd.Categorical([v] * N) if N else pd.Categorical([])
    ds_arr = np.concatenate(cols["decode_step"]) if N else np.zeros(0, dtype=np.int64)
    sel_df = pd.DataFrame({
        "sample_id": cat(sample_id), "benchmark": cat(benchmark), "task_type": cat(task_type),
        "context_length": np.full(N, ctx_len, dtype=np.int64),
        "decode_step": ds_arr,
        "decode_position": np.concatenate(cols["absolute_position"]) if N else ds_arr,
        "absolute_position": np.concatenate(cols["absolute_position"]) if N else ds_arr,
        "layer_id": np.concatenate(cols["layer_id"]) if N else ds_arr,
        "layer_type": cat("CSA"),
        "selected_rank": np.concatenate(cols["selected_rank"]) if N else ds_arr,
        "compressed_kv_index": np.concatenate(cols["compressed_kv_index"]) if N else ds_arr,
        "index_score": np.full(N, np.nan),
        "is_valid": np.ones(N, dtype=bool),
        "compression_ratio": np.concatenate(cols["compression_ratio"]) if N else ds_arr,
        "original_token_start": np.concatenate(cols["original_token_start"]) if N else ds_arr,
        "original_token_end": np.concatenate(cols["original_token_end"]) if N else ds_arr,
        "query_to_entry_distance_tokens": np.concatenate(cols["query_to_entry_distance_tokens"]) if N else ds_arr,
        "previous_step_selected": np.concatenate(cols["previous_step_selected"]) if N else np.zeros(0, dtype=bool),
        "first_seen_decode_step": np.concatenate(cols["first_seen_decode_step"]) if N else ds_arr,
    })

    lmap = {l["layer_id"]: l for l in mc["layer_map"]}
    decode_steps = sorted({r["pos"] - base_pos for r in recs})
    for ds in decode_steps:
        for lid, lm in lmap.items():
            if lm["attention_type"] == "CSA":
                continue
            le_rows.append({"sample_id": sample_id, "benchmark": benchmark, "task_type": task_type,
                            "context_length": ctx_len, "decode_step": ds, "absolute_position": base_pos + ds,
                            "layer_id": lid, "layer_type": lm["attention_type"],
                            "n_candidates_total": None, "n_candidates_visible": None, "configured_top_k": None,
                            "valid_selected_count": None, "selected_score_min": None, "selected_score_max": None,
                            "selected_score_mean": None, "selected_score_std": None, "rank_k_score": None,
                            "rank_k_plus_1_score": None, "boundary_margin": None, "is_sparse_layer": False,
                            "compression_ratio": lm["compression_ratio"]})
    dt_rows = [{"sample_id": sample_id, "benchmark": benchmark, "task_type": task_type, "context_length": ctx_len,
                "decode_step": ds, "decode_token_id": tid, "absolute_position": base_pos + ds,
                "prefill_or_decode": "decode", "token_latency_ns": None} for ds, tid in enumerate(gen_ids)]

    pd.DataFrame(dt_rows).to_parquet(os.path.join(TR, "decode_tokens.parquet"), index=False)
    pd.DataFrame(le_rows).to_parquet(os.path.join(TR, "layer_events.parquet"), index=False)
    sel_df.to_parquet(os.path.join(TR, "selected_kv.parquet"), index=False)
    pd.DataFrame(ss_rows).to_parquet(os.path.join(TR, "score_summaries.parquet"), index=False)
    checks = {}
    for fn in os.listdir(TR):
        fp = os.path.join(TR, fn)
        if os.path.isfile(fp):
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            checks[fn] = h.hexdigest()
    json.dump(checks, open(os.path.join(TR, "trace_checksums.json"), "w"), indent=2)
    print(f"ingested(fast) {os.path.basename(run_dir)}: csa_layers={len(layers)} decode_steps={len(decode_steps)} selected_rows={N}")


if __name__ == "__main__":
    main()
