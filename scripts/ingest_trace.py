#!/usr/bin/env python3
"""Ingest a run's raw trace JSONL into Parquet evidence tables (doc Parts IV).

Usage: ingest_trace.py <run_dir>
Reads:  traces/indexer_trace.jsonl, outputs/generations.jsonl, logs/time_and_stderr.log,
        run_manifest.json, model_config.json
Writes: traces/{decode_tokens,layer_events,selected_kv,score_summaries}.parquet,
        traces/trace_checksums.json
"""
import json, os, sys, re, hashlib, statistics
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locality_lib import ranks_from_scores, original_token_range, representative_original_pos

run_dir = sys.argv[1]
TR = os.path.join(run_dir, "traces")


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_token_latencies(run_dir):
    """ds4 DS4_TOKEN_TIMING prints 'ds4: decode eval N took X ms'."""
    lat = {}
    try:
        txt = open(os.path.join(run_dir, "logs", "time_and_stderr.log")).read()
        for m in re.finditer(r"decode eval (\d+) took ([\d.]+) ms", txt):
            lat[int(m.group(1)) - 1] = float(m.group(2)) * 1e6  # ns, 1-based -> 0-based
    except Exception:
        pass
    return lat


def main():
    gen = load_jsonl(os.path.join(run_dir, "outputs", "generations.jsonl"))[0]
    manifest = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    sample_id = gen["sample_id"]
    benchmark = manifest.get("benchmark_name", "RULER")
    task_type = "retrieval"
    ctx_len = manifest.get("context_length_target")
    prompt_tokens = gen.get("prompt_token_count")
    gen_ids = gen["generated_token_ids"]

    recs = load_jsonl(os.path.join(TR, "indexer_trace.jsonl"))
    recs = [r for r in recs if r.get("phase") == 1]  # decode-only
    if not recs:
        print("WARNING: no decode-phase records found", file=sys.stderr)
    base_pos = min((r["pos"] for r in recs), default=0)

    latencies = parse_token_latencies(run_dir)

    # ---- selected_kv + layer_events + score_summaries ----
    sel_rows, le_rows, ss_rows = [], [], []
    # track per (layer, index): previous-step membership + first seen
    prev_sets = {}        # layer -> set(prev step indices)
    first_seen = {}       # (layer, idx) -> decode_step
    csa_layers = sorted({r["layer"] for r in recs})

    # group records by (decode_step, layer)
    recs.sort(key=lambda r: (r["pos"], r["layer"]))
    for r in recs:
        layer = r["layer"]
        ratio = r.get("ratio", 4)
        ds = r["pos"] - base_pos
        abs_pos = r["pos"]
        sel = r.get("sel", [])
        scores = r.get("scores")
        sbi = {c: scores[i] for i, c in enumerate(sel)} if scores else {}
        ranked = ranks_from_scores(sel, sbi)
        prev = prev_sets.get(layer, set())

        # score summary / layer event
        sc = list(sbi.values())
        ss = {
            "sample_id": sample_id, "benchmark": benchmark, "task_type": task_type,
            "context_length": ctx_len, "decode_step": ds, "absolute_position": abs_pos,
            "layer_id": layer, "layer_type": "CSA", "n_candidates_total": r["n_comp"],
            "n_candidates_visible": r["n_comp"], "configured_top_k": r["top_k"],
            "valid_selected_count": r.get("valid_k", len(sel)),
            "selected_score_min": min(sc) if sc else None,
            "selected_score_max": max(sc) if sc else None,
            "selected_score_mean": (sum(sc) / len(sc)) if sc else None,
            "selected_score_std": (statistics.pstdev(sc) if len(sc) > 1 else 0.0) if sc else None,
            "rank_k_score": r.get("rank_k_score"),
            "rank_k_plus_1_score": r.get("rank_kp1_score"),
            "boundary_margin": (r["rank_k_score"] - r["rank_kp1_score"])
                                if ("rank_k_score" in r and "rank_kp1_score" in r) else None,
        }
        ss_rows.append(ss)
        le_rows.append({**{k: ss[k] for k in (
            "sample_id", "benchmark", "task_type", "context_length", "decode_step",
            "absolute_position", "layer_id", "layer_type", "n_candidates_total",
            "n_candidates_visible", "configured_top_k", "valid_selected_count",
            "selected_score_min", "selected_score_max", "selected_score_mean",
            "selected_score_std", "rank_k_score", "rank_k_plus_1_score", "boundary_margin")},
            "is_sparse_layer": True, "compression_ratio": ratio})

        for rank, c in enumerate(ranked):
            ots, ote = original_token_range(c, ratio)
            rep = representative_original_pos(c, ratio)
            key = (layer, c)
            fseen = first_seen.setdefault(key, ds)
            sel_rows.append({
                "sample_id": sample_id, "benchmark": benchmark, "task_type": task_type,
                "context_length": ctx_len, "decode_step": ds, "decode_position": abs_pos,
                "absolute_position": abs_pos, "layer_id": layer, "layer_type": "CSA",
                "selected_rank": rank, "compressed_kv_index": c,
                "index_score": sbi.get(c), "is_valid": True, "compression_ratio": ratio,
                "original_token_start": ots, "original_token_end": ote,
                "query_to_entry_distance_tokens": abs_pos - rep,
                "previous_step_selected": c in prev,
                "first_seen_decode_step": fseen,
            })
        prev_sets[layer] = set(sel)

    # add non-CSA layer rows (SWA/HCA) for completeness, per decode step
    lmap = {l["layer_id"]: l for l in json.load(open(os.path.join(run_dir, "model_config.json")))["layer_map"]}
    decode_steps = sorted({r["pos"] - base_pos for r in recs})
    for ds in decode_steps:
        for lid, lm in lmap.items():
            if lm["attention_type"] == "CSA":
                continue
            le_rows.append({
                "sample_id": sample_id, "benchmark": benchmark, "task_type": task_type,
                "context_length": ctx_len, "decode_step": ds, "absolute_position": base_pos + ds,
                "layer_id": lid, "layer_type": lm["attention_type"],
                "n_candidates_total": None, "n_candidates_visible": None,
                "configured_top_k": None, "valid_selected_count": None,
                "selected_score_min": None, "selected_score_max": None,
                "selected_score_mean": None, "selected_score_std": None,
                "rank_k_score": None, "rank_k_plus_1_score": None, "boundary_margin": None,
                "is_sparse_layer": False, "compression_ratio": lm["compression_ratio"]})

    # ---- decode_tokens ----
    dt_rows = []
    for ds, tid in enumerate(gen_ids):
        dt_rows.append({
            "sample_id": sample_id, "benchmark": benchmark, "task_type": task_type,
            "context_length": ctx_len, "decode_step": ds, "decode_token_id": tid,
            "absolute_position": base_pos + ds, "prefill_or_decode": "decode",
            "token_latency_ns": latencies.get(ds),
        })

    for name, rows in [("decode_tokens", dt_rows), ("layer_events", le_rows),
                       ("selected_kv", sel_rows), ("score_summaries", ss_rows)]:
        df = pd.DataFrame(rows)
        df.to_parquet(os.path.join(TR, f"{name}.parquet"), index=False)
        print(f"  {name}.parquet: {len(df)} rows")

    # ---- checksums ----
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
    print(f"ingested {os.path.basename(run_dir)}: csa_layers={len(csa_layers)} "
          f"decode_steps={len(decode_steps)} selected_rows={len(sel_rows)}")


if __name__ == "__main__":
    main()
