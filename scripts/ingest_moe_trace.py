#!/usr/bin/env python3
"""Ingest a run's raw MoE expert-selection JSONL into a Parquet evidence table.

Usage: ingest_moe_trace.py <run_dir>
Reads:  traces/moe_trace.jsonl  (schema: sv,phase,layer,pos,token,n_expert,n_used,
                                  is_hash,sel[],weights[]), and optionally
        run_manifest.json / outputs/generations.jsonl for sample metadata.
Writes: traces/selected_experts.parquet  (one row per decode_step x layer x expert)

Analogous to ingest_trace.py but for routed-MoE top-k expert selection. Every
FFN layer produces a selection each decode step (43 layers here), vs the CSA
indexer's 21 layers. Expert ids live in a FIXED pool [0, n_expert) with n_used
chosen per token (256/6 for DeepSeek-V4-Flash). The first n_hash layers use
deterministic token-id hash routing (is_hash=1) and are flagged so they can be
separated from the learned biased-top-k layers in analysis.
"""
import json, os, sys, hashlib
import pandas as pd

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


def maybe_load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def main():
    recs = load_jsonl(os.path.join(TR, "moe_trace.jsonl"))
    recs = [r for r in recs if r.get("phase") == 1]  # decode-only
    if not recs:
        print("WARNING: no decode-phase MoE records found", file=sys.stderr)
        pd.DataFrame([]).to_parquet(os.path.join(TR, "selected_experts.parquet"), index=False)
        return

    manifest = maybe_load(os.path.join(run_dir, "run_manifest.json")) or {}
    ctx_len = manifest.get("context_length_target")
    benchmark = manifest.get("benchmark_name", "RULER")
    sample_id = None
    gens = os.path.join(run_dir, "outputs", "generations.jsonl")
    if os.path.exists(gens):
        try:
            sample_id = load_jsonl(gens)[0].get("sample_id")
        except Exception:
            pass

    base_pos = min(r["pos"] for r in recs)
    recs.sort(key=lambda r: (r["pos"], r["layer"]))

    prev_sets = {}    # layer -> set(prev step expert ids)
    first_seen = {}   # (layer, expert) -> decode_step
    rows = []
    layers = sorted({r["layer"] for r in recs})
    for r in recs:
        layer = r["layer"]
        ds = r["pos"] - base_pos
        abs_pos = r["pos"]
        sel = r.get("sel", [])
        weights = r.get("weights") or [None] * len(sel)
        is_hash = int(r.get("is_hash", 0))
        prev = prev_sets.get(layer, set())
        for rank, (e, w) in enumerate(zip(sel, weights)):
            key = (layer, e)
            fseen = first_seen.setdefault(key, ds)
            rows.append({
                "sample_id": sample_id, "benchmark": benchmark, "context_length": ctx_len,
                "decode_step": ds, "absolute_position": abs_pos, "token_id": r.get("token"),
                "layer_id": layer, "is_hash_layer": bool(is_hash),
                "n_expert": r.get("n_expert"), "n_used": r.get("n_used", len(sel)),
                "selected_rank": rank, "expert_id": e, "expert_weight": w,
                "previous_step_selected": e in prev,
                "first_seen_decode_step": fseen,
            })
        prev_sets[layer] = set(sel)

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(TR, "selected_experts.parquet"), index=False)

    # refresh checksums (keep alongside the KV trace checksums if present)
    checks = {}
    cpath = os.path.join(TR, "trace_checksums.json")
    if os.path.exists(cpath):
        checks = maybe_load(cpath) or {}
    for fn in ("moe_trace.jsonl", "selected_experts.parquet"):
        fp = os.path.join(TR, fn)
        if os.path.isfile(fp):
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            checks[fn] = h.hexdigest()
    json.dump(checks, open(cpath, "w"), indent=2)

    n_hash = len({l for l in layers if any(r["layer"] == l and r.get("is_hash") for r in recs)})
    decode_steps = sorted({r["pos"] - base_pos for r in recs})
    print(f"ingested MoE {os.path.basename(run_dir)}: layers={len(layers)} "
          f"(hash={n_hash}) decode_steps={len(decode_steps)} rows={len(df)}")


if __name__ == "__main__":
    main()
