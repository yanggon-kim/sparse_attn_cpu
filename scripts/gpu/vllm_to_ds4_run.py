#!/usr/bin/env python3
"""Adapter: one vLLM-traced request -> a ds4-schema run directory (exp1 §4), so that the unchanged
CPU analysis chain (ingest_trace.py -> validate_trace.py -> analyze_locality.py -> ...) runs on it.

Usage: vllm_to_ds4_run.py --trace <SEL_TRACE dir> --req-json <request.json> --meta-json <meta.json>
                          --run-dir <out_run_dir> [--computing-only]

request.json (written by the runner per request):
  {request_id, sample_id, prompt_token_count, generated_token_ids[], generated_text, finish_reason,
   benchmark_prediction, reference_answer[], is_correct, score, routed_experts[[steps][layers][k]] (optional,
   decode positions only, as returned by vLLM with routed_experts_prompt_start=prompt_len)}
meta.json (per run):
  {run_id, model_name, model_path, benchmark_name, task_subset, context_length_target, max_new_tokens,
   batch_size, vllm_commit, gpu, wall_clock_seconds, gpu_hours, seed, kind, num_layers, top_k,
   index_n_heads, index_head_dim, expert_count, expert_used, first_k_dense, index_topk_freq,
   index_skip_topk_offset, index_topk_pattern, computed_layers[] (layers that compute their own top-k)}

Outputs in run_dir: traces/indexer_trace.jsonl, traces/moe_trace.jsonl (if routed_experts given),
  outputs/generations.jsonl, run_manifest.json, model_config.json, logs/time_and_stderr.log, prompts/sample.json
--computing-only: mark only computing layers as "CSA" in model_config.json and drop shared-layer records
  (exp2 variant (b)); default keeps all layers (variant (a)).
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--req-json", required=True)
ap.add_argument("--meta-json", required=True)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--computing-only", action="store_true")

def _resolve_trace_file(trace_dir, request_id):
    """vLLM's engine-internal request id is '<request_id>-<hex>'; find the by_req file by exact or prefix match."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(request_id))
    d = os.path.join(trace_dir, "by_req")
    exact = os.path.join(d, safe + ".jsonl")
    if os.path.exists(exact):
        return exact
    cands = sorted(f for f in os.listdir(d) if f.startswith(safe + "-") and f.endswith(".jsonl"))
    if len(cands) != 1:
        raise FileNotFoundError(f"{len(cands)} trace files match request {request_id!r} in {d}: {cands[:5]}")
    return os.path.join(d, cands[0])


a = ap.parse_args()

req = json.load(open(a.req_json))
meta = json.load(open(a.meta_json))
rd = a.run_dir
for d in ("traces", "outputs", "logs", "prompts", "analysis"):
    os.makedirs(os.path.join(rd, d), exist_ok=True)

src = _resolve_trace_file(a.trace, req["request_id"])
computed = set(meta.get("computed_layers") or range(meta["num_layers"]))
n_layers = meta["num_layers"]

# ---- indexer_trace.jsonl (phase 1 only, sorted by pos then layer) ----
recs, moe_recs = [], []
with open(src) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("phase") != 1:
            continue
        if r.get("moe"):
            moe_recs.append(r)
            continue
        if a.computing_only and not r.get("topk_computed", True):
            continue
        recs.append(r)
recs.sort(key=lambda r: (r["pos"], r["layer"]))
# de-duplicate (a request can be re-scheduled; keep first occurrence per (pos, layer))
seen, uniq = set(), []
for r in recs:
    k = (r["pos"], r["layer"])
    if k in seen:
        continue
    seen.add(k)
    uniq.append(r)
recs = uniq
n_steps = len({r["pos"] for r in recs})
with open(os.path.join(rd, "traces", "indexer_trace.jsonl"), "w") as f:
    for r in recs:
        out = {"sv": 2, "phase": 1, "layer": r["layer"], "ratio": 1, "pos": r["pos"], "n_comp": r["n_comp"],
               "top_k": r["top_k"], "valid_k": r["valid_k"], "sel": r["sel"],
               "topk_computed": r.get("topk_computed", True), "shared_from_layer": r.get("shared_from_layer", r["layer"])}
        if r.get("scores"):
            out["scores"] = r["scores"]
        f.write(json.dumps(out, separators=(",", ":")) + "\n")

# ---- moe_trace.jsonl: from the hook's MoE records (preferred) or vLLM routed_experts [steps][layers][k] ----
re_ = req.get("routed_experts")
gen_ids = req["generated_token_ids"]
n_moe = 0
if moe_recs:
    base_pos = min((r["pos"] for r in recs), default=req["prompt_token_count"] - 1)
    moe_recs.sort(key=lambda r: (r["pos"], r["layer"]))
    seen = set()
    with open(os.path.join(rd, "traces", "moe_trace.jsonl"), "w") as f:
        for r in moe_recs:
            k = (r["pos"], r["layer"])
            if k in seen:
                continue
            seen.add(k)
            ds = r["pos"] - base_pos
            tok = gen_ids[ds] if 0 <= ds < len(gen_ids) else None
            f.write(json.dumps({"sv": 2, "phase": 1, "layer": r["layer"], "pos": r["pos"], "token": tok,
                                "n_expert": meta["expert_count"], "n_used": meta["expert_used"],
                                "is_hash": False, "sel": r["sel"]}, separators=(",", ":")) + "\n")
            n_moe += 1
elif re_:
    base_pos = min((r["pos"] for r in recs), default=req["prompt_token_count"] - 1)
    first_dense = meta.get("first_k_dense", 3)
    with open(os.path.join(rd, "traces", "moe_trace.jsonl"), "w") as f:
        for ds, per_layer in enumerate(re_):
            pos = base_pos + ds
            tok = gen_ids[ds] if ds < len(gen_ids) else None
            for li, sel in enumerate(per_layer):
                layer = li + first_dense if len(per_layer) == n_layers - first_dense else li
                sel = [int(x) for x in sel if int(x) >= 0]
                if not sel:
                    continue
                f.write(json.dumps({"sv": 2, "phase": 1, "layer": layer, "pos": pos, "token": tok,
                                    "n_expert": meta["expert_count"], "n_used": meta["expert_used"],
                                    "is_hash": False, "sel": sel}, separators=(",", ":")) + "\n")
                n_moe += 1

# ---- generations.jsonl ----
gen = {"run_id": meta["run_id"], "sample_id": req["sample_id"], "prompt_token_count": req["prompt_token_count"],
       "generated_token_count": len(gen_ids), "generated_token_ids": gen_ids,
       "generated_text": req.get("generated_text", ""), "finish_reason": req.get("finish_reason"),
       "benchmark_prediction": req.get("benchmark_prediction"), "reference_answer": req.get("reference_answer", []),
       "is_correct": req.get("is_correct"), "score": req.get("score"), "request_id": req["request_id"]}
with open(os.path.join(rd, "outputs", "generations.jsonl"), "w") as f:
    f.write(json.dumps(gen) + "\n")

# ---- run_manifest.json ----
manifest = {"schema_version": "1", "trace_schema_version": 2, "run_id": meta["run_id"], "backend": "cuda",
            "model_name": meta["model_name"], "model_path": meta["model_path"], "quantization": "fp8 (native)",
            "benchmark_name": meta["benchmark_name"], "task_subset": meta.get("task_subset"),
            "context_length_target": meta["context_length_target"],
            "context_length_actual_tokens": req["prompt_token_count"], "max_new_tokens": meta["max_new_tokens"],
            "decode_parameters": {"temperature": 0, "greedy": True, "seed": meta.get("seed", 42),
                                  "batch_size": meta.get("batch_size", 1)},
            "vllm_commit": meta["vllm_commit"], "gpu": meta["gpu"], "kind": meta.get("kind"),
            "timing": {"wall_clock_seconds": meta.get("wall_clock_seconds"), "gpu_hours": meta.get("gpu_hours")},
            "is_correct": req.get("is_correct"), "sample_id": req["sample_id"], "request_id": req["request_id"],
            "decode_steps_traced": n_steps, "moe_records": n_moe, "variant": "computing_only" if a.computing_only else "all_layers"}
json.dump(manifest, open(os.path.join(rd, "run_manifest.json"), "w"), indent=2)

# ---- model_config.json ----
layer_map = []
for l in range(n_layers):
    is_csa = (l in computed) if a.computing_only else True
    layer_map.append({"layer_id": l, "attention_type": "CSA" if is_csa else "SHARED", "compression_ratio": 1,
                      "topk_computed": l in computed})
mc = {"num_layers": n_layers, "sparse_top_k": meta["top_k"], "indexer_head_count": meta.get("index_n_heads"),
      "indexer_head_dim": meta.get("index_head_dim"), "expert_count": meta["expert_count"],
      "expert_used": meta["expert_used"], "first_k_dense": meta.get("first_k_dense", 3),
      "index_topk_freq": meta.get("index_topk_freq"), "index_skip_topk_offset": meta.get("index_skip_topk_offset"),
      "index_topk_pattern": meta.get("index_topk_pattern"), "layer_map": layer_map}
json.dump(mc, open(os.path.join(rd, "model_config.json"), "w"), indent=2)
json.dump({"sample_id": req["sample_id"], "reference_answer": req.get("reference_answer", []),
           "prompt_token_count": req["prompt_token_count"]}, open(os.path.join(rd, "prompts", "sample.json"), "w"))
open(os.path.join(rd, "logs", "time_and_stderr.log"), "a").write(
    f"vllm run {meta['run_id']} request {req['request_id']} wall={meta.get('wall_clock_seconds')}s\n")
print(f"[adapter] {rd}: {len(recs)} indexer records over {n_steps} decode steps, {n_moe} moe records, "
      f"{len(gen_ids)} generated tokens")
