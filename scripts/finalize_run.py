#!/usr/bin/env python3
"""Assemble per-run manifests + generations.jsonl from raw run artifacts.

Usage: finalize_run.py <run_dir> <length> <prompt_file> <sample_json>
Reads: outputs/logprobs.json, logs/*, ../../model_inspect.txt, ../../model_sha256.txt
Writes: run_manifest.json, model_config.json, machine.json, benchmark_config.json,
        outputs/generations.jsonl
"""
import json, os, sys, re, platform, subprocess, hashlib, datetime

run_dir, length, prompt_file, sample_json = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
EXP = "<WORKDIR>/experiment"
MODEL = "<WORKDIR>/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"


def read(p, default=""):
    try:
        return open(p).read()
    except Exception:
        return default


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---- layer map (verified pattern; L0-1 SWA, even CSA r4, odd HCA r128) ----
def layer_map(n_layer=43):
    out = []
    for il in range(n_layer):
        if il < 2:
            t, r = "SWA", 0
        elif il % 2 == 0:
            t, r = "CSA", 4
        else:
            t, r = "HCA", 128
        out.append({"layer_id": il, "attention_type": t, "compression_ratio": r})
    return out


def parse_inspect():
    txt = read(os.path.join(EXP, "model_inspect.txt"))
    g = lambda pat: (re.search(pat, txt) or [None, None])[1] if re.search(pat, txt) else None
    def num(pat):
        m = re.search(pat, txt)
        return int(m.group(1)) if m else None
    return {
        "num_layers": num(r"layers:\s*(\d+)"),
        "query_head_count": num(r"heads=(\d+)"),
        "kv_head_count": num(r"kv_heads=(\d+)"),
        "attention_head_dim": num(r"head_dim=(\d+)"),
        "sliding_window_size": num(r"swa=(\d+)"),
        "indexer_head_count": num(r"indexer:\s*heads=(\d+)"),
        "indexer_head_dim": num(r"head_dim=(\d+)", ),
        "sparse_top_k": num(r"top_k=(\d+)"),
        "expert_count": num(r"count=(\d+)"),
        "expert_used": num(r"used=(\d+)"),
    }


def parse_timing():
    # ds4 prints "ds4: prefill: X t/s, generation: Y t/s"; /usr/bin/time -v gives RSS/faults.
    err = read(os.path.join(run_dir, "logs", "time_and_stderr.log"))
    out = {}
    m = re.search(r"prefill:\s*([\d.]+)\s*t/s,\s*generation:\s*([\d.]+)\s*t/s", err)
    if m:
        out["prefill_tok_per_s"] = float(m.group(1))
        out["generation_tok_per_s"] = float(m.group(2))
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", err)
    if m:
        out["peak_rss_bytes"] = int(m.group(1)) * 1024
    m = re.search(r"Elapsed \(wall clock\) time.*?:\s*([\d:.]+)", err)
    if m:
        out["wall_clock"] = m.group(1)
        parts = [float(x) for x in m.group(1).split(":")]
        secs = 0.0
        for p in parts:
            secs = secs * 60 + p
        out["wall_clock_seconds"] = secs
    for k, pat in [("major_page_faults", r"Major \(requiring I/O\) page faults:\s*(\d+)"),
                   ("minor_page_faults", r"Minor \(reclaiming a frame\) page faults:\s*(\d+)"),
                   ("swaps", r"Swaps:\s*(\d+)")]:
        m = re.search(pat, err)
        if m:
            out[k] = int(m.group(1))
    return out


def main():
    sample = json.loads(sample_json)
    lp_path = os.path.join(run_dir, "outputs", "logprobs.json")
    lp = json.loads(read(lp_path, "{}"))
    steps = lp.get("steps", [])
    gen_ids = [s["selected"]["id"] for s in steps]
    gen_text = "".join(s["selected"]["text"] for s in steps)
    prompt_tokens = lp.get("prompt_tokens")
    needle = sample.get("needle_value")
    is_correct = bool(needle) and (needle in gen_text)
    timing = parse_timing()
    # Derive throughput from wall-clock when ds4's t/s summary is unavailable.
    if "wall_clock_seconds" in timing and timing["wall_clock_seconds"] > 0 and prompt_tokens:
        # decode ~0.5 tok/s dominates little; approximate prefill rate excluding ~load+decode.
        decode_s = len(gen_ids) / 0.5
        prefill_s = max(1.0, timing["wall_clock_seconds"] - decode_s - 60.0)
        timing["derived_prefill_tok_per_s"] = round(prompt_tokens / prefill_s, 3)
        timing["derived_overall_tok_per_s"] = round((prompt_tokens + len(gen_ids)) / timing["wall_clock_seconds"], 3)

    # generations.jsonl
    gen_rec = {
        "run_id": os.path.basename(run_dir),
        "sample_id": sample["sample_id"],
        "prompt_token_count": prompt_tokens,
        "generated_token_count": len(gen_ids),
        "generated_token_ids": gen_ids,
        "generated_text": gen_text,
        "finish_reason": "length" if len(gen_ids) >= sample["target_max_new_tokens"] else "eos",
        "benchmark_prediction": gen_text.strip()[:200],
        "reference_answer": sample["reference_answer"],
        "is_correct": is_correct,
        "score": 1.0 if is_correct else 0.0,
        **timing,
    }
    with open(os.path.join(run_dir, "outputs", "generations.jsonl"), "w") as f:
        f.write(json.dumps(gen_rec) + "\n")

    insp = parse_inspect()
    # model_config.json
    mc = {
        **insp,
        "hidden_size": 4096,
        "csa_compression_ratio": 4,
        "hca_compression_ratio": 128,
        "indexer_head_dim": 128,
        "query_compression_dim": 1024,
        "kv_layout_description": "MLA (kv_heads=1, head_dim=512); CSA ratio-4 + HCA ratio-128 interleaved; SWA window 128 on layers 0-1.",
        "layer_map": layer_map(insp.get("num_layers") or 43),
    }
    json.dump(mc, open(os.path.join(run_dir, "model_config.json"), "w"), indent=2)

    # machine.json
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True).strip()
        except Exception:
            return None
    machine = {
        "hostname": platform.node(),
        "kernel_version": platform.release(),
        "cpu_model": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2 | xargs"),
        "physical_core_count": sh("lscpu | grep '^Core(s) per socket' | cut -d: -f2 | xargs"),
        "socket_count": sh("lscpu | grep '^Socket(s)' | cut -d: -f2 | xargs"),
        "logical_cpu_count": os.cpu_count(),
        "numa_node_count": sh("lscpu | grep 'NUMA node(s)' | cut -d: -f2 | xargs"),
        "total_ram_bytes": int(sh("grep MemTotal /proc/meminfo | awk '{print $2}'") or 0) * 1024,
        "available_ram_bytes_at_finish": int(sh("grep MemAvailable /proc/meminfo | awk '{print $2}'") or 0) * 1024,
        "compiler": sh("gcc --version | head -1"),
    }
    json.dump(machine, open(os.path.join(run_dir, "machine.json"), "w"), indent=2)

    # run_manifest.json
    manifest = {
        "schema_version": "1",
        "trace_schema_version": 2,
        "run_id": os.path.basename(run_dir),
        "created_utc": utc(),
        "backend": "cpu",
        "device_type": "cpu",
        "model_name": "DeepSeek-V4-Flash",
        "model_path": MODEL,
        "model_file_size": os.path.getsize(MODEL) if os.path.exists(MODEL) else None,
        "model_sha256": read(os.path.join(EXP, "model_sha256.txt")).strip() or None,
        "quantization": "IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8 (imatrix)",
        "weight_dtype": "IQ2_XXS/Q2_K mixed", "kv_dtype": "f16", "indexer_qk_dtype": "f16/q8",
        "tokenizer_name": "DeepSeek-V4-Flash (PreTrainedTokenizerFast)",
        "runtime_name": "ds4 (DwarfStar)", "runtime_commit": read(os.path.join(EXP, "code", "ds4_build.txt")).strip()[:200],
        "benchmark_name": "RULER", "benchmark_version": sample.get("benchmark_version"),
        "benchmark_split": "validation", "task_subset": sample.get("task_subtype"),
        "sample_count": 1,
        "context_length_target": length,
        "context_length_actual_tokens": prompt_tokens,
        "max_new_tokens": sample["target_max_new_tokens"],
        "decode_parameters": {"temperature": 0.0, "greedy": True, "seed": 0, "batch_size": 1,
                              "mtp": False, "speculative": False, "prompt_cache": False},
        "thread_count": 64, "numa_policy": read(os.path.join(EXP, "code", "numa_policy.txt")).strip() or "default",
        "environment_variables": {"DS4_TRACE_LEVEL": "3", "DS4_TRACE_DECODE_ONLY": "1",
                                  "DS4_TRACE_FULL_SCORE_SAMPLE_RATE": "0.002", "OMP_NUM_THREADS": "64"},
        "trace_level": 3,
        "timing": timing,
        "is_correct": is_correct,
    }
    json.dump(manifest, open(os.path.join(run_dir, "run_manifest.json"), "w"), indent=2)

    # benchmark_config.json
    bc = {"benchmark": "RULER", "task": sample.get("task_subtype"), "task_group": "retrieval",
          "context_length_target": length, "num_samples": 1, "max_new_tokens": 128,
          "decoding": "greedy", "tokenizer": "DeepSeek-V4-Flash",
          "ruler_commit": sample.get("benchmark_version")}
    json.dump(bc, open(os.path.join(run_dir, "benchmark_config.json"), "w"), indent=2)

    print(f"finalized {os.path.basename(run_dir)}: correct={is_correct} "
          f"gen_tokens={len(gen_ids)} prompt_tokens={prompt_tokens} "
          f"prefill={timing.get('prefill_tok_per_s')}t/s gen={timing.get('generation_tok_per_s')}t/s")


if __name__ == "__main__":
    main()
