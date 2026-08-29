#!/usr/bin/env python3
"""Run one rung of the ladder (exp1 §5 / exp2 §4) on vLLM TP8 with the selection hook, then adapt every request
into a ds4-schema run directory and run the unchanged analysis chain.

Usage: run_vllm_batch.py --model <path> --model-tag v32 --manifest <prompts>/manifest.jsonl --rung 8192
         --kinds bf ld --out-root <WORKDIR>/runs [--max-num-seqs 16] [--sample-ids id ...] [--no-hook]
         [--thinking chat|thinking] [--attribution-check] [--max-model-len N] [--skip-analysis] [--gpu-mem 0.9]
Per request: <out-root>/<run_id>/{req.json, meta.json, + ds4 files by the adapter, analysis/}
Batch summary: <out-root>/batch_<model-tag>_<rung>_<kinds>.json
run_id = <model-tag>_<sample_id>  (sample_id already encodes source/task/rung/kind/index).
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def chat_prompt_ids(model_path, tok, text, thinking):
    """Return prompt token ids using the model's own chat formatting (thinking off unless requested)."""
    enc = os.path.join(model_path, "encoding", "encoding_dsv32.py")
    if os.path.exists(enc):
        import importlib.util
        spec = importlib.util.spec_from_file_location("encoding_dsv32", enc)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        s = m.encode_messages([{"role": "user", "content": text}], thinking_mode=thinking)
        return tok(s, add_special_tokens=False)["input_ids"], s[:200]
    # GLM: jinja chat template with enable_thinking flag
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                  tokenize=True, enable_thinking=(thinking == "thinking"))
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rung", type=int, required=True)
    ap.add_argument("--kinds", nargs="+", default=["bf", "ld"])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--sample-ids", nargs="*", default=None)
    ap.add_argument("--no-hook", action="store_true")
    ap.add_argument("--thinking", default="chat")
    ap.add_argument("--attribution-check", action="store_true",
                    help="also run the first bf prompt alone (batch 1) under a second request id")
    ap.add_argument("--skip-analysis", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--computing-only", action="store_true", help="exp2 variant (b) adapter flag (also writes (a))")
    ap.add_argument("--force-tokens", type=int, default=None, help="smoke: force exactly N decode tokens for every prompt")
    ap.add_argument("--run-suffix", default="", help="appended to run ids / trace dir (e.g. _smoke)")
    ap.add_argument("--logprobs", type=int, default=None, help="request top-N logprobs per step (exp3 agreement sample)")
    ap.add_argument("--reindex", action="store_true", help="install selhook.reindex_ext.ReindexExt (REINDEX_* env)")
    ap.add_argument("--no-adapter", action="store_true", help="skip the ds4 adapter (exp3 accuracy-only runs)")
    ap.add_argument("--source-filter", default=None, help="only prompts of this source (e.g. ruler)")
    ap.add_argument("--sample-ids-file", default=None, help="file with one sample_id per line to keep")
    a = ap.parse_args()

    node = json.load(open(os.path.join(os.environ["WORKDIR"], "node.json")))
    mcfg = node["models"][os.path.basename(a.model.rstrip("/"))]
    model_name = a.model_name or os.path.basename(a.model.rstrip("/")).replace("-FP8", "")
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    rows = [r for r in rows if r["rung"] == a.rung and r["kind"] in a.kinds]
    if a.sample_ids:
        rows = [r for r in rows if r["sample_id"] in a.sample_ids]
    if a.source_filter:
        rows = [r for r in rows if r["source"] == a.source_filter]
    if a.sample_ids_file:
        keep = {l.strip() for l in open(a.sample_ids_file) if l.strip()}
        rows = [r for r in rows if r["sample_id"] in keep]
    if not rows:
        print("no prompts selected"); return
    pdir = os.path.dirname(os.path.abspath(a.manifest))
    trace_dir = os.path.join(a.out_root, f"trace_{a.model_tag}_{a.rung}_{'_'.join(a.kinds)}{a.run_suffix}")
    if not a.no_hook:
        os.environ["SEL_TRACE"] = trace_dir
        os.makedirs(trace_dir, exist_ok=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)

    prompts, sps, reqs = [], [], []
    for r in rows:
        text = open(os.path.join(pdir, r["text_path"])).read()
        thinking = r.get("thinking", a.thinking)
        ids, head = chat_prompt_ids(a.model, tok, text, thinking)
        lp = {"logprobs": a.logprobs} if a.logprobs else {}
        if a.force_tokens:
            sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=a.force_tokens, min_tokens=a.force_tokens, ignore_eos=True, seed=a.seed, **lp)
        elif r["kind"] == "ld":
            sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048, min_tokens=2048, ignore_eos=True, seed=a.seed, **lp)
        else:
            sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=r["max_new_tokens"], seed=a.seed, **lp)
        prompts.append(TokensPrompt(prompt_token_ids=ids)); sps.append(sp)
        reqs.append({"sample_id": r["sample_id"], "manifest": r, "prompt_token_count": len(ids), "template_head": head})
    if a.attribution_check:
        bf0 = next((i for i, r in enumerate(rows) if r["kind"] == "bf"), 0)
        reqs.append({"sample_id": rows[bf0]["sample_id"] + "_solo", "manifest": rows[bf0],
                     "prompt_token_count": reqs[bf0]["prompt_token_count"], "solo": True})
    max_len = a.max_model_len or (a.rung + 2304)
    max_len = max(max_len, max(q["prompt_token_count"] for q in reqs) + 2100)
    kw = {} if (a.no_hook and not a.reindex) else {"worker_extension_cls": "selhook.reindex_ext.ReindexExt" if a.reindex else "selhook.worker_ext.SelHookExt"}
    t0 = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=8, enforce_eager=True, trust_remote_code=True,
              max_model_len=max_len, enable_prefix_caching=False, seed=a.seed, max_num_seqs=a.max_num_seqs,
              gpu_memory_utilization=a.gpu_mem, **kw)
    load_s = time.time() - t0
    print(f"[load] {load_s:.0f}s max_model_len={max_len} n_prompts={len(prompts)} max_num_seqs={a.max_num_seqs}", flush=True)

    def run(prompt_list, sp_list, id_list):
        t = time.time()
        outs = llm.generate(prompt_list, sp_list, use_tqdm=True, request_id=id_list) if _supports_request_id(llm) \
            else llm.generate(prompt_list, sp_list, use_tqdm=True)
        return outs, time.time() - t

    ids_main = [q["sample_id"] for q in reqs if not q.get("solo")]
    outs, gen_s = run(prompts, sps, ids_main)
    outs_solo, gen_solo_s = [], 0.0
    if a.attribution_check:
        outs_solo, gen_solo_s = run([prompts[bf0]], [sps[bf0]], [reqs[-1]["sample_id"]])
    if not a.no_hook:
        info = llm.collective_rpc("selhook_flush")
        print("[selhook]", info[0] if info else info, flush=True)
    if a.reindex:
        info = llm.collective_rpc("reindex_info")
        print("[reindex]", info[0] if info else info, flush=True)
        summary_reindex = info[0] if info else None
    wall = time.time() - t0
    from score import score as score_fn
    summary = {"model": model_name, "model_tag": a.model_tag, "rung": a.rung, "kinds": a.kinds, "n": len(reqs),
               "reindex": (summary_reindex if a.reindex else None), "reindex_env": {k: v for k, v in os.environ.items() if k.startswith("REINDEX_")},
               "load_seconds": round(load_s, 1), "gen_seconds": round(gen_s + gen_solo_s, 1),
               "wall_seconds": round(wall, 1), "gpu_hours": round(8 * wall / 3600, 3), "max_model_len": max_len,
               "max_num_seqs": a.max_num_seqs, "thinking": a.thinking, "trace_dir": trace_dir, "runs": []}
    all_outs = list(zip([q for q in reqs if not q.get("solo")], outs)) + list(zip([q for q in reqs if q.get("solo")], outs_solo))
    for q, o in all_outs:
        m = q["manifest"]
        c = o.outputs[0]
        is_ok, sc, extracted = score_fn(m["source"], m["task"], c.text, m["reference"], m.get("meta"))
        rid_engine = o.request_id
        run_id = f"{a.model_tag}_{q['sample_id']}{a.run_suffix}"
        rd = os.path.join(a.out_root, run_id)
        os.makedirs(rd, exist_ok=True)
        req = {"request_id": rid_engine, "sample_id": q["sample_id"], "prompt_token_count": len(o.prompt_token_ids),
               "generated_token_ids": list(c.token_ids), "generated_text": c.text, "finish_reason": c.finish_reason,
               "benchmark_prediction": extracted, "reference_answer": m["reference"], "is_correct": is_ok, "score": sc,
               "source": m["source"], "task": m["task"], "kind": m["kind"], "rung": m["rung"], "solo": bool(q.get("solo"))}
        if a.logprobs and c.logprobs:
            req["top_logprobs"] = [[(int(t), round(float(l.logprob), 5)) for t, l in sorted(d.items(), key=lambda kv: -kv[1].logprob)[: a.logprobs]] for d in c.logprobs]
        req["thinking"] = q["manifest"].get("thinking", a.thinking)
        json.dump(req, open(os.path.join(rd, "req.json"), "w"))
        meta = {"run_id": run_id, "model_name": model_name, "model_path": a.model.replace(os.environ["WORKDIR"], "<WORKDIR>"),
                "benchmark_name": m["source"], "task_subset": m["task"], "context_length_target": a.rung,
                "max_new_tokens": a.force_tokens or (2048 if m["kind"] == "ld" else m["max_new_tokens"]), "batch_size": 1 if q.get("solo") else len(prompts),
                "vllm_commit": node["vllm_commit"], "gpu": node["gpu"], "wall_clock_seconds": round(wall, 1),
                "gpu_hours": round(8 * wall / 3600 / max(1, len(reqs)), 4), "seed": a.seed, "kind": m["kind"],
                "num_layers": mcfg["num_hidden_layers"], "top_k": mcfg["index_topk"], "index_n_heads": mcfg["index_n_heads"],
                "index_head_dim": mcfg["index_head_dim"], "expert_count": mcfg["n_routed_experts"],
                "expert_used": mcfg["num_experts_per_tok"], "first_k_dense": mcfg["first_k_dense_replace"],
                "index_topk_freq": mcfg["index_topk_freq"], "index_skip_topk_offset": mcfg["index_skip_topk_offset"],
                "index_topk_pattern": mcfg["index_topk_pattern"], "computed_layers": mcfg["skip_rule"]["computing_layers"],
                "thinking": a.thinking, "max_num_seqs": a.max_num_seqs}
        json.dump(meta, open(os.path.join(rd, "meta.json"), "w"), indent=1)
        summary["runs"].append({"run_id": run_id, "sample_id": q["sample_id"], "source": m["source"], "task": m["task"],
                                "kind": m["kind"], "prompt_tokens": req["prompt_token_count"], "gen_tokens": len(c.token_ids),
                                "finish": c.finish_reason, "is_correct": is_ok, "score": round(sc, 4), "request_id": rid_engine})
        print(f"[out] {run_id}: {req['prompt_token_count']} -> {len(c.token_ids)} tok, {c.finish_reason}, correct={is_ok} score={sc:.3f} :: {c.text[:80]!r}", flush=True)
    sp = os.path.join(a.out_root, f"batch_{a.model_tag}_{a.rung}_{'_'.join(a.kinds)}{a.run_suffix}.json")
    json.dump(summary, open(sp, "w"), indent=1)
    print(f"[batch] wrote {sp}; gen {gen_s:.0f}s wall {wall:.0f}s = {summary['gpu_hours']} GPU-h", flush=True)
    del llm
    if a.skip_analysis or a.no_hook or a.no_adapter:
        return
    # ---- adapter + unchanged analysis chain, per request (sequential; CPU-bound) ----
    py = sys.executable
    cmds = []
    for r in summary["runs"]:
        rd = os.path.join(a.out_root, r["run_id"])
        variants = [("", [])] + ([("_b", ["--computing-only"])] if a.computing_only else [])
        for suffix, flags in variants:
            out_rd = rd if not suffix else rd + suffix
            cmds.append([py, os.path.join(HERE, "vllm_to_ds4_run.py"), "--trace", trace_dir, "--req-json", os.path.join(rd, "req.json"),
                         "--meta-json", os.path.join(rd, "meta.json"), "--run-dir", out_rd] + flags)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=12) as ex:
        for res in ex.map(lambda c: subprocess.run(c, capture_output=True, text=True), cmds):
            print(res.stdout.strip()[-200:], flush=True)
            if res.returncode != 0:
                print("[adapter ERROR]", res.stderr[-500:], flush=True)
    print("[done] adapter finished; run analyze_runs.sh for the chain", flush=True)


def _supports_request_id(llm):
    import inspect
    try:
        return "request_id" in inspect.signature(llm.generate).parameters
    except Exception:
        return False


if __name__ == "__main__":
    main()
