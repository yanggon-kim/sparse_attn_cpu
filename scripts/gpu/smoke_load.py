#!/usr/bin/env python3
"""exp0 §4 smoke: load a model at TP8 (eager, no prefix caching), generate N greedy tokens, twice.
Usage: smoke_load.py --model <path> --out <json> [--max-model-len 16384] [--n 3] [--prompt "..."]
                     [--hook-trace <dir>]   (installs the selection hook via worker_extension_cls)
Writes <json>: load_seconds, gpu_mem_used_mib per GPU after load, token ids of both generations, identical flag.
"""
import argparse, json, os, subprocess, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--hook-trace", default=None)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--routed-experts", action="store_true")
    args = ap.parse_args()

    if args.hook_trace:
        os.environ["SEL_TRACE"] = args.hook_trace

    from vllm import LLM, SamplingParams  # noqa: E402

    kw = {}
    if args.routed_experts:
        kw["enable_return_routed_experts"] = True
    if args.hook_trace:
        kw["worker_extension_cls"] = "selhook.worker_ext.SelHookExt"

    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=8, enforce_eager=True, trust_remote_code=True,
              max_model_len=args.max_model_len, enable_prefix_caching=False, seed=42,
              gpu_memory_utilization=0.90, **kw)
    load_s = time.time() - t0
    mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.n, seed=42)
    gens = []
    for i in range(args.repeat):
        out = llm.generate([args.prompt], sp)
        gens.append({"text": out[0].outputs[0].text, "token_ids": list(out[0].outputs[0].token_ids),
                     "request_id": out[0].request_id})
        g = gens[-1]
        print(f"[gen {i}] text={g['text']!r} ids={g['token_ids']} re_shape={g.get('routed_experts_shape')}", flush=True)
    res = {"model": args.model, "load_seconds": round(load_s, 1), "gpu_mem_used_mib": [int(m) for m in mem],
           "max_model_len": args.max_model_len, "n": args.n, "prompt": args.prompt, "gens": gens,
           "identical": all(g["token_ids"] == gens[0]["token_ids"] for g in gens),
       "worker_extension_cls": kw.get("worker_extension_cls"),
           "hook_trace": args.hook_trace}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
