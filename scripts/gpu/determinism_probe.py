#!/usr/bin/env python3
"""exp3 precondition probe: is greedy decode bit-reproducible run-to-run (and across batch compositions)?
Usage: determinism_probe.py --model <path> --prompt-file <txt> --out <json> [--n-tokens 512] [--repeats 3]
                            [--batch-copies 4] [--max-num-seqs 8] [--enforce-eager 1] [--env KEY=VAL ...]
Runs the SAME prompt: (a) solo, `repeats` times in one engine; (b) in a batch with `batch-copies` other prompts
(distinct filler prompts) once. Reports first divergence steps between all solo repeats and solo-vs-batch, plus the
per-step top-1 logprob agreement. --env sets environment variables BEFORE vLLM import (kernel backend switches);
the LLM kwargs are the campaign's (TP8, prefix caching off, seed 42).
"""
import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--filler-files", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--enforce-eager", type=int, default=1)
    ap.add_argument("--thinking", default="chat")
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--extra-kwargs", default="{}", help="JSON of extra LLM() kwargs")
    a = ap.parse_args()
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_vllm_batch import chat_prompt_ids
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)
    ids, _ = chat_prompt_ids(a.model, tok, open(a.prompt_file).read(), a.thinking)
    fillers = [chat_prompt_ids(a.model, tok, open(f).read(), a.thinking)[0] for f in a.filler_files]
    kw = json.loads(a.extra_kwargs)
    t0 = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=8, enforce_eager=bool(a.enforce_eager), trust_remote_code=True,
              max_model_len=max(len(ids), *(len(f) for f in fillers)) + a.n_tokens + 256 if fillers else len(ids) + a.n_tokens + 256,
              enable_prefix_caching=False, seed=42, max_num_seqs=a.max_num_seqs, gpu_memory_utilization=0.9, **kw)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=a.n_tokens, min_tokens=a.n_tokens, ignore_eos=True,
                        seed=42, logprobs=1)

    def toks(o):
        c = o.outputs[0]
        lp = [max(d.values(), key=lambda x: x.logprob).logprob if d else None for d in (c.logprobs or [])]
        return list(c.token_ids), lp

    res = {"model": a.model, "env": a.env, "extra_kwargs": kw, "enforce_eager": a.enforce_eager, "n_tokens": a.n_tokens,
           "prompt_tokens": len(ids), "load_s": round(time.time() - t0, 1), "solo": [], "batch": None}
    for r in range(a.repeats):
        o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
        t, lp = toks(o)
        res["solo"].append({"token_ids": t, "top1_logprob": lp})
    if fillers:
        outs = llm.generate([TokensPrompt(prompt_token_ids=ids)] + [TokensPrompt(prompt_token_ids=f) for f in fillers],
                            [sp] * (1 + len(fillers)), use_tqdm=False)
        t, lp = toks(outs[0])
        res["batch"] = {"token_ids": t, "top1_logprob": lp, "batch_size": 1 + len(fillers)}

    def div(x, y):
        return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)

    base = res["solo"][0]["token_ids"]
    res["solo_first_divergence"] = [div(base, s["token_ids"]) for s in res["solo"][1:]]
    res["solo_identical"] = all(d is None for d in res["solo_first_divergence"])
    if res["batch"]:
        res["batch_vs_solo_first_divergence"] = div(base, res["batch"]["token_ids"])
    lp0 = res["solo"][0]["top1_logprob"]
    res["solo_max_abs_dlogprob_top1"] = [max((abs(p - q) for p, q in zip(lp0, s["top1_logprob"]) if p is not None and q is not None), default=None)
                                         for s in res["solo"][1:]]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("solo", "batch")}, indent=1))


if __name__ == "__main__":
    main()
