#!/usr/bin/env python3
"""exp3 §7 unit tests for the re-index hook, all in ONE model load (runtime mode switch via collective_rpc):
  1. row read-back: swap two rows of both cache groups and read them back (impl B mechanics)
  2. clean -> clean2 -> ctrl_identity -> perm_once(A) -> perm_periodic(A) -> perm_once(B) -> perm_periodic(B)
     on N prompts, 128 greedy tokens each; report first token divergence vs clean and vs clean2 (the noise floor),
     answer correctness (RULER niah), and the perm_log stats (events, pairs, max touched slot vs seq_len-64).
Usage: reindex_unit_test.py --model <path> --manifest <ladder manifest> --out <json> [--n 4] [--tokens 128]
"""
import argparse, glob, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--rung", type=int, default=8192)
    a = ap.parse_args()
    os.environ["REINDEX_INSTALL"] = "1"
    os.environ["REINDEX_MODE"] = "off"
    log_root = os.path.join(os.path.dirname(os.path.abspath(a.out)), "permlog")
    from run_vllm_batch import chat_prompt_ids
    from score import score as score_fn
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    rows = [r for r in rows if r["rung"] == a.rung and r["source"] == "ruler" and r["task"] == "niah_single_2"][: a.n]
    tok = AutoTokenizer.from_pretrained(a.model)
    pdir = os.path.dirname(os.path.abspath(a.manifest))
    prompts = [TokensPrompt(prompt_token_ids=chat_prompt_ids(a.model, tok, open(os.path.join(pdir, r["text_path"])).read(), "chat")[0]) for r in rows]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=a.tokens, min_tokens=a.tokens, ignore_eos=True, seed=42)
    t0 = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=8, enforce_eager=True, trust_remote_code=True, max_model_len=a.rung + 2304,
              enable_prefix_caching=False, seed=42, max_num_seqs=a.n, gpu_memory_utilization=0.9,
              worker_extension_cls="selhook.reindex_ext.ReindexExt")
    res = {"model": a.model, "n": a.n, "tokens": a.tokens, "load_s": round(time.time() - t0, 1), "modes": {}}
    # 1. read-back test
    rb = llm.collective_rpc("reindex_readback_test")
    res["readback"] = rb[0]
    print("[readback]", rb[0], flush=True)
    modes = [("clean", "off", "A"), ("clean2", "off", "A"), ("ctrl_identity", "ctrl_identity", "A"),
             ("perm_once_A", "perm_once", "A"), ("perm_periodic_A", "perm_periodic", "A"),
             ("perm_once_B", "perm_once", "B"), ("perm_periodic_B", "perm_periodic", "B")]
    for name, mode, impl in modes:
        os.makedirs(os.path.join(log_root, name), exist_ok=True)
        llm.collective_rpc("reindex_set", kwargs={"mode": mode, "impl": impl, "log_dir": os.path.join(log_root, name)})
        outs = llm.generate(prompts, sp, use_tqdm=False)
        info = llm.collective_rpc("reindex_info")[0]
        toks = [list(o.outputs[0].token_ids) for o in outs]
        texts = [o.outputs[0].text for o in outs]
        corr = [score_fn(r["source"], r["task"], t, r["reference"])[0] for r, t in zip(rows, texts)]
        res["modes"][name] = {"mode": mode, "impl": impl, "token_ids": toks, "correct": corr, "n_events": info["n_events"],
                              "n_swaps": info["n_swaps"], "errors": info["errors"], "text_head": [t[:60] for t in texts]}
        print(f"[{name}] correct={corr} events={info['n_events']} swaps={info['n_swaps']} errors={info['errors']}", flush=True)
        # perm-log stats
        logs = glob.glob(os.path.join(log_root, name, "*.jsonl"))
        ev, maxunit, minmargin, moved = 0, -1, None, 0
        for lf in logs:
            for line in open(lf):
                d = json.loads(line)
                if "n_units" not in d:
                    continue
                ev += 1; moved += d["moved"]
                blk = 64 if d["impl"] == "A" else 1
                top = (d["n_units"] - 1) * blk + (blk - 1)
                maxunit = max(maxunit, top)
                m = d["seq_len"] - 64 - top
                minmargin = m if minmargin is None else min(minmargin, m)
        res["modes"][name]["permlog"] = {"files": len(logs), "events": ev, "moved_units": moved,
                                         "max_touched_slot": maxunit, "min_margin_to_seq_len_minus_64": minmargin}

    def div(x, y):
        return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)

    base = res["modes"]["clean"]["token_ids"]
    for name in res["modes"]:
        res["modes"][name]["first_divergence_vs_clean"] = [div(b, t) for b, t in zip(base, res["modes"][name]["token_ids"])]
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: {"correct": v["correct"], "div_vs_clean": v["first_divergence_vs_clean"], "events": v["n_events"],
                          "permlog": v.get("permlog"), "errors": v["errors"]} for k, v in res["modes"].items()}, indent=1))


if __name__ == "__main__":
    main()
