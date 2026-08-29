#!/usr/bin/env python3
"""exp3 tier 1 for one model, in ONE engine load (modes switched via collective_rpc):
  PPL block  : long documents -> prefix P (32K/64K/128K) written to the KV cache, then the prefix is re-indexed
               (hook fires at the prefill-chunk boundary, max_num_batched_tokens = 2048 so the boundary is exact),
               then a 2K continuation is scored teacher-forced (prompt_logprobs). PPL = exp(-mean logprob).
               WikiText-2 windows (1K prefix / 1K scored) as the short anchor.
  ACC block  : RULER niah_single_2 + qa_1 (32K, 128K) generated greedily with the decode-step hook (perm_periodic
               needs decode steps), scored by scripts/gpu/score.py.
Modes: clean, clean2, ctrl_identity, ctrl_numeric (same item processed alongside a filler request), perm_once_A,
perm_once_B, perm_periodic_B (ACC only). Which modes run at which length is given by --plan (default = tier 1).
Usage: exp3_tier1.py --model <path> --tag v32 --out <json> [--n-docs 10] [--n-wt 50] [--n-ruler 5]
"""
import argparse, json, math, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TIER1_PLAN = {32768: ["clean", "clean2", "ctrl_identity", "ctrl_numeric", "perm_once_A", "perm_once_B"],
              65536: ["clean", "clean2", "perm_once_A", "perm_once_B"],
              131072: ["clean", "clean2", "perm_once_B"],
              2048: ["clean", "clean2", "ctrl_identity", "ctrl_numeric", "perm_once_A", "perm_once_B"]}   # 2048 = WikiText-2 prefix (chunk boundary), 1K scored
MODE_CFG = {"clean": ("off", "A"), "clean2": ("off", "A"), "ctrl_identity": ("ctrl_identity", "B"), "ctrl_numeric": ("off", "A"),
            "perm_once_A": ("perm_once", "A"), "perm_once_B": ("perm_once", "B"), "perm_periodic_B": ("perm_periodic", "B")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, default=10)
    ap.add_argument("--n-wt", type=int, default=50)
    ap.add_argument("--n-ruler", type=int, default=5)
    ap.add_argument("--cont", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=2048, help="max_num_batched_tokens (prefill chunk = boundary granularity)")
    ap.add_argument("--plan", default=None, help="JSON {prefix_len: [modes]} overriding the tier-1 plan")
    ap.add_argument("--acc-modes", nargs="*", default=["clean", "perm_once_B", "perm_periodic_B"])
    ap.add_argument("--skip-acc", action="store_true")
    ap.add_argument("--skip-ppl", action="store_true")
    a = ap.parse_args()
    W = os.environ["WORKDIR"]
    os.environ["REINDEX_INSTALL"] = "1"; os.environ["REINDEX_MODE"] = "off"
    plan = {int(k): v for k, v in json.loads(a.plan).items()} if a.plan else TIER1_PLAN
    from run_vllm_batch import chat_prompt_ids
    from score import score as score_fn
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)
    random.seed(42)
    log_root = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"permlog_{a.tag}")
    res = {"model": a.model, "tag": a.tag, "plan": {str(k): v for k, v in plan.items()}, "ppl": [], "acc": [], "timing": {}}

    # ---------- items: long documents (InfiniteBench books, distinct) ----------
    books, seen = [], set()
    for task in ("longbook_sum_eng", "longbook_qa_eng", "longbook_choice_eng"):
        for l in open(os.path.join(W, "benchmark", "data", "InfiniteBench", task + ".jsonl")):
            r = json.loads(l); key = r["context"][:2000]
            if key in seen:
                continue
            seen.add(key); books.append(r["context"])
    random.shuffle(books)
    docs = []
    for ctx in books:
        ids = tok(ctx, add_special_tokens=False)["input_ids"]
        if len(ids) >= 131072 + a.cont + 16:
            docs.append(ids)
        if len(docs) >= a.n_docs:
            break
    print(f"[items] {len(docs)} books >= 133K tokens (from {len(books)} distinct)", flush=True)
    wt = tok(open(os.path.join(W, "benchmark", "data", "wikitext2_test.txt")).read(), add_special_tokens=False)["input_ids"]
    wt_windows = [wt[i:i + 3072] for i in range(0, len(wt) - 3072, 3072)][: a.n_wt]
    bos = tok("", add_special_tokens=True)["input_ids"]  # model's BOS if any

    def ppl_prompt(ids, prefix_len, cont):
        seq = (bos if bos and bos[0] != ids[0] else []) + ids[: prefix_len + cont]
        return TokensPrompt(prompt_token_ids=seq), len(seq) - cont

    t0 = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=8, enforce_eager=True, trust_remote_code=True, max_model_len=131072 + a.cont + 4096,
              enable_prefix_caching=False, seed=42, max_num_seqs=2, max_num_batched_tokens=a.chunk, gpu_memory_utilization=0.9,
              worker_extension_cls="selhook.reindex_ext.ReindexExt")
    res["timing"]["load_s"] = round(time.time() - t0, 1)
    print(f"[load] {res['timing']['load_s']}s", flush=True)
    sp_ppl = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, seed=42)
    filler = TokensPrompt(prompt_token_ids=docs[-1][:4096]) if docs else None

    def mode_cfg(mode):
        """'clean', 'clean2', 'clean3'... -> off; '<name>@<seed>' -> that permutation seed (default 7)."""
        base, _, seed = mode.partition("@")
        if base.startswith("clean"):
            m, impl = "off", "A"
        else:
            m, impl = MODE_CFG[base]
        return m, impl, (int(seed) if seed else 7)

    def set_mode(mode, prefix_len):
        m, impl, seed = mode_cfg(mode)
        d = os.path.join(log_root, f"{mode}_{prefix_len}".replace("@", "_s"))
        os.makedirs(d, exist_ok=True)
        llm.collective_rpc("reindex_set", kwargs={"mode": m, "impl": impl, "seed": seed, "log_dir": d, "prefix_len": prefix_len if m != "off" else 0})

    def score_ppl(out, n_prefix):
        lps = out.prompt_logprobs
        vals = []
        for i in range(n_prefix, len(lps)):
            d = lps[i]
            tid = out.prompt_token_ids[i]
            if d and tid in d:
                vals.append(d[tid].logprob)
        return vals

    # ---------- PPL block ----------
    t_ppl = time.time()
    for prefix_len in (sorted(plan) if not a.skip_ppl else []):
        items = wt_windows if prefix_len == 2048 else docs
        cont = 1024 if prefix_len == 2048 else a.cont
        for mode in plan[prefix_len]:
            set_mode(mode, prefix_len)
            t1 = time.time()
            for di, ids in enumerate(items):
                prompt, n_prefix = ppl_prompt(ids, prefix_len, cont)
                if mode.startswith("ctrl_numeric") and filler is not None:
                    outs = llm.generate([prompt, filler], [sp_ppl, SamplingParams(temperature=0.0, max_tokens=1, seed=42)], use_tqdm=False)
                    out = outs[0]
                else:
                    out = llm.generate([prompt], sp_ppl, use_tqdm=False)[0]
                vals = score_ppl(out, n_prefix)
                info = llm.collective_rpc("reindex_info")[0]
                res["ppl"].append({"mode": mode, "prefix_len": prefix_len, "doc": di, "n_scored": len(vals),
                                   "mean_logprob": (sum(vals) / len(vals)) if vals else None,
                                   "ppl": (math.exp(-sum(vals) / len(vals)) if vals else None), "logprobs": [round(v, 5) for v in vals],
                                   "events": info["n_events"], "errors": info["errors"][:1]})
            n_items = len(items)
            m = [r for r in res["ppl"] if r["mode"] == mode and r["prefix_len"] == prefix_len]
            print(f"[ppl] prefix {prefix_len} {mode:>14}: n={n_items} mean PPL {sum(r['ppl'] for r in m)/len(m):.4f} "
                  f"({time.time()-t1:.0f}s, events so far {info['n_events']}, errors {info['errors'][:1]})", flush=True)
            json.dump(res, open(a.out, "w"))
    res["timing"]["ppl_s"] = round(time.time() - t_ppl, 1)

    # ---------- ACC block (RULER generation) ----------
    if not a.skip_acc:
        t_acc = time.time()
        man = [json.loads(l) for l in open(os.path.join(W, "prompts", "exp3", "manifest.jsonl")) if l.strip()]
        pdir = os.path.join(W, "prompts", "exp3")
        sel = []
        for L in (32768, 131072):
            for task in ("niah_single_2", "qa_1"):
                sel += [r for r in man if r["source"] == "ruler" and r["task"] == task and r["rung"] == L][: a.n_ruler]
        prompts, metas = [], []
        for r in sel:
            ids, _ = chat_prompt_ids(a.model, tok, open(os.path.join(pdir, r["text_path"])).read(), "chat")
            prompts.append(TokensPrompt(prompt_token_ids=ids)); metas.append(r)
        for mode in a.acc_modes:
            m, impl, seed = mode_cfg(mode)
            d = os.path.join(log_root, f"acc_{mode}".replace("@", "_s")); os.makedirs(d, exist_ok=True)
            llm.collective_rpc("reindex_set", kwargs={"mode": m, "impl": impl, "seed": seed, "log_dir": d, "prefix_len": 0})
            t1 = time.time()
            for p, r in zip(prompts, metas):
                sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=r["max_new_tokens"], seed=42, logprobs=1)
                out = llm.generate([p], sp, use_tqdm=False)[0]
                c = out.outputs[0]
                ok, sc, ex = score_fn(r["source"], r["task"], c.text, r["reference"])
                info = llm.collective_rpc("reindex_info")[0]
                res["acc"].append({"mode": mode, "sample_id": r["sample_id"], "task": r["task"], "rung": r["rung"], "correct": ok, "score": sc,
                                   "n_tokens": len(c.token_ids), "token_ids": list(c.token_ids), "text": c.text[:200], "events": info["n_events"],
                                   "errors": info["errors"][:1]})
            got = [x for x in res["acc"] if x["mode"] == mode]
            print(f"[acc] {mode:>14}: correct {sum(1 for x in got if x['correct'])}/{len(got)} ({time.time()-t1:.0f}s)", flush=True)
            json.dump(res, open(a.out, "w"))
        res["timing"]["acc_s"] = round(time.time() - t_acc, 1)
    res["timing"]["total_s"] = round(time.time() - t0, 1)
    res["timing"]["gpu_hours"] = round(8 * res["timing"]["total_s"] / 3600, 2)
    json.dump(res, open(a.out, "w"))
    print("[done]", res["timing"], flush=True)


if __name__ == "__main__":
    main()
