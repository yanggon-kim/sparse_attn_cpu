#!/usr/bin/env python3
"""exp3 tier 2 for one model, ONE engine load (modes switched via collective_rpc). Superset of exp3_tier1.py:
  PPL block  : 30 long documents (InfiniteBench books) -> prefix P in {32K, 64K, 128K}, re-indexed at the exact
               prefill-chunk boundary (max_num_batched_tokens = 2048), 2K continuation scored teacher-forced;
               ALL modes at EVERY prefix: clean, clean2, ctrl_identity, ctrl_numeric, perm_once_A/B x seeds 7,8,9.
               Short anchors: WikiText-2 (150 windows) and PTB (all windows), 2K prefix / 1K scored, same modes.
  ACC block  : RULER niah_single_2, niah_multikey_2, vt, qa_1 x {32K, 64K, 128K} x 25 items + LongBench-v2 100 items
               (stratified over the six domains), greedy generation, modes clean / perm_once_B / perm_periodic_B.
Resumable (--resume): completed (mode, prefix, corpus) / (mode) groups in the output JSON are skipped.
Usage: exp3_tier2.py --model <path> --tag v32 --out <json> [--resume] [--skip-ppl] [--skip-acc]
"""
import argparse, json, math, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ALL_MODES = ["clean", "clean2", "ctrl_identity", "ctrl_numeric", "perm_once_A", "perm_once_A@8", "perm_once_A@9",
             "perm_once_B", "perm_once_B@8", "perm_once_B@9"]
TIER2_PLAN = {32768: ALL_MODES, 65536: ALL_MODES, 131072: ALL_MODES, 2048: ALL_MODES}
MODE_CFG = {"clean": ("off", "A"), "clean2": ("off", "A"), "ctrl_identity": ("ctrl_identity", "B"), "ctrl_numeric": ("off", "A"),
            "perm_once_A": ("perm_once", "A"), "perm_once_B": ("perm_once", "B"), "perm_periodic_B": ("perm_periodic", "B")}
RULER_TASKS = ["niah_single_2", "niah_multikey_2", "vt", "qa_1"]
RULER_RUNGS = [32768, 65536, 131072]
VT_MAX_NEW = 256   # RULER's vt default (30) cuts chat models off mid-trace (the exp3 prompts carry no "Answer:" prefix)


def max_new_for(r):
    return max(r["max_new_tokens"], VT_MAX_NEW) if r["source"] == "ruler" and r["task"] == "vt" else r["max_new_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, default=30)
    ap.add_argument("--n-wt", type=int, default=150)
    ap.add_argument("--n-ptb", type=int, default=10**9)
    ap.add_argument("--n-ruler", type=int, default=25)
    ap.add_argument("--n-lb2", type=int, default=100)
    ap.add_argument("--cont", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--plan", default=None, help="JSON {prefix_len: [modes]} overriding the tier-2 plan")
    ap.add_argument("--acc-modes", nargs="*", default=["clean", "perm_once_B", "perm_periodic_B"])
    ap.add_argument("--skip-acc", action="store_true")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    W = os.environ["WORKDIR"]
    os.environ["REINDEX_INSTALL"] = "1"; os.environ["REINDEX_MODE"] = "off"
    plan = {int(k): v for k, v in json.loads(a.plan).items()} if a.plan else TIER2_PLAN
    from run_vllm_batch import chat_prompt_ids
    from score import score as score_fn
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)
    random.seed(42)
    log_root = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"permlog_{a.tag}")
    res = {"model": a.model, "tag": a.tag, "tier": 2, "plan": {str(k): v for k, v in plan.items()}, "ppl": [], "acc": [], "timing": {}}
    if a.resume and os.path.exists(a.out):
        old = json.load(open(a.out))
        res["ppl"], res["acc"] = old.get("ppl", []), old.get("acc", [])
        n0 = len(res["acc"])
        res["acc"] = [r for r in res["acc"] if not (r.get("task") == "vt" and r.get("max_new_tokens") != VT_MAX_NEW)]
        if n0 != len(res["acc"]):
            print(f"[resume] dropped {n0 - len(res['acc'])} vt rows generated with the 30-token cap", flush=True)
        res["timing"]["resumed_from"] = old.get("timing", {})
        print(f"[resume] {len(res['ppl'])} ppl rows, {len(res['acc'])} acc rows already present", flush=True)

    # ---------- items: long documents (same selection as tier 1: seed 42 shuffle, first n_docs books >= 133K tokens) ----------
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

    def windows(path, n):
        ids = tok(open(path).read(), add_special_tokens=False)["input_ids"]
        return [ids[i:i + 3072] for i in range(0, len(ids) - 3072, 3072)][:n]
    short = {"wikitext2": windows(os.path.join(W, "benchmark", "data", "wikitext2_test.txt"), a.n_wt),
             "ptb": windows(os.path.join(W, "benchmark", "data", "ptb_test.txt"), a.n_ptb)}
    print(f"[items] short windows: " + ", ".join(f"{k} {len(v)}" for k, v in short.items()), flush=True)
    bos = tok("", add_special_tokens=True)["input_ids"]

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
        base, _, seed = mode.partition("@")
        if base.startswith("clean"):
            m, impl = "off", "A"
        else:
            m, impl = MODE_CFG[base]
        return m, impl, (int(seed) if seed else 7)

    def set_mode(mode, prefix_len, corpus=None):
        m, impl, seed = mode_cfg(mode)
        d = os.path.join(log_root, f"{mode}_{prefix_len}" + (f"_{corpus}" if corpus else "")).replace("@", "_s")
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
        groups = [(c, w, 1024) for c, w in short.items()] if prefix_len == 2048 else [("longbook", docs, a.cont)]
        for corpus, items, cont in groups:
            for mode in plan[prefix_len]:
                have = [r for r in res["ppl"] if r["mode"] == mode and r["prefix_len"] == prefix_len and r.get("corpus", "longbook" if prefix_len != 2048 else "wikitext2") == corpus]
                if len(have) >= len(items):
                    print(f"[ppl] prefix {prefix_len} {corpus} {mode:>14}: skip (have {len(have)})", flush=True)
                    continue
                set_mode(mode, prefix_len, corpus if prefix_len == 2048 else None)
                t1 = time.time()
                done = {r["doc"] for r in have}
                for di, ids in enumerate(items):
                    if di in done:
                        continue
                    prompt, n_prefix = ppl_prompt(ids, prefix_len, cont)
                    if mode.startswith("ctrl_numeric") and filler is not None:
                        outs = llm.generate([prompt, filler], [sp_ppl, SamplingParams(temperature=0.0, max_tokens=1, seed=42)], use_tqdm=False)
                        out = outs[0]
                    else:
                        out = llm.generate([prompt], sp_ppl, use_tqdm=False)[0]
                    vals = score_ppl(out, n_prefix)
                    info = llm.collective_rpc("reindex_info")[0]
                    res["ppl"].append({"mode": mode, "prefix_len": prefix_len, "corpus": corpus, "doc": di, "n_scored": len(vals),
                                       "mean_logprob": (sum(vals) / len(vals)) if vals else None,
                                       "ppl": (math.exp(-sum(vals) / len(vals)) if vals else None), "logprobs": [round(v, 5) for v in vals],
                                       "events": info["n_events"], "errors": info["errors"][:1]})
                m = [r for r in res["ppl"] if r["mode"] == mode and r["prefix_len"] == prefix_len and r.get("corpus") == corpus]
                print(f"[ppl] prefix {prefix_len} {corpus} {mode:>14}: n={len(m)} mean PPL {sum(r['ppl'] for r in m)/len(m):.4f} "
                      f"({time.time()-t1:.0f}s, events so far {info['n_events']}, errors {info['errors'][:1]})", flush=True)
                json.dump(res, open(a.out, "w"))
    res["timing"]["ppl_s"] = round(time.time() - t_ppl, 1)

    # ---------- ACC block ----------
    if not a.skip_acc:
        t_acc = time.time()
        man = [json.loads(l) for l in open(os.path.join(W, "prompts", "exp3", "manifest.jsonl")) if l.strip()]
        pdir = os.path.join(W, "prompts", "exp3")
        sel = []
        for L in RULER_RUNGS:
            for task in RULER_TASKS:
                sel += [r for r in man if r["source"] == "ruler" and r["task"] == task and r["rung"] == L][: a.n_ruler]
        lb2 = [r for r in man if r["source"] == "longbench_v2"]
        by_dom = {}
        for r in sorted(lb2, key=lambda r: r["sample_id"]):
            by_dom.setdefault(r["task"], []).append(r)
        quota = {d: max(1, round(a.n_lb2 * len(v) / len(lb2))) for d, v in by_dom.items()}
        picked = [r for d, v in sorted(by_dom.items()) for r in v[: quota[d]]][: a.n_lb2]
        sel += picked
        res["acc_selection"] = {"ruler_tasks": RULER_TASKS, "ruler_rungs": RULER_RUNGS, "n_ruler": a.n_ruler,
                                "lb2_quota": quota, "n_items": len(sel)}
        print(f"[acc] {len(sel)} items ({len(sel) - len(picked)} RULER + {len(picked)} LongBench-v2 {quota})", flush=True)
        for mode in a.acc_modes:
            done = {r["sample_id"] for r in res["acc"] if r["mode"] == mode}
            todo = [r for r in sel if r["sample_id"] not in done]
            if not todo:
                print(f"[acc] {mode:>14}: skip (have {len(done)})", flush=True)
                continue
            m, impl, seed = mode_cfg(mode)
            d = os.path.join(log_root, f"acc_{mode}".replace("@", "_s")); os.makedirs(d, exist_ok=True)
            llm.collective_rpc("reindex_set", kwargs={"mode": m, "impl": impl, "seed": seed, "log_dir": d, "prefix_len": 0})
            t1 = time.time()
            for k, r in enumerate(todo):
                ids, _ = chat_prompt_ids(a.model, tok, open(os.path.join(pdir, r["text_path"])).read(), "chat")
                sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_for(r), seed=42, logprobs=1)
                out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
                c = out.outputs[0]
                ok, sc, ex = score_fn(r["source"], r["task"], c.text, r["reference"])
                info = llm.collective_rpc("reindex_info")[0]
                res["acc"].append({"mode": mode, "sample_id": r["sample_id"], "source": r["source"], "task": r["task"], "rung": r["rung"],
                                   "correct": ok, "score": sc, "extracted": ex, "n_tokens": len(c.token_ids), "max_new_tokens": max_new_for(r), "token_ids": list(c.token_ids),
                                   "text": c.text[:200], "events": info["n_events"], "errors": info["errors"][:1]})
                if (k + 1) % 25 == 0:
                    json.dump(res, open(a.out, "w"))
                    print(f"[acc] {mode:>14}: {k+1}/{len(todo)} ({time.time()-t1:.0f}s)", flush=True)
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
