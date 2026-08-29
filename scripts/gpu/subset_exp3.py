#!/usr/bin/env python3
"""Select a budget-sized, stratified subset of the exp3 manifest (seed 42) and estimate its cost.
Usage: subset_exp3.py --manifest <exp3>/manifest.jsonl --out <exp3>/manifest_subset.jsonl
         [--ruler-tasks niah_single_2 niah_multikey_2 vt qa_1] [--ruler-per-task 50] [--ruler-lengths 32768 65536 131072]
         [--lb2 200] [--ib-per-task 50] [--mmlu 500] [--gpqa 198]
         [--prefill-tps 7000] [--decode-tps 400] [--thinking-tokens 3000]
Prints per-source item counts, prompt tokens, and an estimated node-hours / GPU-hours per mode.
"""
import argparse, json, random
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ruler-tasks", nargs="*", default=["niah_single_2", "niah_multikey_2", "vt", "qa_1"])
ap.add_argument("--ruler-per-task", type=int, default=50)
ap.add_argument("--ruler-lengths", nargs="*", type=int, default=[32768, 65536, 131072])
ap.add_argument("--lb2", type=int, default=200)
ap.add_argument("--ib-per-task", type=int, default=50)
ap.add_argument("--mmlu", type=int, default=500)
ap.add_argument("--gpqa", type=int, default=198)
ap.add_argument("--prefill-tps", type=float, default=7000)
ap.add_argument("--decode-tps", type=float, default=400)
ap.add_argument("--thinking-tokens", type=float, default=3000)
a = ap.parse_args()
random.seed(42)
rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
by = defaultdict(list)
for r in rows:
    by[(r["source"], r["task"], r["rung"])].append(r)
chosen = []
for (src, task, rung), lst in by.items():
    lst = sorted(lst, key=lambda r: r["sample_id"])
    if src == "ruler":
        if task in a.ruler_tasks and rung in a.ruler_lengths:
            chosen += lst[: a.ruler_per_task]
    elif src == "infinitebench":
        chosen += lst[: a.ib_per_task]
    elif src == "mmlu_pro":
        pass
    elif src == "gpqa":
        chosen += lst[: a.gpqa]
lb2 = [r for r in rows if r["source"] == "longbench_v2"]
random.shuffle(lb2)
# stratify by length bucket
per = {"short": a.lb2 // 3, "medium": a.lb2 // 3, "long": a.lb2 - 2 * (a.lb2 // 3)}
cnt = Counter()
for r in lb2:
    L = r["meta"]["length"]
    if cnt[L] < per[L]:
        chosen.append(r); cnt[L] += 1
mm = [r for r in rows if r["source"] == "mmlu_pro"]
random.shuffle(mm)
cats = sorted({r["task"] for r in mm})
per_c = a.mmlu // len(cats) if cats else 0
cc = Counter()
for r in mm:
    if cc[r["task"]] < per_c:
        chosen.append(r); cc[r["task"]] += 1
with open(a.out, "w") as f:
    for r in chosen:
        f.write(json.dumps(r) + "\n")
tot_prompt = sum(r["prompt_tokens_v32"] for r in chosen)
dec = sum((a.thinking_tokens if r.get("thinking") == "thinking" else r["max_new_tokens"]) for r in chosen)
hours = tot_prompt / a.prefill_tps / 3600 + dec / a.decode_tps / 3600
print("items per source:", Counter(r["source"] for r in chosen))
print("items per (source, rung):", sorted(Counter((r["source"], r["rung"]) for r in chosen).items()))
print(f"prompt tokens {tot_prompt/1e6:.1f} M, decode tokens ~{dec/1e6:.2f} M -> ~{hours:.2f} node-h = {8*hours:.1f} GPU-h per mode "
      f"(prefill {a.prefill_tps:.0f} tok/s, decode {a.decode_tps:.0f} tok/s assumed)")
