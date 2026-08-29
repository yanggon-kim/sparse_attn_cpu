#!/usr/bin/env python3
"""exp1 §3 batch-attribution check: the same prompt run alone vs inside a batch must yield identical per-step
selection sets (greedy, prefix caching off). Usage: compare_attribution.py <run_dir_batch> <run_dir_solo>
Compares traces/indexer_trace.jsonl (KV) and traces/moe_trace.jsonl (MoE) record by record, and generated tokens.
"""
import json, sys


def load(p):
    d = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            d[(r["pos"], r["layer"])] = r["sel"]
    return d


a, b = sys.argv[1], sys.argv[2]
ga = json.loads(open(a + "/outputs/generations.jsonl").readline())
gb = json.loads(open(b + "/outputs/generations.jsonl").readline())
ta, tb = ga["generated_token_ids"], gb["generated_token_ids"]
same_tok = ta == tb
div = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
print(f"tokens identical: {same_tok}; first token divergence at decode step {div} of {len(ta)}/{len(tb)}")
# Batched vs solo decode is NOT bit-identical (batch-size-dependent fp8 kernels = the ctrl_numeric effect);
# attribution is judged on the steps BEFORE token divergence: a wrong req_id mapping gives Jaccard ~ k/N
# there, a right one gives ~1.0. After divergence the two decodes legitimately drift.
base_pos = min(int(json.loads(l)["pos"]) for l in open(a + "/traces/indexer_trace.jsonl") if l.strip())
fail = False
for name in ("indexer_trace.jsonl", "moe_trace.jsonl"):
    try:
        da, db = load(f"{a}/traces/{name}"), load(f"{b}/traces/{name}")
    except FileNotFoundError:
        print(f"{name}: missing, skipped"); continue
    keys = sorted(set(da) & set(db))
    pre = [k for k in keys if k[0] - base_pos <= div]      # steps whose input tokens are identical
    post = [k for k in keys if k[0] - base_pos > div]
    def jac(ks):
        v = [len(set(da[k]) & set(db[k])) / max(1, len(set(da[k]) | set(db[k]))) for k in ks]
        return (sum(v) / len(v)) if v else float("nan"), sum(1 for k in ks if da[k] != db[k])
    jpre, dpre = jac(pre); jpost, dpost = jac(post)
    print(f"{name}: {len(keys)} common cells ({len(da)}/{len(db)}); before divergence: {len(pre)} cells, "
          f"mean Jaccard {jpre:.4f}, {dpre} not bit-identical; after: {len(post)} cells, mean Jaccard {jpost:.4f}")
    # PASS criterion: pre-divergence Jaccard far above the random level k/N (~0.25 at 8K for KV, ~0.03 for MoE);
    # bit-identity is not expected because batched vs solo fp8 kernels are not batch-invariant (measured).
    thr = 0.80 if name.startswith("indexer") else 0.50
    if len(da) != len(db) or (pre and jpre < thr):
        fail = True
# structural check: the batch run's positions must track its own prompt length (wrong mapping -> other request's)
pa = sorted({k[0] for k in load(a + "/traces/indexer_trace.jsonl")})
print(f"structural: first pos {pa[0]} == prompt_token_count-1 ({ga['prompt_token_count'] - 1}): {pa[0] == ga['prompt_token_count'] - 1}; "
      f"contiguous steps: {pa[-1] - pa[0] + 1 == len(pa)}")
if pa[0] != ga['prompt_token_count'] - 1 or pa[-1] - pa[0] + 1 != len(pa):
    fail = True
print("ATTRIBUTION " + ("FAIL" if fail else "PASS"))
sys.exit(1 if fail else 0)
