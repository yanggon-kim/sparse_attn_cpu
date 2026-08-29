#!/usr/bin/env python3
"""exp3 §5/§6: build results.csv, per_item.jsonl and agreement_sample.json from the accuracy run batches.
Usage: exp3_results.py --runs-root <WORKDIR>/runs --model-tag v32 --model-name DeepSeek-V3.2 --out docs/reindex_accuracy
         [--official docs/reindex_accuracy/official_numbers.json]
Run ids follow run_vllm_batch.py: <tag>_<sample_id><suffix> with suffix = _<mode>[_<impl>] (clean = "_clean").
Per (benchmark, context, model, impl, mode): accuracy ± 95% bootstrap CI over items (10,000 resamples), paired
delta vs clean with its CI (paired bootstrap on shared items), official value (if any). per_item.jsonl carries
prediction hashes, first divergence step / top-1 agreement / max |dlogit| over the top-20 vs clean when logprobs
were recorded, and selection Jaccard vs clean (replayed through perm_log) when traces were recorded.
"""
import argparse, csv, glob, hashlib, json, os, random, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--runs-root", required=True)
ap.add_argument("--model-tag", required=True)
ap.add_argument("--model-name", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--official", default=None)
ap.add_argument("--n-boot", type=int, default=10000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
random.seed(42)
official = json.load(open(a.official)) if a.official and os.path.exists(a.official) else {}

MODES = ["clean", "clean2", "ctrl_identity", "ctrl_numeric", "perm_once", "perm_periodic"]


def parse_run(rd):
    rid = os.path.basename(rd)
    if not rid.startswith(a.model_tag + "_"):
        return None
    rest = rid[len(a.model_tag) + 1:]
    mode, impl = None, None
    for m in sorted(MODES, key=len, reverse=True):
        for imp in ("_A", "_B", ""):
            suf = f"_{m}{imp}"
            if rest.endswith(suf):
                mode, impl = m, (imp[1:] if imp else None)
                rest = rest[: -len(suf)]
                break
        if mode:
            break
    if not mode:
        return None
    req = json.load(open(os.path.join(rd, "req.json")))
    return {"run_id": rid, "sample_id": rest, "mode": mode, "impl": impl or "-", "source": req["source"], "task": req["task"],
            "context": req.get("rung"), "correct": req.get("is_correct"), "score": req.get("score"),
            "n_tokens": len(req["generated_token_ids"]), "pred_hash": hashlib.sha256(req["generated_text"].encode()).hexdigest()[:16],
            "token_ids": req["generated_token_ids"], "top_logprobs": req.get("top_logprobs"), "rd": rd}


runs = [r for r in (parse_run(rd) for rd in glob.glob(os.path.join(a.runs_root, f"{a.model_tag}_*")) if os.path.exists(os.path.join(rd, "req.json"))) if r]
by = {}
for r in runs:
    by.setdefault((r["source"], r["context"], r["mode"], r["impl"]), {})[r["sample_id"]] = r
clean = {}
for (src, ctx, mode, impl), d in by.items():
    if mode == "clean":
        clean[(src, ctx)] = d


def boot_ci(vals, n=a.n_boot):
    if not vals:
        return (float("nan"), float("nan"))
    m = len(vals)
    res = []
    for _ in range(n):
        s = [vals[random.randrange(m)] for _ in range(m)]
        res.append(sum(s) / m)
    res.sort()
    return res[int(0.025 * n)], res[int(0.975 * n) - 1]


def divergence(x, y):
    return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None if x == y else min(len(x), len(y)))


results, per_item, agreement = [], [], {}
for (src, ctx, mode, impl), d in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, MODES.index(kv[0][2]), kv[0][3])):
    items = [r for r in d.values() if r["correct"] is not None]
    accs = [1.0 if r["correct"] else 0.0 for r in items] if items else [r["score"] for r in d.values()]
    acc = sum(accs) / len(accs) if accs else float("nan")
    lo, hi = boot_ci(accs)
    row = {"benchmark": src, "subset": "all", "context": ctx, "model": a.model_name, "impl": impl, "mode": mode,
           "n_items": len(d), "accuracy": round(acc, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
           "delta_vs_clean": None, "delta_ci_lo": None, "delta_ci_hi": None,
           "official": official.get(a.model_name, {}).get(src, {}).get("value"),
           "official_source": official.get(a.model_name, {}).get(src, {}).get("source"),
           "run_ids": ";".join(sorted(r["run_id"] for r in d.values()))[:4000]}
    cl = clean.get((src, ctx))
    if cl and mode != "clean":
        shared = [s for s in d if s in cl]
        if shared:
            pa = [(1.0 if d[s]["correct"] else 0.0) if d[s]["correct"] is not None else d[s]["score"] for s in shared]
            pc = [(1.0 if cl[s]["correct"] else 0.0) if cl[s]["correct"] is not None else cl[s]["score"] for s in shared]
            diffs = [x - y for x, y in zip(pa, pc)]
            row["delta_vs_clean"] = round(sum(diffs) / len(diffs), 4)
            dlo, dhi = boot_ci(diffs)
            row["delta_ci_lo"], row["delta_ci_hi"] = round(dlo, 4), round(dhi, 4)
            # agreement sample
            divs, ident, top1, maxdl = [], 0, [], []
            for s in shared:
                x, y = d[s]["token_ids"], cl[s]["token_ids"]
                dv = divergence(x, y)
                divs.append(dv if dv is not None else len(x))
                ident += int(x == y)
                if d[s]["top_logprobs"] and cl[s]["top_logprobs"]:
                    n = min(len(d[s]["top_logprobs"]), len(cl[s]["top_logprobs"]))
                    agree = sum(1 for i in range(n) if d[s]["top_logprobs"][i] and cl[s]["top_logprobs"][i]
                                and d[s]["top_logprobs"][i][0][0] == cl[s]["top_logprobs"][i][0][0])
                    top1.append(agree / n if n else float("nan"))
                    # max |dlogit| over the top-20 before divergence (same inputs)
                    upto = divs[-1] if divs[-1] is not None else n
                    md = 0.0
                    for i in range(min(n, upto)):
                        da = dict(d[s]["top_logprobs"][i]); dc = dict(cl[s]["top_logprobs"][i])
                        for t in set(da) & set(dc):
                            md = max(md, abs(da[t] - dc[t]))
                    maxdl.append(md)
                per_item.append({"benchmark": src, "item_id": s, "context": ctx, "mode": mode, "impl": impl,
                                 "correct": d[s]["correct"], "score": d[s]["score"], "prediction_hash": d[s]["pred_hash"],
                                 "n_tokens": d[s]["n_tokens"], "first_divergence_step": dv, "identical_to_clean": x == y,
                                 "top1_agreement": (top1[-1] if top1 else None), "max_dlogit_top20": (maxdl[-1] if maxdl else None)})
            agreement[f"{src}|{ctx}|{mode}|{impl}"] = {
                "n": len(shared), "identical_token_streams": ident,
                "first_divergence_mean": statistics.mean(divs) if divs else None,
                "first_divergence_median": statistics.median(divs) if divs else None,
                "top1_agreement_mean": statistics.mean(top1) if top1 else None,
                "max_dlogit_top20_mean": statistics.mean(maxdl) if maxdl else None,
                "max_dlogit_top20_p90": (sorted(maxdl)[int(0.9 * (len(maxdl) - 1))] if maxdl else None)}
    results.append(row)
    if mode == "clean":
        for s in d:
            per_item.append({"benchmark": src, "item_id": s, "context": ctx, "mode": mode, "impl": impl, "correct": d[s]["correct"],
                             "score": d[s]["score"], "prediction_hash": d[s]["pred_hash"], "n_tokens": d[s]["n_tokens"]})

with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["benchmark"])
    w.writeheader()
    for r in results:
        w.writerow(r)
with open(os.path.join(a.out, "per_item.jsonl"), "w") as f:
    for r in per_item:
        f.write(json.dumps(r) + "\n")
json.dump(agreement, open(os.path.join(a.out, "agreement_sample.json"), "w"), indent=1)
for r in results:
    print(f"{r['benchmark']:>14} ctx={r['context']} {r['mode']:>13} {r['impl']}: n={r['n_items']} acc={r['accuracy']:.3f} "
          f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] delta={r['delta_vs_clean']} official={r['official']}")
