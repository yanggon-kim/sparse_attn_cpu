#!/usr/bin/env python3
"""Turn exp3 tier-2 JSONs (exp3_tier2.py; tier-1 JSONs also accepted) into docs/reindex_accuracy/tier2/:
results.csv, per_item.jsonl, tier2_results.md, reindex_ppl_delta_tier2.png, acc_by_task.csv.
PPL rows are grouped by (model, corpus, prefix); every non-clean mode is compared with `clean` (paired per document),
the clean-vs-clean2 rerun gives the noise floor. Equivalence: |mean ΔPPL| within the clean2 floor bound (max |CI edge|)
and per-token |Δlogprob| p90 within 1.5x max(floor p90, ctrl_numeric p90).
Accuracy rows: per (model, benchmark/task, context, mode) with n, accuracy, Δ vs clean, identical token streams,
plus pooled rows per benchmark. Usage: exp3_tier2_results.py --out <dir> <tier2_v32.json> [...]
"""
import argparse, csv, json, os, random, statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("files", nargs="+")
ap.add_argument("--n-boot", type=int, default=10000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
random.seed(42)
ORDER = ["clean", "clean2", "clean3", "clean4", "ctrl_identity", "ctrl_numeric", "perm_once_A", "perm_once_A@8", "perm_once_A@9",
         "perm_once_B", "perm_once_B@8", "perm_once_B@9", "perm_periodic_B"]


def mode_key(m):
    return (ORDER.index(m) if m in ORDER else len(ORDER), m)


def boot(vals, n):
    if not vals:
        return (float("nan"), float("nan"))
    m = len(vals); r = []
    for _ in range(n):
        s = [vals[random.randrange(m)] for _ in range(m)]
        r.append(sum(s) / m)
    r.sort()
    return r[int(0.025 * n)], r[int(0.975 * n) - 1]


def impl_of(mode):
    b = mode.split("@")[0]
    return "A" if b.endswith("_A") else "B" if (b.endswith("_B") or b == "ctrl_identity") else "-"


rows, per_item, acc_rows = [], [], []
for f in a.files:
    d = json.load(open(f))
    model = os.path.basename(d["model"]).replace("-FP8", "")
    ppl = d["ppl"]
    for r in ppl:
        r.setdefault("corpus", "wikitext2" if r["prefix_len"] == 2048 else "longbook")
    groups = sorted({(r["corpus"], r["prefix_len"]) for r in ppl})
    for corpus, P in groups:
        by = {}
        for r in ppl:
            if r["prefix_len"] == P and r["corpus"] == corpus:
                by.setdefault(r["mode"], {})[r["doc"]] = r
        clean = by.get("clean", {})
        floor, numeric = None, None
        for mode in sorted(by, key=mode_key):
            docs = sorted(set(by[mode]) & set(clean))
            if not docs:
                continue
            mean_ppl = statistics.mean(by[mode][x]["ppl"] for x in docs)
            dppl = [by[mode][x]["ppl"] - clean[x]["ppl"] for x in docs]
            tok_abs = [abs(p - q) for x in docs for p, q in zip(by[mode][x]["logprobs"], clean[x]["logprobs"])]
            lo, hi = boot(dppl, a.n_boot)
            p90 = sorted(tok_abs)[int(0.9 * (len(tok_abs) - 1))] if tok_abs else float("nan")
            row = {"benchmark": corpus, "context": P, "model": model, "impl": impl_of(mode), "mode": mode, "n_items": len(docs),
                   "metric": "ppl", "value": round(mean_ppl, 5), "delta_vs_clean": round(sum(dppl) / len(dppl), 6),
                   "delta_ci_lo": round(lo, 6), "delta_ci_hi": round(hi, 6),
                   "token_abs_dlogprob_mean": round(statistics.mean(tok_abs), 6), "token_abs_dlogprob_p90": round(p90, 6),
                   "n_tokens": len(tok_abs), "events": max(by[mode][x]["events"] for x in docs),
                   "errors": sum(1 for x in docs if by[mode][x].get("errors"))}
            if mode == "clean2":
                floor = row
            if mode == "ctrl_numeric":
                numeric = row
            if floor and mode not in ("clean", "clean2"):
                fl = max(abs(floor["delta_ci_lo"]), abs(floor["delta_ci_hi"]), 1e-9)
                ref_p90 = max(floor["token_abs_dlogprob_p90"], numeric["token_abs_dlogprob_p90"] if numeric else 0.0)
                row["equivalent_to_noise_floor"] = bool(abs(row["delta_vs_clean"]) <= fl and row["token_abs_dlogprob_p90"] <= 1.5 * ref_p90 + 1e-9)
                row["floor_delta_ci"] = f"[{floor['delta_ci_lo']}, {floor['delta_ci_hi']}]"
                row["delta_ci_includes_zero"] = bool(row["delta_ci_lo"] <= 0.0 <= row["delta_ci_hi"])   # weaker, variance-aware check
            rows.append(row)
            for x in docs:
                per_item.append({"benchmark": corpus, "item_id": x, "context": P, "mode": mode, "impl": row["impl"], "model": model,
                                 "ppl": by[mode][x]["ppl"], "delta_ppl_vs_clean": by[mode][x]["ppl"] - clean[x]["ppl"], "n_scored": by[mode][x]["n_scored"]})
    # accuracy block
    acc = d.get("acc", [])
    if acc:
        for r in acc:
            r.setdefault("source", "ruler")
        modes = sorted({r["mode"] for r in acc}, key=mode_key)
        cl = {r["sample_id"]: r for r in acc if r["mode"] == "clean"}

        def acc_row(bench, ctx, mode, its):
            its = [r for r in its if r["sample_id"] in cl]
            if not its:
                return None
            c = sum(1 for r in its if r["correct"]); c0 = sum(1 for r in its if cl[r["sample_id"]]["correct"])
            divs = [next((i for i, (p, q) in enumerate(zip(r["token_ids"], cl[r["sample_id"]]["token_ids"])) if p != q), None) for r in its]
            ident = sum(1 for dv in divs if dv is None and True)
            same_len = sum(1 for r, dv in zip(its, divs) if dv is None and len(r["token_ids"]) == len(cl[r["sample_id"]]["token_ids"]))
            flips = sum(1 for r in its if r["correct"] != cl[r["sample_id"]]["correct"])
            dacc = [(1 if r["correct"] else 0) - (1 if cl[r["sample_id"]]["correct"] else 0) for r in its]
            lo, hi = boot(dacc, a.n_boot) if mode != "clean" else (0.0, 0.0)
            return {"benchmark": bench, "context": ctx, "model": model, "impl": impl_of(mode), "mode": mode, "n_items": len(its), "metric": "accuracy",
                    "value": round(c / len(its), 4), "delta_vs_clean": round((c - c0) / len(its), 4), "delta_ci_lo": round(lo, 4), "delta_ci_hi": round(hi, 4),
                    "flips_vs_clean": flips, "identical_token_streams": same_len,
                    "first_divergence_median": statistics.median([x if x is not None else 9999 for x in divs]),
                    "events": max(r["events"] for r in its), "errors": sum(1 for r in its if r.get("errors"))}
        for mode in modes:
            for src in sorted({r["source"] for r in acc}):
                sub = [r for r in acc if r["mode"] == mode and r["source"] == src]
                if src == "ruler":
                    for task in sorted({r["task"] for r in sub}):
                        for L in sorted({r["rung"] for r in sub if r["task"] == task}):
                            row = acc_row(f"ruler_{task}", L, mode, [r for r in sub if r["task"] == task and r["rung"] == L])
                            if row: acc_rows.append(row)
                    row = acc_row("ruler_all", "pooled", mode, sub)
                    if row: acc_rows.append(row)
                else:
                    for task in sorted({r["task"] for r in sub}):
                        row = acc_row(f"{src}:{task}", "mixed", mode, [r for r in sub if r["task"] == task])
                        if row: acc_rows.append(row)
                    row = acc_row(f"{src}_all", "pooled", mode, sub)
                    if row: acc_rows.append(row)
            for r in [r for r in acc if r["mode"] == mode]:
                per_item.append({"benchmark": r["source"], "task": r["task"], "item_id": r["sample_id"], "context": r["rung"], "mode": mode, "model": model,
                                 "correct": r["correct"], "score": r["score"], "n_tokens": r["n_tokens"], "text": r["text"][:120]})
allrows = rows + acc_rows
keys = []
for r in allrows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in allrows:
        w.writerow(r)
with open(os.path.join(a.out, "per_item.jsonl"), "w") as f:
    for r in per_item:
        f.write(json.dumps(r) + "\n")
# figure: ΔPPL vs clean (paired, 95 % CI) per model, long-book prefixes; short corpora in a second row
models = sorted({r["model"] for r in rows})
cmap = plt.get_cmap("tab20")
fig, axs = plt.subplots(2, len(models), figsize=(5.5 * len(models), 7.5), squeeze=False)
for ci, model in enumerate(models):
    for ri, benches in enumerate([["longbook"], ["wikitext2", "ptb"]]):
        ax = axs[ri][ci]
        pr = [r for r in rows if r["model"] == model and r["benchmark"] in benches and r["mode"] != "clean"]
        xs = sorted({(r["benchmark"], r["context"]) for r in pr}); modes = sorted({r["mode"] for r in pr}, key=mode_key)
        w = 0.8 / max(1, len(modes))
        for i, m in enumerate(modes):
            for j, (b, c) in enumerate(xs):
                r = next((x for x in pr if x["mode"] == m and x["benchmark"] == b and x["context"] == c), None)
                if r is None:
                    continue
                x = j + i * w - 0.4 + w / 2
                ax.bar(x, r["delta_vs_clean"], w, color=cmap(mode_key(m)[0] % 20), label=m if j == 0 else None)
                ax.errorbar(x, r["delta_vs_clean"], yerr=[[r["delta_vs_clean"] - r["delta_ci_lo"]], [r["delta_ci_hi"] - r["delta_vs_clean"]]], fmt="none", ecolor="k", capsize=2, lw=0.8)
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{b} {c//1024}K" if ri else f"{c//1024}K prefix" for b, c in xs], fontsize=8)
        ax.set_title(f"{model}: ΔPPL vs clean (paired, 95 % CI)" if ri == 0 else f"{model}: short anchors", fontsize=9); ax.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "reindex_ppl_delta_tier2.png"), dpi=150)
# markdown
L = ["# exp3 tier 2 — re-index accuracy (perplexity primary, generation accuracy secondary)", "",
     "Teacher-forced perplexity of the scored continuation after the KV prefix was physically re-indexed (impl A: 64-token block "
     "permutation with block-table update; impl B: per-token row permutation, block table untouched; `@8`/`@9` = permutation seeds 8/9, no suffix = seed 7). "
     "`clean2` = identical rerun = run-to-run noise floor; `ctrl_identity` = hook active with identity permutation; `ctrl_numeric` = same item processed "
     "alongside a filler request. Verdict `equivalent_to_noise_floor`: |mean ΔPPL| within the clean2 CI bound and per-token |Δlogprob| p90 within "
     "1.5× max(floor, ctrl_numeric).", "",
     "| model | corpus | prefix | mode | impl | n | PPL | ΔPPL vs clean [95 % CI] | token |Δlogprob| mean / p90 | equivalent |", "|---|---|---:|---|---|---:|---:|---|---|---|"]
for r in rows:
    L.append(f"| {r['model']} | {r['benchmark']} | {r['context']} | {r['mode']} | {r['impl']} | {r['n_items']} | {r['value']:.4f} | {r['delta_vs_clean']:+.5f} [{r['delta_ci_lo']:+.5f}, {r['delta_ci_hi']:+.5f}] | {r['token_abs_dlogprob_mean']:.4f} / {r['token_abs_dlogprob_p90']:.4f} | {r.get('equivalent_to_noise_floor', '—')} |")
L += ["", "| model | benchmark | context | mode | n | accuracy | Δ vs clean [95 % CI] | flips | identical token streams | median first divergence |", "|---|---|---:|---|---:|---:|---|---:|---:|---:|"]
for r in acc_rows:
    L.append(f"| {r['model']} | {r['benchmark']} | {r['context']} | {r['mode']} | {r['n_items']} | {r['value']:.3f} | {r['delta_vs_clean']:+.3f} [{r['delta_ci_lo']:+.3f}, {r['delta_ci_hi']:+.3f}] | {r['flips_vs_clean']} | {r['identical_token_streams']}/{r['n_items']} | {r['first_divergence_median']} |")
L += ["", "![ΔPPL](reindex_ppl_delta_tier2.png)", ""]
open(os.path.join(a.out, "tier2_results.md"), "w").write("\n".join(L))
nv = [r for r in rows if "equivalent_to_noise_floor" in r]
print(f"ppl rows {len(rows)} (verdict rows {len(nv)}, equivalent {sum(1 for r in nv if r['equivalent_to_noise_floor'])}), acc rows {len(acc_rows)}, per-item {len(per_item)}")
for r in nv:
    if not r["equivalent_to_noise_floor"]:
        print("  NOT equivalent:", r["model"], r["benchmark"], r["context"], r["mode"], r["delta_vs_clean"], r["floor_delta_ci"], r["token_abs_dlogprob_p90"])
