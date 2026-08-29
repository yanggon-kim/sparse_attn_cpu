#!/usr/bin/env python3
"""Turn tier-1 JSONs (exp3_tier1.py) into docs/reindex_accuracy: results.csv (PPL + accuracy rows with paired
bootstrap CIs vs clean and the clean-vs-clean2 noise floor), per_item.jsonl, agreement_sample.json, figure, README table.
Usage: exp3_tier1_results.py --out docs/reindex_accuracy <tier1_v32.json> [<tier1_glm52.json> ...]
Equivalence verdict per (model, prefix, mode): |ΔPPL| CI overlaps the clean2 floor interval, i.e. the paired
per-document ΔPPL vs clean is not distinguishable from clean2's — and per-token |Δlogprob| within 3x the floor's p90.
"""
import argparse, csv, json, math, os, random, statistics

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


def boot(vals, n):
    if not vals:
        return (float("nan"), float("nan"))
    m = len(vals); r = []
    for _ in range(n):
        s = [vals[random.randrange(m)] for _ in range(m)]
        r.append(sum(s) / m)
    r.sort()
    return r[int(0.025 * n)], r[int(0.975 * n) - 1]


rows, per_item, agreement = [], [], {}
for f in a.files:
    d = json.load(open(f))
    model = os.path.basename(d["model"]).replace("-FP8", "")
    ppl = d["ppl"]
    prefixes = sorted({r["prefix_len"] for r in ppl})
    for P in prefixes:
        by = {}
        for r in ppl:
            if r["prefix_len"] == P:
                by.setdefault(r["mode"], {})[r["doc"]] = r
        clean = by.get("clean", {})
        floor, numeric = None, None
        order = ["clean", "clean2", "clean3", "clean4", "ctrl_identity", "ctrl_numeric", "perm_once_A", "perm_once_A@8", "perm_once_B", "perm_once_B@8", "perm_once_B@9"]
        for mode in [m for m in order if m in by] + sorted(m for m in by if m not in order):
            docs = sorted(set(by[mode]) & set(clean))
            mean_ppl = statistics.mean(by[mode][x]["ppl"] for x in docs)
            # paired per-document delta of mean log-prob (equivalently log PPL ratio)
            dl = [by[mode][x]["mean_logprob"] - clean[x]["mean_logprob"] for x in docs]
            dppl = [by[mode][x]["ppl"] - clean[x]["ppl"] for x in docs]
            tok_abs = [abs(p - q) for x in docs for p, q in zip(by[mode][x]["logprobs"], clean[x]["logprobs"])]
            lo, hi = boot(dppl, a.n_boot)
            p90 = sorted(tok_abs)[int(0.9 * (len(tok_abs) - 1))] if tok_abs else float("nan")
            base = mode.split("@")[0]
            row = {"benchmark": "wikitext2" if P == 2048 else "longbook_ppl", "context": P, "model": model, "impl": ("A" if base.endswith("_A") else "B" if base.endswith("_B") or base == "ctrl_identity" else "-"),
                   "mode": mode, "n_items": len(docs), "metric": "ppl", "value": round(mean_ppl, 5),
                   "delta_vs_clean": round(sum(dppl) / len(dppl), 6) if dppl else None, "delta_ci_lo": round(lo, 6), "delta_ci_hi": round(hi, 6),
                   "token_abs_dlogprob_mean": round(statistics.mean(tok_abs), 6) if tok_abs else None, "token_abs_dlogprob_p90": round(p90, 6),
                   "n_tokens": len(tok_abs), "events": max((by[mode][x]["events"] for x in docs), default=0)}
            if mode == "clean2":
                floor = row
            if floor and mode.startswith("clean") and mode not in ("clean", "clean2"):
                fl = max(abs(floor["delta_ci_lo"]), abs(floor["delta_ci_hi"]), 1e-9)
                row["equivalent_to_noise_floor"] = bool(abs(row["delta_vs_clean"]) <= fl)   # extra reruns: floor consistency check
            if mode == "ctrl_numeric":
                numeric = row
            if floor and not mode.startswith("clean"):
                # equivalence: (i) |mean ΔPPL| within the clean2 floor bound (max |CI edge|); (ii) per-token |Δlogprob| p90
                # within 1.5x the larger of the floor's and the batch-composition control's p90 (summation-order noise).
                fl = max(abs(floor["delta_ci_lo"]), abs(floor["delta_ci_hi"]), 1e-9)
                ref_p90 = max(floor["token_abs_dlogprob_p90"], (numeric["token_abs_dlogprob_p90"] if numeric else 0.0))
                row["equivalent_to_noise_floor"] = bool(abs(row["delta_vs_clean"]) <= fl and row["token_abs_dlogprob_p90"] <= 1.5 * ref_p90 + 1e-9)
                row["floor_delta_ci"] = f"[{floor['delta_ci_lo']}, {floor['delta_ci_hi']}]"
            rows.append(row)
            for x in docs:
                per_item.append({"benchmark": row["benchmark"], "item_id": x, "context": P, "mode": mode, "impl": row["impl"], "model": model,
                                 "ppl": by[mode][x]["ppl"], "delta_ppl_vs_clean": by[mode][x]["ppl"] - clean[x]["ppl"], "n_scored": by[mode][x]["n_scored"]})
    # accuracy block
    acc = d.get("acc", [])
    if acc:
        modes = []
        for r in acc:
            if r["mode"] not in modes:
                modes.append(r["mode"])
        cl = {r["sample_id"]: r for r in acc if r["mode"] == "clean"}
        for mode in modes:
            for L in sorted({r["rung"] for r in acc}):
                its = [r for r in acc if r["mode"] == mode and r["rung"] == L]
                if not its:
                    continue
                c = sum(1 for r in its if r["correct"])
                divs = [next((i for i, (p, q) in enumerate(zip(r["token_ids"], cl[r["sample_id"]]["token_ids"])) if p != q), None) for r in its if r["sample_id"] in cl]
                ident = sum(1 for dv in divs if dv is None)
                rows.append({"benchmark": "ruler_niah2+qa1", "context": L, "model": model, "impl": ("B" if mode.endswith("_B") else "A" if mode.endswith("_A") else "-"),
                             "mode": mode, "n_items": len(its), "metric": "accuracy", "value": round(c / len(its), 3),
                             "delta_vs_clean": round(c / len(its) - sum(1 for r in its if cl[r["sample_id"]]["correct"]) / len(its), 3) if mode != "clean" else 0.0,
                             "identical_token_streams": ident, "first_divergence_median": statistics.median([x if x is not None else 9999 for x in divs]) if divs else None})
                for r in its:
                    per_item.append({"benchmark": "ruler", "item_id": r["sample_id"], "context": L, "mode": mode, "model": model, "correct": r["correct"],
                                     "n_tokens": r["n_tokens"], "text": r["text"][:120]})
keys = []
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in rows:
        w.writerow(r)
with open(os.path.join(a.out, "per_item.jsonl"), "w") as f:
    for r in per_item:
        f.write(json.dumps(r) + "\n")
# figure: ΔPPL vs clean with CI, per model/context, modes as bars; floor shaded
models = sorted({r["model"] for r in rows if r["metric"] == "ppl"})
fig, axs = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
COL = {"clean2": "#647083", "clean3": "#8b949e", "clean4": "#afb8c1", "ctrl_identity": "#8250df", "ctrl_numeric": "#b54708", "perm_once_A": "#c55a11", "perm_once_A@8": "#e8894a", "perm_once_B": "#1a7f37", "perm_once_B@8": "#3fb950", "perm_once_B@9": "#7ee787"}
for ax, model in zip(axs[0], models):
    pr = [r for r in rows if r["metric"] == "ppl" and r["model"] == model and r["mode"] != "clean"]
    ctxs = sorted({r["context"] for r in pr}); modes = [m for m in COL if any(r["mode"] == m for r in pr)]
    w = 0.8 / max(1, len(modes))
    for i, m in enumerate(modes):
        for j, c in enumerate(ctxs):
            r = next((x for x in pr if x["mode"] == m and x["context"] == c), None)
            if r is None:
                continue
            x = j + i * w - 0.4 + w / 2
            ax.bar(x, r["delta_vs_clean"], w, color=COL[m], label=m if j == 0 else None)
            ax.errorbar(x, r["delta_vs_clean"], yerr=[[r["delta_vs_clean"] - r["delta_ci_lo"]], [r["delta_ci_hi"] - r["delta_vs_clean"]]], fmt="none", ecolor="k", capsize=2, lw=0.8)
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(len(ctxs))); ax.set_xticklabels([f"{c//1024}K" for c in ctxs])
    ax.set_title(f"{model}: ΔPPL vs clean (paired, 95 % CI)"); ax.set_xlabel("prefix length"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "reindex_ppl_delta.png"), dpi=150)
# README table
L = ["# exp3 tier 1 — re-index accuracy (perplexity primary, RULER sanity)", "",
     "Teacher-forced perplexity of a scored continuation after the KV prefix was physically re-indexed (impl A: 64-token block permutation with block-table update; impl B: per-token row permutation, block table untouched). "
     "`clean2` = identical rerun = run-to-run noise floor; `ctrl_identity` = hook active with identity permutation; `ctrl_numeric` = same item processed alongside a filler request. "
     "Verdict `equivalent_to_noise_floor`: |mean ΔPPL| within the clean2 CI bound and per-token |Δlogprob| p90 within 3× the floor's.", "",
     "| model | benchmark | prefix | mode | impl | n | PPL | ΔPPL vs clean [95 % CI] | token |Δlogprob| mean / p90 | equivalent |", "|---|---|---:|---|---|---:|---:|---|---|---|"]
for r in rows:
    if r["metric"] == "ppl":
        L.append(f"| {r['model']} | {r['benchmark']} | {r['context']} | {r['mode']} | {r['impl']} | {r['n_items']} | {r['value']:.4f} | {r['delta_vs_clean']:+.5f} [{r['delta_ci_lo']:+.5f}, {r['delta_ci_hi']:+.5f}] | {r['token_abs_dlogprob_mean']:.4f} / {r['token_abs_dlogprob_p90']:.4f} | {r.get('equivalent_to_noise_floor', '—')} |")
L += ["", "| model | benchmark | context | mode | n | accuracy | Δ vs clean | identical token streams | median first divergence |", "|---|---|---:|---|---:|---:|---:|---:|---:|"]
for r in rows:
    if r["metric"] == "accuracy":
        L.append(f"| {r['model']} | {r['benchmark']} | {r['context']} | {r['mode']} | {r['n_items']} | {r['value']:.2f} | {r['delta_vs_clean']:+.2f} | {r['identical_token_streams']}/{r['n_items']} | {r['first_divergence_median']} |")
L += ["", "![ΔPPL](reindex_ppl_delta.png)", ""]
open(os.path.join(a.out, "tier1_results.md"), "w").write("\n".join(L))
print("\n".join(L[:30]))
print(f"wrote {len(rows)} result rows, {len(per_item)} per-item rows")
