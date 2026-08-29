#!/usr/bin/env python3
"""exp3 §6 figure: grouped bars per benchmark — clean / ctrl_numeric / perm_once / perm_periodic (+ identity) with
95% CIs and the official value as a marker. Usage: make_reindex_figure.py <results.csv> <out.png> [--model NAME]
"""
import argparse, csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("out")
ap.add_argument("--model", default=None)
a = ap.parse_args()
rows = [r for r in csv.DictReader(open(a.results)) if (a.model is None or r["model"] == a.model)]
MODES = ["clean", "ctrl_identity", "ctrl_numeric", "perm_once", "perm_periodic"]
COL = {"clean": "#1f6feb", "ctrl_identity": "#8250df", "ctrl_numeric": "#647083", "perm_once": "#c55a11", "perm_periodic": "#1a7f37"}
groups = defaultdict(dict)
for r in rows:
    key = f"{r['benchmark']}\n{r['context'] or ''}"
    groups[key][(r["mode"], r["impl"])] = r
keys = sorted(groups)
fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(keys)), 4.5))
w = 0.8 / (len(MODES) + 1)
for gi, k in enumerate(keys):
    off = 0
    for mode in MODES:
        for impl in ("-", "A", "B"):
            r = groups[k].get((mode, impl))
            if not r:
                continue
            x = gi + off * w - 0.4 + w / 2
            acc, lo, hi = float(r["accuracy"]), float(r["ci_lo"]), float(r["ci_hi"])
            ax.bar(x, acc, w, color=COL[mode], alpha=0.9 if impl != "B" else 0.55, hatch="//" if impl == "B" else None,
                   label=f"{mode}{'' if impl == '-' else ' (' + impl + ')'}")
            ax.errorbar(x, acc, yerr=[[acc - lo], [hi - acc]], fmt="none", ecolor="black", capsize=2, lw=0.8)
            off += 1
    off_r = groups[k].get(("clean", "-"))
    if off_r and off_r.get("official"):
        try:
            ax.plot([gi - 0.4, gi + 0.4], [float(off_r["official"]) / 100.0] * 2, "k--", lw=1)
        except ValueError:
            pass
h, l = ax.get_legend_handles_labels()
seen, hh, ll = set(), [], []
for x, y in zip(h, l):
    if y not in seen:
        seen.add(y); hh.append(x); ll.append(y)
ax.legend(hh, ll, fontsize=7, ncol=3, loc="lower right")
ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, fontsize=8)
ax.set_ylabel("accuracy (fraction)"); ax.set_ylim(0, 1.05)
ax.set_title(f"Re-index accuracy: clean vs controls vs permuted KV cache ({a.model or 'all models'}); dashed = official")
fig.tight_layout(); fig.savefig(a.out, dpi=150)
print("wrote", a.out)
