#!/usr/bin/env python3
"""Verification pass for a ladder sweep (exp1/exp2 gate): independent re-derivation of the CSV aggregates from the
per-run JSONs, validate-PASS census, and sanity checks (overlap non-increasing / lift increasing with context,
retention non-increasing in lag, recency baseline below overlap, A@99 non-increasing, all sel counts = 2048).
Usage: verify_sweep.py --runs-root <WORKDIR>/runs --model-tag v32 --csv docs/gpu_sweep/R1_kv_locality.csv
         [--r3 docs/gpu_sweep/R3_hotset_coverage.csv] [--exclude _smoke _smoke2 _solo _b]
Exit 1 on any FAIL.
"""
import argparse, csv, glob, json, os, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("--runs-root", required=True)
ap.add_argument("--model-tag", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--r3", default=None)
ap.add_argument("--exclude", nargs="*", default=["_smoke", "_smoke2", "_solo", "_b", "smoke0_capital_s0"])
a = ap.parse_args()
FAIL = []


def check(c, msg):
    print(("  PASS: " if c else "  FAIL: ") + msg)
    if not c:
        FAIL.append(msg)


runs = {}
for rd in sorted(glob.glob(os.path.join(a.runs_root, f"{a.model_tag}_*"))):
    rid = os.path.basename(rd)
    if any(rid.endswith(x) for x in a.exclude):
        continue
    try:
        man = json.load(open(os.path.join(rd, "run_manifest.json")))
        m = json.load(open(os.path.join(rd, "analysis", "metrics_run_summary.json")))
    except Exception:
        continue
    vlog = os.path.join(rd, "analysis", "validate.log")
    vpass = os.path.exists(vlog) and "VALIDATION PASSED" in open(vlog).read()
    hs = None
    try:
        hs = json.load(open(os.path.join(rd, "analysis", "hotset_coverage.json")))
    except Exception:
        pass
    runs[rid] = {"rung": man["context_length_target"], "kind": man.get("kind"), "adj": m["overall_adjacent_overlap_mean"],
                 "lift": m["overall_locality_lift_mean"], "rec": m["overall_recency_overlap_mean"], "ret": m["overall_retention"],
                 "steps": m["n_decode_steps"], "vpass": vpass, "a99": hs["A99_pct_mean"] if hs else None,
                 "n_csa": m["n_csa_layers"]}
print(f"[census] {len(runs)} runs")
check(all(r["vpass"] for r in runs.values()), f"validate_trace PASSED for every run ({sum(r['vpass'] for r in runs.values())}/{len(runs)})")
check(all(r["n_csa"] == runs[next(iter(runs))]["n_csa"] for r in runs.values()), "same CSA layer count in every run")
# re-derive per-rung means and compare with the CSV aggregates
csv_rows = [r for r in csv.DictReader(open(a.csv)) if r["run_id"].startswith("AGG:") and r["kind"] == "all"]
for row in csv_rows:
    rung = int(row["rung"])
    sel = [r for r in runs.values() if r["rung"] == rung]
    if not sel:
        check(False, f"rung {rung}: no runs found for CSV aggregate"); continue
    for key, col in (("adj", "adjacent_overlap"), ("lift", "lift_vs_random"), ("rec", "recency_baseline_overlap")):
        mine = statistics.mean(r[key] for r in sel)
        theirs = float(row[col])
        check(abs(mine - theirs) < 1e-6 and int(row["task"][2:]) == len(sel),
              f"rung {rung} {col}: CSV {theirs:.4f} == re-derived {mine:.4f} over n={len(sel)} (CSV n={row['task'][2:]})")
# sanity across rungs
rungs = sorted({r["rung"] for r in runs.values()})
adj = [statistics.mean(r["adj"] for r in runs.values() if r["rung"] == g) for g in rungs]
lift = [statistics.mean(r["lift"] for r in runs.values() if r["rung"] == g) for g in rungs]
rec = [statistics.mean(r["rec"] for r in runs.values() if r["rung"] == g) for g in rungs]
print("  rungs", rungs); print("  adj ", [round(x, 3) for x in adj]); print("  lift", [round(x, 2) for x in lift]); print("  rec ", [round(x, 3) for x in rec])
check(all(b <= a_ + 0.03 for a_, b in zip(adj, adj[1:])), "adjacent overlap non-increasing with context (3 pt tolerance)")
check(all(b > a_ for a_, b in zip(lift, lift[1:])), "lift strictly increasing with context")
check(all(r < o for r, o in zip(rec, adj)), "recency baseline below adjacent overlap at every rung")
check(all(lift[i] > 1 for i in range(len(lift))), "lift > 1 at every rung")
for rid, r in ((k, v) for k, v in runs.items() if v["kind"] == "bf" and v["steps"] >= 64):   # ld runs can loop periodically; short decodes have too few lag pairs
    lags = [1, 2, 4, 8, 16, 32, 64]
    vals = [r["ret"].get(f"lag_{l}") for l in lags]
    vals = [v for v in vals if v is not None and v == v]
    if any(vals[i + 1] > vals[i] + 0.02 for i in range(len(vals) - 1)):
        check(False, f"{rid}: retention not non-increasing in lag {[round(v, 3) for v in vals]}")
        break
else:
    check(True, "retention non-increasing in lag (2 pt tolerance) in every bf run with >= 64 steps")
a99 = [statistics.mean(r["a99"] for r in runs.values() if r["rung"] == g and r["a99"] is not None) for g in rungs if any(r["a99"] is not None and r["rung"] == g for r in runs.values())]
if a99:
    print("  A@99", [round(x, 1) for x in a99])
    check(all(b <= a_ + 2 for a_, b in zip(a99, a99[1:])), "hot-set A@99 non-increasing with context (2 pt tolerance)")
ld = [r for r in runs.values() if r["kind"] == "ld"]
check(all(r["steps"] >= 2048 for r in ld), f"all {len(ld)} ld runs have >= 2048 decode steps")
if FAIL:
    print(f"\nVERIFY FAILED: {len(FAIL)}"); sys.exit(1)
print("\nVERIFY PASSED")
