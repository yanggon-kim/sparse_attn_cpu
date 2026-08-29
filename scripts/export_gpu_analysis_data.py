#!/usr/bin/env python3
"""Aggregate the GPU-campaign per-run analysis JSONs (sparse_attn_cpu docs/{gpu_sweep,glm_sweep}/runs/<run_id>/analysis/)
into (1) retention_curves.json files for the ramulator v6 policy study and (2) the paper's fig:hotness CSVs.

No trace parsing here -- only metrics_run_summary.json, extended_retention.json, hotset_coverage.json, run_manifest.json.

Outputs
  <exports>/v6_v32/retention_curves.json          families v32_{ruler,bf,ld}_<rung>
  <exports>/v6_glm/retention_curves.json          families glm52_*, glm52b_*, glm5_*
  <repo>/docs/00_doc/data/retention_curves_gpu.json   (both, one file, small)
  <repo>/docs/00_doc/data/hotness_coverage_<ctx>.csv            V3.2 long-decode runs (pool_pct, mean, min, max over (run, layer))
  <repo>/docs/00_doc/data/hotness_coverage_<model>_<ctx>.csv    GLM-5.2 (all layers) / GLM-5
  <repo>/docs/00_doc/data/hotness_retention.csv                 V3.2 ld 64K + 128K (lags 1..1024, working-set windows 1..1024)
  <repo>/docs/00_doc/data/hotness_retention_<model>.csv         GLM-5.2 / GLM-5
  <repo>/docs/00_doc/data/gpu_headline_by_kind.csv              model x variant x rung x kind: n, adj overlap, lift, ret@64/512/1024, A@99, cov10
Family/retention definitions follow docs/00_doc/v6_export_format.md and locality_metrics.md §2.4/§2.8.
Usage: export_gpu_analysis_data.py [--repo-root R] [--exports E] [--data-dir D]
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
LAGS_STD = [1, 2, 4, 8, 16, 32, 64]
LAGS_EXT = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
WINS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CTX_TAG = {8192: "8k", 16384: "16k", 32768: "32k", 65536: "64k", 131072: "128k"}
MODEL_TAG = {("DeepSeek-V3.2", "all_layers"): "v32", ("GLM-5.2", "all_layers"): "glm52",
             ("GLM-5.2", "computing_only"): "glm52b", ("GLM-5", "all_layers"): "glm5"}


def fnum(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.6f}"


def load_runs(repo_root):
    runs = []
    for sweep in ("gpu_sweep", "glm_sweep"):
        root = os.path.join(repo_root, "docs", sweep, "runs")
        if not os.path.isdir(root):
            continue
        for rid in sorted(os.listdir(root)):
            d = os.path.join(root, rid)
            mp_ = os.path.join(d, "run_manifest.json")
            sp = os.path.join(d, "analysis", "metrics_run_summary.json")
            if not (os.path.exists(mp_) and os.path.exists(sp)):
                continue
            man = json.load(open(mp_))
            if "smoke" in rid:
                continue
            m = re.search(r"_(8192|16384|32768|65536|131072)_", rid)
            if not m:
                continue
            summ = json.load(open(sp))
            ext_p = os.path.join(d, "analysis", "extended_retention.json")
            ext = json.load(open(ext_p)) if os.path.exists(ext_p) else None
            hs_p = os.path.join(d, "analysis", "hotset_coverage.json")
            hs = json.load(open(hs_p)) if os.path.exists(hs_p) else None
            tag = MODEL_TAG[(man["model_name"], man.get("variant", "all_layers"))]
            runs.append(dict(run_id=rid, dir=d, sweep=sweep, model=man["model_name"], variant=man.get("variant", "all_layers"),
                             tag=tag, kind=man.get("kind"), rung=int(m.group(1)), benchmark=man.get("benchmark_name"),
                             task=man.get("task_subset"), prompt_tokens=man.get("context_length_actual_tokens"),
                             steps=summ.get("n_decode_steps"), summ=summ, ext=ext, hs=hs))
    return runs


def retention_of(r):
    ret = {int(k.split("_")[1]): float(v) for k, v in r["summ"]["overall_retention"].items()}
    ws = {int(k[1:]): float(v) for k, v in r["summ"]["overall_working_set_ratio"].items()}
    ext = None
    ws_ext = None
    if r["ext"]:
        ext = {int(k): float(v) for k, v in r["ext"]["retention"].items()}
        ws_ext = {int(k): float(v) for k, v in r["ext"]["working_set_ratio"].items()}
    return ret, ws, ext, ws_ext


def nanmean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def build_families(runs):
    fams = {}
    for r in runs:
        ret, ws, ext, ws_ext = retention_of(r)
        keys = [f"{r['tag']}_{r['kind']}_{r['rung']}"]
        if r["kind"] == "bf" and r["benchmark"] == "ruler":
            keys.append(f"{r['tag']}_ruler_{r['rung']}")
        entry = {"steps": r["steps"], "prompt_tokens": r["prompt_tokens"], "benchmark": r["benchmark"], "task": r["task"],
                 "retention": {str(l): ret[l] for l in sorted(ret)},
                 "working_set_ratio": {str(w): ws[w] for w in sorted(ws)},
                 "retention_extended": ({str(l): ext[l] for l in sorted(ext)} if ext else None),
                 "working_set_ratio_extended": ({str(w): ws_ext[w] for w in sorted(ws_ext)} if ws_ext else None),
                 "source": "analysis/metrics_run_summary.json:overall_retention (+ analysis/extended_retention.json)"}
        for key in keys:
            d = fams.setdefault(key, {"family": key.split("_")[1], "model": r["model"], "variant": r["variant"],
                                      "context_rung": r["rung"], "run_kind": r["kind"], "runs": {}})
            d["runs"][r["run_id"]] = entry
    for key, d in fams.items():
        lags = sorted({int(l) for e in d["runs"].values() for l in e["retention"]})
        d["mean_retention"] = {str(l): nanmean([e["retention"].get(str(l)) for e in d["runs"].values()]) for l in lags}
        elags = sorted({int(l) for e in d["runs"].values() if e["retention_extended"] for l in e["retention_extended"]})
        if elags:
            d["mean_retention_extended"] = {str(l): nanmean([e["retention_extended"].get(str(l)) for e in d["runs"].values()
                                                             if e["retention_extended"]]) for l in elags}
            d["mean_working_set_ratio_extended"] = {str(w): nanmean([e["working_set_ratio_extended"].get(str(w)) for e in d["runs"].values()
                                                                     if e["working_set_ratio_extended"]]) for w in WINS}
        d["n_runs"] = len(d["runs"])
        d["mean_steps"] = float(np.mean([e["steps"] for e in d["runs"].values()]))
    return fams


def write_curves(fams, path, models_note):
    out = {
        "description": "Retention at lag L = mean over (layer, step) of |S_t ∩ S_{t-L}| / |S_t| (per-run: unweighted mean over the "
                       "top-k layers of per-layer means for lags 1-64 from metrics_run_summary.json; retention_extended pools all "
                       "(layer, step) pairs, lags 1-2048, from extended_retention.json). GPU campaign (vLLM 0.28.0, 8x B200, k = 2048 "
                       "tokens, ratio 1). " + models_note,
        "families_note": "<model>_ld_<rung>: long-decode runs (2048 forced steps; LongBench-v1 summarization, InfiniteBench En.QA/"
                         "En.Sum) -- use these for lags > 64. <model>_bf_<rung>: benchmark-faithful runs (3-512 decode steps, median "
                         "~20-30; RULER niah+qa, LongBench v1/v2, InfiniteBench) -- short traces, lags >= 16 are unreliable. "
                         "<model>_ruler_<rung>: the RULER subset of bf. Model tags: v32 = DeepSeek-V3.2 (61 layers), glm52 = GLM-5.2 "
                         "all 78 layers (memory-system view), glm52b = GLM-5.2 21 computing layers only, glm5 = GLM-5 (78 layers).",
        "consumer_note": "For the retention_pred policy use a LEAVE-ONE-OUT curve (exclude the evaluated run; the ld families have "
                         "8 runs per rung). retention_lag2048 is NaN everywhere: 2048-step decodes give lags <= 1024.",
        "lags_standard": LAGS_STD, "lags_extended": LAGS_EXT, "working_set_windows": WINS,
        "families": fams,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "work/experiment/scripts/export_gpu_analysis_data.py",
        "campaign_commit_sparse_attn_cpu": "a28245c (analysis) + raw-trace batches, HEAD 1c2a3f2",
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path, {k: v["n_runs"] for k, v in sorted(fams.items())})


def write_hotness(runs, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    prov = []
    for tag, fname_cov, fname_ret in (("v32", "hotness_coverage_{ctx}.csv", "hotness_retention.csv"),
                                      ("glm52", "hotness_coverage_glm52_{ctx}.csv", "hotness_retention_glm52.csv"),
                                      ("glm5", "hotness_coverage_glm5_{ctx}.csv", "hotness_retention_glm5.csv")):
        for rung in (16384, 32768, 65536, 131072):
            ld = [r for r in runs if r["tag"] == tag and r["kind"] == "ld" and r["rung"] == rung and r["hs"]]
            if not ld:
                continue
            pcts = sorted({p for r in ld for p in r["hs"]["coverage_by_pool_pct"]}, key=float)
            rows = []
            for p in pcts:
                vals = [float(pl["cov_by_pool_pct"][p]) for r in ld for pl in r["hs"]["per_layer"].values() if p in pl["cov_by_pool_pct"]]
                rows.append([p, f"{np.mean(vals):.6f}", f"{np.min(vals):.6f}", f"{np.max(vals):.6f}"])
            fn = fname_cov.format(ctx=CTX_TAG[rung])
            with open(os.path.join(data_dir, fn), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["pool_pct", "mean", "min", "max"])
                w.writerows(rows)
            a99 = [r["hs"]["A99_pct_mean"] for r in ld]
            prov.append((fn, tag, rung, [r["run_id"] for r in ld], float(np.mean([r["hs"]["pool_N_mean"] for r in ld])),
                         float(np.mean(a99)), min(a99), max(a99), ld[0]["hs"]["n_csa_layers"]))
            print(fn, len(ld), "runs", "A99 mean", round(float(np.mean(a99)), 1))
        # retention csv: ld 64K and 128K means
        cols = {}
        for rung in (65536, 131072):
            ld = [r for r in runs if r["tag"] == tag and r["kind"] == "ld" and r["rung"] == rung and r["ext"]]
            if not ld:
                continue
            ctx = CTX_TAG[rung]
            cols[f"retention_ld{ctx}"] = {int(l): nanmean([float(r["ext"]["retention"][str(l)]) for r in ld]) for l in LAGS_EXT}
            cols[f"working_set_ratio_ld{ctx}"] = {int(w): nanmean([float(r["ext"]["working_set_ratio"][str(w)]) for r in ld]) for w in WINS}
            prov.append((fname_ret, tag, rung, [r["run_id"] for r in ld], None, None, None, None, ld[0]["ext"]["n_layers"]))
        if cols:
            with open(os.path.join(data_dir, fname_ret), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["lag"] + list(cols))
                for lag in LAGS_EXT:
                    w.writerow([lag] + [fnum(cols[c].get(lag)) for c in cols])
            print(fname_ret, "written")
    return prov


def write_headline(runs, data_dir):
    keys = sorted({(r["tag"], r["rung"], r["kind"]) for r in runs})
    path = os.path.join(data_dir, "gpu_headline_by_kind.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_tag", "model", "variant", "rung", "kind", "n_runs", "mean_decode_steps", "adjacent_overlap", "lift_vs_random",
                    "recency_baseline", "retention_lag64", "retention_lag512", "retention_lag1024", "A99_pct", "cov_top10pct", "pool_N"])
        for tag, rung, kind in keys:
            rs = [r for r in runs if (r["tag"], r["rung"], r["kind"]) == (tag, rung, kind)]
            s = lambda key: nanmean([r["summ"].get(key) for r in rs])  # noqa: E731
            ret64 = nanmean([r["summ"]["overall_retention"].get("lag_64") for r in rs])
            ret512 = nanmean([float(r["ext"]["retention"]["512"]) for r in rs if r["ext"]])
            ret1024 = nanmean([float(r["ext"]["retention"]["1024"]) for r in rs if r["ext"]])
            a99 = nanmean([r["hs"]["A99_pct_mean"] for r in rs if r["hs"]])
            cov10 = nanmean([r["hs"]["coverage_by_pool_pct"]["10"]["mean"] for r in rs if r["hs"]])
            pool = nanmean([r["hs"]["pool_N_mean"] for r in rs if r["hs"]])
            w.writerow([tag, rs[0]["model"], rs[0]["variant"], rung, kind, len(rs), f"{np.mean([r['steps'] for r in rs]):.1f}",
                        fnum(s("overall_adjacent_overlap_mean")), fnum(s("overall_locality_lift_mean")), fnum(s("overall_recency_overlap_mean")),
                        fnum(ret64), fnum(ret512), fnum(ret1024), fnum(a99), fnum(cov10), f"{pool:.0f}"])
    print("wrote", path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.path.join(ROOT, "01_github", "sparse_attn_cpu"))
    ap.add_argument("--exports", default=os.path.join(EXP, "exports"))
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    data_dir = a.data_dir or os.path.join(a.repo_root, "docs", "00_doc", "data")
    runs = load_runs(a.repo_root)
    print(len(runs), "runs loaded")
    fams = build_families(runs)
    v32 = {k: v for k, v in fams.items() if k.startswith("v32_")}
    glm = {k: v for k, v in fams.items() if not k.startswith("v32_")}
    write_curves(v32, os.path.join(a.exports, "v6_v32", "retention_curves.json"), "Model DeepSeek-V3.2 (61 top-k layers).")
    write_curves(glm, os.path.join(a.exports, "v6_glm", "retention_curves.json"), "Models GLM-5.2 (a: 78 layers, b: 21 computing layers) and GLM-5 (78 layers).")
    write_curves(fams, os.path.join(data_dir, "retention_curves_gpu.json"), "All three models.")
    prov = write_hotness(runs, data_dir)
    write_headline(runs, data_dir)
    json.dump([{"file": p[0], "model_tag": p[1], "rung": p[2], "run_ids": p[3], "pool_N_mean": p[4], "A99_pct_mean": p[5],
                "A99_pct_min": p[6], "A99_pct_max": p[7], "n_layers": p[8]} for p in prov],
              open(os.path.join(data_dir, "hotness_provenance.json"), "w"), indent=1)
    print("wrote hotness_provenance.json")


if __name__ == "__main__":
    sys.exit(main())
