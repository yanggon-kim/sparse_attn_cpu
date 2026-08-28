#!/usr/bin/env python3
"""Export DeepSeek-V4 top-k selection traces for the ramulator v6 migration-policy study.

For every run directory under <runs_root> that has traces/selected_kv.parquet (and is not in
SKIP), write <out_dir>/<run_id>.npz + <run_id>.manifest.json, then <out_dir>/retention_curves.json.

npz keys (format spec: sparse_attn_cpu/docs/00_doc/v6_export_format.md):
  sel      uint16 (max index < 65536) or uint32, shape [steps, layers, k], each row sorted
           ascending; unused slots (rank >= valid_k or is_valid == False) filled with the dtype max.
  valid_k  int32 [steps, layers]  number of valid entries in each row.
  pos      int32 [steps]          decode_position (token index of the query = candidate tokens).
  n_comp   int32 [steps]          (pos + 1) // ratio  (compressed candidate count seen by the indexer;
           pos is the 0-based query position, so pos + 1 tokens are in the cache; NOT pos // ratio).
  layers   int32 [layers]         CSA layer ids (the layers that carry a top-k selection).
  k, ratio int scalars; model (str scalar).
V4 units: one entry = one compressed block of `ratio` tokens. V3.2 exports use ratio = 1, k = 2048.

Usage:
  export_v6_traces.py [--runs-root R] [--out-dir O] [--only RUN_ID ...] [--no-jsonl-check]
Default runs-root = <experiment>/runs, out-dir = <experiment>/exports/v6.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
SKIP = {
    # superseded RULER series (-n 128) and the aborted 96K attempt
    "niah_single_2_4096_q2", "niah_single_2_8192_q2", "niah_single_2_16384_q2",
    "niah_single_2_40960_q2", "niah_single_2_65536_q2", "niah_single_2_98304_q2",
}
LAGS_STD = [1, 2, 4, 8, 16, 32, 64]
LAGS_EXT = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_hash(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "n/a"


def jsonl_check(run_dir, pos, n_comp, sel, layers, valid_k, fill):
    """Assert one decode step against the raw indexer_trace.jsonl (first decode record)."""
    p = os.path.join(run_dir, "traces", "indexer_trace.jsonl")
    if not os.path.exists(p):
        return "jsonl missing, check skipped"
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get("phase") == 1 and r.get("layer") in set(layers.tolist()):
                break
        else:
            return "no decode record found, check skipped"
    t = int(np.where(pos == r["pos"])[0][0])
    li = int(np.where(layers == r["layer"])[0][0])
    assert n_comp[t] == r["n_comp"], (n_comp[t], r["n_comp"])
    assert valid_k[t, li] == r["valid_k"], (valid_k[t, li], r["valid_k"])
    js = np.array(sorted(r["sel"]), dtype=np.int64)
    ex = sel[t, li][sel[t, li] != fill].astype(np.int64)
    assert np.array_equal(js, ex), "sel mismatch vs jsonl"
    return f"ok (pos={r['pos']}, layer={r['layer']}, n_comp={r['n_comp']}, valid_k={r['valid_k']})"


def export_run(run_dir, out_dir, do_check=True):
    run_id = os.path.basename(run_dir.rstrip("/"))
    pq = os.path.join(run_dir, "traces", "selected_kv.parquet")
    df = pd.read_parquet(pq, columns=["layer_id", "decode_step", "decode_position",
                                      "compressed_kv_index", "selected_rank", "is_valid",
                                      "compression_ratio"])
    ratios = df.compression_ratio.unique()
    assert len(ratios) == 1, f"{run_id}: mixed compression ratios {ratios}"
    ratio = int(ratios[0])
    layers = np.array(sorted(df.layer_id.unique()), dtype=np.int32)
    steps = np.array(sorted(df.decode_step.unique()))
    assert np.array_equal(steps, np.arange(len(steps))), f"{run_id}: decode_step not contiguous"
    k = int(df.selected_rank.max()) + 1
    T, L = len(steps), len(layers)

    pos_by_step = df.groupby("decode_step").decode_position.agg(["min", "max"])
    assert (pos_by_step["min"] == pos_by_step["max"]).all(), f"{run_id}: pos varies within a step"
    pos = pos_by_step["min"].to_numpy().astype(np.int32)
    n_comp = ((pos + 1) // ratio).astype(np.int32)  # verified against every jsonl record: NOT pos // ratio

    max_idx = int(df.compressed_kv_index.max())
    dtype = np.uint16 if max_idx < 65536 else np.uint32
    fill = np.iinfo(dtype).max
    sel = np.full((T, L, k), fill, dtype=dtype)
    valid_k = np.zeros((T, L), dtype=np.int32)

    lmap = {int(l): i for i, l in enumerate(layers)}
    v = df[df.is_valid & (df.compressed_kv_index >= 0)]
    t_idx = v.decode_step.to_numpy()
    l_idx = v.layer_id.map(lmap).to_numpy()
    r_idx = v.selected_rank.to_numpy()
    sel[t_idx, l_idx, r_idx] = v.compressed_kv_index.to_numpy().astype(dtype)
    np.add.at(valid_k, (t_idx, l_idx), 1)
    # sort each row ascending (fill value sorts last), and make sure every row has valid_k valid slots
    sel.sort(axis=2)
    assert ((sel != fill).sum(axis=2) == valid_k).all(), f"{run_id}: valid_k mismatch"

    manifest_src = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    model = manifest_src.get("model_name", "DeepSeek-V4-Flash")
    gen = os.path.join(run_dir, "outputs", "generations.jsonl")
    prompt_tokens = None
    if os.path.exists(gen):
        with open(gen) as f:
            first = json.loads(f.readline())
        prompt_tokens = first.get("prompt_token_count")

    check = jsonl_check(run_dir, pos, n_comp, sel, layers, valid_k, fill) if do_check else "skipped"

    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"{run_id}.npz")
    np.savez_compressed(npz, sel=sel, valid_k=valid_k, pos=pos, n_comp=n_comp, layers=layers,
                        k=np.int32(k), ratio=np.int32(ratio), model=np.array(model))
    man = {
        "run_id": run_id,
        "model": model,
        "quantization": manifest_src.get("quantization"),
        "engine": "ds4 (antirez, instrumented) CPU",
        "benchmark": manifest_src.get("benchmark_version", manifest_src.get("benchmark_name")),
        "task": manifest_src.get("task_subset"),
        "prompt_tokens": prompt_tokens,
        "context_length_actual_tokens": manifest_src.get("context_length_actual_tokens"),
        "steps": T, "layers": L, "k": k, "ratio": ratio,
        "sel_dtype": np.dtype(dtype).name, "fill_value": int(fill),
        "max_index": max_idx, "pos_first": int(pos[0]), "pos_last": int(pos[-1]),
        "all_rows_full": bool((valid_k == k).all()),
        "source_parquet": pq, "source_parquet_sha256": sha256(pq),
        "npz": npz, "npz_bytes": os.path.getsize(npz),
        "jsonl_one_step_check": check,
        "export_date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "work/experiment/scripts/export_v6_traces.py",
        "script_git_hash_sparse_attn_cpu": git_hash(os.path.join(EXP, "..", "..", "01_github", "sparse_attn_cpu")),
        "units": "V4: each sel entry is a compressed block of `ratio` tokens; n_comp = (pos + 1) // ratio",
    }
    json.dump(man, open(os.path.join(out_dir, f"{run_id}.manifest.json"), "w"), indent=2)
    print(f"{run_id:32s} steps={T:5d} L={L} k={k} ratio={ratio} {np.dtype(dtype).name} "
          f"{man['npz_bytes']/1e6:7.1f} MB  check: {check}")
    return man


def family_of(run_id):
    if run_id.startswith("niah_single_2_"):
        return "ruler", f"ruler_{run_id.split('_')[3]}"
    if run_id.startswith("lb_"):
        return "longbench", "longbench_" + run_id[3:].rsplit("_s", 1)[0]
    if run_id.startswith("longform_"):
        return "longdecode", "longdecode"
    return "other", "other"


def retention_curves(runs_root, manifests, out_dir):
    fams = {}
    for m in manifests:
        rid = m["run_id"]
        rdir = os.path.join(runs_root, rid)
        summ = json.load(open(os.path.join(rdir, "analysis", "metrics_run_summary.json")))
        ret = {int(kk.split("_")[1]): float(vv) for kk, vv in summ["overall_retention"].items()}
        src = "analysis/metrics_run_summary.json:overall_retention"
        ext_p = os.path.join(rdir, "analysis", "extended_retention.json")
        ext = None
        if os.path.exists(ext_p):
            e = json.load(open(ext_p))
            ext = {int(kk): float(vv) for kk, vv in e["retention"].items()}
            src += " + analysis/extended_retention.json:retention (pooled over (layer,step) pairs)"
        fam, key = family_of(rid)
        d = fams.setdefault(key, {"family": fam, "runs": {}})
        d["runs"][rid] = {"steps": m["steps"], "prompt_tokens": m["prompt_tokens"],
                          "retention": {str(l): ret[l] for l in sorted(ret)},
                          "retention_extended": ({str(l): ext[l] for l in sorted(ext)} if ext else None),
                          "source": src}
    for key, d in fams.items():
        lags = sorted({int(l) for r in d["runs"].values() for l in r["retention"]})
        d["mean_retention"] = {str(l): float(np.mean([r["retention"][str(l)] for r in d["runs"].values()
                                                       if str(l) in r["retention"]])) for l in lags}
        d["n_runs"] = len(d["runs"])
    out = {
        "description": "Retention at lag L = mean over (layer, step) of |S_t ∩ S_{t-L}| / |S_t| "
                       "(per-run values are unweighted means over the 21 CSA layers of the per-layer "
                       "means; extended lags for the long-decode run pool all (layer, step) pairs). "
                       "Model DeepSeek-V4-Flash (ds4 CPU, IQ2), k = 512 compressed blocks of 4 tokens.",
        "consumer_note": "For the retention_pred policy use a LEAVE-ONE-OUT curve: when evaluating run X, "
                         "build the curve from the other runs of the same family/benchmark (for the "
                         "single-run families ruler_<L> and longdecode, use the neighbouring contexts / the "
                         "pooled LongBench curve) so the evaluated trace is not fitted to itself. "
                         "Traces shorter than ~500 steps understate the decay beyond lag 64 "
                         "(HANDOFF §3); only longdecode carries lags 128–2048.",
        "lags_standard": LAGS_STD, "lags_extended": LAGS_EXT,
        "families": fams,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "work/experiment/scripts/export_v6_traces.py",
    }
    p = os.path.join(out_dir, "retention_curves.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p, "families:", {k: v["n_runs"] for k, v in fams.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=os.path.join(EXP, "runs"))
    ap.add_argument("--out-dir", default=os.path.join(EXP, "exports", "v6"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-jsonl-check", action="store_true")
    a = ap.parse_args()
    runs = sorted(d for d in os.listdir(a.runs_root)
                  if os.path.exists(os.path.join(a.runs_root, d, "traces", "selected_kv.parquet"))
                  and d not in SKIP)
    if a.only:
        runs = [r for r in runs if r in set(a.only)]
    mans = []
    for r in runs:
        mans.append(export_run(os.path.join(a.runs_root, r), a.out_dir, not a.no_jsonl_check))
    retention_curves(a.runs_root, mans, a.out_dir)
    tot = sum(m["npz_bytes"] for m in mans)
    print(f"{len(mans)} runs exported, total {tot/1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
