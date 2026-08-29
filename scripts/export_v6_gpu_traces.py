#!/usr/bin/env python3
"""Export the GPU-campaign (vLLM, DeepSeek-V3.2 / GLM-5.2 / GLM-5) top-k selection traces to the v6 npz schema.

Source: the sharded run dirs committed in sparse_attn_cpu (docs/<sweep>/runs/<run_id>/, campaign commit a28245c +
raw-trace batches, HEAD 1c2a3f2): traces/indexer_trace.jsonl.gz.partNN with SHARDS.json sha256s.
Per run this script
  1. verifies the sha256 of every file listed in SHARDS.json (parts + manifests + analysis files),
  2. reassembles the gz shards under <gpu-dir>/<run_id>/traces/<name>.jsonl.gz (off-repo; the raw JSONL is NOT
     written -- it is decompressed in a stream and its raw sha256 is checked against SHARDS.json),
  3. parses the phase-1 (decode) records and writes <out-dir>/<run_id>.npz + <run_id>.manifest.json in the schema of
     docs/00_doc/v6_export_format.md: ratio = 1, k = 2048, n_comp = pos + 1, ids = per-request token positions,
     uint16 when max index < 65536 else uint32.
The selected_kv.parquet step of the CPU pipeline is skipped on purpose: the v6 schema is built directly from the
JSONL (the 256 M-row parquet of a 2048-step run would cost ~10 GB per run and adds nothing).
GLM-5.2 "computing-only" variant (b): the `_b` run dirs carry analysis only; their selection is the subset of the
(a) trace restricted to the layers whose model_config layer_map says CSA (topk_computed layers). This script writes
<run_id>_b.npz from the same parse when --variants-b is given.

Usage:
  export_v6_gpu_traces.py --sweep gpu_sweep|glm_sweep [--repo-root R] [--gpu-dir G] [--out-dir O]
                          [--only RUN_ID ...] [--workers 4] [--variants-b] [--priority]
Defaults: repo-root = <root>/01_github/sparse_attn_cpu, gpu-dir = <experiment>/exports/gpu,
          out-dir = <experiment>/exports/v6_v32 (gpu_sweep) or v6_glm (glm_sweep).
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time

import numpy as np

try:
    import orjson as _oj
    loads = _oj.loads
except Exception:  # pragma: no cover
    loads = json.loads

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
CACHE_NOTE = ("vLLM 0.28.0 default kv_cache_dtype=auto: bf16 MLA latent rows 576 x 2 B = 1152 B/token/layer "
              "+ 132 B fp8 indexer key/token/layer (not the 656 B fp8_ds_mla layout assumed in the campaign docs)")
RUNG_RE = re.compile(r"_(8192|16384|32768|65536|131072)_")


def sha256_file(path, bufsize=1 << 24):
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


def run_kind(run_id):
    return "ld" if re.search(r"_ld\d+$", run_id) else "bf"


def rung_of(run_id):
    m = RUNG_RE.search(run_id)
    return int(m.group(1)) if m else None


def verify_and_reassemble(run_dir, gpu_run_dir, shards):
    """Check every sha256 in SHARDS.json; concatenate the parts into <gpu_run_dir>/traces/<name>.gz.
    Returns {name: (gz_path, raw_sha256_expected)} for the trace files."""
    bad = []
    traces = {}
    for rel, val in shards["files"].items():
        if isinstance(val, dict):  # the logical gz entry: {"parts", "gz_bytes", "raw_bytes", "raw_sha256"}
            traces[rel] = val
            continue
        p = os.path.join(run_dir, rel)
        if not os.path.exists(p) or sha256_file(p) != val:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"sha256 mismatch/missing: {bad}")
    out = {}
    os.makedirs(os.path.join(gpu_run_dir, "traces"), exist_ok=True)
    for rel, meta in traces.items():  # rel = traces/indexer_trace.jsonl.gz
        parts = [os.path.join(run_dir, f"{rel}.part{i:02d}") for i in range(meta["parts"])]
        dst = os.path.join(gpu_run_dir, rel)
        if not (os.path.exists(dst) and os.path.getsize(dst) == meta["gz_bytes"]):
            with open(dst + ".tmp", "wb") as fo:
                for part in parts:
                    with open(part, "rb") as fi:
                        while True:
                            b = fi.read(1 << 24)
                            if not b:
                                break
                            fo.write(b)
            os.replace(dst + ".tmp", dst)
        assert os.path.getsize(dst) == meta["gz_bytes"], f"{rel}: gz size {os.path.getsize(dst)} != {meta['gz_bytes']}"
        out[os.path.basename(rel)] = (dst, meta["raw_sha256"], meta["raw_bytes"])
    return out


def parse_indexer(gz_path, raw_sha_expected):
    """Stream-decompress the indexer trace, check the raw sha256, return per-step arrays."""
    h = hashlib.sha256()
    recs = []  # (pos, layer, sel list, n_comp, valid_k, top_k, ratio, topk_computed, shared_from)
    with gzip.open(gz_path, "rb") as f:
        for line in f:
            h.update(line)
            if not line.strip():
                continue
            r = loads(line)
            if r.get("phase") != 1:
                continue
            recs.append(r)
    got = h.hexdigest()
    if got != raw_sha_expected:
        raise RuntimeError(f"raw sha256 mismatch {got} != {raw_sha_expected}")
    return recs


def build_arrays(recs, layer_filter=None):
    pos_all = sorted({r["pos"] for r in recs})
    layers = sorted({r["layer"] for r in recs if layer_filter is None or r["layer"] in layer_filter})
    if not layers:
        raise RuntimeError("no layers after filter")
    k = max(r["top_k"] for r in recs)
    ratios = {r.get("ratio", 1) for r in recs}
    assert ratios == {1}, f"unexpected ratio set {ratios}"
    pos = np.array(pos_all, dtype=np.int32)
    T, L = len(pos), len(layers)
    tmap = {p: i for i, p in enumerate(pos_all)}
    lmap = {l: i for i, l in enumerate(layers)}
    max_idx = max((max(r["sel"]) for r in recs if r["sel"] and (layer_filter is None or r["layer"] in layer_filter)),
                  default=0)
    dtype = np.uint16 if max_idx < 65536 else np.uint32
    fill = np.iinfo(dtype).max
    sel = np.full((T, L, k), fill, dtype=dtype)
    valid_k = np.zeros((T, L), dtype=np.int32)
    n_comp = np.zeros(T, dtype=np.int32)
    seen = np.zeros((T, L), dtype=bool)
    for r in recs:
        if layer_filter is not None and r["layer"] not in layer_filter:
            continue
        t, li = tmap[r["pos"]], lmap[r["layer"]]
        if seen[t, li]:
            raise RuntimeError(f"duplicate record pos={r['pos']} layer={r['layer']}")
        seen[t, li] = True
        s = np.asarray(r["sel"], dtype=np.int64)
        s = s[s >= 0]
        assert len(s) == r["valid_k"], (len(s), r["valid_k"])
        assert r["n_comp"] == r["pos"] + 1, (r["n_comp"], r["pos"])
        n_comp[t] = r["n_comp"]
        s.sort()
        sel[t, li, :len(s)] = s.astype(dtype)
        valid_k[t, li] = len(s)
    if not seen.all():
        missing = int((~seen).sum())
        raise RuntimeError(f"{missing} (step, layer) cells without a record")
    assert (n_comp == pos + 1).all()
    assert ((sel != fill).sum(axis=2) == valid_k).all()
    return dict(sel=sel, valid_k=valid_k, pos=pos, n_comp=n_comp, layers=np.array(layers, dtype=np.int32),
                k=k, ratio=1, dtype=dtype, fill=fill, max_idx=max_idx)


def write_npz(arr, out_dir, run_id, model, variant, manifest_src, meta, shards_sha, gz_path, extra):
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"{run_id}.npz")
    np.savez_compressed(npz, sel=arr["sel"], valid_k=arr["valid_k"], pos=arr["pos"], n_comp=arr["n_comp"],
                        layers=arr["layers"], k=np.int32(arr["k"]), ratio=np.int32(arr["ratio"]),
                        model=np.array(model))
    T, L = arr["valid_k"].shape
    man = {
        "run_id": run_id,
        "model": model,
        "variant": variant,
        "quantization": manifest_src.get("quantization"),
        "engine": "vLLM 0.28.0 (2cf0a69) TP8 eager, 8x B200, FLASHINFER_MLA_SPARSE + FLASHINFER_TRTLLM fp8 MoE",
        "benchmark": manifest_src.get("benchmark_name"),
        "task": manifest_src.get("task_subset"),
        "sample_id": manifest_src.get("sample_id"),
        "run_kind": manifest_src.get("kind", run_kind(run_id)),
        "context_rung": manifest_src.get("context_length_target"),
        "prompt_tokens": manifest_src.get("context_length_actual_tokens"),
        "context_length_actual_tokens": manifest_src.get("context_length_actual_tokens"),
        "max_new_tokens": manifest_src.get("max_new_tokens"),
        "decode_steps": T,
        "steps": T, "layers": L, "k": arr["k"], "ratio": 1,
        "num_model_layers": meta.get("num_layers"),
        "index_topk_freq": meta.get("index_topk_freq"),
        "index_skip_topk_offset": meta.get("index_skip_topk_offset"),
        "sel_dtype": np.dtype(arr["dtype"]).name, "fill_value": int(arr["fill"]),
        "max_index": int(arr["max_idx"]), "pos_first": int(arr["pos"][0]), "pos_last": int(arr["pos"][-1]),
        "all_rows_full": bool((arr["valid_k"] == arr["k"]).all()),
        "cache_layout_note": CACHE_NOTE,
        "source_run_dir": extra["source_rel"],
        "source_shards_json_sha256": shards_sha,
        "source_gz": gz_path, "source_raw_sha256_verified": True,
        "npz": npz, "npz_bytes": os.path.getsize(npz),
        "export_date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "work/experiment/scripts/export_v6_gpu_traces.py",
        "script_git_hash_sparse_attn_cpu": extra["git"],
        "campaign_commit_sparse_attn_cpu": "a28245c (analysis) + raw-trace batches, HEAD 1c2a3f2",
        "units": "ids are per-request 0-based token positions (ratio 1); n_comp = pos + 1 = tokens in the cache; "
                 "steps are decode steps only (the vLLM hook traces decode; the first records are the prefill tail)",
    }
    if variant == "computing_only":
        man["note"] = ("layers restricted to the top-k computing layers of GLM-5.2 (index_topk_freq 4, offset 3); "
                       "the shared layers reproduce their producer's set bit-for-bit (exp2 smoke), so the all-layer "
                       "npz of the same run_id (without _b) is the memory-system view")
    json.dump(man, open(os.path.join(out_dir, f"{run_id}.manifest.json"), "w"), indent=2)
    return man


def export_one(args):
    run_dir, gpu_dir, out_dir, variants_b, gitrev, repo_root = args
    run_id = os.path.basename(run_dir.rstrip("/"))
    t0 = time.time()
    try:
        shards = json.load(open(os.path.join(run_dir, "SHARDS.json")))
        shards_sha = sha256_file(os.path.join(run_dir, "SHARDS.json"))
        gpu_run_dir = os.path.join(gpu_dir, run_id)
        traces = verify_and_reassemble(run_dir, gpu_run_dir, shards)
        if "indexer_trace.jsonl.gz" not in traces:
            return (run_id, "skip: no indexer trace", 0)
        gz, raw_sha, _ = traces["indexer_trace.jsonl.gz"]
        recs = parse_indexer(gz, raw_sha)
        manifest_src = json.load(open(os.path.join(run_dir, "run_manifest.json")))
        meta = json.load(open(os.path.join(run_dir, "meta.json"))) if os.path.exists(os.path.join(run_dir, "meta.json")) else {}
        model = manifest_src.get("model_name")
        extra = {"source_rel": os.path.relpath(run_dir, repo_root), "git": gitrev}
        arr = build_arrays(recs)
        man = write_npz(arr, out_dir, run_id, model, manifest_src.get("variant", "all_layers"), manifest_src, meta,
                        shards_sha, gz, extra)
        msg = f"steps={man['steps']} L={man['layers']} k={man['k']} {man['sel_dtype']} {man['npz_bytes']/1e6:.1f} MB"
        if variants_b:
            bdir = run_dir + "_b"
            if os.path.isdir(bdir):
                mc = json.load(open(os.path.join(bdir, "model_config.json")))
                comp = {l["layer_id"] for l in mc["layer_map"] if l["attention_type"] == "CSA"}
                arr_b = build_arrays(recs, layer_filter=comp)
                man_b = json.load(open(os.path.join(bdir, "run_manifest.json")))
                extra_b = {"source_rel": os.path.relpath(bdir, repo_root) + " (analysis) + " + extra["source_rel"] + " (trace)", "git": gitrev}
                mb = write_npz(arr_b, out_dir, run_id + "_b", model, "computing_only", man_b, meta, shards_sha, gz, extra_b)
                msg += f" | _b L={mb['layers']} {mb['npz_bytes']/1e6:.1f} MB"
        return (run_id, "ok " + msg, time.time() - t0)
    except Exception as e:  # noqa: BLE001
        return (run_id, f"FAIL {type(e).__name__}: {e}", time.time() - t0)


def priority_key(run_id):
    kind = run_kind(run_id)
    rung = rung_of(run_id) or 0
    model = run_id.split("_")[0]
    mrank = {"v32": 0, "glm52": 1, "glm5": 2}.get(model, 3)
    krank = 0 if kind == "ld" else 1
    rrank = {131072: 0, 65536: 1, 32768: 2, 16384: 3, 8192: 4}.get(rung, 5)
    return (mrank, 0 if rung >= 65536 else 1, krank, rrank, run_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, choices=["gpu_sweep", "glm_sweep"])
    ap.add_argument("--repo-root", default=os.path.join(ROOT, "01_github", "sparse_attn_cpu"))
    ap.add_argument("--gpu-dir", default=os.path.join(EXP, "exports", "gpu"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variants-b", action="store_true", help="also write <run_id>_b.npz (GLM-5.2 computing-only)")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(EXP, "exports", "v6_v32" if a.sweep == "gpu_sweep" else "v6_glm")
    runs_root = os.path.join(a.repo_root, "docs", a.sweep, "runs")
    runs = sorted((d for d in os.listdir(runs_root)
                   if os.path.isdir(os.path.join(runs_root, d, "traces")) and not d.endswith("_b")
                   and os.path.exists(os.path.join(runs_root, d, "SHARDS.json"))), key=priority_key)
    if a.only:
        runs = [r for r in runs if r in set(a.only)]
    if a.skip_existing:
        runs = [r for r in runs if not os.path.exists(os.path.join(out_dir, r + ".manifest.json"))]
    gitrev = git_hash(a.repo_root)
    jobs = [(os.path.join(runs_root, r), a.gpu_dir, out_dir, a.variants_b, gitrev, a.repo_root) for r in runs]
    print(f"{len(jobs)} runs -> {out_dir} (workers {a.workers})", flush=True)
    fails = []
    with mp.Pool(a.workers) as pool:
        for rid, msg, secs in pool.imap(export_one, jobs):
            print(f"{rid:48s} {msg}  [{secs:.0f}s]", flush=True)
            if msg.startswith("FAIL"):
                fails.append((rid, msg))
    print(f"done: {len(jobs) - len(fails)} ok, {len(fails)} failed", flush=True)
    for rid, msg in fails:
        print("  ", rid, msg)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
