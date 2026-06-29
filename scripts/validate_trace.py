#!/usr/bin/env python3
"""Validate trace integrity + unit-test metric helpers (doc §19, §36).
Usage: validate_trace.py <run_dir>   (run_dir optional -> unit tests only)
Exits non-zero on any failure.
"""
import json, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locality_lib import (topk_indices, jaccard, overlap_fraction, weighted_overlap,
                          random_expected_overlap_fraction, ranks_from_scores)

FAIL = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAIL.append(msg)


def unit_tests():
    print("[unit tests]")
    # top-k unique
    check(topk_indices([0.1, 0.9, 0.5, 0.3], 2) == [1, 2], "top-k picks highest two")
    # ties -> index asc tie-break
    check(topk_indices([1.0, 1.0, 1.0], 2) == [0, 1], "ties broken by ascending index")
    # all negative
    check(topk_indices([-5.0, -1.0, -9.0], 1) == [1], "all-negative top-1")
    # N < k
    check(topk_indices([0.2, 0.4], 5) == [1, 0], "N<k returns all, ranked")
    # jaccard
    check(abs(jaccard([1, 2, 3], [2, 3, 4]) - 0.5) < 1e-9, "jaccard 2/4=0.5")
    check(jaccard([], []) == 1.0, "jaccard empty=1")
    # overlap fraction
    check(abs(overlap_fraction([1, 2, 3, 4], [2, 4]) - 0.5) < 1e-9, "overlap 2/4")
    # weighted overlap identical lists = 1
    check(abs(weighted_overlap([3, 1, 2], [3, 1, 2]) - 1.0) < 1e-9, "weighted overlap self=1")
    # weighted overlap disjoint = 0
    check(weighted_overlap([1, 2], [3, 4]) == 0.0, "weighted overlap disjoint=0")
    # random expected
    check(abs(random_expected_overlap_fraction(512, 1024) - 0.5) < 1e-9, "random overlap k/N")
    check(random_expected_overlap_fraction(512, 256) == 1.0, "random overlap N<=k -> 1")
    # ranks fallback (no scores) = ascending index
    check(ranks_from_scores([5, 2, 9], {}) == [2, 5, 9], "rank fallback ascending index")


def integrity(run_dir):
    print(f"[integrity] {run_dir}")
    TR = os.path.join(run_dir, "traces")
    sel = pd.read_parquet(os.path.join(TR, "selected_kv.parquet"))
    le = pd.read_parquet(os.path.join(TR, "layer_events.parquet"))
    dt = pd.read_parquet(os.path.join(TR, "decode_tokens.parquet"))
    ss = pd.read_parquet(os.path.join(TR, "score_summaries.parquet"))
    gen = json.loads(open(os.path.join(run_dir, "outputs", "generations.jsonl")).read().splitlines()[0])
    mc = json.load(open(os.path.join(run_dir, "model_config.json")))
    n_csa = sum(1 for l in mc["layer_map"] if l["attention_type"] == "CSA")

    # indices in range
    bad = 0
    for _, g in sel.groupby(["decode_step", "layer_id"]):
        nc = ss[(ss.decode_step == g.decode_step.iloc[0]) & (ss.layer_id == g.layer_id.iloc[0])]
        if len(nc):
            ncv = nc.n_candidates_visible.iloc[0]
            if (g.compressed_kv_index >= ncv).any() or (g.compressed_kv_index < 0).any():
                bad += 1
    check(bad == 0, "all selected indices within [0, n_candidates)")

    # ranks contiguous per (step,layer)
    rank_ok = True
    for _, g in sel.groupby(["decode_step", "layer_id"]):
        rs = sorted(g.selected_rank.tolist())
        if rs != list(range(len(rs))):
            rank_ok = False
            break
    check(rank_ok, "selected ranks contiguous 0..k-1 per (step,layer)")

    # valid_selected_count matches selected rows
    cnt_ok = True
    sel_counts = sel.groupby(["decode_step", "layer_id"]).size()
    for (dstep, lid), n in sel_counts.items():
        row = ss[(ss.decode_step == dstep) & (ss.layer_id == lid)]
        if len(row) and int(row.valid_selected_count.iloc[0]) != n:
            cnt_ok = False
            break
    check(cnt_ok, "valid_selected_count matches #selected rows")

    # every decode step has all CSA layer events
    csa = le[le.is_sparse_layer]
    per_step = csa.groupby("decode_step").layer_id.nunique()
    check((per_step == n_csa).all(), f"every decode step has all {n_csa} CSA layer events")

    # decode_tokens count matches generation
    check(len(dt) == gen["generated_token_count"], "decode_tokens rows == generated tokens")

    # tracing on/off identity (smoke marker, if recorded)
    sm = os.path.join(os.path.dirname(run_dir), "..", "smoke_identity.txt")
    if os.path.exists(sm):
        check(open(sm).read().strip() == "IDENTICAL", "tracing on/off produced identical tokens (smoke)")


def main():
    unit_tests()
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        integrity(sys.argv[1])
    if FAIL:
        print(f"\nVALIDATION FAILED: {len(FAIL)} check(s) failed")
        sys.exit(1)
    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
