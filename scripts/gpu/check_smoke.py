#!/usr/bin/env python3
"""exp0 §5 / exp1 §3 unit checks on a hook trace.
Usage: check_smoke.py --trace <SEL_TRACE dir> --req <request_id> --n-gen <generated tokens> --num-layers 61
                      [--top-k 2048] [--prompt-len N] [--hookoff-json a.json --hookon-json b.json]
                      [--identity-out <file>] [--expected-computed <json list>]
Prints PASS/FAIL per check; exits 1 on any FAIL.
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--req", required=True)
ap.add_argument("--n-gen", type=int, required=True)
ap.add_argument("--num-layers", type=int, default=61)
ap.add_argument("--top-k", type=int, default=2048)
ap.add_argument("--prompt-len", type=int, default=None)
ap.add_argument("--hookoff-json", default=None)
ap.add_argument("--hookon-json", default=None)
ap.add_argument("--identity-out", default=None)
ap.add_argument("--expected-computed", default=None, help="JSON list of layer ids that compute top-k")

def _resolve_trace_file(trace_dir, request_id):
    """vLLM's engine-internal request id is '<request_id>-<hex>'; find the by_req file by exact or prefix match."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(request_id))
    d = os.path.join(trace_dir, "by_req")
    exact = os.path.join(d, safe + ".jsonl")
    if os.path.exists(exact):
        return exact
    cands = sorted(f for f in os.listdir(d) if f.startswith(safe + "-") and f.endswith(".jsonl"))
    if len(cands) != 1:
        raise FileNotFoundError(f"{len(cands)} trace files match request {request_id!r} in {d}: {cands[:5]}")
    return os.path.join(d, cands[0])


a = ap.parse_args()
FAIL = []


def check(c, msg):
    print(("  PASS: " if c else "  FAIL: ") + msg)
    if not c:
        FAIL.append(msg)


p = _resolve_trace_file(a.trace, a.req)
recs = [json.loads(l) for l in open(p) if l.strip()]
moe = [r for r in recs if r.get("moe")]
recs = [r for r in recs if not r.get("moe")]
dec = [r for r in recs if r["phase"] == 1]
steps = sorted({r["pos"] for r in dec})
print(f"[trace] {p}: {len(recs)} records, {len(dec)} decode, steps={len(steps)} pos {steps[:1]}..{steps[-1:]}")
check(len(dec) == a.num_layers * a.n_gen, f"decode records == {a.num_layers} layers x {a.n_gen} steps (got {len(dec)})")
check(len(steps) == a.n_gen, f"{a.n_gen} distinct decode positions (got {len(steps)})")
check(all(b - x == 1 for x, b in zip(steps, steps[1:])), "pos increases by exactly 1 per step")
if a.prompt_len is not None:
    check(steps and steps[0] == a.prompt_len - 1, f"first decode pos == prompt_len-1 ({a.prompt_len - 1})")
bad_range = sum(1 for r in dec if r["sel"] and (min(r["sel"]) < 0 or max(r["sel"]) > r["pos"]))
check(bad_range == 0, "all sel in [0, pos]")
check(all(len(r["sel"]) <= a.top_k for r in dec), f"len(sel) <= {a.top_k}")
check(all(len(r["sel"]) == min(a.top_k, r["n_comp"]) for r in dec), "len(sel) == min(top_k, n_comp)")
check(all(r["n_comp"] == r["pos"] + 1 for r in dec), "n_comp == pos + 1")
check(all(len(set(r["sel"])) == len(r["sel"]) for r in dec), "no duplicate indices in sel")
check(all(r["sel"] == sorted(r["sel"]) for r in dec), "sel sorted ascending")
per_step = {}
for r in dec:
    per_step.setdefault(r["pos"], []).append(r["layer"])
check(all(sorted(v) == list(range(a.num_layers)) for v in per_step.values()),
      f"each step has layers 0..{a.num_layers - 1} exactly once")
if a.expected_computed:
    exp = set(json.loads(a.expected_computed))
    got = {r["layer"] for r in dec if r.get("topk_computed", True)}
    check(got == exp, f"computing-layer set == config skip rule ({len(exp)} layers)")
    # shared layers duplicate their producer bit-for-bit
    by = {(r["pos"], r["layer"]): r for r in dec}
    ok = all(r["sel"] == by[(r["pos"], r["shared_from_layer"])]["sel"] for r in dec if not r.get("topk_computed", True))
    check(ok, "every shared layer's sel == producer layer's sel")
if moe:
    mdec = [r for r in moe if r["phase"] == 1]
    n_moe_layers = len({r["layer"] for r in mdec})
    check(len(mdec) == n_moe_layers * a.n_gen, f"MoE records == {n_moe_layers} layers x {a.n_gen} steps (got {len(mdec)})")
    check(all(len(r["sel"]) == 8 and all(0 <= e < 256 for e in r["sel"]) for r in mdec), "8 expert ids in [0,256) per MoE record")
    check(all(len(set(r["sel"])) == 8 for r in mdec), "8 distinct experts per record")
    check(min(r["layer"] for r in mdec) == 3, "MoE layers start at 3 (layers 0-2 dense)")
if a.hookoff_json and a.hookon_json:
    off = json.load(open(a.hookoff_json))["gens"][0]["token_ids"]
    on = json.load(open(a.hookon_json))["gens"][0]["token_ids"]
    same = off == on
    check(same, f"hook-on tokens == hook-off tokens ({on} vs {off})")
    if a.identity_out:
        open(a.identity_out, "w").write("IDENTICAL\n" if same else "DIFFERENT\n")
if FAIL:
    print(f"\nCHECKS FAILED: {len(FAIL)}")
    sys.exit(1)
print("\nALL CHECKS PASSED")
