"""exp3 re-index hook: physically re-order a request's already-written KV prefix between decode steps.

Two implementations (exp3 §2), both applied identically to EVERY layer tensor of BOTH KV-cache groups
(MLA latent `kv_c_and_k_pe_cache`, 656 B rows, and the indexer key cache, 132 B rows; vLLM layout
(num_blocks, block_size=64, row_bytes) per layer, see flashinfer_mla_sparse.py:127-134 / indexer.py:180):
  A  block-granular ("page-table" baseline): swap the CONTENTS of two 64-token blocks of one request in every
     layer tensor, and swap the two entries in the request's block-table rows (both groups' BlockTable.np and the
     runner's CachedRequestState.block_ids). Index space stays "position"; the table translates.
  B  entry-granular ("re-index" design): swap two ROWS (slots) inside the request's written prefix in every layer
     tensor; block tables untouched. Indices then ARE physical order; perm_log.jsonl replays them to original order.
Modes (REINDEX_MODE): off | ctrl_identity | perm_once | perm_periodic ; REINDEX_IMPL=A|B ; REINDEX_SEED (7);
REINDEX_PERIOD (4 steps); REINDEX_FRAC (0.10 of the prefix per periodic event); REINDEX_LOG=<dir> (perm_log).
Rules: swaps only within one request; only positions < seq_len-64 (never the block being appended to);
applied before the runner's _prepare_inputs computes this step's slot_mapping (we wrap execute_model), after
torch.cuda.synchronize(); identical on every TP rank (deterministic per-request RNG: seed ^ hash(request id)),
which is required because the MLA/indexer caches are replicated across TP ranks.
Also imports selhook.worker_ext so the selection trace can run in the same process (SEL_TRACE).
"""
import hashlib
import json
import os

import numpy as np
import torch

import selhook.worker_ext as selhook_trace  # noqa: F401  (installs the trace hook if SEL_TRACE is set)

CFG = {"mode": os.environ.get("REINDEX_MODE", "off"), "impl": os.environ.get("REINDEX_IMPL", "A"),
       "seed": int(os.environ.get("REINDEX_SEED", "7")), "period": int(os.environ.get("REINDEX_PERIOD", "4")),
       "frac": float(os.environ.get("REINDEX_FRAC", "0.10")), "log_dir": os.environ.get("REINDEX_LOG")}
INSTALL = CFG["mode"] != "off" or os.environ.get("REINDEX_INSTALL", "0") == "1"
BLOCK = 64

_st = {"installed": False, "rank": None, "runner": None, "groups": None, "req": {}, "errors": [], "n_swaps": 0,
       "n_events": 0, "log_files": {}}


def _rng(req_id):
    h = int(hashlib.sha256(f"{CFG['seed']}:{req_id}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(h)


def _groups(runner):
    """[(group_id, [layer kv tensors...], block_table)] for the two attention groups."""
    fc = runner.vllm_config.compilation_config.static_forward_context
    out = []
    for g, grp in enumerate(runner.kv_cache_config.kv_cache_groups):
        tensors = []
        for name in grp.layer_names:
            layer = fc.get(name)
            t = getattr(layer, "kv_cache", None)
            if isinstance(t, (list, tuple)):
                t = t[0]
            if t is not None and t.numel() > 0:
                tensors.append(t)
        if tensors:
            out.append((g, tensors, runner.input_batch.block_table[g]))
    widths = sorted({int(t.shape[-1]) for _, ts, _ in out for t in ts})
    _st["indexer_row_bytes"] = widths[0] if len(widths) > 1 else None   # 132 (indexer) vs 576/656 (MLA latent)
    _st["row_widths"] = widths
    return out


def _log(req_id, rec):
    LOG_DIR = CFG["log_dir"]
    if not LOG_DIR:
        return
    f = _st["log_files"].get(req_id)
    if f is None:
        os.makedirs(LOG_DIR, exist_ok=True)
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(req_id))
        f = open(os.path.join(LOG_DIR, safe + ".jsonl"), "a")
        _st["log_files"][req_id] = f
    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    f.flush()


def _apply_perm_A(runner, req_state, row, perm):
    """Impl A: physical permutation of the request's logical blocks 0..nb-1: content of logical block i moves
    to the physical block that logical block perm[i] occupied; the block-table row is updated so that logical
    block i still resolves to its own content (index space stays 'position'). One gather per layer tensor."""
    nb = len(perm)
    perm_t = None
    for g, tensors, bt in _st["groups"]:
        ids = req_state.block_ids[g]
        old = np.asarray(ids[:nb], dtype=np.int64)          # physical block per logical block
        dst = old[perm]                                     # content of logical i goes to physical old[perm[i]]
        if perm_t is None or perm_t[0] is not tensors[0].device:
            pass
        src_t = torch.as_tensor(old, device=tensors[0].device)
        dst_t = torch.as_tensor(dst, device=tensors[0].device)
        for t in tensors:
            t[dst_t] = t[src_t].clone()
        for i in range(nb):
            ids[i] = int(dst[i])
        bt.block_table.np[row, :nb] = dst
        # GPU copy of the block table is refreshed by _prepare_inputs -> commit_block_table(num_reqs)


def _apply_perm_B(runner, req_state, row, perm):
    """Impl B: physical permutation of the request's logical positions 0..n-1 inside the written prefix:
    content of position p moves to the slot of position perm[p]; block tables untouched, so the model now
    sees rows in physical order (indices are physical; perm_log replays them). One gather per layer tensor.
    Layouts (per 64-token block): MLA latent cache = 64 contiguous rows (576 B bf16 or 656 B fp8_ds_mla);
    indexer k-cache = [64 x head_dim fp8 values][64 x 4-byte fp32 scales] (csrc indexer_k_quant_and_cache),
    so its values and scales are permuted as two separate row sets."""
    n = len(perm)
    pos = np.arange(n, dtype=np.int64)
    for g, tensors, bt in _st["groups"]:
        ids = np.asarray(req_state.block_ids[g], dtype=np.int64)
        b_src, o_src = ids[pos // BLOCK], pos % BLOCK
        b_dst, o_dst = ids[perm // BLOCK], perm % BLOCK
        dev = tensors[0].device
        bs, os_, bd, od = (torch.as_tensor(x, device=dev) for x in (b_src, o_src, b_dst, o_dst))
        for t in tensors:
            W = t.shape[-1]
            if W == _st.get("indexer_row_bytes"):
                hd = W - 4                                   # head_dim (128) + 4-byte scale
                blk = t.view(t.shape[0], BLOCK * W)
                vals = blk[:, : BLOCK * hd].view(t.shape[0], BLOCK, hd)
                scl = blk[:, BLOCK * hd:].view(t.shape[0], BLOCK, 4)
                vals[bd, od] = vals[bs, os_].clone()
                scl[bd, od] = scl[bs, os_].clone()
            else:
                t[bd, od] = t[bs, os_].clone()


def _plan_step(req_id, seq_len, step_idx):
    """Return (kind, perm) for this request at this decode step, or None. perm is a permutation over the
    eligible units (blocks for impl A, positions for impl B): unit i's content moves to unit perm[i]."""
    limit = seq_len - BLOCK  # positions strictly below this are eligible (never the block being appended to)
    if limit <= 1:
        return None
    MODE, IMPL, PERIOD, FRAC = CFG["mode"], CFG["impl"], CFG["period"], CFG["frac"]
    if MODE == "off":
        return None
    r = _st["req"].setdefault(req_id, {"rng": _rng(req_id), "done_once": False, "step": 0})
    r["step"] = step_idx
    n_units = (limit // BLOCK) if IMPL == "A" else limit
    if n_units < 2:
        return None
    if MODE == "ctrl_identity":
        return ("identity", np.arange(n_units, dtype=np.int64))
    if MODE in ("perm_once", "perm_periodic") and not r["done_once"]:
        r["done_once"] = True
        return ("perm_once", r["rng"].permutation(n_units).astype(np.int64))
    if MODE == "perm_periodic" and step_idx > 0 and step_idx % PERIOD == 0:
        k = max(1, int(n_units * FRAC / 2))          # k disjoint pairs = FRAC of the units
        idx = r["rng"].choice(n_units, size=min(2 * k, n_units - (n_units % 2)), replace=False)
        perm = np.arange(n_units, dtype=np.int64)
        a_, b_ = idx[0::2], idx[1::2]
        perm[a_], perm[b_] = b_, a_
        return ("periodic", perm)
    return None


def _install():
    if _st["installed"] or not INSTALL:
        return
    from vllm.v1.worker import gpu_model_runner as gmr
    orig_exec = gmr.GPUModelRunner.execute_model

    def execute_model(self, scheduler_output, *a, **kw):
        if _st["rank"] is None:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            _st["rank"] = get_tensor_model_parallel_rank()
            _st["runner"] = self
        try:
            if _st["groups"] is None:
                _st["groups"] = _groups(self)
            torch.cuda.synchronize()
            nst = scheduler_output.num_scheduled_tokens
            # fresh per-request computed-token counts for this step (the runner's own copy lags one step)
            cr = scheduler_output.scheduled_cached_reqs
            fresh = dict(zip(cr.req_ids, cr.num_computed_tokens))
            prefix_len = CFG.get("prefix_len")
            for rid, n in nst.items():
                st = self.requests.get(rid)
                if st is None or rid not in fresh:
                    continue
                seq_len = int(fresh[rid])            # tokens already in the cache for this request
                if n == 1:
                    step_idx = len(st.output_token_ids)   # between decode steps
                elif prefix_len and seq_len >= prefix_len and n > 1:
                    step_idx = 0                          # chunk boundary: prefix written, continuation next
                    seq_len = min(seq_len, prefix_len)    # permute the prefix only (teacher-forced PPL block)
                    r0 = _st["req"].get(rid)
                    if r0 is not None and r0.get("done_once"):
                        continue
                else:
                    continue
                plan = _plan_step(rid, seq_len, step_idx)
                if not plan:
                    continue
                kind, perm = plan
                IMPL = CFG["impl"]
                row = self.input_batch.req_id_to_index.get(rid)
                if row is None:
                    continue
                n_units = len(perm)
                # assertions: same request only (by construction); nothing at/after seq_len - 64
                if IMPL == "A":
                    assert n_units * BLOCK <= seq_len - BLOCK, "block beyond prefix"
                    assert n_units <= len(st.block_ids[0]), "more blocks than allocated"
                else:
                    assert n_units <= seq_len - BLOCK, "slot beyond prefix"
                moved = int((perm != np.arange(n_units)).sum())
                if moved:
                    if IMPL == "A":
                        _apply_perm_A(self, st, row, perm)
                    else:
                        _apply_perm_B(self, st, row, perm)
                    torch.cuda.synchronize()
                _st["n_swaps"] += moved // 2 if kind == "periodic" else moved
                _st["n_events"] += 1
                if _st["rank"] == 0:
                    rec = {"event": kind, "impl": IMPL, "mode": CFG["mode"], "step": step_idx, "seq_len": seq_len,
                           "n_units": n_units, "moved": moved, "at_prefill_boundary": n > 1}
                    if kind == "periodic":
                        ch = np.flatnonzero(perm != np.arange(n_units))
                        rec["pairs"] = [(int(i), int(perm[i])) for i in ch if i < perm[i]]
                    elif kind == "perm_once":
                        rec["perm"] = perm.tolist()
                    _log(rid, rec)
        except Exception as e:
            _st["errors"].append(repr(e))
            if len(_st["errors"]) < 3 and _st["rank"] == 0:
                import traceback
                _log("_errors", {"error": repr(e), "tb": traceback.format_exc()[-2000:]})
        return orig_exec(self, scheduler_output, *a, **kw)

    gmr.GPUModelRunner.execute_model = execute_model
    _st["installed"] = True


_install()


class ReindexExt(selhook_trace.SelHookExt):
    def reindex_set(self, mode=None, impl=None, seed=None, period=None, frac=None, log_dir=None, prefix_len=None, reset=True):
        """Switch mode/impl at runtime (all TP ranks receive the same RPC); resets per-request state.
        prefix_len: when set, a request whose scheduled chunk starts at >= prefix_len tokens gets its prefix
        permuted once before that chunk (teacher-forced PPL block); 0/None disables the chunk-boundary trigger."""
        for k, v in (("mode", mode), ("impl", impl), ("seed", seed), ("period", period), ("frac", frac), ("log_dir", log_dir), ("prefix_len", prefix_len)):
            if v is not None:
                CFG[k] = v
        if reset:
            _st["req"] = {}
            for f in _st["log_files"].values():
                f.close()
            _st["log_files"] = {}
        return dict(CFG)

    def reindex_info(self):
        MODE, IMPL, SEED, PERIOD, FRAC = CFG["mode"], CFG["impl"], CFG["seed"], CFG["period"], CFG["frac"]
        return {"mode": MODE, "impl": IMPL, "seed": SEED, "period": PERIOD, "frac": FRAC, "rank": _st["rank"],
                "n_events": _st["n_events"], "n_swaps": _st["n_swaps"], "errors": _st["errors"][:5],
                "groups": [(g, len(t)) for g, t, _ in (_st["groups"] or [])]}

    def reindex_groups_info(self):
        runner = _st["runner"] or getattr(self, "model_runner", None)
        fc = runner.vllm_config.compilation_config.static_forward_context
        out = []
        for g, grp in enumerate(runner.kv_cache_config.kv_cache_groups):
            names = list(grp.layer_names)
            info = {"group": g, "n_layers": len(names), "spec": type(grp.kv_cache_spec).__name__,
                    "page_size_bytes": getattr(grp.kv_cache_spec, "page_size_bytes", None), "names_head": names[:2], "layers": []}
            for name in names[:2]:
                layer = fc.get(name)
                t = getattr(layer, "kv_cache", None)
                info["layers"].append({"name": name, "layer_type": type(layer).__name__ if layer is not None else None,
                                       "kv_type": type(t).__name__, "shape": (list(t.shape) if hasattr(t, "shape") else (len(t) if t is not None else None)),
                                       "numel": (int(t.numel()) if hasattr(t, "numel") else None)})
            out.append(info)
        out.append({"fc_keys_head": list(fc.keys())[:4], "n_fc": len(fc), "runner_kv_caches": len(getattr(runner, "kv_caches", []))})
        return out

    def reindex_readback_test(self):
        """Unit test (exp3 §7): swap rows 0 and 1 of the first tensor of each group (impl B style) and read back."""
        if _st["runner"] is None:
            _st["runner"] = getattr(self, "model_runner", None)
        if _st["groups"] is None:
            _st["groups"] = _groups(_st["runner"])
        res = {}
        for g, tensors, _ in _st["groups"]:
            t = tensors[0]
            a0, a1 = t[1, 0].clone(), t[1, 1].clone()
            tmp = t[1, 0].clone(); t[1, 0].copy_(t[1, 1]); t[1, 1].copy_(tmp)
            ok = bool(torch.equal(t[1, 0], a1) and torch.equal(t[1, 1], a0))
            tmp = t[1, 0].clone(); t[1, 0].copy_(t[1, 1]); t[1, 1].copy_(tmp)  # restore
            res[str(g)] = {"shape": list(t.shape), "dtype": str(t.dtype), "row_bytes": int(t.shape[-1]) * t.element_size(), "swap_readback_ok": ok}
        return res
