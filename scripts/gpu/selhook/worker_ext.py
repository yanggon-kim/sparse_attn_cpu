"""Selection-trace hook (KV top-k + MoE routed experts) for vLLM 0.28.0, installed in every TP worker.

Mechanism (verified against vllm-src @ v0.28.0 on 8xB200, FLASHINFER_MLA_SPARSE + FLASHINFER_TRTLLM fp8 MoE):
- worker_extension_cls import side effect runs in every worker before init_device/load_model
  (vllm/v1/worker/worker_base.py:265-291).
- KV hook point: MLAAttention.forward_impl (layers/attention/mla_attention.py:722), entered after the layer's
  indexer filled impl.topk_indices_buffer ([tokens, 2048] request-local positions, -1 = empty) for ALL rows,
  prefill or decode (forward_mqa alone misses long prefill chunks, which use the dense-MHA path).
- Token rows are laid out per request as [query_start_loc[i], query_start_loc[i+1]); the LAST row of a request
  is the token whose next-token prediction this step produces, and its selection is the gather at
  pos = seq_lens[i]-1. NOTE: the sparse builder counts short prefills (query_len <= reorder threshold) as
  "decodes", so num_decode_tokens != num_decodes in general; the row plan below does not rely on it.
- Rows recorded per step: the last row of every request; phase=1 if the request is decoding (query_len==1)
  or its prefill completes this step (seq_len == num_prompt_tokens); other rows (mid-prefill chunk) get
  phase=0 and are dropped unless SEL_TRACE_PREFILL=1.
- Attribution: attn_metadata.req_id_per_token[row] is the input_batch row; input_batch.req_ids is read AFTER
  the step (the runner may reorder the batch inside _prepare_inputs).
- MoE: per MoERunner, a capture fn (monolithic kernel: fused_experts.set_capture_fn; else router.set_capture_fn)
  receives topk_ids [tokens, k] (logical ids, EPLB off) in the same token order; the same rows are kept.
- Emitted only from TP rank 0; a background thread appends to <SEL_TRACE>/by_req/<request_id>.jsonl:
    {"req","layer","pos","n_comp","top_k","valid_k","sel":[sorted],"phase",
     "topk_computed","shared_from_layer"}                              (KV, one per layer per step)
    {"req","moe":1,"layer","pos","sel":[k expert ids],"phase"}          (MoE, one per MoE layer per step)
Env: SEL_TRACE=<dir> enables; SEL_TRACE_PREFILL=1 keeps mid-prefill rows; SEL_TRACE_MOE=0 disables MoE capture.
"""
import atexit
import json
import os
import queue
import threading

import torch

TRACE_DIR = os.environ.get("SEL_TRACE")
TRACE_PREFILL = os.environ.get("SEL_TRACE_PREFILL", "0") == "1"
TRACE_MOE = os.environ.get("SEL_TRACE_MOE", "1") == "1"

_state = {
    "installed": False,
    "is_rank0": None,
    "runner": None,
    "computed": None,        # layer_id -> bool (this layer computes its own top-k)
    "row_plan": None,        # per step: (rows_gpu LongTensor, [(batch_row, pos, qlen), ...])
    "kv_recs": [],           # per step: (layer_name, topk_rows_gpu)
    "moe_recs": [],          # per step: (layer_id, topk_ids_rows_gpu)
    "n_written": 0,
    "n_steps": 0,
    "layer_ids": {},
    "moe_bound": None,
    "errors": [],
}
_q = queue.Queue(maxsize=8192)
_files = {}
_writer = None


def _safe(req):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(req))


def _writer_loop():
    while True:
        item = _q.get()
        if item is None:
            break
        req, lines = item
        f = _files.get(req)
        if f is None:
            os.makedirs(os.path.join(TRACE_DIR, "by_req"), exist_ok=True)
            f = open(os.path.join(TRACE_DIR, "by_req", _safe(req) + ".jsonl"), "a")
            _files[req] = f
        f.write("".join(lines))
        _state["n_written"] += len(lines)


def _flush_all():
    for f in _files.values():
        f.flush()


def _shutdown():
    if _writer is not None:
        _q.put(None)
        _writer.join(timeout=120)
    for f in _files.values():
        f.close()


def _layer_index(name):
    lid = _state["layer_ids"].get(name)
    if lid is None:
        from vllm.model_executor.models.utils import extract_layer_index
        lid = extract_layer_index(name)
        _state["layer_ids"][name] = lid
    return lid


def _build_computed_map(model):
    computed = {}
    for name, mod in model.named_modules():
        if name.endswith("self_attn") and hasattr(mod, "indexer"):
            try:
                lid = _layer_index(name)
            except Exception:
                continue
            ok = getattr(mod, "indexer", None) is not None
            mla = getattr(mod, "mla_attn", None)
            if mla is not None and getattr(mla, "skip_topk", False):
                ok = False
            computed[lid] = bool(ok)
    return computed


def _bind_moe(model):
    """Install per-layer capture fns on every MoERunner (mirrors bind_routed_experts_capturer without its
    KV-cache-group precheck). Returns {model_layer_id: how}."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    bound = {}
    for name, module in model.named_modules():
        if not isinstance(module, MoERunner):
            continue
        try:
            layer_id = _layer_index(name)          # model layer index from the module path
        except Exception:
            layer_id = int(module.layer_id)

        def capture_fn(topk_ids, layer_id=layer_id):
            plan = _state["row_plan"]
            if plan is None or not _state["is_rank0"]:
                return
            rows = plan[0]
            if int(rows.max()) < topk_ids.shape[0]:
                _state["moe_recs"].append((layer_id, topk_ids[rows].clone()))

        qm = module._quant_method
        fe = getattr(getattr(getattr(qm, "moe_kernel", None), "impl", None), "fused_experts", None)
        if getattr(qm, "is_monolithic", False) and fe is not None and hasattr(fe, "set_capture_fn"):
            fe.set_capture_fn(capture_fn)
            bound[layer_id] = "monolithic:" + type(fe).__name__
        elif hasattr(module, "router") and hasattr(module.router, "set_capture_fn"):
            module.router.set_capture_fn(capture_fn)
            bound[layer_id] = "router:" + type(module.router).__name__
        else:
            bound[layer_id] = "UNBOUND"
    return bound


def _make_row_plan(attn_metadata):
    """Rows to record this step: the last token row of every request."""
    nr = int(attn_metadata.num_reqs)
    qsl = attn_metadata.query_start_loc[: nr + 1].cpu().tolist()
    sl = attn_metadata.seq_lens[:nr].cpu().tolist()
    rows, meta = [], []
    for i in range(nr):
        qlen = qsl[i + 1] - qsl[i]
        if qlen <= 0:
            continue
        rows.append(qsl[i + 1] - 1)
        meta.append((i, int(sl[i]) - 1, int(qlen)))
        if TRACE_PREFILL and qlen > 1:
            for j in range(qsl[i], qsl[i + 1] - 1):
                rows.append(j)
                meta.append((i, int(sl[i]) - (qsl[i + 1] - 1 - j) - 1, -1))  # -1 = mid-prefill row
    rows_t = torch.tensor(rows, dtype=torch.long, device=attn_metadata.query_start_loc.device)
    return (rows_t, meta)


def _drain_step():
    kv, moe, plan = _state["kv_recs"], _state["moe_recs"], _state["row_plan"]
    _state["kv_recs"], _state["moe_recs"], _state["row_plan"] = [], [], None
    if plan is None or (not kv and not moe):
        return
    torch.cuda.synchronize()
    runner = _state["runner"]
    req_ids = list(runner.input_batch.req_ids)
    meta = plan[1]
    computed = _state["computed"] or {}
    rowinfo = []
    for (brow, pos, qlen) in meta:
        rid = req_ids[brow] if brow < len(req_ids) else f"row{brow}"
        if qlen == 1:
            phase = 1
        elif qlen > 1:
            st = runner.requests.get(rid)
            npt = int(getattr(st, "num_prompt_tokens", -1)) if st is not None else -1
            phase = 1 if pos + 1 == npt else 0
        else:
            phase = 0
        rowinfo.append((rid, pos, phase))
    keep = [k for k, (_, _, ph) in enumerate(rowinfo) if ph == 1 or TRACE_PREFILL]
    if not keep:
        _state["n_steps"] += 1
        return
    order = sorted({_layer_index(n) for (n, _) in kv})
    last_comp, producer = None, {}
    for lid in order:
        if computed.get(lid, True):
            last_comp = lid
        producer[lid] = last_comp if last_comp is not None else lid
    per_req = {}
    for (name, buf) in kv:
        lid = _layer_index(name)
        arr = buf.cpu().numpy()
        for k in keep:
            rid, pos, phase = rowinfo[k]
            sel = arr[k]
            sel = sel[sel >= 0]
            sel.sort()
            rec = {"req": rid, "layer": lid, "pos": pos, "n_comp": pos + 1, "top_k": int(arr.shape[1]),
                   "valid_k": int(len(sel)), "sel": sel.tolist(), "phase": phase,
                   "topk_computed": bool(computed.get(lid, True)), "shared_from_layer": int(producer.get(lid, lid))}
            per_req.setdefault(rid, []).append(json.dumps(rec, separators=(",", ":")) + "\n")
    for (lid, ids) in moe:
        arr = ids.cpu().numpy()
        for k in keep:
            rid, pos, phase = rowinfo[k]
            rec = {"req": rid, "moe": 1, "layer": int(lid), "pos": pos, "sel": [int(x) for x in arr[k]], "phase": phase}
            per_req.setdefault(rid, []).append(json.dumps(rec, separators=(",", ":")) + "\n")
    for rid, lines in per_req.items():
        _q.put((rid, lines))
    _state["n_steps"] += 1


def _install():
    global _writer
    if _state["installed"] or not TRACE_DIR:
        return
    import importlib
    from vllm.v1.worker import gpu_model_runner as gmr

    # Hook point: MLAAttention.forward_impl (vllm/model_executor/layers/attention/mla_attention.py:722). It runs
    # for every sparse layer on every step AFTER the layer's indexer filled self.impl.topk_indices_buffer, and
    # regardless of whether the tokens then go through the dense-MHA prefill path or forward_mqa (long prefill
    # chunks never reach forward_mqa on the FlashInfer sparse backend, which lost decode step 0 in the first smoke).
    from vllm.model_executor.layers.attention import mla_attention as mla_mod
    orig_exec = gmr.GPUModelRunner.execute_model
    orig_fi = mla_mod.MLAAttention.forward_impl
    impl_classes = [mla_mod.MLAAttention]

    def forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output, *a, **kw):
        if _state["is_rank0"] and _state["runner"] is not None and attn_metadata is not None:
            try:
                buf = getattr(self.impl, "topk_indices_buffer", None)
                if buf is not None and getattr(self.impl, "is_sparse", False):
                    if _state["row_plan"] is None:
                        _state["row_plan"] = _make_row_plan(attn_metadata)
                    rows = _state["row_plan"][0]
                    _state["kv_recs"].append((self.layer_name, buf[rows].clone()))
            except Exception as e:
                _state["errors"].append(f"forward_impl: {e!r}")
        return orig_fi(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output, *a, **kw)

    def execute_model(self, scheduler_output, *a, **kw):
        if _state["is_rank0"] is None:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            _state["is_rank0"] = get_tensor_model_parallel_rank() == 0
            if _state["is_rank0"]:
                _state["computed"] = _build_computed_map(self.model)
                if TRACE_MOE:
                    try:
                        _state["moe_bound"] = _bind_moe(self.model)
                    except Exception as e:
                        _state["errors"].append(f"bind_moe: {e!r}")
                os.makedirs(TRACE_DIR, exist_ok=True)
                json.dump({"topk_computed": _state["computed"], "impl_classes": [c.__name__ for c in impl_classes],
                           "moe_bound": _state["moe_bound"], "errors": _state["errors"]},
                          open(os.path.join(TRACE_DIR, "layer_computed.json"), "w"), indent=1)
        if _state["is_rank0"]:
            _state["runner"] = self
            _state["row_plan"], _state["kv_recs"], _state["moe_recs"] = None, [], []
        out = orig_exec(self, scheduler_output, *a, **kw)
        if _state["is_rank0"]:
            try:
                _drain_step()
            except Exception as e:
                _state["errors"].append(f"drain: {e!r}")
                _state["row_plan"], _state["kv_recs"], _state["moe_recs"] = None, [], []
        return out

    mla_mod.MLAAttention.forward_impl = forward_impl
    gmr.GPUModelRunner.execute_model = execute_model
    _writer = threading.Thread(target=_writer_loop, daemon=True)
    _writer.start()
    atexit.register(_shutdown)
    _state["installed"] = True


_install()


class SelHookExt:
    """Mixed into the worker class; methods callable via llm.collective_rpc("selhook_flush")."""

    def selhook_flush(self):
        import time
        t0 = time.time()
        while not _q.empty() and time.time() - t0 < 900:
            time.sleep(0.05)
        _flush_all()
        return {"n_written": _state["n_written"], "n_steps": _state["n_steps"], "is_rank0": _state["is_rank0"],
                "errors": _state["errors"][:20], "trace_dir": TRACE_DIR}

    def selhook_info(self):
        return {"installed": _state["installed"], "is_rank0": _state["is_rank0"], "computed": _state["computed"],
                "moe_bound": _state["moe_bound"], "errors": _state["errors"][:20], "trace_dir": TRACE_DIR}
