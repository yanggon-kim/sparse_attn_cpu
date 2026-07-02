# Collecting DeepSeek-V4 Selection Histories in vLLM

**Audience:** an agentic AI (or engineer) that will *run DeepSeek-V4 in vLLM on a GPU* and record two
per-token, per-layer **selection histories**:

- **(A) CSA lightning-indexer top-k KV indices** — for each generated token and each CSA (ratio-4) layer,
  which compressed-KV blocks the sparse attention kept.
- **(B) MoE routed-expert ids** — for each token and each MoE layer, which experts it was routed to.

These are the two "who-did-I-select" traces. The purpose is temporal-locality / routing-stability analysis
(how much the selected set overlaps between adjacent decode steps), the GPU analogue of the CPU `ds4` study
in this repo.

> **Provenance & status.** All file/line anchors are for `vllm-project/vllm` @ commit **`2b753ad20`** (line
> numbers may drift — a few lines of context are given). This guide was written by reading the source; it
> was **not executed here** (this box is CPU-only; DeepSeek-V4 in vLLM needs a GPU + the checkpoint). Run
> the **Validation checklist** (§8) first on a 2–3 token generation before trusting a long run.

---

## 1. Prerequisites & how to run

- A GPU that can hold the DeepSeek-V4 checkpoint (e.g. `deepseek-ai/DeepSeek-V4-Flash`), CUDA, and a
  vLLM build with the DeepSeek-V4 kernels.
- **Run these two selections require Python-level hooks to fire every step**, so:
  - **`enforce_eager=True`** — the indexer op is `@eager_break_during_capture`
    (`vllm/model_executor/layers/sparse_attn_indexer.py:294`); eager mode guarantees the Python wrappers run
    on every decode step instead of being captured into a CUDA graph.
  - **`tensor_parallel_size=1` and no expert parallelism (EP)** — so expert ids are **global logical** ids
    (no EPLB physical remap; see §7). Indexer indices are also cleanest single-GPU (no KV sharding).
- Greedy decoding (`temperature=0`) for reproducibility.

```python
# run_harness.py  — drives a few decode steps with the collector installed
import os
os.environ["VLLM_SEL_TRACE"] = "/path/to/out_dir"     # collector writes here (see §6)
import vllm_selection_collector as C                    # the module in §6
C.install()                                             # monkeypatch BEFORE constructing LLM

from vllm import LLM, SamplingParams
llm = LLM(model="deepseek-ai/DeepSeek-V4-Flash",
          enforce_eager=True, tensor_parallel_size=1, trust_remote_code=True,
          max_model_len=8192)                            # size to your GPU
out = llm.generate(["<your long prompt here>"],
                   SamplingParams(temperature=0.0, max_tokens=128))
print(out[0].outputs[0].text)
C.finalize()                                            # flush + close trace files
```

---

## 2. Where each selection is produced (map)

| Selection | Producer file | Tensor | Shape | Hook target |
|---|---|---|---|---|
| (A) indexer top-k KV | `vllm/model_executor/layers/sparse_attn_indexer.py` | `topk_indices_buffer` | `(num_tokens, topk_tokens)` int32, `-1`-padded | `SparseAttnIndexer.forward_cuda` (class @685, method @751) |
| (B) MoE experts | `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` | `topk_ids` | `(num_tokens, num_experts_per_tok [+shared])` int32/64, `-1`=padding | `fused_topk_bias` (MegaMoE) / `BaseRouter._select_experts` (FusedMoE) |

Only **even/CSA (compress_ratio==4)** layers have an indexer (`attention.py:249` builds it only then), so (A)
is emitted for ~half the layers; every layer's FFN emits (B).

---

## 3. Part A — Indexer top-k KV indices

**Producer.** `sparse_attn_indexer.py`: the module-level op writes the per-token top-k into
`topk_indices_buffer` after the top-k kernels and returns it:

```python
# vllm/model_executor/layers/sparse_attn_indexer.py  (decode path, ~573-647)
topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
# ... cooperative_topk / persistent_topk / top_k_per_row_decode write topk_indices ...
return topk_indices_buffer          # shape (num_tokens, topk_tokens), int32, -1 for empty slots
```

`SparseAttnIndexer.forward_cuda` (`@751`) calls that op and returns the buffer; the instance knows its layer
through `self.k_cache.prefix` (`@766`).

**What to record**, per (layer, decode_step, token-row): the list of selected indices in that row with `-1`
stripped, plus `n_valid` and `n_candidates` (= `topk_tokens` cap; the true `n_comp` you can also read from
attn metadata if needed).

**Semantics / gotchas (read before analyzing):**

1. **These are COMPRESSED-KV block indices, not raw token positions.** Index `c` ≈ raw tokens
   `[c·4, c·4+3]` for CSA (ratio `m=4`). The number of compressed blocks grows by 1 every `m` tokens.
   Do **not** compare a compressed index to a raw position.
2. **Append-only / stable numbering.** A finalized compressed block keeps its index as the sequence grows,
   so a given index refers to the same content at step *n* and *n+1* — comparing sets across steps is valid.
3. **`-1` is padding** for unused top-k slots — strip it.
4. **This is the top-k over compressed blocks only.** The sliding-window (recent uncompressed) tokens are a
   *separate, moving* set combined later in the kernel (`combine_topk_swa_indices`); do **not** mix them into
   the locality of the *selection*.
5. **Log logical, not physical.** `topk_indices_buffer` holds logical indices; the block-table remap to
   physical cache slots happens downstream (`nvidia/flashmla.py compute_global_topk_indices_and_lens`). Log
   the buffer (logical) — physical slots are not stable across steps.
6. **int32, max ~250K at 1M context** — no overflow.

---

## 4. Part B — MoE routed-expert ids

**Producer.** `fused_topk_bias(...)` returns `(topk_weights, topk_ids)`:

```python
# vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py  (~159-194)
topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)  # [num_tokens, topk]
# ... sqrt(softplus) scores, +bias for selection, top-k, renorm ...
# hash layers: expert_ids = hash_indices_table[input_tokens]; topk_ids.copy_(expert_ids)
return topk_weights, topk_ids
```

DeepSeek-V4 has **two MoE backends**; cover both:

- **MegaMoE** (`kernel_config.moe_backend == "deep_gemm_mega_moe"`): `DeepseekV4MoE.forward` calls
  `fused_topk_bias(...)` directly (`nvidia/model.py:703`) → `topk_ids` visible there.
- **FusedMoE** (default otherwise): the ids are produced inside `BaseRouter._select_experts`
  (`base_router.py:260`), which even has a built-in capture hook:
  ```python
  # base_router.py:296
  if self.capture_fn is not None:
      self.capture_fn(topk_ids)        # logical ids, BEFORE EPLB remap
  ```

**What to record**, per (layer, decode_step, token-row): the expert ids in that row (and weights if wanted).

**Semantics / gotchas:**

1. **Global logical expert ids** (range `[0, global_num_experts)`) — provided you run **single-GPU, no EP**.
   With EP/EPLB the ids get remapped to physical experts (`_apply_eplb_mapping`); capture **before** that
   (the `capture_fn` / `_select_experts` return, or `fused_topk_bias` output) to keep logical ids.
2. **Shared experts are appended** as extra columns with fixed ids `[global_num_experts, +num_shared)`
   (`fused_topk_bias_router.py:392-411`) → row width = `num_experts_per_tok + num_shared`. Split by id range
   if you want routed-only.
3. **Hash-routing layers** (`layer_idx < num_hash_layers`): `topk_ids` is filled from the `tid2eid[token_id]`
   table (deterministic, not learned) — still the same tensor, so it's captured the same way; you may want to
   flag/exclude these layers in analysis.
4. **`-1` padding** appears on the MegaMoE path for padded tokens (`nvidia/model.py:465`) — strip it.
5. `renormalize`, `scoring_func` only affect **weights**, not ids. `num_experts_per_tok` = the row width.

---

## 5. Attribution — layer, step, token

- **Layer id:** `from vllm.model_executor.models.utils import extract_layer_index` (`utils.py:804`,
  `extract_layer_index(name)` pulls the int from `"model.layers.N.…"`). For the indexer use
  `self.k_cache.prefix`; for MoE use `self.prefix` on `DeepseekV4MoE` (`nvidia/model.py:522`).
- **Decode step:** simplest is a **global counter** bumped once per model forward (patch
  `DeepseekV4Model.forward`; each decode forward = 1 new token, or `next_n` with MTP). During single-sequence
  greedy decode, `absolute_position ≈ prompt_len + decode_step` (as in ds4).
- **Token row:** each tensor's row index is the token within the batch/step. For a single prompt and
  `tensor_parallel_size=1`, decode has one row per step (or `next_n` rows under speculative/MTP — see §8).
- Richer metadata (per-request `seq_lens`, `num_decodes`, `slot_mapping`) is available via
  `from vllm.forward_context import get_forward_context; get_forward_context().attn_metadata` if you need
  exact positions per request in a batch.

---

## 6. The collector (drop-in monkeypatch module)

Save as `vllm_selection_collector.py`, `import` it and call `install()` **before** building `LLM`. It is
env-gated (`VLLM_SEL_TRACE=<dir>`), buffered, and writes two JSONL files.

```python
# vllm_selection_collector.py
import os, json, threading, atexit

_OUT = os.environ.get("VLLM_SEL_TRACE")
_lock = threading.Lock()
_step = 0                      # global decode-step counter (bumped per model forward)
_cur_moe_layer = threading.local()
_files = {}
_orig = {}

def _extract_layer(name):
    try:
        from vllm.model_executor.models.utils import extract_layer_index
        return int(extract_layer_index(name))
    except Exception:
        # fallback: first integer in the dotted name
        for p in str(name).split("."):
            if p.isdigit():
                return int(p)
        return -1

def _fh(kind):
    f = _files.get(kind)
    if f is None:
        os.makedirs(_OUT, exist_ok=True)
        f = open(os.path.join(_OUT, f"{kind}.jsonl"), "w")
        _files[kind] = f
    return f

def _emit(kind, rec):
    with _lock:
        f = _fh(kind); f.write(json.dumps(rec) + "\n")

def _rows_no_pad(t):
    # t: 2D int tensor (num_tokens, k) -> list[list[int]] with -1 stripped
    out = []
    for row in t.detach().to("cpu").tolist():
        out.append([int(x) for x in row if x != -1])
    return out

# ---------- (A) indexer top-k KV ----------
def _wrap_indexer(orig_forward_cuda):
    def wrapped(self, *a, **kw):
        buf = orig_forward_cuda(self, *a, **kw)       # returns topk_indices_buffer
        try:
            layer = _extract_layer(getattr(self.k_cache, "prefix", ""))
            sel = _rows_no_pad(buf)
            for tok, idxs in enumerate(sel):
                _emit("indexer", {"source": "indexer", "layer": layer, "decode_step": _step,
                                  "token_row": tok, "n_valid": len(idxs), "sel": idxs})
        except Exception as e:      # never break generation
            _emit("errors", {"where": "indexer", "err": repr(e)})
        return buf
    return wrapped

# ---------- (B) MoE experts ----------
def _log_moe(topk_ids):
    layer = getattr(_cur_moe_layer, "id", -1)
    for tok, ids in enumerate(_rows_no_pad(topk_ids)):
        _emit("moe", {"source": "moe", "layer": layer, "decode_step": _step,
                      "token_row": tok, "experts": ids})

def _wrap_moe_forward(orig_forward):
    def wrapped(self, *a, **kw):
        prev = getattr(_cur_moe_layer, "id", None)
        _cur_moe_layer.id = _extract_layer(getattr(self, "prefix", ""))   # stamp current MoE layer
        try:
            return orig_forward(self, *a, **kw)
        finally:
            _cur_moe_layer.id = prev
    return wrapped

def _wrap_fused_topk_bias(orig):                 # MegaMoE path
    def wrapped(*a, **kw):
        w, ids = orig(*a, **kw)
        try: _log_moe(ids)
        except Exception as e: _emit("errors", {"where": "fused_topk_bias", "err": repr(e)})
        return w, ids
    return wrapped

def _wrap_select_experts(orig):                  # FusedMoE path
    def wrapped(self, *a, **kw):
        w, ids = orig(self, *a, **kw)
        try: _log_moe(ids)
        except Exception as e: _emit("errors", {"where": "select_experts", "err": repr(e)})
        return w, ids
    return wrapped

# ---------- step counter ----------
def _wrap_model_forward(orig):
    def wrapped(self, *a, **kw):
        global _step
        _step += 1
        return orig(self, *a, **kw)
    return wrapped

def install():
    assert _OUT, "set VLLM_SEL_TRACE=<dir> before install()"
    import vllm.model_executor.layers.sparse_attn_indexer as sai
    import vllm.models.deepseek_v4.nvidia.model as dsm
    from vllm.model_executor.layers.fused_moe.router import base_router as br

    _orig["idx"] = sai.SparseAttnIndexer.forward_cuda
    sai.SparseAttnIndexer.forward_cuda = _wrap_indexer(_orig["idx"])
    # (ROCm: also wrap SparseAttnIndexer.forward_hip)

    _orig["moe_fwd"] = dsm.DeepseekV4MoE.forward
    dsm.DeepseekV4MoE.forward = _wrap_moe_forward(_orig["moe_fwd"])
    _orig["ftb"] = dsm.fused_topk_bias                    # patch at the USE site (imported name)
    dsm.fused_topk_bias = _wrap_fused_topk_bias(_orig["ftb"])
    _orig["sel"] = br.BaseRouter._select_experts
    br.BaseRouter._select_experts = _wrap_select_experts(_orig["sel"])

    _orig["mdl"] = dsm.DeepseekV4Model.forward
    dsm.DeepseekV4Model.forward = _wrap_model_forward(_orig["mdl"])

def finalize():
    with _lock:
        for f in _files.values():
            f.flush(); f.close()
        _files.clear()

atexit.register(finalize)
```

> **Note on the two MoE patches.** `dsm.fused_topk_bias` catches the **MegaMoE** backend (it imports the name
> at `nvidia/model.py:36`, so patch it *on that module*, not on the router module). `BaseRouter._select_experts`
> catches the **FusedMoE** backend. Only one fires per layer depending on `moe_backend`; both use the
> thread-local layer id stamped by the `DeepseekV4MoE.forward` wrapper. (You may instead use the built-in
> `router.set_capture_fn(...)` for the FusedMoE path.)

---

## 7. Output schema

Two JSONL files in `$VLLM_SEL_TRACE/`, one record per (source, layer, decode_step, token_row):

```jsonc
// indexer.jsonl
{"source":"indexer","layer":6,"decode_step":12,"token_row":0,"n_valid":512,"sel":[0,1,2,57,667, ...]}
// moe.jsonl
{"source":"moe","layer":7,"decode_step":12,"token_row":0,"experts":[3,88,120,201,255,256]}   // 256 = shared
```

- `layer` = decoder layer index; `decode_step` = global forward counter; `token_row` = row within the step.
- indexer `sel` = **compressed-KV block indices** (`-1` stripped). moe `experts` = **global logical expert ids**
  (routed ids `< global_num_experts`; ids `≥ global_num_experts` are shared experts).
- This mirrors the `ds4` JSONL, so the same overlap/retention/working-set analysis (`analyze_locality.py`)
  applies: adjacent-step overlap `|Uₜ∩Uₜ₋₁|/|Uₜ|`, retention at lags, working set, reuse distance — per layer,
  for **both** the indexer KV set and the MoE expert set.

---

## 8. Validation checklist (run first, on ~3 tokens)

1. `enforce_eager=True` — confirm `indexer.jsonl` gets **new records every decode step** (if it only has
   prefill and then nothing, the hook is being captured into a CUDA graph → eager is not on).
2. `tensor_parallel_size=1`, **no EP** — else `moe` ids may be physical (post-EPLB), not logical.
3. Indexer records appear only for CSA layers (even/ratio-4) — expect ~half the layers.
4. MoE records appear for **every** layer; shared-expert ids (`≥ global_num_experts`) present; hash-layer
   records (small `layer`) look deterministic per token.
5. Strip `-1`; check indexer `n_valid ≤ index_topk` and values `< n_comp`; check `experts` ids in
   `[0, global_num_experts + num_shared)`.
6. Under speculative decoding / MTP there may be **`next_n` rows per step** — either disable spec-decode for a
   clean single-row-per-step trace, or key records by `(decode_step, token_row)` and reconcile with accepted
   tokens.
7. The `errors.jsonl` file should stay empty (the wrappers swallow exceptions to never break generation —
   check it's empty).

## 9. Pitfalls recap
- Cuda graph vs eager (§8.1) is the #1 reason hooks "don't fire."
- Indexer indices are **compressed-block, logical, append-only**; MoE ids are **global logical experts**.
- Separate the sliding-window set from the indexer top-k; separate shared experts from routed.
- Keep TP=1 / no-EP for stable, logical ids in both traces.
