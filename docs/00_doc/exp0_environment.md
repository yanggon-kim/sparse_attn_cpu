# exp0 — Environment, models, storage, hook unit checks

*Gate to pass before any measurement: a 3-token TP8 generation of DeepSeek-V3.2 with the hook installed
and the hook's unit checks green. Budget: <= 5 GPU-h (most of it model download and weight loading).*

## 1. Inventory the node (record everything in `<WORKDIR>/node.json`)

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv   # expect 8x H200 141 GB (or B200)
nvcc --version; python3 -c "import torch;print(torch.__version__, torch.version.cuda)"
df -h /  /nvme* 2>/dev/null; free -h; nproc
```
- Decide the trace root now: a local NVMe with >= 1.5 TB free. Export `WORKDIR=<that path>/sparse_attn_gpu`
  and create `$WORKDIR/{models,runs,prompts,logs}`. `/` must never be the trace target.

## 2. vLLM pin

The experiment files cite `vllm-project/vllm` @ **`5559679`** (2026-07-26); the repo's older guide cites
`2b753ad20`. Either install a release that contains both `deepseek_v2.py::GlmMoeDsaForCausalLM` and the
`fp8_ds_mla` sparse FlashMLA path, or build from source at the pinned commit:

```bash
git clone https://github.com/vllm-project/vllm && cd vllm && git checkout 5559679
pip install -e . -v          # or `uv pip install vllm==<release>` if its git sha contains the symbols below
python3 - <<'PY'
import vllm, subprocess
from vllm.model_executor.models.registry import ModelRegistry
print(vllm.__version__)
print("GlmMoeDsaForCausalLM" in ModelRegistry.get_supported_archs())
PY
grep -n "def forward_mqa" vllm/v1/attention/backends/mla/flashmla_sparse.py           # expect ~838
grep -n "^def sparse_attn_indexer" vllm/model_executor/layers/sparse_attn_indexer.py   # expect ~296
grep -n "def triton_convert_req_index_to_global_index" vllm/v1/attention/backends/mla/sparse_utils.py  # ~120
grep -n "index_topk_freq" vllm/model_executor/models/deepseek_v2.py                    # ~1092
```
Record the resolved commit in every `run_manifest.json` (`vllm_commit`). If any grep lands on a different
line, update the citation in the exp file you are following (the line numbers are hints, the symbols are
the contract). Also install: `pip install pandas pyarrow matplotlib nltk datasets`.

## 3. Models

| model | HF id | dtype | size | note |
|---|---|---|---|---|
| DeepSeek-V3.2 | `deepseek-ai/DeepSeek-V3.2` | native FP8 | ~690 GB | main target; `index_topk` 2048, `index_n_heads` 64, `index_head_dim` 128, 61 layers |
| GLM-5.2 | `zai-org/GLM-5.2` (verify exact id) | FP8 | ~750 GB | DSA with `index_topk_freq` (expected 4) |
| GLM-5 | `zai-org/GLM-5` (verify) | FP8 | ~750 GB | **only if** `config.json` has `index_topk` |

```bash
hf download deepseek-ai/DeepSeek-V3.2 --local-dir $WORKDIR/models/DeepSeek-V3.2      # HF_TOKEN if gated
python3 -c "import json;c=json.load(open('$WORKDIR/models/DeepSeek-V3.2/config.json'));print({k:c.get(k) for k in ['num_hidden_layers','index_topk','index_n_heads','index_head_dim','index_topk_freq','index_skip_topk_offset','index_topk_pattern','n_routed_experts','num_experts_per_tok']})"
```
Do the same `config.json` print for each GLM checkpoint and paste the dicts into `node.json`. The GLM
skip rule is `max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0` → skipped layers
(`deepseek_v2.py:1097-1100`); list the layers that compute their own top-k.

## 4. Smoke: 3-token generation, TP8, eager

```python
from vllm import LLM, SamplingParams
llm = LLM(model=f"{WORKDIR}/models/DeepSeek-V3.2", tensor_parallel_size=8, enforce_eager=True,
          trust_remote_code=True, max_model_len=16384, enable_prefix_caching=False, seed=42)
out = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=3))
print(out[0].outputs[0].text, out[0].outputs[0].token_ids)
```
Record: load time, per-GPU memory after load (`nvidia-smi`), the 3 token ids. Run it twice → token ids
must be identical (determinism baseline for exp3). `enable_prefix_caching=False` is mandatory for exp3
(block swaps break prefix-hash reuse) and harmless elsewhere — keep it off for the whole campaign.

## 5. Hook unit checks (the collector of exp1 §2, installed but with one prompt)

Run the same 3-token prompt with `SEL_TRACE=$WORKDIR/runs/smoke0` set:
- [ ] `indexer_trace.jsonl` has exactly `61 x 3` decode records (`phase=1`) for one request, plus prefill
      records only if prefill tracing was on (default off).
- [ ] every `sel[]` value lies in `[0, pos]`, `len(sel) <= 2048`, no `-1` survives, `n_comp == pos + 1`.
- [ ] `layer` values are `0..60`, each exactly once per step.
- [ ] tokens with hook on == tokens with hook off (write `IDENTICAL` to `$WORKDIR/runs/smoke_identity.txt`).
- [ ] MoE hook (if enabled): 61 - 3 = 58 routed-MoE layers x 3 steps, 8 expert ids in `[0, 256)` each.
- [ ] `scripts/ingest_trace.py $WORKDIR/runs/smoke0 && scripts/validate_trace.py $WORKDIR/runs/smoke0`
      exit 0 after the adapter (exp1 §4) has written the ds4-schema files.

## 6. Storage layout

```
$WORKDIR/runs/<run_id>/{run_manifest.json, model_config.json, prompts/, outputs/generations.jsonl,
                        traces/indexer_trace.jsonl(.gz), traces/moe_trace.jsonl(.gz), traces/*.parquet,
                        logs/time_and_stderr.log, analysis/}
```
`run_id` convention: `<model>_<bench>_<task>_<ctx>_<kind>_s<sample>` e.g. `v32_ruler_niah2_32768_bf_s03`
(`bf` = benchmark-faithful, `ld` = long-decode). Compress traces after ingest (`gzip`), keep the parquet.

## 7. Gate

All of §4 and §5 green, `node.json` written, `HANDOFF.md` "Status" gets a dated line
"exp0 passed on <node>, vLLM <commit>, DSv3.2 loaded in <min> min, <GB>/GPU". Then go to
`exp1_dsv32_gather_index.md` §5 (smoke run).
