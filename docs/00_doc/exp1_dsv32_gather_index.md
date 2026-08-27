# exp1 — DeepSeek-V3.2 gather-index statistics (R1 KV locality, R2 MoE, R3 hot-set) on vLLM

*Goal: the V3.2 analogue of the CPU V4 study, at 8K–128K context and >= 2K decode steps, with per-request
attribution. Output: `docs/gpu_sweep/` (tables, plots, summary md). Budget ~50 GPU-h.*

## 1. What is selected and where (vLLM `5559679`; re-grep before patching)

- **Producer**: `sparse_attn_indexer()` — `vllm/model_executor/layers/sparse_attn_indexer.py:296`
  (`@eager_break_during_capture` at :295, hence `enforce_eager`). It clears `topk_indices_buffer[:n] = -1`
  (:411), writes the decode top-k at :593-:667 and returns the buffer (:667). Shape
  `[max_num_batched_tokens, index_topk]` int32, **request-local positions**, `-1` = empty. Decode
  `seq_lens` come from `decode_metadata.seq_lens` (:557; 2-D `(B, next_n)` under spec-decode, :558).
  Scores (the `logits` fed to `top_k_per_row_decode`, :635) exist only here — record them optionally,
  sub-sampled (rate 0.002 as on the CPU).
- **Consumer = the hook point**: `FlashMLASparseImpl.forward_mqa`
  (`vllm/v1/attention/backends/mla/flashmla_sparse.py:838`): `topk_indices = self.topk_indices_buffer[:num_actual_toks]`
  (:857-858) with `attn_metadata` (`FlashMLASparseMetadata`, :146: `req_id_per_token` :156, `block_table`
  :155, `num_decodes/num_prefills/num_decode_tokens` :160-162, `seq_lens` :163 — Optional, present on
  the decode path) and `layer` (an `AttentionLayer`, `layer_name` = `model.layers.N.self_attn...`) all in
  scope. Hook here: wrap `forward_mqa`, read the buffer, attribute rows via `req_id_per_token`.
- **Translation to physical slots** (not needed for statistics, needed for exp3):
  `triton_convert_req_index_to_global_index` (`mla/sparse_utils.py:120`):
  `slot = block_table[req][idx // 64] * 64 + idx % 64`, called at :597 / :645 / :739.
- **Sizes**: latent 656 B/token/layer (`vllm/v1/kv_cache_interface.py:404-406`, `fp8_ds_mla`);
  indexer key 132 B (`deepseek_v2.py:697-698`: `head_dim + head_dim // 128 * 4`). k = 2048
  (`config.index_topk`, `deepseek_v2.py:663`). All 61 layers have an indexer (`index_topk_freq` absent = 1).
- **MoE routing**: `BaseRouter._select_experts`
  (`vllm/model_executor/layers/fused_moe/router/base_router.py:260`) returns `(topk_weights, topk_ids)`;
  wrap it as in the repo guide §4 (keep *logical* ids: read before any EPLB remap; EPLB off anyway).
  V3.2: 256 routed experts, 8 used, 1 shared, first 3 layers dense.

## 2. Collector (adapt `docs/vllm_selection_history_collection_guide.md` §6)

Env-gated monkeypatch installed before `LLM()` — under TP8 it must be installed **in every worker**
(pass it as a `worker_extension_cls` or via `VLLM_PLUGINS`; emit only from TP rank 0 — the buffer is
replicated, `wq_b` is `ReplicatedLinear`). One record per (request, layer, decode step):

```json
{"req": "<vllm request_id>", "layer": 17, "pos": 32767, "n_comp": 32768, "top_k": 2048,
 "valid_k": 2048, "sel": [0, 1, 5, ...], "scores": [..optional..], "phase": 1}
```
`pos = seq_lens[req] - 1` (the token being generated attends to `[0, pos]`); `n_comp = seq_lens[req]`;
`sel` = the row with `-1` stripped, **sorted ascending** (order carries no rank information in vLLM —
`valid_k == len(sel)`); `phase` 1 = decode (`num_decodes` rows), 0 = prefill (off by default; sample
1 in 64 steps if wanted). Map `req` → `sample_id` via the run's prompt manifest. Write per-request
files (`traces/by_req/<sample_id>.jsonl`) so batched decode splits into ds4-style single-request runs.

## 3. Verification (smoke gate, 8K RULER prompt, 256 steps, batch 1)

- [ ] all `sel` in `[0, pos]`; `len(sel) <= 2048`; `len(sel) == min(2048, n_comp)` for `n_comp >= 2048`
- [ ] 61 records per decode step per request; `pos` increases by exactly 1 per step
- [ ] hook-on vs hook-off `token_ids` identical (write `IDENTICAL` to `runs/smoke_identity.txt`)
- [ ] batch-attribution check: run the same prompt alone and in a batch of 4 different prompts → its
      per-step `sel` sets identical (greedy, no prefix caching); if not, the `req_id_per_token` mapping is wrong
- [ ] plausibility vs CPU V4: adjacent overlap at 8K expected in 0.6–0.9; lift > 1; recency baseline
      (fraction of `sel` within the last 2048 positions) well below the overlap
- [ ] full chain runs: `ingest_trace.py` → `validate_trace.py` (PASS) → `analyze_locality.py` →
      `analyze_hotset_coverage.py`, and MoE: `ingest_moe_trace.py` → `analyze_moe_locality.py`

## 4. Adapter to the ds4 run-directory schema (scripts unchanged)

`scripts/vllm_to_ds4_run.py <trace_dir> <sample_id> <run_dir>` (to be written on the node) produces:

| ds4 file / field | from vLLM |
|---|---|
| `traces/indexer_trace.jsonl` `{sv:2, phase, layer, ratio, pos, n_comp, top_k, valid_k, sel[], scores[]?, rank_k_score?, rank_kp1_score?}` | records of §2 for one request; `ratio: 1` (V3.2 has no compression; `original_token_range(c,1) = [c,c]`); `phase: 1`; rank scores only when scores were logged |
| `traces/moe_trace.jsonl` `{sv:2, phase, layer, pos, token, n_expert:256, n_used:8, is_hash:false, sel[8], weights[8]}` | MoE records; `is_hash` always false (V3.2 has no hash layers); dense layers 0–2 omitted |
| `outputs/generations.jsonl` (one line) `{run_id, sample_id, prompt_token_count, generated_token_count, generated_token_ids[], generated_text, finish_reason, benchmark_prediction, reference_answer[], is_correct, score}` | vLLM `RequestOutput` + benchmark scorer |
| `run_manifest.json` `{schema_version:"1", trace_schema_version:2, run_id, backend:"cuda", model_name:"DeepSeek-V3.2", model_path, quantization:"fp8 (native)", benchmark_name, task_subset, context_length_target, context_length_actual_tokens, max_new_tokens, decode_parameters{temperature:0, greedy:true, seed, batch_size}, vllm_commit, gpu:"8xH200", timing{wall_clock_seconds, gpu_hours}, is_correct}` | the runner; `context_length_target` is the ladder rung (8192…131072) — `analyze_hotset_coverage.py` reads it |
| `model_config.json` `{num_layers:61, sparse_top_k:2048, indexer_head_count:64, indexer_head_dim:128, expert_count:256, expert_used:8, layer_map:[{layer_id, attention_type:"CSA", compression_ratio:1} x 61]}` | constant; `attention_type` must be `"CSA"` for every indexer layer (`validate_trace.py` counts them; `ingest_trace.py` treats non-CSA as dense) |
| `logs/time_and_stderr.log` | vLLM stderr (token latencies are optional; the regex `decode eval N took X ms` is ds4-only) |

The analysis is index-set agnostic (`locality_lib.py`), so `ratio: 1` and k = 2048 flow through; random
baseline becomes `k / n_comp`.

## 5. Ladder and run kinds

| rung (input tokens) | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| prompts | RULER `niah_single_2` + `qa_1` (`benchmark/RULER` prepare.py at each `--max_seq_length`), LongBench v1 summarization (multi_news/gov_report/qmsum, ≤ rung), LongBench v2, InfiniteBench (En.QA/En.MC/Retrieve.PassKey); ~20 prompts per rung, balanced over sources |
| kind **bf** (benchmark-faithful) | native output length (RULER 128, LongBench v1 512, v2 / InfiniteBench per task), scored; gives task-realistic step counts and the accuracy sanity table |
| kind **ld** (long-decode) | same prompts, `min_tokens=2048, max_tokens=2048, ignore_eos=True` → >= 2K decode steps for retention curves; use long-form prompts (summarization/QA) for ld to avoid degenerate repetition; 5–10 per rung suffice |

n ≈ 20 per rung (bf) + 5–10 (ld). Decode batch 8–16 requests with attribution; `max_model_len` = rung +
2,304. Order: 8K first (both kinds, full chain, review), then 16K … 128K. Prefill is fast on GPU; the
128K rung is limited by KV memory (~48 KB/token/61 layers → 16 x 130K tokens ≈ 100 GB, fine on 8 x H200
after ~690 GB weights).

## 6. Outputs → `docs/gpu_sweep/`

- `R1_kv_locality.csv`: per run adjacent overlap, retention at lags 1/8/64/512/2048, lift, recency
  baseline, working set (from `analysis/metrics_run_summary.json`); per-rung mean ± std ± bootstrap CI.
- `R2_moe_locality.csv`: adjacent overlap of the 8 routed experts, lift vs 8/256, per-layer concentration.
- `R3_hotset_coverage.csv`: A@99 % of pool and top-10 % share per rung (the two numbers the ramulator
  testcase reads: `MEASURED_OVERLAP`, `MEASURED_TOP10`).
- plots (`generate_kv_plots.py`, `generate_moe_plots.py`, `generate_longbench_plots.py`) and
  `gpu_sweep_summary.md` with the V4-CPU curve overlaid for comparison; every number tagged with run ids.
