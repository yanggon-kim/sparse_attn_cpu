# v6 trace export format (selection traces for the ramulator migration-policy study)

One file pair per run under `<WORKDIR>/experiment/exports/v6/` (off-repo, ~2.5 MB per 160-step V4 run,
102 MB for the 42 V4 runs): `<run_id>.npz` + `<run_id>.manifest.json`, plus one `retention_curves.json`.
Producer: `scripts/export_v6_traces.py` (reads `runs/<id>/traces/selected_kv.parquet` only; one decode
step is asserted against the raw `indexer_trace.jsonl`). Invocation (defaults = all runs, skip the
superseded `_q2` RULER series and the aborted 96K):

    python3 scripts/export_v6_traces.py [--runs-root <experiment>/runs] [--out-dir <experiment>/exports/v6] [--only <run_id> ...]

## `<run_id>.npz` (`np.load`, compressed)
| key | dtype / shape | meaning |
|---|---|---|
| `sel` | `uint16` if max index < 65536 else `uint32`; `[steps, layers, k]` | selected entry ids per (decode step, indexer layer), **sorted ascending per row**; unused slots (rank >= `valid_k`, or a -1 / invalid id in the source) hold the dtype max (0xFFFF / 0xFFFFFFFF) and sort last |
| `valid_k` | `int32 [steps, layers]` | number of valid ids in the row (= `min(k, n_comp)` in practice; < k only while the pool is smaller than k) |
| `pos` | `int32 [steps]` | `decode_position`: 0-based token index of the query token of that step |
| `n_comp` | `int32 [steps]` | candidate pool size seen by the indexer = `(pos + 1) // ratio` (verified against every ds4 record; **not** `pos // ratio`) |
| `layers` | `int32 [layers]` | model layer ids that carry a top-k selection (V4-Flash: the 21 CSA layers 2,4,…,42) |
| `k`, `ratio` | `int32` scalars | top-k and tokens-per-entry |
| `model` | str scalar | e.g. `DeepSeek-V4-Flash` |

Units. **V4**: an entry is a *compressed block* of `ratio` = 4 tokens (k = 512 blocks; pool N = `n_comp`).
**V3.2 / GLM (GPU exports)**: emit `ratio = 1`, `k = 2048`, `pos` = the query's 0-based position so that
`n_comp = pos + 1` = N tokens; ids are per-request token positions (translate through the block table
before writing, see `exp1_dsv32_gather_index.md` §4); vLLM's -1 padding becomes fill + `valid_k`. Steps
are decode steps only (traces are decode-only; the first steps serve as the prefill tail).

## `<run_id>.manifest.json`
`run_id, model, quantization, engine, benchmark, task, prompt_tokens, context_length_actual_tokens, steps,
layers, k, ratio, sel_dtype, fill_value, max_index, pos_first, pos_last, all_rows_full, source_parquet,
source_parquet_sha256, npz, npz_bytes, jsonl_one_step_check, export_date, script,
script_git_hash_sparse_attn_cpu, units`.

## `retention_curves.json`
`families.<family>` (`ruler_<ctx>`, `longbench_<task>`, `longdecode`) → `runs.<run_id>.retention{lag: value}`
(lags 1…64 from `analysis/metrics_run_summary.json`; `retention_extended` up to lag 2048 for the long-decode
run from `analysis/extended_retention.json`) and `mean_retention` over the family's runs. Retention at lag
L = mean over (layer, step) of |S_t ∩ S_{t−L}| / |S_t|. The consumer (`retention_pred` policy) must use a
leave-one-out curve (exclude the evaluated run; see `consumer_note` in the file).

Consumer: ramulator `examples/v6_policy/traces.py` (branch `feature/hbm-lpddr-basedie-v6`).
