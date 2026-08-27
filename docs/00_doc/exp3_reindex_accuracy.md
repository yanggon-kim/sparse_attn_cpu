# exp3 — Re-index accuracy: clean vs re-indexed KV cache on DeepSeek-V3.2

*Goal: the GPU version of the ds4 correctness experiment (`ramulator2/00_doc/01_design/v5/
v5_reindex_correctness_ds4.md`, `examples/v5_ds4_reindex_correctness/compare_runs.py`): show that physically
re-indexing the KV cache during decode does not change benchmark accuracy beyond floating-point noise, on
benchmarks with official published numbers. Output: `docs/reindex_accuracy/` for the paper figure. Budget
~40 GPU-h. Run last.*

## 1. Why this is expected to hold (the claim being tested)

Attention and the indexer are permutation-invariant over the set of past entries; RoPE is applied at
insert (verified in vLLM: the 656 B row holds the RoPE'd 64-dim part); the top-k index is only ever
used through the block table (`sparse_utils.py:120`). Therefore any physical reordering that is applied
identically to the latent cache and the indexer key cache — and that the index path follows — is
invisible to the model except for summation order. The ds4 result: identity control bit-identical,
one-time permutation 127/128 tokens, periodic swaps diverge at the same step as a `-fno-fast-math`
control (step 35), selection Jaccard 0.967 in original-index space.

## 2. Two implementations (A mandatory, B optional but the stronger paper argument)

Both act between decode steps in the worker (wrap `GPUModelRunner.execute_model`: apply swaps *before*
`_prepare_inputs` computes the step's `slot_mapping`, after the previous step's KV insert has landed;
`torch.cuda.synchronize()` around the swap). Prefix caching **off**, spec-decode/MTP off, no KV
connector. Swaps only touch a request's own blocks and only positions `< seq_len - 64` (never the block
being appended to). Both caches per layer: MLA `kv_c_and_k_pe_cache` (656 B rows) and the indexer
`k_cache` (132 B rows) — vLLM allocates one tensor per layer per group; the block table is shared by all
layers of a group, so a block swap must be applied to all 61 layer tensors of both groups.

**A — block-granular, page-table style.** Pick two 64-token blocks `i, j` of one request; copy block
contents with `ops.swap_blocks(src, dst, block_size_in_bytes, block_mapping)` (`vllm/_custom_ops.py:2775`;
same tensor as src and dst via a scratch block, or `cache[[bi,bj]] = cache[[bj,bi]]` with a clone) for
every layer of both groups, and permute the request's row in `BlockTable.block_table.np` (columns `i`, `j`)
before `commit_block_table` (`vllm/v1/worker/block_table.py:184`; `MultiGroupBlockTable` :244 wraps one
`BlockTable` per group). Index space stays "position"; the table translates. This models a system with a
page table — the *baseline* the paper argues against — and is the easy correctness check.

**B — entry-granular, re-index style (the design).** Physically swap individual rows (slots) inside a
request's already-written prefix, identically in both caches for the layer, and **do not touch the block
table**. Because the indexer scans the key cache in block-table order and emits *that* order as indices,
and the latent gather resolves those indices through the same block table, the index space simply
becomes physical order — exactly the ds4 hook (`kv_perm_apply` swapping rows of `attn_comp_kv` and
`index_comp_kv`). No kernel change and no index translation is needed; the plan's variant "permute the
global indices after `triton_convert_req_index_to_global_index`" is only required if the two caches were
permuted differently — do not do that. Keep a per-request `perm_log.jsonl` (`init` map per layer, `swap`
events with step and pairs) in the `compare_runs.py` format so selections can be replayed to
original-index space.

## 3. Modes and controls (per benchmark item, all greedy, seed 42)

| run | what | expectation |
|---|---|---|
| `clean` | unmodified vLLM, hook off | must land within the official published range (§4) **before anything else runs**; run twice → identical tokens |
| `ctrl_identity` | swap machinery active with identity pairs (copies a block/row onto itself) | bit-identical to `clean` |
| `ctrl_numeric` | `clean` model under a different but numerically-equivalent execution config (e.g. `max_num_seqs` 1 vs 8, or eager vs cudagraph) | the fp-noise floor: same size of divergence as the permuted runs |
| `perm_once` | one uniform random permutation (seed 7) of the prefix after prefill | task metrics within CI of `clean` |
| `perm_periodic` | additionally every 4 steps swap 10 % of the prefix (disjoint random pairs) | same |

Use implementation A for all rows first; repeat `perm_once`/`perm_periodic` with B if time allows.

## 4. Benchmarks and the official-number gate

Long-context first: **RULER** 13 tasks at 32K/64K/128K (`benchmark/RULER` in this repo has the data
prep and `scripts/eval/evaluate.py`; 100 items/task/length is enough with bootstrap CIs, budget permitting
the official 500), **LongBench v2** (503 MC items, EM; download from `THUDM/LongBench` v2), **InfiniteBench**
subset (En.MC, En.QA, Retrieve.PassKey, Retrieve.Number). Short-context sanity: **GPQA-Diamond** (198
items) and an **MMLU-Pro** stratified subset (1,000 items) via lm-eval-harness or the official scripts
(neither is installed on the CPU box; install on the node).

Fetch the official DeepSeek-V3.2 numbers for these from the HF model card / tech report (record URL,
date, prompt format) into `docs/reindex_accuracy/official_numbers.json`. **Gate**: `clean` within the
official range (or within 1 point where only a point estimate is published) per benchmark; if not, fix
the prompt template / max tokens / scorer **once** and re-run; if `clean` is still outside the range,
**stop and report** (`GPU_CAMPAIGN.md` §5(d)) — otherwise the comparison is meaningless. If official numbers
cannot be found for a benchmark, that is a §5(b) stop for that benchmark (say which; the others proceed).
On pass write `docs/00_doc/reports/exp3_clean_<date>.md` and continue to the controls and permuted runs
without pausing. Same for GLM-5.2 if it is added.

## 5. Metrics

- Benchmark accuracy per run mode ± 95 % bootstrap CI over items (10,000 resamples), and the paired
  difference `mode - clean` with its CI (paired bootstrap on the same items).
- ds4-style agreement on a sample (20 items per benchmark, selections logged with the exp1 hook):
  identical-token count and first divergence step; per-step top-1 logit agreement and max |Δlogit| over
  the top-20 (`logprobs=20` in `SamplingParams`); selection Jaccard per (step, layer) after replaying
  `perm_log.jsonl`, restricted to the prefix window where the token streams still agree; task success.
  Port `compare_runs.py`: replace ds4's `logprobs.json` reader with vLLM's `RequestOutput.logprobs`.
- Report `ctrl_numeric` next to the permuted runs in every table (the claim is "as invisible as a
  numerically-equivalent config", not token-exact reproduction).

## 6. Outputs → `docs/reindex_accuracy/`

- `results.csv`: `benchmark, subset, context, model, impl (A|B), mode, n_items, accuracy, ci_lo, ci_hi,
  delta_vs_clean, delta_ci_lo, delta_ci_hi, official, official_source, run_ids`
- `per_item.jsonl`: `{benchmark, item_id, context, mode, impl, correct, prediction_hash, n_tokens,
  first_divergence_step, top1_agreement, max_dlogit_top20, sel_jaccard_mean}` (one line per item × mode)
- `official_numbers.json`, `agreement_sample.json`, `README.md` (setup, the gate result, the figure
  script `make_reindex_figure.py` that reads `results.csv` — grouped bars per benchmark: clean /
  ctrl_numeric / perm_once / perm_periodic with CIs and the official value as a marker).

## 7. Verification checklist

- [ ] `clean` twice → identical `token_ids` on every item (else determinism is broken; fix before continuing)
- [ ] `ctrl_identity` == `clean` bit-for-bit on 20 items
- [ ] after a swap in mode B, re-reading a swapped row pair from both caches shows the exchange
      (unit test on a 3-token run: swap rows 0 and 1 of layer 0, read back)
- [ ] `perm_log.jsonl` replay maps every logged `sel` index back into `[0, pos]` with no duplicates
- [ ] no swap touches a block/slot at or beyond `seq_len - 64`; no cross-request swap (assert in code)
- [ ] `results.csv` row count = benchmarks × modes × impls; every row has `run_ids`
- [ ] gate report `docs/00_doc/reports/exp3_reindex_<date>.md` written; then the final report
      (`reports/final_<date>.md`, `GPU_CAMPAIGN.md` §6)
