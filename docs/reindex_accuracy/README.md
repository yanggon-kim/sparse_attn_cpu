# exp3 — Re-index accuracy (tier 1)

**Question.** Does physically re-ordering a request's KV cache during decode change the model's outputs beyond the
run-to-run numerical noise of the inference stack? (SPLEX paper correctness figure.)

**Setup.** vLLM 0.28.0, 8× B200 (TP8, eager, prefix caching off, greedy). Hook: `scripts/gpu/selhook/reindex_ext.py`
(worker extension; mode switched at runtime via `collective_rpc("reindex_set")`). Both KV caches of every layer are
permuted identically (MLA latent cache 576 B bf16 rows — vLLM's default `auto` dtype here — and the indexer key cache,
whose 64-token block is `[64×128 fp8][64×4 B scales]`).
- **impl A** (page-table baseline): permute whole 64-token blocks and update the block-table row.
- **impl B** (re-index design): permute individual rows inside the written prefix; block table untouched, so indices
  become physical; `permlog_*/…jsonl` records the permutation for replay.
- Modes: `clean`, `clean2/3/4` (identical reruns = **noise floor**), `ctrl_identity` (hook on, identity permutation),
  `ctrl_numeric` (same item processed alongside a filler request = batch-composition noise), `perm_once_{A,B}[@seed]`
  (uniform random permutation of the prefix, seed 7/8/9), `perm_periodic_B` (generation only: every 4 decode steps 10 % of
  the prefix swapped in disjoint pairs).

**Why perplexity.** Greedy generation is not bit-reproducible on this stack (`determinism_probes.md`): fp summation order
flips near-tie tokens and the texts drift, so "identical tokens" cannot be the criterion. Teacher-forced scoring feeds the
same fixed tokens in every mode and reads their log-probs, so the comparison is paired per token and the only variability
is numerical. **Primary metric:** perplexity of a 2K continuation after a 32K/64K/128K prefix that was re-indexed at the
prefill-chunk boundary (10 InfiniteBench books ≥133K tokens, seed 42; `max_num_batched_tokens=2048` makes the boundary
exact) plus WikiText-2 (50 windows, 2K prefix / 1K scored). **Secondary:** RULER niah_single_2 + qa_1 (5+5 items at 32K
and 128K) generated greedily with re-indexing between decode steps.

**Equivalence rule** (`scripts/gpu/exp3_tier1_results.py`): a mode is *equivalent to the noise floor* when (i) its mean
paired ΔPPL vs `clean` lies within the `clean2` floor bound (max |edge| of the clean2 paired-bootstrap 95 % CI, 10,000
resamples) and (ii) its per-token |Δlogprob| p90 is within 1.5× the larger of the floor's and `ctrl_numeric`'s p90.
Token-level noise is heavy-tailed for every mode including identical reruns (~1.5 % of tokens shift >0.5 nats: MoE
routing flips at near-ties), so the token distribution is compared as a whole, not by exact match.

**Files.** `results.csv` (one row per model × benchmark × prefix × mode), `per_item.jsonl`, `tier1_results.md` (tables),
`reindex_ppl_delta.png`, `unit_test_v32.json` (hook unit test), `determinism_probes.md`, `official_numbers.json`
(MMLU-Pro/GPQA official values — those benchmarks are deferred to tier 3; GPQA dropped, gated).

**Tier-1 result (2026-08-29): 77 of 78 mode rows equivalent to the noise floor across DeepSeek-V3.2, GLM-5.2, GLM-5 (impl A and B); the exception is one high-variance document at one seed (V3.2, 32K, seed 7), cleared by seeds 8/9. RULER answers unchanged. See `tier1_results.md`.**
