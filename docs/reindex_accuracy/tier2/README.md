# exp3 tier 2 — re-index perplexity at scale (DeepSeek-V3.2; stopped after the PPL block by user decision, 2026-08-29)

Same protocol as tier 1 (`../README.md`), scaled up on **DeepSeek-V3.2** in one engine load (`scripts/gpu/exp3_tier2.py`,
vLLM 0.28.0, 8× B200 TP8, eager, prefix caching off, `max_num_batched_tokens=2048`):

| block | items | modes | status |
|---|---|---|---|
| long books (InfiniteBench, ≥133K tokens, seed-42 selection; tier-1's 10 are the first 10) | **30 docs** × prefix 32K / 64K / 128K, 2K scored | clean, clean2, ctrl_identity, ctrl_numeric, perm_once_A / perm_once_B × seeds 7, 8, 9 (**all 10 modes at every prefix**) | done — 900 rows |
| WikiText-2 (2K prefix / 1K scored) | 93 windows (the whole test set) | same 10 modes | done — 930 rows |
| PTB test (Mikolov split, 2K prefix / 1K scored) | 32 windows (the whole test set) | same 10 modes | done — 320 rows |
| generation accuracy: RULER niah_single_2 / niah_multikey_2 / vt / qa_1 × 32K/64K/128K × 25 + LongBench-v2 100 | 400 items | clean / perm_once_B / perm_periodic_B | **stopped**: only `clean` 325/400 recorded (300 RULER + 25 LongBench-v2); no re-indexed generation |

GPU time: PPL block 2.4 h wall (≈19 GPU-h) + 1.1 h of the ACC block (≈9 GPU-h). GLM-5.2 / GLM-5 tier 2 was not started.

**Result (PPL, 40 verdict rows):** 38/40 equivalent to the identical-rerun noise floor under the tier-1 rule; every one of
the 40 rows has a paired-ΔPPL 95 % CI that overlaps the floor's, and every long-book row (30 of 30) passes outright.
The two flagged rows are PTB impl-B seeds 7 and 9 (+0.0060 and +0.0081 PPL on 5.797, i.e. +0.1 %, ≈1 SE each, seed 8 −0.0019,
CIs include 0): the rule compares a mean against a floor bound derived from the *lower-variance* clean2 distribution, and impl B's
per-token noise is the batch-composition (`ctrl_numeric`) level, not the rerun level, so on 32 windows the mean test is
under-powered. `delta_ci_includes_zero` (added in `exp3_tier2_results.py`) is the variance-aware complement: true for both.

Long-book headline (30 docs, ΔPPL vs clean; floor = clean2):
- 32K: floor −0.0013 [−0.0042, +0.0015]; impl A −0.0001 / +0.0010 / +0.0012; impl B +0.0010 / −0.0013 / −0.0013 (seeds 7/8/9); identity +0.0023; numeric +0.0012.
- 64K: floor −0.0019 [−0.0060, +0.0012]; impl A +0.0018 / +0.0017 / +0.0009; impl B +0.0005 / +0.0023 / +0.0023; identity +0.0021; numeric +0.0037.
- 128K: floor −0.0018 [−0.0043, +0.0005]; impl A −0.0028 / −0.0022 / −0.0005; impl B −0.0015 / −0.0034 / −0.0002; identity −0.0008; numeric −0.0015.
- The tier-1 outlier (impl B seed 7 at 32K, one document, +0.057) is absorbed at n = 30: +0.0010 [−0.0016, +0.0039].
- Token-level |Δlogprob| p90 at 32K/64K/128K: 0.048/0.065/0.091 nats for the rerun floor and 0.049–0.051 / 0.063–0.067 / 0.089–0.094 for every other mode.
  At 2K (short anchors) impl B and `ctrl_numeric` share a higher p90 (0.20–0.24 vs 0.05–0.10 for impl A / identity / rerun): row permutation
  changes the sparse-gather summation order the way a different batch composition does; block permutation does not.

Clean generation accuracy recorded before the stop (V3.2, greedy, chat template, no thinking): niah_single_2 25/25, niah_multikey_2 25/25 and
vt 25/25 at 32K/64K/128K; qa_1 18/17/14 of 25; LongBench-v2 14/25 (code 5/10, in-context learning 9/15). RULER `vt` needs
`max_new_tokens=256` here (RULER's 30-token default assumes its "Answer: … they are:" prefix, which the exp3 prompts omit; with
30 tokens the chat model is cut off mid-trace and scores 0/25 — fixed in `exp3_tier2.py`, rows regenerated).

Files: `results.csv` (PPL rows + clean accuracy rows), `per_item.jsonl`, `tier2_results.md` (tables), `reindex_ppl_delta_tier2.png`,
`tier2_v32.json` (raw: every scored token's log-prob per mode/doc, generation token ids), `permlogs/permlog_t2_v32.tar.gz.part*`
(+ `SHA256SUMS`; `cat permlogs/*.part* | tar xz` → per-event permutation logs for replay). Produced by
`scripts/gpu/exp3_tier2.py` → `scripts/gpu/exp3_tier2_results.py --out docs/reindex_accuracy/tier2 tier2_v32.json`.
