# START HERE — exp3 "re-index accuracy" (perplexity comparison), tiers 1 and 2

Written 2026-08-29 for an agent with no memory of the campaign. Everything referenced below is in this directory
(`docs/reindex_accuracy/`) or in `docs/00_doc/`; the raw numbers can be regenerated from the JSON files with the two
results scripts — never hand-edit tables.

## 1. The question and why the answer is a perplexity number

The SPLEX design physically **re-indexes** (moves) KV-cache entries of a DeepSeek-style sparse-attention (DSA) model
during inference so that hot entries sit together. Attention and the sparse indexer are permutation-invariant over the
set of cached entries (positions are baked in at insertion), so moving rows must not change the model's outputs — up to
floating-point summation-order noise. exp3 exists to *demonstrate* that on real models: DeepSeek-V3.2, GLM-5.2, GLM-5.

Comparing generated text was the original plan, but greedy decode on this stack (vLLM 0.28.0, 8× B200, TP8) is **not
bit-reproducible**: identical reruns diverge after 10–24 tokens at near-tie logits and the texts drift (11 kernel/backend
configurations probed, none deterministic — `determinism_probes.md`). So the primary metric is **teacher-forced
perplexity**, InfiniGen-style: the same fixed tokens are scored in every configuration, so the comparison is paired per
token and the only variability is numerical. A **noise floor** is measured in the same engine by running the unmodified
baseline twice (`clean` vs `clean2`). A configuration is "equivalent" when it is indistinguishable from that floor.

## 2. Protocol (identical in tier 1 and tier 2)

- Engine: vLLM 0.28.0 (tag `2cf0a69`), TP8, `enforce_eager=True`, prefix caching off, `max_num_batched_tokens=2048`
  (prefill chunk = 2048 tokens, so the re-index point is an exact chunk boundary), `max_num_seqs=2`, seed 42.
- Item: a document prefix of *P* ∈ {32K, 64K, 128K} tokens is prefilled; at the chunk boundary where ≥ *P* tokens are in
  the cache the hook permutes the prefix; the next 2K tokens are prefilled with `prompt_logprobs=1` and their log-probs
  are the score. PPL = exp(−mean logprob). Short anchors: WikiText-2 (and, tier 2, PTB) windows, 2K prefix / 1K scored.
- Hook: `scripts/gpu/selhook/reindex_ext.py` (vLLM worker extension, mode switched at runtime with
  `collective_rpc("reindex_set", …)`). Both KV caches of every layer are permuted identically: the MLA latent cache
  (bf16, 576 B rows — vLLM's default `auto` dtype on B200) and the indexer key cache (64-token block laid out as
  `[64×128 fp8 values][64×4 B fp32 scales]` — getting this layout wrong broke retrieval outright in an early version,
  which is the proof that the test can fail).
- Configurations ("modes"):
  | mode | meaning |
  |---|---|
  | `clean` | baseline |
  | `clean2` (`clean3/4` in the tier-1 diagnostic) | identical rerun = **noise floor** |
  | `ctrl_identity` | hook active, identity permutation (mechanism on, nothing moved) |
  | `ctrl_numeric` | baseline item processed alongside a 4K filler request = batch-composition noise |
  | `perm_once_A[@seed]` | **impl A**: permute whole 64-token blocks and update the block table (what a page-table system would do) |
  | `perm_once_B[@seed]` | **impl B**: permute individual rows inside the prefix, block table untouched — the SPLEX design (indices become physical; `permlog_*` records the permutation) |
  | `perm_periodic_B` | generation only: every 4 decode steps 10 % of the prefix swapped in disjoint pairs |
  Seed suffix `@8` / `@9` = permutation seeds 8 / 9; no suffix = seed 7.
- Equivalence rule (`exp3_tier1_results.py`, `exp3_tier2_results.py`): a mode is equivalent to the floor when
  (i) its mean paired ΔPPL vs `clean` lies within the `clean2` floor bound (max |edge| of clean2's paired-bootstrap 95 %
  CI, 10,000 resamples) **and** (ii) its per-token |Δlogprob| 90th percentile is ≤ 1.5× max(floor p90, `ctrl_numeric`
  p90). Tier 2 adds the weaker, variance-aware column `delta_ci_includes_zero`.
- Secondary evidence: greedy generation on RULER (needle retrieval / QA) with re-indexing between decode steps, scored by
  `scripts/gpu/score.py`; reported as accuracy, Δ vs clean, number of token streams bit-identical to clean, median first
  divergence step.

## 3. Status (2026-08-29)

| tier | models | PPL items | modes | generation | verdict |
|---|---|---|---|---|---|
| 1 (done) | V3.2, GLM-5.2, GLM-5 | 10 books × 32K/64K/128K (2K scored) + WikiText-2 50 windows | 32K: all six; 64K: clean, clean2, A, B; 128K: clean, clean2, B; V3.2 extra: clean3/4, A@8, B@8, B@9 at 32K | RULER niah_single_2 + qa_1, 5 + 5 at 32K and 128K, modes clean / perm_once_B / perm_periodic_B | **77/78 rows in the floor**; RULER accuracy identical in every mode (16/20, 18/20, 20/20) |
| 2 (V3.2 only, stopped) | V3.2 | 30 books × 32K/64K/128K + WikiText-2 93 windows + PTB 32 windows | **all 10 modes at every prefix** (clean, clean2, identity, numeric, A×3 seeds, B×3 seeds) | `clean` only, 325/400 items (RULER niah_single_2 / niah_multikey_2 / vt / qa_1 × 3 lengths × 25 + LongBench-v2 25) — stopped by the user before the re-indexed passes | **38/40 rows in the floor, all 30 long-book rows**; tier-1's single exception disappears at n = 30 |

Not run: GLM-5.2 / GLM-5 tier 2; tier-2 re-indexed generation; tier 3 (MMLU-Pro thinking, InfiniteBench tasks,
impl A in generation). GPQA is gated (no access) and was dropped. GPU-h spent on exp3 ≈ 15 (tier 1) + 28 (tier 2).

## 4. Headline numbers (ΔPPL vs clean; floor = clean2; paired over documents)

Tier 2, DeepSeek-V3.2, 30 books (baseline PPL 1.3058 / 1.2678 / 1.3674 at 32K / 64K / 128K):
| config | 32K | 64K | 128K |
|---|---|---|---|
| identical rerun (floor) | −0.0013 [−0.0042, +0.0015] | −0.0019 [−0.0060, +0.0012] | −0.0018 [−0.0043, +0.0005] |
| identity re-index | +0.0023 | +0.0021 | −0.0008 |
| batch-composition control | +0.0012 | +0.0037 | −0.0015 |
| impl A seeds 7 / 8 / 9 | −0.0001 / +0.0010 / +0.0012 | +0.0018 / +0.0017 / +0.0009 | −0.0028 / −0.0022 / −0.0005 |
| impl B seeds 7 / 8 / 9 | +0.0010 / −0.0013 / −0.0013 | +0.0005 / +0.0023 / +0.0023 | −0.0015 / −0.0034 / −0.0002 |
| token \|Δlogprob\| p90, every mode incl. floor | 0.048–0.051 | 0.063–0.067 | 0.089–0.094 |

Tier 1, 10 books, impl B vs floor: V3.2 −0.0009 (seed 8) / +0.0014 / −0.0012 vs −0.0008 / +0.0015 / −0.0031;
GLM-5.2 −0.0073 / +0.0107 / −0.0005 vs −0.0058 / +0.0130 / −0.0052 (baseline 1.883 / 2.215 / 2.596);
GLM-5 −0.0049 / +0.0041 / −0.0006 vs −0.0081 / +0.0070 / −0.0040 (baseline 1.610 / 1.888 / 2.092).
WikiText-2: all modes within ±0.003 of baseline (3.001 / 2.984 / 2.863 for the three models in tier 1; 3.244 in tier 2).

Interpretation notes an agent should not re-discover:
- Token-level noise is heavy-tailed *for every mode including identical reruns*: ~1.5 % of tokens shift > 0.5 nats
  (MoE routing flips at near-ties). Compare distributions (p90), never exact token matches.
- At 32K–128K the p90 of every mode equals the floor's. At 2K (short anchors) impl B and `ctrl_numeric` share a higher
  p90 (0.20–0.24 vs 0.05–0.10 for impl A / identity / rerun): row permutation changes the sparse-gather summation order
  the way a different batch composition does; block permutation does not. It is numerical, not semantic.
- The two tier-2 rows that fail rule (i) are PTB impl-B seeds 7 and 9: +0.0060 / +0.0081 on PPL 5.797 (+0.1 %, ≈1 SE
  each, seed 8 −0.0019, both CIs include 0). Rule (i) compares a mean against a bound derived from the lower-variance
  rerun distribution and is under-powered on 32 windows. The tier-1 exception (V3.2 32K impl B seed 7, +0.0062) was one
  document (+0.057); seeds 8/9 and two extra reruns cleared it, and at 30 documents the same seed gives +0.0010.
- Generation: RULER `vt` (variable tracking) needs `max_new_tokens=256` with our prompt format — the exp3 prompts omit
  RULER's "Answer: … they are:" prefix, so with RULER's 30-token default the chat model is cut off mid-trace (0/25).
  Fixed in `exp3_tier2.py`; tier-1 RULER used only niah_single_2 and qa_1 and is unaffected.

## 5. Raw data inventory (all committed; nothing lives only on the node)

| file | what | schema |
|---|---|---|
| `tier1_v32.json`, `tier1_glm52.json`, `tier1_glm5.json` | tier-1 raw output of `exp3_tier1.py` (V3.2 includes the seed/rerun diagnostic merged in) | `ppl[]`: `{mode, prefix_len, doc, n_scored, mean_logprob, ppl, logprobs[2048 or 1024], events, errors}`; `acc[]`: `{mode, sample_id, task, rung, correct, score, n_tokens, token_ids[], text, events, errors}`; `timing`, `plan` |
| `tier2/tier2_v32.json` (29 MB) | tier-2 raw output of `exp3_tier2.py` | same plus `corpus` (`longbook` / `wikitext2` / `ptb`) in `ppl[]`; `acc[]` adds `source`, `extracted`, `max_new_tokens`; `acc_selection` |
| `results.csv`, `tier2/results.csv` | one row per model × benchmark × prefix × mode (PPL rows: value, ΔPPL, CI, token p90, verdict; accuracy rows) | produced by the results scripts |
| `per_item.jsonl`, `tier2/per_item.jsonl` | per-document ΔPPL and per-item generation outcomes | |
| `tier1_results.md`, `tier2/tier2_results.md`, `reindex_ppl_delta.png`, `tier2/reindex_ppl_delta_tier2.png` | tables and figures | |
| `permlogs/permlog_{v32,v32diag,glm52,glm52acc,glm5}.tar.gz.part*`, `tier2/permlogs/permlog_t2_v32.tar.gz.part*` (+ `SHA256SUMS`) | per-event permutation logs written by the hook (`init` / `swap` records: request, step, impl, seq_len, n_units, moved, the permutation) — replay maps physical → original index for impl B | `cat …part* \| tar xz` |
| `unit_test_v32.json` | hook unit test (row read-back after an impl-B swap, mode sweep on 4 prompts) | |
| `determinism_probes.md`, `docs/00_doc/data_gpu/determinism/*.json` | the 11 non-determinism probes | |
| `official_numbers.json` | MMLU-Pro / GPQA official values (tier 3, unused) | |
| `docs/00_doc/data_gpu/logs/exp3_*` | run logs and launch scripts (paths scrubbed to `<WORKDIR>`) | |
| `docs/00_doc/data_gpu/prompts_exp3/` | manifests of the 5,932 exp3 prompts (texts regenerate with `scripts/gpu/build_exp3_prompts.py`, seed 42) | |

Documents: InfiniteBench books (`longbook_sum/qa/choice_eng`, de-duplicated on the first 2,000 characters), shuffled with
`random.seed(42)`, first *n* with ≥ 131072 + 2048 + 16 tokens under the model's tokenizer; tier 2's 30 start with tier 1's
10. WikiText-2 test and PTB test (Mikolov split) tokenized as one stream and cut into 3,072-token windows.

## 6. Reports and prose

- `README.md` (tier-1 method + verdict), `tier2/README.md` (tier-2 scope, verdict, the two flagged rows, what was stopped).
- `paper_accuracy_section.md` — paper-ready methodology / results / takeaway text, Table 1 (tier-1 PPL), Table 2
  (tier-1 RULER), Table 3 addendum (tier-2 V3.2 PPL).
- Gate reports: `docs/00_doc/reports/exp3_stop_20260829.md` (why perplexity), `exp3_clean_20260829.md`,
  `exp3_reindex_20260829.md` (tier 1), `exp3_tier2_20260829.md`. Tracker: `docs/00_doc/PROGRESS.md`.
- Campaign spec the protocol deviates from: `docs/00_doc/exp3_reindex_accuracy.md` (written before the determinism
  finding; accuracy-with-official-numbers gate replaced by the perplexity protocol with user approval).

## 7. How to regenerate, extend, or resume

```bash
cd 01_github/sparse_attn_cpu
python3 scripts/gpu/exp3_tier1_results.py --out docs/reindex_accuracy docs/reindex_accuracy/tier1_v32.json \
        docs/reindex_accuracy/tier1_glm52.json docs/reindex_accuracy/tier1_glm5.json      # tables/figure from raw
python3 scripts/gpu/exp3_tier2_results.py --out docs/reindex_accuracy/tier2 docs/reindex_accuracy/tier2/tier2_v32.json
```
(Both need only Python + matplotlib; `--n-boot 500` for a quick check.) The results scripts accept several JSONs and any
mode names of the form `<base>[@seed]`; new modes need a line in `MODE_CFG` of the runner.

On the GPU node (`source $WORKDIR/env/activate.sh`; models, prompts, benchmark data as listed in `HANDOFF.md` §GPU):
```bash
python scripts/gpu/exp3_tier2.py --model $WORKDIR/models/GLM-5.2-FP8 --tag t2_glm52 --out $WORKDIR/runs/exp3/tier2_glm52.json --resume
python scripts/gpu/exp3_tier2.py --model $WORKDIR/models/DeepSeek-V3.2 --tag t2_v32 --out $WORKDIR/runs/exp3/tier2_v32.json --resume --skip-ppl
```
The first runs GLM-5.2 tier 2 from scratch (≈2.5 h wall for the PPL block, ≈5 h more for generation); the second resumes
V3.2's generation block (perm_once_B, perm_periodic_B) — `--resume` skips every (mode, prefix, corpus) group already
complete in the JSON, and `--plan '{"32768": ["clean","perm_once_B"]}'` restricts the PPL plan. Run detached
(`setsid nohup … < /dev/null &`), one model at a time, and kill leftover vLLM workers by PID before a relaunch.
Copy the finished JSON into this directory, split `permlog_<tag>` into ≤ 40 MB parts, rerun the results script, and
extend `paper_accuracy_section.md` — commits must stay ≤ 1.5 GB per push and files ≤ 45 MB.
