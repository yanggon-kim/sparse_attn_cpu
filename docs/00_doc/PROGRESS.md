# GPU campaign progress (B200 x8 node)

Top-level tracker for the campaign in `GPU_CAMPAIGN.md`. One line per phase; details in `reports/`.
Node: 8x NVIDIA B200 183 GB (1.47 TB HBM), SM 10.0, driver 595.91.07 / CUDA 13.2 runtime, 192 cores,
1.9 TB RAM, NVMe `<WORKDIR>` = 28 TB. Started 2026-08-28.

| phase | status | started | finished | report | GPU-h (cum.) |
|---|---|---|---|---|---|
| setup (env, models, benchmarks) | DONE | 2026-08-28 20:25 UTC | 2026-08-28 21:05 UTC | (this file, §Setup) | 0 |
| exp0 environment gate | PASS | 2026-08-28 21:10 UTC | 2026-08-28 22:05 UTC | `reports/exp0_20260828.md` | 5 |
| exp1 smoke | PASS | 2026-08-28 22:05 UTC | 2026-08-28 22:50 UTC | `reports/exp1_smoke_20260828.md` | 7.5 |
| exp1 ladder 8K-128K | PASS | 2026-08-28 22:35 UTC | 2026-08-29 03:45 UTC | `reports/exp1_ladder_20260829.md` | 23 |
| exp2 GLM-5.2 + GLM-5 | PASS | 2026-08-29 03:45 UTC | 2026-08-29 08:40 UTC | `reports/exp2_20260829.md` | 48 |
| exp3 tier 1 (PPL + RULER sanity, 3 models) | PASS | 2026-08-29 09:10 UTC | 2026-08-29 13:50 UTC | `reports/exp3_clean_20260829.md`, `reports/exp3_reindex_20260829.md` | 65 |
| exp3 tier 2 (V3.2 PPL ×30 docs, all modes/seeds; stopped before re-indexed generation) | PASS (PPL) / stopped | 2026-08-29 16:58 UTC | 2026-08-29 21:25 UTC | `reports/exp3_tier2_20260829.md` | 93 |
| exp3 tier 2 V3.2 generation (clean + perm_once_B complete, perm_periodic_B 225/400) | partial — node taken by another workload | 2026-08-31 02:35 UTC | 2026-08-31 05:54 UTC | `reports/exp3_tier2_20260829.md` §addendum | 128 |
| exp3 tier 2 GLM-5.2 / GLM-5 PPL, tier 3 | not run (blocked: GPUs busy) | | | | |
| final | DONE | | 2026-08-29 14:00 UTC | `reports/final_20260829.md` | 65 |

## Setup (2026-08-28, no GPU time spent)
- Env: uv venv Python 3.12 at `<WORKDIR>/env/vllm`; **vLLM 0.28.0** (tag `2cf0a69`, 2026-08-24; contains the
  cited pin `5559679`), torch 2.13.0+cu130, flashinfer 0.6.16.post3, lm-eval 0.4.12, transformers 5.16.1.
  nvcc 13.0 from `/opt/pytorch/cuda` (`CUDA_HOME`). Activate: `source <WORKDIR>/env/activate.sh`.
  vLLM source clone at `<WORKDIR>/env/vllm-src` @ v0.28.0 for grepping hook sites.
- Hook symbol line drift vs docs (symbols intact): `sparse_utils.py::triton_convert_req_index_to_global_index`
  120 -> 154; `deepseek_v2.py index_topk_freq` 1092 -> 1087; `forward_mqa` 838 (same); `sparse_attn_indexer` 296
  (same); `_select_experts` 260 (same); `GlmMoeDsaForCausalLM` registry 117 -> 118. FlashMLA sparse supports
  capability family 100 (`vllm/v1/attention/ops/flashmla.py:71-72`).
- Models on NVMe (`<WORKDIR>/models/`): `DeepSeek-V3.2` (native FP8, 163 shards, 643 GB), `GLM-5.2-FP8`
  (141 shards, 704 GB), `GLM-5-FP8` (142 shards, 705 GB). All ungated. **Deviation:** `zai-org/GLM-5.2` and
  `zai-org/GLM-5` are BF16 (1.4 TB each, do not fit); the official `-FP8` repos are used instead (indexer
  projections listed in `modules_to_not_convert`, i.e. unquantized).
- Config facts: V3.2 61 layers, index_topk 2048, 64 idx heads. GLM-5.2: 78 layers, index_topk 2048, 32 idx heads,
  `index_topk_freq` 4, `index_skip_topk_offset` **3** (doc expected 2). GLM-5: 78 layers, index_topk 2048, no
  freq -> every layer computes top-k -> **GLM-5 has DSA, will be run**. All three: 256 experts / 8 used,
  kv_lora_rank 512, rope 64.
- Benchmarks: RULER @ `38da79d` + Paul Graham essays + nltk punkt; LongBench v1 (`zai-org/LongBench`),
  LongBench v2 (`zai-org/LongBench-v2`), InfiniteBench (`xinrongzhang2022/InfiniteBench`), MMLU-Pro
  (`TIGER-Lab/MMLU-Pro`) under `<WORKDIR>/benchmark/`. **GPQA (`Idavidrein/gpqa`) is gated: access request
  pending from the user.**
- Official numbers (VERIFY item): DeepSeek-V3.2 tech report (HF `assets/paper.pdf`, Table 2, thinking mode,
  T=1.0, 128K ctx): MMLU-Pro 85.0 EM, GPQA-Diamond 82.4. GLM-5.2 card: GPQA-Diamond 91.2; GLM-5 card: 86.0.
  **No official RULER / LongBench-v2 / InfiniteBench numbers found for any of the three models** (to be
  re-checked against the LongBench-v2 / RULER leaderboards at exp3; else §5(b) per benchmark).

## exp0 (2026-08-28) — PASS
- Backend on B200: FLASHINFER_MLA_SPARSE + FLASHINFER_TRTLLM fp8 MoE (monolithic). Hook = `scripts/gpu/selhook/`
  (worker_extension_cls), KV via `forward_mqa`, MoE via kernel `set_capture_fn`. Smoke tokens `[11111,16,455]`
  identical hook-off/on; 16/16 unit checks; full ds4 chain PASS on `v32_smoke0_capital_s0`. Details in the report.

## exp1 smoke (2026-08-28) — PASS
- 8K niah ×4 batched + solo, 256 steps: adj overlap 0.817 / lift 3.02× / recency 0.43 / A@99 67 % (CPU V4 8K: 0.790 / 2.92× / 65.7 %).
- Not run-to-run deterministic (solo diverges at step 38 between identical runs) → exp3 needs a deterministic kernel config test.
- Fast twins: `ingest_trace_fast.py` (verified identical), `analyze_locality_fast.py` (original too slow at GPU scale).

## exp1 ladder generation (2026-08-28/29)
- Rungs 8K/16K/32K/64K/128K: 28/28/28/27/24 runs, GPU-h 1.32/1.42/1.60/1.80/2.13 (wall 10-16 min each incl. load).
  All RULER niah retrieved at every rung; ld runs 2048 forced steps. Analysis via memory-aware queue
  (`queue_analysis.sh`), packaging via `finish_rung.sh` (traces sharded <45 MB into `docs/gpu_sweep/runs/`).
- Determinism probe for exp3 queued on the GPU between exp1 and exp2 (`determinism_probe.py`).

## Determinism probes for exp3 (2026-08-29 00:00-01:00 UTC, DeepSeek-V3.2, 8K niah prompt, 512 greedy steps x3 solo + batch-of-4)
| config | solo run-to-run identical? | first divergence (steps) |
|---|---|---|
| baseline (FLASHINFER_MLA_SPARSE + FLASHINFER_TRTLLM fp8 MoE, eager, TP8) | no | 10 / 24 |
| MoE backend DeepGEMM | no | 10 / 10 |
| MoE backend Triton | no | 24 / 10 |
| MoE backend FlashInfer-CUTLASS | unsupported (block-quant fp8) | — |
| VLLM_BATCH_INVARIANT=1 | unsupported (no batch-invariant sparse-MLA backend) | — |
| NCCL-only allreduce (no custom/symm-mem) | no | 10 / 10 |
Pre-divergence top-1 logprob noise between identical runs: mean 0.008-0.011, max 0.04-0.08 nats; divergence happens at
near-ties (e.g. step 10: -0.576 vs -0.693). Remaining probes: NCCL+DeepGEMM, cudagraph, FlashMLA-sparse attention.
Consequence: bit-exact "clean x2" is not achievable on this stack; exp3 equivalence will be judged against the
run-to-run noise floor (clean vs clean2) and ctrl_numeric — a §5(b) item to confirm with the user before exp3.
GPU-h for probes ~8.

## exp1 ladder (2026-08-29) — PASS
- 135 runs; adj overlap 0.843/0.790/0.751/0.741/0.723, lift 2.6/5.3/10.2/21.9/41.5×, A@99 65/45/30/17/10 %, top-10 % coverage
  0.30/0.57/0.79/0.94/0.98 at 8K/16K/32K/64K/128K; MoE adj overlap ≈0.25 (8×) context-independent. verify_sweep PASS.
- `docs/gpu_sweep/` committed incl. 135 sharded raw-trace run dirs (13 GB).

## exp2 (2026-08-29) — PASS
- GLM-5.2 (a/b) and GLM-5 ladders (135 runs each): adj overlap 0.846/0.789/0.750/0.722/0.702 (GLM-5.2 a), 0.849/0.791/0.753/0.730/0.715 (GLM-5)
  vs V3.2 0.843/0.790/0.751/0.741/0.723; A@99 62/43/29/18/12 and 62/43/29/19/11 vs 65/45/30/17/10; MoE ≈0.29 (9×). verify PASS ×3.
- exp3 hook unit test passed for impl A and impl B after the indexer-cache layout fix (`docs/reindex_accuracy/unit_test_v32.json`).

## exp3 — user decisions (2026-08-29 09:00 UTC) and tier-1 plan
- Determinism: bit-identical reruns are impossible on this stack (11 probes); equivalence is judged against the
  run-to-run noise floor (clean vs clean2) with a teacher-forced **perplexity** primary metric (InfiniGen-style).
- Budget: **tier 1 ≈ 8 GPU-h per model** ("important minimum first, expand later"); tiers 2/3 deferred.
- GPQA-Diamond: **dropped** (gated, no access). MMLU-Pro thinking deferred to tier 3.
- Tier-1 content per model (one engine load, modes switched via RPC):
  * PPL block: 10 long documents (InfiniteBench books) → prefix 32K/64K/128K, 2K scored continuation; modes
    clean, clean2, ctrl_identity, ctrl_numeric (batch-composition control), perm_once A, perm_once B at 32K;
    clean, clean2, perm_once A, perm_once B at 64K; clean, clean2, perm_once B at 128K; WikiText-2 50×2K windows
    (1K prefix / 1K scored) for all modes. perm_periodic is exercised in the generation block (needs decode steps).
  * Accuracy sanity: RULER niah_single_2 + qa_1, 5 + 5 items at 32K and 128K; modes clean, perm_once B, perm_periodic B.
  * Gate: identity ≡ clean within noise; every perm-mode ΔPPL inside the clean-vs-clean2 interval (paired bootstrap);
    RULER answers unchanged. Outputs → `docs/reindex_accuracy/`.

## exp3 tier 1 — DeepSeek-V3.2 (2026-08-29 09:20-09:55 UTC, 4.1 GPU-h)
- Long-book PPL (10 docs, 2K scored): 32K clean 1.0939 / clean2 1.0931 / identity 1.0942 / numeric 1.0923 / perm A 1.0925 / perm B 1.1002
  (B mean driven by one document, +0.057; 9/10 docs within ±0.004 — seed-repeat diagnostic running);
  64K clean 1.1357 / clean2 1.1372 / A 1.1360 / B 1.1371; 128K clean 1.0966 / clean2 1.0934 / B 1.0954.
- Token-level noise identical across modes incl. the identical rerun (~1.5 % of tokens shift >0.5 nats: MoE routing flips at near-ties).
- RULER niah+qa (32K,128K; 5+5 each): clean 16/20, perm_once B 16/20, perm_periodic B 16/20 — the same 4 qa misses in every mode;
  16/20 token streams bit-identical to clean under re-indexing (the 4 that differ diverge at steps 2-43, i.e. the noise regime).
- WikiText block re-run with the corrected 2K-prefix / 1K-scored geometry (first pass had no chunk boundary -> no events).

## exp3 tier 1 — all three models (2026-08-29 14:00 UTC) — PASS
- 77/78 (model × benchmark × prefix × mode) rows equivalent to the identical-rerun noise floor; the one exception (V3.2 impl B
  seed 7 at 32K) is a single high-variance document (seeds 8/9 and extra clean reruns confirm). RULER answers unchanged in
  every mode. `docs/reindex_accuracy/tier1_results.md`. Campaign total ≈ 65 GPU-h; final report `reports/final_20260829.md`.

## Repository / push status (2026-08-29 15:35 UTC)
- origin/main holds everything: analysis artifacts + reports (a28245c), and the 540 raw-trace run dirs
  (`docs/gpu_sweep/runs/` 135 + `docs/glm_sweep/runs/` 135 + 135 `_b` + 135 GLM-5) pushed in 26 batches of <=1.5 GB
  (01bac9d..66a323e) because GitHub rejects a single >2 GB push. Trees verified identical to the local full-history
  branch `gpu-campaign-raw-traces` (tag `raw-traces-full`). Shards are <=45 MB gz parts with SHARDS.json sha256s;
  reassemble with `cat traces/<name>.jsonl.gz.part* | gunzip` then `scripts/gpu/ingest_trace_fast.py`.

## exp3 tier 2 — DeepSeek-V3.2 (2026-08-29 16:58–21:25 UTC, ≈28 GPU-h) — PASS (PPL); stopped by the user after the PPL block
- 30 books × 32K/64K/128K × all 10 modes (clean, clean2, identity, numeric, perm_once A/B × seeds 7/8/9) + WikiText-2 (93 windows) + PTB (32
  windows): 38/40 rows equivalent to the rerun floor, all 30 long-book rows pass; the 2 flagged PTB impl-B rows are +0.1 % ≈ 1 SE with CIs
  including 0. Tier-1's seed-7 outlier vanishes at n = 30 (+0.0010 [−0.0016, +0.0039]). `docs/reindex_accuracy/tier2/`.
- Generation block reached `clean` 325/400 (niah/vt 25/25 everywhere, qa_1 18/17/14, LongBench-v2 14/25) and was stopped; no re-indexed
  generation rows for tier 2 (tier-1 RULER rows remain that evidence). RULER `vt` needs a 256-token cap with our prompt format (fixed in script).
- GLM-5.2 / GLM-5 tier 2 not started; `exp3_tier2.py --resume` continues from the JSON if ever resumed (≈10 h wall per model).

## exp3 tier 2 generation — DeepSeek-V3.2 (2026-08-31 02:35–05:54 UTC, ≈35 GPU-h) — partial
- 400 items (RULER niah_single_2 / niah_multikey_2 / vt / qa_1 × 32K/64K/128K × 25 + LongBench-v2 100): baseline 329/400,
  `perm_once_B` 331/400 (+2), `perm_periodic_B` 210/225 = baseline on the same items with 0 flips. Needle tasks 25/25 everywhere;
  all flips are qa_1/LongBench-v2 near-ties. 245/400 and 145/225 token streams bit-identical; 0 hook errors, 20,183 events.
- Interrupted by an external SIGTERM at 05:54 UTC; since 05:58 the node runs another project's Qwen2.5-72B API servers
  (`<NODE>/symbiosys_72b/`). Remaining: 175 periodic items, the `clean2` generation floor, GLM-5.2/GLM-5 tier-2 PPL —
  all `--resume`-able once 8 GPUs are free (TP=8 required; GLM FP8 does not fit on 4 GPUs).
