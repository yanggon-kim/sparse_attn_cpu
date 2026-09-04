# HANDOFF.history.md — archive for the DeepSeek sparse-selection locality study

Superseded, dated and historical material moved out of `HANDOFF.md` (the live file, beside this one).
Nothing loads this archive; it exists so that every number, decision and incident removed from the
live HANDOFF stays findable. Part 1 is the curated chronology; Part 2 is the pre-restructure
`HANDOFF.md` verbatim.

---

# Part 1 — chronology

## 2026-07-05 → 2026-08-11 — DeepSeek-V4 CPU campaign (ds4)
Full narrative, per-run wall-clocks and the corrections made along the way live in the previous
session's auto-memory file
`~/.claude/projects/-home-yanggon-99-personal-project-05-deepseek/memory/deepseek-v4-kv-locality-experiment.md`
(read-only history). Headline numbers survive in the live HANDOFF's Key-numbers block. Milestones:

- 2026-07-05: MoE logger `moe_log_selection` added to `ds4.c`; RULER 4K/8K/16K re-run as
  `runs/niah_single_2_{L}_moe_q2` with both traces (commits c8b80c2, d6a383a, cb26b59).
- 2026-07-06: 32K point (commit cb26b59 / versel 208c562) — KV adj 0.672, lift 10.53×; MoE 0.348
  (the dip was n=1 content noise).
- 2026-07-07: 64K point (sparse_attn_cpu ae8c6c9 / versel 9c85ee0, ~25 h 54 m) — KV adj 0.668,
  lift 21.37×; MoE back to 0.369. Needle correct in full once `-n 256` replaced `-n 128`.
- 2026-08-07 → 08-11: LongBench summarization collection, 36 runs, ~3.5 days, 15,073 decode steps,
  4.5 GB raw retained locally. Published sparse_attn_cpu fea819f + versel deaaaa9.
- Superseded claim: "retention plateaus at ~0.5" — a short-trace (≤128 step) artifact, corrected by
  the 3,019-step `longform_p16k_g4k_q2` run. The live gotcha ("decode ≥1–2 K steps") is its residue.
- Superseded claim: the 64K needle "failure" was a 128-token truncation artifact, not a retrieval
  failure (the model wrote 6 of 7 digits; the indexer selected the needle block in 88 % of cells).

### Two RULER run series (discrepancy log, 2026-08-27)
The *published* series is `niah_single_2_{4096,8192,16384,32768,65536}_moe_q2` (`-n 256`, 117–163
decode steps, KV+MoE traces). An older series `niah_single_2_{4096,8192,16384,40960,65536}_q2`
(`-n 128`) also exists: 40K adj 0.659 / lift 13.2×, 64K adj 0.670 / lift 21.4×. Quote the `_moe_q2`
series. The ramulator digest's §2 table still shows the old series — carried into the live HANDOFF's
open issues. `runs/niah_single_2_98304_q2` is an aborted 96K attempt (died in prefill at layer 34/43,
no outputs, 468 KB).

## 2026-08-23/24 — re-index correctness runs on ds4 for the ramulator v5 design
Run by the ramulator project with the `kv_perm_apply` hook (4K NIAH, `-n 128`, ~55 min each on 64
threads). Results are in the live Key-numbers block. Raw runs (90 MB, not ours to delete):
`<HOME>/0007_26summer/01_ramulator/tmp_v5_gather/ds4_reindex/`; write-up
`ramulator2/00_doc/01_design/v5/v5_reindex_correctness_ds4.md`.

## 2026-08-27 — owner-onboarding cross-check (no reruns)
Every published V4 number re-read from its source artifact and confirmed against the Vercel/GitHub
write-ups: RULER adj overlap 0.868/0.790/0.718/0.672/0.668 and lift 1.72/2.92/5.72/10.53/21.37× from
`runs/niah_single_2_{L}_moe_q2/analysis/metrics_run_summary.json`
(`overall_adjacent_overlap_mean`, `overall_locality_lift_mean`); hot-set A@99 79.5/65.7/47.5/31.4/
20.4 % of pool (pool 1,030/1,912/4,095/8,039/16,393; per-layer range 9.9–38.4 % at 64K) from
`analysis/hotset_coverage.json` (`A99_pct_mean`, `A99_pct_range`); LongBench aggregate (36 runs)
multi_news 0.914 / gov_report 0.755 / qmsum 0.732 adj, pooled 0.801, A@99 92.0/78.7/61.4 % from
`analysis_longbench/longbench_aggregate.json` (identical to the published copy in
`sparse_attn_cpu/docs/longbench_sweep/`). All matched.

Also surveyed that day: `ds4.c` carries +410/−8 uncommitted lines (instrumentation + the v5 re-index
hook); `build_longbench_prompts.py` verified byte-identical on re-run (deterministic, seed 42).

## 2026-08-27/28 — GPU campaign package (the spec that was handed to the rented node)
`01_github/sparse_attn_cpu/docs/00_doc/GPU_CAMPAIGN.md` was the top-level file to hand to the agent
on the 8-GPU node (its §8 held the ready-to-paste prompt); it links `exp0_environment.md` (pin/install
vLLM, models, TP8 smoke, hook unit checks), `exp1_dsv32_gather_index.md` (V3.2 hook at
`flashmla_sparse.py:838 forward_mqa`, adapter to the ds4 run-dir schema so ingest/validate/analyze run
unchanged, 8K–128K ladder, bf/ld run kinds), `exp2_glm_gather_index.md` (GLM-5.2 index-share handling,
GLM-5 DSA check), `exp3_reindex_accuracy.md` (block-granular A vs entry-granular B re-index,
ds4-mirrored modes/controls, benchmark set + official-number gate,
`docs/reindex_accuracy/{results.csv,per_item.jsonl,README.md}`). Citations verified against the vLLM
checkout `5559679` (`<HOME>/0007_26summer/03_vLLM/vllm`). 2026-08-28:
`docs/00_doc/locality_metrics.md` = the metric reference (formulas from the code, JSON keys, V3.2
adapter caveats, V4 worked table), linked from GPU_CAMPAIGN §2 and exp1 §4; committed + pushed.

### Prerequisites (old §7b; SATISFIED 2026-08-29)
Rental account/credentials for an 8-GPU node with ≥ ~1 TB HBM (H200/B200-class; ~1,000 GPU-h budget)
and the go date; confirmation of the model (`deepseek-ai/DeepSeek-V3.2` native FP8, plus GLM-5/5.2 if
wanted); HF token if any target repo is gated; the session URL for the commit trailer. Everything else
(vLLM pin, hook points, schema adapter, ladder, benchmarks, re-index implementations, gates) was
specified in `GPU_CAMPAIGN.md` + `exp0..exp3`; the GPU agent resolved the "VERIFY ON THE NODE" list
(GPU_CAMPAIGN.md §7) itself. Traces stayed on the node's NVMe (1–2 TB); only analysis artifacts came
back into `sparse_attn_cpu/docs/`.

## 2026-08-28 — v6 trace exports (V4) and paper hotness data
- v6 V4 exports (sparse_attn_cpu 3c63760): `work/experiment/exports/v6/<run_id>.npz` +
  `.manifest.json` for the 42 V4 runs (5 RULER `_moe_q2`, 36 `lb_*`, `longform_p16k_g4k_q2`; 18,799
  steps, 102 MB; uint16, k = 512, ratio = 4, 21 CSA layers) + `retention_curves.json` (9 families).
  Verified: adjacent overlap recomputed from the 64K npz = 0.6682 = `metrics_run_summary.json`.
  Two multi_news runs (s0, s1) have rows with `valid_k < k` (pool < 512).
- fig:hotness data (sparse_attn_cpu fa69406): `docs/00_doc/data/hotness_coverage_{64k,32k,16k}.csv`
  (mean/min/max over 21 CSA layers of `coverage_by_pool_pct`, RULER `_moe_q2`), `hotness_retention.csv`
  (64K RULER lags 1–64 + `longform_p16k_g4k_q2` lags 1–2048 with working-set ratios), `README.md`.
  These V4 files were later renamed `*_v4.csv` when the GPU data replaced them.

## 2026-08-29 — GPU campaign delivered (8× B200, vLLM 0.28.0)
Pull: sparse_attn_cpu origin/main 758e78f → 1c2a3f2 = 30 commits (a28245c analysis+reports, 26
raw-trace batches, 3d8153c/20ed678 node artifacts, 1c2a3f2 smoke run dirs; ~35 GB of gz shards, plain
git blobs, no LFS). **Incident:** a `git pull` of that size must run detached — an interrupted checkout
leaves untracked partial files that block the merge; the 23 GB partial copy from that incident sits in
`work/experiment/exports/partial_checkout_20260829/` and is safe to delete.

Environment: 8× B200 (183 GB, SM 10.0, driver 595.91.07, CUDA 13.0), vLLM 0.28.0 tag 2cf0a69 (contains
pin 5559679), torch 2.13.0+cu130, flashinfer 0.6.16.post3; backend FLASHINFER_MLA_SPARSE (not
FLASHMLA) + TRTLLM fp8 MoE. exp0/exp1 smoke/exp1 ladder/exp2 smoke/exp2/exp3 tier 1 all PASS; ~65 of
1,000 GPU-h, 17.5 h wall. Reports `docs/00_doc/reports/{exp0,exp1_smoke,exp1_ladder,exp2_smoke,exp2,
exp3_clean,exp3_reindex,exp3_stop,final}_2026082x.md`, `node.json`, `docs/00_doc/PROGRESS.md`.

VERIFY deviations resolved on the node:
- GLM-5 HAS DSA (78 layers, no index sharing) — answered the open pre-node question.
- GLM-5.2 `index_skip_topk_offset` = 3 (the doc said 2) → 21 computing layers {0,1,2,6,10,…,74},
  `index_topk_freq` 4; shared layers reproduce the producer's set bit-for-bit → variants (a) all-78-layer
  view and (b) computing-only differ ≤ 0.01 overlap.
- GLM checkpoints are BF16 → used the official `-FP8` repos.
- Decode is NOT run-to-run deterministic (11 probes, `docs/reindex_accuracy/determinism_probes.md`) →
  exp3 bit-identity replaced by teacher-forced PPL vs an identical-rerun noise floor (user decision).
- GPQA gated → dropped; MMLU-Pro deferred → no official-number gate.
- exp3 tier 2 run for V3.2 only; per-index scores not traced (score columns NaN); fast analysis twins
  `scripts/gpu/*_fast.py`.
- **KV latent cache is bf16 (576 B rows = 1152 B/token/layer + 132 B indexer), not the 656-B fp8
  layout assumed in the design docs** — the paper keeps the fp8 byte accounting with a footnote.

Campaign data shape: 540 run dirs (v32 135, glm52 135, glm52_b 135 analysis-only, glm5 135); per model
28/28/28/27/24 runs at 8K/16K/32K/64K/128K = bf 20/20/20/19/16 (RULER niah+qa 8, LongBench v1 4
(≤ 32K), v2 4, InfiniteBench 4; decode 3–512 steps, median ~20–30) + ld 8/rung (2048 forced steps;
LB-v1 summarization, IB En.QA/En.Sum). Not delivered by the campaign: npz exports, updated
`docs/00_doc/data/` files (regenerated by us), Git LFS; `retention_lag2048` is NaN for every run
(2048-step decodes give lags ≤ 1024).

### v6 GPU exports (2026-08-29, ours)
`scripts/export_v6_gpu_traces.py` (copy in repo `scripts/`), driver `exports/run_v6_gpu_export.sh`,
log `exports/v6_gpu_export.log` (`--skip-existing` resumes). Pipeline per run: every sha256 in
`SHARDS.json` verified (parts + manifests + analysis), parts concatenated to
`exports/gpu/<run_id>/traces/<name>.jsonl.gz` (raw JSONL never written; its sha256 checked on the
decompressed stream), records parsed straight to the npz — the 256 M-row `selected_kv.parquet` of the
CPU pipeline is skipped on purpose. Cost ~50 s and 11 GB RSS per 2048-step run; 4 workers pinned
(`taskset -c 40-47`, OMP 1). Result (35 min, 420 run dirs, 0 sha256 or export failures): `v6_v32/`
146 npz / 9.4 GB (per rung 8K–128K: bf 20/20/20/19/16 = 0.15–0.25 GB each, ld 8/rung =
1.1/1.5/1.7/1.8/2.1 GB, + 11 smoke); `v6_glm/` 415 npz / 19.1 GB (glm52 135 + 5 smoke, glm52b 135 + 5,
glm5 135; ld rungs 0.5–2.8 GB); dtypes uint16 453 / uint32 108 (128K and some 64K ld). Reassembled gz
shards `exports/gpu/` 33 GB. Sanity: 64K ld npz recomputed layer-0 adjacent overlap 0.40 (layer 0 is
the least local; run mean 0.79); GLM-5.2 `_b` npz == (a) npz restricted to the 21 computing layers,
shared layer 3 == producer layer 2 bit-for-bit, recomputed `_b` adjacent overlap 0.7409 vs campaign
0.741. Reassemble a shard by hand: `cat traces/x.jsonl.gz.part* | gunzip`.

### exp3 tier 1 detail (2026-08-29)
Teacher-forced PPL, 10 InfiniteBench books (32K/64K/128K prefix, 2K scored) + WikiText-2 (50 × 2K);
modes clean / clean2(-4) / ctrl_identity / ctrl_numeric / perm_once_A (64-token block + block-table) /
perm_once_B (row permutation, block table untouched = the design) / perm_periodic_B (RULER generation,
every 4 steps 10 % swapped). 77/78 mode rows within the identical-rerun noise floor; the exception was
V3.2 impl B seed 7 @32K (one document, cleared by seeds 8/9). V3.2 books impl B dPPL −0.0009/+0.0014/
−0.0012 vs floor −0.0008/+0.0015/−0.0031; RULER niah+qa accuracy unchanged in every mode (V3.2 16/20,
GLM-5.2 18/20, GLM-5 20/20).

## 2026-08-31 — exp3 tier 2 (V3.2) pulled in
Commits d2a6d8f (PPL block), 4e7468a (START_HERE.md), 1759b17 (generation block); 25 files, ~230 MB
permlog shards. Headline numbers are in the live Key-numbers block. Detail retired from the live file:
tier 1's single exception (impl B seed 7 @32K) disappears at n = 30 (+0.0010 [−0.0016, +0.0039]); the
2 flagged tier-2 rows are PTB impl-B seeds 7/9 (+0.006/+0.008 on PPL 5.80, ≈1 SE, CIs include 0, and a
batch-composition control reproduces the token p90 — an under-powered mean test, not a re-index
effect); per-token |Δlogprob| p90 identical across all modes (0.048–0.051 / 0.063–0.067 / 0.089–0.094
nats at 32K/64K/128K); all 10 generation flips are qa_1 / LongBench-v2 near-ties. The run was
interrupted by an external SIGTERM 2026-08-31 05:54 UTC (node taken by four external
`Qwen2.5-72B-Instruct` vLLM servers on GPUs 0–7 at 05:58 UTC). GPU-h: tier 2 ≈ 63; campaign cumulative
≈ 128 of 1,000.

## Older decisions (moved out of "Recent decisions")
- MoE context-scaling plot uses a 0-based lift axis — autoscale exaggerated a 16 → 14.9× dip into a
  visible drop, and the point of the plot is that MoE locality is context-independent (2026-07-06).

---

# Part 2 — `HANDOFF.md` as it stood before the 2026-09-04 restructure (verbatim, 332 lines)

# HANDOFF.md — DeepSeek sparse-selection locality study

Written 2026-08-25 for an agent with no memory of prior sessions; refined 2026-08-27/28 by the
`deepseek-owner` agent (onboarding survey: repo states, cross-checks, ds4 hook, consumers). Read this first, then
`~/.claude/projects/-home-yanggon-99-personal-project-05-deepseek/memory/deepseek-v4-kv-locality-experiment.md`
(the auto-memory file: full history of runs, numbers, gotchas — the single most important doc).

## 1. Purpose
Measure the **temporal locality of sparse selections** inside DeepSeek-V4 (and, next, V3.2 / GLM-5) for
an architecture paper (ISCA/HPCA/ASPLOS target): which KV entries the CSA/DSA "lightning indexer" gathers
per token (top-k), and which MoE experts the router picks, and how much those sets overlap across
consecutive decode steps. Outputs feed a hot-set/prefetch cache design. Three result families:
R1 KV top-k locality, R2 MoE routing locality, R3 hot-set coverage (% of pool to cover 99% of accesses).

## 2. Layout (absolute paths)
- `work/experiment/` — the whole pipeline. `runs/<id>/{traces,outputs,analysis,prompts,logs}`;
  `scripts/`; `prompts/`; `benchmark/{RULER,longbench}`; `analysis_moe/` (RULER-sweep plots);
  `analysis_longbench/` (aggregate + plots). `EXPERIMENT_SUMMARY.md` = RULER write-up.
- `work/ds4/` — antirez/ds4 C engine (HEAD 80ebbc3) **+ uncommitted local changes in `ds4.c` only**
  (+410/−8 lines, 2026-08-27 survey): (a) instrumentation — KV logger `indexer_log_selection`, MoE
  logger `moe_log_selection` (= `sparse_attn_cpu/docs/ds4_instrumentation.patch`, still reverse-applies
  clean); (b) the **v5 re-index correctness hook** added 2026-08-24 by the ramulator project
  (`kv_perm_apply`, env `DS4_PERM_MODE/SEED/PERIOD/FRAC/HCA/LOG/IDENTITY`; default MODE=0 = inert).
  Patch copy: `ramulator2/examples/v5_ds4_reindex_correctness/ds4_reindex_hook.patch` (absolute-path
  headers, needs `-p7`; it is 2 lines behind ds4.c — lacks the `DS4_PERM_IDENTITY` control).
  Keep both; never revert. Binaries (untracked): `ds4` (built 08-24 10:15, hook without IDENTITY),
  `ds4_new` (current source), `ds4_nofm` (current source, `-fno-fast-math`). `make cpu` rebuilds `ds4`.
- `work/models/` — `DeepSeek-V4-Flash-IQ2XXS-...gguf` (81 GB, the workhorse);
  `v32/DeepSeek-V3.2-UD-TQ1_0.gguf` (151 GB, dense, NO DSA indexer — useless for sparse data);
  `v32_4L/DeepSeek-V3.2-4Layers-Q8_0.gguf` (16 GB, arch deepseek32 WITH indexer; dev-only).
- `work/llama.cpp/` — pinned to 683f0c72e (2026-07-09) with local patches (see §5). Not pushed anywhere.
- `01_github/sparse_attn_cpu/` — published repo (git@github.com:yanggon-kim/sparse_attn_cpu). HEAD = 1759b17
  (fast-forwarded 2026-08-31: exp3 tier-2 commits from the GPU node — d2a6d8f PPL block, 4e7468a START_HERE.md,
  1759b17 generation block; 25 files, ~230 MB permlog shards; before them 73db862 our v6/analysis exports,
  1c2a3f2 GPU-campaign raw traces). Holds `docs/00_doc/GPU_CAMPAIGN.md`, `exp0..exp3`, and a path-scrubbed copy
  of this HANDOFF.md (refresh it when this file changes and a commit is due).
- `01_github/versel_distribute/` — static site → https://versel-distribute.vercel.app. Local HEAD deaaaa9,
  clean, **6 commits behind origin/main (e00b92a, user's Part X/Part III-of-05_ramulator edits; none
  touch `04_sparse_attn/02_part3*`, `03_part4*`; `index.html` 1 line)** — rebase before any push.
  Has its own `CLAUDE.md` (READ IT): plain HTML, relative links only, every report self-contained,
  `index.html` is the only nav; "apply the edit rule" = add `<a class="report-link">` for new pages.
- `papers/`, `00_doc/`, `study_deepseek_v4_vllm/`, `PLAN.md`, `RESULTS.md` — background docs.

## 3. Current status
**Works (published):**
- RULER 4K/8K/16K/32K/64K sweep on ds4 (`runs/niah_single_2_{L}_moe_q2`): KV adj overlap
  0.868/0.790/0.718/0.672/0.668, lift 1.7/2.9/5.7/10.5/21.4×; MoE learned 0.33–0.38 (~16× random,
  context-independent); hash layers 0–2 ≈ random. Hot-set A@99 80%→20%.
- LongBench summarization (36 runs, `runs/lb_{multi_news,gov_report,qmsum}_s{0..11}_q2`): all 36 rc=0,
  validate PASS, 15,073 decode steps, 4.5 GB raw retained locally. Aggregate:
  `work/experiment/analysis_longbench/longbench_aggregate.json`. Real tasks lie ON the RULER curve.
- Long-decode run `runs/longform_p16k_g4k_q2` (3,019 steps) — retention keeps decaying; short traces
  (<~500 steps) falsely show a plateau. Always decode ≥1–2K steps for retention curves.
- Vercel reports Part III (KV, `04_sparse_attn/02_part3_cpu_kv_locality/`) and Part IV (MoE,
  `03_part4_moe_locality/`) include RULER + LongBench sections. Parts I/II/V/VI/VII are the user's own.
- **Re-index correctness runs for the ramulator v5 design (2026-08-23/24, run by that project with
  the hook above, 4K NIAH, `-n 128`, ~55 min each on 64 threads):** identity control bit-identical to
  baseline; one-time random row permutation 127/128 tokens identical, needle retrieved, selection Jaccard
  0.967 in original-index space; periodic swaps and the `-fno-fast-math` control both diverge at step 35
  → residual = summation-order noise. Raw runs (90 MB, not ours to delete):
  `<HOME>/0007_26summer/01_ramulator/tmp_v5_gather/ds4_reindex/`; write-up
  `ramulator2/00_doc/01_design/v5/v5_reindex_correctness_ds4.md`.
- Two RULER run series exist: the *published* `niah_single_2_{4096,8192,16384,32768,65536}_moe_q2`
  (`-n 256`, 117–163 steps, KV+MoE traces) and the older `niah_single_2_{4096,8192,16384,40960,65536}_q2`
  (`-n 128`; 40K adj 0.659 / lift 13.2×, 64K 0.670 / 21.4×). Quote the `_moe_q2` series; the ramulator
  digest's §2 table still shows the old series (see §8). `runs/niah_single_2_98304_q2` is an **aborted**
  96K attempt (died in prefill at layer 34/43, no outputs, 468 KB) — the runner will redo it if asked.

**Cross-checked 2026-08-27 (owner onboarding, no reruns):** RULER adj overlap
0.868/0.790/0.718/0.672/0.668 and lift 1.72/2.92/5.72/10.53/21.37× = `runs/niah_single_2_{L}_moe_q2/
analysis/metrics_run_summary.json` (`overall_adjacent_overlap_mean`, `overall_locality_lift_mean`);
hot-set A@99 79.5/65.7/47.5/31.4/20.4 % of pool (pool 1,030/1,912/4,095/8,039/16,393; per-layer range
9.9–38.4 % at 64K) = `analysis/hotset_coverage.json` (`A99_pct_mean`, `A99_pct_range`); LongBench
aggregate (36 runs) multi_news 0.914 / gov_report 0.755 / qmsum 0.732 adj, pooled 0.801, A@99
92.0/78.7/61.4 % = `analysis_longbench/longbench_aggregate.json` (identical to the published copy in
`sparse_attn_cpu/docs/longbench_sweep/`). All match the Vercel/GitHub write-ups.

- **v6 trace exports for the ramulator policy study (2026-08-28, sparse_attn_cpu 3c63760):**
  `work/experiment/exports/v6/<run_id>.npz` + `.manifest.json` for the 42 V4 runs (5 RULER `_moe_q2`,
  36 `lb_*`, `longform_p16k_g4k_q2`; 18,799 steps, 102 MB total; all uint16, k = 512, ratio = 4, 21 CSA
  layers) + `retention_curves.json` (9 families). Format: `sparse_attn_cpu/docs/00_doc/v6_export_format.md`.
  Regenerate: `python3 work/experiment/scripts/export_v6_traces.py` (~4 min; `--only <run_id>`; log
  `exports/v6_export.log`). Verified: adjacent overlap recomputed from the 64K npz = 0.6682 =
  `metrics_run_summary.json`. **Gotcha: `n_comp = (pos + 1) // ratio`, not `pos // ratio`** (asserted
  against the jsonl for every run). Two multi_news runs (s0, s1) have rows with `valid_k < k` (pool < 512).
  Consumer: ramulator-owner `examples/v6_policy/traces.py` on branch v6.

- **fig:hotness data for the paper (2026-08-28, sparse_attn_cpu fa69406, pushed):** `docs/00_doc/data/
  hotness_coverage_{64k,32k,16k}.csv` (mean/min/max over 21 CSA layers of `coverage_by_pool_pct`, RULER
  `_moe_q2` runs), `hotness_retention.csv` (64K RULER lags 1–64 + `longform_p16k_g4k_q2` lags 1–2048, with
  working-set ratios), `README.md` (provenance, units, V3.2-replaces-in-same-schema note). Regenerate:
  `python3 work/experiment/scripts/export_hotness_fig_data.py` (copy in repo `scripts/`). Consumer: paper-owner.

- **GPU campaign delivered (2026-08-29; sparse_attn_cpu origin/main pulled 758e78f → 1c2a3f2 = 30 commits: a28245c
  analysis+reports, 26 raw-trace batches, 3d8153c/20ed678 node artifacts, 1c2a3f2 smoke run dirs; ~35 GB of gz shards,
  plain git blobs, no LFS; a `git pull` of that size must run detached — an interrupted checkout leaves untracked
  partial files that block the merge; the 23 GB partial copy of that incident sits in `work/experiment/exports/
  partial_checkout_20260829/`, safe to delete):**
  8x B200 (183 GB, SM 10.0, driver 595.91.07, CUDA 13.0), vLLM 0.28.0 tag 2cf0a69 (contains pin 5559679), torch 2.13.0+cu130,
  flashinfer 0.6.16.post3; backend FLASHINFER_MLA_SPARSE (not FLASHMLA) + TRTLLM fp8 MoE; **KV latent cache is bf16
  (576 B rows = 1152 B/token/layer + 132 B indexer), not the 656-B fp8 layout assumed in the docs**. exp0/exp1 smoke/
  exp1 ladder/exp2 smoke/exp2/exp3 tier 1 all PASS; ~65 of 1,000 GPU-h, 17.5 h wall. Reports `docs/00_doc/reports/
  {exp0,exp1_smoke,exp1_ladder,exp2_smoke,exp2,exp3_clean,exp3_reindex,exp3_stop,final}_2026082x.md`, `node.json`,
  `docs/00_doc/PROGRESS.md`. VERIFY deviations: GLM-5 HAS DSA (run, 78 layers, no sharing); GLM-5.2 `index_skip_topk_offset`
  = 3 (doc said 2) -> 21 computing layers {0,1,2,6,10,..,74}, `index_topk_freq` 4, shared layers reproduce the producer's set
  bit-for-bit -> variants (a) all-78-layer view and (b) computing-only differ <= 0.01 overlap; GLM checkpoints are BF16 ->
  official `-FP8` repos; decode is NOT run-to-run deterministic (11 probes, `docs/reindex_accuracy/determinism_probes.md`)
  -> exp3 bit-identity replaced by teacher-forced PPL vs identical-rerun noise floor (user decision); GPQA gated -> dropped;
  exp3 tier 2 run for V3.2 only (see the tier-2 bullet below; GLM tier 2 / tier 3 not run); per-index scores not
  traced (score columns NaN); fast analysis twins `scripts/gpu/*_fast.py`.
  - **Data:** `docs/gpu_sweep/` (V3.2: R1/R2/R3/accuracy CSVs with per-run rows + CI columns, `sweep_v32.json`,
    `gpu_sweep_summary.md`, 4 PNGs) and `docs/glm_sweep/` (GLM-5.2 a/b + GLM-5, same layout, `side_by_side.png`);
    per run `docs/<sweep>/runs/<run_id>/{req,meta,run_manifest,model_config}.json, outputs/generations.jsonl,
    analysis/{metrics_run_summary,hotset_coverage,extended_retention,moe_metrics_run_summary}.json + parquets,
    traces/indexer_trace.jsonl.gz.partNN (+ moe_trace), SHARDS.json (sha256 per file, parts <= 45 MB)`. 540 run dirs
    (v32 135, glm52 135, glm52_b 135 analysis-only, glm5 135); per model 28/28/28/27/24 runs at 8K/16K/32K/64K/128K =
    bf 20/20/20/19/16 (RULER niah+qa 8, LongBench v1 4 (<= 32K), v2 4, InfiniteBench 4; **decode 3-512 steps, median
    ~20-30**) + ld 8/rung (2048 forced steps; LB-v1 summarization, IB En.QA/En.Sum). Reassemble: `cat traces/x.jsonl.gz.part*
    | gunzip`. `docs/reindex_accuracy/{README.md,results.csv,per_item.jsonl,tier1_results.md,tier1_{v32,glm52,glm5}.json,
    unit_test_v32.json,determinism_probes.md,official_numbers.json,paper_accuracy_section.md,reindex_ppl_delta.png}`.
    **v6 exports produced by us (2026-08-29, see the next bullet).** Not delivered by the campaign: npz exports,
    updated `docs/00_doc/data/` files (now regenerated), Git LFS (plain blobs); `retention_lag2048` is NaN for every
    run (2048-step decodes give lags <= 1024).
  - **v6 GPU exports (2026-08-29, `scripts/export_v6_gpu_traces.py`, copy in repo `scripts/`):** one `<run_id>.npz` +
    `.manifest.json` per campaign run in `work/experiment/exports/v6_v32/` (DeepSeek-V3.2, incl. the 10 smoke dirs) and
    `exports/v6_glm/` (GLM-5.2 all-layer `<id>.npz`, computing-only `<id>_b.npz` derived from the same trace by layer
    filter, GLM-5), plus `retention_curves.json` in each (families `<tag>_{ld,bf,ruler}_<rung>`, tags v32/glm52/glm52b/
    glm5, lags to 1024 from `extended_retention.json`; `scripts/export_gpu_analysis_data.py`). Schema =
    `v6_export_format.md`: ratio 1, k 2048, `n_comp = pos + 1` (asserted per record), uint16 below 65536 else uint32,
    manifest carries `run_kind` (bf|ld), `decode_steps`, `cache_layout_note` (bf16 1152 B/token/layer + 132 B), source
    run dir + SHARDS.json sha. Pipeline per run: every sha256 in `SHARDS.json` verified (parts + manifests + analysis),
    parts concatenated to `exports/gpu/<run_id>/traces/<name>.jsonl.gz` (raw JSONL never written; its sha256 is
    checked on the decompressed stream), records parsed straight to the npz — the 256 M-row `selected_kv.parquet` of the
    CPU pipeline is skipped on purpose. Cost: ~50 s and 11 GB RSS per 2048-step run; 4 workers pinned (`taskset -c
    40-47`, OMP 1); driver `exports/run_v6_gpu_export.sh`, log `exports/v6_gpu_export.log` (`--skip-existing` resumes).
    Sanity: 64K ld npz recomputed layer-0 adjacent overlap 0.40 (layer 0 is the least local; run mean 0.79); GLM-5.2
    `_b` npz == (a) npz restricted to the 21 computing layers, shared layer 3 == producer layer 2 bit-for-bit, recomputed
    `_b` adjacent overlap 0.7409 vs campaign 0.741. **Result (2026-08-29, 35 min, 420 run dirs, 0 sha256 or export
    failures):** `v6_v32/` 146 npz / 9.4 GB (per rung 8K–128K: bf 20/20/20/19/16 = 0.15–0.25 GB each rung, ld 8/rung =
    1.1/1.5/1.7/1.8/2.1 GB, + 11 smoke), `v6_glm/` 415 npz / 19.1 GB (glm52 135 + 5 smoke, glm52b 135 + 5, glm5 135; ld
    rungs 0.5–2.8 GB); dtypes uint16 453 / uint32 108 (128K and some 64K ld). Reassembled gz shards `exports/gpu/` 33 GB.
    Consumer paths for ramulator-owner: `work/experiment/exports/v6_v32/{<run_id>.npz,<run_id>.manifest.json,
    retention_curves.json}` and `exports/v6_glm/` (same; `<id>_b.npz` = computing-only view).
  - **`docs/00_doc/data/` regenerated (ld runs only, 8 per rung):** `hotness_coverage_{16k,32k,64k,128k}.csv` (V3.2;
    band pooled over run × layer), `hotness_coverage_{glm52,glm5}_*.csv`, `hotness_retention{,_glm52,_glm5}.csv` (ld 64K
    + 128K, lags 1–1024), `gpu_headline_by_kind.csv` (tag × rung × kind), `retention_curves_gpu.json`,
    `hotness_provenance.json`; V4 files renamed `*_v4.csv`; README rewritten. **ld-only headline, V3.2 64K / 128K:**
    adj overlap 0.793 / 0.799, lift 24.8× / 43.6×, ret@64 0.643 / 0.627, ret@512 0.478 / 0.473, ret@1024 0.351 / 0.344,
    cov10 0.894 / 0.948, A@99 24.8 % / 17.8 % (N 65K / 113K); GLM-5.2(a) 0.753 / 0.751, ret@64 0.546 / 0.530, ret@1024
    0.303 / 0.291, cov10 0.834 / 0.894, A@99 32.2 / 25.2 %; GLM-5 0.774 / 0.764, ret@64 0.565 / 0.562, ret@1024 0.316 /
    0.298, cov10 0.853 / 0.910, A@99 31.9 / 23.8 %. 16K / 32K ld A@99: V3.2 66.5 / 46.6, GLM-5.2 68.6 / 50.4, GLM-5 65.7 / 49.2.
  - **Key numbers V3.2 (rung mean over bf+ld, 95 % bootstrap CI over runs, `gpu_sweep_summary.md`; k = 2048 tokens,
    61 layers, ratio 1):** adj overlap 8K-128K **0.843/0.790/0.751/0.741/0.723**, lift 2.56/5.27/10.2/21.9/41.5x, recency
    0.49-0.31; ret@64 0.722/0.627/0.584/0.602/0.609; ld ret@512 0.580/0.508/0.487/0.478/0.473, ret@1024 0.433/0.388/
    0.369/0.351/0.344; A@99 **65.0/44.7/30.3/16.7/9.7 %** (pool 6.6K/14K/28K/61K/119K), cov@10 % (MEASURED_TOP10)
    0.304/0.565/0.788/0.935/0.978; MoE 0.24-0.26 (7.7-8.3x, 58 learned layers, 8/256) context-independent.
    **Caveat (mine, from the per-run CSV rows):** bf and ld runs differ a lot at 64K/128K — adj overlap bf 0.719/0.684
    vs ld 0.793/0.799; A@99 bf 13.4/5.7 % vs ld 24.8/17.8 %; cov10 bf 0.952/0.993 vs ld 0.894/0.948 — the short bf
    decodes (median 20 steps at 64K/128K) inflate hot-set concentration exactly as HANDOFF §3 warns. Quote hot-set /
    retention numbers from the ld (2048-step) rows or state the mix; overlap/lift are fine either way.
  - **GLM (same ladder):** adj overlap GLM-5.2(a) 0.846/0.789/0.750/0.722/0.702, (b) 0.847/0.791/0.745/0.714/0.692,
    GLM-5 0.849/0.791/0.753/0.730/0.715; lift 2.6->40x; A@99 GLM-5.2 62.1/42.7/29.1/18.0/11.5, GLM-5 61.5/42.5/28.9/18.6/
    11.0 (ld-only 64K/128K: 32.2/25.2 and 31.9/23.8); cov10 0.30/0.57/0.80/0.92/0.96; ld ret@1024 GLM-5.2 0.44->0.29,
    GLM-5 0.44->0.30; MoE 0.28-0.31 (9-10x, 75 learned layers). Generality: all three within +-0.02 overlap / +-2 pt A@99.
  - **exp3 tier 1:** teacher-forced PPL, 10 InfiniteBench books (32K/64K/128K prefix, 2K scored) + WikiText-2 (50 x 2K);
    modes clean/clean2(-4)/ctrl_identity/ctrl_numeric/perm_once_A (64-token block + block-table)/perm_once_B (row
    permutation, block table untouched = the design)/perm_periodic_B (RULER generation, every 4 steps 10 % swapped).
    **77/78 mode rows within the identical-rerun noise floor** (exception V3.2 impl B seed 7 @32K, one document, cleared
    by seeds 8/9); e.g. V3.2 books impl B dPPL -0.0009/+0.0014/-0.0012 vs floor -0.0008/+0.0015/-0.0031; RULER
    niah+qa accuracy unchanged in every mode (V3.2 16/20, GLM-5.2 18/20, GLM-5 20/20). No official-number gate (no
    published PPL; MMLU-Pro/GPQA deferred/dropped).
  - **exp3 tier 2 (V3.2 only; PPL 2026-08-29 + generation addendum 2026-08-31; pulled here 2026-08-31, commits
    d2a6d8f/4e7468a/1759b17):** PPL block COMPLETE — 30 long books × {32K,64K,128K} × 10 modes (clean, clean2,
    ctrl_identity, ctrl_numeric, perm_once_A/B × seeds 7/8/9) + WikiText-2 (93 windows) + PTB (32) = 2,150 rows,
    1,505 re-index events, 0 hook errors; **38/40 mode rows within the identical-rerun floor and ALL 30 long-book
    rows pass** — tier 1's single exception (impl B seed 7 @32K) disappears at n = 30 (+0.0010 [−0.0016, +0.0039]);
    the 2 flagged rows are PTB impl-B seeds 7/9 (+0.006/+0.008 on PPL 5.80, ≈1 SE, CIs include 0, batch-composition
    control reproduces the token p90 — under-powered mean test, not a re-index effect). Baseline PPL 1.3058/1.2678/
    1.3674 at 32K/64K/128K; per-token |Δlogprob| p90 identical across all modes (0.048–0.051/0.063–0.067/0.089–0.094
    nats). Re-indexed GENERATION (400 items: RULER niah_single_2/niah_multikey_2/vt/qa_1 × 3 lengths × 25 +
    LongBench-v2 100): clean 329/400, perm_once_B **331/400 (+2**, all 10 flips are qa_1/LB-v2 near-ties; niah/vt
    25/25 at every length in every mode), perm_periodic_B partial 225/400 → **210 vs 210 baseline on the same items,
    0 flips**. Interrupted by an external SIGTERM 2026-08-31 05:54 UTC (node taken by another workload); NOT run:
    last 175 perm_periodic_B items, the clean2 generation floor, GLM-5.2/GLM-5 tier 2, tier 3. Resume:
    `scripts/gpu/exp3_tier2.py --resume` (TP8 required for reduction-order comparability). Artifacts:
    `docs/reindex_accuracy/tier2/{tier2_results.md,results.csv,per_item.jsonl,tier2_v32.json,permlogs/}`,
    `docs/reindex_accuracy/START_HERE.md` (the exp3 study's own onboarding doc), `docs/00_doc/reports/
    exp3_tier2_20260829.md` (+ 2026-08-31 addendum §8–11), `paper_accuracy_section.md` tier-2 addendum with
    ready-to-adapt Tables 3/4. GPU-h: tier 2 ≈63; campaign cumulative ≈128 of 1,000.
  - **Consumers to notify:** ramulator `MEASURED_OVERLAP`/`MEASURED_TOP10` (V3.2 values above replace V4), v6 policy
    study (needs the npz exports), evaluation §6 hit-ratio inputs, paper §3.3 (generality claim in exp2 report §6) and
    the correctness paragraph (`paper_accuracy_section.md` — now with the tier-2 addendum + Tables 3/4: 30-doc PPL
    equivalence and task-accuracy-under-re-indexing table), `ref_dsv4_kv_locality_study.md` digest.

**In progress / parked:**
- DeepSeek-V3.2 on CPU: engine patched and loads, but the only CPU-fittable GGUF (Unsloth TQ1_0) is
  dense-stripped; DSA-preserving GGUFs (`sszymczyk/DeepSeek-V3.2-*-light-GGUF`) are 404–714 GB.
  Standalone DSA tracer (`work/llama.cpp/examples/eval-callback/eval-callback.cpp`, target
  `llama-eval-callback`) works at short context but crashes on prompts >~256 tokens
  (`ops.cpp set_rows: leaf_90 uninitialized`) while `llama-server` handles the same prompt fine.
  Fix path: hook the trace into llama-server's decode path instead. PARKED — user pivoted to GPU.
- **GPU campaign: EXECUTED and delivered 2026-08-29/31** (8× B200, vLLM 0.28.0; see the "GPU campaign
  delivered" bullet above for the full result set). §7/§7b below are kept as the historical package spec.
  **Open GPU work, all needing 8 free GPUs at TP8** (the node was taken over by another workload
  2026-08-31 05:58 UTC — four external `Qwen2.5-72B-Instruct` vLLM servers on GPUs 0–7):
  (a) exp3 tier 2 V3.2 generation — last 175 `perm_periodic_B` items + the `clean2` identical-rerun
  generation floor (the floor for the "identical token streams" column); (b) GLM-5.2 / GLM-5 tier-2
  PPL (~2.5 h each) + ACC; (c) exp3 tier 3 (not designed/run). Resume verbatim with
  `scripts/gpu/exp3_tier2.py --resume` (`--skip-ppl` for the generation block only); TP=8 is required —
  TP=4 changes reduction order and breaks comparability with the recorded baseline, and the GLM FP8
  weights do not fit in 4 GPUs. ≈10 h wall per model if resumed.

## 4. Recent decisions and why
- Stay on **native FP8 / 8 GPUs, >= ~1 TB HBM** (GPU model open, 2026-08-27; not H100: 640 GB < ~690 GB weights; not 6/7 GPUs: 128 heads &
  256 experts don't divide; not QuantTrio AWQ: it 4-bit-quantizes the indexer itself — verified
  `indexer.wq_b.qweight` in its tensor map). NVFP4 builds are Blackwell-only.
- Hot-set "hotness" = per-layer selection frequency over the decode (offline oracle ranking,
  upper bound); pool N = final candidate count; budgets nested.
- LongBench: 512-output summarization tasks chosen for decode-step count; ≤20K-token filter, seed 42.
- MoE context-scaling plot uses a 0-based lift axis (autoscale exaggerated a 16→14.9× dip).

## 5. Gotchas
- ds4: no `--seed`; t/s not printed under `--dump-logprobs` (throughput derived from `/usr/bin/time`);
  `DS4_TRACE_DECODE_ONLY=1` essential (prefill trace = GBs); RULER `prepare.py` needs
  `work/experiment/shim/python`→python3 on PATH. CPU prefill is O(n²): 64K ≈ 26 h, 128K ≈ 75 h.
- **Shared machine** (2× Xeon, 251 GB RAM). Other users run Vivado (≈100 GB). Check `free -h` before
  loading anything ≥80 GB; a thrashing 161 GB load once froze the box. User asked to yield memory.
- Always launch long runs detached: `setsid nohup ... < /dev/null &` + a background watcher loop; a
  network drop once killed a 24 h run. Runners skip completed runs (guard on `outputs/generations.jsonl`).
- Harness: a bare trailing `sleep` in a Bash call gets blocked; use until-loops or run_in_background.
- llama.cpp local patches (uncommitted, in `work/llama.cpp`): `ggml_cont()` before the MLA absorbed
  matmuls in `src/models/deepseek2.cpp` (~line 295) and `src/llama-graph.cpp` (~line 2427) — required
  for CPU (scheduler assert `cur_backend_id != -1`); `dsa_topk` tensor rename in
  `src/models/deepseek32.cpp` (~line 345); diagnostic prints in `ggml/src/ggml-backend.cpp:~1242` and
  `ggml/src/ggml-cpu/ops.cpp:~4946`. HF `unsloth/DeepSeek-V3.2-Exp-GGUF` is gated; use `-GGUF` (no Exp).
- Repo hygiene: scrub `<HOME>/...work` → `<WORKDIR>` in anything committed to sparse_attn_cpu.
  versel remote diverges often (user commits there) → `git fetch && git rebase origin/main` before push.
  WebFetch caches 15 min: verify live pages with a `?v=` query string.
- Vercel Part I is the user's B200/vLLM study — never edit its numbers.
- **Scripts have no `--help` and act on any argv**: `generate_longbench_plots.py X` writes plots into a
  directory named `X`; `build_longbench_prompts.py` ignores args and regenerates `prompts/longbench/` +
  the manifest (deterministic, seed 42 — verified byte-identical 2026-08-27); `aggregate_longbench.py`
  rewrites the aggregate JSON (deterministic). `analyze_hotset_coverage.py`/`analyze_moe_concentration.py`
  with no run dirs print empty tables. Read the script header instead of probing with `--help`.
  (A stray `scripts/--help/` directory from such a probe may still exist — safe to delete.)

## 6. Frequently used commands
```bash
EXP=<WORKDIR>/experiment
cd <WORKDIR>/ds4 && make cpu          # rebuild engine
# RULER sweep (both traces): run-ids niah_single_2_<L>_moe_q2
bash $EXP/scripts/run_experiment_moe.sh "4096 8192 16384" 256
# LongBench: build prompts, run (resumable), analyze, aggregate, plot
python3 $EXP/scripts/build_longbench_prompts.py
setsid nohup bash $EXP/scripts/run_longbench.sh > $EXP/logs_longbench_driver.log 2>&1 < /dev/null &
bash $EXP/scripts/analyze_longbench_all.sh          # ingest→validate→R1→R2→R3 per run
python3 $EXP/scripts/aggregate_longbench.py         # → analysis_longbench/longbench_aggregate.json
python3 $EXP/scripts/generate_longbench_plots.py    # → analysis_longbench/plots/lb_0{1,2,3}*.png
# Per-run pieces
python3 $EXP/scripts/{ingest_trace,validate_trace,analyze_locality}.py <run_dir>
python3 $EXP/scripts/{ingest_moe_trace,analyze_moe_locality}.py <run_dir>
python3 $EXP/scripts/analyze_moe_concentration.py <plot_dir> <run_dir...>
python3 $EXP/scripts/analyze_hotset_coverage.py <plot_dir> <run_dir...>
python3 $EXP/scripts/generate_moe_plots.py <out> <run_dir...>; generate_kv_plots.py likewise
# V3.2 DSA dev tracer (4-layer model; short prompts only until the long-context bug is fixed)
cd <WORKDIR>/llama.cpp && cmake --build build -j48 --target llama-eval-callback
DSA_TRACE_OUTPUT=/tmp/dsa.jsonl ./build/bin/llama-eval-callback -m ../models/v32_4L/DeepSeek-V3.2-4Layers-Q8_0.gguf -t 48 -c 4096 -n 8 -p "..." -ngl 0
```
Trace env for ds4 runs: `DS4_TRACE_OUTPUT=<dir> DS4_MOE_TRACE=1 DS4_TRACE_LEVEL=3 DS4_TRACE_DECODE_ONLY=1
DS4_TRACE_FULL_SCORE_SAMPLE_RATE=0.002 DS4_TRACE_FLUSH_INTERVAL=32 OMP_NUM_THREADS=64`.
Trace records: `indexer_trace.jsonl {phase,layer,pos,n_comp,top_k,sel[],scores[]}`,
`moe_trace.jsonl {phase,layer,pos,token,is_hash,sel[6],weights[6]}`.

## 7. Next steps (priority)
0. **[HISTORICAL — the package below was executed 2026-08-29/31; kept as the method spec]** GPU campaign package (2026-08-27): `01_github/sparse_attn_cpu/docs/00_doc/
   GPU_CAMPAIGN.md` is the top-level file to hand to the agent on the rented 8-GPU node (its §8 has the
   ready-to-paste prompt); it links `exp0_environment.md` (pin/install vLLM, models, TP8 smoke, hook unit
   checks), `exp1_dsv32_gather_index.md` (V3.2 hook at `flashmla_sparse.py:838 forward_mqa`, adapter to the
   ds4 run-dir schema so ingest/validate/analyze run unchanged, 8K–128K ladder, bf/ld run kinds),
   `exp2_glm_gather_index.md` (GLM-5.2 index-share handling, GLM-5 DSA check), `exp3_reindex_accuracy.md`
   (block-granular A vs entry-granular B re-index, ds4-mirrored modes/controls, benchmark set + official-number
   gate, `docs/reindex_accuracy/{results.csv,per_item.jsonl,README.md}`). Citations verified against vLLM
   checkout `5559679` (`<HOME>/0007_26summer/03_vLLM/vllm`).
   **2026-08-28:** `docs/00_doc/locality_metrics.md` = the metric reference (formulas from the code, JSON keys,
   V3.2 adapter caveats, V4 worked table) linked from GPU_CAMPAIGN §2 and exp1 §4; committed + pushed.
   **Revised 2026-08-27 (user decisions):** GPU type generic (8 GPUs, >= ~1 TB HBM; exp0 §1 detects model/SM/
   driver/HBM and picks kernels — first VERIFY item); execution mode = run start to finish **without review
   stops**: gates are self-checks, per-gate report is a file `docs/00_doc/reports/<phase>_<date>.md` + HANDOFF
   Status line + per-experiment summary md; stop only for §5 (a)–(d) (gate fails after one retry / method-
   changing VERIFY outcome / undecided decision incl. budget > 20 % overrun / clean run outside official range).
1. **Top open item — resume exp3 tier 2 when 8 GPUs are free** (see §3 "Open GPU work"): the 175
   remaining `perm_periodic_B` generation items, the `clean2` generation floor, then GLM-5.2/GLM-5
   tier-2 PPL+ACC. Nothing else in the campaign is outstanding. (Items 1–2 of the old pre-node prep
   list and the GLM-5 `index_topk` question are DONE: GLM-5 has DSA, 78 layers, no index sharing.)
2. Re-sweep the v6 migration-policy study on the V3.2 exports (`exports/v6_v32/`) — the v6 conclusions
   were fit on V4 traces, and V3.2 is *more* local at 64K (0.79 ld vs 0.67), so they should hold but
   are not yet verified. Consumer: ramulator-owner, branch v6.
3. Optional: ROUGE-score the LongBench generations (`runs/lb_*/outputs/generations.jsonl`) for a
   quality-sanity table; optional 128K CPU RULER point (~75 h, needs prompt via
   `benchmark/RULER prepare.py --max_seq_length 131072` + `build_prompts.py` LENGTHS edit).
4. If the CPU V3.2 tracer is resumed: move the `dsa_trace_cb` hook into `llama-server`.

## 7b. GPU campaign — prerequisites (SATISFIED 2026-08-29; historical)
Rental account/credentials for an 8-GPU node with >= ~1 TB HBM (H200/B200-class; ~1,000 GPU-h budget) and the go date; confirmation of the
model (`deepseek-ai/DeepSeek-V3.2` native FP8, plus GLM-5/5.2 if wanted); HF token if any target repo is
gated; the session URL for the commit trailer. Everything else (vLLM pin, hook points, schema adapter,
ladder, benchmarks, re-index implementations, gates) is specified in `GPU_CAMPAIGN.md` + `exp0..exp3`
(see §7 item 0); the GPU agent resolves the "VERIFY ON THE NODE" list (GPU_CAMPAIGN.md §7) itself.
Traces land on the node's NVMe (1–2 TB); only analysis artifacts come back into `sparse_attn_cpu/docs/`.

## 8. Interfaces / dependencies
- Engines: antirez/ds4 (MIT, 80ebbc3) with `docs/ds4_instrumentation.patch` (in sparse_attn_cpu; reverse-
  applies clean to HEAD); llama.cpp (master ~July 2026; `deepseek32` arch + DSA KV cache).
- Data: NVIDIA/RULER (38da79d), zai-org/LongBench `data.zip` (extracted in `benchmark/longbench/`),
  DeepSeek tokenizer in `work/experiment/tokenizer/`. HF anonymous downloads work (`hf download`).
- Python: pandas/pyarrow/matplotlib/tokenizers; metric helpers in `scripts/locality_lib.py` (id-set-agnostic
  — reused by KV, MoE, and hot-set analyses).
- Publishing convention: analysis artifacts + scripts go to sparse_attn_cpu (`docs/*_sweep/`), raw traces
  stay local; report pages to versel (assets in sibling `assets/`, verify with html.parser + live fetch).
- **Consumers of these numbers (read-only for us; report changes to main, never edit):**
  - Ramulator `01_ramulator/01_github/ramulator2/examples/v5_reindex_gather_testcase.py` —
    `MEASURED_OVERLAP = {4096: 0.869, 8192: 0.790, 16384: 0.718, 32768: 0.672, 65536: 0.668}` and
    `MEASURED_TOP10 = {0.201, 0.349, 0.575, 0.767, 0.907}` (coverage of the hottest 10 % of the pool,
    `docs/kv_hotset_coverage.md` 10 % row) calibrate its synthetic selection process. (4K 0.869 is the
    old-series value; published 0.868 — immaterial.)
  - Ramulator digest `ramulator2/00_doc/01_design/v5/ref_dsv4_kv_locality_study.md`: §3–§6 match our
    artifacts; **§2 table is the old `_q2` series (4K/8K/16K/40K/64K, "128 decode tokens", 64K adj 0.670,
    lift 13.2× at 40K)**, not the published 32K/64K `_moe_q2` points (0.672/0.668, 10.53×/21.37×).
    `v5_reindex_correctness_ds4.md` §2 calls `ds4_nofm` "the unmodified engine" — it is built from the
    hooked source (hook inert at MODE=0), and its hook patch lacks the `DS4_PERM_IDENTITY` lines.
  - Evaluation `ramulator2/00_doc/03_evaluation/` (§6 hit-ratio inputs) and the paper's §3.3
    (`00_doc/02_writing_paper/`) quote the digest; any re-measurement must be announced to both.
