# fig:hotness data (paper §3.3 / evaluation) — provenance

**Current files = GPU campaign (2026-08-29): DeepSeek-V3.2, GLM-5.2, GLM-5 on 8× B200, vLLM 0.28.0 (tag `2cf0a69`,
contains the docs pin `5559679`), TP8 eager, native FP8 weights, greedy, top-k = 2048 tokens (ratio 1, no block
compression), kv_cache_dtype `auto` = bf16 MLA latent rows (1152 B/token/layer) + 132 B fp8 indexer key/token/layer.**
Campaign commit in this repo: `a28245c` (analysis + reports) followed by the raw-trace batches (HEAD `1c2a3f2`);
per-run sources are `docs/{gpu_sweep,glm_sweep}/runs/<run id>/analysis/`. Exported by
`scripts/export_gpu_analysis_data.py` (deterministic; reads only the analysis JSONs listed below).
Metric definitions: `../locality_metrics.md` (§2.4 retention, §2.5 working set, §2.8 hot-set coverage).
The previous CPU / DeepSeek-V4-Flash files are kept as `*_v4.csv` (same columns; provenance at the end of this file).

**All GPU rows use the long-decode ("ld") runs only** — 8 runs per rung, 2048 forced decode steps each
(LongBench-v1 summarization at ≤ 32K, InfiniteBench En.QA / En.Sum at 64K/128K; run ids in `hotness_provenance.json`).
The benchmark-faithful ("bf") runs decode only 3–512 steps (median ≈ 20 at 64K/128K) and inflate hot-set
concentration (64K A@99: 13 % bf vs 25 % ld); the campaign's own `gpu_sweep_summary.md` averages both kinds.
`gpu_headline_by_kind.csv` gives every rung split by kind so a consumer can pick either view.

| file | model (tag) | run ids | source JSON (under `runs/<run id>/analysis/`) | script that produced the JSON |
|---|---|---|---|---|
| `hotness_coverage_{16k,32k,64k,128k}.csv` | DeepSeek-V3.2 (`v32`), 61 top-k layers | the 8 `v32_*_<rung>_ld0x` runs of the rung | `hotset_coverage.json` → `per_layer.<layer>.cov_by_pool_pct` | `analyze_hotset_coverage.py` |
| `hotness_coverage_glm52_*.csv` | GLM-5.2 all 78 layers (`glm52`, memory-system view; the 57 shared layers reuse their producer's set bit-for-bit) | `glm52_*_ld0x` | same | same |
| `hotness_coverage_glm5_*.csv` | GLM-5, 78 layers (`glm5`) | `glm5_*_ld0x` | same | same |
| `hotness_retention.csv` | DeepSeek-V3.2 | the 8 ld runs at 64K (`retention_ld64k`, `working_set_ratio_ld64k`) and at 128K (`*_ld128k`) | `extended_retention.json` → `retention.<lag>`, `working_set_ratio.<w>` (lags 1–1024, windows 1–1024) | `scripts/gpu/extended_retention_fast.py` |
| `hotness_retention_glm52.csv`, `hotness_retention_glm5.csv` | GLM-5.2 (all layers), GLM-5 | same rungs | same | same |
| `gpu_headline_by_kind.csv` | all four tags (`glm52b` = GLM-5.2 computing layers only) × rung × kind | every campaign run (540) | `metrics_run_summary.json`, `extended_retention.json`, `hotset_coverage.json` | `scripts/gpu/analyze_locality_fast.py` etc. |
| `retention_curves_gpu.json` | all tags | every campaign run | same | — (schema: `../v6_export_format.md`; families `<tag>_{ld,bf,ruler}_<rung>`) |
| `hotness_provenance.json` | — | run ids, pool N, A@99 mean/min/max over the 8 runs, layer count per CSV | — | — |

## Columns and units

- `hotness_coverage_*.csv`: `pool_pct` = hot region size as % of the final candidate pool N (offline oracle: tokens
  ranked by per-layer selection frequency over the whole decode, nested budgets); `mean` / `min` / `max` = fraction of
  all top-k selections (0–1) that fall inside that region, **pooled over (run, layer) pairs of the rung** (8 runs × 61
  or 78 layers; the min–max pair is the per-layer band across runs). Headline (V3.2, ld): 128K cov@10 % = 0.948,
  A@99 = 17.8 % of pool (per-run 13.8–25.3 %, N ≈ 113K); 64K 0.894 / 24.8 % (20.9–29.0 %); 32K 46.6 %; 16K 66.5 %.
  GLM-5.2: 64K 32.2 %, 128K 25.2 %; GLM-5: 31.9 %, 23.8 %.
- `hotness_retention*.csv`: `lag` = decode steps; `retention_*` = mean over the 8 runs of the run's pooled
  (layer, step) fraction of the entries selected at step t that are selected again at step t+lag; `working_set_ratio_*`
  = distinct entries in a window of `lag` steps divided by `lag·k` (1.0 = no reuse). Empty cell = lag/window not
  computable (lag 2048 needs > 2048 steps). Random-selection baselines k/N: 64K ≈ 0.031, 128K ≈ 0.018.
  V3.2: 0.79 (lag 1) → 0.64 (64) → 0.48 (512) → 0.35 (1024) at 64K; 0.80 → 0.63 → 0.47 → 0.34 at 128K — still decaying.
- `gpu_headline_by_kind.csv`: `adjacent_overlap`, `lift_vs_random`, `recency_baseline`, `retention_lag64` from
  `metrics_run_summary.json` (unweighted layer means); `retention_lag512/1024` from `extended_retention.json` (ld only);
  `A99_pct`, `cov_top10pct`, `pool_N` from `hotset_coverage.json`; all means over the runs of the cell.

## Previous CPU / V4-Flash files (`*_v4.csv`, exported 2026-08-28 by `scripts/export_hotness_fig_data.py`)

DeepSeek-V4-Flash, IQ2_XXS GGUF, ds4 CPU engine, greedy, 21 CSA layers (even layers 2–42), top-k = 512 compressed
blocks of 4 tokens (ratio 4). `hotness_coverage_{64k,32k,16k}_v4.csv` = `runs/niah_single_2_{65536,32768,16384}_moe_q2`
(RULER NIAH single-2; 158/117/~120 decode steps; pool N = 16,393 / 8,039 / 4,095 blocks), min/max over the 21 layers of one
run; 64K cov@10 % = 0.907, A@99 = 20.4 %. `hotness_retention_v4.csv`: `*_ruler64k` = `niah_single_2_65536_moe_q2`
(158 steps, lags 1–64; flattens at lag ≥ 16 because few pairs remain), `*_longform` = `longform_p16k_g4k_q2` (17K
prompt + 3,018 decode steps, lags 1–2048: 0.73 → 0.51 (64) → 0.37 (512) → 0.17 (2048)).
