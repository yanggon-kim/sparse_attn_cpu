# fig:hotness data (paper §3.3 / evaluation) — provenance

Exported 2026-08-28 by `scripts/export_hotness_fig_data.py` (run from `<WORKDIR>/experiment/`; deterministic,
reads the analysis JSONs listed below). All numbers: **DeepSeek-V4-Flash, IQ2_XXS GGUF, ds4 CPU engine,
greedy decode, 21 CSA layers (even layers 2–42), top-k = 512 compressed blocks of 4 tokens (ratio 4)**.
Metric definitions: `../locality_metrics.md` (§2.4 retention, §2.5 working set, §2.8 hot-set coverage).

| file | run id | source JSON (under `<WORKDIR>/experiment/runs/<run id>/analysis/`) | script that produced the JSON |
|---|---|---|---|
| `hotness_coverage_64k.csv` | `niah_single_2_65536_moe_q2` (RULER NIAH single-2, 64K context, 158 decode steps, pool N = 16,393 blocks) | `hotset_coverage.json` → `coverage_by_pool_pct` | `analyze_hotset_coverage.py` |
| `hotness_coverage_32k.csv` | `niah_single_2_32768_moe_q2` (32K, 117 steps, N = 8,039) | same | same |
| `hotness_coverage_16k.csv` | `niah_single_2_16384_moe_q2` (16K, N = 4,095) | same | same |
| `hotness_retention.csv` (cols `*_ruler64k`) | `niah_single_2_65536_moe_q2` (158 steps, lags 1–64) | `metrics_run_summary.json` → `overall_retention.lag_L`, `overall_working_set_ratio.wW` | `analyze_locality.py` |
| `hotness_retention.csv` (cols `*_longform`) | `longform_p16k_g4k_q2` (17K long-form prompt + 3,018 decode steps, lags 1–2048, windows 1–1024) | `extended_retention.json` → `retention.L`, `working_set_ratio.W` | `extended_retention.py` |

## Columns and units

- `hotness_coverage_*.csv`: `pool_pct` = hot region size as % of the final candidate pool N (offline
  oracle: blocks ranked by per-layer selection frequency over the whole decode, nested budgets);
  `mean`/`min`/`max` = fraction of all top-k selections (0–1) that fall inside that region, mean / min / max
  **over the 21 CSA layers** (the min–max pair is the per-layer band). 64K: cov@10 % = 0.907, A@99 = 20.4 %
  of pool (per-layer 9.9–38.4 %); 32K: 0.767 / 31.4 %; 16K: 0.575 / 47.5 %.
- `hotness_retention.csv`: `lag` = decode steps; `retention_*` = mean fraction (0–1) of the entries selected
  at step t that are selected again at step t+lag (per-layer mean, then mean over layers; the two runs
  aggregate slightly differently, see `locality_metrics.md` §2.4); `working_set_ratio_*` = distinct entries in
  a window of `lag` steps divided by `lag·k` (1.0 = no reuse). Empty cells = lag not computed for that run.
  Random-selection baselines: 64K RULER k/N ≈ 0.031; long-form ≈ 0.11 (pool ≈ 4.5–5.2K blocks over the decode).
- Short-trace caveat: the 158-step RULER curve flattens at lag ≥ 16 because few (t, t+lag) pairs remain;
  the 3,018-step run shows the true two-timescale decay 0.73 → 0.51 (64) → 0.37 (512) → 0.17 (2048).

## Status

CPU / V4-Flash data. The DeepSeek-V3.2 GPU campaign (`../GPU_CAMPAIGN.md`, exp1) will replace these files
**in the same schema** (same columns; k = 2048 tokens, no block compression; run ids `dsv32_*`).
