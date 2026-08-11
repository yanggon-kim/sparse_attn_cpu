# LongBench Validation: KV + MoE Selection Locality on a Real Benchmark (DeepSeek-V4-Flash, ds4 CPU)

The RULER sweep established the locality results at n=1 per context length on a synthetic needle task. This
collection re-measures **Results 1 (KV top-k locality), 2 (MoE routing locality), and 3 (hot-set coverage)**
on **LongBench summarization** — a real benchmark — with **12 samples × 3 tasks = 36 runs**, giving
cross-sample error bars. Same instrumented ds4 engine (DeepSeek-V4-Flash IQ2, CPU, greedy), both traces
(`indexer_trace.jsonl` top-512 KV / 21 CSA layers + `moe_trace.jsonl` top-6-of-256 / 43 layers) per decode
token, decode-phase only.

## Setup
- **Tasks:** the three 512-output summarization sets — `multi_news` (~2.8K tokens), `gov_report` (~10.5K),
  `qmsum` (~12.3K, query-focused) — official prompt templates and official `max_gen=512`
  (zai-org/LongBench data.zip, v1).
- **Selection:** per task, filter to prompts ≤ 20K tokens (12/200 gov_report, 22/200 qmsum excluded), then a
  fixed-seed (42) random pick of 12. Manifest: `longbench_sweep/longbench_samples.jsonl`.
- **Scale:** 36/36 runs completed (rc=0, trace validation PASS), ~3.5 days wall-clock; **15,073 decode
  steps**, 316,533 KV + 648,139 MoE selection records (~4.5 GB raw, kept local; summaries ship here).
- **Pipeline:** `build_longbench_prompts.py` → `run_longbench.sh` → per-run R1/R2/R3 chain
  (`analyze_longbench_all.sh`) → `aggregate_longbench.py` (mean ± std + bootstrap CI, per task and pooled)
  → `longbench_sweep/longbench_aggregate.json`; figures via `generate_longbench_plots.py`.

## Headline table (mean ± std across samples, n=12/task)

| task | prompt tok | decode steps | KV adj overlap | KV lift | recency | MoE learned adj | hot-set A@99 |
|---|--:|--:|--:|--:|--:|--:|--:|
| multi_news | 2,778 | 491 | 0.914 ± 0.057 | 1.35 ± 0.35× | 0.729 | 0.372 ± 0.014 | 92.0 ± 3.2% |
| gov_report | 10,493 | 512 | 0.755 ± 0.048 | 3.87 ± 1.51× | 0.302 | 0.353 ± 0.007 | 78.7 ± 9.0% |
| qmsum | 12,290 | 253 | 0.732 ± 0.031 | 4.40 ± 1.14× | 0.232 | 0.333 ± 0.010 | 61.4 ± 4.9% |
| **pooled** | 8,520 | 419 | 0.801 ± 0.094 | 3.21 ± 1.73× | 0.421 | 0.353 ± 0.019 | 77.4 ± 14.1% |

Pooled KV retention: 0.801 (lag 1) → 0.689 (lag 8) → 0.605 (lag 64).

## Finding 1 — real tasks lie on the RULER curve
Plot all 36 samples as (prompt tokens → adjacent overlap / lift) and overlay the 5-point RULER sweep: the
LongBench points **fall on the same curve** (`plots/lb_01_kv_vs_context.png`). Overlap declines and lift
rises with context exactly as the synthetic sweep predicted (1.35× at 2.8K → 4.4× at 12.3K), and the
recency baseline collapses (0.73 → 0.23) — the **semantic (non-recency) locality story generalizes from
needle retrieval to real summarization/QA**, now with per-task std of only ±0.03–0.06.

## Finding 2 — MoE routing locality is task- AND context-independent
Learned-layer adjacent overlap is **0.333–0.372 (~15–16× the 6/256 random baseline)** on all three tasks,
sitting in/near the RULER sweep's 0.348–0.376 band; hash layers stay at the random floor (~0.023)
(`plots/lb_02_moe_by_task.png`). Small but consistent task ordering (multi_news > gov_report > qmsum)
suggests mild content dependence — the query-focused task routes slightly less repetitively.

## Finding 3 — hot-set coverage scales with pool size, as predicted
A@99 (fraction of the candidate pool needed to cover 99% of top-k accesses) shrinks as the pool grows:
92% (2.8K) → 79% (10.5K) → 61% (12.3K), consistent with the RULER curve that reaches ~20% at 64K.
qmsum needs the smallest hot set at comparable length — query-conditioning concentrates the selection.

## Notes & caveats
- qmsum generates ~253 steps (query answers EOS early); multi_news/gov_report run near the full 512.
- `is_correct` is not meaningful here (no needle); generation quality evidence = the stored summaries in
  each run's `generations.jsonl` (ROUGE scoring optional, not run).
- IQ2 quantized / CPU reference path / greedy decode, as in the RULER sweep; ≤20K-token filter excluded the
  long tail (34 of 600 candidate samples) to bound per-run wall-clock — both noted for methods sections.
- Raw traces (~4.5 GB) retained locally in `runs/lb_*_q2/`; everything here reproduces from the shipped
  manifest + scripts.
