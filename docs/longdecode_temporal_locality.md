# Long-Decode Temporal Locality (DeepSeek-V4-Flash, CPU)

A complement to the context-length study (4K–64K needle runs): instead of scaling the **prefill input**,
this run scales the **decode output** — a long-form generation that produces **~3,000 decode steps** so we
can measure temporal locality over a long horizon.

## Setup
- **Task:** long-form completion. Prompt = 17,273 tokens (a large slice of Paul Graham essay prose ending
  mid-thought + a "keep writing at length" cue); the model then **generates 3,019 tokens** greedily
  (`temperature=0`, `-n 4096`) and stops at **EOS** on its own — coherent essay output, **no repetition**.
- **Regime:** candidate pool ~4,695 compressed blocks, top-k 512 → **~11% kept** (≈ the needle-16K sparsity).
- **Data:** 3,018 decode steps × 21 CSA layers = **63,378 indexer selection records** (~24× more steps than
  any needle run). Engine: ds4 CPU, IQ2. Wall-clock ~6h51m.

## Finding 1 — retention keeps decaying; the earlier "plateau" was a short-trace artifact
With only ~128 decode steps, retention looked like it flattened near 0.5 by lag 64. Over 3,018 steps it is
clearly a **two-timescale** decay — a durable near-term core plus slow long-term turnover toward the ~11%
random floor:

| lag (steps) | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| retained | 0.73 | 0.67 | 0.62 | 0.59 | 0.56 | 0.54 | 0.51 | 0.48 | 0.43 | 0.37 | 0.27 | 0.17 |

The selected KV set is stable for *tens* of steps but **rotates almost completely over ~2,000 steps**
(`analysis/extended_retention.png`). This refines the needle-run story, where retention curves capped at
lag 64 suggested a flat plateau.

## Finding 2 — task dependence: generation is recency-driven, retrieval is not
At the **same** sparsity and the **same** adjacent overlap as the needle-16K point:

| metric | long-form generation (this run) | needle retrieval (16K) |
|---|--:|--:|
| adjacent overlap | 0.726 | 0.718 |
| **recency-baseline overlap** | **0.409** | **0.157** |
| locality lift vs random | 6.7× | 5.7× |

Long-form generation attends heavily to the **recent** context it just produced (recency baseline 0.41),
whereas needle retrieval reaches back to an **old** fixed fact (0.16). Same top-level overlap, different
locality character — a genuine task-dependence result.

## Finding 3 — working set
Over a 1,024-step decode window the selection touches **~78%** of all compressed blocks (working-set ratio
0.007 = ~3,670 of ~4,695), consistent with the long-term turnover in Finding 1.

## Caveats
Single sample; IQ2 quantized / CPU reference path; long-form *completion* (not a standard benchmark), Paul
Graham essay continuation; logical compressed-KV indices (see the main experiment summary). The raw trace
(`indexer_trace.jsonl` 430 MB, `selected_kv.parquet` 513 MB) is kept local — this folder ships the analysis
artifacts (`metrics_run_summary.json`, `extended_retention.json`, `extended_retention.png`,
`generations.jsonl`) from which the numbers reproduce via `scripts/analyze_locality.py` +
`scripts/extended_retention.py`.
