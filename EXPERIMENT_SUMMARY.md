# Experiment Summary — DeepSeek-V4 Sparse-Attention KV Temporal Locality

**Date:** 2026-06-29 · **Engine:** ds4 (CPU) · **Model:** DeepSeek-V4-Flash IQ2 (81 GB) ·
**Benchmark:** official NVIDIA/RULER `niah_single_2` · **Decode:** greedy, 128 tokens, deterministic.

## What was run
Three samples — one `niah_single_2` retrieval prompt at **4K, 8K, 16K** tokens — traced at the
CSA lightning-indexer top-k selection point during autoregressive decode. All three **correctly
retrieved the needle**. Wall-clock: 54 min / 1h45 / 4h09 (≈6h48 total), peak RSS ~80 GB, no swapping.
Each run emits the immutable trace (`traces/*.parquet`), per-metric analysis, tables, plots, and a
per-sample revisit package. The raw trace is the primary evidence; every number below is reproducible
from it via `scripts/analyze_locality.py`.

## Headline result
The indexer's per-token KV selection is **strongly temporally local, and increasingly *non-random*
as context grows**. With a fixed top-k=512, longer context = more candidates = sparser selection,
and the selection becomes far more structured than chance:

| Context | n_candidates | kept | adjacent overlap | locality lift vs random | recency-baseline overlap |
|--------:|-------------:|-----:|-----------------:|------------------------:|-------------------------:|
| 4K  | ~1008 | 51% | **0.869** | 1.7× | 0.43 |
| 8K  | ~1887 | 27% | **0.790** | 2.9× | 0.30 |
| 16K | ~4080 | 13% | **0.718** | **5.7×** | 0.16 |

## Answers to the research questions (doc §1)
1. **Adjacent-token locality.** 72–87% of the compressed-KV indices selected at decode step *t* are
   re-selected at *t+1* (rank-aware weighted overlap is essentially identical: 0.72–0.86), so it is
   the *high-rank* entries that persist. Churn is only 13% (4K) → 28% (16K).
2. **Longer-horizon retention.** Retention does **not** decay to zero — it plateaus on a persistent
   hot set: at lag 64, 76% (4K) / 61% (8K) / 50% (16K) of the current set was already selected 64
   steps earlier. Some blocks (needle + sinks) persist for the **entire** decode (persistence max =
   full length).
3. **Layer dependence.** All 21 CSA layers show high locality (16K per-layer adjacent overlap 0.62–0.88;
   min L24, max L36); the pattern is broad, not confined to a few layers. Cross-layer Jaccard
   (semantic position agreement, *not* physical sharing) is moderate and falls with context: 0.49 → 0.22.
4. **Task dependence.** Only retrieval (`niah_single_2`) this pass; multi-hop/aggregation deferred.
5. **Context-length dependence.** Adjacent overlap drifts **down** with context (0.87→0.72) because
   more candidates allow more churn, but locality **lift** rises sharply (1.7×→5.7×) — relative to
   chance the selection gets *more* structured at longer context.
6. **Correctness dependence.** All three correct; correct-vs-incorrect contrast needs the (deferred)
   larger sample set.
7. **Systems relevance.** Reuse is near-total: logical cold-access fraction is 1.4% (4K) → 4.0% (16K),
   i.e. ~96–99% of block accesses re-touch a previously-used block. The **working set is tiny**: a
   64-step decode window touches only 2.6% (4K) / 3.8% (8K) / 5.6% (16K) of `64×top-k` distinct blocks.
   A small retained hot set would capture almost all accesses.

## The hot set *is* the answer: needle-block retention
Mapping the planted needle's token position to its compressed-KV block and checking how often that
block is selected during decode:

| Context | needle block selected (% of layer×step cells) | random baseline | CSA layers selecting it *every* step |
|--------:|----------------------------------------------:|----------------:|-------------------------------------:|
| 4K  | **100%** | 50%  | 21/21 |
| 8K  | **100%** | 27%  | 19/21 |
| 16K | **95%**  | 12.5%| 15/21 |

The indexer pins the answer-bearing KV block as a permanent member of the per-token hot set, far above
chance — a direct mechanistic explanation for both the high temporal locality and the correct retrieval.

## Is it just recency? No.
Adjacent overlap (0.72–0.87) ≫ the deterministic recency baseline (most-recent-512 compressed
entries: 0.16–0.43), and the gap widens with context. Access-age confirms it: only **2–6%** of
selections fall in the most-recent 1% of context; **44–57%** are in the oldest half. The indexer
reaches back to semantically relevant *old* positions (the planted needle is old), not just recent ones.

## Allowed claims (doc §40 form)
- "On RULER `niah_single_2` at 16K context, the median CSA layer re-selected **71.8%** of its compressed
  KV entries between adjacent decode tokens — **5.7×** the random-selection expectation."
- "The 64-token working set was **5.6%** of `64×top-k` at 16K — strong reuse of a small hot set."
- "Selection is semantic, not recency-driven: adjacent overlap 0.72 vs recency-baseline 0.16 at 16K,
  with only 6% of selections in the most-recent 1% of context."

## Caveats (doc §41)
Q2 quantized runtime (not full precision); CPU reference path (not GPU production); **logical** KV
reuse (not physical cache hits); cross-layer overlap = semantic agreement, not shared tensors;
**n=1 sample per context length** → point estimates only, no cross-sample confidence intervals.

## Minimum acceptance criteria (doc §37) — status
Benchmark selected+documented ✓ · end-to-end run ✓ · token & per-layer CSA traces ✓ · selected
indices+scores ✓ · run manifest ✓ · model/dataset revisions recorded ✓ · trace checksums ✓ ·
tracing on/off identical tokens ✓ · adjacent overlap ✓ · multi-lag retention ✓ · working-set ✓ ·
reuse-distance ✓ · per-layer plots ✓ · per-sample package ✓ · code/build/run commands saved ✓ ·
GPU adapter points documented ✓ · trace-overhead (running, appended below).

## Trace overhead (doc §32)
Measured on a 2.3K-token prompt, 16 decode tokens, identical prefill (decode-only tracing):

| Trace level | prefill t/s | generation t/s | wall clock | trace bytes |
|---|---:|---:|---:|---:|
| off       | 1.45 | 0.62 | 27:02 | 0 |
| metadata (L0) | 1.46 | 0.62 | 26:53 | 46 KB |
| selected (L1) | 1.43 | 0.61 | 27:18 | 2.1 MB |

Overhead is within measurement noise — ~1–2% on decode tok/s, ≈0% overall (prefill is untraced).
Decode-only buffered writes are cheap relative to the ~1.6 s/token MoE forward. Selected-level trace
volume ≈ 2.1 MB per 16 decode tokens (≈ top-k indices+scores × 21 CSA layers).

## Where everything is
`experiment/` — `benchmark_decision.md`, `README.md` (workflow + GPU adapter), `runs/<id>/` (manifests,
traces/*.parquet, analysis/, sample_reports/), `tables/` (11 CSV+MD), `plots/` (34 PNG), `scripts/`.
