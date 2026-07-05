# MoE Expert-Selection Locality (DeepSeek-V4-Flash, ds4 CPU)

A companion to the CSA lightning-indexer KV-selection study. DeepSeek-V4 is sparse in **two** places per
layer: the attention selects a small set of past KV entries (the KV study), and the **feed-forward MoE**
routes each token to **6 of 256** experts. This run measures the temporal locality of that *expert*
selection, collected from the **same ds4 CPU decode pass** as the KV indexer trace, over a 4K/8K/16K sweep.

## Setup
- **Engine:** instrumented `antirez/ds4` (single-file C, CPU), DeepSeek-V4-Flash **IQ2** (~81 GB), 64 threads,
  greedy decode. Instrumentation: `docs/ds4_instrumentation.patch` (now covers **both** the CSA indexer and
  MoE); enable MoE tracing with `DS4_MOE_TRACE=1` alongside the existing `DS4_TRACE_*` vars → the engine
  writes `<DS4_TRACE_OUTPUT>/moe_trace.jsonl`, one record per (layer, decode step):
  `{sv,phase,layer,pos,token,n_expert,n_used,is_hash,sel[6],weights[6]}`.
- **Task:** RULER `niah_single_2` at 4K / 8K / 16K, one sample each (all three retrieve the needle correctly).
- **Coverage:** MoE routes on **every** layer's FFN (43 layers) — denser than the CSA indexer's 21 layers.
  **40** layers use learned biased top-6 routing; the **first 3** use deterministic token-id **hash** routing
  and are reported separately throughout. Config: 256 experts, 6 active, +1 shared.
- **Pipeline:** `scripts/ingest_moe_trace.py` → `traces/selected_experts.parquet`;
  `scripts/analyze_moe_locality.py` (reuses `scripts/locality_lib.py`) → `analysis/moe_metrics_*`;
  `scripts/generate_moe_plots.py` → the plots in `moe_locality_4k8k16k/plots/`.
  Runner: `scripts/run_experiment_moe.sh "4096 8192 16384"`.
- **Random baseline:** two independent choices of 6 experts from 256 share **6/256 = 0.0234** on average — the
  yardstick for every overlap below.

## Finding 1 — learned routing is strongly temporally local
Across the 40 learned layers, consecutive tokens reuse ~**37%** of their 6 experts — about **2.2 of 6**
persist step to step, **16×** the random baseline.

| context | adjacent overlap | experts carried / 6 | jaccard | churn | locality lift |
|---|--:|--:|--:|--:|--:|
| 4K | 0.376 | 2.26 | 0.257 | 0.624 | 16.0× |
| 8K | 0.373 | 2.24 | 0.256 | 0.627 | 15.9× |
| 16K | 0.374 | 2.24 | 0.256 | 0.626 | 16.0× |

## Finding 2 — it is context-independent (unlike KV selection)
The single sharpest contrast with the attention side. KV-selection locality **rises** with context (the
candidate pool grows, so lift climbs 1.7×→5.7× over 4K→16K in the KV study). MoE routing does **not**: its
pool is **fixed at 256 experts** regardless of context, so the regime is identical at every length —
adjacent overlap **0.376 / 0.373 / 0.374** at 4K / 8K / 16K (within 0.002). See
`plots/moe_05_context_scaling.png`.

**Implication:** context length is the right sweep axis for KV locality but *not* for MoE locality. For
expert caching the informative axis is decode **length** (how long a hot-expert set stays warm), not prompt
length.

## Finding 3 — hash-routed layers are not local (by construction)
The first 3 layers route by a fixed `token-id → experts` table. Because consecutive generated tokens are
usually different ids, their expert sets barely overlap — adjacent overlap ≈ **0.02**, essentially the
random floor (lift ≈ 1). This is a property of the *routing rule*, not the content; they are always reported
separately so they don't understate learned locality. See `plots/moe_04_hash_vs_learned.png`.

| routing (16K) | layers | adjacent overlap | lift |
|---|--:|--:|--:|
| learned top-6 | 40 | 0.374 | 16.0× |
| hash-routed | 3 | 0.021 | 0.9× |
| random baseline | — | 0.023 | 1.0× |

## Finding 4 — durable core, broad working set
Retention (learned layers) decays from 0.37 (lag 1) to a plateau near **0.15** by lag 64 — still **6×** the
random floor: a couple of experts persist across many steps while most churn. But the working set is broad:
over a 64-step window a layer touches ~**75** distinct experts (ratio ≈ 0.195 of 64·6), and across the full
decode ~**120 of 256** (~47% of the pool). So the routing has a sticky core yet keeps reaching new experts —
an expert cache needs capacity for the tail, not just the top few. (`plots/moe_02_retention.png`,
`plots/moe_03_working_set.png`.)

| retention lag | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 4K | 0.376 | 0.264 | 0.203 | 0.174 | 0.166 | 0.143 | 0.162 |
| 8K | 0.373 | 0.270 | 0.207 | 0.173 | 0.176 | 0.150 | 0.127 |
| 16K | 0.374 | 0.270 | 0.214 | 0.167 | 0.134 | 0.153 | 0.144 |

## Finding 5 — per-layer routing structure: is adjacent overlap really "temporal"?
Adjacent overlap (Findings 1–4) conflates two things. A layer can look temporally local simply because it
has **favorite experts** — a concentrated marginal usage distribution — even if consecutive tokens were
independent. Looking at each layer's expert decisions *over the whole sequence* (not just t vs t−1) separates
them. Analysis: `scripts/analyze_moe_concentration.py`; figure `plots/moe_06_per_layer_concentration.png`;
per-run numbers in `moe_locality_4k8k16k/<ctx>/moe_concentration.json`.

**Every learned layer is concentrated; hash layers are not.** Over a decode, each of the 40 learned layers
routes with a clear preferred committee: the single most-used expert fires for **22%–85%** of tokens
(uniform = 2.3%), just **7–32 experts cover 50%** of the layer's routing, normalized entropy 0.63–0.85
(< 1.0). The 3 hash layers are near-uniform (top-expert rate 0.12–0.16, ~50 experts for 50%, entropy 0.92) —
the negative control. Every learned layer still *touches ~half* the 256-pool (88–171 distinct experts), so
the shape is **concentrated core + long tail**, never a fixed 6. Peakiest: L15, L14 (top-expert rate 0.85,
7 experts = 50%); flattest learned: L39 (0.22), L23 (0.25); peaky layers cluster mid-stack (L11–18, L24–26).

**Static-vs-dynamic decomposition.** Split each layer's overlap into *static preference* — the overlap you'd
get if consecutive tokens were independent draws from that layer's own usage distribution,
`(Σ p_i²)/6` with `p_i` = per-expert usage rate — and *dynamic correlation* = observed / static. For the
learned layers (consistent across 4K/8K/16K): observed **0.374**, static **~0.17** (7.4× the uniform 0.023),
dynamic **~2.3×**. So the ~16× "locality lift" factors as **≈7.4× static preference × ≈2.3× temporal
correlation**. Both are real: static concentration is the larger *multiplier* off random; the step-to-step
correlation adds *at least as much* overlap in absolute terms (observed−static ≈ 0.20 vs static−uniform ≈ 0.14).

**Two modes, and a division of labor.** `corr(static, observed) = +0.86` and
`corr(dynamic-mult, concentration) = −0.86`: the extra stickiness is strongest exactly where concentration is
weakest. Learned layers span two mechanisms that both land near 0.37 — **"specialist" layers** (peaky:
L14/15/24/26) get overlap mostly from static preference (dynamic ≈ 1.5×), while **"context-tracking" layers**
(broad: L9/20/21/23) have the highest dynamic multipliers (3.1–3.2×). By depth: concentration falls
shallow→deep (top-expert rate 0.58→0.48) while the dynamic multiplier rises (2.18→2.49) — **early learned
layers route by fixed specialists, deep layers route by context**; the middle third has the highest overall
overlap. Hash layers even have dynamic multiplier < 1 (observed *below* their marginal) — deterministic
token-id routing on ever-changing tokens is slightly anti-correlated step to step.

## What ships here
`moe_locality_4k8k16k/` holds, per context, the MoE + KV run summaries
(`moe_metrics_run_summary.json`, `kv_metrics_run_summary.json`), the per-layer MoE table
(`moe_metrics_sample_layer.parquet`), the per-layer concentration + decomposition
(`moe_concentration.json`), the generation (`generations.jsonl`), and the 6 MoE + 3 KV plots under
`plots/` (including `moe_06_per_layer_concentration.png`). The raw `moe_trace.jsonl` (~tens of MB) and `selected_experts.parquet` are kept local; all numbers
above reproduce from the summaries via the scripts named in **Setup**. The KV summaries here are the
sparse-attention locality collected in the *same* runs — they reproduce the earlier 4K/8K/16K KV points
(adjacent overlap 0.868 / 0.790 / 0.718; lift 1.72× / 2.92× / 5.72×).

## Caveats
Single sample per length; IQ2 quantized / CPU reference path (quantization may shift the router's top-6 vs
full precision); logical expert ids (not a physical cache — an upper bound that ignores expert-load timing
and bandwidth); greedy decode on one retrieval task; the 3 hash layers are deterministic token-id routing.
Because the expert pool is fixed, the 4K/8K/16K axis varies token content/position, not the sparsity regime.
