# Locality metrics of DSA/CSA top-k selections — definitions, code, and reproduction

Reference for the temporal-locality statistics this repo reports (R1 KV locality, R2 MoE locality,
R3 hot-set coverage), written so that the GPU campaign (`GPU_CAMPAIGN.md`, `exp1_dsv32_gather_index.md`)
computes *identical* statistics from DeepSeek-V3.2 raw traces. Every formula below is the one in the
code (`scripts/locality_lib.py`, `analyze_locality.py`, `analyze_hotset_coverage.py`,
`analyze_moe_locality.py`, `analyze_moe_concentration.py`, `extended_retention.py`, `ingest_trace.py`,
`ingest_moe_trace.py`); function names are cited so the code, not this text, is the authority.

## 1. Notation and which records feed which metric

- Layer ℓ, decode step t (0-based, `decode_step = pos − min(pos)` over decode records, `ingest_trace.py`).
- **S_t^ℓ** = the set of selected indices at (ℓ, t); k = |S_t| (V4-Flash: k = 512 *compressed* blocks,
  ratio 4, so index c covers original tokens [4c, 4c+3]; V3.2: k = 2048 *raw* token positions, ratio 1).
- **N_t^ℓ** = the candidate pool at (ℓ, t) = trace field `n_comp` → parquet `n_candidates_visible`
  (V4: ≈ pos/4; V3.2: N = pos). When N_t < k the whole pool is selected (`valid_k` < `top_k`).
- **Phase**: decode only — both ingesters drop `phase != 1` records (`ingest_trace.py`,
  `ingest_moe_trace.py`); prefill never enters any statistic.
- **Layers**: KV metrics use every layer that emits an indexer record (V4: the 21 CSA layers 2,4,…,42;
  SWA/HCA layers have no top-k and appear only as `is_sparse_layer=False` rows in `layer_events.parquet`;
  V3.2: all 61 DSA layers). MoE metrics use every routed-MoE layer; layers flagged `is_hash=1` (V4:
  layers 0–2, token-id hash routing) are computed but reported in a separate `"hash"` block and **excluded
  from the `"learned"` numbers** (`analyze_moe_locality.py`, `group_summary`). V3.2 has no hash layers.
- Per-run "overall" numbers are **mean over layers of the per-layer mean over steps** (see §2.10).

## 2. Definitions (with the implementing function)

All set helpers are in `locality_lib.py` and are id-agnostic (KV indices, expert ids, anything hashable).

### 2.1 Adjacent overlap — `overlap_fraction(cur, prev)`
`o_t = |S_t ∩ S_{t−1}| / |S_t|` — the fraction of the *current* selection already selected one step
earlier (asymmetric; equals |S_t ∩ S_{t-1}|/k when k is full). Computed only when step t−1 exists for
that layer (`analyze_locality.py`: `steps[i-1] == t-1`). `churn = 1 − o_t`. Per-step values live in
`analysis/metrics_token_layer.parquet` (`adjacent_overlap`), per-layer means in
`metrics_sample_layer.parquet` (`adjacent_overlap_mean`), run-level in `metrics_run_summary.json`
(`overall_adjacent_overlap_mean`).

### 2.2 Jaccard — `jaccard(a, b)`
`J_t = |S_t ∩ S_{t−1}| / |S_t ∪ S_{t−1}|` (empty ∪ → 1.0). With |S_t| = |S_{t−1}| = k,
`J = o/(2−o)`. Key `adjacent_jaccard` / `overall_adjacent_jaccard_mean`. Also used for cross-layer
similarity at the same step (`cross_layer_jaccard.parquet`, `moe_cross_layer_jaccard.parquet`).

### 2.3 Lift vs random — `random_expected_overlap_fraction(k, n)`
Random baseline = expected overlap of two independent uniform k-subsets of N: `k/N` (returns 1.0 if
N ≤ k, NaN if N ≤ 0). **N is taken per step**: `n = cur["n_comp"]` = `n_candidates_visible` at (ℓ, t),
and `k = top_k` = `configured_top_k` (constant 512 / 2048), *not* `valid_k`. Lift is the **mean of
per-step ratios** `o_t / (k/N_t)` (`lift.append(o / exp)`), not the ratio of means. Key
`locality_lift_mean` / `overall_locality_lift_mean`. Because N grows with context while o_t plateaus,
lift rises monotonically with context (1.7× → 21× at 4K → 64K).

### 2.4 Retention at lag L, retention curve
`R(L) = mean_t overlap_fraction(S_t, S_{t−L})` over every t for which step t−L exists in that layer
(`analyze_locality.py`, `LAGS = [1,2,4,8,16,32,64]`; `R(1)` = adjacent overlap). Keys
`retention_lag_{L}` per layer, `overall_retention.lag_{L}` run-level. For long decodes
`scripts/extended_retention.py` recomputes it for `LAGS = [1,…,64,128,256,512,1024,2048]` and writes
`analysis/extended_retention.json` (`retention.{L}`). Note the aggregation differs: `extended_retention.py`
pools all (layer, step) pairs into one mean, while `analyze_locality.py` takes the mean of per-layer
means; with equal step counts per layer the two coincide.

### 2.5 Working set at window w
`WS_t(w) = |∪_{j=t−w+1..t} S_j|` over the last w steps of that layer (truncated at the start of decode),
reported as `working_set_w{w}_mean` (absolute) and `working_set_ratio_w{w}_mean` = mean of
`WS_t(w)/(w·k)` (`WINDOWS = [1,…,64]`; run-level `overall_working_set_ratio.w{w}`). Ratio 1.0 means no
reuse inside the window; the V4 64K value 0.079 at w = 64 means 64 steps touch only 7.9 % of 64·k
distinct entries. `extended_retention.py` extends to w ≤ 1024 (`working_set_ratio.{w}`; there k is
taken as the largest set in the window).

### 2.6 Recency baseline — `analyze_locality.py` ("recency baseline overlap")
`rec_t = overlap_fraction(S_t, {N_t−k, …, N_t−1})` — how much of the selection a "keep the most recent
k entries" policy would have covered. Key `recency_overlap_mean` / `overall_recency_overlap_mean`.
For V3.2 the recency set is the last 2048 token positions.

### 2.7 Weighted overlap, new/evicted, score stability
- `weighted_overlap(cur_ranked, prev_ranked)`: DCG-weighted overlap, `w(r) = 1/log2(r+2)` over the
  score-ranked lists (`ranks_from_scores`: score desc, index asc; ascending index when no scores),
  `Σ_{shared} min(w_cur, w_prev) / Σ w_cur`. Requires `scores[]`; with no scores it degenerates to an
  index-order weighting and should not be compared across engines. Key `weighted_overlap`.
- `new_evicted(cur, prev)` → (`new_entries`, `evicted_entries`) = |S_t \ S_{t−1}|, |S_{t−1} \ S_t|
  (per step in `metrics_token_layer.parquet`); `new_entries` per step is the step's miss count under a
  "cache exactly the previous selection" policy.
- Pearson of shared entries' scores across adjacent steps (`score_pearson_mean`), `mean_abs_dscore`,
  `boundary_margin = rank_k_score − rank_kp1_score` — all optional (need scores).
- Also in the per-layer table: reuse distance over the ranked access stream (`reuse_p50/p90/p99`,
  `reuse_cold_fraction`), persistence run length (`persistence_mean_run`), and access age
  (`age_*`, `frac_recent/middle/old` with cuts at 1 % and 50 % of `context_length_target`).

### 2.8 Hot-set coverage A@α, B, cov@p — `analyze_hotset_coverage.py: per_layer_coverage`
Per layer, over the **whole decode** (offline oracle): selection frequency `freq[c]` = number of steps
in which c ∈ S_t; sort descending; `cum[i]` = cumulative share of all selection events T = Σ freq.
- **A@α**: `H_α = searchsorted(cum, α) + 1` = smallest prefix of the frequency ranking carrying ≥ α of
  all selection events; reported as `pct_of_pool = 100·H_α / pool_N` for α ∈ {.90,.95,.99,.999}
  (`ALPHAS`). JSON: `per_layer[ℓ].A[α].{H, pct_of_pool}`; run-level `A_pct_by_alpha[α]`,
  `A99_pct_mean`, `A99_pct_range = [min, max over layers]`.
- **pool_N** = `max(n_candidates_visible)` for the layer = the **final** candidate count of the decode
  (not per-step). At 64K V4: 16,393 blocks ≈ (65,536 + 158 steps)/4.
- **B**: smallest n such that the 1st-percentile step containment `|S_t ∩ hot_n|/|S_t| ≥ 0.99`
  (`p1_containment`, binary search) — "misses > 1 % of a step on ≤ 1 % of steps"; B ≥ A@99.
  Keys `B_n`, `B_pct_of_pool`, `B99_pct_mean`.
- **cov@p** (`cov_by_pool_pct[p]`, `POOL_PCT_GRID = [1,2,3,5,7,10,15,20,25,30,40,50,60,75,100]`):
  the forward sweep — share of selection events served by the hottest `round(p/100·pool_N)` entries
  (`cum[n−1]`; 1.0 once n ≥ number of distinct selected entries). Run-level
  `coverage_by_pool_pct[p].{mean,min,max}` over layers; **cov10 %** = `coverage_by_pool_pct["10"].mean`
  is the value the ramulator testcase quotes as `MEASURED_TOP10`.
- `distinct_selected` / `distinct_pct_of_pool`: entries ever selected (A@100).

### 2.9 MoE analogues — `analyze_moe_locality.py`, `analyze_moe_concentration.py`
Same helpers over expert ids: S_t^ℓ = the `n_used` routed experts (V4: 6 of 256; V3.2: 8 of 256).
Random baseline is the **fixed** `n_used/n_expert` (V4 6/256 = 0.0234; V3.2 8/256 = 0.03125),
so MoE lift does not scale with context. Keys: `moe_metrics_run_summary.json` →
`learned.{adjacent_overlap_mean, locality_lift_mean, retention.lag_L, working_set_ratio.wW,
n_distinct_experts_mean}` and the same under `hash`.
Concentration (`analyze_moe_concentration.py: perlayer`, output `moe_concentration.json`), per layer:
`p_i` = usage rate of expert i = (#steps with i ∈ S_t)/#steps; `top_expert_rate = max_i p_i`;
`n_experts_50pct` = smallest prefix of the descending slot-count ranking carrying 50 % of slots;
`top6_cov`, `top32_cov`, `entropy_norm`; **static preference**
`static_overlap = Σ_i p_i² / n_used` = the adjacent overlap expected if steps were independent draws from
the layer's own marginal; **dynamic factor** `dynamic_mult = observed_overlap / static_overlap`
(observed = that layer's `adjacent_overlap_mean` from R2). Run-level `learned.{static_overlap,
observed_overlap, dynamic_mult, top_expert_rate, n_experts_50pct}`, `corr_static_observed`,
`corr_dynamic_toprate`.

### 2.10 Aggregation
Per-step → per-layer: plain mean over the layer's steps (NaNs skipped). Per-layer → run: plain mean over
layers (`sl.<col>.mean()`), unweighted by step count; `layer_groups` = shallow/middle/deep thirds of the
sorted layer list. Run → sweep: the RULER sweep is one run per context (n = 1, no CI);
`aggregate_longbench.py` reports mean ± std and a percentile bootstrap CI over runs
(`locality_lib.bootstrap_ci`, 2000 resamples, seed 12345). GPU runs with n ≈ 20 per rung should use the
LongBench-style aggregation, never pool steps across runs.

## 3. Why each metric matters for SPLEX (ranked)

SPLEX keeps a compact hot region of the latent cache in HBM and swaps entries lazily, per layer, at
step barriers; the indexer keys stay HBM-pinned. What the migration policy needs, in order:

1. **Retention curve R(L)** — the decay law of "an entry selected now is selected again L steps later"
   sets the admission rule (admit after how many hits), the residence time before eviction, and the
   swap rate per round; a two-timescale decay (V4: 0.73 → 0.51 @64 → 0.17 @2048) means a durable core
   plus slow turnover, i.e. a small steady swap stream rather than periodic flushes. Needs ≥ 1–2 K steps.
2. **cov@p and A@α** — the "how much room buys how much hit ratio" curve: `coverage_by_pool_pct` is the
   oracle HBM hit ratio of a hot region holding p % of the pool (the equal-room argument: at the same
   f_avail, hotness-aware placement vs. ratio placement); A@99 is the room needed for a 99 % hit ratio.
   Per-layer spread (A@99 at 64K 9.9–38.4 % of pool across the 21 CSA layers) motivates **per-layer**
   HBM budgets instead of one global f. Oracle ranking = upper bound for any online policy.
3. **Adjacent overlap / new_entries** — lag-1 predictability: `new_entries` per step bounds the
   per-round miss count of a "keep last selection" policy; but it is granularity-sensitive (V4 blocks of
   4 tokens vs. V3.2 tokens) and says nothing beyond one step.
4. **Working set WS(w)** — sizes the staging/swap buffer and the DMA window: how many distinct entries a
   burst of w rounds touches.
5. **Recency baseline** — shows a sliding-window cache is the wrong policy (V4 64K: 0.116 vs 0.668).
6. **Lift** — narrative only (how far from random); it is o/(k/N) and grows with N by construction.
7. **MoE overlap / concentration** — expert-weight placement is a separate, fixed-pool problem
   (~16× random, context-independent, ~7× of it static preference); informs weight tiering, not KV.

## 4. Worked example — DeepSeek-V4-Flash, RULER niah_single_2 (ds4 CPU, IQ2_XXS, greedy, `-n 256`)

Runs `runs/niah_single_2_{L}_moe_q2`; all analysis from `analyze_locality.py`, `analyze_hotset_coverage.py`,
`analyze_moe_locality.py`, `analyze_moe_concentration.py` on those run dirs.

| context | steps | adj overlap | Jaccard | lift | recency | R(64) | WS ratio w64 | pool_N | A@99 % (layer range) | cov10 % | MoE learned adj (lift) | static × dyn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4K  | 153 | 0.868 | 0.771 | 1.72  | 0.429 | 0.753 | 0.026 | 1,030  | 79.5 (70.7–87.6) | 20.1 | 0.376 (16.0×) | 0.174 × 2.28 |
| 8K  | 163 | 0.790 | 0.662 | 2.92  | 0.308 | 0.606 | 0.039 | 1,912  | 65.7 (52.1–81.4) | 34.9 | 0.373 (15.9×) | 0.166 × 2.37 |
| 16K | 117 | 0.718 | 0.573 | 5.72  | 0.157 | 0.498 | 0.056 | 4,095  | 47.5 (26.5–66.2) | 57.5 | 0.374 (16.0×) | 0.173 × 2.27 |
| 32K | 117 | 0.672 | 0.520 | 10.53 | 0.100 | 0.457 | 0.067 | 8,039  | 31.4 (16.4–48.4) | 76.7 | 0.348 (14.9×) | 0.166 × 2.22 |
| 64K | 158 | 0.668 | 0.516 | 21.37 | 0.116 | 0.469 | 0.079 | 16,393 | 20.4 (9.9–38.4)  | 90.7 | 0.369 (15.7×) | 0.170 × 2.28 |

Hash layers 0–2: adj overlap 0.020–0.023 = the 6/256 floor. Long-decode retention
(`runs/longform_p16k_g4k_q2`, 17K prompt + 3,018 steps, `extended_retention.py`): R(1) 0.726, R(64) 0.513,
R(512) 0.366, R(2048) 0.170; recency 0.409 (generation is recency-driven, needle retrieval is not).
LongBench (36 runs, `aggregate_longbench.py`): adj 0.914/0.755/0.732, A@99 92.0/78.7/61.4 %.

## 5. Recipe to reproduce from raw data

Run-dir layout expected by every script: `runs/<id>/{traces/indexer_trace.jsonl, traces/moe_trace.jsonl,
outputs/generations.jsonl, run_manifest.json, model_config.json, logs/time_and_stderr.log}`.

```bash
S=scripts; R=runs/<id>
python3 $S/ingest_trace.py $R          # → traces/{selected_kv,score_summaries,layer_events,decode_tokens}.parquet
python3 $S/validate_trace.py $R        # unit tests + integrity; must print VALIDATION PASSED
python3 $S/analyze_locality.py $R      # → analysis/metrics_run_summary.json, metrics_{sample_layer,token_layer}.parquet
python3 $S/analyze_hotset_coverage.py <plot_dir> $R [more runs]   # → analysis/hotset_coverage.json (+2 plots)
python3 $S/extended_retention.py $R    # long decodes only → analysis/extended_retention.json
python3 $S/ingest_moe_trace.py $R      # → traces/selected_experts.parquet
python3 $S/analyze_moe_locality.py $R  # → analysis/moe_metrics_run_summary.json, moe_metrics_sample_layer.parquet
python3 $S/analyze_moe_concentration.py <plot_dir> $R [more runs]  # → analysis/moe_concentration.json
```

Input schema (one JSON object per line):
- `indexer_trace.jsonl`: `{sv:2, phase:0|1, layer, ratio, pos, n_comp, top_k, valid_k, all,
  rank_k_score?, rank_kp1_score?, sel:[...], scores?:[...]}` — `sel` lists the selected indices
  (ascending, only valid ones, length `valid_k`); `scores[i]` is `sel[i]`'s indexer score (optional;
  enables 2.7). `ratio` defaults to 4 if absent (`ingest_trace.py`) — **V3.2 must write `ratio: 1`**.
- `moe_trace.jsonl`: `{sv, phase, layer, pos, token, n_expert, n_used, is_hash, sel:[n_used], weights:[n_used]}`.
- `generations.jsonl` (first line used): `sample_id, generated_token_ids[], generated_token_count,
  prompt_token_count, is_correct`; `run_manifest.json`: `context_length_target`, `benchmark_name`;
  `model_config.json`: `layer_map:[{layer_id, attention_type, compression_ratio}]` (every indexer layer
  `"CSA"`; `validate_trace.py` requires all CSA layers present at every step).

Where each statistic lives: adjacent overlap / Jaccard / lift / recency / retention (≤64) / working set →
`metrics_run_summary.json` keys in §2.1–2.6; extended lags → `extended_retention.json`; A@α, B, cov@p →
`hotset_coverage.json` (§2.8); MoE → `moe_metrics_run_summary.json`, `moe_concentration.json` (§2.9).

V3.2 adapter (`exp1_dsv32_gather_index.md` §4, `scripts/vllm_to_ds4_run.py`): map the vLLM hook JSONL
(`sparse_attn_indexer` top-k output per request, per layer, per step) to the fields above with
`ratio: 1`, `n_comp = pos` (tokens visible to the query, after the block-table translation is undone —
`sel` must be *logical token positions* in the request, not physical slots), `top_k: 2048`,
`valid_k = len(sel)`, `phase: 1` for decode steps, `is_hash: 0`, `n_used: 8`. Then k = 2048 and N = pos
flow through unchanged; the random baseline becomes 2048/pos. **Granularity caveat**: V4 numbers are over
4-token blocks (k = 512 of N ≈ pos/4); V3.2 over tokens (k = 2048 of N = pos). k/N is identical at equal
context, so lift is comparable, but overlap, A@α and cov@p are not directly comparable — token-level
selections can differ inside a block that a block-level trace would count as retained. Report both
token-level and a 4-token-block-folded view (`sel // 4`, deduplicated) for V3.2 if V4 comparison is wanted.
Decode ≥ 2K steps for any retention curve; the ladder's 256-step smoke run is for gates only.

## 6. Pitfalls

- **Short traces show a false retention plateau**: at ≤ 160 steps R(64) looks flat at ~0.47–0.5; with
  3,018 steps it keeps decaying to 0.17 at lag 2048. Quote retention beyond lag 64 only from ≥ 1–2 K-step runs.
- **−1 padding**: ds4 writes only valid indices (`sel` length = `valid_k`; no padding). vLLM's indexer pads
  the top-k tensor with −1 when fewer than k candidates exist (early positions) — the adapter must drop
  them before writing `sel`, or every early step gets a spurious shared "entry" −1 and `validate_trace.py`
  fails its range check.
- **Per-request attribution in batched decode**: rows of the indexer output are ordered by the batch's
  token layout, not by request; map each row through the scheduler's request/position mapping
  (`req_id_per_token`) and verify with the alone-vs-batched identity gate (exp1 §3). Mixed prefill/decode
  batches also require `phase` to be set per row.
- **pool_N is the final pool**: A@α "% of pool" divides by the last step's N, so short decodes and long
  decodes at the same prompt length are comparable, but a decode-scaling run (small prompt, long decode)
  has a pool dominated by generated tokens.
- **Oracle A@α / cov@p are upper bounds**: the ranking uses the whole decode's frequencies; any online
  policy (LRU, frequency with a warm-up) sees a lower hit ratio at the same room — the swap-rate that
  the migration engine pays is measured in the ramulator testcase, not here.
- `random_expected_overlap_fraction` returns 1.0 when N ≤ k (steps where the whole pool is selected, e.g.
  pos < 8192 for V3.2 at k = 2048 — irrelevant for V4 at ≥ 4K but present in V3.2 short-context rungs);
  those steps contribute lift = o_t ≈ 1 and should be excluded when N ≤ k dominates a rung.
- Scores are optional: without `scores[]`, `weighted_overlap` uses index order and the score-stability
  fields are NaN — do not compare `weighted_overlap` between a scored and an unscored trace.
- The per-run numbers are unweighted means over layers; a single outlier layer (V4 layer 2 has A@99 38 %)
  moves the mean — always report the layer range next to it.
