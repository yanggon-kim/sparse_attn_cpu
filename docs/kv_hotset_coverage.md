# Hot-set coverage of the top-k KV selection (DeepSeek-V4-Flash, ds4 CPU)

An applied read of the CSA lightning-indexer locality: of all candidate compressed-KV positions, how many must
be kept **hot** (in fast memory) to cover the top-k=512 selection over a decode? Computed from the existing 5
runs (4K/8K/16K/32K/64K) — no re-runs. Script: `scripts/analyze_hotset_coverage.py`; per-run numbers in
`moe_locality_sweep/<ctx>/hotset_coverage.json`; figures `moe_locality_sweep/plots/hotset_0{1,2}_*.png`.

## Two definitions (per CSA layer, then averaged over the 21 CSA layers)
For a layer, count each candidate position's selection frequency over all decode steps; rank positions by
frequency; pool size `N` = max candidates visible (final pool).
- **Metric A — serve 99% of accesses (mean hit-rate).** Smallest freq-ranked hot set whose selections account
  for 99% of all top-k selection *events*. A cache of these serves 99% of accesses (1% mean miss).
- **Metric B — never stall.** Smallest fixed hot set such that ≥99% of each step's 512 selected entries are
  present on ≥99% of steps. Stricter; always ≥ A.

## Result — the hot-set fraction shrinks sharply with context

| context | candidate pool | kept / step | A: 99% of accesses | A: 99.9% | B: never-stall | full union |
|---|--:|--:|--:|--:|--:|--:|
| 4K | ~1,030 | 50% | **80%** (~819) | 88% | 88% | 93% |
| 8K | ~1,912 | 27% | **66%** (~1,256) | 77% | 78% | 81% |
| 16K | ~4,095 | 12.5% | **48%** (~1,945) | 57% | 57% | 58% |
| 32K | ~8,039 | 6.4% | **31%** (~2,524) | 37% | 38% | 38% |
| 64K | ~16,393 | 3.1% | **20%** (~3,344) | 24% | 25% | 25% |

**Headline:** to cover 99% of the top-k selection (access-weighted) you need ~80% of candidates at 4K, falling
to **~20% at 64K**. The sparser the regime (longer context), the more concentrated the selection — so the hot
set is a *shrinking fraction* of the (growing) pool.

## Forward sweep — coverage at a fixed hot-set budget
The inverse view: fix the hot set at a given **% of the candidate pool** and read off what fraction of top-k
accesses it covers (access-weighted, mean over the 21 CSA layers). Cell = % of accesses covered.

| hot set (% of pool) | 4K | 8K | 16K | 32K | 64K |
|---|--:|--:|--:|--:|--:|
| 1% | 2.0 | 3.7 | 7.9 | 15.0 | 27.8 |
| 2% | 4.1 | 7.4 | 15.4 | 27.7 | 46.5 |
| 5% | 10.2 | 18.4 | 34.7 | 54.1 | 75.1 |
| 10% | 20.1 | 34.9 | 57.5 | 76.7 | 90.7 |
| 15% | 29.8 | 49.2 | 72.0 | 87.6 | 96.0 |
| 20% | 39.5 | 60.8 | 81.5 | 93.2 | 98.1 |
| 25% | 48.6 | 70.3 | 87.9 | 96.3 | 99.1 |
| 30% | 57.0 | 77.9 | 92.0 | 98.1 | 99.6 |
| 40% | 71.7 | 88.3 | 96.6 | 99.6 | 99.9 |
| 50% | 83.2 | 94.4 | 98.7 | 100.0 | 100.0 |
| 75% | 97.9 | 99.6 | 100.0 | 100.0 | 100.0 |
| 100% | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

The longer the context, the more **front-loaded** the coverage: at 64K, **10% of the pool covers ~91%** of
accesses and 20% covers ~98%; at 4K the same budgets cover only 20% / 40% (top-512 of ~1,030 keeps ~half the
pool each step → near-linear, no concentrated hot core). Full grid per run in
`moe_locality_sweep/<ctx>/hotset_coverage.json` (`coverage_by_pool_pct`).

## Four notes
1. **Fraction shrinks (80%→20%), absolute size grows (~819→~3,344 positions).** The pool grows faster than the
   fraction shrinks; a physical cache is ~1.6× (4K) → ~6.5× (64K) the per-step 512-budget.
2. **Never-stall (B) barely exceeds access-weighted (A), and ≈ the full working set.** E.g. 64K: A@99 = 20%,
   B = 25%, union = 25%. Serving 99% of *accesses* is cheap; *never* missing >1% on any step costs nearly the
   whole working set — the gap A→B is the cold tail of rarely-reused positions.
3. **Large per-layer spread.** At 64K, A@99 ranges **10%–38%** across the 21 CSA layers (deep/stable layers need
   the smallest hot set) → per-layer cache budgets pay off.
4. **Caveat.** The candidate pool grows during decode, so a late-appearing position has fewer chances to be
   selected; using max pool size as the denominator and raw frequency is a first-order choice (slightly
   conservative for A). IQ2 / CPU / n=1 per length, as elsewhere.
