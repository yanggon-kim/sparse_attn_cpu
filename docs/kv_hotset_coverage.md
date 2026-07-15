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
