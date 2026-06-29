# Benchmark Decision Note (doc §4)

```
Selected benchmark:        RULER (official NVIDIA/RULER)
Selected task subset:      niah_single_2  (single-needle retrieval: word key -> number value)
                           [vt multi-hop, cwe aggregation generators prepared; deferred to expansion]
Selected split/version:    validation; RULER commit pinned in code/ruler_commit.txt
Planned context lengths:   4096, 8192, 16384 tokens
Planned number of samples: 1 per context length (3 total this pass; expandable)
Planned max_new_tokens:    128 (greedy, temperature 0)
Reason for selection:      Decision-tree rule #1 (first instrumentation run -> RULER). niah gives
                           controlled context length, known evidence position (the needle), and a
                           single persistent retrieval target -> clearest first read of decode-time
                           KV-selection locality. CPU prefill (~1.2 tok/s) makes the doc's 32K/64K
                           pilot infeasible (~7-8 hr/sample); 4K/8K/16K keeps total ~7 hr while still
                           giving real CSA sparsity (n_comp >> top_k=512 above ~2K tokens).
Expected locality pattern: Single retrieval -> small persistent hot set; high adjacent-token overlap
                           and slow retention decay; the needle's compressed block retained across
                           decode steps in layers that drive the answer.
Dataset source/version:    RULER synthetic niah; haystack = Paul Graham essays
                           (json/PaulGrahamEssays.json, downloaded via RULER's script);
                           length calibrated with the DeepSeek-V4-Flash tokenizer; actual ds4 token
                           counts recorded per run (context_length_actual_tokens).
```

## Deviations from the doc (documented per decision rule #6)
- **Context length** scaled from 32K/64K to 4K/8K/16K — CPU runtime constraint (recorded above).
- **Sample count** reduced to 1/length for this pass (user-directed ~7 hr budget); pipeline supports
  arbitrary counts for the expansion.
- **Single generation per sample** serves both RULER recall scoring and the trace (the standard RULER
  prompt + substring/recall scoring on the same greedy 128-token output) — avoids a second expensive prefill.
