# exp2 — GLM-5 / GLM-5.2 gather-index statistics (cross-model generality)

*Goal: the exp1 curves on a second DSA model family, same prompts, same ladder, same scripts. Output:
`docs/glm_sweep/` with V3.2-vs-GLM side-by-side. Budget ~30 GPU-h. Run only after the exp1 ladder is
committed.*

## 1. What is different from exp1

- **Model class**: `GlmMoeDsaForCausalLM` (`vllm/model_executor/models/registry.py:117`) subclasses
  `DeepseekV2ForCausalLM` (`deepseek_v2.py:1930`) — same indexer, same attention backend, same
  `forward_mqa` hook point, same 656 B / 132 B cache layout (verify `kv_lora_rank`, `qk_rope_head_dim`,
  `index_head_dim` in its config; if they differ the byte sizes differ, not the hook).
- **Index sharing ("IndexShare")**: with `index_topk_freq = F > 1` only every F-th backbone layer runs the
  indexer; the others reuse the shared `topk_indices_buffer` left by the last computing layer
  (`deepseek_v2.py:1085-1103`: `is_v32 = hasattr(config, "index_topk")`; skip rule
  `max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0`, or the explicit
  `index_topk_pattern` string with `"S"` = skip). Checkpoint indexer weights of skipped layers are
  dropped at load (`:1566-1575`). Expected for GLM-5.2: `F = 4`, offset 2 → **VERIFY** from `config.json`.
- **GLM-5 (plain)**: whether it has DSA at all is not established locally. Decide from its `config.json`:
  `index_topk` present → same path, run it; absent → dense attention, no indexer, **skip GLM-5**, say so
  in the summary and in the exp2 gate report, and continue with GLM-5.2 only (this is a method-changing
  VERIFY outcome — `GPU_CAMPAIGN.md` §5(b) — report it, but GLM-5.2 alone still satisfies exp2).
  Do not spend GPU-h on a download before this check.

## 2. What to record (additions to the exp1 record)

```json
{"req": "...", "layer": 5, "pos": 32767, "n_comp": 32768, "top_k": 2048, "valid_k": 2048,
 "sel": [...], "topk_computed": false, "shared_from_layer": 4, "phase": 1}
```
- `topk_computed`: true if this layer's indexer ran (`not _skip_topk`, read it from the attention module
  at hook time — the `Indexer`/`DeepseekV2MLAAttention` object carries the flag; log which attribute you
  used). false → the record duplicates the producing layer's set; keep it (the *gather* happens in every
  layer, which is what the memory system sees) but mark it.
- `shared_from_layer`: the last layer with `topk_computed == true` at this step.
- Everything else as exp1 §2; MoE hook unchanged (record `n_expert`, `n_used` from the config).

Adapter (exp1 §4): identical, plus `model_config.json` gets `index_topk_freq`, `index_skip_topk_offset`,
`index_topk_pattern` and each `layer_map` entry gets `topk_computed`. The analysis scripts ignore the
extra fields. **Report two variants of every R1/R3 number**: (a) all 61 layers (memory-system view),
(b) computing layers only (algorithm view; the shared layers would otherwise inflate per-step overlap
by construction — a step's shared layers agree with their producer at 1.0).

## 3. Verification (smoke gate, 8K RULER prompt, 256 steps)

Self-check as in exp1 §3: green → ladder; one retry on failure, then stop.

- [ ] the set of layers with `topk_computed == true` equals the skip-rule prediction from the config
- [ ] for a skipped layer, `sel` equals `sel` of `shared_from_layer` at the same step, bit-for-bit
- [ ] all exp1 §3 checks (range, count, hook-on == hook-off, batch attribution, full chain)
- [ ] `n_comp` and `pos` consistent with GLM's tokenizer (prompt token counts differ from V3.2 — record both)

## 4. Ladder

Same prompts as exp1 (re-tokenized; a rung is defined by the *V3.2* token count, GLM's count recorded
in `context_length_actual_tokens`), same two run kinds (bf / ld), n ≈ 20 bf + 5 ld per rung, batch 8–16.
If budget is short, drop 128K first, then the ld kind at 16K/32K.

## 5. Outputs → `docs/glm_sweep/`

- `R1_kv_locality.csv`, `R2_moe_locality.csv`, `R3_hotset_coverage.csv` as exp1 §6, one row set per
  model (`DeepSeek-V3.2`, `GLM-5.2`, `GLM-5` if run), with the (a)/(b) variant column for GLM.
- `side_by_side.png`: adjacent overlap and lift vs context, one line per model, V4-CPU dotted.
- `glm_sweep_summary.md`: the config dicts, the skip-rule layer list, the (a) vs (b) delta, and the
  one-sentence generality claim for the paper (with run ids).
- Gate (self-check): side-by-side curves and the summary committed → `docs/00_doc/reports/exp2_<date>.md`,
  then continue to exp3.
