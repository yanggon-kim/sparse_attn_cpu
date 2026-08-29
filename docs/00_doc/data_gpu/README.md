# GPU campaign — node-side artifacts (added 2026-08-29 so nothing depends on the rented node)
- `determinism/` — the 11 determinism-probe JSONs (token ids + top-1 logprobs, 3 solo runs + 1 batched per config).
- `batch_summaries/` — per-rung runner summaries (`batch_<tag>_<rung>_bf_ld.json`, smoke batches).
- `prompts_ladder/` — the 135 exp1/exp2 prompts (raw user text) + `manifest.jsonl`.
- `prompts_exp3/` — exp3 manifests (full 5,932-item set and the tier-1 subset); texts regenerate deterministically with
  `scripts/gpu/build_exp3_prompts.py` (RULER via prepare.py seed 42, LongBench-v2/InfiniteBench/MMLU-Pro from HF).
- `logs/` — runner/analysis/probe logs and the orchestration scripts used on the node.
- `env/` — `pip_freeze.txt`, `activate.sh`, `node.json`, `nvidia-smi` snapshot.
- `reindex_unit/`, `smoke0/` — exp3 hook unit test and the exp0 3-token smoke trace.
- `../../reindex_accuracy/permlogs/` — exp3 permutation logs (tar.gz, 45 MB parts: `cat <name>.tar.gz.part* | tar xz`).
- Smoke run dirs (`*_smoke*`, `v32_smoke0_capital_s0`) are packaged under `docs/{gpu,glm}_sweep/runs/` like the ladder runs.
