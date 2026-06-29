# Benchmark inputs

Official **NVIDIA/RULER** (commit `38da79d79519ef87aa46ae804f838e1eab7f86d7`, Apache-2.0), task
`niah_single_2`, at context lengths 4K / 8K / 16K (one sample each).

- `prompts/niah_single_2_<len>_s0.txt` — the exact prompt fed to the model (RULER `input` + `answer_prefix`).
- `prompts/samples.jsonl` — per-sample metadata: needle value, reference answer, prompt SHA-256, lengths.
- `ruler_data/niah_single_2_<len>_validation.jsonl` — the raw RULER output records.

## Needles (ground truth)
| Context | needle value | key |
|--------:|-------------:|-----|
| 4K  | 2338687 | harmonious-uniform |
| 8K  | 7210606 | (see samples.jsonl) |
| 16K | 7210606 | (see samples.jsonl) |

## Regenerate
```bash
git clone https://github.com/NVIDIA/RULER && (cd RULER && git checkout 38da79d7)
# DeepSeek tokenizer: tokenizer.json + tokenizer_config.json from
#   https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash  -> ./tokenizer/
# Paul Graham haystack:
python RULER/scripts/data/synthetic/json/download_paulgraham_essay.py
# RULER's prepare.py invokes `python` (not python3): add a `python`->python3 shim to PATH.
for L in 4096 8192 16384; do
  python RULER/scripts/data/prepare.py --save_dir out/$L --benchmark synthetic \
    --task niah_single_2 --subset validation --tokenizer_path ./tokenizer \
    --tokenizer_type hf --max_seq_length $L --num_samples 1 --model_template_type base
done
# Then assemble input+answer_prefix into prompt .txt files (see ../scripts/build_prompts.py).
```
The needle values/positions depend on RULER's `--random_seed` (default 42); the files here are that seed.
