#!/bin/bash
# Prepare RULER niah_single_2 prompts at 4K/8K/16K using the DeepSeek tokenizer.
set -e
EXP=<WORKDIR>/experiment
export PATH="$EXP/shim:$PATH"   # provides `python` -> python3 for RULER subprocess
TOK="$EXP/tokenizer"
cd "$EXP/benchmark/RULER/scripts/data"
for L in 4096 8192 16384; do
  SUB=$([ "$L" = 4096 ] && echo raw || echo raw_$L)
  python3 prepare.py --save_dir "$EXP/data/$SUB" --benchmark synthetic \
    --task niah_single_2 --subset validation --tokenizer_path "$TOK" \
    --tokenizer_type hf --max_seq_length "$L" --num_samples 1 --model_template_type base
done
python3 "$EXP/scripts/build_prompts.py"
