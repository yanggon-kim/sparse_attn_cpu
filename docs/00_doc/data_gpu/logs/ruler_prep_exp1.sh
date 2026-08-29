#!/bin/bash
source <WORKDIR>/env/activate.sh
cd $WORKDIR/benchmark/RULER/scripts/data
for L in 8192 16384 32768 65536 131072; do for T in niah_single_2 qa_1; do
  [ -f $WORKDIR/prompts/ruler/$L/$T/validation.jsonl ] && continue
  echo "== $T $L $(date -u +%T)"
  python prepare.py --save_dir $WORKDIR/prompts/ruler/$L --benchmark synthetic --task $T --subset validation \
    --tokenizer_path $WORKDIR/models/DeepSeek-V3.2 --tokenizer_type hf --max_seq_length $L --num_samples 8 \
    --random_seed 42 --model_template_type base 2>&1 | tail -n 3
done; done; echo RULER_PREP_DONE
