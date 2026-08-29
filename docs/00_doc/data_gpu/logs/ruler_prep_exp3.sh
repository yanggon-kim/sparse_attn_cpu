#!/bin/bash
source <WORKDIR>/env/activate.sh
cd $WORKDIR/benchmark/RULER/scripts/data
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2"
for L in 32768 65536 131072; do for T in $TASKS; do
  [ -f $WORKDIR/prompts/ruler_exp3/$L/$T/validation.jsonl ] && continue
  echo "== $T $L $(date -u +%T)"
  python prepare.py --save_dir $WORKDIR/prompts/ruler_exp3/$L --benchmark synthetic --task $T --subset validation \
    --tokenizer_path $WORKDIR/models/DeepSeek-V3.2 --tokenizer_type hf --max_seq_length $L --num_samples 100 \
    --random_seed 42 --model_template_type base 2>&1 | tail -n 2
done; done; echo RULER_EXP3_PREP_DONE
