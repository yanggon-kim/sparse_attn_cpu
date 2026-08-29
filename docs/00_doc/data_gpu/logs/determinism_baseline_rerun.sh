#!/bin/bash
source <WORKDIR>/env/activate.sh
S=<HOME>/00_sparse_attn/01_github/sparse_attn_cpu/scripts/gpu
while ! grep -q SWEEP_DONE $WORKDIR/logs/determinism_sweep.log 2>/dev/null; do sleep 20; done
python $S/determinism_probe.py --model $WORKDIR/models/DeepSeek-V3.2 --prompt-file $WORKDIR/prompts/ladder/ruler_niah_single_2_8192_s00.txt --filler-files $WORKDIR/prompts/ladder/ruler_niah_single_2_8192_s01.txt $WORKDIR/prompts/ladder/ruler_niah_single_2_8192_s02.txt $WORKDIR/prompts/ladder/ruler_niah_single_2_8192_s03.txt --n-tokens 512 --repeats 3 --out $WORKDIR/runs/determinism/baseline.json > $WORKDIR/logs/determinism_baseline.log 2>&1
echo BASELINE_RERUN_DONE
