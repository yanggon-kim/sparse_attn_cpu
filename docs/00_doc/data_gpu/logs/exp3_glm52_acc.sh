#!/bin/bash
source <WORKDIR>/env/activate.sh
S=<HOME>/00_sparse_attn/01_github/sparse_attn_cpu/scripts/gpu
while ! grep -q GLM5_TIER1_DONE $WORKDIR/logs/after_glm52_exp3.log 2>/dev/null; do sleep 20; done
sleep 30
cd $WORKDIR && python $S/exp3_tier1.py --model $WORKDIR/models/GLM-5.2-FP8 --tag glm52acc --out $WORKDIR/runs/exp3/tier1_glm52_acc.json --skip-ppl > $WORKDIR/logs/exp3_tier1_glm52_acc.log 2>&1
echo "GLM52_ACC_DONE $?"
