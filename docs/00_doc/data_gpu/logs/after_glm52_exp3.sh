#!/bin/bash
source <WORKDIR>/env/activate.sh
S=<HOME>/00_sparse_attn/01_github/sparse_attn_cpu/scripts/gpu
while ! grep -qE "\[done\]|Traceback" $WORKDIR/logs/exp3_tier1_glm52.log 2>/dev/null; do sleep 20; done
sleep 30
cd $WORKDIR && python $S/exp3_tier1.py --model $WORKDIR/models/GLM-5-FP8 --tag glm5 --out $WORKDIR/runs/exp3/tier1_glm5.json > $WORKDIR/logs/exp3_tier1_glm5.log 2>&1
echo GLM5_TIER1_DONE
