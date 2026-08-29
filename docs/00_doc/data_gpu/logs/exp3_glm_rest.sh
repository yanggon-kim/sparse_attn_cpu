#!/bin/bash
source <WORKDIR>/env/activate.sh
S=<HOME>/00_sparse_attn/01_github/sparse_attn_cpu/scripts/gpu
cd $WORKDIR
python $S/exp3_tier1.py --model $WORKDIR/models/GLM-5-FP8 --tag glm5 --out $WORKDIR/runs/exp3/tier1_glm5.json > $WORKDIR/logs/exp3_tier1_glm5.log 2>&1
echo "GLM5 exit $? $(date -u +%T)"
python $S/exp3_tier1.py --model $WORKDIR/models/GLM-5.2-FP8 --tag glm52acc --out $WORKDIR/runs/exp3/tier1_glm52_acc.json --skip-ppl > $WORKDIR/logs/exp3_tier1_glm52_acc.log 2>&1
echo "GLM52 ACC exit $? $(date -u +%T)"
echo EXP3_GLM_REST_DONE
