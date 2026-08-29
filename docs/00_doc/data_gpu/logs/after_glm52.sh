#!/bin/bash
source <WORKDIR>/env/activate.sh
S=<HOME>/00_sparse_attn/01_github/sparse_attn_cpu/scripts/gpu
while ! grep -q LADDER_DONE $WORKDIR/logs/ladder_glm52.log 2>/dev/null; do sleep 30; done
# re-launch the (b)-variant packaging for GLM-5.2 with the patched finish_rung (analysis only)
for R in 8192 16384 32768 65536 131072; do
  setsid nohup bash $S/finish_rung.sh glm52 GLM-5.2 $R $WORKDIR/logs/rung${R}_runs_glm52_b.txt docs/glm_sweep > $WORKDIR/logs/finish_glm52_${R}_b.log 2>&1 < /dev/null &
done
bash $S/run_ladder.sh $WORKDIR/models/GLM-5-FP8 glm5 GLM-5 docs/glm_sweep 8192 16384 32768 65536 131072
