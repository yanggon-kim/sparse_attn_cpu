#!/bin/bash
# exp3 tier 2 chain: V3.2 -> GLM-5.2 -> GLM-5, one engine load each, resumable
source <WORKDIR>/env/activate.sh
S=<REPO>/scripts/gpu
cd $WORKDIR
for spec in "DeepSeek-V3.2 v32" "GLM-5.2-FP8 glm52" "GLM-5-FP8 glm5"; do
  set -- $spec
  for attempt in 1 2 3; do
    python $S/exp3_tier2.py --model $WORKDIR/models/$1 --tag t2_$2 --out $WORKDIR/runs/exp3/tier2_$2.json --resume >> $WORKDIR/logs/exp3_tier2_$2.log 2>&1
    grep -q "\[done\]" $WORKDIR/logs/exp3_tier2_$2.log && break
    echo "attempt $attempt failed for $2, cleaning up" >> $WORKDIR/logs/exp3_tier2_chain.log
    pkill -f "[e]xp3_tier2.py" ; sleep 20
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 30
  done
  echo "$2 finished $(date -u)" >> $WORKDIR/logs/exp3_tier2_chain.log
done
echo TIER2_CHAIN_DONE >> $WORKDIR/logs/exp3_tier2_chain.log
