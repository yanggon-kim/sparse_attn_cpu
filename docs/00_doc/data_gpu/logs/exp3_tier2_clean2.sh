#!/bin/bash
# after round 2: identical-rerun generation baseline for V3.2 (stream-identity noise floor)
source <WORKDIR>/env/activate.sh
S=<REPO>/scripts/gpu
while ! grep -q TIER2_ROUND2_DONE $WORKDIR/logs/exp3_tier2_chain2.log 2>/dev/null; do sleep 60; done
sleep 60
cd $WORKDIR
for attempt in 1 2 3; do
  python $S/exp3_tier2.py --model $WORKDIR/models/DeepSeek-V3.2 --tag t2_v32 --out $WORKDIR/runs/exp3/tier2_v32.json \
    --resume --skip-ppl --acc-modes clean2 >> $WORKDIR/logs/exp3_tier2_v32.log 2>&1
  tail -50 $WORKDIR/logs/exp3_tier2_v32.log | grep -q "\[done\]" && break
  pkill -f "[e]xp3_tier2.py"; sleep 20
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 40
done
echo "v32 clean2 generation finished $(date -u)" >> $WORKDIR/logs/exp3_tier2_chain2.log
echo TIER2_CLEAN2_DONE >> $WORKDIR/logs/exp3_tier2_chain2.log
