#!/bin/bash
# exp3 tier 2, round 2: (a) V3.2 re-indexed generation, (b) GLM-5.2 PPL, (c) GLM-5 PPL. One engine load each, resumable.
source <WORKDIR>/env/activate.sh
S=<REPO>/scripts/gpu
cd $WORKDIR
run() {   # $1 tag  $2 model dir  $3.. extra args
  local tag=$1 model=$2; shift 2
  for attempt in 1 2 3; do
    python $S/exp3_tier2.py --model $WORKDIR/models/$model --tag t2_$tag --out $WORKDIR/runs/exp3/tier2_$tag.json --resume "$@" \
      >> $WORKDIR/logs/exp3_tier2_$tag.log 2>&1
    tail -50 $WORKDIR/logs/exp3_tier2_$tag.log | grep -q "\[done\]" && break
    echo "attempt $attempt failed for $tag $(date -u)" >> $WORKDIR/logs/exp3_tier2_chain2.log
    pkill -f "[e]xp3_tier2.py"; sleep 20
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 40
  done
  echo "$tag finished $(date -u)" >> $WORKDIR/logs/exp3_tier2_chain2.log
}
run v32   DeepSeek-V3.2 --skip-ppl
run glm52 GLM-5.2-FP8   --skip-acc
run glm5  GLM-5-FP8     --skip-acc
echo TIER2_ROUND2_DONE >> $WORKDIR/logs/exp3_tier2_chain2.log
