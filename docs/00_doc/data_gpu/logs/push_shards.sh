#!/bin/bash
# Bring the raw-trace run dirs from the local branch into main in ~1.5 GB commits, pushing after each.
cd <HOME>/00_sparse_attn/01_github/sparse_attn_cpu
SRC=gpu-campaign-raw-traces
LIMIT=$((1500*1024*1024))
batch=(); size=0; n=0
flush() {
  [ ${#batch[@]} = 0 ] && return
  n=$((n+1))
  git checkout -q $SRC -- "${batch[@]}"
  git add -- "${batch[@]}"
  git commit -q -F - <<EOM
raw traces batch $n: ${#batch[@]} run dirs ($(( size / 1048576 )) MB) — sharded gz JSONL + analysis from the GPU sweeps

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01W3cwGyLxyjrS7tgfrBx5mN
EOM
  for try in 1 2 3; do git push -q origin main 2>><WORKDIR>/logs/push_shards.err && break; echo "push retry $try"; sleep 20; done
  echo "$(date -u +%T) batch $n pushed: ${#batch[@]} dirs $(( size / 1048576 )) MB -> $(git rev-parse --short HEAD)"
  batch=(); size=0
}
for d in $(git ls-tree -d --name-only $SRC docs/gpu_sweep/runs/ docs/glm_sweep/runs/ | sort); do
  git ls-files --error-unmatch "$d" >/dev/null 2>&1 && continue   # already on main
  s=$(git ls-tree -r -l $SRC "$d" | awk '{sum+=$4} END{print sum+0}')
  if [ $((size + s)) -gt $LIMIT ] && [ ${#batch[@]} -gt 0 ]; then flush; fi
  batch+=("$d"); size=$((size + s))
done
flush
echo PUSH_SHARDS_DONE
