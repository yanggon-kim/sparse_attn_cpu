#!/bin/bash
[ -n "${WORKDIR:-}" ] && [ -f "$WORKDIR/env/activate.sh" ] && source "$WORKDIR/env/activate.sh" > /dev/null 2>&1 || true
# Run the whole ladder for one model sequentially (exp1 §5 / exp2 §4) and launch the analysis queue + packaging.
# Usage: run_ladder.sh <model_path> <model_tag> <model_name> <docs_sweep_dir> [--computing-only] [rungs...]
#   e.g. run_ladder.sh $WORKDIR/models/GLM-5.2-FP8 glm52 GLM-5.2 docs/glm_sweep --computing-only 8192 16384 32768 65536 131072
set -u
MODEL=$1; TAG=$2; NAME=$3; DOCS=$4; shift 4
EXTRA=""; if [ "${1:-}" = "--computing-only" ]; then EXTRA="--computing-only"; shift; fi
RUNGS=${@:-"8192 16384 32768 65536 131072"}
S=$(cd "$(dirname "$0")" && pwd)
MAN=$WORKDIR/prompts/ladder/manifest.jsonl
for R in $RUNGS; do
  MAXSEQ=32; [ "$R" -ge 131072 ] && MAXSEQ=24
  if [ -f "$WORKDIR/runs/batch_${TAG}_${R}_bf_ld.json" ]; then echo "[skip] rung $R already generated"; else
    echo "=== $(date -u +%T) rung $R generate"
    python3 "$S/run_vllm_batch.py" --model "$MODEL" --model-tag "$TAG" --model-name "$NAME" --manifest "$MAN" --rung $R --kinds bf ld \
      --out-root "$WORKDIR/runs" --max-num-seqs $MAXSEQ $EXTRA > "$WORKDIR/logs/ladder_${TAG}_${R}.log" 2>&1
    grep -E "Traceback|\[batch\]|\[selhook\]|\[done\]" "$WORKDIR/logs/ladder_${TAG}_${R}.log" | tail -n 3 | cut -c1-200
  fi
  python3 - "$WORKDIR/runs/batch_${TAG}_${R}_bf_ld.json" "$WORKDIR/runs" > "$WORKDIR/logs/rung${R}_runs_${TAG}.txt" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d["runs"]:
    print(sys.argv[2] + "/" + r["run_id"])
PY
  if [ -n "$EXTRA" ]; then sed 's/$/_b/' "$WORKDIR/logs/rung${R}_runs_${TAG}.txt" > "$WORKDIR/logs/rung${R}_runs_${TAG}_b.txt"; fi
  setsid nohup bash "$S/queue_analysis.sh" 350 20 "$WORKDIR/logs/rung${R}_runs_${TAG}.txt" $([ -n "$EXTRA" ] && echo "$WORKDIR/logs/rung${R}_runs_${TAG}_b.txt") > "$WORKDIR/logs/queue_${TAG}_${R}.log" 2>&1 < /dev/null &
  setsid nohup bash "$S/finish_rung.sh" "$TAG" "$NAME" "$R" "$WORKDIR/logs/rung${R}_runs_${TAG}.txt" "$DOCS" > "$WORKDIR/logs/finish_${TAG}_${R}.log" 2>&1 < /dev/null &
  if [ -n "$EXTRA" ]; then
    setsid nohup bash "$S/finish_rung.sh" "$TAG" "$NAME" "$R" "$WORKDIR/logs/rung${R}_runs_${TAG}_b.txt" "$DOCS" > "$WORKDIR/logs/finish_${TAG}_${R}_b.log" 2>&1 < /dev/null &
  fi
done
echo LADDER_DONE
