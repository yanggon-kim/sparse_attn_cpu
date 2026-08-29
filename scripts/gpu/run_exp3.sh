#!/bin/bash
[ -n "${WORKDIR:-}" ] && [ -f "$WORKDIR/env/activate.sh" ] && source "$WORKDIR/env/activate.sh" > /dev/null 2>&1 || true
# exp3 orchestration for one model: clean (x2) -> ctrl_identity -> ctrl_numeric -> perm_once -> perm_periodic,
# impl A for every mode, then impl B for the perm modes. One vLLM engine per (mode, rung bucket).
# Usage: run_exp3.sh <model_path> <model_tag> <manifest> <out_root> [modes...]
#   env: EXP3_SOURCES="ruler longbench_v2 infinitebench mmlu_pro gpqa" (default all), EXP3_MAXSEQ (16),
#        EXP3_LOGPROBS (20, only for the agreement sample runs), EXP3_SAMPLE_IDS_FILE (subset for the sample)
set -u
MODEL=$1; TAG=$2; MANIFEST=$3; OUT=$4; shift 4
MODES=${@:-"clean clean2 ctrl_identity ctrl_numeric perm_once perm_periodic perm_once_B perm_periodic_B"}
S=$(cd "$(dirname "$0")" && pwd)
SRCS=${EXP3_SOURCES:-"ruler longbench_v2 infinitebench mmlu_pro gpqa"}
MAXSEQ=${EXP3_MAXSEQ:-16}
for MODE in $MODES; do
  IMPL=A; base=$MODE
  case $MODE in *_B) IMPL=B; base=${MODE%_B};; esac
  case $base in
    clean|clean2)   env_mode=off;  extra="";;
    ctrl_identity)  env_mode=ctrl_identity; extra="--reindex";;
    ctrl_numeric)   env_mode=off;  extra="--max-num-seqs 1";;   # different but numerically-equivalent config
    perm_once)      env_mode=perm_once; extra="--reindex";;
    perm_periodic)  env_mode=perm_periodic; extra="--reindex";;
  esac
  for RUNG in 32768 65536 131072; do
    for SRC in $SRCS; do
      n=$(python3 - "$MANIFEST" "$RUNG" "$SRC" <<'PY'
import json,sys
m,r,s=sys.argv[1],int(sys.argv[2]),sys.argv[3]
print(sum(1 for l in open(m) if l.strip() and json.loads(l)["rung"]==r and json.loads(l)["source"]==s))
PY
)
      [ "$n" = 0 ] && continue
      suffix="_${MODE}"
      [ -f "$OUT/batch_${TAG}_${RUNG}_bf${suffix}_${SRC}.json" ] && { echo "[skip] $MODE $RUNG $SRC"; continue; }
      echo "=== $(date -u +%T) $MODE impl=$IMPL rung=$RUNG src=$SRC n=$n"
      REINDEX_MODE=$env_mode REINDEX_IMPL=$IMPL REINDEX_SEED=7 REINDEX_PERIOD=4 REINDEX_FRAC=0.10 \
      REINDEX_LOG=$OUT/permlog_${TAG}_${RUNG}${suffix}_${SRC} \
      python3 "$S/run_vllm_batch.py" --model "$MODEL" --model-tag "$TAG" --manifest "$MANIFEST" --rung $RUNG --kinds bf \
        --out-root "$OUT" --max-num-seqs ${MAXSEQ} --run-suffix "${suffix}" --no-hook --no-adapter \
        --source-filter "$SRC" --logprobs ${EXP3_LOGPROBS:-0} $extra > "$WORKDIR/logs/exp3_${TAG}_${MODE}_${RUNG}_${SRC}.log" 2>&1
      mv "$OUT/batch_${TAG}_${RUNG}_bf${suffix}.json" "$OUT/batch_${TAG}_${RUNG}_bf${suffix}_${SRC}.json" 2>/dev/null
      grep -E "Traceback|\[batch\]|\[reindex\]" "$WORKDIR/logs/exp3_${TAG}_${MODE}_${RUNG}_${SRC}.log" | tail -n 3 | cut -c1-200
    done
  done
done
echo EXP3_DONE
