#!/bin/bash
[ -n "${WORKDIR:-}" ] && [ -f "$WORKDIR/env/activate.sh" ] && source "$WORKDIR/env/activate.sh" > /dev/null 2>&1 || true
# After the per-run chains of a rung are done: hot-set coverage (R3) for every run of the rung, then package
# raw traces + analysis into the repo (sharded), then aggregate the model's tables.
# Usage: finish_rung.sh <model-tag> <model-name> <rung> <runs_list_file> <docs_sweep_dir>
set -u
TAG=$1; NAME=$2; RUNG=$3; LIST=$4; DOCS=$5
S=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$S/.." && pwd)
# wait for chains
for RD in $(cat "$LIST"); do
  for i in $(seq 1 720); do grep -q ANALYZE_RUNS_DONE "$RD/analyze.log" 2>/dev/null && break; sleep 10; done
  grep -q ANALYZE_RUNS_DONE "$RD/analyze.log" || echo "TIMEOUT waiting $RD"
done
echo "=== $(date -u +%T) hotset for rung $RUNG"
python3 "$S/analyze_hotset_coverage.py" "$WORKDIR/runs/_hotset_${TAG}_${RUNG}" $(cat "$LIST") > "$WORKDIR/logs/hotset_${TAG}_${RUNG}.log" 2>&1 || echo "HOTSET_FAIL"
echo "=== $(date -u +%T) sharding"
for RD in $(cat "$LIST"); do
  case "$RD" in *_b) # derived variant (computing-only adapter): package analysis only, traces are a subset of the _a run
    python3 "$S/gpu/shard_traces.py" --run-dir "$RD" --dest "$REPO/$DOCS/runs/$(basename $RD)" --no-traces | tail -1;;
  *) python3 "$S/gpu/shard_traces.py" --run-dir "$RD" --dest "$REPO/$DOCS/runs/$(basename $RD)" | tail -1;;
  esac
done
echo "=== $(date -u +%T) aggregate"
python3 "$S/gpu/aggregate_gpu_sweep.py" --runs-root "$WORKDIR/runs" --model-tag "$TAG" --model-name "$NAME" --out "$REPO/$DOCS"
echo FINISH_RUNG_DONE
