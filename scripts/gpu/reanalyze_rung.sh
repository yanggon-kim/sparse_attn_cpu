#!/bin/bash
[ -n "${WORKDIR:-}" ] && [ -f "$WORKDIR/env/activate.sh" ] && source "$WORKDIR/env/activate.sh" > /dev/null 2>&1 || true
# Wait for a rung's chains, re-run analyze_locality_fast.py on every run (idempotent), then finish_rung.sh.
# Usage: reanalyze_rung.sh <model-tag> <model-name> <rung> <runs_list_file> <docs_sweep_dir> [parallel]
set -u
TAG=$1; NAME=$2; RUNG=$3; LIST=$4; DOCS=$5; P=${6:-14}
S=$(cd "$(dirname "$0")" && pwd)
for RD in $(cat "$LIST"); do
  for i in $(seq 1 720); do grep -q ANALYZE_RUNS_DONE "$RD/analyze.log" 2>/dev/null && break; sleep 10; done
done
echo "=== $(date -u +%T) reanalyze rung $RUNG"
cat "$LIST" | xargs -P $P -I{} sh -c 'python3 '"$S"'/analyze_locality_fast.py {} >> {}/analyze.log 2>&1'
echo "=== $(date -u +%T) reanalyze done"
bash "$S/finish_rung.sh" "$TAG" "$NAME" "$RUNG" "$LIST" "$DOCS"
