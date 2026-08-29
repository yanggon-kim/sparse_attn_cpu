#!/bin/bash
[ -n "${WORKDIR:-}" ] && [ -f "$WORKDIR/env/activate.sh" ] && source "$WORKDIR/env/activate.sh" > /dev/null 2>&1 || true
# Run the unchanged CPU analysis chain on ds4-schema run dirs produced by vllm_to_ds4_run.py.
# Usage: analyze_runs.sh [--fast] <run_dir> [<run_dir> ...]
#   --fast : use scripts/gpu/ingest_trace_fast.py (numpy; identical parquet schema, needed for 2K-step runs)
# Per run: ingest -> validate (exit!=0 -> WARN) -> analyze_locality -> extended_retention -> ingest_moe -> analyze_moe
set -u
S=$(cd "$(dirname "$0")/.." && pwd)
FAST=0; [ "${1:-}" = "--fast" ] && { FAST=1; shift; }
for RD in "$@"; do
  RD=${RD%/}
  [ -f "$RD/traces/indexer_trace.jsonl" ] || { echo "[skip no trace] $RD"; continue; }
  echo "--- $(date -u +%T) $RD"
  if [ $FAST = 1 ]; then python3 "$S/gpu/ingest_trace_fast.py" "$RD" || { echo "INGEST_FAIL $RD"; continue; }
  else python3 "$S/ingest_trace.py" "$RD" || { echo "INGEST_FAIL $RD"; continue; }; fi
  python3 "$S/validate_trace.py" "$RD" > "$RD/analysis/validate.log" 2>&1 && echo "validate PASS" || echo "VALIDATE_WARN $RD (see analysis/validate.log)"
  if [ $FAST = 1 ]; then python3 "$S/gpu/analyze_locality_fast.py" "$RD" || echo "ANALYZE_FAIL $RD"; else python3 "$S/analyze_locality.py" "$RD" || echo "ANALYZE_FAIL $RD"; fi
  if [ $FAST = 1 ]; then python3 "$S/gpu/extended_retention_fast.py" "$RD" > "$RD/analysis/extended_retention.log" 2>&1 || echo "EXTRET_WARN $RD"; else python3 "$S/extended_retention.py" "$RD" > "$RD/analysis/extended_retention.log" 2>&1 || echo "EXTRET_WARN $RD"; fi
  if [ -f "$RD/traces/moe_trace.jsonl" ]; then
    python3 "$S/ingest_moe_trace.py" "$RD" && python3 "$S/analyze_moe_locality.py" "$RD" || echo "MOE_FAIL $RD"
  fi
done
echo ANALYZE_RUNS_DONE
