#!/bin/bash
# Run the full R1/R2/R3 analysis chain on every completed LongBench run (runs/lb_*_q2).
# Idempotent: re-analyzes only runs missing analysis/metrics_run_summary.json unless FORCE=1.
set -u
EXP=<WORKDIR>/experiment
cd "$EXP"
for RUN in runs/lb_*_q2; do
  [ -d "$RUN" ] || continue
  if [ ! -f "$RUN/outputs/generations.jsonl" ] || [ ! -s "$RUN/traces/indexer_trace.jsonl" ]; then
    echo "[skip] $(basename "$RUN") (incomplete)"; continue
  fi
  if [ "${FORCE:-0}" != "1" ] && [ -f "$RUN/analysis/hotset_coverage.json" ]; then
    echo "[ok]   $(basename "$RUN") already analyzed"; continue
  fi
  echo "=== $(basename "$RUN") ==="
  python3 scripts/ingest_trace.py       "$RUN" || { echo "[FAIL ingest] $RUN"; continue; }
  python3 scripts/validate_trace.py     "$RUN" > "$RUN/analysis/validate.log" 2>&1 \
      && echo "  validate: PASS" || echo "  validate: FAIL (see analysis/validate.log)"
  python3 scripts/analyze_locality.py   "$RUN" || echo "[FAIL analyze_locality] $RUN"
  python3 scripts/ingest_moe_trace.py   "$RUN" || echo "[FAIL ingest_moe] $RUN"
  python3 scripts/analyze_moe_locality.py      "$RUN" || echo "[FAIL analyze_moe] $RUN"
  python3 scripts/analyze_moe_concentration.py /tmp/lb_conc_plots "$RUN" || echo "[FAIL moe_conc] $RUN"
  python3 scripts/analyze_hotset_coverage.py   /tmp/lb_hotset_plots "$RUN" || echo "[FAIL hotset] $RUN"
done
echo "ANALYZE_ALL_DONE"
