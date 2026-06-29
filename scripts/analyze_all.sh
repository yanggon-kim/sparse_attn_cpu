#!/bin/bash
# Orchestrate full post-run analysis over all completed runs (doc Part XI).
set -u
EXP=<WORKDIR>/experiment
S="$EXP/scripts"
RUNS="$EXP/runs"
echo "=== unit tests + per-run ingest/validate/analyze/report ==="
python3 "$S/validate_trace.py" || { echo "unit tests failed"; exit 1; }
for RUN in "$RUNS"/*/; do
  [ -f "$RUN/outputs/generations.jsonl" ] || { echo "[skip incomplete] $RUN"; continue; }
  [ -f "$RUN/traces/indexer_trace.jsonl" ] || { echo "[skip no trace] $RUN"; continue; }
  echo "--- $RUN ---"
  python3 "$S/ingest_trace.py" "$RUN"
  python3 "$S/validate_trace.py" "$RUN" || echo "WARN: validation issues in $RUN"
  python3 "$S/analyze_locality.py" "$RUN"
  python3 "$S/make_sample_report.py" "$RUN"
done
echo "=== combined tables + plots ==="
python3 "$S/generate_tables.py" "$RUNS" "$EXP/tables"
python3 "$S/generate_plots.py" "$RUNS" "$EXP/plots"
echo "=== DONE: tables in $EXP/tables, plots in $EXP/plots ==="
