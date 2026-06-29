#!/bin/bash
# Run the niah_single_2 sparse-attention KV trace experiment on CPU.
# Sequential over context lengths (resumable: skips completed runs).
# Deterministic greedy decode, Level-3 trace (decode-phase only), per-token timing.
set -u
EXP=<WORKDIR>/experiment
DS4=<WORKDIR>/ds4/ds4
M=<WORKDIR>/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
TASK=niah_single_2
LENGTHS=${1:-"4096 8192 16384"}

# --- code provenance (once) ---
mkdir -p "$EXP/code"
{ echo "ds4 build: make cpu  (gcc -O3 -ffast-math -march=native -std=c99)"; "$DS4" --help 2>/dev/null | head -1; } > "$EXP/code/ds4_build.txt"
sha256sum <WORKDIR>/ds4/ds4.c | awk '{print "ds4.c sha256:",$1}' >> "$EXP/code/ds4_build.txt"
git -C "$EXP/benchmark/RULER" rev-parse HEAD > "$EXP/code/ruler_commit.txt" 2>/dev/null
echo "default (numactl not installed; single-process, 64 threads, no explicit NUMA binding)" > "$EXP/code/numa_policy.txt"

NUMACTL=""
command -v numactl >/dev/null 2>&1 && NUMACTL="numactl --interleave=all"

for L in $LENGTHS; do
  RUN_ID="${TASK}_${L}_q2"
  RUN="$EXP/runs/$RUN_ID"
  if [ -f "$RUN/outputs/generations.jsonl" ]; then
    echo "[skip] $RUN_ID already complete"; continue
  fi
  mkdir -p "$RUN/traces/full_scores" "$RUN/outputs" "$RUN/logs" "$RUN/prompts" "$RUN/analysis"
  PF="$EXP/prompts/${TASK}_${L}_s0.txt"
  cp "$PF" "$RUN/prompts/"
  grep "\"${TASK}_${L}_s0\"" "$EXP/prompts/samples.jsonl" > "$RUN/prompts/sample.json"
  SAMPLE_JSON=$(cat "$RUN/prompts/sample.json")
  CTX=$(( L + 768 ))
  echo "[run] $RUN_ID ctx=$CTX  $(date -u +%H:%M:%S)"
  free -h | head -2 > "$RUN/logs/mem_before.log"

  DS4_TRACE_OUTPUT="$RUN/traces" \
  DS4_TRACE_LEVEL=3 \
  DS4_TRACE_DECODE_ONLY=1 \
  DS4_TRACE_FULL_SCORE_SAMPLE_RATE=0.002 \
  DS4_TRACE_FLUSH_INTERVAL=32 \
  DS4_TOKEN_TIMING=1 \
  OMP_NUM_THREADS=64 \
  $NUMACTL /usr/bin/time -v "$DS4" --cpu -m "$M" \
      --prompt-file "$PF" -c "$CTX" -t 64 --temp 0 -n 128 \
      --dump-logprobs "$RUN/outputs/logprobs.json" \
      > "$RUN/logs/stdout.log" 2> "$RUN/logs/time_and_stderr.log"
  RC=$?
  free -h | head -2 > "$RUN/logs/mem_after.log"
  echo "RUN_EXIT=$RC" >> "$RUN/logs/time_and_stderr.log"
  grep -oE "pswpin [0-9]+|pswpout [0-9]+" /proc/vmstat > "$RUN/logs/vmstat_swap.log" 2>/dev/null

  if [ $RC -ne 0 ]; then
    echo "[ERROR] $RUN_ID failed rc=$RC — see logs/time_and_stderr.log; not finalizing." | tee -a "$RUN/logs/finalize.log"
    continue
  fi
  python3 "$EXP/scripts/finalize_run.py" "$RUN" "$L" "$PF" "$SAMPLE_JSON" 2>&1 | tee -a "$RUN/logs/finalize.log"
  echo "[done] $RUN_ID rc=$RC trace_lines=$(wc -l < "$RUN/traces/indexer_trace.jsonl" 2>/dev/null)  $(date -u +%H:%M:%S)"
done
echo "ALL_RUNS_COMPLETE $(date -u +%H:%M:%S)"
