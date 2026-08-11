#!/bin/bash
# LongBench summarization collection on ds4 (DeepSeek-V4-Flash IQ2, CPU).
# Runs every sample in prompts/longbench_samples.jsonl sequentially (multi_news -> gov_report -> qmsum,
# shortest-first within each task), emitting BOTH the KV indexer trace and the MoE trace per run.
# Resumable: skips runs whose outputs/generations.jsonl exists.
# Usage: run_longbench.sh [max_runs]   (optional cap, e.g. 1 for a smoke run)
set -u
EXP=<WORKDIR>/experiment
DS4=<WORKDIR>/ds4/ds4
M=<WORKDIR>/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MANIFEST="$EXP/prompts/longbench_samples.jsonl"
MAXRUNS=${1:-9999}

mkdir -p "$EXP/code"
{ echo "ds4 build: make cpu  (gcc -O3 -ffast-math -march=native -std=c99)"; "$DS4" --help 2>/dev/null | head -1; } > "$EXP/code/ds4_build_longbench.txt"
sha256sum <WORKDIR>/ds4/ds4.c | awk '{print "ds4.c sha256:",$1}' >> "$EXP/code/ds4_build_longbench.txt"

n_done=0
while IFS= read -r SAMPLE_JSON; do
  [ $n_done -ge $MAXRUNS ] && break
  SID=$(echo "$SAMPLE_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_id'])")
  PF=$(echo "$SAMPLE_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['prompt_file'])")
  LEN=$(echo "$SAMPLE_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['context_length_target'])")
  RUN_ID="${SID}_q2"
  RUN="$EXP/runs/$RUN_ID"
  if [ -f "$RUN/outputs/generations.jsonl" ]; then
    echo "[skip] $RUN_ID already complete"; continue
  fi
  mkdir -p "$RUN/traces/full_scores" "$RUN/outputs" "$RUN/logs" "$RUN/prompts" "$RUN/analysis"
  cp "$PF" "$RUN/prompts/"
  echo "$SAMPLE_JSON" > "$RUN/prompts/sample.json"
  CTX=$(( LEN + 512 + 256 ))
  echo "[run] $RUN_ID prompt_tok=$LEN ctx=$CTX  $(date -u +%F_%H:%M:%S)"
  free -h | head -2 > "$RUN/logs/mem_before.log"

  DS4_TRACE_OUTPUT="$RUN/traces" \
  DS4_MOE_TRACE=1 \
  DS4_TRACE_LEVEL=3 \
  DS4_TRACE_DECODE_ONLY=1 \
  DS4_TRACE_FULL_SCORE_SAMPLE_RATE=0.002 \
  DS4_TRACE_FLUSH_INTERVAL=32 \
  DS4_TOKEN_TIMING=1 \
  OMP_NUM_THREADS=64 \
  /usr/bin/time -v "$DS4" --cpu -m "$M" \
      --prompt-file "$PF" -c "$CTX" -t 64 --temp 0 -n 512 \
      --dump-logprobs "$RUN/outputs/logprobs.json" \
      > "$RUN/logs/stdout.log" 2> "$RUN/logs/time_and_stderr.log"
  RC=$?
  free -h | head -2 > "$RUN/logs/mem_after.log"
  echo "RUN_EXIT=$RC" >> "$RUN/logs/time_and_stderr.log"

  if [ $RC -ne 0 ]; then
    echo "[ERROR] $RUN_ID failed rc=$RC — see logs/time_and_stderr.log; not finalizing." | tee -a "$RUN/logs/finalize.log"
    n_done=$((n_done+1)); continue
  fi
  python3 "$EXP/scripts/finalize_run.py" "$RUN" "$LEN" "$PF" "$SAMPLE_JSON" 2>&1 | tee -a "$RUN/logs/finalize.log"
  echo "[done] $RUN_ID rc=$RC kv=$(wc -l < "$RUN/traces/indexer_trace.jsonl" 2>/dev/null) moe=$(wc -l < "$RUN/traces/moe_trace.jsonl" 2>/dev/null)  $(date -u +%F_%H:%M:%S)"
  n_done=$((n_done+1))
done < "$MANIFEST"
echo "ALL_LONGBENCH_RUNS_COMPLETE $(date -u +%F_%H:%M:%S)"
