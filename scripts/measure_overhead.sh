#!/bin/bash
# Trace-overhead measurement (doc §32): off / metadata / selected on one small sample.
# Run AFTER the main runs (it competes for CPU). Uses a short prompt for speed.
set -u
EXP=<WORKDIR>/experiment
DS4=<WORKDIR>/ds4/ds4
M=<WORKDIR>/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
OUT="$EXP/overhead"; mkdir -p "$OUT/t"
PF="$EXP/overhead_prompt.txt"
run() { # name level
  local name=$1 lvl=$2
  local env=""
  [ "$lvl" != "off" ] && env="DS4_TRACE_OUTPUT=$OUT/t DS4_TRACE_DECODE_ONLY=1 DS4_TRACE_LEVEL=$lvl"
  rm -f "$OUT/t/indexer_trace.jsonl"
  env $env /usr/bin/time -v "$DS4" --cpu -m "$M" --prompt-file "$PF" -c 2816 -t 64 --temp 0 -n 16 \
    > "$OUT/${name}.out" 2> "$OUT/${name}.time"
  local tb=0; [ -f "$OUT/t/indexer_trace.jsonl" ] && tb=$(wc -c < "$OUT/t/indexer_trace.jsonl")
  echo "$name: $(grep -oE 'prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s' "$OUT/${name}.time" || true) trace_bytes=$tb"
}
run off off
run metadata 0
run selected 1
