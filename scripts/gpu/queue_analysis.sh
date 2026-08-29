#!/bin/bash
[ -n "${WORKDIR:-}" ] && [ -f "$WORKDIR/env/activate.sh" ] && source "$WORKDIR/env/activate.sh" > /dev/null 2>&1 || true
# Memory-aware analysis queue: run analyze_runs.sh --fast on every listed run dir that is not done and not
# currently being processed, starting a new one only when free memory and the running count allow.
# Usage: queue_analysis.sh <min_free_gb> <max_running> <runs_list_file> [<runs_list_file> ...]
set -u
MINFREE=$1; MAXRUN=$2; shift 2
S=$(cd "$(dirname "$0")" && pwd)
LISTS="$@"
while true; do
  pending=0
  for L in $LISTS; do for RD in $(cat "$L"); do
    grep -q ANALYZE_RUNS_DONE "$RD/analyze.log" 2>/dev/null && continue
    pending=$((pending+1))
    pgrep -f "analyze_runs.sh --fast $RD\$" > /dev/null && continue        # in progress
    pgrep -f "$RD" | grep -v pgrep > /dev/null && continue                 # some stage running on it
    free=$(free -g | awk '/Mem:/{print $7}')
    running=$(pgrep -c -f '^bash .*analyze_runs.sh --fast')
    if [ "$free" -ge "$MINFREE" ] && [ "$running" -lt "$MAXRUN" ]; then
      echo "$(date -u +%T) start $RD (free ${free}G, running $running)"
      setsid nohup bash "$S/analyze_runs.sh" --fast "$RD" > "$RD/analyze.log" 2>&1 < /dev/null &
      sleep 20
    fi
  done; done
  [ "$pending" = 0 ] && { echo "QUEUE_DONE $(date -u +%T)"; exit 0; }
  sleep 30
done
