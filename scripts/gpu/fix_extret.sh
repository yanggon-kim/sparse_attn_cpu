#!/bin/bash
[ -n "${WORKDIR:-}" ] && [ -f "$WORKDIR/env/activate.sh" ] && source "$WORKDIR/env/activate.sh" > /dev/null 2>&1 || true
# Re-run extended_retention_fast.py on runs whose chain reported EXTRET_WARN (killed original), after the chain is done.
# Usage: fix_extret.sh <runs_list_file>...
set -u
S=$(cd "$(dirname "$0")" && pwd)
for L in "$@"; do for RD in $(cat "$L"); do
  for i in $(seq 1 720); do grep -q ANALYZE_RUNS_DONE "$RD/analyze.log" 2>/dev/null && break; sleep 10; done
  if grep -q EXTRET_WARN "$RD/analyze.log" 2>/dev/null || [ ! -f "$RD/analysis/extended_retention.json" ]; then
    echo "$(date -u +%T) fix $RD"; python3 "$S/extended_retention_fast.py" "$RD" > "$RD/analysis/extended_retention.log" 2>&1 && sed -i 's/EXTRET_WARN/EXTRET_FIXED/' "$RD/analyze.log"
  fi
done; done
echo FIX_EXTRET_DONE
