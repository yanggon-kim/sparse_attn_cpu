#!/bin/bash
set -u
source <WORKDIR>/env/activate.sh
R=<HOME>/00_sparse_attn/01_github/sparse_attn_cpu; W=$WORKDIR; cd $R
D=docs/00_doc/data_gpu; mkdir -p $D/{determinism,batch_summaries,prompts_ladder,prompts_exp3,logs,env,reindex_unit,smoke0} docs/reindex_accuracy/permlogs
scrub() { python3 - "$1" <<'PY'
import sys,os
p=sys.argv[1]; W='<WORKDIR>'
try:
    s=open(p).read()
except Exception: sys.exit()
s2=s.replace(W,'<WORKDIR>').replace(os.path.expanduser('~'),'<HOME>')
if s2!=s: open(p,'w').write(s2)
PY
}
cp $W/runs/determinism/*.json $D/determinism/; cp $W/runs/batch_*.json $D/batch_summaries/
cp $W/prompts/ladder/*.txt $W/prompts/ladder/manifest.jsonl $D/prompts_ladder/
cp $W/prompts/exp3/manifest.jsonl $W/prompts/exp3/manifest_subset.jsonl $D/prompts_exp3/
cp $W/logs/*.log $W/logs/*.sh $D/logs/ 2>/dev/null; rm -f $D/logs/download_*.log $D/logs/env_install.log
cp $W/runs/reindex_unit/*.json $D/reindex_unit/; cp -r $W/runs/reindex_unit/permlog $D/reindex_unit/ 2>/dev/null
cp $W/runs/smoke0/*.json $W/runs/smoke_identity.txt $D/smoke0/; cp -r $W/runs/smoke0/trace $D/smoke0/trace
pip freeze > $D/env/pip_freeze.txt; cp $W/env/activate.sh $D/env/; cp $W/node.json $D/env/node.json; nvidia-smi > $D/env/nvidia_smi.txt
for p in $W/runs/exp3/permlog_*; do n=$(basename $p); tar czf - -C $W/runs/exp3 $n | split -b 45m - docs/reindex_accuracy/permlogs/$n.tar.gz.part; done
find $D docs/reindex_accuracy/permlogs -type f | while read f; do case "$f" in *.txt|*.json|*.jsonl|*.log|*.sh|*.md) scrub "$f";; esac; done
# smoke / exp0 run dirs (traces sharded)
for rd in $(ls -d $W/runs/*_smoke* $W/runs/v32_smoke0_capital_s0 $W/runs/_cmp_* 2>/dev/null | grep -v '/_cmp_'); do
  b=$(basename $rd); case $b in glm52*) dest=docs/glm_sweep/runs/$b;; *) dest=docs/gpu_sweep/runs/$b;; esac
  [ -f $rd/traces/indexer_trace.jsonl ] && python3 scripts/gpu/shard_traces.py --run-dir $rd --dest $dest > /dev/null
done
cat > $D/README.md <<'EOM'
# GPU campaign — node-side artifacts (added 2026-08-29 so nothing depends on the rented node)
- `determinism/` — the 11 determinism-probe JSONs (token ids + top-1 logprobs, 3 solo runs + 1 batched per config).
- `batch_summaries/` — per-rung runner summaries (`batch_<tag>_<rung>_bf_ld.json`, smoke batches).
- `prompts_ladder/` — the 135 exp1/exp2 prompts (raw user text) + `manifest.jsonl`.
- `prompts_exp3/` — exp3 manifests (full 5,932-item set and the tier-1 subset); texts regenerate deterministically with
  `scripts/gpu/build_exp3_prompts.py` (RULER via prepare.py seed 42, LongBench-v2/InfiniteBench/MMLU-Pro from HF).
- `logs/` — runner/analysis/probe logs and the orchestration scripts used on the node.
- `env/` — `pip_freeze.txt`, `activate.sh`, `node.json`, `nvidia-smi` snapshot.
- `reindex_unit/`, `smoke0/` — exp3 hook unit test and the exp0 3-token smoke trace.
- `../../reindex_accuracy/permlogs/` — exp3 permutation logs (tar.gz, 45 MB parts: `cat <name>.tar.gz.part* | tar xz`).
- Smoke run dirs (`*_smoke*`, `v32_smoke0_capital_s0`) are packaged under `docs/{gpu,glm}_sweep/runs/` like the ladder runs.
EOM
git add -A docs/00_doc/data_gpu docs/reindex_accuracy/permlogs
git commit -q -F - <<'EOM'
Add node-side artifacts: determinism probes, batch summaries, prompt ladder + exp3 manifests, logs, env, unit test, exp3 permutation logs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01W3cwGyLxyjrS7tgfrBx5mN
EOM
git push -q origin main && echo "extras pushed $(git rev-parse --short HEAD)"
# smoke run dirs in <=1.5 GB batches
size=0; batch=()
flush() { [ ${#batch[@]} = 0 ] && return; git add -- "${batch[@]}"; git commit -q -m "raw traces: smoke/exp0 run dirs (${#batch[@]} dirs)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01W3cwGyLxyjrS7tgfrBx5mN"; git push -q origin main && echo "smoke batch pushed $(git rev-parse --short HEAD) ${#batch[@]} dirs"; batch=(); size=0; }
for d in $(git status --short -uall | awk '{print $2}' | grep -E '^docs/(gpu|glm)_sweep/runs/' | cut -d/ -f1-4 | sort -u); do
  s=$(du -sb $d | cut -f1); if [ $((size+s)) -gt $((1500*1024*1024)) ] && [ ${#batch[@]} -gt 0 ]; then flush; fi; batch+=("$d"); size=$((size+s)); done; flush
echo EXTRAS_DONE
