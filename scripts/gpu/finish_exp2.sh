#!/bin/bash
[ -n "${WORKDIR:-}" ] && [ -f "$WORKDIR/env/activate.sh" ] && source "$WORKDIR/env/activate.sh" > /dev/null 2>&1 || true
# exp2 §5 outputs: per-model/variant aggregates, side-by-side plots with DeepSeek-V3.2 and the CPU V4 curve, summary md.
# Usage: finish_exp2.sh   (run after the GLM-5.2 and GLM-5 ladders are analyzed and packaged)
set -u
S=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$S/../.." && pwd)
D=$REPO/docs/glm_sweep
mkdir -p "$D"
python3 "$S/aggregate_gpu_sweep.py" --runs-root "$WORKDIR/runs" --model-tag glm52 --model-name GLM-5.2 --out "$D" --variant all_layers --suffix _glm52_a
python3 "$S/aggregate_gpu_sweep.py" --runs-root "$WORKDIR/runs" --model-tag glm52 --model-name GLM-5.2 --out "$D" --variant computing_only --suffix _glm52_b
python3 "$S/aggregate_gpu_sweep.py" --runs-root "$WORKDIR/runs" --model-tag glm5 --model-name GLM-5 --out "$D" --variant all_layers --suffix _glm5
# combined R1/R2/R3 tables (one row set per model/variant)
python3 - "$D" <<'PY'
import csv, glob, os, sys
D = sys.argv[1]
for name in ("R1_kv_locality", "R2_moe_locality", "R3_hotset_coverage", "accuracy_by_source"):
    rows, keys = [], []
    for f in sorted(glob.glob(os.path.join(D, f"{name}_*.csv"))):
        for r in csv.DictReader(open(f)):
            rows.append(r)
            for k in r:
                if k not in keys:
                    keys.append(k)
    with open(os.path.join(D, f"{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    print("combined", name, len(rows), "rows")
PY
python3 "$S/make_gpu_sweep_summary.py" --out "$D" --title "DSA gather-index statistics: DeepSeek-V3.2 vs GLM-5.2 vs GLM-5 on 8x B200 (exp2)" \
  --sweeps "$REPO/docs/gpu_sweep/sweep_v32.json" "$D/sweep_glm52_glm52_a.json" "$D/sweep_glm52_glm52_b.json" "$D/sweep_glm5_glm5.json" \
  --labels "DeepSeek-V3.2 (61 layers)" "GLM-5.2 (a) all 78 layers" "GLM-5.2 (b) 21 computing layers" "GLM-5 (78 layers)" \
  --md glm_sweep_summary.md --png-prefix glm
cp "$D/glm_01_context_scaling.png" "$D/side_by_side.png"
echo FINISH_EXP2_DONE
