#!/bin/bash
source <WORKDIR>/env/activate.sh
S=<HOME>/00_sparse_attn/01_github/sparse_attn_cpu/scripts/gpu
while ! grep -q BASELINE_RERUN_DONE $WORKDIR/logs/determinism_baseline_rerun.log 2>/dev/null; do sleep 20; done
P="--model $WORKDIR/models/DeepSeek-V3.2 --prompt-file $WORKDIR/prompts/ladder/ruler_niah_single_2_8192_s00.txt --filler-files $WORKDIR/prompts/ladder/ruler_niah_single_2_8192_s01.txt $WORKDIR/prompts/ladder/ruler_niah_single_2_8192_s02.txt $WORKDIR/prompts/ladder/ruler_niah_single_2_8192_s03.txt --n-tokens 512 --repeats 3"
run() { name=$1; shift; echo "=== $name $(date -u +%T)"; python $S/determinism_probe.py $P --out $WORKDIR/runs/determinism/$name.json "$@" > $WORKDIR/logs/determinism_$name.log 2>&1; grep -E "solo_identical|Traceback|Error:" $WORKDIR/logs/determinism_$name.log | head -3; }
run nccl_only --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 --extra-kwargs '{"disable_custom_all_reduce": true}'
run nccl_only_deepgemm --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 NCCL_ALGO=Ring NCCL_PROTO=Simple --extra-kwargs '{"disable_custom_all_reduce": true, "moe_backend": "deep_gemm"}'
run cudagraph_default --enforce-eager 0
echo SWEEP2_DONE
