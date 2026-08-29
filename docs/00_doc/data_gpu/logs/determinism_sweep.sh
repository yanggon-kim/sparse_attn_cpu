#!/bin/bash
source <WORKDIR>/env/activate.sh
S=<HOME>/00_sparse_attn/01_github/sparse_attn_cpu/scripts/gpu
while pgrep -f "determinism_probe.py.*baseline" > /dev/null; do sleep 15; done
P="--model $WORKDIR/models/DeepSeek-V3.2 --prompt-file $WORKDIR/prompts/ladder/ruler_niah_single_2_8192_s00.txt --filler-files $WORKDIR/prompts/ladder/ruler_niah_single_2_8192_s01.txt $WORKDIR/prompts/ladder/ruler_niah_single_2_8192_s02.txt $WORKDIR/prompts/ladder/ruler_niah_single_2_8192_s03.txt --n-tokens 512 --repeats 3"
run() { name=$1; shift; echo "=== $name $(date -u +%T)"; python $S/determinism_probe.py $P --out $WORKDIR/runs/determinism/$name.json "$@" > $WORKDIR/logs/determinism_$name.log 2>&1; grep -E "solo_identical|solo_first_divergence|batch_vs_solo|Traceback|Error" $WORKDIR/logs/determinism_$name.log | head -5; }
run moe_deepgemm --extra-kwargs '{"moe_backend":"deep_gemm"}'
run moe_flashinfer_cutlass --extra-kwargs '{"moe_backend":"flashinfer_cutlass"}'
run batch_invariant --env VLLM_BATCH_INVARIANT=1
run moe_triton --extra-kwargs '{"moe_backend":"triton"}'
run attn_flashmla_sparse --env VLLM_ATTENTION_BACKEND=FLASHMLA_SPARSE
echo SWEEP_DONE
