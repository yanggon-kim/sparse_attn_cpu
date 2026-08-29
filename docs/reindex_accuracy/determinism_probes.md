# Determinism probes (exp3 precondition) — DeepSeek-V3.2 on 8x B200, vLLM 0.28.0 — 2026-08-29

Same 8K RULER niah prompt (8047 tokens), greedy, 512 forced decode steps, run 3x solo in one engine + once in a batch of 4.
`solo_div` = first token divergence between solo repeat 0 and repeats 1/2 (None = identical); `batch_div` = vs the batched copy.

| probe | LLM kwargs / env | eager | solo identical | solo_div | batch_div | max abs Δlogprob(top-1) | load s |
|---|---|---|---|---|---|---|---|
| attn_flashmla_nccl_deepgemm | `{'attention_backend': 'FLASHMLA_SPARSE', 'disable_custom_all_reduce': True, 'moe_backend': 'deep_gemm'}` `['VLLM_ALLREDUCE_USE_SYMM_MEM=0']` | 1 | False | [23, 23] | 10 | [1.944, 1.68] | 120.1 |
| attn_flashmla_sparse | `{}` `['VLLM_ATTENTION_BACKEND=FLASHMLA_SPARSE']` | 1 | False | [24, 24] | 19 | [2.169, 2.179] | 141.5 |
| attn_flashmla_sparse2 | `{'attention_backend': 'FLASHMLA_SPARSE'}` `[]` | 1 | False | [10, 23] | 23 | [2.411, 2.412] | 193.7 |
| baseline | `{}` `[]` | 1 | False | [10, 24] | 24 | [2.317, 2.008] | 141.9 |
| cudagraph_default | `{}` `[]` | 0 | False | [23, 10] | 10 | [2.206, 1.817] | 600.1 |
| moe_deepgemm | `{'moe_backend': 'deep_gemm'}` `[]` | 1 | False | [10, 10] | 10 | [1.956, 2.15] | 136.9 |
| moe_triton | `{'moe_backend': 'triton'}` `[]` | 1 | False | [24, 10] | 39 | [1.78, 2.172] | 117.3 |
| nccl_only | `{'disable_custom_all_reduce': True}` `['VLLM_ALLREDUCE_USE_SYMM_MEM=0']` | 1 | False | [10, 10] | 10 | [2.023, 1.848] | 163.5 |
| nccl_only_deepgemm | `{'disable_custom_all_reduce': True, 'moe_backend': 'deep_gemm'}` `['VLLM_ALLREDUCE_USE_SYMM_MEM=0', 'NCCL_ALGO=Ring', 'NCCL_PROTO=Simple']` | 1 | False | [22, 10] | 30 | [2.55, 2.567] | 118.4 |

Unsupported configurations: `moe_backend=flashinfer_cutlass` (block-quantized FP8 not supported), `VLLM_BATCH_INVARIANT=1` (no batch-invariant sparse-MLA backend); `VLLM_ATTENTION_BACKEND` env is ignored in 0.28 (use the `attention_backend` kwarg; the FlashMLA-sparse runs above used it).
Pre-divergence top-1 logprob noise between identical runs: mean ~0.01, max ~0.04-0.08 nats; divergences occur at near-ties.
Conclusion: bit-exact run-to-run reproducibility is not available on this stack for DeepSeek-V3.2 (sparse MLA + FP8 MoE at TP8);
the exp3 'clean x2 identical' precondition (exp3 §7) cannot be met by configuration. Proposed replacement: judge re-indexed runs against
the measured run-to-run noise floor (clean vs clean2) and ctrl_numeric, on accuracy (paired bootstrap) and on token/logit agreement.
