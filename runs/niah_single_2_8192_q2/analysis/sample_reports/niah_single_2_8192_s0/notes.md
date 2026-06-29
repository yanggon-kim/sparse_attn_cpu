# Sample report: niah_single_2_8192_s0

- **Correct:** True  (reference=['7210606'], prediction='The user is asking about a specific "special magic number fo')
- **Context length (target):** 8192 tokens; decode steps: 128
- **Needle value:** 7210606 (HF token position ~4267)
- **Highest-locality layer:** L36 adjacent_overlap=0.908
- **Lowest-locality layer:** L10 adjacent_overlap=0.711
- **Overall adjacent overlap:** 0.790; locality lift vs random: 2.91x
- **Recency-baseline overlap:** 0.299 (separates semantic locality from pure recency)

## Anomalies / notes
- Q2 quantized, CPU reference path; logical (not physical) KV reuse.
