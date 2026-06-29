# Sample report: niah_single_2_16384_s0

- **Correct:** True  (reference=['7210606'], prediction='The user is asking about a specific number mentioned in the ')
- **Context length (target):** 16384 tokens; decode steps: 117
- **Needle value:** 7210606 (HF token position ~8487)
- **Highest-locality layer:** L36 adjacent_overlap=0.882
- **Lowest-locality layer:** L24 adjacent_overlap=0.619
- **Overall adjacent overlap:** 0.718; locality lift vs random: 5.72x
- **Recency-baseline overlap:** 0.157 (separates semantic locality from pure recency)

## Anomalies / notes
- Q2 quantized, CPU reference path; logical (not physical) KV reuse.
