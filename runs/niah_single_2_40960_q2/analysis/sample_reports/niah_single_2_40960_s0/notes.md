# Sample report: niah_single_2_40960_s0

- **Correct:** True  (reference=['5107245'], prediction='The user is asking about a specific magic number mentioned i')
- **Context length (target):** 40960 tokens; decode steps: 106
- **Needle value:** 5107245 (HF token position ~9722)
- **Highest-locality layer:** L36 adjacent_overlap=0.852
- **Lowest-locality layer:** L24 adjacent_overlap=0.549
- **Overall adjacent overlap:** 0.659; locality lift vs random: 13.15x
- **Recency-baseline overlap:** 0.119 (separates semantic locality from pure recency)

## Anomalies / notes
- Q2 quantized, CPU reference path; logical (not physical) KV reuse.
