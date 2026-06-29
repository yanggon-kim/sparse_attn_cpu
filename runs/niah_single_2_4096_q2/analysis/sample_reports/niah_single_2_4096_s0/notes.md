# Sample report: niah_single_2_4096_s0

- **Correct:** True  (reference=['2338687'], prediction='The user is asking about a "special magic number for harmoni')
- **Context length (target):** 4096 tokens; decode steps: 128
- **Needle value:** 2338687 (HF token position ~1421)
- **Highest-locality layer:** L36 adjacent_overlap=0.947
- **Lowest-locality layer:** L10 adjacent_overlap=0.827
- **Overall adjacent overlap:** 0.869; locality lift vs random: 1.71x
- **Recency-baseline overlap:** 0.429 (separates semantic locality from pure recency)

## Anomalies / notes
- Q2 quantized, CPU reference path; logical (not physical) KV reuse.
