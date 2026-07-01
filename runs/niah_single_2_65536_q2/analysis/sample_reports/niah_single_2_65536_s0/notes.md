# Sample report: niah_single_2_65536_s0

- **Correct:** False  (reference=['5107245'], prediction='The user is asking about a special magic number for roasted-')
- **Context length (target):** 65536 tokens; decode steps: 128
- **Needle value:** 5107245 (HF token position ~16764)
- **Highest-locality layer:** L36 adjacent_overlap=0.862
- **Lowest-locality layer:** L10 adjacent_overlap=0.545
- **Overall adjacent overlap:** 0.670; locality lift vs random: 21.42x
- **Recency-baseline overlap:** 0.106 (separates semantic locality from pure recency)

## Anomalies / notes
- Q2 quantized, CPU reference path; logical (not physical) KV reuse.
