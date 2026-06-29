# Attribution

This experiment builds on third-party components. Each remains under its own license; only the
original analysis/instrumentation code and the measured data in this repo are authored here.

| Component | Source | Version | License | Use here |
|---|---|---|---|---|
| ds4 ("DwarfStar") inference engine | https://github.com/antirez/ds4 | commit `80ebbc396aee40eedc1d829222f3362d10fa4c6c` | MIT | CPU inference of DeepSeek-V4-Flash; instrumented via `ds4_instrumentation.patch` |
| RULER benchmark | https://github.com/NVIDIA/RULER | commit `38da79d79519ef87aa46ae804f838e1eab7f86d7` | Apache-2.0 | `niah_single_2` task generation (4K/8K/16K) |
| DeepSeek-V4-Flash model | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash | — | per model card | the model under study |
| DeepSeek-V4-Flash GGUF (IQ2) | https://huggingface.co/antirez/deepseek-v4-gguf | `…IQ2XXS…imatrix.gguf` | per repo | the exact quantized weights run |
| DeepSeek-V4-Flash tokenizer | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash | `tokenizer.json` | per model card | RULER length calibration |
| Paul Graham essays (RULER haystack) | http://www.paulgraham.com / RULER download script | — | author's | haystack text in the prompts |

**Not redistributed here** (download from the sources above): the 81 GB GGUF weights, the DeepSeek
tokenizer, and the RULER repository itself. `docs/ds4_instrumentation.patch` contains only the added
lines (apply to a clean `antirez/ds4` checkout), not the upstream source.
