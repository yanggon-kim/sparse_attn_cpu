# Sparse-Attention KV Temporal Locality in DeepSeek-V4 (CPU)

Measuring **which compressed-KV entries the DeepSeek-V4 CSA "lightning indexer" selects during
autoregressive decode, and how that selection persists across decode steps** — run entirely on CPU.

This repo contains the **raw evidence** (per-token, per-layer KV-selection traces), the **benchmark
inputs**, the **analysis pipeline**, and the **results**. It is written so that another person — or an
AI agent — can reproduce the whole thing from scratch by following this README.

> **TL;DR result.** With a fixed top-k = 512, the indexer's per-token selection is strongly temporally
> local, and it becomes *more* structured (less random) as context grows: adjacent-step overlap
> 0.87 → 0.79 → 0.72 → 0.66 at 4K/8K/16K/40K, while the **locality lift over a random selector rises
> 1.7× → 2.9× → 5.7× → 13.2×** (roughly doubling per ~2× of context).
> The needle's KV block is pinned in 95–100% of layer×step cells. Selection is semantic, not recency.
> All measurements are reproducible from `runs/*/traces/` via `scripts/analyze_locality.py`.

---

## 1. Model used

- **Model:** DeepSeek-V4-Flash — `deepseek-ai/DeepSeek-V4-Flash` (Mixture-of-Experts, **284B total /
  13B active**, 1M-token context, hybrid attention).
- **Architecture (43 layers), verified at runtime and used by the analysis:**
  - Layers **0–1: SWA** (sliding-window attention, window 128, compression ratio 0).
  - **Even layers 2–42: CSA** (Compressed Sparse Attention, ratio 4) — these run the **top-k = 512
    lightning indexer**; *this is the only place sparse selection happens* (21 layers).
  - **Odd layers 3–41: HCA** (Heavily-Compressed Attention, ratio 128, no top-k).
  - Compressed index `c` (ratio `r`) maps to original tokens `[c·r, c·r + r − 1]` (contiguous, non-overlapping).
- **Exact weights run:** GGUF quant
  `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` (~81 GB) from
  Hugging Face `antirez/deepseek-v4-gguf`.
  **SHA-256:** `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668` (also in each `runs/*/run_manifest.json`).
- **Why IQ2 / CPU:** this is an instrumentation + systems study, not a quality benchmark. IQ2 fits in
  RAM and the CPU "reference" path is the easiest place to log the indexer's exact top-k choices.
  Results are for *this quantized CPU runtime* (see Caveats).

## 2. Inference engine

- **Engine:** `antirez/ds4` ("DwarfStar"), a single-file C inference engine for DeepSeek-V4 — commit
  `80ebbc396aee40eedc1d829222f3362d10fa4c6c` (MIT).
- **Instrumentation:** apply `docs/ds4_instrumentation.patch` (124-line addition to `ds4.c`). It adds an
  env-gated, buffered JSONL logger at the indexer top-k site (`indexer_log_selection`), emitting one
  record per CSA layer per token. **It does not change model outputs** (verified: tracing on/off gives
  byte-identical greedy tokens).
  - Env controls: `DS4_TRACE_OUTPUT` (dir), `DS4_TRACE_LEVEL` (0 meta / 1 selected / 2 boundary /
    3 full-score subset), `DS4_TRACE_DECODE_ONLY=1` (skip prefill — essential, else 16K ≈ 2 GB),
    `DS4_TRACE_FULL_SCORE_SAMPLE_RATE`, `DS4_TRACE_FLUSH_INTERVAL`.
- **Build:** `make cpu` (gcc, `-O3 -ffast-math -march=native`). Helper: `scripts/build_cpu.sh`.

## 3. Benchmark

- **Benchmark:** official **NVIDIA/RULER** — commit `38da79d79519ef87aa46ae804f838e1eab7f86d7` (Apache-2.0).
- **Task:** `niah_single_2` (single needle-in-a-haystack: a word *key* → number *value*, hidden in a
  Paul Graham essay haystack). One sample at each context length **4K / 8K / 16K / 40K**.
- **Length calibration tokenizer:** `deepseek-ai/DeepSeek-V4-Flash` (`PreTrainedTokenizerFast`,
  `tokenizer.json`). RULER sizes the haystack with this tokenizer; the actual ds4 token count is
  recorded per run (`context_length_actual_tokens`).
- **Decode:** deterministic greedy (`--temp 0`), 128 new tokens. A single generation serves both RULER
  recall scoring (substring match) and the trace.
- The exact generated prompts are in `benchmark/prompts/`; the RULER `validation.jsonl` files in
  `benchmark/ruler_data/`. See `benchmark/README.md` to regenerate.

## 4. How to reproduce (step by step)

```bash
# 0. Prereqs: ~256 GB RAM, multi-core x86-64, ~120 GB disk for traces, gcc/make/cmake, Python 3.10+
pip install numpy pandas pyarrow matplotlib transformers tokenizers tabulate scipy \
            nltk tqdm pyyaml wonderwords html2text tenacity requests

# 1. Engine: clone ds4, apply the instrumentation patch, build the CPU binary
git clone https://github.com/antirez/ds4 && cd ds4
git checkout 80ebbc396aee40eedc1d829222f3362d10fa4c6c
git apply /path/to/this/repo/docs/ds4_instrumentation.patch
make cpu                                    # -> ./ds4

# 2. Weights: download the IQ2 GGUF (~81 GB) from antirez/deepseek-v4-gguf
#    DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf

# 3. Benchmark: get the DeepSeek tokenizer + RULER, generate the prompts
#    (tokenizer.json + tokenizer_config.json from huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
git clone https://github.com/NVIDIA/RULER && (cd RULER && git checkout 38da79d7)
#    RULER's prepare.py calls `python` (not python3) -> put a `python`->python3 shim on PATH.
#    Download the Paul Graham essay corpus: RULER/scripts/data/synthetic/json/download_paulgraham_essay.py
bash scripts/prepare_benchmark.sh           # -> benchmark/prompts/*.txt  (see script for exact RULER args)

# 4. Run (deterministic, traced; resumable). Per-sample env used:
#    DS4_TRACE_OUTPUT=<run>/traces DS4_TRACE_LEVEL=3 DS4_TRACE_DECODE_ONLY=1 \
#    DS4_TRACE_FULL_SCORE_SAMPLE_RATE=0.002 OMP_NUM_THREADS=64 \
#    ds4 --cpu -m <gguf> --prompt-file <prompt> -c <len+768> -t 64 --temp 0 -n 128 \
#        --dump-logprobs <run>/outputs/logprobs.json
bash scripts/run_experiment.sh "4096 8192 16384"

# 5. Analyze: ingest -> validate -> metrics -> tables -> plots -> per-sample reports
bash scripts/analyze_all.sh
```

**Pipeline scripts** (`scripts/`): `run_experiment.sh` (run), `finalize_run.py` (manifests),
`ingest_trace.py` (JSONL → Parquet evidence tables + checksums), `validate_trace.py` (unit tests +
integrity, exits non-zero on failure), `analyze_locality.py` (all metrics), `generate_tables.py`,
`generate_plots.py`, `make_sample_report.py`, `locality_lib.py` (metric helpers).
*Note: scripts use absolute paths from the original run environment (identifiers scrubbed to
`<host>`/`<user>`/`<WORKDIR>`); adjust paths for your machine.*

## 5. What is measured (metric definitions)

For decode step `t` and CSA layer `l`, let `U[t,l]` = the set of selected compressed-KV indices.
- **Adjacent overlap** `|U[t]∩U[t-1]| / |U[t]|`; **Jaccard**; **churn** `1-overlap`; rank-aware
  **weighted overlap** (DCG weights `1/log2(rank+2)`).
- **Retention(lag)** mean over `t` of `|U[t]∩U[t-lag]| / |U[t]|`, lags 1/2/4/8/16/32/64.
- **Working set(w)** = |union of `U` over a w-step window|; normalized `÷ (w·top_k)`.
- **Reuse distance** over the logical `(layer, index)` access stream; **access age** = query position −
  representative original position of the block.
- **Random baseline** `E[overlap] ≈ top_k / n_candidates`; **locality lift** = observed / random.
- **Recency baseline** = the most-recent `top_k` compressed entries (separates semantics from recency).

## 6. Results (this run, n = 1 per length, all needles retrieved correctly)

| Context | n_candidates | kept | adjacent overlap | retention@64 | **lift vs random** | working-set@64 | wall clock |
|--------:|-------------:|-----:|-----------------:|-------------:|-------------------:|---------------:|-----------:|
| 4K  | ~1008  | 51% | 0.87 | 0.76 | 1.7× | 2.6% of 64·top-k | 54 min |
| 8K  | ~1887  | 27% | 0.79 | 0.61 | 2.9× | 3.8% | 1h45 |
| 16K | ~4080  | 13% | 0.72 | 0.50 | 5.7× | 5.6% | 4h09 |
| 40K | ~10222 |  5% | 0.66 | 0.43 | **13.2×** | 7.3% | 12h58 |

Hardware: 2× Intel Xeon Silver 4514Y (64 threads, AVX-512/AMX), 251 GB RAM; CPU prefill ≈ 0.88–1.35 tok/s
(slower at longer context due to the O(n²) indexer cost), peak RSS 81–107 GB, no swapping.
Full write-up: [`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md).

## 7. Repository layout
```
EXPERIMENT_SUMMARY.md      headline findings + answers to the research questions
benchmark_decision.md      benchmark selection note (RULER decision tree)
docs/00_doc/               the original experiment specification (first instruction)
docs/ds4_instrumentation.patch   the ds4.c instrumentation (apply to antirez/ds4)
docs/ATTRIBUTION.md        third-party components + licenses
scripts/                   the full run + analysis pipeline
benchmark/prompts/         the 3 generated prompts + samples.jsonl
benchmark/ruler_data/      RULER niah_single_2 validation.jsonl (4K/8K/16K)
runs/<id>/                 per-run manifests, traces/*.parquet + indexer_trace.jsonl, analysis/, logs/
tables/                    11 summary tables (CSV + Markdown)
plots/                     34 plots (per-run + combined)
overhead/                  trace-overhead measurement
```

## 8. Caveats
Q2-quantized runtime (not full-precision behavior); CPU reference path (not GPU production); **logical**
KV reuse (not physical cache hits); cross-layer overlap = semantic position agreement, not shared
tensors; **n = 1 sample per context length** → point estimates only, no cross-sample confidence
intervals. See `EXPERIMENT_SUMMARY.md` and `docs/00_doc/` for the full methodology.

## 9. Attribution & license
Builds on `antirez/ds4` (MIT), `NVIDIA/RULER` (Apache-2.0), and `deepseek-ai/DeepSeek-V4-Flash`
(model + tokenizer). See `docs/ATTRIBUTION.md`. **This repository ships no LICENSE file** — the
original-work code/analysis here carries no explicit license grant; third-party components remain under
their own licenses.
