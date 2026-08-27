# GPU campaign: DeepSeek-V3.2 / GLM gather-index statistics and re-index accuracy

*Top-level instructions for the AI agent running on the rented 8-GPU node. Written 2026-08-27 on the
CPU box, before any GPU time was bought. Self-contained: read this file first, then the four experiment
files it links. Nothing here has been executed on a GPU yet; every "VERIFY ON THE NODE" item is an
unverified assumption.*

## 1. Purpose

The SPLEX paper (custom-HBM base die, KV-cache re-indexing) needs three things this campaign produces:

| exp | question | consumer |
|---|---|---|
| exp1 | Which KV entries does DeepSeek-V3.2's DSA indexer gather per decode step (top-k = 2048 of 656 B latents, all 61 layers), and how do those sets overlap / persist / concentrate as context grows 8K–128K? | ramulator selection-process calibration (`MEASURED_OVERLAP`, `MEASURED_TOP10`), paper §3.3 |
| exp2 | Same statistics on GLM-5.2 (and GLM-5 if it has DSA) — cross-model generality | paper §3.3 (one sentence + one curve) |
| exp3 | Does physically re-indexing the KV cache during decode change model quality? Clean vs re-indexed runs on benchmarks with official published numbers | paper correctness figure |

The CPU study in this repo answered exp1 for DeepSeek-**V4**-Flash only (compressed CSA, k = 512, IQ2
quant); V3.2 is the model the paper is built around and it never ran on the CPU (no DSA-preserving GGUF fits).

## 2. What already exists in this repo (`git@github.com:yanggon-kim/sparse_attn_cpu.git`)

- CPU results (V4-Flash, ds4 engine): RULER 4K–64K sweep (`docs/kv_hotset_coverage.md`, `docs/moe_selection_locality.md`,
  `docs/moe_locality_sweep/`), LongBench v1 36 runs (`docs/longbench_locality.md`, `docs/longbench_sweep/`),
  long-decode 3,019-step run (`docs/longdecode_temporal_locality.md`). Key CPU numbers, quoted for
  comparison: adjacent-step overlap 0.868/0.790/0.718/0.672/0.668 at 4K/8K/16K/32K/64K, lift over random
  1.7/2.9/5.7/10.5/21.4x, hot-set A@99 80%→20% of pool.
- Analysis chain (`scripts/`): `ingest_trace.py` → `validate_trace.py` → `analyze_locality.py` (R1 KV
  locality) → `analyze_hotset_coverage.py` (R3), plus `ingest_moe_trace.py` / `analyze_moe_locality.py`
  (R2), `generate_*_plots.py`, `aggregate_longbench.py`. They consume one run directory in the ds4 schema
  (defined in `exp1_dsv32_gather_index.md` §4). **Reuse them unchanged**; adapt the trace, not the scripts.
- `docs/vllm_selection_history_collection_guide.md`: a source-read (never executed) hook guide for
  V4 in vLLM @ `2b753ad20`. Its collector pattern is reusable; its schema is not (see the adapter).
- `HANDOFF.md`: CPU-side history, gotchas, and the original GPU plan.

## 3. Node assumptions and budget

- **GPU type is not decided yet.** Requirement, stated generically: **8 GPUs with >= ~1 TB of HBM in total**
  (H200/B200-class, i.e. >= ~128 GB per GPU), so that `deepseek-ai/DeepSeek-V3.2` native FP8 (~690 GB) plus
  the KV cache of the 128K rung fits at TP = 8, `enforce_eager=True`. The method is identical on any such
  node; only kernels and speed differ. **The first exp0 step is therefore to detect the GPU model, SM
  version, driver/CUDA, and per-GPU HBM, and pick the kernels accordingly** (exp0 §1, §7 first item);
  record the result in `node.json` and in every `run_manifest.json` (`gpu` field). H100 (8 × 80 GB =
  640 GB) does not fit; 6/7 GPUs do not divide 128 heads / 256 experts.
- GLM-5.2 (`zai-org/GLM-5.2`, ~750 B, FP8) fits the same node; it is served by vLLM's
  `GlmMoeDsaForCausalLM` on the DeepSeek code path.
- Storage: a decode trace is 2048 int32 per (layer, step) = 500 KB/step over 61 layers; a 2K-step run is
  ~1 GB in binary, 3–5x as JSONL. Plan **1–2 TB of local NVMe** for the whole ladder; keep raw traces there.
- Budget ~1,000 GPU-h = ~125 node-hours. Rough split: exp0 5 h, exp1 50 h, exp2 30 h, exp3 40 h.
  Track hours in the run manifests; stop a ladder early rather than overrun.

## 4. Ground rules

1. Greedy decoding (`temperature=0`, `top_p=1`), seed recorded anyway (`seed=42`), `enforce_eager=True`.
2. Every record is **per-request attributed** (request id → sample id, layer, absolute position). Batched
   decode is allowed (8–16 requests) but never mix records across requests.
3. Every run directory has `run_manifest.json` (model, revision, vLLM commit, node, sampling params, prompt
   set, wall-clock, GPU-h) — the analysis scripts read it.
4. Raw traces stay **off-repo** (NVMe). Only analysis artifacts (JSON/CSV/PNG/MD, small manifests) are
   committed: `docs/gpu_sweep/` (exp1), `docs/glm_sweep/` (exp2), `docs/reindex_accuracy/` (exp3).
5. Commit conventions: scrub every absolute path to `<WORKDIR>` before committing; message = one summary
   line + a body stating run ids and numbers; trailers on every commit:
   `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and `Claude-Session: <url of your session>`.
   Stage explicit paths (never `git add -A`). Push to `origin main` only when the user says so.
6. Gate each phase (§5). Gates are **self-checks**: when a gate passes, write the gate report file (§6) and
   proceed automatically — no review pause. Do not start the ladder until the smoke gate passes; do not
   start exp3 re-index runs until the clean run reproduces the official numbers.
7. Report numbers with their run id, model, context, decode length, and analysis script.
8. Short traces (< ~500 decode steps) falsely show retention plateaus — retention curves need >= 2K steps.

## 5. Execution order and gates

The campaign runs **start to finish without review stops**. Each gate is a self-check the agent evaluates
itself; on pass it writes the gate report (§6) and continues to the next phase. **Stop and report to the
user only when:**

- (a) a gate fails after one reasonable retry (fix the obvious cause once — a wrong line number, a missing
  package, a too-large batch — and re-run; a second failure is a stop);
- (b) a VERIFY item (§7) turns out different from the assumptions **in a way that changes the method**
  (e.g. GLM-5 has no DSA, official benchmark numbers unavailable, model gated with no token, sparse FlashMLA
  kernels unsupported on the detected SM);
- (c) a decision is needed that these docs do not settle (budget overrun > 20 % of the §3 split, storage
  short, a kernel/path unsupported on the node with no documented alternative);
- (d) the exp3 clean accuracy run is outside the official range after the prompt-template / max-tokens /
  scorer fix of exp3 §4.

Everything else (VERIFY items that differ but do not change the method, dropped ladder rungs per exp2 §4,
line-number drift) is recorded in the reports and the campaign continues.

```
exp0 environment ── gate: 3-token TP8 generation, hook unit checks pass
  └─ exp1 smoke (one 8K RULER prompt, 256 steps) ── gate: all sel in [0,pos], count <= 2048,
     61 layers/step, hook-on == hook-off tokens, full analysis chain runs, curve plausible vs CPU V4
       └─ exp1 ladder 8K→128K (shortest first, both run kinds) ── gate: R1/R2/R3 tables + plots committed
            └─ exp2 GLM (same prompts) ── gate: side-by-side curves committed
                 └─ exp3 accuracy: clean run within official range → identity control → perm_once → perm_periodic
```

Experiment files (same directory): `exp0_environment.md`, `exp1_dsv32_gather_index.md`,
`exp2_glm_gather_index.md`, `exp3_reindex_accuracy.md`.

## 6. Report format

The per-gate report is a **file, not a pause**. At each gate (pass or fail) and at the end write:

1. `docs/00_doc/reports/<phase>_<YYYYMMDD>.md` (phases: `exp0`, `exp1_smoke`, `exp1_ladder`, `exp2`,
   `exp3_clean`, `exp3_reindex`, `final`): at most 15 lines — what ran, run ids, headline numbers with
   model/context/decode length, files committed with hashes, deviations from the assumptions, GPU-h used
   so far vs the §3 split, and the gate verdict (PASS / FAIL+retry / STOP with the §5 reason letter).
2. The per-experiment summary md named in each exp file (`docs/gpu_sweep/gpu_sweep_summary.md`,
   `docs/glm_sweep/glm_sweep_summary.md`, `docs/reindex_accuracy/README.md`) once that experiment has data.
3. A dated line appended to the "Status" section of `HANDOFF.md` in the repo root (append a GPU section;
   do not rewrite CPU history).

Commit these together with the analysis artifacts (§4 conventions) and continue. The only time the agent
addresses the user directly is a §5 stop condition or the final report (`reports/final_<date>.md`, plus
the same 15 lines as the chat message).

## 7. VERIFY ON THE NODE (unverified as of 2026-08-27)

- [ ] **GPU model, SM version, driver/CUDA, per-GPU HBM** (`nvidia-smi`; exp0 §1). Total HBM must be >= ~1 TB
      over 8 GPUs; pick kernels for the SM (FlashMLA sparse / `fp8_ds_mla` availability — Hopper sm_90 and
      Blackwell sm_100 take different kernel paths; a missing path is a §5(c) stop). Record in `node.json`
      and in the `gpu` field of every `run_manifest.json`.
- [ ] vLLM pin: the hook citations below are for checkout `5559679` (2026-07-26); the older guide used
      `2b753ad20`. Pin whichever supports V3.2 FP8 + `GlmMoeDsaForCausalLM`; record the commit; re-locate
      every cited line with `grep` before patching.
- [ ] GLM-5 (not 5.2): does its HF `config.json` have `index_topk`? If not, it has no DSA — run GLM-5.2 only.
- [ ] GLM-5.2 `index_topk_freq` (expected 4) and `index_skip_topk_offset` (expected 2) from its config.
- [ ] Official DeepSeek-V3.2 and GLM-5.2 benchmark numbers (HF model cards / tech reports): RULER,
      LongBench v2, GPQA-Diamond, MMLU-Pro. None are stored locally; fetch, cite the URL and date.
- [ ] Storage: NVMe size, and that `/` is not the trace target.
- [ ] Whether the FP8 model files are gated on HF (token needed).

## 8. Ready-to-paste prompt for the GPU agent

```
You are running a measurement campaign on this 8-GPU node. Clone
git@github.com:yanggon-kim/sparse_attn_cpu.git and read docs/00_doc/GPU_CAMPAIGN.md in full, then
the four experiment files it links (exp0..exp3). Follow the execution order and gates in
GPU_CAMPAIGN.md §5 exactly and run the whole campaign start to finish WITHOUT stopping for review:
begin with docs/00_doc/exp0_environment.md (first step: detect the GPU model, SM, driver/CUDA and
per-GPU HBM, and pick kernels accordingly — the GPU type was not known when these docs were written),
then the exp1 smoke run, the exp1 ladder, exp2, exp3. Gates are self-checks: when a gate passes, write
the gate report file (docs/00_doc/reports/<phase>_<date>.md, GPU_CAMPAIGN.md §6), update the HANDOFF.md
Status section and the per-experiment summary md, commit, and continue. Stop and report to me ONLY for
the four conditions in GPU_CAMPAIGN.md §5: a gate fails after one retry; a VERIFY item (§7) differs in a
way that changes the method; a decision the docs do not settle (budget overrun > 20 %, storage short,
kernel unsupported on the SM); or the exp3 clean run is outside the official range. Work through every
"VERIFY ON THE NODE" item in §7 before spending GPU time and record what differs in the exp0 report.
Raw traces go to local NVMe, never into the repo; commit only analysis artifacts under docs/gpu_sweep,
docs/glm_sweep, docs/reindex_accuracy with the commit conventions in §4 (scrub paths to <WORKDIR>, add
the two trailers). Do not push until I say so. Finish with docs/00_doc/reports/final_<date>.md and the
same <=15 lines as your last message.
```
