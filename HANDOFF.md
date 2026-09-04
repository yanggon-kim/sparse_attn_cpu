# HANDOFF.md — DeepSeek sparse-selection locality study

State of the `deepseek-owner` sub-project. Read this first, then the previous session's auto-memory
`~/.claude/projects/-home-yanggon-99-personal-project-05-deepseek/memory/deepseek-v4-kv-locality-experiment.md`
(read-only history of the V4 CPU runs). Superseded and dated material — campaign narratives, resolved
VERIFY items, onboarding cross-checks, older decisions — lives in `HANDOFF.history.md` beside this file.

## 1. Purpose
Measure the **temporal locality of sparse selections** inside DeepSeek-V4, V3.2 and GLM-5/5.2 for an
architecture paper (ISCA/HPCA/ASPLOS target): which KV entries the CSA/DSA "lightning indexer" gathers
per token (top-k), which MoE experts the router picks, and how much those sets overlap across
consecutive decode steps. Outputs feed a hot-set/prefetch cache design. Three result families:
R1 KV top-k locality, R2 MoE routing locality, R3 hot-set coverage (% of pool to cover 99 % of accesses).
A fourth line (exp3) shows that re-indexing the KV cache does not change model quality.

## 2. Layout (absolute paths)
- `work/experiment/` — the whole pipeline. `runs/<id>/{traces,outputs,analysis,prompts,logs}`;
  `scripts/`; `prompts/`; `benchmark/{RULER,longbench}`; `analysis_moe/` (RULER-sweep plots);
  `analysis_longbench/` (aggregate + plots); `exports/` (v6 npz exports, see §4). `EXPERIMENT_SUMMARY.md`
  = RULER write-up.
- `work/ds4/` — antirez/ds4 C engine (HEAD 80ebbc3) **+ uncommitted local changes in `ds4.c` only**:
  (a) instrumentation — KV logger `indexer_log_selection`, MoE logger `moe_log_selection`
  (= `sparse_attn_cpu/docs/ds4_instrumentation.patch`, still reverse-applies clean); (b) the **v5
  re-index correctness hook** added by the ramulator project (`kv_perm_apply`, env
  `DS4_PERM_MODE/SEED/PERIOD/FRAC/HCA/LOG/IDENTITY`; default MODE=0 = inert). Patch copy:
  `ramulator2/examples/v5_ds4_reindex_correctness/ds4_reindex_hook.patch` (absolute-path headers, needs
  `-p7`; 2 lines behind ds4.c — lacks the `DS4_PERM_IDENTITY` control). **Keep both; never revert.**
  Binaries (untracked): `ds4` (hook without IDENTITY), `ds4_new` (current source), `ds4_nofm`
  (current source, `-fno-fast-math`). `make cpu` rebuilds `ds4`. `work/ds4/AGENT.md` is the user's file.
- `work/models/` — `DeepSeek-V4-Flash-IQ2XXS-...gguf` (81 GB, the CPU workhorse);
  `v32/DeepSeek-V3.2-UD-TQ1_0.gguf` (151 GB, dense, NO DSA indexer — useless for sparse data);
  `v32_4L/DeepSeek-V3.2-4Layers-Q8_0.gguf` (16 GB, arch deepseek32 WITH indexer; dev-only).
- `work/llama.cpp/` — pinned to 683f0c72e with local uncommitted patches (§6). Not pushed anywhere.
- `01_github/sparse_attn_cpu/` — the published repo (git@github.com:yanggon-kim/sparse_attn_cpu, branch
  `main`; we are its only writer). Holds `docs/00_doc/GPU_CAMPAIGN.md`, `exp0..exp3`, `docs/gpu_sweep/`,
  `docs/glm_sweep/`, `docs/reindex_accuracy/`, `docs/00_doc/data/`, and a path-scrubbed copy of this
  HANDOFF plus `HANDOFF.history.md` — **refresh both whenever this file changes and a commit is due**
  (scrub the absolute work-tree prefix to `<WORKDIR>` and the home prefix to `<HOME>`).
- `01_github/versel_distribute/` — a second clone of the static site → https://versel-distribute.vercel.app.
  **Read-only for us**: `vercel-owner` is the only agent that commits there. When results need a page
  change, put the final figures/tables/text under our root and describe the change in the report.
  Parts III (`04_sparse_attn/02_part3_cpu_kv_locality/`) and IV (`03_part4_moe_locality/`) carry our
  numbers; Parts I/II/V+ are the user's own — never edit their numbers. Read its `CLAUDE.md` before
  preparing any page material.
- `papers/`, `00_doc/`, `study_deepseek_v4_vllm/`, `PLAN.md`, `RESULTS.md` — background docs.

## 3. Current status
- **V4 CPU campaign: complete and published.** RULER 4K–64K sweep (`runs/niah_single_2_{L}_moe_q2`),
  36 LongBench summarization runs (`runs/lb_*_q2`, 15,073 decode steps, 4.5 GB raw kept locally), and
  one 3,019-step long-decode run (`runs/longform_p16k_g4k_q2`). Vercel Parts III/IV carry the results.
- **GPU campaign: executed and delivered** (8× B200, vLLM 0.28.0). V3.2 + GLM-5.2 + GLM-5 locality
  ladders 8K–128K with MoE, hot-set and retention; exp3 re-index accuracy tier 1 (all three models) and
  tier 2 (V3.2 PPL complete, generation partial). Data in `sparse_attn_cpu/docs/{gpu_sweep,glm_sweep,
  reindex_accuracy}/`, reports in `docs/00_doc/reports/`, onboarding doc
  `docs/reindex_accuracy/START_HERE.md`. ~128 of the 1,000 GPU-h budget used.
- **v6 trace exports delivered** for the ramulator policy study: V4 `exports/v6/`, V3.2
  `exports/v6_v32/`, GLM `exports/v6_glm/` (`<id>_b.npz` = GLM-5.2 computing-only view). Format:
  `sparse_attn_cpu/docs/00_doc/v6_export_format.md`.
- **Parked:** DeepSeek-V3.2 on CPU. The engine is patched and loads, but the only CPU-fittable GGUF
  (Unsloth TQ1_0) is dense-stripped and DSA-preserving GGUFs are 404–714 GB; the standalone tracer
  (`work/llama.cpp/examples/eval-callback/eval-callback.cpp`, target `llama-eval-callback`) crashes on
  prompts > ~256 tokens (`ops.cpp set_rows: leaf_90 uninitialized`) while `llama-server` handles the
  same prompt. Fix path: hook the trace into llama-server's decode path. The user pivoted to GPU.
- **Blocked:** the remaining GPU work needs 8 free GPUs at TP=8; the node was taken over by another
  workload (four external `Qwen2.5-72B-Instruct` vLLM servers). See §7.

## 4. Key numbers
Every figure below carries its source artifact. Model/context/decode-length matter — see the bf/ld
caveat. Superseded blocks are in `HANDOFF.history.md`.

**DeepSeek-V3.2, GPU ladder (the headline set).** k = 2048 tokens, 61 layers, ratio 1; rung mean over
bf + ld runs with 95 % bootstrap CI over runs. Source: `sparse_attn_cpu/docs/gpu_sweep/gpu_sweep_summary.md`
+ the per-run R1/R2/R3 CSVs. Rungs 8K / 16K / 32K / 64K / 128K:
- adjacent overlap **0.843 / 0.790 / 0.751 / 0.741 / 0.723**; lift 2.56 / 5.27 / 10.2 / 21.9 / 41.5×;
  recency baseline 0.49 → 0.31
- ret@64 0.722 / 0.627 / 0.584 / 0.602 / 0.609; ld ret@512 0.580 / 0.508 / 0.487 / 0.478 / 0.473;
  ld ret@1024 0.433 / 0.388 / 0.369 / 0.351 / 0.344
- A@99 **65.0 / 44.7 / 30.3 / 16.7 / 9.7 %** (pool 6.6K / 14K / 28K / 61K / 119K);
  cov@10 % (MEASURED_TOP10) 0.304 / 0.565 / 0.788 / 0.935 / 0.978
- MoE adjacent overlap 0.24–0.26 (7.7–8.3×, 58 learned layers, 8 of 256), context-independent

**Long-decode (ld) rows only** — 8 runs/rung, 2048 forced steps. Source `docs/00_doc/data/
gpu_headline_by_kind.csv` + `hotness_*.csv`. 64K / 128K: adj 0.793 / 0.799, lift 24.8 / 43.6×,
ret@64 0.643 / 0.627, ret@1024 0.351 / 0.344, cov10 0.894 / 0.948, A@99 24.8 / 17.8 % (N 65K / 113K);
16K / 32K A@99 66.5 / 46.6.

> **Caveat — always check the run kind.** bf and ld diverge sharply at 64K/128K: adj overlap bf
> 0.719 / 0.684 vs ld 0.793 / 0.799; A@99 bf 13.4 / 5.7 % vs ld 24.8 / 17.8 %; cov10 bf 0.952 / 0.993
> vs ld 0.894 / 0.948. The short bf decodes (median 20–30 steps at those rungs) inflate hot-set
> concentration. **Quote hot-set and retention from ld rows** or state the mix; overlap and lift are
> fine either way.

**GLM-5.2 / GLM-5 (same ladder).** Source `docs/glm_sweep/`. Adjacent overlap GLM-5.2(a) 0.846 / 0.789 /
0.750 / 0.722 / 0.702, (b) computing-only 0.847 / 0.791 / 0.745 / 0.714 / 0.692, GLM-5 0.849 / 0.791 /
0.753 / 0.730 / 0.715; lift 2.6 → 40×; A@99 GLM-5.2 62.1 / 42.7 / 29.1 / 18.0 / 11.5, GLM-5 61.5 / 42.5 /
28.9 / 18.6 / 11.0 (ld-only 64K/128K: 32.2 / 25.2 and 31.9 / 23.8); cov10 0.30 / 0.57 / 0.80 / 0.92 / 0.96;
ld ret@1024 GLM-5.2 0.44 → 0.29, GLM-5 0.44 → 0.30; MoE 0.28–0.31 (9–10×, 75 learned layers).
**Generality:** all three models within ±0.02 overlap and ±2 pt A@99 of each other.

**DeepSeek-V4, CPU (ds4).** Still the calibration source for the ramulator testcase. RULER `_moe_q2`
series, sources `runs/niah_single_2_{L}_moe_q2/analysis/{metrics_run_summary,hotset_coverage}.json`:
adjacent overlap 0.868 / 0.790 / 0.718 / 0.672 / 0.668 at 4K/8K/16K/32K/64K, lift 1.72 / 2.92 / 5.72 /
10.53 / 21.37×; A@99 79.5 / 65.7 / 47.5 / 31.4 / 20.4 % of pool (pool 1,030 / 1,912 / 4,095 / 8,039 /
16,393; per-layer range 9.9–38.4 % at 64K); MoE learned 0.33–0.38 (~16× random, context-independent),
hash layers 0–2 ≈ random. LongBench, 36 runs, source `analysis_longbench/longbench_aggregate.json`
(= `sparse_attn_cpu/docs/longbench_sweep/`): multi_news 0.914 / gov_report 0.755 / qmsum 0.732 adjacent,
pooled 0.801; A@99 92.0 / 78.7 / 61.4 % — real tasks lie ON the RULER curve. Long decode
`longform_p16k_g4k_q2` (3,019 steps): retention 0.73@lag1 → 0.51@64 → 0.37@512 → 0.17@2048, no plateau.

**Re-index accuracy (exp3).** V3.2 tier 2, source `docs/reindex_accuracy/tier2/tier2_results.md` and
`docs/reindex_accuracy/paper_accuracy_section.md` (ready-to-adapt Tables 3/4): PPL block complete —
30 long books × {32K, 64K, 128K} × 10 modes + WikiText-2 (93 windows) + PTB (32) = 2,150 rows, 1,505
re-index events, 0 hook errors; **38/40 mode rows and all 30 long-book rows inside the identical-rerun
noise floor** (the 2 flagged are PTB impl-B seeds, ≈1 SE, CIs include 0). Baseline PPL 1.3058 / 1.2678 /
1.3674 at 32K/64K/128K. Generation, 400 items (RULER niah_single_2 / niah_multikey_2 / vt / qa_1 × 3
lengths × 25 + LongBench-v2 100): clean 329/400, perm_once_B **331/400**, perm_periodic_B partial
225/400 → **210 vs 210 baseline on the same items, 0 flips**; niah/vt 25/25 at every length in every
mode. Tier 1 (all three models): 77/78 mode rows within the noise floor; RULER niah+qa accuracy
unchanged in every mode. **Token streams are NOT identical** (perm_once_B 173/300 RULER identical) —
claim PPL equivalence + task accuracy, never token-exactness. V4 on ds4: identity control bit-identical;
one-time permutation 127/128 tokens identical, needle retrieved, selection Jaccard 0.967 in
original-index space; periodic swaps and the `-fno-fast-math` control both diverge at step 35 → the
residual is summation-order noise (write-up `ramulator2/00_doc/01_design/v5/v5_reindex_correctness_ds4.md`).

**Export inventory.** `exports/v6/` 42 V4 runs, 18,799 steps, 102 MB (k = 512, ratio 4, 21 CSA layers);
`exports/v6_v32/` 146 npz / 9.4 GB; `exports/v6_glm/` 415 npz / 19.1 GB (glm52 + glm52b + glm5);
`exports/gpu/` 33 GB of reassembled gz shards. Each directory also holds `retention_curves.json`.
Schema (both generations): `sparse_attn_cpu/docs/00_doc/v6_export_format.md`. Regenerate with
`scripts/export_v6_traces.py` (V4, ~4 min) or `scripts/export_v6_gpu_traces.py` via
`exports/run_v6_gpu_export.sh` (~35 min, `--skip-existing` resumes).

**Cache layout fact.** The GPU campaign's KV latent cache is **bf16**: 576 B rows = 1152 B/token/layer
plus the 132 B indexer key — not the 656 B fp8 layout the design docs assume. The paper keeps the fp8
byte accounting with a footnote.

## 5. Recent decisions and why
- **Quote hot-set and retention from the long-decode rows** (2048 steps), or state the bf/ld mix.
  Short benchmark decodes (median 20–30 steps at 64K/128K) inflate hot-set concentration — the same
  effect that produced the retracted "retention plateau" claim on V4.
- **V3.2 is the headline model for consumers**; V4 stays as the CPU-engine correctness vehicle and the
  existing ramulator calibration. V3.2 is *more* local at 64K than V4 (0.74 rung mean / 0.79 ld vs 0.67),
  so V4-fit policy conclusions should hold, but that is not yet verified.
- **Re-index correctness is claimed as PPL equivalence + task accuracy, never token-exactness** (user).
  Decode on 8× B200 is not run-to-run deterministic (11 probes,
  `docs/reindex_accuracy/determinism_probes.md`), so bit-identity is not measurable there.
- **exp3 must run at TP = 8.** TP = 4 changes reduction order and breaks comparability with the recorded
  baseline, and the GLM FP8 weights do not fit in 4 GPUs.
- **GPU campaigns run start to finish without review stops** (user): gates are self-checks, each phase
  writes `docs/00_doc/reports/<phase>_<date>.md` plus a HANDOFF status line; stop only for a gate failing
  after one retry, a method-changing VERIFY outcome, an undecided decision (incl. a >20 % budget
  overrun), or a clean run outside the official range.
- **GPU type is left generic** (8 GPUs, ≥ ~1 TB HBM) (user); `exp0` detects model/SM/driver/HBM and picks
  kernels as its first VERIFY item, so the package is portable across rentals.
- **Native FP8, 8 GPUs, ≥ ~1 TB HBM** (user): H100 is out (640 GB < ~690 GB of weights); 6 or 7 GPUs are
  out (128 heads and 256 experts do not divide); QuantTrio AWQ is out because it 4-bit-quantizes the
  indexer itself (verified `indexer.wq_b.qweight` in its tensor map). NVFP4 builds are Blackwell-only.
- **No official-number accuracy gate for exp3**: GPQA is gated and was dropped, MMLU-Pro deferred, and
  neither model publishes a comparable PPL — the identical-rerun noise floor is the reference instead.
- **Hot-set "hotness" = per-layer selection frequency over the decode** — an offline oracle ranking, so
  every coverage number is an upper bound; pool N = final candidate count; budgets are nested.
- **LongBench uses the three 512-output summarization tasks** (multi_news, gov_report, qmsum) because the
  metric needs decode steps, not prompt length; ≤20K-token filter, seed 42, official templates.

## 6. Gotchas
- ds4: no `--seed`; t/s not printed under `--dump-logprobs` (derive throughput from `/usr/bin/time`);
  `DS4_TRACE_DECODE_ONLY=1` is essential (a prefill trace is GBs); RULER `prepare.py` needs
  `work/experiment/shim/python` → python3 on PATH. CPU prefill is O(n²): 64K ≈ 26 h, 128K ≈ 75 h.
- **Shared machine** (2× Xeon, 251 GB RAM; other users run Vivado ≈ 100 GB). Check `free -h` before
  loading anything ≥ 80 GB — a thrashing 161 GB load once froze the box.
- Launch long runs detached (`setsid nohup … < /dev/null &`) plus a background watcher loop; a network
  drop once killed a 24 h run. Runners skip completed runs (guard on `outputs/generations.jsonl`).
  A large `git pull` of the trace repo must be detached too: an interrupted checkout leaves untracked
  partial files that block the merge.
- Harness: a bare trailing `sleep` in a Bash call is blocked; use until-loops or run_in_background.
- llama.cpp local patches (uncommitted, in `work/llama.cpp`): `ggml_cont()` before the MLA absorbed
  matmuls in `src/models/deepseek2.cpp` (~line 295) and `src/llama-graph.cpp` (~line 2427) — required on
  CPU (scheduler assert `cur_backend_id != -1`); `dsa_topk` tensor rename in `src/models/deepseek32.cpp`
  (~line 345); diagnostic prints in `ggml/src/ggml-backend.cpp:~1242` and `ggml/src/ggml-cpu/ops.cpp:~4946`.
  HF `unsloth/DeepSeek-V3.2-Exp-GGUF` is gated; use `-GGUF` (no Exp).
- Repo hygiene: before committing anything to sparse_attn_cpu, scrub the absolute work-tree prefix to
  `<WORKDIR>` and the home prefix to `<HOME>`; raw traces stay local. WebFetch caches 15 min — verify
  live pages with a `?v=` query string.
- **Scripts have no `--help` and act on any argv**: `generate_longbench_plots.py X` writes plots into a
  directory named `X`; `build_longbench_prompts.py` ignores args and regenerates `prompts/longbench/` +
  the manifest (deterministic, seed 42); `aggregate_longbench.py` rewrites the aggregate JSON
  (deterministic); `analyze_hotset_coverage.py` / `analyze_moe_concentration.py` with no run dirs print
  empty tables. Read the script header instead of probing with `--help`.

## 7. Open issues
- **Top item — resume exp3 tier 2 when 8 GPUs are free at TP = 8**: the last 175 `perm_periodic_B`
  generation items and the `clean2` identical-rerun generation floor (the floor for the "identical token
  streams" column), then GLM-5.2 / GLM-5 tier-2 PPL + ACC (~2.5 h each), then tier 3 (not yet designed).
  Resume verbatim with `scripts/gpu/exp3_tier2.py --resume` (`--skip-ppl` for the generation block only);
  ≈10 h wall per model. Nothing else in the campaign is outstanding.
- **v6 policy re-sweep on the V3.2 exports** (`exports/v6_v32/`) is not done — the v6 conclusions were
  fit on V4 traces. Consumer: ramulator-owner, branch v6.
- **Ramulator digest `ref_dsv4_kv_locality_study.md` §2** still tabulates the old `_q2` V4 series
  (4K/8K/16K/40K/64K, "128 decode tokens", 64K adj 0.670, lift 13.2× at 40K) instead of the published
  `_moe_q2` points (32K/64K = 0.672 / 0.668, 10.53× / 21.37×). Report to ramulator-owner; we do not edit it.
- **`v5_reindex_correctness_ds4.md` §2** calls `ds4_nofm` "the unmodified engine" — it is built from the
  hooked source (hook inert at MODE=0) and its patch copy lacks the `DS4_PERM_IDENTITY` lines.
- `runs/niah_single_2_98304_q2` is an aborted 96K attempt (no outputs, 468 KB); the runner will redo it.
- `work/experiment/exports/partial_checkout_20260829/` is a 23 GB partial copy from an interrupted pull —
  safe to delete.
- CPU V3.2 tracer parked (§3); if resumed, move the `dsa_trace_cb` hook into `llama-server`.
- Optional: ROUGE-score the LongBench generations (`runs/lb_*/outputs/generations.jsonl`) for a
  quality-sanity table; optional 128K CPU RULER point (~75 h, needs a prompt via
  `benchmark/RULER prepare.py --max_seq_length 131072` + a `build_prompts.py` LENGTHS edit).

## 8. Frequently used commands
```bash
EXP=<WORKDIR>/experiment
cd <WORKDIR>/ds4 && make cpu          # rebuild engine
# RULER sweep (both traces): run-ids niah_single_2_<L>_moe_q2
bash $EXP/scripts/run_experiment_moe.sh "4096 8192 16384" 256
# LongBench: build prompts, run (resumable), analyze, aggregate, plot
python3 $EXP/scripts/build_longbench_prompts.py
setsid nohup bash $EXP/scripts/run_longbench.sh > $EXP/logs_longbench_driver.log 2>&1 < /dev/null &
bash $EXP/scripts/analyze_longbench_all.sh          # ingest→validate→R1→R2→R3 per run
python3 $EXP/scripts/aggregate_longbench.py         # → analysis_longbench/longbench_aggregate.json
python3 $EXP/scripts/generate_longbench_plots.py    # → analysis_longbench/plots/lb_0{1,2,3}*.png
# Per-run pieces
python3 $EXP/scripts/{ingest_trace,validate_trace,analyze_locality}.py <run_dir>
python3 $EXP/scripts/{ingest_moe_trace,analyze_moe_locality}.py <run_dir>
python3 $EXP/scripts/analyze_moe_concentration.py <plot_dir> <run_dir...>
python3 $EXP/scripts/analyze_hotset_coverage.py <plot_dir> <run_dir...>
python3 $EXP/scripts/generate_moe_plots.py <out> <run_dir...>; generate_kv_plots.py likewise
# v6 exports for the ramulator policy study
python3 $EXP/scripts/export_v6_traces.py                       # V4  → exports/v6/
bash    $EXP/exports/run_v6_gpu_export.sh                      # GPU → exports/v6_v32/, exports/v6_glm/
# V3.2 DSA dev tracer (4-layer model; short prompts only until the long-context bug is fixed)
cd <WORKDIR>/llama.cpp && cmake --build build -j48 --target llama-eval-callback
DSA_TRACE_OUTPUT=/tmp/dsa.jsonl ./build/bin/llama-eval-callback -m ../models/v32_4L/DeepSeek-V3.2-4Layers-Q8_0.gguf -t 48 -c 4096 -n 8 -p "..." -ngl 0
```
Trace env for ds4 runs: `DS4_TRACE_OUTPUT=<dir> DS4_MOE_TRACE=1 DS4_TRACE_LEVEL=3 DS4_TRACE_DECODE_ONLY=1
DS4_TRACE_FULL_SCORE_SAMPLE_RATE=0.002 DS4_TRACE_FLUSH_INTERVAL=32 OMP_NUM_THREADS=64`.
Trace records: `indexer_trace.jsonl {phase,layer,pos,n_comp,top_k,sel[],scores[]}`,
`moe_trace.jsonl {phase,layer,pos,token,is_hash,sel[6],weights[6]}`.
Export gotcha: `n_comp = (pos + 1) // ratio`, not `pos // ratio` (asserted per run against the jsonl).

## 9. Interfaces / dependencies
- Engines: antirez/ds4 (MIT, 80ebbc3) with `docs/ds4_instrumentation.patch` (in sparse_attn_cpu;
  reverse-applies clean to HEAD); llama.cpp (master ~July 2026; `deepseek32` arch + DSA KV cache);
  vLLM 0.28.0 tag 2cf0a69 on the GPU node (contains pin 5559679).
- Data: NVIDIA/RULER (38da79d), zai-org/LongBench `data.zip` (extracted in `benchmark/longbench/`),
  InfiniteBench + LongBench-v2 on the GPU node, DeepSeek tokenizer in `work/experiment/tokenizer/`.
  HF anonymous downloads work (`hf download`).
- Python: pandas / pyarrow / matplotlib / tokenizers; metric helpers in `scripts/locality_lib.py`
  (id-set-agnostic — reused by KV, MoE and hot-set analyses). Metric reference:
  `sparse_attn_cpu/docs/00_doc/locality_metrics.md`.
- Publishing convention: analysis artifacts + scripts go to sparse_attn_cpu (`docs/*_sweep/`), raw traces
  stay local; report-page material goes to `vercel-owner` through main, never committed by us.
- **Consumers of these numbers (read-only for us; report needed changes to main, never edit):**
  - Ramulator `ramulator2/examples/v5_reindex_gather_testcase.py` — `MEASURED_OVERLAP` and
    `MEASURED_TOP10` calibrate its synthetic selection process; currently the V4 values
    (`{4096: 0.869, 8192: 0.790, 16384: 0.718, 32768: 0.672, 65536: 0.668}` and
    `{0.201, 0.349, 0.575, 0.767, 0.907}`). The V3.2 rung means in §4 are the replacement set.
  - Ramulator v6 policy study (`examples/v6_policy/traces.py`, branch v6) — consumes the npz exports.
  - Ramulator digest `ramulator2/00_doc/01_design/v5/ref_dsv4_kv_locality_study.md` (see §7 for its
    two known staleness items).
  - Evaluation `ramulator2/00_doc/03_evaluation/` (§6 hit-ratio inputs) and the paper's §3.3 +
    correctness paragraph (`docs/reindex_accuracy/paper_accuracy_section.md`, incl. the tier-2 addendum
    with Tables 3/4). Both quote the digest — any re-measurement must be announced to both owners.
