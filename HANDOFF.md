<!-- Repo copy of the project-root HANDOFF.md. Paths scrubbed per repo convention:
     <PROJECT> = the local project root; <WORKDIR> = <PROJECT>/work; <RAMULATOR_WS>, <VLLM_CHECKOUT>,
     <CLAUDE_MEMORY_DIR> = sibling checkouts / the Claude auto-memory dir on the CPU box. -->

# HANDOFF.md — DeepSeek sparse-selection locality study

Written 2026-08-25 for an agent with no memory of prior sessions; refined 2026-08-27 by the
`deepseek-owner` agent (onboarding survey: repo states, cross-checks, ds4 hook, consumers). Read this first, then
`<CLAUDE_MEMORY_DIR>/memory/deepseek-v4-kv-locality-experiment.md`
(the auto-memory file: full history of runs, numbers, gotchas — the single most important doc).

## 1. Purpose
Measure the **temporal locality of sparse selections** inside DeepSeek-V4 (and, next, V3.2 / GLM-5) for
an architecture paper (ISCA/HPCA/ASPLOS target): which KV entries the CSA/DSA "lightning indexer" gathers
per token (top-k), and which MoE experts the router picks, and how much those sets overlap across
consecutive decode steps. Outputs feed a hot-set/prefetch cache design. Three result families:
R1 KV top-k locality, R2 MoE routing locality, R3 hot-set coverage (% of pool to cover 99% of accesses).

## 2. Layout (absolute paths)
- `work/experiment/` — the whole pipeline. `runs/<id>/{traces,outputs,analysis,prompts,logs}`;
  `scripts/`; `prompts/`; `benchmark/{RULER,longbench}`; `analysis_moe/` (RULER-sweep plots);
  `analysis_longbench/` (aggregate + plots). `EXPERIMENT_SUMMARY.md` = RULER write-up.
- `work/ds4/` — antirez/ds4 C engine (HEAD 80ebbc3) **+ uncommitted local changes in `ds4.c` only**
  (+410/−8 lines, 2026-08-27 survey): (a) instrumentation — KV logger `indexer_log_selection`, MoE
  logger `moe_log_selection` (= `sparse_attn_cpu/docs/ds4_instrumentation.patch`, still reverse-applies
  clean); (b) the **v5 re-index correctness hook** added 2026-08-24 by the ramulator project
  (`kv_perm_apply`, env `DS4_PERM_MODE/SEED/PERIOD/FRAC/HCA/LOG/IDENTITY`; default MODE=0 = inert).
  Patch copy: `ramulator2/examples/v5_ds4_reindex_correctness/ds4_reindex_hook.patch` (absolute-path
  headers, needs `-p7`; it is 2 lines behind ds4.c — lacks the `DS4_PERM_IDENTITY` control).
  Keep both; never revert. Binaries (untracked): `ds4` (built 08-24 10:15, hook without IDENTITY),
  `ds4_new` (current source), `ds4_nofm` (current source, `-fno-fast-math`). `make cpu` rebuilds `ds4`.
- `work/models/` — `DeepSeek-V4-Flash-IQ2XXS-...gguf` (81 GB, the workhorse);
  `v32/DeepSeek-V3.2-UD-TQ1_0.gguf` (151 GB, dense, NO DSA indexer — useless for sparse data);
  `v32_4L/DeepSeek-V3.2-4Layers-Q8_0.gguf` (16 GB, arch deepseek32 WITH indexer; dev-only).
- `work/llama.cpp/` — pinned to 683f0c72e (2026-07-09) with local patches (see §5). Not pushed anywhere.
- `01_github/sparse_attn_cpu/` — published repo (git@github.com:yanggon-kim/sparse_attn_cpu). HEAD = the
  2026-08-27 GPU-campaign commits (e9e6610 package; + the "no review stops / generic GPU" revision), pushed to
  origin main. Holds `docs/00_doc/GPU_CAMPAIGN.md`, `exp0..exp3`, and a path-scrubbed copy of this HANDOFF.md
  (refresh it when this file changes and a commit is due).
- `01_github/versel_distribute/` — static site → https://versel-distribute.vercel.app. Local HEAD deaaaa9,
  clean, **6 commits behind origin/main (e00b92a, user's Part X/Part III-of-05_ramulator edits; none
  touch `04_sparse_attn/02_part3*`, `03_part4*`; `index.html` 1 line)** — rebase before any push.
  Has its own `CLAUDE.md` (READ IT): plain HTML, relative links only, every report self-contained,
  `index.html` is the only nav; "apply the edit rule" = add `<a class="report-link">` for new pages.
- `papers/`, `00_doc/`, `study_deepseek_v4_vllm/`, `PLAN.md`, `RESULTS.md` — background docs.

## 3. Current status
**Works (published):**
- RULER 4K/8K/16K/32K/64K sweep on ds4 (`runs/niah_single_2_{L}_moe_q2`): KV adj overlap
  0.868/0.790/0.718/0.672/0.668, lift 1.7/2.9/5.7/10.5/21.4×; MoE learned 0.33–0.38 (~16× random,
  context-independent); hash layers 0–2 ≈ random. Hot-set A@99 80%→20%.
- LongBench summarization (36 runs, `runs/lb_{multi_news,gov_report,qmsum}_s{0..11}_q2`): all 36 rc=0,
  validate PASS, 15,073 decode steps, 4.5 GB raw retained locally. Aggregate:
  `work/experiment/analysis_longbench/longbench_aggregate.json`. Real tasks lie ON the RULER curve.
- Long-decode run `runs/longform_p16k_g4k_q2` (3,019 steps) — retention keeps decaying; short traces
  (<~500 steps) falsely show a plateau. Always decode ≥1–2K steps for retention curves.
- Vercel reports Part III (KV, `04_sparse_attn/02_part3_cpu_kv_locality/`) and Part IV (MoE,
  `03_part4_moe_locality/`) include RULER + LongBench sections. Parts I/II/V/VI/VII are the user's own.
- **Re-index correctness runs for the ramulator v5 design (2026-08-23/24, run by that project with
  the hook above, 4K NIAH, `-n 128`, ~55 min each on 64 threads):** identity control bit-identical to
  baseline; one-time random row permutation 127/128 tokens identical, needle retrieved, selection Jaccard
  0.967 in original-index space; periodic swaps and the `-fno-fast-math` control both diverge at step 35
  → residual = summation-order noise. Raw runs (90 MB, not ours to delete):
  `<RAMULATOR_WS>/tmp_v5_gather/ds4_reindex/`; write-up
  `ramulator2/00_doc/01_design/v5/v5_reindex_correctness_ds4.md`.
- Two RULER run series exist: the *published* `niah_single_2_{4096,8192,16384,32768,65536}_moe_q2`
  (`-n 256`, 117–163 steps, KV+MoE traces) and the older `niah_single_2_{4096,8192,16384,40960,65536}_q2`
  (`-n 128`; 40K adj 0.659 / lift 13.2×, 64K 0.670 / 21.4×). Quote the `_moe_q2` series; the ramulator
  digest's §2 table still shows the old series (see §8). `runs/niah_single_2_98304_q2` is an **aborted**
  96K attempt (died in prefill at layer 34/43, no outputs, 468 KB) — the runner will redo it if asked.

**Cross-checked 2026-08-27 (owner onboarding, no reruns):** RULER adj overlap
0.868/0.790/0.718/0.672/0.668 and lift 1.72/2.92/5.72/10.53/21.37× = `runs/niah_single_2_{L}_moe_q2/
analysis/metrics_run_summary.json` (`overall_adjacent_overlap_mean`, `overall_locality_lift_mean`);
hot-set A@99 79.5/65.7/47.5/31.4/20.4 % of pool (pool 1,030/1,912/4,095/8,039/16,393; per-layer range
9.9–38.4 % at 64K) = `analysis/hotset_coverage.json` (`A99_pct_mean`, `A99_pct_range`); LongBench
aggregate (36 runs) multi_news 0.914 / gov_report 0.755 / qmsum 0.732 adj, pooled 0.801, A@99
92.0/78.7/61.4 % = `analysis_longbench/longbench_aggregate.json` (identical to the published copy in
`sparse_attn_cpu/docs/longbench_sweep/`). All match the Vercel/GitHub write-ups.

**In progress / parked:**
- DeepSeek-V3.2 on CPU: engine patched and loads, but the only CPU-fittable GGUF (Unsloth TQ1_0) is
  dense-stripped; DSA-preserving GGUFs (`sszymczyk/DeepSeek-V3.2-*-light-GGUF`) are 404–714 GB.
  Standalone DSA tracer (`work/llama.cpp/examples/eval-callback/eval-callback.cpp`, target
  `llama-eval-callback`) works at short context but crashes on prompts >~256 tokens
  (`ops.cpp set_rows: leaf_90 uninitialized`) while `llama-server` handles the same prompt fine.
  Fix path: hook the trace into llama-server's decode path instead. PARKED — user pivoted to GPU.
- GPU campaign (decided, not started): rent **8 GPUs with >= ~1 TB HBM total (H200/B200-class; GPU type
  NOT decided yet — exp0 detects it and picks kernels)**, native FP8
  `deepseek-ai/DeepSeek-V3.2`, vLLM TP8, `enforce_eager=True`; ladder 8K/16K/32K/64K/128K input,
  **~2K decode steps** per run, n≈20/rung; RULER + LongBench v1 + LongBench v2 + InfiniteBench;
  decode-only trace + sampled prefill. Budget ~1,000 GPU-h. GLM-5/5.2 (~750B, also DSA) fit the
  same node → cross-model generality opportunity. Guide: `sparse_attn_cpu/docs/vllm_selection_history_collection_guide.md`.

## 4. Recent decisions and why
- Stay on **native FP8 / 8 GPUs, >= ~1 TB HBM** (GPU model open, 2026-08-27; not H100: 640 GB < ~690 GB weights; not 6/7 GPUs: 128 heads &
  256 experts don't divide; not QuantTrio AWQ: it 4-bit-quantizes the indexer itself — verified
  `indexer.wq_b.qweight` in its tensor map). NVFP4 builds are Blackwell-only.
- Hot-set "hotness" = per-layer selection frequency over the decode (offline oracle ranking,
  upper bound); pool N = final candidate count; budgets nested.
- LongBench: 512-output summarization tasks chosen for decode-step count; ≤20K-token filter, seed 42.
- MoE context-scaling plot uses a 0-based lift axis (autoscale exaggerated a 16→14.9× dip).

## 5. Gotchas
- ds4: no `--seed`; t/s not printed under `--dump-logprobs` (throughput derived from `/usr/bin/time`);
  `DS4_TRACE_DECODE_ONLY=1` essential (prefill trace = GBs); RULER `prepare.py` needs
  `work/experiment/shim/python`→python3 on PATH. CPU prefill is O(n²): 64K ≈ 26 h, 128K ≈ 75 h.
- **Shared machine** (2× Xeon, 251 GB RAM). Other users run Vivado (≈100 GB). Check `free -h` before
  loading anything ≥80 GB; a thrashing 161 GB load once froze the box. User asked to yield memory.
- Always launch long runs detached: `setsid nohup ... < /dev/null &` + a background watcher loop; a
  network drop once killed a 24 h run. Runners skip completed runs (guard on `outputs/generations.jsonl`).
- Harness: a bare trailing `sleep` in a Bash call gets blocked; use until-loops or run_in_background.
- llama.cpp local patches (uncommitted, in `work/llama.cpp`): `ggml_cont()` before the MLA absorbed
  matmuls in `src/models/deepseek2.cpp` (~line 295) and `src/llama-graph.cpp` (~line 2427) — required
  for CPU (scheduler assert `cur_backend_id != -1`); `dsa_topk` tensor rename in
  `src/models/deepseek32.cpp` (~line 345); diagnostic prints in `ggml/src/ggml-backend.cpp:~1242` and
  `ggml/src/ggml-cpu/ops.cpp:~4946`. HF `unsloth/DeepSeek-V3.2-Exp-GGUF` is gated; use `-GGUF` (no Exp).
- Repo hygiene: scrub `/home/yanggon/...work` → `<WORKDIR>` in anything committed to sparse_attn_cpu.
  versel remote diverges often (user commits there) → `git fetch && git rebase origin/main` before push.
  WebFetch caches 15 min: verify live pages with a `?v=` query string.
- Vercel Part I is the user's B200/vLLM study — never edit its numbers.
- **Scripts have no `--help` and act on any argv**: `generate_longbench_plots.py X` writes plots into a
  directory named `X`; `build_longbench_prompts.py` ignores args and regenerates `prompts/longbench/` +
  the manifest (deterministic, seed 42 — verified byte-identical 2026-08-27); `aggregate_longbench.py`
  rewrites the aggregate JSON (deterministic). `analyze_hotset_coverage.py`/`analyze_moe_concentration.py`
  with no run dirs print empty tables. Read the script header instead of probing with `--help`.
  (A stray `scripts/--help/` directory from such a probe may still exist — safe to delete.)

## 6. Frequently used commands
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
# V3.2 DSA dev tracer (4-layer model; short prompts only until the long-context bug is fixed)
cd <WORKDIR>/llama.cpp && cmake --build build -j48 --target llama-eval-callback
DSA_TRACE_OUTPUT=/tmp/dsa.jsonl ./build/bin/llama-eval-callback -m ../models/v32_4L/DeepSeek-V3.2-4Layers-Q8_0.gguf -t 48 -c 4096 -n 8 -p "..." -ngl 0
```
Trace env for ds4 runs: `DS4_TRACE_OUTPUT=<dir> DS4_MOE_TRACE=1 DS4_TRACE_LEVEL=3 DS4_TRACE_DECODE_ONLY=1
DS4_TRACE_FULL_SCORE_SAMPLE_RATE=0.002 DS4_TRACE_FLUSH_INTERVAL=32 OMP_NUM_THREADS=64`.
Trace records: `indexer_trace.jsonl {phase,layer,pos,n_comp,top_k,sel[],scores[]}`,
`moe_trace.jsonl {phase,layer,pos,token,is_hash,sel[6],weights[6]}`.

## 7. Next steps (priority)
0. **GPU campaign package written (2026-08-27, not executed):** `01_github/sparse_attn_cpu/docs/00_doc/
   GPU_CAMPAIGN.md` is the top-level file to hand to the agent on the rented 8-GPU node (its §8 has the
   ready-to-paste prompt); it links `exp0_environment.md` (pin/install vLLM, models, TP8 smoke, hook unit
   checks), `exp1_dsv32_gather_index.md` (V3.2 hook at `flashmla_sparse.py:838 forward_mqa`, adapter to the
   ds4 run-dir schema so ingest/validate/analyze run unchanged, 8K–128K ladder, bf/ld run kinds),
   `exp2_glm_gather_index.md` (GLM-5.2 index-share handling, GLM-5 DSA check), `exp3_reindex_accuracy.md`
   (block-granular A vs entry-granular B re-index, ds4-mirrored modes/controls, benchmark set + official-number
   gate, `docs/reindex_accuracy/{results.csv,per_item.jsonl,README.md}`). Citations verified against vLLM
   checkout `5559679` (`<VLLM_CHECKOUT>`).
   **Revised 2026-08-27 (user decisions):** GPU type generic (8 GPUs, >= ~1 TB HBM; exp0 §1 detects model/SM/
   driver/HBM and picks kernels — first VERIFY item); execution mode = run start to finish **without review
   stops**: gates are self-checks, per-gate report is a file `docs/00_doc/reports/<phase>_<date>.md` + HANDOFF
   Status line + per-experiment summary md; stop only for §5 (a)–(d) (gate fails after one retry / method-
   changing VERIFY outcome / undecided decision incl. budget > 20 % overrun / clean run outside official range).
1. Remaining pre-node prep (no GPU needed): pre-build prompt sets (RULER 8K–128K via `prepare.py`,
   LongBench v1/v2, InfiniteBench manifests in the `longbench_samples.jsonl` schema); optionally draft
   `scripts/vllm_to_ds4_run.py` (adapter of exp1 §4) and test it on a synthetic trace (`scripts/_synth_test.py`).
2. GLM-5/5.2 hook = same code path (`GlmMoeDsaForCausalLM`); the only open point is whether plain GLM-5 has
   `index_topk` (checked on the node, exp2 §1).
3. Optional: ROUGE-score the LongBench generations (`runs/lb_*/outputs/generations.jsonl`) for a
   quality-sanity table; optional 128K CPU RULER point (~75 h, needs prompt via
   `benchmark/RULER prepare.py --max_seq_length 131072` + `build_prompts.py` LENGTHS edit).
4. If the CPU V3.2 tracer is resumed: move the `dsa_trace_cb` hook into `llama-server`.

## 7b. GPU campaign — prerequisites the user must supply before it can start
Rental account/credentials for an 8-GPU node with >= ~1 TB HBM (H200/B200-class; ~1,000 GPU-h budget) and the go date; confirmation of the
model (`deepseek-ai/DeepSeek-V3.2` native FP8, plus GLM-5/5.2 if wanted); HF token if any target repo is
gated; the session URL for the commit trailer. Everything else (vLLM pin, hook points, schema adapter,
ladder, benchmarks, re-index implementations, gates) is specified in `GPU_CAMPAIGN.md` + `exp0..exp3`
(see §7 item 0); the GPU agent resolves the "VERIFY ON THE NODE" list (GPU_CAMPAIGN.md §7) itself.
Traces land on the node's NVMe (1–2 TB); only analysis artifacts come back into `sparse_attn_cpu/docs/`.

## 8. Interfaces / dependencies
- Engines: antirez/ds4 (MIT, 80ebbc3) with `docs/ds4_instrumentation.patch` (in sparse_attn_cpu; reverse-
  applies clean to HEAD); llama.cpp (master ~July 2026; `deepseek32` arch + DSA KV cache).
- Data: NVIDIA/RULER (38da79d), zai-org/LongBench `data.zip` (extracted in `benchmark/longbench/`),
  DeepSeek tokenizer in `work/experiment/tokenizer/`. HF anonymous downloads work (`hf download`).
- Python: pandas/pyarrow/matplotlib/tokenizers; metric helpers in `scripts/locality_lib.py` (id-set-agnostic
  — reused by KV, MoE, and hot-set analyses).
- Publishing convention: analysis artifacts + scripts go to sparse_attn_cpu (`docs/*_sweep/`), raw traces
  stay local; report pages to versel (assets in sibling `assets/`, verify with html.parser + live fetch).
- **Consumers of these numbers (read-only for us; report changes to main, never edit):**
  - Ramulator `01_ramulator/01_github/ramulator2/examples/v5_reindex_gather_testcase.py` —
    `MEASURED_OVERLAP = {4096: 0.869, 8192: 0.790, 16384: 0.718, 32768: 0.672, 65536: 0.668}` and
    `MEASURED_TOP10 = {0.201, 0.349, 0.575, 0.767, 0.907}` (coverage of the hottest 10 % of the pool,
    `docs/kv_hotset_coverage.md` 10 % row) calibrate its synthetic selection process. (4K 0.869 is the
    old-series value; published 0.868 — immaterial.)
  - Ramulator digest `ramulator2/00_doc/01_design/v5/ref_dsv4_kv_locality_study.md`: §3–§6 match our
    artifacts; **§2 table is the old `_q2` series (4K/8K/16K/40K/64K, "128 decode tokens", 64K adj 0.670,
    lift 13.2× at 40K)**, not the published 32K/64K `_moe_q2` points (0.672/0.668, 10.53×/21.37×).
    `v5_reindex_correctness_ds4.md` §2 calls `ds4_nofm` "the unmodified engine" — it is built from the
    hooked source (hook inert at MODE=0), and its hook patch lacks the `DS4_PERM_IDENTITY` lines.
  - Evaluation `ramulator2/00_doc/03_evaluation/` (§6 hit-ratio inputs) and the paper's §3.3
    (`00_doc/02_writing_paper/`) quote the digest; any re-measurement must be announced to both.
