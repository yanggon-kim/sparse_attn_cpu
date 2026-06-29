# DeepSeek-V4 Sparse-Attention KV Temporal Locality Experiment Plan

## 0. Purpose

This document defines a reproducible experiment for measuring **temporal locality in the compressed KV entries selected by DeepSeek-V4 sparse attention during autoregressive decoding**.

The central question is:

> For each generated token and each sparse-attention layer, which compressed KV entries are selected, and how much does that selected set overlap with the selected set for nearby decode steps?

The experiment must preserve enough raw evidence to support later re-analysis, while also producing standardized summary metrics. The implementation must work first on the current Linux CPU server and later be reusable on GPU backends without changing the logical trace format or analysis definitions.

This is an **instrumentation and systems experiment**, not only a benchmark accuracy run. Benchmark scores are useful for validating that the traced execution remains meaningful, but the primary output is the layer-by-layer, token-by-token KV-selection trace.

---

## 1. Research Questions

The implementation and analysis must answer the following questions.

1. **Adjacent-token temporal locality**
   - At the same sparse-attention layer, how many compressed KV indices selected for decode step `t` are selected again at step `t+1`?
   - How stable are the top ranks and index scores?

2. **Longer-horizon temporal locality**
   - How much of the selected set at step `t` is still present at `t+2`, `t+4`, `t+8`, `t+16`, and later?
   - What is the per-layer working-set growth over a generation window?

3. **Layer dependence**
   - Which CSA layers show high or low temporal locality?
   - Do shallow, middle, and deep CSA layers behave differently?
   - Do different layers select semantically similar historical positions, even though their physical KV tensors are separate?

4. **Task dependence**
   - Does locality differ between simple retrieval, multi-hop retrieval, aggregation, long-form generation, and dense global-memory tasks?

5. **Context-length dependence**
   - Does locality increase, decrease, or remain stable as the context grows?

6. **Correctness dependence**
   - Are locality patterns different for correct and incorrect benchmark responses?

7. **Systems relevance**
   - How many unique compressed KV blocks are needed over short decode windows?
   - What fraction of accesses could potentially hit a retained hot set?
   - How large is the reuse distance in logical compressed-KV blocks?

---

## 2. Scope and Terminology

### 2.1 Primary object of observation

For decode step `t` and sparse-attention layer `l`, define:

```text
S[t, l] = ordered list of compressed KV indices selected by the lightning indexer
```

The corresponding unordered set is:

```text
U[t, l] = set(S[t, l])
```

The experiment focuses primarily on **CSA layers**, because CSA performs sparse top-k selection. HCA layers do not use the same sparse top-k selection mechanism and should be logged as a separate attention type.

### 2.2 Logical locality vs. physical locality

The initial experiment measures **logical KV locality**:

```text
(layer_id, compressed_kv_index)
```

This is portable across CPU and GPU backends.

Physical locality is backend-specific and may additionally include:

```text
logical_page_id
allocator_block_id
virtual_address_offset
NUMA_node
GPU_page_id
device_buffer_offset
```

Do not confuse cross-layer similarity with physical cache reuse. The same compressed index number in two different layers refers to different layer-owned KV tensors.

### 2.3 Expected model configuration

DeepSeek-V4-Flash is described as using interleaved CSA and HCA after initial layers, with CSA compression and a fixed sparse top-k. However, the implementation must **not hard-code paper values**. It must read or infer the actual runtime configuration and record:

- number of transformer layers;
- attention type for every layer;
- CSA compression ratio;
- HCA compression ratio;
- sparse top-k;
- sliding-window size;
- indexer head count and head dimension;
- model quantization;
- actual tensor dtypes.

If the runtime differs from the paper, the runtime values are the source of truth.

---

# Part I — Benchmark Selection

## 3. Candidate Benchmarks

The first implementation should use **exactly one primary benchmark**, not all benchmarks at once. The purpose is to validate the tracing pipeline on a controlled workload before expanding.

### 3.1 RULER — recommended default for the first implementation

**Use when:** the goal is controlled, interpretable locality analysis across known task structures and context lengths.

Recommended task categories:

- single-needle or single-key retrieval;
- multi-key retrieval;
- multi-hop tracing;
- aggregation or counting;
- variable-position retrieval.

Why it is useful:

- context length can be controlled;
- evidence positions are known;
- task categories induce different expected access patterns;
- synthetic construction makes interpretation easier;
- it supports controlled comparisons across context lengths.

Expected locality patterns:

- single retrieval may create a small persistent hot set;
- multi-hop tasks may produce phase changes;
- aggregation may produce a broader working set.

**Default decision:** choose RULER unless the primary objective is specifically long-form generation.

---

### 3.2 LongGenBench — best for long consecutive decode traces

**Use when:** the main question is stability across many generated tokens.

Why it is useful:

- produces longer outputs than multiple-choice benchmarks;
- provides enough consecutive decode steps for retention curves and reuse-distance analysis;
- enables analysis across 128–512 generated tokens.

Expected locality patterns:

- stable topic segments may show strong persistence;
- transitions between sections or reasoning stages may cause abrupt churn.

---

### 3.3 LongMemEval — best for realistic long-term conversational memory

**Use when:** the goal is to study retrieval from old sessions, knowledge updates, or temporal reasoning.

Useful task categories:

- single-session retrieval;
- multi-session reasoning;
- temporal reasoning;
- knowledge update;
- abstention or no-answer cases.

Expected locality patterns:

- persistent access to one historical session;
- switching between sessions;
- old/new fact competition.

---

### 3.4 LongBench-v2 — broad realism validation

**Use when:** the goal is to show that locality behavior generalizes across realistic document, dialogue, code, and structured-data tasks.

Advantages:

- diverse task types;
- realistic long-context inputs;
- useful as a secondary validation suite.

Limitation:

- many tasks produce short answers, which gives too few decode steps for strong temporal analysis.

Recommended approach:

- preserve one official-scoring run;
- separately create a trace-only run that requests a 128–256 token explanation;
- never report the trace-only prompt as the official benchmark score.

---

### 3.5 MRCR or Michelangelo-style dense-memory stress tests

**Use when:** the goal is to find a lower-locality counterexample or a workload requiring broadly distributed global memory.

Why it is useful:

- stresses dense global retrieval;
- prevents over-generalizing from needle-style tasks;
- may reveal high churn, large working sets, or long reuse distances.

This benchmark is especially important after a high-locality result has already been observed on a simpler benchmark.

---

### 3.6 Custom controls

These are not substitutes for a public benchmark, but should eventually be included.

#### No-context control

Append a question unrelated to the long context.

Purpose:

- measure background or false-positive global selection;
- estimate the indexer’s context-independent access floor.

#### Local-only control

Place all answer-relevant information inside the recent local window.

Purpose:

- test whether global compressed-KV selection becomes unnecessary or remains active;
- isolate recency effects.

#### Random-selection control

Replace the indexer output with a random set of equal size, only in a separate diagnostic run.

Purpose:

- estimate expected overlap and working-set growth under a non-semantic selector;
- never mix this run with normal benchmark results.

---

## 4. Benchmark Decision Procedure

Before modifying the runtime, the agent must produce a short benchmark decision note.

Create:

```text
experiment/benchmark_decision.md
```

It must contain:

```text
Selected benchmark:
Selected task subset:
Selected split/version:
Planned context lengths:
Planned number of samples:
Planned max_new_tokens:
Reason for selection:
Expected locality pattern:
Dataset source and commit/version:
```

### Required selection rule

Use the following decision tree:

1. If this is the first instrumentation run, select **RULER**.
2. If the main goal is at least 128 consecutive generated tokens, select **LongGenBench**.
3. If realistic dialogue memory is the priority, select **LongMemEval**.
4. If broad realism is the priority, select **LongBench-v2**.
5. If testing a low-locality or dense-memory counterexample, select **MRCR**.
6. If the selected benchmark cannot be obtained or run, document the reason and choose the next closest option.

### Recommended first pilot

```text
Benchmark: RULER
Task subsets:
  - one retrieval task
  - one multi-hop task
  - one aggregation task
Context lengths:
  - 32K
  - 64K
Samples:
  - 3 per task per context length for smoke testing
  - 10–20 per task per context length for the pilot
Max new tokens:
  - 128
Decoding:
  - greedy
```

Do not begin with 256K or 512K until the tracer is validated and runtime cost is measured.

---

# Part II — Reproducible Run Design

## 5. Experiment Phases

### Phase A — code discovery

The agent must locate:

1. indexer-score computation;
2. top-k selection;
3. selected-index tensor;
4. sparse core-attention invocation;
5. per-layer attention-type dispatch;
6. decode-step loop;
7. model configuration loading;
8. CPU tensor storage;
9. GPU tensor readback APIs, if present.

The most important instrumentation point is:

```text
after top-k selection
before sparse core attention consumes the selected indices
```

### Phase B — minimal deterministic smoke test

Run one short sample with:

```text
context length: 2K–8K
max new tokens: 4–8
temperature: 0
batch size: 1
tracing: enabled
```

Verify:

- trace file is written;
- all selected indices are within valid bounds;
- layer IDs and attention types are correct;
- token IDs are identical with tracing on and off;
- no malformed or partial records exist.

### Phase C — benchmark pilot

Run the selected benchmark at 32K and 64K with a small sample count.

### Phase D — full experiment

Expand sample count and context length only after:

- trace correctness checks pass;
- storage volume is known;
- runtime overhead is measured;
- analysis scripts produce expected metrics.

### Phase E — GPU replication

Reuse the same trace schema and analysis pipeline with a GPU-specific trace adapter.

---

## 6. Deterministic Decode Configuration

The default experiment must use:

```text
batch_size = 1
temperature = 0
greedy decoding
top_p disabled
top_k sampling disabled
fixed random seed
speculative decoding disabled
MTP drafting disabled, if separately configurable
continuous batching disabled
prompt caching disabled for the first run
max_new_tokens fixed
```

Record every decode parameter in the run manifest.

If the runtime cannot disable MTP or speculative decoding, record:

- draft token IDs;
- accepted token IDs;
- rejected token IDs;
- model-forward step boundaries.

The initial locality result should preferably come from pure one-token-at-a-time decode.

---

## 7. Quantization Strategy

For the current CPU server:

1. Use the smallest practical quantization for the initial instrumentation pilot.
2. Record the exact model file, file size, and cryptographic checksum.
3. Treat Q2/Q4 results as results for that quantized runtime, not automatically as official full-precision model behavior.
4. After the pipeline works, repeat a small subset on a higher-quality quantization.
5. Compare:
   - generated-token agreement;
   - adjacent-token overlap;
   - selected-index Jaccard similarity;
   - rank stability.

Recommended progression:

```text
Q2: implementation and pipeline validation
Q4: confirmation on a smaller benchmark subset
GPU-native checkpoint: later replication
```

---

# Part III — Instrumentation Requirements

## 8. Instrumentation Architecture

Implement a backend-neutral trace API.

Suggested conceptual interface:

```c
typedef struct {
    uint64_t run_id_hash;
    uint64_t sample_id_hash;
    uint32_t decode_step;
    uint32_t absolute_position;
    uint32_t layer_id;
    uint32_t layer_type;
    uint32_t n_candidates;
    uint32_t n_visible;
    uint32_t configured_top_k;
    uint32_t valid_k;
    const uint32_t *selected_indices;
    const float *selected_scores;
} indexer_trace_event;

void trace_indexer_selection(const indexer_trace_event *event);
```

The exact language and structures may differ, but the logical fields must remain stable.

### Required implementation properties

- disabled by default;
- enabled by CLI flag or environment variable;
- does not modify tensor values;
- does not reorder selected indices;
- uses buffered writes;
- does not call `printf` for every selected entry;
- flushes safely at sample completion and process exit;
- reports I/O failure explicitly;
- stores schema version in every run;
- supports CPU and GPU adapters.

Suggested environment controls:

```text
DS4_TRACE_ENABLE=1
DS4_TRACE_LEVEL=selected
DS4_TRACE_OUTPUT=/path/to/run_dir
DS4_TRACE_FULL_SCORE_SAMPLE_RATE=0.01
DS4_TRACE_FLUSH_INTERVAL=64
```

---

## 9. Trace Levels

### Level 0 — metadata only

Stores run and sample metadata, generated tokens, and timings.

### Level 1 — selected indices

Stores all selected compressed-KV indices and selected scores.

This is the required default.

### Level 2 — top-k boundary diagnostics

Additionally stores:

- score at rank `k`;
- score at rank `k+1`;
- score margin;
- candidate-score summary statistics;
- finite candidate count.

### Level 3 — full score vectors for a calibration subset

Stores the entire candidate score vector only for a small configurable subset of:

- samples;
- decode steps;
- layers.

Full score vectors are potentially enormous and must not be enabled for all events by default.

---

# Part IV — Data to Save

## 10. Directory Layout

Use one immutable directory per run.

```text
experiment/
├── benchmark_decision.md
├── code/
│   ├── git_commit.txt
│   ├── git_status.txt
│   ├── local_changes.patch
│   └── build_command.txt
├── runs/
│   └── <run_id>/
│       ├── run_manifest.json
│       ├── machine.json
│       ├── model_config.json
│       ├── benchmark_config.json
│       ├── prompts/
│       │   ├── samples.jsonl
│       │   └── tokenized_inputs/
│       ├── outputs/
│       │   ├── generations.jsonl
│       │   └── benchmark_scores.json
│       ├── traces/
│       │   ├── decode_tokens.parquet
│       │   ├── layer_events.parquet
│       │   ├── selected_kv.parquet
│       │   ├── score_summaries.parquet
│       │   ├── full_scores/
│       │   └── trace_checksums.json
│       ├── logs/
│       │   ├── stdout.log
│       │   ├── stderr.log
│       │   ├── timing.log
│       │   └── memory.log
│       ├── analysis/
│       │   ├── metrics_token_layer.parquet
│       │   ├── metrics_sample_layer.parquet
│       │   ├── metrics_run_summary.json
│       │   ├── tables/
│       │   └── plots/
│       └── README.md
└── scripts/
    ├── run_experiment.sh
    ├── validate_trace.py
    ├── analyze_locality.py
    ├── generate_tables.py
    └── generate_plots.py
```

Do not overwrite a completed run directory. Create a new `run_id`.

---

## 11. Run Manifest Schema

`run_manifest.json` must include at least:

```text
schema_version
run_id
created_utc
hostname
user
working_directory
backend
device_type
device_count
cpu_model
cpu_socket_count
physical_core_count
logical_cpu_count
numa_node_count
total_ram_bytes
available_ram_bytes_at_start
swap_total_bytes
kernel_version
compiler
compiler_version
build_type
build_flags
git_repository
git_commit
git_dirty
patch_sha256
model_name
model_variant
model_path
model_file_size
model_sha256
quantization
weight_dtype
kv_dtype
indexer_qk_dtype
tokenizer_name
tokenizer_sha256
runtime_name
runtime_version
runtime_commit
benchmark_name
benchmark_version
benchmark_commit
benchmark_split
task_subset
sample_count
context_length_target
context_length_actual_distribution
max_new_tokens
decode_parameters
thread_count
numa_policy
environment_variables
trace_level
trace_schema_version
analysis_version
```

Also save the exact shell command used to launch the run.

---

## 12. Model Configuration Schema

`model_config.json` must record actual runtime values:

```text
num_layers
hidden_size
layer_map:
  - layer_id
  - attention_type: SWA | CSA | HCA | other
csa_compression_ratio
hca_compression_ratio
sparse_top_k
sliding_window_size
indexer_head_count
indexer_head_dim
query_head_count
attention_head_dim
query_compression_dim
kv_layout_description
```

Do not rely only on model-family defaults.

---

## 13. Sample-Level Records

`prompts/samples.jsonl` must contain one record per benchmark sample:

```text
run_id
sample_id
benchmark
benchmark_version
split
task_type
task_subtype
source_record_id
context_length_characters
context_length_tokens
prompt_length_tokens
query_length_tokens
target_max_new_tokens
prompt_sha256
tokenized_prompt_sha256
reference_answer
reference_answer_hash
evidence_positions_original_tokens
evidence_ranges_original_tokens
needle_positions
metadata
```

Store either the original prompt or a path to an immutable local copy. If licensing prevents storing full text, store:

- dataset ID;
- source record ID;
- dataset revision;
- preprocessing script commit;
- prompt hash;
- token IDs.

---

## 14. Generation-Level Records

`outputs/generations.jsonl` must include:

```text
run_id
sample_id
prompt_token_count
generated_token_count
generated_token_ids
generated_text
finish_reason
benchmark_prediction
reference_answer
is_correct
score
generation_start_utc
generation_end_utc
prefill_time_seconds
decode_time_seconds
tokens_per_second
peak_rss_bytes
error
```

Save generated token IDs even if generated text is saved.

---

## 15. Decode-Token Trace Schema

`traces/decode_tokens.parquet` must have one row per generated token:

```text
run_id
sample_id
benchmark
task_type
context_length
decode_step
decode_token_id
decode_token_text
absolute_position
generated_prefix_hash
model_forward_id
accepted
prefill_or_decode
token_start_time_ns
token_end_time_ns
token_latency_ns
rss_bytes
```

Use UTF-8-safe escaping for token text.

---

## 16. Layer-Event Trace Schema

`traces/layer_events.parquet` must have one row per decode step per layer:

```text
run_id
sample_id
benchmark
task_type
context_length
decode_step
decode_token_id
absolute_position
layer_id
layer_type
is_sparse_layer
compression_ratio
n_candidates_total
n_candidates_visible
configured_top_k
valid_selected_count
selected_score_min
selected_score_max
selected_score_mean
selected_score_std
rank_k_score
rank_k_plus_1_score
boundary_margin
indexer_time_ns
topk_time_ns
attention_time_ns
trace_write_time_ns
```

For non-CSA layers:

- `is_sparse_layer = false`;
- sparse-selection fields may be null;
- record HCA visible count or SWA window count separately.

---

## 17. Selected-KV Trace Schema

`traces/selected_kv.parquet` is the primary evidence table. It must have one row per selected compressed KV entry:

```text
run_id
sample_id
benchmark
task_type
task_subtype
context_length
decode_step
decode_token_id
decode_position
absolute_position
layer_id
layer_type
selected_rank
compressed_kv_index
index_score
is_valid
compression_ratio
original_token_start
original_token_end
query_to_entry_distance_tokens
logical_page_id
allocator_block_id
buffer_offset_bytes
previous_step_selected
first_seen_decode_step
trace_timestamp_ns
```

The following fields are required at minimum:

```text
sample_id
benchmark
task_type
context_length
decode_token_id
decode_position
layer_id
selected_rank
compressed_kv_index
index_score
```

### Mapping compressed indices to original-token ranges

Store both:

```text
compressed_kv_index
original_token_start
original_token_end
```

Do not assume a simple non-overlapping range if the compressor uses overlapping source windows. Implement the mapping from the actual runtime compression logic.

If an exact source range cannot be reconstructed, store:

```text
nominal_original_token_start
nominal_original_token_end
mapping_exact = false
```

---

## 18. Full Score Storage

For selected calibration events, save:

```text
run_id
sample_id
decode_step
layer_id
n_candidates
scores_dtype
scores_shape
scores_file
scores_sha256
```

Use a compressed binary format such as:

- NumPy `.npy` with compression in a containing archive;
- Zarr;
- Arrow IPC;
- compressed raw binary with an explicit header.

Do not save full score vectors as text.

---

## 19. Trace Integrity

For every completed run:

1. compute SHA-256 for all trace files;
2. save them in `trace_checksums.json`;
3. validate row counts;
4. verify selected ranks are contiguous;
5. verify indices are in range;
6. verify `valid_selected_count` matches selected rows;
7. verify all decode steps have expected CSA layer events;
8. verify token IDs match the generation output;
9. verify tracing on/off gives identical generated token IDs for a smoke sample.

The validator must exit non-zero on failure.

---

# Part V — Statistical Processing

## 20. Primary Metrics

Let `U[t,l]` be the unordered selected-index set for decode step `t` and layer `l`.

### 20.1 Adjacent-token overlap

```text
adjacent_overlap[t,l] = |U[t,l] ∩ U[t-1,l]| / |U[t,l]|
```

For fixed top-k, this is the fraction of current selected indices that were selected at the previous decode step.

### 20.2 Adjacent-token Jaccard similarity

```text
adjacent_jaccard[t,l] =
    |U[t,l] ∩ U[t-1,l]| /
    |U[t,l] ∪ U[t-1,l]|
```

### 20.3 Churn

```text
churn[t,l] = 1 - adjacent_overlap[t,l]
new_entries[t,l] = |U[t,l] - U[t-1,l]|
evicted_entries[t,l] = |U[t-1,l] - U[t,l]|
```

### 20.4 Rank-aware overlap

Compute at least one rank-sensitive metric.

Recommended simple metric:

```text
weighted_overlap[t,l] =
    sum over shared index s of min(w_t(s), w_t-1(s))
```

where:

```text
w(rank) = 1 / log2(rank + 2)
```

Normalize by the total weight of one top-k list.

Optionally compute Rank-Biased Overlap.

### 20.5 Score stability

For shared selected indices:

```text
score_correlation
mean_absolute_score_change
rank_correlation
```

Also report:

```text
rank_k_score
rank_k_plus_1_score
boundary_margin
```

A small boundary margin indicates unstable top-k membership.

---

## 21. Multi-Step Retention Metrics

For decode lag `delta`:

```text
retention[delta,l] =
    mean_t |U[t,l] ∩ U[t-delta,l]| / |U[t,l]|
```

Required lags:

```text
1, 2, 4, 8, 16, 32, 64
```

Only compute a lag when enough decode steps exist.

Produce one retention curve per:

- layer;
- task type;
- context length;
- correctness group.

---

## 22. Working-Set Metrics

For a decode window of width `w`:

```text
working_set[t,l,w] =
    | union of U[i,l] for i in [t-w+1, t] |
```

Required windows:

```text
1, 2, 4, 8, 16, 32, 64
```

Also compute normalized working-set growth:

```text
working_set_ratio = working_set / (w * top_k)
```

Interpretation:

- near `1/top_k` to small values: strong reuse;
- near `1`: little reuse.

Report:

```text
mean
median
p90
p99
```

---

## 23. Reuse-Distance Metrics

Build a logical access stream of:

```text
(layer_id, compressed_kv_index)
```

Use selected-rank order within each layer event unless the runtime exposes the actual memory-access order.

For every repeated logical block, compute:

- access distance: number of intervening selections;
- unique reuse distance: number of distinct logical blocks accessed since last use;
- reuse time in decode steps;
- consecutive persistence length.

Report:

```text
cold_access_fraction
p50_reuse_distance
p90_reuse_distance
p99_reuse_distance
mean_reuse_step_gap
persistence_run_length_distribution
```

Label these metrics as **logical reuse metrics**, unless actual memory-reference order is captured.

---

## 24. Access-Age Metrics

For selected compressed entry `s` at query position `t`:

```text
entry_age_tokens = t - representative_original_position(s)
```

Report:

```text
mean_age
median_age
p90_age
p99_age
fraction_recent
fraction_middle
fraction_old
```

Choose age buckets relative to context length, for example:

```text
recent: last 1% of context
middle: 1%–50%
old: oldest 50%
```

Also provide absolute-token buckets.

---

## 25. Cross-Layer Metrics

For CSA layers `l1` and `l2` at the same decode step:

```text
cross_layer_jaccard[t,l1,l2] =
    |U[t,l1] ∩ U[t,l2]| /
    |U[t,l1] ∪ U[t,l2]|
```

Important interpretation:

- this measures semantic position agreement;
- it is not direct physical cache reuse;
- layer-owned KV tensors are distinct.

Produce:

- layer-by-layer similarity matrix;
- shallow/middle/deep layer group summaries;
- same-token cross-layer heatmaps.

---

## 26. Random and Recency Baselines

### 26.1 Random expected overlap

For `k` selections from `N` visible candidates, the expected intersection of two independent random selections is approximately:

```text
E[intersection] = k^2 / N
E[overlap_fraction] = k / N
```

Compute locality lift:

```text
locality_lift =
    observed_adjacent_overlap /
    expected_random_overlap_fraction
```

Handle cases where `N <= k`.

### 26.2 Recency baseline

Construct a deterministic baseline selecting the most recent `k` compressed entries.

Compare:

- overlap;
- working set;
- access age;
- benchmark accuracy where feasible.

This separates semantic locality from locality caused solely by recency.

---

## 27. Aggregation Hierarchy

Do not treat all token-layer events as independent samples.

Aggregate in this order:

1. event-level metrics;
2. sample-layer metrics;
3. sample metrics;
4. task/context groups;
5. run summary.

Use bootstrap confidence intervals by resampling **samples**, not individual decode tokens.

Required reporting:

```text
mean
median
standard deviation
p10
p25
p75
p90
p99
95% bootstrap confidence interval
sample count
token count
```

Separate:

- correct samples;
- incorrect samples;
- truncated outputs;
- failed runs.

---

# Part VI — Required Outputs

## 28. Tables

Generate at least:

1. run configuration summary;
2. benchmark/task sample counts;
3. per-layer adjacent overlap;
4. per-layer churn;
5. retention at lags 1/2/4/8/16/32/64;
6. working-set size at windows 1/4/16/64;
7. reuse-distance percentiles;
8. access-age percentiles;
9. correct vs. incorrect comparison;
10. context-length comparison;
11. quantization comparison when available;
12. CPU vs. GPU comparison when available.

Use CSV and Markdown versions.

---

## 29. Plots

Generate separate plots for:

1. adjacent overlap by layer;
2. churn by layer;
3. retention curve by layer;
4. working-set growth by layer;
5. reuse-distance CDF;
6. access-age distribution;
7. cross-layer similarity heatmap;
8. decode-step × compressed-index access raster for selected layers;
9. decode-step × layer overlap heatmap;
10. boundary-margin distribution;
11. context-length scaling;
12. correct vs. incorrect samples;
13. benchmark task comparison.

For access rasters, avoid plotting every layer simultaneously. Select representative shallow, middle, and deep CSA layers.

---

## 30. Per-Sample Revisit Package

Every sample must be independently revisitable.

Create:

```text
analysis/sample_reports/<sample_id>/
```

with:

```text
metadata.json
prompt_reference.json
generation.txt
generation_tokens.json
benchmark_result.json
selected_kv_subset.parquet
token_layer_metrics.parquet
access_raster.png
retention_plot.png
notes.md
```

`notes.md` should summarize:

- whether the answer was correct;
- where the known evidence was located;
- which layers had highest and lowest locality;
- any abrupt phase changes;
- any suspicious trace anomalies.

---

# Part VII — CPU Execution Requirements

## 31. CPU-Specific Controls

Record:

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
thread affinity
NUMA policy
numactl command
CPU frequency governor
transparent huge-page setting
memory interleaving policy
```

Recommended initial policy on a multi-socket server:

```text
numactl --interleave=all
```

But benchmark both interleaved and local allocation only if CPU systems performance is part of the research question.

Monitor:

```text
RSS
major page faults
minor page faults
swap in/out
CPU utilization
NUMA misses if available
```

Avoid swapping. Abort or flag a run if sustained swap activity occurs.

---

## 32. Trace-Overhead Measurement

For one smoke sample, run:

1. tracing disabled;
2. metadata-only tracing;
3. selected-index tracing;
4. full-score calibration tracing.

Report:

```text
prefill time
decode time
tokens/s
peak RSS
trace bytes
trace overhead percentage
```

Instrumentation should not be inside the measured top-k interval unless explicitly measuring trace overhead.

---

# Part VIII — GPU-Reusable Design

## 33. Backend-Neutral Logical Schema

The following fields must remain identical across CPU and GPU:

```text
run_id
sample_id
decode_step
decode_token_id
absolute_position
layer_id
layer_type
n_candidates_visible
configured_top_k
valid_selected_count
selected_rank
compressed_kv_index
index_score
original_token_range
```

Backend-specific fields must be additive, not replacements.

---

## 34. GPU Trace Adapter

On GPU:

1. instrument after the device top-k kernel;
2. copy only selected indices and selected scores by default;
3. use asynchronous device-to-host copies;
4. use a dedicated stream if safe;
5. batch readback across layers or steps;
6. avoid full device synchronization per layer;
7. store GPU event timing separately;
8. record kernel names and launch configuration.

Add GPU fields:

```text
gpu_model
gpu_uuid
driver_version
runtime_version
cuda_or_rocm_version
kernel_name
kernel_variant
stream_id
device_buffer_offset
device_page_id
copy_start_ns
copy_end_ns
kernel_start_ns
kernel_end_ns
```

If async tracing changes results or scheduling, provide a synchronous validation mode for a small subset.

---

## 35. CPU–GPU Comparability

To compare CPU and GPU traces:

1. use the same prompt token IDs;
2. use greedy decoding;
3. use the same model checkpoint and quantization if possible;
4. compare generated tokens step by step;
5. compare selected sets only while generated prefixes remain identical;
6. report divergence step;
7. report selection Jaccard before divergence;
8. separate numerical-difference effects from backend effects.

Never compare token `t` traces after the generated prefixes have diverged without explicitly labeling the comparison as non-aligned.

---

# Part IX — Validation and Acceptance Criteria

## 36. Correctness Tests

### Unit tests

Create synthetic candidate-score arrays with known top-k outputs.

Test:

- unique scores;
- ties;
- all negative scores;
- `N < k`;
- invalid or masked candidates;
- deterministic tie-breaking;
- large index values;
- multiple tokens;
- multiple layers.

### Integration tests

Verify:

- tracing does not change token IDs;
- every CSA layer emits one event per decode step;
- top-k indices match a reference CPU top-k for a small event;
- selected scores match source scores;
- visible candidate count is correct;
- original-token mapping is valid.

---

## 37. Minimum Acceptance Criteria

The initial implementation is complete only when all of the following are true:

- [ ] one public benchmark is selected and documented;
- [ ] one sample runs end to end;
- [ ] token-by-token traces are saved;
- [ ] layer-by-layer CSA traces are saved;
- [ ] selected indices and scores are present;
- [ ] run manifest is complete;
- [ ] model and dataset revisions are recorded;
- [ ] trace checksums are generated;
- [ ] tracing on/off produces identical generated token IDs;
- [ ] adjacent overlap is computed;
- [ ] multi-lag retention is computed;
- [ ] working-set metrics are computed;
- [ ] reuse-distance metrics are computed;
- [ ] per-layer plots are generated;
- [ ] per-sample revisit package is generated;
- [ ] trace overhead is measured;
- [ ] code patch and exact build/run commands are saved;
- [ ] GPU adapter points are documented.

---

# Part X — Recommended Initial Execution Plan

## 38. Concrete First Run

### Benchmark

```text
RULER
```

### Task subset

Choose one task from each group:

```text
retrieval
multi-hop
aggregation
```

### Context lengths

```text
32K
64K
```

### Sample counts

```text
Smoke:
  1 sample per task at 32K

Pilot:
  10 samples per task per context length
```

### Decode length

```text
128 generated tokens
```

If the benchmark normally expects a short answer, preserve:

1. an official-answer run for correctness;
2. a trace run requesting a concise explanation up to 128 tokens.

### Model progression

```text
Q2 for smoke and pilot
Q4 for a smaller confirmation subset
GPU-native model later
```

### Analysis priority

1. adjacent-token overlap;
2. retention at 1/2/4/8/16/32/64;
3. working-set growth;
4. access raster;
5. cross-layer differences;
6. random-overlap lift;
7. correct vs. incorrect comparison.

---

## 39. Expansion Plan

After the RULER pilot succeeds:

1. add LongGenBench for long consecutive generation;
2. add MRCR for a lower-locality stress case;
3. add LongMemEval for realistic historical memory;
4. add LongBench-v2 for broad realism;
5. add no-context and local-only controls;
6. repeat a representative subset on GPU.

---

# Part XI — Final Deliverable Structure

The agent must leave the repository in a state where another researcher can run:

```text
1. environment check
2. model build
3. benchmark preparation
4. one command to execute a run
5. one command to validate traces
6. one command to analyze a run
7. one command to generate tables and plots
```

Recommended top-level instructions:

```bash
./scripts/check_environment.sh
./scripts/build_cpu.sh
./scripts/prepare_benchmark.sh --benchmark ruler
./scripts/run_experiment.sh --config configs/ruler_32k.yaml
python scripts/validate_trace.py runs/<run_id>
python scripts/analyze_locality.py runs/<run_id>
python scripts/generate_tables.py runs/<run_id>
python scripts/generate_plots.py runs/<run_id>
```

The exact commands may differ, but the workflow must be equally explicit and reproducible.

---

# Part XII — Reporting Rules

## 40. Claims That Are Allowed

Examples:

- “On RULER retrieval tasks at 64K context, layer 20 reused a median of X% of selected compressed KV entries between adjacent decode tokens.”
- “The 64-token working set was Y times the single-token top-k.”
- “MRCR exhibited lower temporal locality than RULER under the same context length and decode configuration.”
- “The observed overlap was Z times larger than the random-selection expectation.”

## 41. Claims That Require Caution

Do not automatically claim:

- that CPU results equal GPU production behavior;
- that Q2 results equal official full-precision behavior;
- that cross-layer overlap is physical cache reuse;
- that high logical overlap guarantees CPU or GPU cache hits;
- that one benchmark represents all long-context workloads;
- that a short-answer benchmark provides enough temporal samples;
- that trace-enabled latency equals uninstrumented latency.

---

# 42. Final Checklist for the Agent

Before starting:

- [ ] inspect the runtime;
- [ ] inspect model configuration;
- [ ] select one benchmark;
- [ ] write `benchmark_decision.md`;
- [ ] define a run schema version.

Before full execution:

- [ ] implement backend-neutral trace callback;
- [ ] add CPU writer;
- [ ] add validation tests;
- [ ] verify deterministic output;
- [ ] measure trace volume and overhead.

After execution:

- [ ] checksum raw traces;
- [ ] validate records;
- [ ] compute required metrics;
- [ ] generate tables and plots;
- [ ] create per-sample reports;
- [ ] preserve code commit and patch;
- [ ] document GPU instrumentation points;
- [ ] write a concise experiment summary with limitations.

---

## References for the Agent

The experiment design is based on the following architectural and evaluation concepts:

- DeepSeek-V4 hybrid attention with CSA, HCA, sliding-window attention, compressed indexer keys, per-token top-k sparse selection, and heterogeneous KV-cache management.
- FlashMemory-DeepSeek-V4 evaluation using LongBench-v2, LongMemEval, and RULER, plus MRCR as a dense global-memory failure case.
- Standard cache-locality concepts: overlap, churn, retention, working-set size, reuse distance, and access-age distribution.

The raw trace is the primary evidence. Summary statistics and plots must always be reproducible from the immutable raw trace plus the saved analysis code.
