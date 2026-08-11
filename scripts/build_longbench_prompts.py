#!/usr/bin/env python3
"""Build ds4 prompts + a sample manifest from LongBench summarization tasks.

Usage: build_longbench_prompts.py
Reads  benchmark/longbench/{gov_report,qmsum,multi_news}.jsonl  (from zai-org/LongBench data.zip)
Writes prompts/longbench/<task>_s<k>.txt and prompts/longbench_samples.jsonl
       (schema-compatible with finalize_run.py: sample_id, reference_answer, target_max_new_tokens, ...)

Selection: filter to prompts <= MAX_PROMPT_TOKENS (bounds per-run wall-clock), then a fixed-seed
random sample of N_PER_TASK per task (reproducible, unbiased). Templates are the official ones from
THUDM/LongBench config/dataset2prompt.json (fetched 2026-07-24, hardcoded for reproducibility);
max_gen=512 is the official cap for all three tasks (config/dataset2maxlen.json).
"""
import hashlib
import json
import os
import random

from tokenizers import Tokenizer

EXP = "<WORKDIR>/experiment"
TASKS = ["multi_news", "gov_report", "qmsum"]  # shortest-first ordering for the runner
N_PER_TASK = 12
MAX_PROMPT_TOKENS = 20000
MAX_GEN = 512
SEED = 42

# official LongBench prompt templates (THUDM/LongBench config/dataset2prompt.json)
TEMPLATES = {
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
                  "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. "
             "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n"
             "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
             "Query: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\n"
                  "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
}

tok = Tokenizer.from_file(os.path.join(EXP, "tokenizer", "tokenizer.json"))
os.makedirs(os.path.join(EXP, "prompts", "longbench"), exist_ok=True)

manifest = []
for task in TASKS:
    recs = [json.loads(l) for l in open(os.path.join(EXP, "benchmark", "longbench", f"{task}.jsonl"))]
    cands = []
    for i, r in enumerate(recs):
        prompt = TEMPLATES[task].format(context=r["context"], input=r.get("input", ""))
        n_tok = len(tok.encode(prompt).ids)
        cands.append((i, n_tok, prompt, r))
    fit = [c for c in cands if c[1] <= MAX_PROMPT_TOKENS]
    excluded = len(cands) - len(fit)
    rng = random.Random(SEED)
    chosen = rng.sample(fit, min(N_PER_TASK, len(fit)))
    chosen.sort(key=lambda c: c[1])  # shortest-first within the task
    print(f"{task}: {len(cands)} samples, {excluded} excluded (> {MAX_PROMPT_TOKENS} tok), "
          f"picked {len(chosen)} (seed {SEED}); tok range {chosen[0][1]}-{chosen[-1][1]}")
    for k, (i, n_tok, prompt, r) in enumerate(chosen):
        sid = f"lb_{task}_s{k}"
        pf = os.path.join(EXP, "prompts", "longbench", f"{task}_s{k}.txt")
        with open(pf, "w") as f:
            f.write(prompt)
        manifest.append({
            "sample_id": sid,
            "benchmark": "LongBench",
            "benchmark_version": "zai-org/LongBench data.zip (v1)",
            "split": "test",
            "task_type": "summarization",
            "task_subtype": task,
            "source_record_id": r["_id"],
            "source_record_index": i,
            "context_length_target": n_tok,
            "context_length_characters": len(prompt),
            "reported_length_tokens": n_tok,
            "prompt_file": pf,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "reference_answer": r["answers"],
            "answer_prefix": "",
            "target_max_new_tokens": MAX_GEN,
        })

out = os.path.join(EXP, "prompts", "longbench_samples.jsonl")
with open(out, "w") as f:
    for m in manifest:
        f.write(json.dumps(m) + "\n")
print(f"wrote {len(manifest)} prompts + manifest {out}")
