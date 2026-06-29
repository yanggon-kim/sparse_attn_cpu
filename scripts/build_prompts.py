#!/usr/bin/env python3
"""Assemble ds4 prompt files + sample metadata from RULER niah validation.jsonl.

For each context length, the full model prompt is RULER's `input + answer_prefix`.
Writes:
  prompts/niah_single_2_<L>.txt      (the prompt fed to ds4)
  prompts/samples.jsonl              (one metadata record per sample, doc §13)
"""
import json, hashlib, os, sys

BASE = "<WORKDIR>/experiment"
LENGTHS = [4096, 8192, 16384]
TASK = "niah_single_2"
OUT_PROMPTS = os.path.join(BASE, "prompts")
os.makedirs(OUT_PROMPTS, exist_ok=True)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def raw_path(L):
    sub = "raw" if L == 4096 else f"raw_{L}"
    return os.path.join(BASE, "data", sub, TASK, "validation.jsonl")


def main():
    samples = []
    for L in LENGTHS:
        rec = json.loads(open(raw_path(L)).read().splitlines()[0])
        prompt = rec["input"] + rec["answer_prefix"]
        sample_id = f"{TASK}_{L}_s0"
        pf = os.path.join(OUT_PROMPTS, f"{sample_id}.txt")
        with open(pf, "w") as f:
            f.write(prompt)
        samples.append({
            "sample_id": sample_id,
            "benchmark": "RULER",
            "benchmark_version": open(os.path.join(BASE, "benchmark", "RULER_commit.txt")).read().strip(),
            "split": "validation",
            "task_type": "retrieval",
            "task_subtype": TASK,
            "source_record_id": rec.get("index"),
            "context_length_target": L,
            "context_length_characters": len(prompt),
            "reported_length_tokens": rec["length"],
            "prompt_file": pf,
            "prompt_sha256": sha(prompt),
            "reference_answer": rec["outputs"],
            "reference_answer_hash": sha(json.dumps(rec["outputs"])),
            "answer_prefix": rec["answer_prefix"],
            "needle_value": rec["outputs"][0] if rec["outputs"] else None,
            "token_position_answer_hf": rec.get("token_position_answer"),
            "target_max_new_tokens": 128,
        })
    with open(os.path.join(OUT_PROMPTS, "samples.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(samples)} prompts to {OUT_PROMPTS}")
    for s in samples:
        print(f"  {s['sample_id']}: chars={s['context_length_characters']} "
              f"reported_tok={s['reported_length_tokens']} needle={s['needle_value']}")


if __name__ == "__main__":
    main()
