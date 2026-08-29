#!/usr/bin/env python3
"""Build the exp3 accuracy benchmark set (exp3 §4) as a manifest compatible with run_vllm_batch.py.
Usage: build_exp3_prompts.py --out <WORKDIR>/prompts/exp3 [--ruler <WORKDIR>/prompts/ruler_exp3]
         [--data <WORKDIR>/benchmark/data] [--n-ruler 100] [--n-mmlu 1000] [--lb2-all]
Items ("rung" = the max_model_len bucket the runner should use; "kind" = bf):
  ruler:  13 tasks x {32K,64K,128K} x n-ruler (official prepare.py output, base template), max_new per task
  longbench_v2: all 503 (official zero-shot template, max_new 128), rung by prompt length (<=32K/64K/128K)
  infinitebench: longbook_choice_eng (En.MC, all 229), longbook_qa_eng (En.QA, 351 -> first 100), passkey (100),
                 number_string (100); middle-truncated to 128K when longer (official practice)
  mmlu_pro: stratified subset (n-mmlu over 14 categories, seed 42), official 0-shot CoT-free prompt
            "Answer: X" format; gpqa_diamond: all 198 if the dataset is present (gated), official MC prompt
Thinking mode per user decision: mmlu_pro / gpqa -> "thinking" (max_new 8192); long-context -> "chat".
"""
import argparse, csv, glob, json, os, random

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--ruler", default=os.path.join(os.environ.get("WORKDIR", ""), "prompts", "ruler_exp3"))
ap.add_argument("--data", default=os.path.join(os.environ.get("WORKDIR", ""), "benchmark", "data"))
ap.add_argument("--tokenizer", default=os.path.join(os.environ.get("WORKDIR", ""), "models", "DeepSeek-V3.2"))
ap.add_argument("--n-ruler", type=int, default=100)
ap.add_argument("--n-mmlu", type=int, default=1000)
ap.add_argument("--n-ib-qa", type=int, default=100)
a = ap.parse_args()
random.seed(42)
tok = AutoTokenizer.from_pretrained(a.tokenizer)
os.makedirs(a.out, exist_ok=True)
samples = []
RUNGS = [32768, 65536, 131072]


def ntok(t):
    return len(tok(t, add_special_tokens=False)["input_ids"])


def rung_for(n):
    for r in RUNGS:
        if n + 2304 <= r:
            return r
    return None


def truncate_middle(text, target):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= target:
        return text, False
    h = target // 2
    return tok.decode(ids[:h]) + "\n...\n" + tok.decode(ids[-h:]), True


def add(sid, source, task, rung, text, max_new, ref, meta=None, thinking="chat"):
    path = os.path.join(a.out, sid + ".txt")
    open(path, "w").write(text)
    samples.append({"sample_id": sid, "source": source, "task": task, "rung": rung, "kind": "bf", "max_new_tokens": max_new,
                    "prompt_tokens_v32": ntok(text), "reference": ref, "meta": meta or {}, "text_path": os.path.basename(path),
                    "thinking": thinking})


# ---- RULER 13 tasks ----
RULER_MAXNEW = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}
for L in RUNGS:
    for tdir in sorted(glob.glob(os.path.join(a.ruler, str(L), "*"))):
        task = os.path.basename(tdir)
        p = os.path.join(tdir, "validation.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()][: a.n_ruler]
        mx = next(v for k, v in RULER_MAXNEW.items() if task.startswith(k))
        for i, r in enumerate(rows):
            add(f"ruler_{task}_{L}_i{i:03d}", "ruler", task, L, r["input"], mx, r["outputs"], {"index": r.get("index", i)})
    print(f"[ruler] {L}: {sum(1 for s in samples if s['source']=='ruler' and s['rung']==L)} items")

# ---- LongBench v2 (all 503) ----
LB2 = ("Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\n"
       "What is the correct answer to this question: {question}\nChoices:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n\n"
       "Format your response as follows: \"The correct answer is (insert answer here)\".")
lb2 = json.load(open(os.path.join(a.data, "LongBench-v2", "data.json")))
n_tr = 0
for r in lb2:
    ctx, trunc = truncate_middle(r["context"], 131072 - 2304 - 600)
    text = LB2.format(context=ctx, question=r["question"], A=r["choice_A"], B=r["choice_B"], C=r["choice_C"], D=r["choice_D"])
    rung = rung_for(ntok(text)) or 131072
    n_tr += trunc
    add(f"lb2_{r['_id']}", "longbench_v2", r["domain"], rung, text, 128, [r["answer"]],
        {"_id": r["_id"], "difficulty": r["difficulty"], "length": r["length"], "truncated": trunc})
print(f"[lb2] 503 items, {n_tr} truncated to 128K")

# ---- InfiniteBench ----
IB = {
    "passkey": ("There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is", 6),
    "number_string": ("There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is", 12),
    "longbook_choice_eng": ("Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is", 40),
    "longbook_qa_eng": ("Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:", 40),
}
for task, (tpl, mx) in IB.items():
    rows = [json.loads(l) for l in open(os.path.join(a.data, "InfiniteBench", task + ".jsonl")) if l.strip()]
    if task in ("passkey", "number_string", "longbook_qa_eng"):
        rows = rows[: a.n_ib_qa]
    for r in rows:
        ctx, trunc = truncate_middle(r["context"], 131072 - 2304 - 400)
        if task == "longbook_choice_eng":
            o = r["options"]
            text = tpl.format(context=ctx, question=r["input"], OPTION_A=o[0], OPTION_B=o[1], OPTION_C=o[2], OPTION_D=o[3])
            ref = [o.index(r["answer"][0]) if r["answer"][0] in o else -1, r["answer"][0]]
        elif task == "longbook_qa_eng":
            text = tpl.format(context=ctx, question=r["input"]); ref = r["answer"]
        else:
            text = tpl.format(context=ctx, input=r["input"]); ref = r["answer"] if isinstance(r["answer"], list) else [r["answer"]]
        rung = rung_for(ntok(text)) or 131072
        add(f"ib_{task}_i{r['id']:03d}", "infinitebench", task, rung, text, mx, ref, {"id": r["id"], "truncated": trunc})
    print(f"[ib] {task}: {len(rows)}")

# ---- MMLU-Pro stratified subset ----
import pandas as pd
mp = pd.read_parquet(glob.glob(os.path.join(a.data, "MMLU-Pro", "data", "test-*.parquet"))[0])
cats = sorted(mp.category.unique())
per = a.n_mmlu // len(cats)
chosen = []
for c in cats:
    sub = mp[mp.category == c].sample(frac=1.0, random_state=42)
    chosen.append(sub.head(per))
mpc = pd.concat(chosen)
extra = mp[~mp.question_id.isin(mpc.question_id)].sample(frac=1.0, random_state=42).head(a.n_mmlu - len(mpc))
mpc = pd.concat([mpc, extra])
LET = "ABCDEFGHIJ"
for _, r in mpc.iterrows():
    opts = "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(r.options))
    text = (f"The following is a multiple choice question about {r.category}. Think step by step and then finish your "
            f"answer with \"The answer is (X)\" where X is the correct letter choice.\n\nQuestion: {r.question}\nOptions:\n{opts}")
    add(f"mmlu_{int(r.question_id):05d}", "mmlu_pro", r.category, 32768, text, 8192, [LET[int(r.answer_index)]],
        {"question_id": int(r.question_id)}, thinking="thinking")
print(f"[mmlu_pro] {len(mpc)} items over {len(cats)} categories")

# ---- GPQA Diamond (gated; optional) ----
gp = glob.glob(os.path.join(a.data, "gpqa", "**", "gpqa_diamond.csv"), recursive=True)
if gp:
    rows = list(csv.DictReader(open(gp[0])))
    for i, r in enumerate(rows):
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        rnd = random.Random(42 + i)
        order = list(range(4)); rnd.shuffle(order)
        letters = "ABCD"
        shown = "\n".join(f"({letters[k]}) {opts[order[k]]}" for k in range(4))
        ans = letters[order.index(0)]
        text = (f"What is the correct answer to this question: {r['Question']}\n\nChoices:\n{shown}\n\n"
                f"Think step by step, then format your final answer as \"The answer is (X)\".")
        add(f"gpqa_{i:03d}", "gpqa", "diamond", 32768, text, 8192, [ans], {"record_id": r.get("Record ID")}, thinking="thinking")
    print(f"[gpqa] {len(rows)} items")
else:
    print("[gpqa] dataset not present (gated) -> skipped")

with open(os.path.join(a.out, "manifest.jsonl"), "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")
from collections import Counter
print("total", len(samples), Counter((s["source"], s["rung"]) for s in samples))
