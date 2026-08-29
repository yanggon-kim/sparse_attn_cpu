#!/usr/bin/env python3
"""Build the exp1/exp2 prompt ladder (exp1 §5): per rung {8K,16K,32K,64K,128K} ~20 benchmark-faithful (bf)
prompts balanced over sources + 8 long-decode (ld) prompts, tokenized with the DeepSeek-V3.2 tokenizer.

Usage: build_gpu_prompts.py --out <WORKDIR>/prompts/ladder [--rungs 8192 16384 32768 65536 131072]
       [--ruler <WORKDIR>/prompts/ruler] [--data <WORKDIR>/benchmark/data] [--n-bf-per-source 4] [--n-ld 8]
Writes <out>/manifest.jsonl (one line per sample) and <out>/<sample_id>.txt (the raw *user message* text;
the runner applies the model's chat template). Manifest fields:
  sample_id, source, task, rung, kind (bf|ld), max_new_tokens, prompt_tokens_v32 (of the raw text, no template),
  reference (list of acceptable answers or the reference summary), meta (task-specific), text_path
Rung membership: a prompt belongs to rung R if R/2 < tokens_with_template <= R - 2304 (room for decode).
Sources per rung: RULER niah_single_2, RULER qa_1, LongBench v1 (multi_news/gov_report/qmsum, mixed),
LongBench v2 (MC), InfiniteBench (passkey, number_string, En.MC, En.QA). Long contexts are middle-truncated to
fit the rung (InfiniteBench / LongBench-v2 only; recorded in meta.truncated). Deterministic (seed 42).
"""
import argparse, json, os, random, zipfile

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--rungs", nargs="+", type=int, default=[8192, 16384, 32768, 65536, 131072])
ap.add_argument("--ruler", default=os.path.join(os.environ.get("WORKDIR", ""), "prompts", "ruler"))
ap.add_argument("--data", default=os.path.join(os.environ.get("WORKDIR", ""), "benchmark", "data"))
ap.add_argument("--tokenizer", default=os.path.join(os.environ.get("WORKDIR", ""), "models", "DeepSeek-V3.2"))
ap.add_argument("--n-bf-per-source", type=int, default=4)
ap.add_argument("--n-ld", type=int, default=8)
ap.add_argument("--headroom", type=int, default=2304)
a = ap.parse_args()
random.seed(42)
tok = AutoTokenizer.from_pretrained(a.tokenizer)
os.makedirs(a.out, exist_ok=True)
TEMPLATE_OVERHEAD = 16  # tokens added by the chat template (bos + role markers); measured, conservative


def ntok(text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def fits(n, rung):
    return rung // 2 < n + TEMPLATE_OVERHEAD <= rung - a.headroom


def truncate_middle(text, target_tokens):
    """Keep head and tail so that the token count is ~target_tokens (InfiniteBench-style)."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= target_tokens:
        return text, False
    half = target_tokens // 2
    return tok.decode(ids[:half]) + "\n...\n" + tok.decode(ids[-half:]), True


samples = []


def add(sample_id, source, task, rung, kind, text, max_new, reference, meta=None):
    n = ntok(text)
    path = os.path.join(a.out, sample_id + ".txt")
    open(path, "w").write(text)
    samples.append({"sample_id": sample_id, "source": source, "task": task, "rung": rung, "kind": kind,
                    "max_new_tokens": max_new, "prompt_tokens_v32": n, "reference": reference,
                    "meta": meta or {}, "text_path": os.path.basename(path)})


# ---------------- RULER (already sized per rung by prepare.py) ----------------
RULER_MAXNEW = {"niah_single_2": 128, "qa_1": 32}
for rung in a.rungs:
    for task in ("niah_single_2", "qa_1"):
        p = os.path.join(a.ruler, str(rung), task, "validation.jsonl")
        if not os.path.exists(p):
            print(f"[warn] missing {p}")
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        for i, r in enumerate(rows[: a.n_bf_per_source]):
            add(f"ruler_{task}_{rung}_s{i:02d}", "ruler", task, rung, "bf", r["input"], RULER_MAXNEW[task],
                r["outputs"], {"index": r.get("index", i), "length": r.get("length")})

# ---------------- LongBench v1 summarization (official prompts, max_gen 512) ----------------
LB1_PROMPT = {
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
}
lb1 = {}
with zipfile.ZipFile(os.path.join(a.data, "LongBench", "data.zip")) as z:
    for task in LB1_PROMPT:
        rows = [json.loads(l) for l in z.read(f"data/{task}.jsonl").decode().splitlines() if l.strip()]
        for r in rows:
            r["_text"] = LB1_PROMPT[task].format(context=r["context"], input=r.get("input", ""))
            r["_n"] = ntok(r["_text"])
        lb1[task] = rows
for rung in a.rungs:
    pool = []
    for task, rows in lb1.items():
        pool += [(task, r) for r in rows if fits(r["_n"], rung)]
    random.shuffle(pool)
    bf = pool[: a.n_bf_per_source]
    ld = pool[a.n_bf_per_source: a.n_bf_per_source + max(0, a.n_ld - 2)]
    for i, (task, r) in enumerate(bf):
        add(f"lb1_{task}_{rung}_s{i:02d}", "longbench_v1", task, rung, "bf", r["_text"], 512, r["answers"],
            {"_id": r.get("_id"), "length": r.get("length")})
    for i, (task, r) in enumerate(ld):
        add(f"lb1_{task}_{rung}_ld{i:02d}", "longbench_v1", task, rung, "ld", r["_text"], 2048, r["answers"],
            {"_id": r.get("_id"), "length": r.get("length")})
    print(f"[lb1] rung {rung}: pool {len(pool)} -> bf {len(bf)} ld {len(ld)}")

# ---------------- LongBench v2 (MC; official zero-shot template, max_gen 128) ----------------
LB2_PROMPT = ("Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\n"
              "What is the correct answer to this question: {question}\nChoices:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n\n"
              "Format your response as follows: \"The correct answer is (insert answer here)\".")
lb2 = json.load(open(os.path.join(a.data, "LongBench-v2", "data.json")))
random.shuffle(lb2)
for rung in a.rungs:
    chosen, tried = [], 0
    for r in lb2:
        if len(chosen) >= a.n_bf_per_source:
            break
        tried += 1
        ctx_target = rung - a.headroom - TEMPLATE_OVERHEAD - 400
        ctx, trunc = truncate_middle(r["context"], ctx_target)
        text = LB2_PROMPT.format(context=ctx, question=r["question"], A=r["choice_A"], B=r["choice_B"],
                                 C=r["choice_C"], D=r["choice_D"])
        n = ntok(text)
        if not fits(n, rung):
            continue
        if trunc and r["length"] == "short" and rung >= 65536:
            continue
        chosen.append((r, text, trunc))
    for i, (r, text, trunc) in enumerate(chosen):
        add(f"lb2_{r['domain'].split()[0].lower()}_{rung}_s{i:02d}", "longbench_v2", r["domain"], rung, "bf", text, 128,
            [r["answer"]], {"_id": r["_id"], "difficulty": r["difficulty"], "length": r["length"], "truncated": trunc})
    print(f"[lb2] rung {rung}: {len(chosen)} chosen (tried {tried})")

# ---------------- InfiniteBench (official prompts + max_gen) ----------------
IB = {
    "passkey": ("There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is", 6),
    "number_string": ("There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is", 12),
    "longbook_choice_eng": ("Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is", 40),
    "longbook_qa_eng": ("Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:", 40),
}
ib = {}
for task in IB:
    rows = [json.loads(l) for l in open(os.path.join(a.data, "InfiniteBench", task + ".jsonl")) if l.strip()]
    random.shuffle(rows)
    ib[task] = rows


def ib_text(task, r, ctx):
    tpl = IB[task][0]
    if task == "longbook_choice_eng":
        o = r["options"]
        return tpl.format(context=ctx, question=r["input"], OPTION_A=o[0], OPTION_B=o[1], OPTION_C=o[2], OPTION_D=o[3])
    if task == "longbook_qa_eng":
        return tpl.format(context=ctx, question=r["input"])
    return tpl.format(context=ctx, input=r["input"])


def ib_ref(task, r):
    if task == "longbook_choice_eng":
        return [r["options"].index(r["answer"][0]) if r["answer"][0] in r["options"] else -1, r["answer"][0]]
    return r["answer"] if isinstance(r["answer"], list) else [r["answer"]]


for rung in a.rungs:
    n_each = max(1, a.n_bf_per_source // len(IB))
    for task in IB:
        rows = ib[task]
        chosen = []
        for r in rows:
            if len(chosen) >= n_each:
                break
            ctx_target = rung - a.headroom - TEMPLATE_OVERHEAD - 300
            if task in ("passkey", "number_string"):
                # needle tasks: truncating the middle may delete the needle -> keep the needle-bearing half
                ids = tok(r["context"], add_special_tokens=False)["input_ids"]
                if len(ids) > ctx_target:
                    needle = str(r["answer"][0] if isinstance(r["answer"], list) else r["answer"])
                    head = tok.decode(ids[:ctx_target])
                    tail = tok.decode(ids[-ctx_target:])
                    ctx = head if needle in head else (tail if needle in tail else None)
                    if ctx is None:
                        continue
                    trunc = True
                else:
                    ctx, trunc = r["context"], False
            else:
                ctx, trunc = truncate_middle(r["context"], ctx_target)
            text = ib_text(task, r, ctx)
            if not fits(ntok(text), rung):
                continue
            chosen.append((r, text, trunc))
        for i, (r, text, trunc) in enumerate(chosen):
            add(f"ib_{task}_{rung}_s{i:02d}", "infinitebench", task, rung, "bf", text, IB[task][1], ib_ref(task, r),
                {"id": r["id"], "truncated": trunc})
        print(f"[ib] rung {rung} {task}: {len(chosen)}")
    # ld fill: En.QA (2) always, then En.Sum (longbook_sum_eng, official prompt) until n_ld per rung is reached
    n_ld_have = sum(1 for s in samples if s["rung"] == rung and s["kind"] == "ld")
    want = max(2, a.n_ld - n_ld_have)
    k = 0
    for r in ib["longbook_qa_eng"][a.n_bf_per_source: a.n_bf_per_source + 40]:
        if k >= min(2, want):
            break
        ctx, trunc = truncate_middle(r["context"], rung - a.headroom - TEMPLATE_OVERHEAD - 300)
        text = ib_text("longbook_qa_eng", r, ctx)
        if fits(ntok(text), rung):
            add(f"ib_longbook_qa_eng_{rung}_ld{k:02d}", "infinitebench", "longbook_qa_eng", rung, "ld", text, 2048,
                ib_ref("longbook_qa_eng", r), {"id": r["id"], "truncated": trunc})
            k += 1
    want -= k
    k = 0
    if want > 0:
        SUM_TPL = "Summarize the following book.\n\n{context}\n\nSummary:"
        if "longbook_sum_eng" not in ib:
            rows = [json.loads(l) for l in open(os.path.join(a.data, "InfiniteBench", "longbook_sum_eng.jsonl")) if l.strip()]
            random.shuffle(rows)
            ib["longbook_sum_eng"] = rows
        for r in ib["longbook_sum_eng"]:
            if k >= want:
                break
            ctx, trunc = truncate_middle(r["context"], rung - a.headroom - TEMPLATE_OVERHEAD - 100)
            text = SUM_TPL.format(context=ctx)
            if fits(ntok(text), rung):
                add(f"ib_longbook_sum_eng_{rung}_ld{k:02d}", "infinitebench", "longbook_sum_eng", rung, "ld", text, 2048,
                    r["answer"] if isinstance(r["answer"], list) else [r["answer"]], {"id": r["id"], "truncated": trunc})
                k += 1

with open(os.path.join(a.out, "manifest.jsonl"), "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")
from collections import Counter
print("total", len(samples), Counter((s["rung"], s["kind"]) for s in samples))
print(Counter((s["rung"], s["source"]) for s in samples if s["kind"] == "bf"))
