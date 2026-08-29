"""Benchmark scorers for the GPU campaign (ported from the official scripts, kept minimal & deterministic).
score(source, task, prediction, reference, meta) -> (is_correct: bool|None, score: float, extracted: str)
- ruler: niah_* = all needle values present (string_match_all); qa_* = any answer substring (string_match_part).
- longbench_v1 (multi_news/gov_report/qmsum): ROUGE-L F1 vs the best reference (official metric), no correctness.
- longbench_v2: letter after "The correct answer is" (official regex), exact match.
- infinitebench: passkey/number_string = answer digits in prediction; longbook_choice_eng = letter/option match;
  longbook_qa_eng = max token-F1 over answers (official qa_f1_score), correct if F1 >= 0.5? -> score only.
- gpqa / mmlu_pro: letter extraction from "Answer: X" / "the answer is (X)" / last standalone letter.
"""
import re
import string
from collections import Counter


def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred, ref):
    p, r = _norm(pred).split(), _norm(ref).split()
    common = Counter(p) & Counter(r)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(r)
    return 2 * prec * rec / (prec + rec)


def _rouge_l(pred, ref):
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return sc.score(ref, pred)["rougeL"].fmeasure
    except Exception:
        return _f1(pred, ref)


LETTER_RES = [
    re.compile(r"The correct answer is\s*\(?([A-D])\)?", re.I),
    re.compile(r"answer is\s*:?\s*\(?([A-J])\)?", re.I),
    re.compile(r"Answer\s*:\s*\(?([A-J])\)?", re.I),
    re.compile(r"\\boxed\{\(?([A-J])\)?\}"),
]


def extract_letter(text, letters="ABCD"):
    for rx in LETTER_RES:
        m = rx.findall(text)
        if m:
            c = m[-1].upper()
            if c in letters:
                return c
    m = re.findall(r"\(([A-J])\)", text)
    if m and m[-1] in letters:
        return m[-1]
    m = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", text.strip()[:12])
    if m and m[0] in letters:
        return m[0]
    return ""


def score(source, task, prediction, reference, meta=None):
    pred = prediction or ""
    refs = reference if isinstance(reference, list) else [reference]
    if source == "ruler":
        refs_s = [str(r) for r in refs]
        if task.startswith("qa"):
            # RULER string_match_part: any reference answer appears in the prediction
            hit = any(r.lower() in pred.lower() for r in refs_s)
            return hit, float(hit), pred.strip()[:64]
        # RULER string_match_all (niah_*, vt, cwe, fwe): fraction of reference items present; correct iff all
        hits = sum(1 for r in refs_s if r in pred)
        return hits == len(refs_s), hits / max(1, len(refs_s)), pred.strip()[:64]
    if source == "longbench_v1":
        best = max(_rouge_l(pred, str(r)) for r in refs) if refs else 0.0
        return None, best, pred.strip()[:64]
    if source == "longbench_v2":
        c = extract_letter(pred)
        return c == refs[0], float(c == refs[0]), c
    if source == "infinitebench":
        if task in ("passkey", "number_string"):
            ans = str(refs[0])
            digits = re.findall(r"\d+", pred)
            hit = ans in pred or (bool(digits) and digits[0] == ans)
            return hit, float(hit), (digits[0] if digits else pred.strip()[:32])
        if task == "longbook_choice_eng":
            idx, ans_text = refs[0], refs[1]
            c = extract_letter(pred)
            want = "ABCD"[idx] if isinstance(idx, int) and 0 <= idx < 4 else ""
            hit = (c == want) or (str(ans_text).strip().lower() in pred.strip().lower())
            return hit, float(hit), c or pred.strip()[:32]
        if task == "longbook_qa_eng":
            best = max(_f1(pred, str(r)) for r in refs) if refs else 0.0
            return best >= 0.5, best, pred.strip()[:64]
    if source in ("gpqa", "mmlu_pro"):
        letters = "ABCD" if source == "gpqa" else "ABCDEFGHIJ"
        c = extract_letter(pred, letters)
        return c == refs[0], float(c == refs[0]), c
    return None, 0.0, pred.strip()[:64]
