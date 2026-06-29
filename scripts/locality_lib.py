#!/usr/bin/env python3
"""Shared helpers for sparse-attention KV temporal-locality analysis.

Pure functions over selected-index sets/lists so they are unit-testable
(see validate_trace.py). Logical compressed-KV space throughout.
"""
import math


def topk_indices(scores, k):
    """Deterministic top-k by score desc, index asc tie-break. Returns ranked indices."""
    k = min(k, len(scores))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:k]


def ranks_from_scores(sel, scores_by_index):
    """Assign selected_rank (0=highest score) to selected indices.
    sel: list of compressed indices. scores_by_index: dict idx->score or None.
    Falls back to ascending-index order when scores are absent."""
    if scores_by_index:
        return sorted(sel, key=lambda c: (-scores_by_index.get(c, float("-inf")), c))
    return sorted(sel)


def jaccard(a, b):
    a, b = set(a), set(b)
    u = a | b
    return (len(a & b) / len(u)) if u else 1.0


def overlap_fraction(cur, prev):
    """|cur ∩ prev| / |cur| — fraction of current set retained from prev."""
    cur = set(cur)
    if not cur:
        return float("nan")
    return len(cur & set(prev)) / len(cur)


def churn(cur, prev):
    return 1.0 - overlap_fraction(cur, prev)


def new_evicted(cur, prev):
    cur, prev = set(cur), set(prev)
    return len(cur - prev), len(prev - cur)


def rank_weight(rank):
    """DCG-style positional weight; rank is 0-based."""
    return 1.0 / math.log2(rank + 2)


def weighted_overlap(cur_ranked, prev_ranked):
    """Rank-aware overlap: sum over shared indices of min(w_cur, w_prev),
    normalized by total weight of one top-k list. Inputs are ranked index lists."""
    wc = {c: rank_weight(r) for r, c in enumerate(cur_ranked)}
    wp = {c: rank_weight(r) for r, c in enumerate(prev_ranked)}
    shared = set(wc) & set(wp)
    num = sum(min(wc[c], wp[c]) for c in shared)
    denom = sum(wc.values())
    return (num / denom) if denom else float("nan")


def random_expected_overlap_fraction(k, n):
    """E[overlap_fraction] for two independent uniform k-subsets of N candidates ≈ k/N."""
    if n <= 0:
        return float("nan")
    if n <= k:
        return 1.0
    return k / n


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    """Rank correlation via Pearson on ranks (average ranks for ties)."""
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    if len(xs) < 2:
        return float("nan")
    return pearson(rank(xs), rank(ys))


def bootstrap_ci(samples, stat=lambda v: sum(v) / len(v), n_boot=2000, seed=12345):
    """Percentile bootstrap CI by resampling the list `samples` (sample-level)."""
    if len(samples) < 2:
        v = stat(samples) if samples else float("nan")
        return (v, v)
    # deterministic LCG to avoid Math.random-style nondeterminism
    state = seed
    def rnd():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF
    boots = []
    n = len(samples)
    for _ in range(n_boot):
        res = [samples[int(rnd() * n) % n] for _ in range(n)]
        boots.append(stat(res))
    boots.sort()
    return (percentile(boots, 0.025), percentile(boots, 0.975))


def original_token_range(c, ratio):
    return c * ratio, c * ratio + ratio - 1


def representative_original_pos(c, ratio):
    return c * ratio + ratio // 2
