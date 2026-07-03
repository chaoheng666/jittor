import math

import numpy as np


def row_zscore(scores):
    scores = np.asarray(scores, dtype=np.float32)
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (scores - mean) / std


def softmax(scores, temperature=1.0):
    scores = np.asarray(scores, dtype=np.float64)
    temperature = max(float(temperature), 1e-6)
    scores = scores / temperature
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    total = exp_scores.sum(axis=1, keepdims=True)
    total = np.where(total <= 0, 1.0, total)
    return exp_scores / total


def reciprocal_rank(scores, positive_index=0):
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[int(positive_index)]
    rank = 1 + int(np.sum(scores > pos))
    return 1.0 / rank


def hit_at_k(scores, positive_index=0, k=10):
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[int(positive_index)]
    rank = 1 + int(np.sum(scores > pos))
    return 1.0 if rank <= k else 0.0


def ranking_metrics(score_rows, positive_index=0, hit_k=10):
    rows = list(score_rows)
    if not rows:
        return {"mrr": 0.0, f"hit{hit_k}": 0.0, "queries": 0}
    rr = [reciprocal_rank(row, positive_index) for row in rows]
    hits = [hit_at_k(row, positive_index, hit_k) for row in rows]
    return {
        "mrr": float(np.mean(rr)),
        f"hit{hit_k}": float(np.mean(hits)),
        "queries": int(len(rows)),
    }


def probability_report(probs, expected_cols=100, row_sum_tol=1e-5):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        return {
            "valid": False,
            "reason": "not_2d",
            "rows": int(probs.shape[0]) if probs.ndim else 0,
            "cols": 0,
        }
    row_sums = probs.sum(axis=1)
    has_nan = bool(np.isnan(probs).any())
    has_inf = bool(np.isinf(probs).any())
    has_negative = bool((probs < -1e-12).any())
    cols_ok = int(probs.shape[1]) == int(expected_cols)
    max_row_sum_error = float(np.max(np.abs(row_sums - 1.0))) if len(row_sums) else 0.0
    const_rows = int(np.sum(np.std(probs, axis=1) < 1e-12)) if probs.size else 0
    valid = (
        cols_ok
        and not has_nan
        and not has_inf
        and not has_negative
        and max_row_sum_error <= row_sum_tol
    )
    return {
        "valid": bool(valid),
        "rows": int(probs.shape[0]),
        "cols": int(probs.shape[1]),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "has_negative": has_negative,
        "max_row_sum_error": max_row_sum_error,
        "const_rows": const_rows,
    }


def topk_unseen_stats(scores, candidates, seen_dst, ks=(1, 5)):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array")
    totals = {int(k): 0 for k in ks}
    unseen = {int(k): 0 for k in ks}
    top1 = []
    for row_scores, row_candidates in zip(scores, candidates):
        order = np.argsort(-row_scores)
        for k in ks:
            k = int(k)
            chosen = [row_candidates[i] for i in order[:k]]
            totals[k] += len(chosen)
            unseen[k] += sum(dst not in seen_dst for dst in chosen)
        top1.append(row_candidates[int(order[0])])
    out = {}
    for k in ks:
        k = int(k)
        out[f"top{k}_unseen_frac_pred"] = unseen[k] / max(totals[k], 1)
    out["top1_values"] = top1
    return out


def finite_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default
