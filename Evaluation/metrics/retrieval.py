from __future__ import annotations

import math
from typing import List, Sequence


def recall_at_k(relevance: Sequence[int]) -> float:
    """Single-gold recall@k: 1 if any relevant in top-k else 0.

    This assumes a single gold chunk per query (the QAPair contract):
    any hit in the top-k means the gold was found.
    """
    if not relevance:
        return 0.0
    return 1.0 if any(int(r) > 0 for r in relevance) else 0.0


def precision_at_k(relevance: Sequence[int]) -> float:
    if not relevance:
        return 0.0
    hits = sum(1 for r in relevance if int(r) > 0)
    return hits / float(len(relevance))


def mean_reciprocal_rank(relevance: Sequence[int]) -> float:
    for i, r in enumerate(relevance, start=1):
        if int(r) > 0:
            return 1.0 / float(i)
    return 0.0


def dcg_at_k(relevance: Sequence[int]) -> float:
    score = 0.0
    for i, rel in enumerate(relevance, start=1):
        gain = float(rel)
        if gain <= 0:
            continue
        score += gain / math.log2(i + 1)
    return score


def ndcg_at_k(relevance: Sequence[int]) -> float:
    """Binary nDCG@k."""
    if not relevance:
        return 0.0
    dcg = dcg_at_k(relevance)
    ideal = sorted((int(r) for r in relevance), reverse=True)
    idcg = dcg_at_k(ideal)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def score_retrieval(relevance: Sequence[int], metrics: Sequence[str]) -> dict:
    out = {}
    metric_set = {m.lower() for m in metrics}
    if "recall" in metric_set:
        out["recall_at_k"] = recall_at_k(relevance)
    if "precision" in metric_set:
        out["precision_at_k"] = precision_at_k(relevance)
    if "mrr" in metric_set:
        out["mrr"] = mean_reciprocal_rank(relevance)
    if "ndcg" in metric_set:
        out["ndcg_at_k"] = ndcg_at_k(relevance)
    return out
