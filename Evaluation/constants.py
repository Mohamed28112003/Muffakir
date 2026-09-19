"""Dependency-free evaluation metric names shared by the SDK and UI."""

ALL_METRICS = [
    "recall",
    "precision",
    "mrr",
    "ndcg",
    "faithfulness",
    "answer_correctness",
    "llm_judge_rating",
]

RETRIEVAL_METRICS = {"recall", "precision", "mrr", "ndcg"}
GENERATION_METRICS = {"faithfulness", "answer_correctness", "llm_judge_rating"}

# LLM Judge Rating is intentionally opt-in because it adds an independent
# judge-model call per sample. Keep the historical implicit/default metric set
# stable even though ALL_METRICS now advertises the new supported metric.
DEFAULT_METRICS = [
    "recall",
    "precision",
    "mrr",
    "ndcg",
    "faithfulness",
    "answer_correctness",
]

__all__ = ["ALL_METRICS", "DEFAULT_METRICS", "RETRIEVAL_METRICS", "GENERATION_METRICS"]
