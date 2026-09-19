"""Evaluation public API with lazy pandas-backed imports."""

from Muffakir._lazy import load_attribute
from .constants import ALL_METRICS, DEFAULT_METRICS, GENERATION_METRICS, RETRIEVAL_METRICS

__all__ = [
    "ALL_METRICS", "DEFAULT_METRICS", "RETRIEVAL_METRICS", "GENERATION_METRICS",
    "EvaluationReport", "EvalSampleResult", "RetrievalScores",
    "GenerationScores", "load_evaluation_dataset", "EvalRunner",
]
_LAZY_EXPORTS = {
    "EvaluationReport": ("Evaluation.models", "EvaluationReport"),
    "EvalSampleResult": ("Evaluation.models", "EvalSampleResult"),
    "RetrievalScores": ("Evaluation.models", "RetrievalScores"),
    "GenerationScores": ("Evaluation.models", "GenerationScores"),
    "load_evaluation_dataset": ("Evaluation.dataset", "load_evaluation_dataset"),
    "EvalRunner": ("Evaluation.runner", "EvalRunner"),
}


def __getattr__(name):
    return load_attribute(name, _LAZY_EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
