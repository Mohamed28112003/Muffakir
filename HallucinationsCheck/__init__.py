"""Hallucination-checking public API with lazy strategy imports."""

from importlib import import_module

__all__ = [
    "BaseHallucinationChecker",
    "HallucinationResult",
    "ContextGroundingChecker",
    "NLIChecker",
    "SemanticSimilarityChecker",
    "TextCleanerChecker",
    "create_hallucination_checker",
    "HallucinationsCheck",
]

_LAZY_EXPORTS = {
    "BaseHallucinationChecker": ("HallucinationsCheck.base", "BaseHallucinationChecker"),
    "HallucinationResult": ("HallucinationsCheck.base", "HallucinationResult"),
    "ContextGroundingChecker": ("HallucinationsCheck.context_grounding", "ContextGroundingChecker"),
    "NLIChecker": ("HallucinationsCheck.nli", "NLIChecker"),
    "SemanticSimilarityChecker": ("HallucinationsCheck.semantic_similarity", "SemanticSimilarityChecker"),
    "TextCleanerChecker": ("HallucinationsCheck.text_cleaner", "TextCleanerChecker"),
    "create_hallucination_checker": ("HallucinationsCheck.factory", "create_hallucination_checker"),
    "HallucinationsCheck": ("HallucinationsCheck.HallucinationsCheck", "HallucinationsCheck"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
