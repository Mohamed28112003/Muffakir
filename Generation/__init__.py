"""Generation public API with lazy implementation imports."""

from importlib import import_module

__all__ = ["AnswerGenerator", "ContextRelevanceChecker", "DocumentRetriever", "RAGGenerationPipeline"]
_LAZY_EXPORTS = {
    "AnswerGenerator": ("Generation.AnswerGenerator", "AnswerGenerator"),
    "ContextRelevanceChecker": ("Generation.ContextRelevanceChecker", "ContextRelevanceChecker"),
    "DocumentRetriever": ("Generation.DocumentRetriever", "DocumentRetriever"),
    "RAGGenerationPipeline": ("Generation.RAGGenerationPipeline", "RAGGenerationPipeline"),
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
