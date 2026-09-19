"""Reranker public API with lazy strategy imports."""

from importlib import import_module

from .base import BaseReranker
from .factory import (
    RerankerSpec,
    create_reranker,
    get_reranker_spec,
    list_reranker_specs,
    register_reranker,
)
from .Reranker import Reranker

__all__ = [
    "BaseReranker",
    "SemanticSimilarityReranker",
    "BM25Reranker",
    "CrossEncoderReranker",
    "PointwiseReranker",
    "LLMReranker",
    "RemoteReranker",
    "RerankerSpec",
    "create_reranker",
    "get_reranker_spec",
    "list_reranker_specs",
    "register_reranker",
    "Reranker",
]

_LAZY_EXPORTS = {
    "SemanticSimilarityReranker": ("Reranker.semantic_similarity", "SemanticSimilarityReranker"),
    "BM25Reranker": ("Reranker.bm25", "BM25Reranker"),
    "CrossEncoderReranker": ("Reranker.cross_encoder", "CrossEncoderReranker"),
    "PointwiseReranker": ("Reranker.pointwise", "PointwiseReranker"),
    "LLMReranker": ("Reranker.llm", "LLMReranker"),
    "RemoteReranker": ("Reranker.remote", "RemoteReranker"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
