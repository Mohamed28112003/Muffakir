"""Embedding public API with lazy concrete provider imports."""

from importlib import import_module

from .base import BaseEmbeddingProvider
from .factory import create_embedding_provider
from .EmbeddingProvider import EmbeddingProvider

__all__ = [
    "BaseEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "CohereEmbeddingProvider",
    "create_embedding_provider",
    "EmbeddingProvider",
]

_LAZY_EXPORTS = {
    "SentenceTransformerEmbeddingProvider": ("Embedding.sentence_transformer", "SentenceTransformerEmbeddingProvider"),
    "OpenAIEmbeddingProvider": ("Embedding.openai", "OpenAIEmbeddingProvider"),
    "CohereEmbeddingProvider": ("Embedding.cohere", "CohereEmbeddingProvider"),
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
