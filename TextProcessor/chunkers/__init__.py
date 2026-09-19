"""Chunker public API with lazy strategy imports."""

from Muffakir._lazy import load_attribute
from .base import BaseChunker
from .factory import create_chunker

__all__ = [
    "BaseChunker", "create_chunker", "FixedSizeChunker", "RecursiveChunker",
    "SlidingWindowChunker", "ContextualChunker", "SemanticChunker",
]
_LAZY_EXPORTS = {
    "FixedSizeChunker": ("TextProcessor.chunkers.fixed_size", "FixedSizeChunker"),
    "RecursiveChunker": ("TextProcessor.chunkers.recursive", "RecursiveChunker"),
    "SlidingWindowChunker": ("TextProcessor.chunkers.sliding_window", "SlidingWindowChunker"),
    "ContextualChunker": ("TextProcessor.chunkers.contextual", "ContextualChunker"),
    "SemanticChunker": ("TextProcessor.chunkers.semantic", "SemanticChunker"),
}


def __getattr__(name):
    return load_attribute(name, _LAZY_EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
