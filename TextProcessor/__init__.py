"""Text processing public API with lazy implementation imports."""

from importlib import import_module

__all__ = [
    "ChunkingAndProcessing",
    "MuffakirChunking",
    "MuffakirTextCleaner",
    "BaseChunker",
    "create_chunker",
    "RecursiveChunker",
    "FixedSizeChunker",
    "SlidingWindowChunker",
    "SemanticChunker",
    "ContextualChunker",
]

_LAZY_EXPORTS = {
    "ChunkingAndProcessing": ("TextProcessor.ChunkingAndProcessing", "ChunkingAndProcessing"),
    "MuffakirChunking": ("TextProcessor.MuffakirChunking", "MuffakirChunking"),
    "MuffakirTextCleaner": ("TextProcessor.MuffakirTextCleaner", "MuffakirTextCleaner"),
    "BaseChunker": ("TextProcessor.chunkers", "BaseChunker"),
    "create_chunker": ("TextProcessor.chunkers", "create_chunker"),
    "RecursiveChunker": ("TextProcessor.chunkers", "RecursiveChunker"),
    "FixedSizeChunker": ("TextProcessor.chunkers", "FixedSizeChunker"),
    "SlidingWindowChunker": ("TextProcessor.chunkers", "SlidingWindowChunker"),
    "SemanticChunker": ("TextProcessor.chunkers", "SemanticChunker"),
    "ContextualChunker": ("TextProcessor.chunkers", "ContextualChunker"),
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
