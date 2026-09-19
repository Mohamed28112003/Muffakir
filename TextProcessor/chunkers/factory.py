from typing import Any
from .base import BaseChunker
from Muffakir.optional_dependencies import require_optional_dependency

def create_chunker(strategy: str, **kwargs) -> BaseChunker:
    """
    Factory to create a chunker strategy by name.
    
    Args:
        strategy: "recursive", "character", "token", "sliding_window", "semantic", "contextual"
        **kwargs: Strategy-specific parameters (size, overlap, embeddings, etc.)
    """
    strategy = strategy.lower()
    
    if strategy == "recursive":
        require_optional_dependency("rag")
        from .recursive import RecursiveChunker
        return RecursiveChunker(**kwargs)
        
    elif strategy == "character":
        require_optional_dependency("rag")
        from .fixed_size import FixedSizeChunker
        return FixedSizeChunker(unit="character", **kwargs)
        
    elif strategy == "token":
        require_optional_dependency("rag")
        require_optional_dependency("token")
        from .fixed_size import FixedSizeChunker
        return FixedSizeChunker(unit="token", **kwargs)
        
    elif strategy == "sliding_window":
        from .sliding_window import SlidingWindowChunker
        return SlidingWindowChunker(**kwargs)
        
    elif strategy == "semantic":
        require_optional_dependency("semantic")
        from .semantic import SemanticChunker
        return SemanticChunker(**kwargs)
        
    elif strategy.startswith("contextual_"):
        from .contextual import ContextualChunker
        # Expected format: "contextual_recursive"
        base_name = strategy.replace("contextual_", "")
        # We can pop specific contextual kwargs if passed
        include_source = kwargs.pop("include_source", True)
        include_prev_chunk = kwargs.pop("include_prev_chunk", False)
        base_chunker = create_chunker(base_name, **kwargs)
        return ContextualChunker(base_chunker, include_source, include_prev_chunk)
        
    # Alias support for old code
    elif strategy == "spacy":
        # We dropped spaCy sentence chunker per user request
        # Fallback to recursive
        import logging
        logging.getLogger(__name__).warning(
            "spaCy chunking is deprecated. Falling back to recursive."
        )
        require_optional_dependency("rag")
        from .recursive import RecursiveChunker
        return RecursiveChunker(**kwargs)

    raise ValueError(f"Unknown chunking strategy: '{strategy}'")
