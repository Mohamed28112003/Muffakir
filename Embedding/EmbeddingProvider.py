from typing import Optional, Any
from langchain_core.embeddings import Embeddings
from .factory import create_embedding_provider
from .base import BaseEmbeddingProvider


def EmbeddingProvider(
    model_name: str = 'mohamed2811/Muffakir_Embedding',
    provider: str = "sentence_transformers",
    api_key: Optional[str] = None,
    cache_dir: str = '.embedding_cache',
    batch_size: int = 32,
    custom_embeddings: Optional[Embeddings] = None,
    device: str = "auto",
    **kwargs: Any
) -> BaseEmbeddingProvider:
    """
    Backward-compatible entry point for initializing embedding providers.
    Delegates directly to create_embedding_provider factory.
    """
    return create_embedding_provider(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        cache_dir=cache_dir,
        batch_size=batch_size,
        custom_embeddings=custom_embeddings,
        device=device,
        **kwargs
    )
