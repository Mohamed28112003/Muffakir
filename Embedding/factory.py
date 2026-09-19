import logging
from typing import Optional, Any
from langchain_core.embeddings import Embeddings
from .base import BaseEmbeddingProvider
from Muffakir.optional_dependencies import require_optional_dependency

logger = logging.getLogger(__name__)


class _CustomWrapper(BaseEmbeddingProvider):
    """Wraps a user-supplied LangChain ``Embeddings`` instance with caching.

    Defined at module level so it remains picklable (required when the
    embedding provider is shared across processes, e.g. Composer ``n_jobs>1``).
    """

    def __init__(self, emb: Embeddings, model_name: str, cache_dir: str, batch_size: int):
        super().__init__(
            model_name=model_name,
            provider_name="custom",
            cache_dir=cache_dir,
            batch_size=batch_size,
        )
        self.emb = emb

    def _embed_documents_raw(self, texts):
        return self.emb.embed_documents(texts)

    def _embed_query_raw(self, text):
        return self.emb.embed_query(text)


def available_providers() -> str:
    """Human-readable list of available embedding provider names."""
    return "'sentence_transformers', 'openai', 'cohere'"


def create_embedding_provider(
    provider: str = "sentence_transformers",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    cache_dir: str = '.embedding_cache',
    batch_size: int = 32,
    custom_embeddings: Optional[Embeddings] = None,
    device: str = "auto",
    **kwargs: Any
) -> BaseEmbeddingProvider:
    """
    Factory function to instantiate pluggable Embedding Providers.

    Args:
        provider (str): 'sentence_transformers' / 'huggingface' / 'local', 'openai', 'cohere', or 'custom'.
        model_name (str, optional): Model identifier.
        api_key (str, optional): Provider API key.
        cache_dir (str): Directory path for disk caching.
        batch_size (int): Batch size for bulk document encoding.
        custom_embeddings (Embeddings, optional): Custom injected LangChain Embeddings instance.

    Returns:
        BaseEmbeddingProvider: An instance inheriting from BaseEmbeddingProvider.

    Raises:
        ValueError: if ``provider`` is unknown or empty.
    """
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be a non-empty string.")

    provider_clean = provider.lower().strip()

    if custom_embeddings is not None:
        return _CustomWrapper(custom_embeddings, model_name or "custom_model", cache_dir, batch_size)

    if provider_clean in ("sentence_transformers", "sentence_transformer", "huggingface", "local"):
        require_optional_dependency("local")
        from .sentence_transformer import SentenceTransformerEmbeddingProvider
        default_model = model_name or "mohamed2811/Muffakir_Embedding"
        return SentenceTransformerEmbeddingProvider(
            model_name=default_model,
            cache_dir=cache_dir,
            batch_size=batch_size,
            device=device,
        )

    elif provider_clean in ("openai", "langchain_openai"):
        require_optional_dependency("openai")
        from .openai import OpenAIEmbeddingProvider
        default_model = model_name or "text-embedding-3-small"
        return OpenAIEmbeddingProvider(
            model_name=default_model,
            api_key=api_key,
            cache_dir=cache_dir,
            batch_size=batch_size
        )

    elif provider_clean in ("cohere", "langchain_cohere"):
        require_optional_dependency("cohere")
        from .cohere import CohereEmbeddingProvider
        default_model = model_name or "embed-multilingual-v3.0"
        return CohereEmbeddingProvider(
            model_name=default_model,
            api_key=api_key,
            cache_dir=cache_dir,
            batch_size=batch_size
        )

    raise ValueError(
        f"Unknown embedding provider: '{provider}'. "
        f"Available options: {available_providers()}."
    )
