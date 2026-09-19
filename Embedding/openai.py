import os
import logging
from typing import List, Optional
from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embedding provider using langchain_openai.OpenAIEmbeddings.
    Default model: 'text-embedding-3-small'.
    """

    def __init__(
        self,
        model_name: str = 'text-embedding-3-small',
        api_key: Optional[str] = None,
        cache_dir: str = '.embedding_cache',
        batch_size: int = 32
    ):
        super().__init__(
            model_name=model_name,
            provider_name="openai",
            cache_dir=cache_dir,
            batch_size=batch_size
        )

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAIEmbeddingProvider requires an API key. Pass `api_key=...` "
                "or set the OPENAI_API_KEY environment variable."
            )

        try:
            from langchain_openai import OpenAIEmbeddings
            self._underlying = OpenAIEmbeddings(
                model=self.model_name,
                api_key=resolved_key
            )
        except ImportError as e:
            raise ImportError(
                "langchain_openai is required for OpenAI embeddings. "
                "Install via `pip install langchain-openai`."
            ) from e

    def _embed_documents_raw(self, texts: List[str]) -> List[List[float]]:
        return self._underlying.embed_documents(texts)

    def _embed_query_raw(self, text: str) -> List[float]:
        return self._underlying.embed_query(text)
