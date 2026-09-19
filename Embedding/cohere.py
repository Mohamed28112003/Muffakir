import os
import logging
from typing import List, Optional
from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """
    Cohere embedding provider using langchain_cohere.CohereEmbeddings.
    Default model: 'embed-multilingual-v3.0'.
    """

    def __init__(
        self,
        model_name: str = 'embed-multilingual-v3.0',
        api_key: Optional[str] = None,
        cache_dir: str = '.embedding_cache',
        batch_size: int = 32
    ):
        super().__init__(
            model_name=model_name,
            provider_name="cohere",
            cache_dir=cache_dir,
            batch_size=batch_size
        )

        resolved_key = api_key or os.environ.get("COHERE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "CohereEmbeddingProvider requires an API key. Pass `api_key=...` "
                "or set the COHERE_API_KEY environment variable."
            )

        try:
            from langchain_cohere import CohereEmbeddings
            self._underlying = CohereEmbeddings(
                model=self.model_name,
                cohere_api_key=resolved_key
            )
        except ImportError as e:
            raise ImportError(
                "langchain_cohere is required for Cohere embeddings. "
                "Install via `pip install langchain-cohere`."
            ) from e

    def _embed_documents_raw(self, texts: List[str]) -> List[List[float]]:
        return self._underlying.embed_documents(texts)

    def _embed_query_raw(self, text: str) -> List[float]:
        return self._underlying.embed_query(text)
