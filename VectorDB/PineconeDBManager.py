import logging
from typing import List, Optional, Any

from .base import BaseVectorDBManager
from Embedding.EmbeddingProvider import EmbeddingProvider

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


class PineconeDBManager(BaseVectorDBManager):
    """
    Pinecone Vector Database Provider using langchain_pinecone.
    """

    def __init__(
        self,
        index_name: str = "muffakir-index",
        api_key: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        model_name: str = "mohamed2811/Muffakir_Embedding",
    ):
        provider = embedding_provider or EmbeddingProvider(model_name=model_name)
        super().__init__(embedding_provider=provider)

        self.index_name = index_name
        # Store the API key as an instance attribute — never mutate os.environ in a library
        self._api_key = api_key
        self._vector_store = None

        try:
            from langchain_pinecone import PineconeVectorStore
            self._PineconeVectorStoreClass = PineconeVectorStore
        except ImportError:
            raise ImportError(
                "langchain-pinecone is required for Pinecone provider. "
                "Install via `pip install langchain-pinecone`."
            )

    @property
    def vector_store(self) -> Any:
        if self._vector_store is None:
            kwargs = {"index_name": self.index_name, "embedding": self.embedding_provider}
            if self._api_key:
                kwargs["pinecone_api_key"] = self._api_key
            self._vector_store = self._PineconeVectorStoreClass(**kwargs)
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """Ingest documents into Pinecone index."""
        if not documents:
            return

        kwargs = {
            "documents": documents,
            "embedding": self.embedding_provider,
            "index_name": self.index_name,
        }
        if self._api_key:
            kwargs["pinecone_api_key"] = self._api_key

        self._vector_store = self._PineconeVectorStoreClass.from_documents(**kwargs)
        self._all_documents.extend(documents)

    def load_all_documents(self) -> List[Document]:
        """
        Pinecone does not support bulk document retrieval from the SDK.
        Returns the in-memory cache populated during add_documents().
        """
        return list(self._all_documents)
