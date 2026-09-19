import logging
from typing import List, Optional, Any, Dict
from .base import BaseVectorDBManager
from Embedding.EmbeddingProvider import EmbeddingProvider

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


class MilvusDBManager(BaseVectorDBManager):
    """
    Milvus Vector Database Provider using langchain_milvus.
    Supports both local file-based Milvus Lite (`./milvus_local.db`) and remote cluster (`http://localhost:19530`).
    """

    def __init__(
        self,
        collection_name: str = "ArabicBooks",
        connection_args: Optional[Dict[str, Any]] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        model_name: str = "mohamed2811/Muffakir_Embedding"
    ):
        provider = embedding_provider or EmbeddingProvider(model_name=model_name)
        super().__init__(embedding_provider=provider)

        self.collection_name = collection_name
        self.connection_args = connection_args or {"uri": "./milvus_local.db"}
        self._vector_store = None

        try:
            from langchain_milvus import Milvus
            self._MilvusClass = Milvus
        except ImportError:
            raise ImportError(
                "langchain-milvus and pymilvus are required for Milvus provider. "
                "Install via `pip install langchain-milvus pymilvus`."
            )

    @property
    def vector_store(self) -> Any:
        if self._vector_store is None:
            self._vector_store = self._MilvusClass(
                embedding_function=self.embedding_provider,
                collection_name=self.collection_name,
                connection_args=self.connection_args
            )
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """Ingest documents into Milvus store."""
        if not documents:
            return

        self._vector_store = self._MilvusClass.from_documents(
            documents=documents,
            embedding_function=self.embedding_provider,
            collection_name=self.collection_name,
            connection_args=self.connection_args
        )
        self._all_documents.extend(documents)

    def load_all_documents(self) -> List[Document]:
        """
        Milvus does not support simple bulk retrieval without a query.
        Returns the in-memory cache populated during add_documents().
        For BM25 hybrid search after restart, re-index documents explicitly.
        """
        return list(self._all_documents)
