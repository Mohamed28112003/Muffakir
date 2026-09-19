import logging
from typing import List, Optional, Any
from .base import BaseVectorDBManager
from Embedding.EmbeddingProvider import EmbeddingProvider

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


class QdrantDBManager(BaseVectorDBManager):
    """
    Qdrant Vector Database Provider using langchain_qdrant.
    """

    def __init__(
        self,
        collection_name: str = "ArabicBooks",
        location: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        model_name: str = "mohamed2811/Muffakir_Embedding"
    ):
        provider = embedding_provider or EmbeddingProvider(model_name=model_name)
        super().__init__(embedding_provider=provider)

        self.collection_name = collection_name
        self.location = location
        self.api_key = api_key
        self.path = path
        self._vector_store = None

        try:
            from qdrant_client import QdrantClient
            from langchain_qdrant import QdrantVectorStore

            if self.path:
                client = QdrantClient(path=self.path)
            else:
                client = QdrantClient(url=self.location, api_key=self.api_key)

            self._client = client
            self._QdrantVectorStoreClass = QdrantVectorStore
        except ImportError:
            raise ImportError(
                "qdrant-client and langchain-qdrant are required for Qdrant provider. "
                "Install via `pip install qdrant-client langchain-qdrant`."
            )

    @property
    def vector_store(self) -> Any:
        if self._vector_store is None:
            # Initialize empty vector store connection if no documents added yet
            self._vector_store = self._QdrantVectorStoreClass(
                client=self._client,
                collection_name=self.collection_name,
                embedding=self.embedding_provider
            )
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """Ingest documents into Qdrant store."""
        if not documents:
            return

        self._vector_store = self._QdrantVectorStoreClass.from_documents(
            documents=documents,
            embedding=self.embedding_provider,
            client=self._client,
            collection_name=self.collection_name
        )
        self._all_documents.extend(documents)

    def load_all_documents(self) -> List[Document]:
        """
        Retrieve all documents from the Qdrant collection.
        Used for BM25 hybrid indexing after process restart.
        """
        try:
            from qdrant_client.models import ScrollRequest
            records, _ = self._client.scroll(
                collection_name=self.collection_name,
                limit=10_000,
                with_payload=True,
            )
            docs = []
            for record in records:
                payload = record.payload or {}
                content = payload.pop("page_content", "")
                docs.append(Document(page_content=content, metadata=payload))
            return docs
        except Exception as e:
            from Muffakir.exceptions import RetrievalError

            logger.error("Could not load all documents from Qdrant: %s", e)
            raise RetrievalError(
                f"Failed to load documents from Qdrant collection '{self.collection_name}': {e}"
            ) from e
