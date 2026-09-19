from abc import ABC, abstractmethod
from typing import List, Optional, Any
import logging
import threading

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_core.vectorstores import VectorStore
from Embedding.EmbeddingProvider import EmbeddingProvider

logger = logging.getLogger(__name__)


class BaseVectorDBManager(ABC):
    """
    Abstract Base Class for Vector Database Managers in Muffakir RAG.

    All VectorDB backends must inherit from this class and implement:
    - vector_store: property returning the underlying LangChain VectorStore
    - add_documents(): ingest documents into the backend
    - load_all_documents(): retrieve all persisted documents from the backend
    """

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self._all_documents: List[Document] = []
        self._all_documents_lock = threading.Lock()

    @property
    @abstractmethod
    def vector_store(self) -> VectorStore:
        """Return the underlying LangChain VectorStore instance."""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    def load_all_documents(self) -> List[Document]:
        """
        Retrieve all documents from the persistent vector store.

        Used by HybridRAG for BM25 in-memory indexing. Each backend must
        implement this to fetch documents from its own storage layer, since
        `_all_documents` is ephemeral and lost on process restart.

        Returns:
            List[Document]: All documents currently stored in this backend.
        """
        pass

    def _assert_store_ready(self) -> None:
        """
        Raise a descriptive RuntimeError if the vector store is not initialized.
        Called by search methods to give a clear error instead of an AttributeError.
        """
        if self.vector_store is None:
            raise RuntimeError(
                "No documents have been indexed yet. "
                "Call add_documents() before running a search."
            )

    def similarity_search(self, query: str, k: int = 2) -> List[Document]:
        """Perform similarity vector search."""
        self._assert_store_ready()
        return self.vector_store.similarity_search(query, k=k)

    def max_marginal_relevance_search(self, query: str, k: int = 2, fetch_k: int = 12) -> List[Document]:
        """Perform Maximal Marginal Relevance (MMR) search."""
        self._assert_store_ready()
        if hasattr(self.vector_store, "max_marginal_relevance_search"):
            return self.vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
        # Fallback to standard similarity search if MMR is not supported by backend
        return self.vector_store.similarity_search(query, k=k)

    def get_all_documents(self) -> List[Document]:
        """
        Returns all stored documents for local BM25 Hybrid indexing.

        Delegates to `load_all_documents()` if the in-memory cache is empty
        (e.g. after a process restart when the backend has persisted data).
        """
        if not self._all_documents:
            with self._all_documents_lock:
                if not self._all_documents:
                    self._all_documents = self.load_all_documents()
        return self._all_documents
