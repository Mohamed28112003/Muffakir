import os
import logging
from typing import List, Optional, Any

from .base import BaseVectorDBManager
from Embedding.EmbeddingProvider import EmbeddingProvider

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


class FAISSDBManager(BaseVectorDBManager):
    """
    FAISS (Facebook AI Similarity Search) Vector Database Provider.
    Supports in-memory vector storage with optional disk persistence.
    """

    def __init__(
        self,
        folder_path: str = "./faiss_index",
        index_name: str = "index",
        embedding_provider: Optional[EmbeddingProvider] = None,
        model_name: str = "mohamed2811/Muffakir_Embedding",
    ):
        provider = embedding_provider or EmbeddingProvider(model_name=model_name)
        super().__init__(embedding_provider=provider)

        self.folder_path = folder_path
        self.index_name = index_name
        self._vector_store = None

        try:
            from langchain_community.vectorstores import FAISS
            self._FAISSClass = FAISS
        except ImportError:
            raise ImportError(
                "faiss-cpu (or faiss-gpu) is required for FAISS provider. "
                "Install via `pip install faiss-cpu langchain-community`."
            )

        # Attempt to load existing index if present
        if os.path.exists(self.folder_path):
            try:
                self._vector_store = self._FAISSClass.load_local(
                    folder_path=self.folder_path,
                    embeddings=self.embedding_provider,
                    index_name=self.index_name,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Loaded existing FAISS index from %s", self.folder_path)
            except Exception as e:
                from Trace.observability import mark_current_stage_error

                mark_current_stage_error(e, "fallback")
                logger.debug("Could not load FAISS index from %s: %s", self.folder_path, e)

    @property
    def vector_store(self) -> Any:
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to FAISS index and save to disk."""
        if not documents:
            return

        if self._vector_store is None:
            self._vector_store = self._FAISSClass.from_documents(
                documents=documents,
                embedding=self.embedding_provider,
            )
        else:
            self._vector_store.add_documents(documents)

        # Save to disk
        os.makedirs(self.folder_path, exist_ok=True)
        self._vector_store.save_local(
            folder_path=self.folder_path,
            index_name=self.index_name,
        )
        self._all_documents.extend(documents)

    def load_all_documents(self) -> List[Document]:
        """
        FAISS is an in-memory index with no native document storage retrieval.
        Returns the in-memory cache populated during add_documents().
        For BM25 hybrid search after restart, re-index documents explicitly.
        """
        return list(self._all_documents)
