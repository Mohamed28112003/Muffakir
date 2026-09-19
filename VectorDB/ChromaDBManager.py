import logging
from typing import Any, List, Optional

from .base import BaseVectorDBManager
from Embedding.EmbeddingProvider import EmbeddingProvider
# Kept as a module attribute for backward-compatible monkeypatching.  The
# actual optional adapter is imported only when the manager is constructed.
Chroma = None

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


class ChromaDBManager(BaseVectorDBManager):
    """
    Chroma Vector Database Provider.
    """

    def __init__(
        self,
        path: str = "./muffakir_db",
        collection_name: str = "ArabicBooks",
        embedding_provider: Optional[EmbeddingProvider] = None,
        model_name: str = "mohamed2811/Muffakir_Embedding",
    ):
        provider = embedding_provider or EmbeddingProvider(model_name=model_name)
        super().__init__(embedding_provider=provider)

        self.path = path
        self.collection_name = collection_name
        chroma_class = Chroma
        if chroma_class is None:
            from Muffakir.optional_dependencies import require_optional_dependency

            require_optional_dependency("chroma")
            from langchain_chroma import Chroma as chroma_class

        self._vector_store = chroma_class(
            collection_name=self.collection_name,
            persist_directory=self.path,
            embedding_function=self.embedding_provider,
        )

    @property
    def vector_store(self) -> Any:
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the Chroma collection and persist if supported."""
        if not documents:
            return
        ids = [f"chunk_{i}_{hash(doc.page_content)}" for i, doc in enumerate(documents)]
        self._vector_store.add_documents(documents=documents, ids=ids)
        # Chroma 0.4+ auto-persists; older versions require explicit persist()
        if hasattr(self._vector_store, "persist"):
            try:
                self._vector_store.persist()
            except Exception as e:
                from Trace.observability import mark_current_stage_error

                mark_current_stage_error(e, "fallback")
                logger.debug("Chroma persist() call skipped: %s", e)
        self._all_documents.extend(documents)

    def load_all_documents(self) -> List[Document]:
        """
        Retrieve all documents from the Chroma collection.
        Used for BM25 hybrid indexing after process restart.
        """
        try:
            collection = self._vector_store._collection
            result = collection.get(include=["documents", "metadatas"])
            docs = []
            for content, metadata in zip(
                result.get("documents", []),
                result.get("metadatas", []),
            ):
                docs.append(Document(page_content=content, metadata=metadata or {}))
            return docs
        except Exception as e:
            from Muffakir.exceptions import RetrievalError

            logger.error("Could not load all documents from Chroma: %s", e)
            raise RetrievalError(
                f"Failed to load documents from Chroma collection '{self.collection_name}': {e}"
            ) from e

    def get_collection_count(self) -> int:
        """Return the number of documents in the Chroma collection."""
        try:
            return self._vector_store._collection.count()
        except Exception as e:
            from Muffakir.exceptions import RetrievalError

            logger.error("Could not get Chroma collection count: %s", e)
            raise RetrievalError(
                f"Failed to get document count from Chroma collection '{self.collection_name}': {e}"
            ) from e
