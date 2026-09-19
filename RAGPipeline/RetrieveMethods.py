from __future__ import annotations

import logging
import threading
from typing import List, Optional, Any, Dict

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from LLMProvider.LLMProvider import LLMProvider

logger = logging.getLogger(__name__)

# Backward-compatible monkeypatch seams, resolved only when their retrieval
# modes execute. Importing LangChain's retriever namespace eagerly can pull in
# local ML runtimes, defeating the lightweight SDK import boundary.
_UNRESOLVED = object()
EnsembleRetriever = _UNRESOLVED
ContextualCompressionRetriever = _UNRESOLVED
LLMChainExtractor = _UNRESOLVED


def _load_ensemble_retriever():
    global EnsembleRetriever
    if EnsembleRetriever is not _UNRESOLVED:
        return EnsembleRetriever
    try:
        from langchain.retrievers import EnsembleRetriever as loaded
    except ImportError:
        try:
            from langchain_community.retrievers import EnsembleRetriever as loaded
        except ImportError:
            try:
                from langchain_classic.retrievers import EnsembleRetriever as loaded
            except ImportError:
                loaded = None
    EnsembleRetriever = loaded
    return loaded


def _load_contextual_components():
    global ContextualCompressionRetriever, LLMChainExtractor
    if (
        ContextualCompressionRetriever is not _UNRESOLVED
        or LLMChainExtractor is not _UNRESOLVED
    ):
        return ContextualCompressionRetriever, LLMChainExtractor
    try:
        from langchain.retrievers import ContextualCompressionRetriever as retriever
        from langchain.retrievers.document_compressors import LLMChainExtractor as extractor
    except ImportError:
        try:
            from langchain_community.retrievers import ContextualCompressionRetriever as retriever
            from langchain_community.document_compressors import LLMChainExtractor as extractor
        except ImportError:
            retriever = extractor = None
    ContextualCompressionRetriever = retriever
    LLMChainExtractor = extractor
    return retriever, extractor


class RetrieveMethods:
    """
    Retrieval strategies (Similarity Search, MMR, Hybrid RAG, Contextual RAG).
    Supports all pluggable VectorDB managers (Chroma, Qdrant, Pinecone, FAISS, Milvus).
    Includes in-memory BM25 index caching for high-performance Hybrid search.
    """

    def __init__(self, vector_store: Any = None, db_manager: Any = None):
        self.db_manager = db_manager
        if db_manager is not None:
            self.vector_store = db_manager.vector_store
        else:
            self.vector_store = vector_store

        # Cache for BM25 retriever to avoid rebuilding the index on every query
        self._bm25_retriever_cache: Optional[Any] = None
        self._bm25_lock = threading.Lock()

    def invalidate_bm25_cache(self) -> None:
        """Invalidate the cached BM25 retriever (e.g. after new documents are ingested)."""
        with self._bm25_lock:
            self._bm25_retriever_cache = None
        logger.debug("BM25 retriever cache invalidated.")

    def _get_all_documents(self) -> List[Document]:
        """Extract or retrieve all documents with full metadata preserved for BM25 indexing."""
        if self.db_manager is not None and hasattr(self.db_manager, "get_all_documents"):
            docs = self.db_manager.get_all_documents()
            if docs:
                return docs

        # Fallback for Chroma or vector stores with .get() method
        if hasattr(self.vector_store, "get"):
            try:
                res = self.vector_store.get()
                raw_docs = res.get("documents", [])
                raw_metas = res.get("metadatas", []) or []

                documents: List[Document] = []
                for i, doc in enumerate(raw_docs):
                    meta = raw_metas[i] if i < len(raw_metas) and isinstance(raw_metas[i], dict) else {}
                    if isinstance(doc, Document):
                        documents.append(doc)
                    else:
                        documents.append(Document(page_content=str(doc), metadata=meta))
                return documents
            except Exception as e:
                logger.debug(f"Could not retrieve documents via vector_store.get(): {e}")

        return []

    def similarity_search(self, query: str, k: int = 2) -> List[Document]:
        """Standard dense vector similarity search."""
        return self.vector_store.similarity_search(query, k=k)

    def max_marginal_relevance_search(self, query: str, k: int = 2, fetch_k: int = 12) -> List[Document]:
        """Maximal Marginal Relevance (MMR) search for diverse retrieval."""
        if hasattr(self.vector_store, "max_marginal_relevance_search"):
            return self.vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
        return self.vector_store.similarity_search(query, k=k)

    def _get_or_build_bm25_retriever(self) -> Optional[Any]:
        """Lazily build or retrieve cached BM25 retriever.

        Double-checked locking: avoids taking the lock on the common
        already-cached path, while preventing concurrent callers (e.g. multiple
        EvalRunner worker threads under hybrid retrieval) from each redundantly
        rebuilding the same BM25 index.
        """
        if self._bm25_retriever_cache is not None:
            return self._bm25_retriever_cache

        with self._bm25_lock:
            if self._bm25_retriever_cache is not None:
                return self._bm25_retriever_cache

            all_docs = self._get_all_documents()
            if not all_docs:
                return None

            formatted_docs = [
                doc if isinstance(doc, Document) else Document(page_content=str(doc))
                for doc in all_docs
            ]

            try:
                from Muffakir.optional_dependencies import require_optional_dependency

                require_optional_dependency("bm25")
                from langchain_community.retrievers import BM25Retriever

                self._bm25_retriever_cache = BM25Retriever.from_documents(formatted_docs)
                logger.debug(f"Built and cached BM25Retriever index over {len(formatted_docs)} documents.")
                return self._bm25_retriever_cache
            except Exception as e:
                logger.warning(f"Failed to build BM25 retriever: {e}")
                return None

    def hybrid_search(self, query: str, k: int = 2) -> List[Document]:
        """
        Hybrid RAG combining Dense Vector Search + Sparse BM25 Keyword Search.
        Uses Reciprocal Rank Fusion (RRF) for optimal score blending with cached BM25 index.
        """
        # 1. Vector Search
        vector_docs = self.vector_store.similarity_search(query, k=k * 2)

        # 2. Get or build cached BM25 retriever
        bm25_retriever = self._get_or_build_bm25_retriever()
        if bm25_retriever is None:
            logger.warning("BM25 index cannot be built because document store is empty. Returning vector results.")
            return vector_docs[:k]

        # 3. BM25 Search
        try:
            bm25_retriever.k = k * 2
            bm25_docs = bm25_retriever.invoke(query)
        except Exception as e:
            logger.warning(f"BM25 retriever error: {e}. Falling back to vector search.")
            return vector_docs[:k]

        # 4. Use LangChain EnsembleRetriever if available, otherwise built-in Reciprocal Rank Fusion (RRF)
        ensemble_retriever_class = _load_ensemble_retriever()
        if ensemble_retriever_class is not None:
            try:
                vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
                ensemble = ensemble_retriever_class(
                    retrievers=[vector_retriever, bm25_retriever],
                    weights=[0.5, 0.5]
                )
                results = ensemble.invoke(query)
                seen = set()
                deduped = []
                for doc in results:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        deduped.append(doc)
                return deduped[:k]
            except Exception as ex:
                logger.debug(f"EnsembleRetriever invocation error: {ex}, switching to RRF.")

        # Reciprocal Rank Fusion (RRF) fallback algorithm
        return self._reciprocal_rank_fusion(vector_docs, bm25_docs, k=k)

    # Backward-compatible alias for PascalCase
    HybridRAG = hybrid_search

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Document],
        bm25_results: List[Document],
        k: int = 2,
        rrf_k: int = 60
    ) -> List[Document]:
        """Reciprocal Rank Fusion (RRF) algorithm to combine rank lists."""
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(vector_results):
            content = doc.page_content
            doc_map[content] = doc
            doc_scores[content] = doc_scores.get(content, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, doc in enumerate(bm25_results):
            content = doc.page_content
            doc_map[content] = doc
            doc_scores[content] = doc_scores.get(content, 0.0) + 1.0 / (rrf_k + rank + 1)

        sorted_contents = sorted(doc_scores.keys(), key=lambda c: doc_scores[c], reverse=True)
        return [doc_map[content] for content in sorted_contents[:k]]

    def contextual_search(
        self,
        query: str,
        k: int = 2,
        llm_provider: Optional[LLMProvider] = None
    ) -> List[Document]:
        """Contextual Compression Retrieval using LLM extractor."""
        if llm_provider is None:
            raise ValueError("llm_provider is required for Contextual search")

        contextual_retriever_class, extractor_class = _load_contextual_components()
        if contextual_retriever_class is not None and extractor_class is not None:
            try:
                compressor = extractor_class.from_llm(llm_provider.get_llm())
                compression_retriever = contextual_retriever_class(
                    base_compressor=compressor,
                    base_retriever=self.vector_store.as_retriever(search_kwargs={"k": k})
                )
                return compression_retriever.invoke(query)
            except Exception as e:
                from Muffakir.exceptions import RetrievalError

                logger.error(f"Contextual compression failed: {e}")
                raise RetrievalError(f"Contextual compression retrieval failed: {e}") from e

        # Fallback to standard similarity search — only when contextual
        # compression support isn't available in this environment at all,
        # not on a runtime failure (that raises above instead).
        return self.similarity_search(query, k=k)

    # Backward-compatible alias for PascalCase
    ContextualRAG = contextual_search
