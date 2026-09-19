import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

import sys

from langchain_community.retrievers import BM25Retriever

from Muffakir.Enums import RetrievalMethod
from Muffakir.exceptions import RetrievalError
from RAGPipeline import RAGPipelineManager, RetrieveMethods

# `RAGPipeline/__init__.py` re-exports the RetrieveMethods *class* under the
# same attribute name as the submodule, shadowing `RAGPipeline.RetrieveMethods`
# as a module path — look the actual module up via sys.modules instead.
retrieve_methods_module = sys.modules["RAGPipeline.RetrieveMethods"]


class MockVectorStore:
    def __init__(self, docs=None):
        self.docs = docs or [
            Document(page_content="الذكاء الاصطناعي يغير العالم.", metadata={"source": "doc1.txt", "chunk_id": 0}),
            Document(page_content="معالجة اللغات الطبيعية فرع مهم.", metadata={"source": "doc2.txt", "chunk_id": 1}),
            Document(page_content="تعلم الآلة يعتمد على البيانات.", metadata={"source": "doc3.txt", "chunk_id": 2}),
        ]

    def similarity_search(self, query: str, k: int = 2):
        return self.docs[:k]

    def max_marginal_relevance_search(self, query: str, k: int = 2, fetch_k: int = 12):
        return self.docs[:k]

    def as_retriever(self, search_kwargs=None):
        k = (search_kwargs or {}).get("k", 2)
        mock = MagicMock()
        mock.invoke = MagicMock(return_value=self.docs[:k])
        return mock

    def get(self):
        return {
            "documents": [d.page_content for d in self.docs],
            "metadatas": [d.metadata for d in self.docs],
        }


class MockDBManager:
    def __init__(self, docs=None):
        self.docs = docs or [
            Document(page_content="الذكاء الاصطناعي يغير العالم.", metadata={"source": "doc1.txt", "chunk_id": 0}),
            Document(page_content="معالجة اللغات الطبيعية فرع مهم.", metadata={"source": "doc2.txt", "chunk_id": 1}),
        ]
        self.vector_store = MockVectorStore(self.docs)
        self.added_docs = []

    def get_all_documents(self):
        return self.docs

    def add_documents(self, documents):
        self.added_docs.extend(documents)
        self.docs.extend(documents)


# ---------------------------------------------------------------------------
# RetrieveMethods Tests
# ---------------------------------------------------------------------------

def test_retrieve_methods_similarity_search():
    store = MockVectorStore()
    retriever = RetrieveMethods(vector_store=store)
    results = retriever.similarity_search("الذكاء", k=2)
    assert len(results) == 2
    assert results[0].page_content == "الذكاء الاصطناعي يغير العالم."


def test_retrieve_methods_mmr_search():
    store = MockVectorStore()
    retriever = RetrieveMethods(vector_store=store)
    results = retriever.max_marginal_relevance_search("الذكاء", k=1)
    assert len(results) == 1


def test_retrieve_methods_metadata_preservation_from_chroma_get():
    store = MockVectorStore()
    retriever = RetrieveMethods(vector_store=store)
    all_docs = retriever._get_all_documents()
    assert len(all_docs) == 3
    assert all_docs[0].metadata.get("source") == "doc1.txt"
    assert all_docs[0].metadata.get("chunk_id") == 0


def test_retrieve_methods_bm25_caching():
    db_manager = MockDBManager()
    retriever = RetrieveMethods(db_manager=db_manager)

    assert retriever._bm25_retriever_cache is None

    # First call builds and caches BM25 retriever
    bm25_1 = retriever._get_or_build_bm25_retriever()
    assert bm25_1 is not None
    assert retriever._bm25_retriever_cache is bm25_1

    # Second call returns cached instance
    bm25_2 = retriever._get_or_build_bm25_retriever()
    assert bm25_2 is bm25_1

    # Invalidate cache
    retriever.invalidate_bm25_cache()
    assert retriever._bm25_retriever_cache is None


def test_retrieve_methods_bm25_build_concurrent_calls_build_once(monkeypatch):
    """Concurrent first-access calls to _get_or_build_bm25_retriever() must build
    the BM25 index exactly once — the double-checked lock must prevent every
    racing thread from redundantly rebuilding it."""
    import threading
    import time

    build_calls = {"count": 0}
    real_from_documents = BM25Retriever.from_documents

    def slow_from_documents(docs):
        build_calls["count"] += 1
        time.sleep(0.05)
        return real_from_documents(docs)

    monkeypatch.setattr(BM25Retriever, "from_documents", staticmethod(slow_from_documents))

    db_manager = MockDBManager()
    retriever = RetrieveMethods(db_manager=db_manager)

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(retriever._get_or_build_bm25_retriever()))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert build_calls["count"] == 1
    assert all(r is retriever._bm25_retriever_cache for r in results)


def test_retrieve_methods_hybrid_search_and_alias():
    db_manager = MockDBManager()
    retriever = RetrieveMethods(db_manager=db_manager)

    results = retriever.hybrid_search("الذكاء", k=2)
    assert len(results) <= 2
    assert len(results) > 0

    # Test PascalCase backward-compatible alias
    results_alias = retriever.HybridRAG("الذكاء", k=2)
    assert len(results_alias) <= 2


def test_reciprocal_rank_fusion():
    retriever = RetrieveMethods(vector_store=MockVectorStore())
    vec_results = [
        Document(page_content="doc A"),
        Document(page_content="doc B"),
    ]
    bm25_results = [
        Document(page_content="doc B"),
        Document(page_content="doc C"),
    ]
    fused = retriever._reciprocal_rank_fusion(vec_results, bm25_results, k=2)
    assert len(fused) == 2
    # "doc B" is in both top ranks, so it should be ranked first by RRF
    assert fused[0].page_content == "doc B"


def test_retrieve_methods_contextual_search_fallback(monkeypatch):
    monkeypatch.setattr(retrieve_methods_module, "ContextualCompressionRetriever", None)
    monkeypatch.setattr(retrieve_methods_module, "LLMChainExtractor", None)
    store = MockVectorStore()
    retriever = RetrieveMethods(vector_store=store)

    with pytest.raises(ValueError):
        retriever.contextual_search("query", k=2, llm_provider=None)

    mock_llm_provider = MagicMock()
    mock_llm_provider.get_llm.return_value = MagicMock()

    # Fallback to similarity search when ContextualCompressionRetriever is not available
    results = retriever.contextual_search("الذكاء", k=2, llm_provider=mock_llm_provider)
    assert len(results) == 2

    # Alias check
    results_alias = retriever.ContextualRAG("الذكاء", k=2, llm_provider=mock_llm_provider)
    assert len(results_alias) == 2


def test_retrieve_methods_contextual_search_runtime_failure_raises(monkeypatch):
    """A compression crash must raise, not silently downgrade to similarity search."""
    mock_compressor = MagicMock()
    mock_extractor_cls = MagicMock()
    mock_extractor_cls.from_llm.return_value = mock_compressor

    mock_compression_retriever_instance = MagicMock()
    mock_compression_retriever_instance.invoke.side_effect = RuntimeError("LLM down")
    mock_compression_retriever_cls = MagicMock(return_value=mock_compression_retriever_instance)

    monkeypatch.setattr(retrieve_methods_module, "ContextualCompressionRetriever", mock_compression_retriever_cls)
    monkeypatch.setattr(retrieve_methods_module, "LLMChainExtractor", mock_extractor_cls)

    store = MockVectorStore()
    retriever = RetrieveMethods(vector_store=store)
    mock_llm_provider = MagicMock()
    mock_llm_provider.get_llm.return_value = MagicMock()

    with pytest.raises(RetrievalError) as excinfo:
        retriever.contextual_search("query", k=2, llm_provider=mock_llm_provider)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# RAGPipelineManager Tests
# ---------------------------------------------------------------------------

def test_rag_pipeline_manager_init_and_store():
    db_manager = MockDBManager()
    manager = RAGPipelineManager(
        db_manager=db_manager,
        k=2,
        retrieve_method=RetrievalMethod.SIMILARITY_SEARCH,
    )

    new_doc = Document(page_content="وثيقة جديدة تماماً.", metadata={"source": "new.txt"})
    manager.store_documents([new_doc])

    assert new_doc in db_manager.added_docs
    assert manager.retriever._bm25_retriever_cache is None  # Cache was invalidated


def test_rag_pipeline_manager_query_dispatch(monkeypatch):
    monkeypatch.setattr(retrieve_methods_module, "ContextualCompressionRetriever", None)
    monkeypatch.setattr(retrieve_methods_module, "LLMChainExtractor", None)
    db_manager = MockDBManager()
    manager = RAGPipelineManager(
        db_manager=db_manager,
        k=2,
    )

    # 1. Similarity
    res_sim = manager.query_similar_documents("الذكاء", method=RetrievalMethod.SIMILARITY_SEARCH)
    assert len(res_sim) == 2

    # 2. MMR
    res_mmr = manager.query_similar_documents("الذكاء", method=RetrievalMethod.MAX_MARGINAL_RELEVANCE)
    assert len(res_mmr) == 2

    # 3. Hybrid
    res_hyb = manager.query_similar_documents("الذكاء", method=RetrievalMethod.HYBRID)
    assert len(res_hyb) <= 2

    # 4. Contextual (with mock LLM provider)
    manager.llm_provider = MagicMock()
    manager.llm_provider.get_llm.return_value = MagicMock()
    res_ctx = manager.query_similar_documents("الذكاء", method=RetrievalMethod.CONTEXTUAL)
    assert len(res_ctx) == 2


def test_rag_pipeline_manager_generate_answer_forwards_overrides():
    """generate_answer()'s k/retrieve_method overrides must reach
    generation_pipeline.generate_response() as explicit call arguments — never
    via mutating shared manager state (that mutation used to race under
    concurrent ask() calls)."""
    db_manager = MockDBManager()
    manager = RAGPipelineManager(
        db_manager=db_manager,
        k=2,
        retrieve_method=RetrievalMethod.SIMILARITY_SEARCH,
    )
    manager.generation_pipeline = MagicMock()
    manager.generation_pipeline.generate_response.return_value = {"answer": "ok"}

    manager.generate_answer("query", k=7, retrieve_method=RetrievalMethod.HYBRID)

    manager.generation_pipeline.generate_response.assert_called_once_with(
        "query", k=7, retrieve_method=RetrievalMethod.HYBRID
    )
    # Constructed manager state itself is untouched by the call.
    assert manager.k == 2
    assert manager.retrieve_method == RetrievalMethod.SIMILARITY_SEARCH


def test_rag_pipeline_manager_generate_answer_defaults_to_none_overrides():
    """Omitting overrides forwards None, letting downstream layers fall back
    to their own constructed defaults (unchanged behavior for existing callers)."""
    db_manager = MockDBManager()
    manager = RAGPipelineManager(db_manager=db_manager, k=2)
    manager.generation_pipeline = MagicMock()
    manager.generation_pipeline.generate_response.return_value = {"answer": "ok"}

    manager.generate_answer("query")

    manager.generation_pipeline.generate_response.assert_called_once_with(
        "query", k=None, retrieve_method=None
    )


def test_rag_pipeline_manager_unsupported_method():
    db_manager = MockDBManager()
    manager = RAGPipelineManager(db_manager=db_manager)

    with pytest.raises(ValueError):
        manager.query_similar_documents("query", method="non_existent_method")
