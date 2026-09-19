"""
Test suite for the Generation module (pytest).

No optional SDKs or real LLM/DB are required: the LLM provider and pipeline
manager are mocked. This validates AnswerGenerator, ContextRelevanceChecker,
DocumentRetriever, and RAGGenerationPipeline (vector path, adaptive web-search
fallback, hallucination check, and the None-query guard).
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from Generation.AnswerGenerator import AnswerGenerator
from Generation.ContextRelevanceChecker import ContextRelevanceChecker
from Generation.DocumentRetriever import DocumentRetriever
from Generation.RAGGenerationPipeline import RAGGenerationPipeline
from Muffakir.exceptions import GenerationError


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------
class _MockLLMResponse:
    def __init__(self, content):
        self.content = content


class _MockLLM:
    """Mock LLM that returns a fixed response, or raises on demand."""
    def __init__(self, response_text="test answer", raise_exc=None):
        self._response_text = response_text
        self._raise = raise_exc
        self.invoke_calls = 0

    def get_llm(self):
        return self

    def invoke(self, prompt):
        self.invoke_calls += 1
        if self._raise:
            raise self._raise
        return _MockLLMResponse(self._response_text)


class _MockPromptManager:
    """Mock prompt manager with configurable language and templates."""
    def __init__(self, language="ar", templates=None):
        self.language = language
        self._templates = templates or {
            "generation": "Context: {context}\nQuestion: {question}\nAnswer:",
            "context_relevance": "Q: {question}\nC: {context}\nVerdict:",
        }

    def get_prompt(self, name):
        return self._templates.get(name, "test prompt {context} {question}")


class _MockPipelineManager:
    """Mock RAGPipelineManager with configurable documents."""
    def __init__(self, docs=None):
        self._docs = docs or []
        self.query_calls = 0

    def query_similar_documents(self, query, k, method=None):
        self.query_calls += 1
        self.last_method = method
        return self._docs[:k]


# ---------------------------------------------------------------------------
# AnswerGenerator
# ---------------------------------------------------------------------------
def test_answer_generator_str_context():
    llm = _MockLLM("my answer")
    gen = AnswerGenerator(llm, _MockPromptManager())
    result = gen.generate_answer("what is X?", "some context string")
    assert result == "my answer"
    assert llm.invoke_calls == 1


def test_answer_generator_document_list():
    llm = _MockLLM("answer from docs")
    gen = AnswerGenerator(llm, _MockPromptManager())
    docs = [Document(page_content="doc1 content", metadata={}),
            Document(page_content="doc2 content", metadata={})]
    result = gen.generate_answer("what?", docs)
    assert result == "answer from docs"
    assert llm.invoke_calls == 1


def test_answer_generator_llm_failure_raises_generation_error():
    """LLM failure raises a typed GenerationError instead of a fake answer."""
    llm = _MockLLM(raise_exc=RuntimeError("LLM down"))
    gen = AnswerGenerator(llm, _MockPromptManager(language="ar"))
    with pytest.raises(GenerationError) as excinfo:
        gen.generate_answer("what?", "context")
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_answer_generator_fallback_prompt():
    """If the prompt template fails to format, a fallback prompt is used."""
    llm = _MockLLM("fallback answer")
    pm = _MockPromptManager(templates={"generation": "bad {nonexistent} template"})
    gen = AnswerGenerator(llm, pm)
    result = gen.generate_answer("what?", "ctx")
    assert result == "fallback answer"


# ---------------------------------------------------------------------------
# ContextRelevanceChecker
# ---------------------------------------------------------------------------
def test_context_relevance_relevant():
    llm = _MockLLM("relevant")
    checker = ContextRelevanceChecker(llm, _MockPromptManager())
    assert checker.is_relevant("question", "context text") is True


def test_context_relevance_not_relevant():
    llm = _MockLLM("not_relevant")
    checker = ContextRelevanceChecker(llm, _MockPromptManager())
    assert checker.is_relevant("question", "context text") is False


def test_context_relevance_empty_context():
    """Empty context returns False (no info to answer)."""
    checker = ContextRelevanceChecker(_MockLLM(), _MockPromptManager())
    assert checker.is_relevant("question", "") is False


def test_context_relevance_failure_defaults_true():
    """On LLM failure, defaults to True (avoid unnecessary web search)."""
    llm = _MockLLM(raise_exc=RuntimeError("LLM error"))
    checker = ContextRelevanceChecker(llm, _MockPromptManager())
    assert checker.is_relevant("question", "context") is True


# ---------------------------------------------------------------------------
# DocumentRetriever
# ---------------------------------------------------------------------------
def test_retriever_single_query():
    docs = [Document(page_content="doc1", metadata={"source": "a.pdf", "chunk_id": 1}),
            Document(page_content="doc2", metadata={"source": "b.pdf", "chunk_id": 2})]
    pm = _MockPipelineManager(docs)
    retriever = DocumentRetriever(pm)
    results = retriever.retrieve_documents("query", k=2)
    assert len(results) == 2
    assert results[0]["page_content"] == "doc1"
    assert results[0]["metadata"]["chunk_id"] == 1


def test_retriever_forwards_method_override():
    """method must reach pipeline_manager.query_similar_documents() as an
    explicit argument (not via shared mutable state)."""
    from Muffakir.Enums import RetrievalMethod

    docs = [Document(page_content="doc1", metadata={"source": "a.pdf"})]
    pm = _MockPipelineManager(docs)
    retriever = DocumentRetriever(pm)

    retriever.retrieve_documents("query", k=1, method=RetrievalMethod.HYBRID)
    assert pm.last_method == RetrievalMethod.HYBRID

    retriever.retrieve_documents(["q1", "q2"], k=1, method=RetrievalMethod.SIMILARITY_SEARCH)
    assert pm.last_method == RetrievalMethod.SIMILARITY_SEARCH


def test_retriever_metadata_preserved():
    """P1: full metadata dict is preserved (not just 'source')."""
    docs = [Document(page_content="text", metadata={"source": "x.pdf", "chunk_id": 42, "page": 3})]
    pm = _MockPipelineManager(docs)
    retriever = DocumentRetriever(pm)
    results = retriever.retrieve_documents("query", k=1)
    assert results[0]["metadata"]["chunk_id"] == 42
    assert results[0]["metadata"]["page"] == 3


def test_retriever_multi_query_dedup():
    """Multi-query expansion deduplicates by page_content."""
    doc = Document(page_content="shared content", metadata={"source": "a.pdf"})
    pm = _MockPipelineManager([doc])
    retriever = DocumentRetriever(pm)
    results = retriever.retrieve_documents(["q1", "q2"], k=1)
    assert len(results) == 1


def test_retriever_format_documents():
    retriever = DocumentRetriever(_MockPipelineManager())
    items = [{"page_content": "text", "metadata": {"source": "x.pdf", "chunk_id": 5}}]
    docs = retriever.format_documents(items)
    assert len(docs) == 1
    assert docs[0].page_content == "text"
    assert docs[0].metadata["chunk_id"] == 5


# ---------------------------------------------------------------------------
# RAGGenerationPipeline
# ---------------------------------------------------------------------------
def test_pipeline_vector_path():
    """Standard vector DB path: retrieves, generates, returns docs."""
    docs = [Document(page_content="relevant content", metadata={"source": "a.pdf"})]
    pm_mgr = _MockPipelineManager(docs)
    pipe = RAGGenerationPipeline(
        pipeline_manager=pm_mgr, llm_provider=_MockLLM("generated answer"),
        prompt_manager=_MockPromptManager(), query_transformer=None, hallucination=None, k=1,
    )
    result = pipe.generate_response("question")

    assert result["answer"] == "generated answer"
    assert result["context_source"] == "vector_db"
    assert len(result["retrieved_documents"]) == 1
    assert result["retrieved_documents"][0] == "relevant content"


def test_pipeline_observability_records_success_and_recovered_llm_problem():
    import queue

    from Trace.observability import observation_context

    docs = [Document(page_content="relevant content", metadata={})]
    failing_grader_llm = _MockLLM(raise_exc=TimeoutError("grader timed out"))
    pipe = RAGGenerationPipeline(
        pipeline_manager=_MockPipelineManager(docs),
        llm_provider=failing_grader_llm,
        prompt_manager=_MockPromptManager(),
        query_transformer=None,
        hallucination=None,
        web_search_provider=MagicMock(),
        adaptive_web_search=True,
        k=1,
    )
    # Generation still needs to succeed after the relevance checker falls
    # back. Give the generator a separate successful test double.
    pipe.generator.llm_provider = _MockLLM("answer")

    events = queue.Queue()
    with observation_context(events, trial_id=2, sample_index=4, attempt=1):
        result = pipe.generate_response("question")

    assert result["answer"] == "answer"
    operations = []
    while not events.empty():
        kind, event = events.get_nowait()
        assert kind == "operation"
        operations.append(event)
    by_stage = {event["stage"]: event for event in operations}
    assert by_stage["vector_search"]["outcome"] == "success"
    assert by_stage["relevance_check"]["outcome"] == "error"
    assert by_stage["relevance_check"]["recovery"] == "fallback"
    assert by_stage["generation"]["outcome"] == "success"
    assert by_stage["relevance_check"]["trial_id"] == 2


def test_pipeline_generate_response_forwards_k_and_method():
    """generate_response()'s k/retrieve_method overrides must reach
    DocumentRetriever.retrieve_documents() as explicit call arguments, and an
    omitted k must fall back to the pipeline's own constructed self.k."""
    from Muffakir.Enums import RetrievalMethod

    docs = [Document(page_content="relevant content", metadata={"source": "a.pdf"})]
    pm_mgr = _MockPipelineManager(docs)
    pipe = RAGGenerationPipeline(
        pipeline_manager=pm_mgr, llm_provider=_MockLLM("generated answer"),
        prompt_manager=_MockPromptManager(), query_transformer=None, hallucination=None, k=1,
    )

    pipe.generate_response("question", k=5, retrieve_method=RetrievalMethod.HYBRID)
    assert pm_mgr.last_method == RetrievalMethod.HYBRID

    # Omitted k/method fall back to the pipeline's constructed defaults.
    pipe.generate_response("question")
    assert pm_mgr.last_method is None


def test_pipeline_web_search_fallback():
    """Adaptive: context graded not_relevant → web search → cleared retrieved_documents."""
    docs = [Document(page_content="irrelevant", metadata={"source": "a.pdf"})]
    pm_mgr = _MockPipelineManager(docs)

    class MockRelevanceChecker:
        def is_relevant(self, query, documents):
            return False

    from WebSearch.models import WebSearchResult
    web_provider = MagicMock()
    web_provider.search.return_value = WebSearchResult(
        content="web content", sources=[{"title": "Web Result", "url": "http://example.com"}]
    )

    pipe = RAGGenerationPipeline(
        pipeline_manager=pm_mgr, llm_provider=_MockLLM("web answer"),
        prompt_manager=_MockPromptManager(), query_transformer=None, hallucination=None,
        reranker=None, web_search_provider=web_provider,
        adaptive_web_search=True, k=1,
    )
    pipe.relevance_checker = MockRelevanceChecker()
    result = pipe.generate_response("question")

    assert result["answer"] == "web answer"
    assert result["context_source"] == "web_search"
    assert result["used_context"]
    assert len(result["retrieved_documents"]) == 0  # P1 fix: cleared
    assert len(result["web_sources"]) == 1
    assert result["stage_timings_ms"]["relevance_check_ms"] is not None
    assert result["stage_timings_ms"]["web_search_ms"] is not None
    assert result["pipeline_latency_ms"] is not None


def test_pipeline_hallucination_check():
    """Hallucination check is applied to the answer."""
    hallucination = MagicMock()
    hallucination.check_answer.return_value = "cleaned answer"
    pipe = RAGGenerationPipeline(
        pipeline_manager=_MockPipelineManager([]), llm_provider=_MockLLM("raw answer"),
        prompt_manager=_MockPromptManager(), query_transformer=None,
        hallucination=hallucination, k=1,
    )
    result = pipe.generate_response("question")
    assert result["answer"] == "cleaned answer"
    assert hallucination.check_answer.call_count == 1


def test_pipeline_none_query_guard():
    """P2: query transformer returning None falls back to original query."""
    bad_transformer = MagicMock()
    bad_transformer.transform_query.return_value = None
    docs = [Document(page_content="content", metadata={"source": "a.pdf"})]
    pm_mgr = _MockPipelineManager(docs)
    pipe = RAGGenerationPipeline(
        pipeline_manager=pm_mgr, llm_provider=_MockLLM("answer"),
        prompt_manager=_MockPromptManager(), query_transformer=bad_transformer,
        hallucination=None, k=1,
    )
    result = pipe.generate_response("original question")
    assert result["answer"] == "answer"


def test_pipeline_no_basicconfig():
    """P0 regression: no basicConfig called during pipeline operation."""
    import logging
    root_before = list(logging.getLogger().handlers)
    pipe = RAGGenerationPipeline(
        pipeline_manager=_MockPipelineManager([]), llm_provider=_MockLLM(),
        prompt_manager=_MockPromptManager(), query_transformer=None, hallucination=None, k=1,
    )
    pipe.generate_response("question")
    assert list(logging.getLogger().handlers) == root_before


# ---------------------------------------------------------------------------
# Per-stage timing (stage_timings_ms)
# ---------------------------------------------------------------------------
def test_pipeline_reports_stage_timings_for_retrieval_and_generation():
    docs = [Document(page_content="relevant content", metadata={"source": "a.pdf"})]
    pm_mgr = _MockPipelineManager(docs)
    pipe = RAGGenerationPipeline(
        pipeline_manager=pm_mgr, llm_provider=_MockLLM("generated answer"),
        prompt_manager=_MockPromptManager(), query_transformer=None, hallucination=None, k=1,
    )
    result = pipe.generate_response("question")

    timings = result["stage_timings_ms"]
    assert timings["retrieval_ms"] is not None and timings["retrieval_ms"] >= 0.0
    assert timings["generation_ms"] is not None and timings["generation_ms"] >= 0.0
    # Stages that weren't configured for this pipeline report None, not 0.0 --
    # a trace reader must be able to tell "skipped" apart from "ran instantly".
    assert timings["query_transform_ms"] is None
    assert timings["rerank_ms"] is None
    assert timings["hallucination_check_ms"] is None


def test_pipeline_reports_query_transform_timing_when_configured():
    transformer = MagicMock()
    transformer.transform_query.return_value = "rewritten question"
    transformer.name = "rewrite"
    docs = [Document(page_content="content", metadata={"source": "a.pdf"})]
    pipe = RAGGenerationPipeline(
        pipeline_manager=_MockPipelineManager(docs), llm_provider=_MockLLM("answer"),
        prompt_manager=_MockPromptManager(), query_transformer=transformer, hallucination=None, k=1,
    )
    result = pipe.generate_response("question")
    assert result["stage_timings_ms"]["query_transform_ms"] is not None
    assert result["transformed_query"] == "rewritten question"
    assert result["query_transform_strategy"] == "rewrite"


def test_pipeline_reports_rerank_timing_when_configured():
    reranker = MagicMock()
    docs = [Document(page_content="content", metadata={"source": "a.pdf"})]
    reranker.rerank.return_value = docs
    pipe = RAGGenerationPipeline(
        pipeline_manager=_MockPipelineManager(docs), llm_provider=_MockLLM("answer"),
        prompt_manager=_MockPromptManager(), query_transformer=None, hallucination=None,
        reranker=reranker, k=1,
    )
    result = pipe.generate_response("question")
    assert result["stage_timings_ms"]["rerank_ms"] is not None


def test_pipeline_reports_hallucination_check_timing_when_configured():
    hallucination = MagicMock()
    hallucination.check_answer.return_value = "cleaned answer"
    pipe = RAGGenerationPipeline(
        pipeline_manager=_MockPipelineManager([]), llm_provider=_MockLLM("raw answer"),
        prompt_manager=_MockPromptManager(), query_transformer=None,
        hallucination=hallucination, k=1,
    )
    result = pipe.generate_response("question")
    assert result["stage_timings_ms"]["hallucination_check_ms"] is not None


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
