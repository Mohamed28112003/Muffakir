from langchain_core.documents import Document

from Muffakir.telemetry import run_timed_retrieval


class _Timing:
    def __init__(self):
        self.total = 0.0

    def get_totals(self):
        return {"total_ms": self.total, "count": 1}


class _EmbeddingProvider:
    def __init__(self):
        self.timing = _Timing()


def test_timed_retrieval_separates_retrieval_embedding_from_reranking_work():
    provider = _EmbeddingProvider()
    docs = [Document(page_content="one")]

    def retrieve(query, k):
        provider.timing.total += 5.0
        return docs

    def rerank(query, documents):
        # Semantic reranking can use the same embedding provider. This work
        # must stay inside rerank_ms rather than query_embedding_ms.
        provider.timing.total += 50.0
        return documents

    result = run_timed_retrieval(
        "question", 3, retrieve, rerank=rerank, embedding_provider=provider
    )

    assert result.documents == docs
    assert result.stage_timings_ms["query_embedding_ms"] == 5.0
    assert result.stage_timings_ms["vector_search_ms"] >= 0.0
    assert result.stage_timings_ms["rerank_ms"] is not None
    assert result.stage_timings_ms["query_transform_ms"] is None
    assert result.pipeline_latency_ms >= 0.0


def test_timed_retrieval_preserves_a_genuine_zero_and_marks_skipped_stages_none():
    result = run_timed_retrieval("question", 1, lambda query, k: [])

    assert result.stage_timings_ms["query_embedding_ms"] is None
    assert result.stage_timings_ms["vector_search_ms"] >= 0.0
    assert result.stage_timings_ms["rerank_ms"] is None
    assert result.stage_timings_ms["generation_ms"] is None


def test_timed_retrieval_retains_the_selected_query_transformation_output():
    class _Transformer:
        name = "multi_query"

        def transform_query(self, query):
            return [query, "expanded retrieval query"]

    queries = []
    result = run_timed_retrieval(
        "original question", 1,
        lambda query, k: queries.append(query) or [],
        query_transformer=_Transformer(),
    )

    assert result.transformed_query == ["original question", "expanded retrieval query"]
    assert result.query_transform_strategy == "multi_query"
    assert queries == ["original question", "expanded retrieval query"]
