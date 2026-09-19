"""Small, dependency-light helpers for per-query pipeline telemetry."""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Union

from Trace.context import get_current_sample
from Trace.observability import observe_stage, provider_identity


PIPELINE_STAGE_KEYS = (
    "query_transform_ms",
    "query_embedding_ms",
    "vector_search_ms",
    "rerank_ms",
    "relevance_check_ms",
    "web_search_ms",
    "generation_ms",
    "hallucination_check_ms",
)


def empty_stage_timings() -> Dict[str, Optional[float]]:
    """Return a fresh timing map where skipped stages are explicitly absent."""
    return {key: None for key in PIPELINE_STAGE_KEYS}


@dataclass
class RetrievalTelemetryResult:
    """Internal retrieval result used by evaluators without changing the list API."""

    documents: List[Any] = field(default_factory=list)
    pipeline_latency_ms: float = 0.0
    stage_timings_ms: Dict[str, Optional[float]] = field(default_factory=empty_stage_timings)
    transformed_query: Optional[Union[str, List[str]]] = None
    query_transform_strategy: Optional[str] = None


def _embedding_total_ms(embedding_provider: Any) -> Optional[float]:
    timing = getattr(embedding_provider, "timing", None)
    if timing is None:
        return None

    sample = get_current_sample()
    if sample is not None and hasattr(timing, "get_sample_total_ms"):
        return float(timing.get_sample_total_ms(*sample))

    if hasattr(timing, "get_totals"):
        return float((timing.get_totals() or {}).get("total_ms", 0.0))
    return None


def run_timed_retrieval(
    query: str,
    effective_k: int,
    retrieve_one: Callable[[str, int], List[Any]],
    *,
    query_transformer: Optional[Any] = None,
    rerank: Optional[Callable[[str, List[Any]], List[Any]]] = None,
    embedding_provider: Optional[Any] = None,
) -> RetrievalTelemetryResult:
    """Run transform -> retrieval -> rerank and return documents plus timings."""
    pipeline_start = time.perf_counter()
    timings = empty_stage_timings()
    search_query: Union[str, List[str]] = query
    transformed_query: Optional[Union[str, List[str]]] = None
    query_transform_strategy: Optional[str] = None

    if query_transformer is not None:
        stage_start = time.perf_counter()
        transform_provider, transform_model = provider_identity(
            getattr(query_transformer, "llm_provider", None)
        )
        with observe_stage(
            "query_transform", "llm",
            provider=transform_provider, model=transform_model,
        ):
            transformed = query_transformer.transform_query(query)
        timings["query_transform_ms"] = (time.perf_counter() - stage_start) * 1000.0
        if transformed:
            search_query = transformed
        transformed_query = search_query
        strategy = getattr(query_transformer, "name", None)
        query_transform_strategy = strategy if isinstance(strategy, str) else None

    if isinstance(search_query, list):
        primary_query = str(search_query[0]) if search_query else query
        search_queries = [str(item) for item in search_query if str(item).strip()]
    else:
        primary_query = str(search_query)
        search_queries = [primary_query]

    embedding_before = _embedding_total_ms(embedding_provider)
    retrieval_start = time.perf_counter()
    effective_queries = search_queries or [query]
    with observe_stage("vector_search", "vector_db"):
        if len(effective_queries) == 1:
            # Preserve the legacy single-query result exactly, including ordering.
            documents = list(retrieve_one(effective_queries[0], effective_k) or [])
        else:
            documents = []
            seen_content = set()
            for sub_query in effective_queries:
                for document in retrieve_one(sub_query, effective_k) or []:
                    content = getattr(document, "page_content", str(document)).strip()
                    if content not in seen_content:
                        seen_content.add(content)
                        documents.append(document)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
    embedding_after = _embedding_total_ms(embedding_provider)

    if embedding_before is not None and embedding_after is not None:
        query_embedding_ms = max(embedding_after - embedding_before, 0.0)
        timings["query_embedding_ms"] = query_embedding_ms
        timings["vector_search_ms"] = max(retrieval_ms - query_embedding_ms, 0.0)
    else:
        # The backend did not expose embedding telemetry. The retrieval stage
        # is still measured accurately, but cannot be split further.
        timings["vector_search_ms"] = retrieval_ms

    if rerank is not None:
        stage_start = time.perf_counter()
        with observe_stage("rerank", "reranker"):
            documents = rerank(primary_query, documents)
        timings["rerank_ms"] = (time.perf_counter() - stage_start) * 1000.0

    return RetrievalTelemetryResult(
        documents=documents,
        pipeline_latency_ms=(time.perf_counter() - pipeline_start) * 1000.0,
        stage_timings_ms=timings,
        transformed_query=transformed_query,
        query_transform_strategy=query_transform_strategy,
    )
