from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Union

if TYPE_CHECKING:
    # Deferred: Generation/__init__.py eagerly imports this module, and
    # Muffakir/__init__.py's eager chain imports back into Generation — a
    # module-level import here would be circular. `from __future__ import
    # annotations` above already makes annotations lazy strings, so this is
    # only needed for static type checkers, not runtime.
    from Muffakir.Enums import RetrievalMethod
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt
from Generation.AnswerGenerator import AnswerGenerator
from Generation.ContextRelevanceChecker import ContextRelevanceChecker
from Generation.DocumentRetriever import DocumentRetriever
from HallucinationsCheck.HallucinationsCheck import HallucinationsCheck
from QueryTransformer.QueryTransformer import QueryTransformer
from Reranker.Reranker import Reranker, rerank_documents
from WebSearch.base import BaseWebSearchProvider
from Muffakir.telemetry import _embedding_total_ms, empty_stage_timings
from Trace.observability import observe_stage, provider_identity

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


class RAGGenerationPipeline:
    """Orchestrates the generation phase of the RAG pipeline.

    Flow: query-transform → retrieve → rerank → (adaptive: grade relevance →
    web-search fallback) → generate → hallucination-check.

    All collaborators (LLM, prompt manager, query transformer, reranker,
    hallucination checker, web search provider) are injected via the
    constructor for testability and decoupling.
    """

    def __init__(
        self,
        pipeline_manager,
        llm_provider: LLMProvider,
        prompt_manager: MuffakirPrompt,
        query_transformer: Optional[QueryTransformer],
        hallucination: Optional[HallucinationsCheck],
        reranker: Optional[Reranker] = None,
        web_search_provider: Optional[BaseWebSearchProvider] = None,
        adaptive_web_search: bool = False,
        k: int = 5,
    ):
        self.pipeline_manager = pipeline_manager
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.query_transformer = query_transformer
        self.hallucination = hallucination
        self.reranker = reranker
        self.web_search_provider = web_search_provider
        self.adaptive_web_search = adaptive_web_search
        self.k = k
        self.logger = logging.getLogger(__name__)
        self._llm_provider_name, self._llm_model_name = provider_identity(llm_provider)

        self.generator = AnswerGenerator(llm_provider, prompt_manager)
        self.relevance_checker = None
        if self.adaptive_web_search and self.web_search_provider and prompt_manager:
            self.relevance_checker = ContextRelevanceChecker(llm_provider, prompt_manager)

    def _process_vector_db_query(
        self,
        query: str,
        k: Optional[int] = None,
        retrieve_method: Optional[RetrievalMethod] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: transform → retrieve → rerank → (adaptive) → generate → check.

        Each stage is timed with time.perf_counter() into the returned dict's
        "stage_timings_ms" -- a single central seam (this method), so no
        change is needed in DocumentRetriever/AnswerGenerator/etc. themselves.
        A stage that's skipped (e.g. no query_transformer configured) reports
        None rather than 0.0, so a trace reader can tell "not run" apart from
        "ran instantly".
        """
        pipeline_start = time.perf_counter()
        original_query = query
        transformed_query = None
        query_transform_strategy = None
        effective_k = k if k is not None else self.k
        stage_timings_ms = empty_stage_timings()
        # Kept for older trace readers. New readers use the embedding/vector split.
        stage_timings_ms["retrieval_ms"] = None

        if self.query_transformer:
            _start = time.perf_counter()
            query_llm = getattr(self.query_transformer, "llm_provider", None)
            query_provider, query_model = provider_identity(query_llm)
            with observe_stage(
                "query_transform", "llm", provider=query_provider, model=query_model
            ):
                query = self._query_transform(query)
            stage_timings_ms["query_transform_ms"] = (time.perf_counter() - _start) * 1000.0
            # Guard against a transformer that returns None.
            if query is None:
                self.logger.warning("Query transformer returned None; using original query.")
                query = original_query
            transformed_query = query
            strategy = getattr(self.query_transformer, "name", None)
            query_transform_strategy = strategy if isinstance(strategy, str) else None

        # Handle list of queries (e.g. Multi-Query Expansion) vs single query string
        if isinstance(query, list):
            primary_query = query[0] if query else original_query
            search_queries = query
        else:
            primary_query = query
            search_queries = query

        retriever = DocumentRetriever(self.pipeline_manager)
        embedding_provider = getattr(self.pipeline_manager, "embedding_provider", None)
        embedding_before = _embedding_total_ms(embedding_provider)
        _start = time.perf_counter()
        with observe_stage("vector_search", "vector_db"):
            retrieval_result = retriever.retrieve_documents(
                search_queries, effective_k, method=retrieve_method
            )
            formatted_documents = retriever.format_documents(retrieval_result)
        retrieval_ms = (time.perf_counter() - _start) * 1000.0
        stage_timings_ms["retrieval_ms"] = retrieval_ms
        embedding_after = _embedding_total_ms(embedding_provider)
        if embedding_before is not None and embedding_after is not None:
            query_embedding_ms = max(embedding_after - embedding_before, 0.0)
            stage_timings_ms["query_embedding_ms"] = query_embedding_ms
            stage_timings_ms["vector_search_ms"] = max(retrieval_ms - query_embedding_ms, 0.0)
        else:
            stage_timings_ms["vector_search_ms"] = retrieval_ms

        if self.reranker:
            _start = time.perf_counter()
            reranker_name = type(getattr(self.reranker, "_reranker", self.reranker)).__name__
            with observe_stage("rerank", "reranker", provider=reranker_name):
                formatted_documents = self._rerank_documents(primary_query, formatted_documents)
            stage_timings_ms["rerank_ms"] = (time.perf_counter() - _start) * 1000.0

        context_source = "vector_db"
        web_sources: List[Dict[str, str]] = []
        generation_context: Union[List[Document], str] = formatted_documents

        # Adaptive RAG Mode B: grade relevance before generation; fall back to web search
        if (
            self.adaptive_web_search
            and self.web_search_provider
            and self.relevance_checker
        ):
            _start = time.perf_counter()
            with observe_stage(
                "relevance_check",
                "llm",
                provider=self._llm_provider_name,
                model=self._llm_model_name,
            ):
                is_relevant = self.relevance_checker.is_relevant(
                    primary_query, formatted_documents
                )
            stage_timings_ms["relevance_check_ms"] = (time.perf_counter() - _start) * 1000.0
            if not is_relevant:
                self.logger.info(
                    "Retrieved context graded not_relevant; falling back to web search"
                )
                _start = time.perf_counter()
                web_result = None
                web_error = None
                with observe_stage(
                    "web_search",
                    "web_search",
                    provider=type(self.web_search_provider).__name__,
                ) as web_observation:
                    try:
                        web_result = self.web_search_provider.search(primary_query)
                    except Exception as e:
                        web_error = e
                        web_observation.mark_error(e, "fallback")
                if web_error is None and web_result is not None:
                    stage_timings_ms["web_search_ms"] = (time.perf_counter() - _start) * 1000.0
                    generation_context = web_result.content or ""
                    web_sources = web_result.sources or []
                    context_source = "web_search"
                    # When using web search, don't report the vector DB docs
                    # that were graded as not relevant — they weren't used.
                    formatted_documents = []
                else:
                    # The vector context fallback is intentional and does not
                    # alter execution behavior, but it remains observable.
                    self.logger.error(
                        f"Adaptive web search failed, using vector context: {web_error}",
                        exc_info=True,
                    )
                    context_source = "vector_db"
                    generation_context = formatted_documents

        _start = time.perf_counter()
        with observe_stage(
            "generation",
            "llm",
            provider=self._llm_provider_name,
            model=self._llm_model_name,
        ):
            answer = self.generator.generate_answer(primary_query, generation_context)
        stage_timings_ms["generation_ms"] = (time.perf_counter() - _start) * 1000.0

        if self.hallucination:
            _start = time.perf_counter()
            with observe_stage(
                "hallucination_check",
                "llm",
                provider=self._llm_provider_name,
                model=self._llm_model_name,
            ):
                answer = self._hallucination_check(answer, context=generation_context, query=primary_query)
            stage_timings_ms["hallucination_check_ms"] = (time.perf_counter() - _start) * 1000.0

        return {
            "answer": answer,
            # Keep this payload explicit: the caller's question is already
            # available as ``question`` in evaluation traces, while this value
            # records exactly what was supplied to retrieval/generation.
            "transformed_query": transformed_query,
            "query_transform_strategy": query_transform_strategy,
            "retrieved_documents": [doc.page_content for doc in formatted_documents],
            "source_metadata": [doc.metadata for doc in formatted_documents],
            "context_source": context_source,
            "web_sources": web_sources,
            "used_context": (
                generation_context
                if isinstance(generation_context, str)
                else "\n\n".join(doc.page_content for doc in generation_context)
            ),
            "stage_timings_ms": stage_timings_ms,
            "pipeline_latency_ms": (time.perf_counter() - pipeline_start) * 1000.0,
        }

    def _query_transform(self, query: str):
        """Transform the query using the query transformer."""
        if self.query_transformer:
            return self.query_transformer.transform_query(query)

    def _rerank_documents(self, query: str, formatted_documents: List[Document]) -> List[Document]:
        """Rerank the retrieved documents if a reranker is provided."""
        return rerank_documents(self.reranker, query, formatted_documents)

    def _hallucination_check(self, answer: str, context: str = "", query: str = "") -> str:
        """Check for hallucinations in the generated answer, grounded in the retrieved context."""
        if self.hallucination:
            return self.hallucination.check_answer(answer, context=context, query=query)
        return answer

    def generate_response(
        self,
        query: str,
        k: Optional[int] = None,
        retrieve_method: Optional[RetrievalMethod] = None,
    ) -> Dict[str, Any]:
        """Main entry point to generate a response.

        :param k: override the top-k count for this call only.
        :param retrieve_method: override the retrieval method for this call only.
        """
        return self._process_vector_db_query(query, k=k, retrieve_method=retrieve_method)
