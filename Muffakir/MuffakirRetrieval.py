from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from LLMProvider.LLMProvider import LLMProvider
from LLMProvider.parameters import resolve_parameters
from PromptManager.PromptManager import MuffakirPrompt
from QueryTransformer.QueryTransformer import QueryTransformer
from Reranker.Reranker import Reranker, rerank_documents
from Muffakir.Enums import (
    RETRIEVAL_MAPPING,
    RetrievalMethod,
    ProviderName,
    PROVIDER_MAPPING,
    resolve_provider_name,
)
from Muffakir.exceptions import ConfigurationError, MuffakirError, RetrievalError
from Muffakir.telemetry import RetrievalTelemetryResult, run_timed_retrieval

logger = logging.getLogger(__name__)


def _retrieve_methods_class():
    existing = globals().get("RetrieveMethods")
    if existing is not None:
        return existing
    from RAGPipeline.RetrieveMethods import RetrieveMethods as loaded

    globals()["RetrieveMethods"] = loaded
    return loaded


def __getattr__(name: str):
    if name == "RetrieveMethods":
        return _retrieve_methods_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class MuffakirRetrieval:
    """
    Retrieval-only RAG facade: exercises embedding/chunking/retrieval-method/
    reranking/query-transformation exactly like MuffakirRAG, but never builds
    or calls a main generation LLM, AnswerGenerator, or HallucinationsCheck.

    EvalRunner-compatible: implements get_similar_documents() with the same
    signature MuffakirRAG.get_similar_documents() has, so it is a drop-in
    `rag` for retrieval-metric-only evaluation.
    """

    def __init__(
        self,
        rag_config: Dict[str, Any],
        embedding_provider: Any,
        db_manager: Any,
        prompt_manager: Optional[MuffakirPrompt] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.config = rag_config
        self.embedding_provider = embedding_provider
        self.db_manager = db_manager

        from Muffakir.dependency_validation import validate_retrieval_dependencies

        validate_retrieval_dependencies(self.config)

        self.retriever = _retrieve_methods_class()(db_manager=self.db_manager)
        self.k = self.config.get("k", 5)
        self.fetch_k = self.config.get("fetch_k", 7)
        configured_method = self.config.get("retrieval_method", "similarity_search")
        self.retrieve_method = RETRIEVAL_MAPPING.get(configured_method)
        if self.retrieve_method is None:
            # Mirrors MuffakirRAG.__init__'s up-front rejection of an unknown
            # retrieval_method -- without this the dispatch below would silently
            # degrade any unrecognized value to similarity_search, so a typo'd
            # Composer sweep value would report scores for the wrong method.
            available_methods = ", ".join(RETRIEVAL_MAPPING.keys())
            raise ConfigurationError(
                f"Unsupported retrieval method: {configured_method}. "
                f"Available methods: {available_methods}"
            )

        self.prompt_manager = prompt_manager or MuffakirPrompt(
            language=self.config.get("language", "ar"),
            overrides=self.config.get("prompt_overrides"),
        )

        self.query_transformer: Optional[QueryTransformer] = None
        if self.config.get("query_transformer"):
            qt_provider = self.config.get("query_transform_llm_provider")
            qt_model = self.config.get("query_transform_llm_model")
            if not qt_provider or not qt_model:
                raise ConfigurationError(
                    "query_transformer is enabled in retrieval-only mode but "
                    "query_transform_llm_provider/query_transform_llm_model are "
                    "not set -- retrieval-only mode has no main LLM to fall "
                    "back to, unlike full-RAG mode. Set both explicitly."
                )
            qt_llm_provider = LLMProvider(
                parameters=resolve_parameters(self.config, "query_transform"),
                # Retrieval-only mode hides the main api_key field in ComposerUI
                # (no main LLM exists), so the query-transform LLM gets its own
                # dedicated credential field; fall back to the main api_key for
                # a full_rag-style config that supplies only that.
                api_key=self.config.get("query_transform_api_key") or self.config.get("api_key"),
                provider=PROVIDER_MAPPING.get(qt_provider, ProviderName.OPENAI),
                model=qt_model,
                temperature=self.config.get("llm_temperature", 0.0),
                max_tokens=self.config.get("llm_max_tokens", 4096),
                base_url=(
                    self.config.get("query_transform_base_url")
                    or self.config.get("llm_base_url")
                    or self.config.get("base_url")
                ),
            )
            self.query_transformer = QueryTransformer(
                llm_provider=qt_llm_provider,
                prompt_manager=self.prompt_manager,
                prompt="query_rewrite",
                strategy=self.config.get("query_transformer_strategy", "rewrite"),
            )

        if self.retrieve_method == RetrievalMethod.CONTEXTUAL and self.query_transformer is None:
            raise ConfigurationError(
                "retrieval_method='contextual' requires an LLM in retrieval-only "
                "mode. Enable query_transformer with query_transform_llm_provider/"
                "query_transform_llm_model, or choose a different retrieval_method."
            )

        self.reranker: Optional[Reranker] = None
        if self.config.get("reranking"):
            reranking_method = str(
                self.config.get("reranking_method", "semantic_similarity")
            ).strip().lower()
            reranker_llm_provider = (
                self.query_transformer.llm_provider
                if self.query_transformer
                else None
            )
            llm_override_requested = any(
                self.config.get(key)
                for key in (
                    "reranker_llm_provider",
                    "reranker_llm_model",
                    "reranker_llm_api_key",
                    "reranker_llm_base_url",
                )
            )
            if reranking_method in {"llm", "llm_reranker", "llm_based", "llm-based"}:
                if llm_override_requested:
                    provider = self.config.get("reranker_llm_provider")
                    model = self.config.get("reranker_llm_model")
                    if not provider or not model:
                        raise ConfigurationError(
                            "Dedicated LLM reranker requires reranker_llm_provider "
                            "and reranker_llm_model."
                        )
                    reranker_llm_provider = LLMProvider(
                        parameters=resolve_parameters(self.config, "reranker"),
                        api_key=self.config.get("reranker_llm_api_key"),
                        provider=resolve_provider_name(provider),
                        model=model,
                        temperature=self.config.get("llm_temperature", 0.0),
                        max_tokens=self.config.get("llm_max_tokens", 4096),
                        base_url=self.config.get("reranker_llm_base_url"),
                    )
                elif (
                    (reranker_llm_provider is None or self.config.get("reranker_llm_parameters") is not None)
                    and self.config.get("query_transform_llm_provider")
                    and self.config.get("query_transform_llm_model")
                ):
                    reranker_llm_provider = LLMProvider(
                        parameters=resolve_parameters(self.config, "reranker"),
                        api_key=(
                            self.config.get("query_transform_api_key")
                            or self.config.get("api_key")
                        ),
                        provider=resolve_provider_name(
                            self.config["query_transform_llm_provider"]
                        ),
                        model=self.config["query_transform_llm_model"],
                        temperature=self.config.get("llm_temperature", 0.0),
                        max_tokens=self.config.get("llm_max_tokens", 4096),
                        base_url=(
                            self.config.get("query_transform_base_url")
                            or self.config.get("llm_base_url")
                            or self.config.get("base_url")
                        ),
                    )
                elif reranker_llm_provider is None:
                    raise ConfigurationError(
                        "LLM reranking in retrieval-only mode requires a dedicated "
                        "reranker LLM or a configured query-transform LLM to reuse."
                    )
            self.reranker = Reranker(
                embedding_provider=self.embedding_provider,
                reranking_method=reranking_method,
                cross_encoder_model_name=self.config.get("reranking_model"),
                llm_provider=reranker_llm_provider,
                prompt_manager=self.prompt_manager,
                device=self.config.get("device", "auto"),
                remote_base_url=self.config.get("reranker_base_url"),
                remote_api_key=self.config.get("reranker_api_key"),
                remote_model=self.config.get("reranker_model"),
                remote_timeout=self.config.get("reranker_timeout_seconds", 30.0),
                remote_options=self.config.get("reranker_options"),
            )

    def _retrieve_one(self, search_query: str, effective_k: int) -> List[Document]:
        """Dispatch a single query to the configured retrieval method.

        Mirrors RAGPipelineManager.query_similar_documents' dispatch exactly.
        """
        if self.retrieve_method == RetrievalMethod.MAX_MARGINAL_RELEVANCE:
            return self.retriever.max_marginal_relevance_search(search_query, effective_k, self.fetch_k)
        if self.retrieve_method == RetrievalMethod.HYBRID:
            return self.retriever.hybrid_search(search_query, effective_k)
        if self.retrieve_method == RetrievalMethod.CONTEXTUAL:
            return self.retriever.contextual_search(
                query=search_query,
                k=effective_k,
                llm_provider=self.query_transformer.llm_provider if self.query_transformer else None,
            )
        return self.retriever.similarity_search(search_query, effective_k)

    def get_similar_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Transform (if enabled) -> retrieve -> rerank (if enabled). No LLM
        generation call anywhere in this path."""
        return self.get_similar_documents_with_trace(query, k=k).documents

    def get_similar_documents_with_trace(
        self, query: str, k: Optional[int] = None
    ) -> RetrievalTelemetryResult:
        """Telemetry-aware retrieval used by EvalRunner.

        The existing ``get_similar_documents`` method remains list-returning for
        public API compatibility.
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided for document retrieval.")
            return RetrievalTelemetryResult()

        effective_k = k if k is not None else self.k

        try:
            rerank = None
            if self.reranker:
                rerank = lambda rerank_query, documents: rerank_documents(
                    self.reranker, rerank_query, documents
                )
            return run_timed_retrieval(
                query,
                effective_k,
                self._retrieve_one,
                query_transformer=self.query_transformer,
                rerank=rerank,
                embedding_provider=self.embedding_provider,
            )

        except MuffakirError:
            # Already a typed error (ConfigurationError from this class's own
            # fail-fast checks, or a provider error like a rate-limit/timeout
            # raised by the query transformer's LLM call) -- propagate as-is
            # so its retryable/error_code semantics aren't lost by re-wrapping.
            raise
        except Exception as e:
            # Mirrors MuffakirRAG.get_similar_documents: an unexpected failure
            # must surface as a typed MuffakirError so EvalRunner aborts the
            # trial with an error code instead of recording a bogus 0.0 score.
            self.logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            raise RetrievalError(f"Failed to retrieve documents: {e}") from e

    def ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """No answer generation in retrieval-only mode. Fail loudly rather
        than an ambiguous AttributeError, matching MuffakirSearch's existing
        fail-fast convention for its own unsupported operation."""
        raise ConfigurationError(
            "MuffakirRetrieval has no answer generation -- do not request "
            "faithfulness/answer_correctness/llm_judge_rating metrics for a "
            "retrieval-only run. Use recall/precision/mrr/ndcg instead."
        )
