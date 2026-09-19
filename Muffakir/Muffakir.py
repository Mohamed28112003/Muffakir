from __future__ import annotations

from typing import Dict, Any, Optional, List, Union
import logging
import os
from pathlib import Path

from LLMProvider.LLMProvider import LLMProvider
from LLMProvider.parameters import resolve_parameters
from Embedding.EmbeddingProvider import EmbeddingProvider
from PromptManager.PromptManager import MuffakirPrompt
from QueryTransformer.QueryTransformer import QueryTransformer
from HallucinationsCheck.HallucinationsCheck import HallucinationsCheck
from Muffakir.Enums import (
    ProviderName,
    RetrievalMethod,
    PROVIDER_MAPPING,
    RETRIEVAL_MAPPING,
    resolve_provider_name,
    resolve_retrieval_method,
)
from Muffakir.constants import DEFAULT_RAG_CONFIG
from Muffakir.exceptions import (
    ConfigurationError,
    MuffakirError,
    GenerationError,
    RetrievalError,
    DocumentIndexError,
)

try:
    from langchain_core.documents import Document
except ImportError:
    # pyrefly: ignore [missing-import]
    from langchain.schema import Document

from Reranker.Reranker import Reranker
from Muffakir.telemetry import RetrievalTelemetryResult, run_timed_retrieval

logger = logging.getLogger(__name__)

_LAZY_COMPONENTS = {
    "ChunkingAndProcessing": ("TextProcessor.ChunkingAndProcessing", "ChunkingAndProcessing"),
    "RAGPipelineManager": ("RAGPipeline.RAGPipelineManager", "RAGPipelineManager"),
}


def _component(name: str):
    value = globals().get(name)
    if value is not None:
        return value
    from importlib import import_module

    module_name, attribute = _LAZY_COMPONENTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __getattr__(name: str):
    if name in _LAZY_COMPONENTS:
        return _component(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class MuffakirRAG:
    """
    Multilingual RAG (Retrieval-Augmented Generation) library.

    A configurable RAG library providing document processing, retrieval, and
    question-answering capabilities.
    """

    # Centralized default configuration values
    DEFAULT_CONFIG = DEFAULT_RAG_CONFIG

    # Provider and retrieval mappings referenced from centralized Enums
    PROVIDER_MAPPING = PROVIDER_MAPPING
    RETRIEVAL_MAPPING = RETRIEVAL_MAPPING

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        prompt_manager: Optional[MuffakirPrompt] = None,
        embedding_provider: Optional["EmbeddingProvider"] = None,
        db_manager: Optional[Any] = None,
        llm_provider: Optional[LLMProvider] = None,
        **kwargs: Any,
    ):
        """
        Initialize MuffakirRAG with configuration.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary containing settings.
            prompt_manager (Optional[MuffakirPrompt]): Injected prompt manager.
            embedding_provider (Optional): Injected embedding provider.
            db_manager (Optional[BaseVectorDBManager]): Injected VectorDB manager.
            llm_provider (Optional[LLMProvider]): Injected pre-configured LLMProvider.
            **kwargs: Configuration keys passed directly as keyword arguments.
        """
        self.logger = logging.getLogger(__name__)

        user_config = dict(config) if config else {}
        user_config.update(kwargs)

        # Map documents_path to data_dir for backward/ergonomic compatibility
        if "documents_path" in user_config and "data_dir" not in user_config:
            user_config["data_dir"] = user_config["documents_path"]

        # Merge user config with defaults
        self.config = self._merge_config(user_config)

        # Injected shared components
        self._injected_embedding_provider = embedding_provider
        self._injected_db_manager = db_manager
        self._injected_llm_provider = llm_provider

        # Validate required parameters
        self._validate_config()

        from Muffakir.dependency_validation import validate_rag_dependencies

        validate_rag_dependencies(
            self.config,
            injected_embedding=self._injected_embedding_provider is not None,
            injected_vector_db=self._injected_db_manager is not None,
            injected_llm=(
                self._injected_llm_provider is not None
                and not isinstance(self._injected_llm_provider, str)
            ),
        )

        # Initialize components
        self._initialize_components(prompt_manager)

        # Process documents and setup RAG pipeline
        self._setup_pipeline()

        self.logger.info("MuffakirRAG initialized successfully.")

    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with default values."""
        config = self.DEFAULT_CONFIG.copy()
        config.update(user_config)
        config["search_provider_config"] = dict(
            user_config.get(
                "search_provider_config",
                config.get("search_provider_config") or {},
            )
        )
        config["vector_db_config"] = dict(
            user_config.get("vector_db_config", config.get("vector_db_config") or {})
        )
        config["document_parser_config"] = dict(
            user_config.get(
                "document_parser_config",
                config.get("document_parser_config") or {},
            )
        )
        return config

    def _normalize_adaptive_search_config(self):
        """Normalize adaptive web search settings and apply Firecrawl alias."""
        provider = (self.config.get("search_provider") or "tavily").lower().strip()
        self.config["search_provider"] = provider
        sp_config = dict(self.config.get("search_provider_config") or {})

        if self.config.get("fire_crawl_api") and "api_key" not in sp_config:
            if provider in ("firecrawl", "fire_crawl", "fire-crawl"):
                sp_config["api_key"] = self.config["fire_crawl_api"]

        self.config["search_provider_config"] = sp_config

    def _validate_config(self):
        """Validate required configuration parameters."""
        skip_ingestion = bool(self.config.get("skip_document_ingestion", False))
        required_params = []

        # A string passed as llm_provider= is a provider name, not a real injected instance.
        has_real_injected_provider = (
            self._injected_llm_provider is not None
            and not isinstance(self._injected_llm_provider, str)
        )
        # If a string was passed, propagate it into the config so downstream validation sees it.
        # NOTE: must NOT use setdefault here — DEFAULT_CONFIG already inserts the key with value
        # None, so setdefault would silently leave it as None. Use explicit assignment instead.
        if isinstance(self._injected_llm_provider, str) and not self.config.get("llm_provider"):
            self.config["llm_provider"] = self._injected_llm_provider
        
        provider_val = (self.config.get("llm_provider") or "").lower().strip()
        if not has_real_injected_provider:
            # Local endpoints (Ollama, local vLLM / base_url) don't strictly require an API key
            if provider_val not in ("ollama", "vllm") and not self.config.get("base_url") and not self.config.get("llm_base_url"):
                required_params.append("api_key")
            required_params.extend(["llm_provider", "llm_model"])

        if not skip_ingestion and self._injected_db_manager is None:
            required_params.append("data_dir")

        for param in required_params:
            if not self.config.get(param):
                raise ValueError(f"Required parameter '{param}' is missing from configuration")

        # Validate data directory exists (only when ingesting documents)
        if not skip_ingestion and self._injected_db_manager is None:
            data_path = Path(self.config["data_dir"])
            if not data_path.exists():
                raise FileNotFoundError(f"Data directory does not exist: {self.config['data_dir']}")

        # Validate provider (if creating one from config)
        if not has_real_injected_provider and self.config.get("llm_provider"):
            if self.config["llm_provider"] not in self.PROVIDER_MAPPING:
                available_providers = ", ".join(self.PROVIDER_MAPPING.keys())
                raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}. Available providers: {available_providers}")

        # Validate retrieval method
        if self.config["retrieval_method"] not in self.RETRIEVAL_MAPPING:
            available_methods = ", ".join(self.RETRIEVAL_MAPPING.keys())
            raise ValueError(f"Unsupported retrieval method: {self.config['retrieval_method']}. Available methods: {available_methods}")

        self._normalize_adaptive_search_config()
        if self.config.get("adaptive_web_search"):
            self._validate_adaptive_web_search_config()

        self.logger.info("Configuration validated successfully.")

    def _validate_adaptive_web_search_config(self):
        """Validate web search credentials when Adaptive RAG is enabled."""
        provider = self.config.get("search_provider", "tavily")
        supported = ("firecrawl", "tavily", "serpapi", "fire_crawl", "fire-crawl", "serp_api", "serp-api", "google")
        if provider not in supported:
            raise ValueError(
                f"Unsupported search_provider: {provider}. "
                "Available: 'firecrawl', 'tavily', 'serpapi'."
            )

        sp_config = self.config.get("search_provider_config") or {}
        search_api_key = sp_config.get("api_key")

        if provider in ("firecrawl", "fire_crawl", "fire-crawl"):
            if not search_api_key and not self.config.get("fire_crawl_api"):
                raise ValueError(
                    "adaptive_web_search with Firecrawl requires an API key. Pass "
                    "search_provider_config={'api_key': '...'} or fire_crawl_api='...'."
                )
        elif provider == "tavily":
            if not search_api_key and not os.getenv("TAVILY_API_KEY"):
                raise ValueError(
                    "adaptive_web_search with Tavily requires an API key. Pass "
                    "search_provider_config={'api_key': '...'} or set TAVILY_API_KEY."
                )
        elif provider in ("serpapi", "serp_api", "serp-api", "google"):
            if not search_api_key and not os.getenv("SERPAPI_API_KEY"):
                raise ValueError(
                    "adaptive_web_search with SerpAPI requires an API key. Pass "
                    "search_provider_config={'api_key': '...'} or set SERPAPI_API_KEY."
                )

    def _initialize_components(self, prompt_manager: Optional[MuffakirPrompt] = None):
        """Initialize all core components."""
        self.logger.info("Initializing core components...")

        # Initialize LLM Provider (reuse injected instance if provided)
        # Guard: if someone accidentally passed a provider-name string as llm_provider=,
        # treat it as a config key and fall through to the normal build path.
        if self._injected_llm_provider is not None and not isinstance(self._injected_llm_provider, str):
            self.llm_provider = self._injected_llm_provider
            self.logger.info("Injected LLM Provider used.")
        else:
            # If a string was passed as llm_provider=, treat it as the provider name.
            # Use direct assignment (not setdefault) since DEFAULT_CONFIG seeds the key with None.
            if isinstance(self._injected_llm_provider, str) and not self.config.get("llm_provider"):
                self.config["llm_provider"] = self._injected_llm_provider
                self.logger.info(
                    f"llm_provider='{self._injected_llm_provider}' interpreted as provider name (not an instance)."
                )
            self.llm_provider = LLMProvider(
                parameters=resolve_parameters(self.config),
                api_key=self.config.get("api_key"),
                provider=self.PROVIDER_MAPPING.get(self.config.get("llm_provider", "openai"), ProviderName.OPENAI),
                model=self.config.get("llm_model", "gpt-4o-mini"),
                temperature=self.config.get("llm_temperature", 0.0),
                max_tokens=self.config.get("llm_max_tokens", 4096),
                base_url=self.config.get("llm_base_url") or self.config.get("base_url"),
            )
            self.logger.info(f"LLM Provider initialized: {self.config.get('llm_provider')}")

        # Initialize Embedding Provider (reuse injected instance if provided, e.g. shared across trials)
        if self._injected_embedding_provider is not None:
            self.embedding_provider = self._injected_embedding_provider
            self.logger.info(f"Embedding Provider injected: {self.config['embedding_model']} ({self.config.get('embedding_provider', 'sentence_transformers')})")
        else:
            self.embedding_provider = EmbeddingProvider(
                model_name=self.config["embedding_model"],
                provider=self.config.get("embedding_provider", "sentence_transformers"),
                api_key=self.config.get("api_key"),
                batch_size=self.config.get("embedding_batch_size", 16),
                device=self.config.get("device", "auto"),
            )
            self.logger.info(f"Embedding Provider initialized: {self.config['embedding_model']} ({self.config.get('embedding_provider', 'sentence_transformers')})")

        # Initialize Prompt Manager
        if prompt_manager:
            self.prompt_manager = prompt_manager
            self.logger.info("Custom Prompt Manager injected.")
        else:
            self.prompt_manager = MuffakirPrompt(
                language=self.config.get("language", "ar"),
                overrides=self.config.get("prompt_overrides"),
            )
            self.logger.info(f"Prompt Manager initialized with language: {self.config.get('language', 'ar')}")

        # Initialize Document Parser
        self.doc_parser = None
        if self.config.get("document_parser"):
            from DocumentParser import create_document_parser
            self.doc_parser = create_document_parser(
                provider=self.config["document_parser"],
                **self.config.get("document_parser_config", {})
            )
            self.logger.info(f"Document Parser initialized: {self.config['document_parser']}")
        # Backward compatibility for old static azure config
        elif self.config.get("azure_endpoint") and self.config.get("azure_api_key"):
            from DocumentParser import create_document_parser
            self.doc_parser = create_document_parser(
                provider="azure",
                endpoint=self.config["azure_endpoint"],
                api_key=self.config["azure_api_key"]
            )
            self.logger.info("Document Parser initialized: azure (backward compatibility)")

        # Initialize Chunking Orchestrator
        self.chunking = None
        if self.config.get("chunking"):
            self.chunking = self.config["chunking"]
            self.logger.info(f"Custom Chunking injected: {self.chunking.chunker.name}")
        else:
            from TextProcessor.MuffakirChunking import MuffakirChunking
            self.chunking = MuffakirChunking(
                chunker=self.config.get("chunking_method", "recursive"),
                chunker_config={
                    "size": self.config.get("chunk_size", 600),
                    "overlap": self.config.get("chunk_overlap", 200)
                },
                language=self.config.get("language", "auto")
            )
            self.logger.info(f"Chunking initialized: {self.chunking.chunker.name}")

        # Initialize optional components
        self.query_transformer = None
        if self.config["query_transformer"]:
            strategy = self.config.get("query_transformer_strategy", "rewrite")
            qt_provider_override = self.config.get("query_transform_llm_provider")
            qt_model_override = self.config.get("query_transform_llm_model")
            if qt_provider_override or qt_model_override or self.config.get("query_transform_llm_parameters") is not None:
                query_transform_llm_provider = LLMProvider(
                    parameters=resolve_parameters(self.config, "query_transform"),
                    api_key=self.config.get("query_transform_api_key") or self.config.get("api_key"),
                    provider=self.PROVIDER_MAPPING.get(
                        qt_provider_override or self.config.get("llm_provider", "openai"),
                        ProviderName.OPENAI,
                    ),
                    model=qt_model_override or self.config.get("llm_model", "gpt-4o-mini"),
                    temperature=self.config.get("llm_temperature", 0.0),
                    max_tokens=self.config.get("llm_max_tokens", 4096),
                    base_url=(
                        self.config.get("query_transform_base_url")
                        or self.config.get("llm_base_url")
                        or self.config.get("base_url")
                    ),
                )
                self.logger.info(
                    f"Query Transformer using override LLM: {qt_provider_override}/{qt_model_override}"
                )
            else:
                query_transform_llm_provider = self.llm_provider
            self.query_transformer = QueryTransformer(
                llm_provider=query_transform_llm_provider,
                prompt_manager=self.prompt_manager,
                prompt="query_rewrite",
                strategy=strategy,
            )
            self.logger.info(f"Query Transformer enabled (strategy={strategy})")

        self.hallucination_checker = None
        if self.config.get("hallucination_check", True):
            self.hallucination_checker = HallucinationsCheck(
                llm_provider=self.llm_provider,
                prompt_manager=self.prompt_manager,
                method=self.config.get("hallucination_method", "text_cleaner"),
                embedding_provider=self.embedding_provider,
            )
            self.logger.info(
                f"Hallucination Checker enabled: {self.config.get('hallucination_method', 'text_cleaner')}"
            )

        self.reranker = None
        if self.config.get("reranking", False):
            reranking_method = str(
                self.config.get("reranking_method", "semantic_similarity")
            ).strip().lower()
            reranker_llm_provider = self.llm_provider
            llm_override_requested = any(
                self.config.get(key)
                for key in (
                    "reranker_llm_provider",
                    "reranker_llm_model",
                    "reranker_llm_api_key",
                    "reranker_llm_base_url",
                )
            )
            if reranking_method in {"llm", "llm_reranker", "llm_based", "llm-based"} and llm_override_requested:
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
            if (reranking_method in {"llm", "llm_reranker", "llm_based", "llm-based"}
                    and not llm_override_requested and self.config.get("reranker_llm_parameters") is not None):
                reranker_llm_provider = LLMProvider(
                    api_key=self.config.get("api_key"),
                    provider=resolve_provider_name(self.config.get("llm_provider", "openai")),
                    model=self.config.get("llm_model", "gpt-4o-mini"),
                    base_url=self.config.get("llm_base_url") or self.config.get("base_url"),
                    parameters=resolve_parameters(self.config, "reranker"),
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
            self.logger.info(f"Reranker enabled with method: {reranking_method}")

        # Adaptive RAG web search provider (shared WebSearch factory; not full MuffakirSearch)
        self.web_search_provider = None
        if self.config.get("adaptive_web_search"):
            from WebSearch import create_web_search_provider
            self.web_search_provider = create_web_search_provider(
                provider=self.config.get("search_provider", "tavily"),
                **dict(self.config.get("search_provider_config") or {}),
            )
            self.logger.info(
                f"Adaptive web search enabled with provider: "
                f"{self.config.get('search_provider', 'tavily')}"
            )

    def _setup_pipeline(self):
        """Setup the complete RAG pipeline."""
        self.logger.info("Setting up RAG pipeline...")
        skip_ingestion = bool(self.config.get("skip_document_ingestion", False))

        if skip_ingestion:
            # Read-only mode: a shared, pre-indexed VectorDB is reused (e.g. by MuffakirComposer).
            # Skip chunking and indexing; documents are already stored in the shared DB.
            self.logger.info("Skipping document ingestion (skip_document_ingestion=True)")
            self.documents = []
            self.document_processor = None
        else:
            # Initialize document processor
            processor_cls = _component("ChunkingAndProcessing")
            self.document_processor = processor_cls(
                directory_path=self.config["data_dir"],
                document_parser=self.doc_parser,
                muffakir_chunking=self.chunking
            )
            # Process documents
            self.logger.info("Processing documents...")
            self.documents = self.document_processor.process_all(
                chunking_method=self.config["chunking_method"],
                use_ocr=self.config.get("use_ocr", False)
            )
            self.logger.info(f"Processed {len(self.documents)} document chunks.")

        # Initialize RAG Pipeline Manager (reuses injected db_manager if provided)
        pipeline_manager_cls = _component("RAGPipelineManager")
        self.rag_manager = pipeline_manager_cls(
            db_path=self.config["db_path"],
            collection_name=self.config["collection_name"],
            model_name=self.config["embedding_model"],
            vector_db_provider=self.config.get("vector_db_provider", "chroma"),
            vector_db_config=self.config.get("vector_db_config", {}),
            db_manager=self._injected_db_manager,
            embedding_provider=self.embedding_provider,
            llm_provider=self.llm_provider,
            query_transformer=self.query_transformer,
            prompt_manager=self.prompt_manager,
            hallucination=self.hallucination_checker,
            reranker=self.reranker,
            web_search_provider=self.web_search_provider,
            adaptive_web_search=bool(self.config.get("adaptive_web_search")),
            k=self.config["k"],
            fetch_k=self.config["fetch_k"],
            retrieve_method=self.RETRIEVAL_MAPPING[self.config["retrieval_method"]],
        )

        # Store documents in vector database (only when ingesting)
        if not skip_ingestion:
            self.logger.info("Storing documents in vector database...")
            self.rag_manager.store_documents(self.documents)
            self.logger.info("Documents stored successfully.")
        else:
            self.logger.info("Using pre-indexed shared vector database (read-only).")

    def ask(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Ask a question and get an intelligent answer.

        Args:
            question (str): The question in Arabic or English
            **kwargs: Additional parameters to override defaults per query (thread-safe)
                - k (int): Number of documents to retrieve
                - retrieval_method (str): Retrieval method to use

        Returns:
            Dict[str, Any]: Response containing answer and metadata
        """
        if not question or not question.strip():
            return {
                "answer": "يرجى تقديم سؤال صحيح",
                "error": "Empty question provided",
                "sources": []
            }

        try:
            self.logger.info(f"Processing question: {question[:100]}...")

            # Resolve per-query parameter overrides
            k_override = kwargs.get("k", self.rag_manager.k)
            method_override = self.rag_manager.retrieve_method

            if "retrieval_method" in kwargs:
                method_name = kwargs["retrieval_method"]
                if method_name in self.RETRIEVAL_MAPPING:
                    method_override = self.RETRIEVAL_MAPPING[method_name]
                else:
                    self.logger.warning(f"Unknown retrieval method: {method_name}")

            # Pass per-query overrides straight through as call arguments (not via
            # mutating shared self.rag_manager state) so concurrent ask() calls on
            # the same MuffakirRAG instance never race on each other's k/method.
            response = self.rag_manager.generate_answer(
                question, k=k_override, retrieve_method=method_override
            )

            self.logger.info("Answer generated successfully.")
            return response

        except MuffakirError:
            raise
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}", exc_info=True)
            raise GenerationError(f"Failed to generate answer: {e}") from e

    def get_similar_documents(
        self,
        query: str,
        k: Optional[int] = None,
        method: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve similar documents for a given query.

        Args:
            query (str): Search query
            k (int, optional): Number of documents to retrieve
            method (str, optional): Retrieval method to use

        Returns:
            List[Document]: Similar documents
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided for document retrieval.")
            return []

        try:
            k = k or self.config["k"]
            method_enum = self.RETRIEVAL_MAPPING.get(
                method or self.config["retrieval_method"]
            )

            return self.rag_manager.query_similar_documents(
                query=query,
                k=k,
                method=method_enum
            )

        except MuffakirError:
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            raise RetrievalError(f"Failed to retrieve documents: {e}") from e

    def get_similar_documents_with_trace(
        self,
        query: str,
        k: Optional[int] = None,
        method: Optional[str] = None,
    ) -> RetrievalTelemetryResult:
        """Run the configured retrieval architecture and return telemetry.

        This is intentionally separate from the legacy list-returning method so
        existing library callers keep the same interface.
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided for document retrieval.")
            return RetrievalTelemetryResult()

        effective_k = k or self.config["k"]
        method_enum = self.RETRIEVAL_MAPPING.get(method or self.config["retrieval_method"])

        def retrieve_one(search_query: str, requested_k: int) -> List[Document]:
            return self.rag_manager.query_similar_documents(
                query=search_query,
                k=requested_k,
                method=method_enum,
            )

        rerank = None
        if self.reranker:
            from Reranker.Reranker import rerank_documents

            rerank = lambda rerank_query, documents: rerank_documents(
                self.reranker, rerank_query, documents
            )

        try:
            return run_timed_retrieval(
                query,
                effective_k,
                retrieve_one,
                query_transformer=self.query_transformer,
                rerank=rerank,
                embedding_provider=self.embedding_provider,
            )
        except MuffakirError:
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            raise RetrievalError(f"Failed to retrieve documents: {e}") from e

    def add_documents(self, documents: Union[List[Document], List[str]]) -> bool:
        """
        Add new documents to the knowledge base.

        Args:
            documents: List of Document objects or file paths

        Returns:
            bool: Success status
        """
        if not documents:
            self.logger.warning("No documents provided.")
            return False

        try:
            if isinstance(documents[0], str):
                # Process file paths using configured chunking orchestrator
                processed_docs = []
                for file_path in documents:
                    if not Path(file_path).exists():
                        self.logger.warning(f"File not found: {file_path}")
                        continue

                    processor_cls = _component("ChunkingAndProcessing")
                    temp_processor = processor_cls(
                        directory_path=str(Path(file_path).parent),
                        document_parser=self.doc_parser,
                        muffakir_chunking=self.chunking,
                    )
                    docs = temp_processor.process_all(
                        chunking_method=self.config["chunking_method"],
                        use_ocr=self.config.get("use_ocr", False)
                    )
                    processed_docs.extend(docs)
                documents = processed_docs

            if not documents:
                self.logger.warning("No valid documents to add.")
                return False

            self.rag_manager.store_documents(documents)
            self.logger.info(f"Added {len(documents)} new documents.")
            return True

        except MuffakirError:
            raise
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}", exc_info=True)
            raise DocumentIndexError(f"Failed to add documents: {e}") from e

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()

    def __repr__(self) -> str:
        doc_count = len(self.documents) if hasattr(self, 'documents') else 0
        return f"MuffakirRAG(provider={self.config['llm_provider']}, docs={doc_count}, k={self.config['k']})"
