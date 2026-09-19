from typing import Dict, Any, List, Optional
import logging
import os

from LLMProvider.LLMProvider import LLMProvider
from LLMProvider.parameters import resolve_parameters
from PromptManager.PromptManager import MuffakirPrompt
from Muffakir.Enums import ProviderName, PROVIDER_MAPPING, resolve_provider_name, WebSearchProvider
from Muffakir.constants import DEFAULT_SEARCH_CONFIG
from Muffakir.exceptions import ConfigurationError
from Muffakir.Search import Search
from WebSearch import create_web_search_provider

logger = logging.getLogger(__name__)


class MuffakirSearch:
    """
    Search RAG facade: integrates pluggable WebSearch providers (Firecrawl, Tavily, SerpAPI)
    with LLM answer generation.
    """

    # Centralized default configuration values
    DEFAULT_CONFIG = DEFAULT_SEARCH_CONFIG

    # Provider mapping referenced from centralized Enums
    PROVIDER_MAPPING = PROVIDER_MAPPING

    SUPPORTED_SEARCH_PROVIDERS = ("firecrawl", "tavily", "serpapi")

    def __init__(self, config: Dict[str, Any], prompt_manager: Optional[MuffakirPrompt] = None):

        self.logger = logging.getLogger(__name__)

        self.config = self._merge_config(config)
        self._normalize_search_config()
        self._validate_config()
        from Muffakir.dependency_validation import validate_search_dependencies

        validate_search_dependencies(self.config)
        self._initialize_components(prompt_manager)
        self._setup_pipeline()

        self.logger.info("MuffakirSearch initialized successfully.")

    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with default values."""
        config = self.DEFAULT_CONFIG.copy()
        config.update(user_config)
        # Deep-copy nested dict so we don't mutate the class default
        config["search_provider_config"] = dict(
            user_config.get("search_provider_config", config.get("search_provider_config") or {})
        )
        return config

    def _normalize_search_config(self):
        """
        Normalize search provider settings and apply backward-compatible aliases.
        """
        provider = (self.config.get("search_provider") or "firecrawl").lower().strip()
        self.config["search_provider"] = provider

        sp_config = dict(self.config.get("search_provider_config") or {})

        # Backward compatibility: top-level fire_crawl_api -> Firecrawl api_key
        if self.config.get("fire_crawl_api") and "api_key" not in sp_config:
            if provider in ("firecrawl", "fire_crawl", "fire-crawl"):
                sp_config["api_key"] = self.config["fire_crawl_api"]

        # Promote common Firecrawl knobs from top-level config if not already set
        for key in ("max_depth", "time_limit", "max_urls"):
            if key not in sp_config and self.config.get(key) is not None:
                sp_config[key] = self.config[key]

        self.config["search_provider_config"] = sp_config

    def _validate_config(self):
        """Validate required configuration parameters."""
        required_params = ["api_key", "llm_provider", "llm_model"]

        for param in required_params:
            if not self.config.get(param):
                raise ValueError(f"Required parameter '{param}' is missing from configuration")

        if self.config["llm_provider"] not in self.PROVIDER_MAPPING:
            available_providers = ", ".join(self.PROVIDER_MAPPING.keys())
            raise ValueError(
                f"Unsupported LLM provider: {self.config['llm_provider']}. "
                f"Available providers: {available_providers}"
            )

        provider = self.config["search_provider"]
        if provider not in self.SUPPORTED_SEARCH_PROVIDERS and provider not in (
            "fire_crawl",
            "fire-crawl",
            "serp_api",
            "serp-api",
            "google",
        ):
            available = ", ".join(self.SUPPORTED_SEARCH_PROVIDERS)
            raise ValueError(
                f"Unsupported search provider: {provider}. Available providers: {available}"
            )

        # Provider-specific API key checks
        sp_config = self.config.get("search_provider_config") or {}
        search_api_key = sp_config.get("api_key")

        if provider in ("firecrawl", "fire_crawl", "fire-crawl"):
            if not search_api_key and not self.config.get("fire_crawl_api"):
                raise ValueError(
                    "Firecrawl requires an API key. Pass "
                    "search_provider_config={'api_key': '...'} or fire_crawl_api='...'."
                )
        elif provider == "tavily":
            if not search_api_key and not os.getenv("TAVILY_API_KEY"):
                raise ValueError(
                    "Tavily requires an API key. Pass "
                    "search_provider_config={'api_key': '...'} or set TAVILY_API_KEY."
                )
        elif provider in ("serpapi", "serp_api", "serp-api", "google"):
            if not search_api_key and not os.getenv("SERPAPI_API_KEY"):
                raise ValueError(
                    "SerpAPI requires an API key. Pass "
                    "search_provider_config={'api_key': '...'} or set SERPAPI_API_KEY."
                )

        self.logger.info("Configuration validated successfully.")

    def _initialize_components(self, prompt_manager: Optional[MuffakirPrompt] = None):
        """Initialize all core components."""
        self.logger.info("Initializing core components...")

        self.llm_provider = LLMProvider(
            parameters=resolve_parameters(self.config),
            api_key=self.config.get("api_key"),
            provider=resolve_provider_name(self.config["llm_provider"]),
            model=self.config["llm_model"],
            temperature=self.config.get("llm_temperature", 0.0),
            max_tokens=self.config.get("llm_max_tokens", 4096),
            base_url=self.config.get("llm_base_url") or self.config.get("base_url"),
        )
        self.logger.info(f"LLM Provider initialized: {self.config['llm_provider']}")


        if prompt_manager:
            self.prompt_manager = prompt_manager
            self.logger.info("Custom Prompt Manager injected.")
        else:
            self.prompt_manager = MuffakirPrompt(
                language=self.config.get("language", "ar"),
                overrides=self.config.get("prompt_overrides"),
            )
            self.logger.info(
                f"Prompt Manager initialized with language: {self.config.get('language', 'ar')}"
            )

    def _setup_pipeline(self):
        """Setup the search pipeline with a pluggable web search provider."""
        self.logger.info("Setting up Search pipeline...")

        provider_name = self.config["search_provider"]
        provider_config = dict(self.config.get("search_provider_config") or {})

        self.web_search_provider = create_web_search_provider(
            provider=provider_name,
            **provider_config,
        )
        self.logger.info(f"Web Search Provider initialized: {provider_name}")

        self.search_pipeline = Search(
            web_search_provider=self.web_search_provider,
            llm_provider=self.llm_provider,
            prompt_manager=self.prompt_manager,
        )

    @property
    def search_pipline(self) -> Search:
        """Backward-compatible alias for misspelled search_pipline attribute."""
        return self.search_pipeline

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()

    def search(self, query: str) -> Dict[str, Any]:
        """Perform a search using the configured web search provider and LLM."""
        self.logger.info(f"Performing search for query: {query}")

        try:
            results = self.search_pipeline.search_web(query)
            self.logger.info("Search completed successfully.")
            return results
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise e

    def ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """EvalRunner-compatible adapter (matches MuffakirRAG.ask()'s call and
        return shape) so a web-search-only run can be scored by the existing
        evaluation pipeline unchanged. `k`/`retrieval_method` kwargs are
        accepted for interface compatibility but unused — there is no local
        retrieval in web-search-only mode."""
        result = self.search(question)
        return {
            "answer": result["answer"],
            "retrieved_documents": [],
            "source_metadata": [
                {"title": s.get("title"), "url": s.get("url")}
                for s in result.get("sources", [])
            ],
            "context_source": "web_search",
            "web_sources": result.get("sources", []),
            "used_context": result.get("context", ""),
            "stage_timings_ms": result.get("stage_timings_ms", {}),
            "pipeline_latency_ms": result.get("pipeline_latency_ms"),
        }

    def get_similar_documents(self, query: str, k: Optional[int] = None) -> List[Any]:
        """Web-search-only mode has no local retrieval corpus — retrieval
        metrics (recall/precision/mrr/ndcg) are not meaningful here. Fail
        loudly with a clear message rather than an ambiguous AttributeError."""
        raise ConfigurationError(
            "MuffakirSearch has no local retrieval corpus — do not request "
            "retrieval metrics (recall/precision/mrr/ndcg) for a web-search-only "
            "run. Use faithfulness, answer_correctness, or llm_judge_rating instead."
        )
