from typing import Dict, Any
import logging
import time

from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt
from Generation.AnswerGenerator import AnswerGenerator
from WebSearch.base import BaseWebSearchProvider
from Muffakir.telemetry import empty_stage_timings
from Trace.observability import observe_stage, provider_identity


class Search:
    """
    Search orchestrator: runs a pluggable web search provider, then generates an answer.
    """

    def __init__(
        self,
        web_search_provider: BaseWebSearchProvider,
        llm_provider: LLMProvider,
        prompt_manager: MuffakirPrompt,
    ):
        self.logger = logging.getLogger(__name__)
        self.web_search_provider = web_search_provider
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager

        self.generator = AnswerGenerator(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
        )

    def search_web(self, original_query: str) -> Dict[str, Any]:
        pipeline_start = time.perf_counter()
        timings = empty_stage_timings()

        stage_start = time.perf_counter()
        with observe_stage(
            "web_search",
            "web_search",
            provider=type(self.web_search_provider).__name__,
        ):
            result = self.web_search_provider.search(original_query)
        timings["web_search_ms"] = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        llm_name, llm_model = provider_identity(self.llm_provider)
        with observe_stage(
            "generation", "llm", provider=llm_name, model=llm_model
        ):
            answer = self.generator.generate_answer(original_query, result.content)
        timings["generation_ms"] = (time.perf_counter() - stage_start) * 1000.0

        return {
            "answer": answer,
            "sources": result.sources,
            "context": result.content or "",
            "stage_timings_ms": timings,
            "pipeline_latency_ms": (time.perf_counter() - pipeline_start) * 1000.0,
        }
