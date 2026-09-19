from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from PromptManager.PromptManager import MuffakirPrompt
from LLMProvider.LLMProvider import LLMProvider
from LLMProvider.parameters import resolve_parameters
from Muffakir.Enums import ProviderName, PROVIDER_MAPPING, resolve_provider_name
from Muffakir.constants import DEFAULT_EVALUATION_CONFIG
from Evaluation.constants import ALL_METRICS, DEFAULT_METRICS, GENERATION_METRICS

if TYPE_CHECKING:
    from Evaluation.dataset import DatasetInput
    from Evaluation.models import EvaluationReport


class MuffakirEvaluation:
    """
    Custom RAG evaluation facade.

    Option 1: inject an already-built MuffakirRAG via evaluate(rag=..., dataset=...).
    Dataset must match MuffakirSyntheticData / QAPair schema.
    """

    ALL_METRICS = list(ALL_METRICS)

    # Centralized default configuration values
    DEFAULT_CONFIG = DEFAULT_EVALUATION_CONFIG

    # Provider mapping referenced from centralized Enums
    PROVIDER_MAPPING = PROVIDER_MAPPING

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        prompt_manager: Optional[MuffakirPrompt] = None,
    ):
        # NOTE: libraries must not call logging.basicConfig (mutates host root logger).
        self.logger = logging.getLogger(__name__)
        self.config = self._merge_config(config or {})
        self._prompt_manager_override = prompt_manager
        self.prompt_manager = prompt_manager or MuffakirPrompt(
            language=self.config.get("language", "ar"),
            overrides=self.config.get("prompt_overrides"),
        )
        self.logger.info("MuffakirEvaluation initialized")

    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        config = self.DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config

    def _resolve_metrics(self) -> List[str]:
        metrics = self.config.get("metrics")
        if not metrics:
            return list(DEFAULT_METRICS)
        resolved = []
        unknown = []
        for m in metrics:
            key = str(m).lower().strip()
            if key in ALL_METRICS:
                resolved.append(key)
            else:
                unknown.append(m)
        if unknown:
            raise ValueError(
                f"Unknown metrics: {unknown}. Available: {ALL_METRICS}"
            )
        return resolved or list(DEFAULT_METRICS)

    def _resolve_llm_provider(self, rag: Any) -> LLMProvider:
        """
        Use override LLM only when api_key + llm_provider + llm_model are all provided.
        Otherwise reuse rag.llm_provider.
        """
        api_key = self.config.get("api_key")
        provider_name = self.config.get("llm_provider")
        model = self.config.get("llm_model")

        if provider_name and model:
            try:
                canonical_provider = resolve_provider_name(provider_name)
            except ValueError as e:
                raise ValueError(str(e)) from e

            self.logger.info(
                f"Using override LLM for evaluation judges: {provider_name}/{model}"
            )
            return LLMProvider(
                parameters=resolve_parameters(self.config, "judge"),
                api_key=api_key,
                provider=canonical_provider,
                model=model,
                temperature=self.config.get("llm_temperature", 0.0),
                max_tokens=self.config.get("llm_max_tokens", 4096),
                base_url=self.config.get("llm_base_url") or self.config.get("base_url"),
            )


        if not hasattr(rag, "llm_provider") or rag.llm_provider is None:
            raise ValueError(
                "No LLM available for evaluation. Pass api_key/llm_provider/llm_model "
                "in MuffakirEvaluation config, or inject a MuffakirRAG with llm_provider."
            )

        if self.config.get("judge_llm_parameters") is not None or self.config.get("llm_parameters") is not None:
            source = rag.llm_provider
            if not isinstance(getattr(source, "provider", None), str) or not isinstance(getattr(source, "model", None), str):
                raise ValueError("Parameter-only judge override needs a Muffakir LLMProvider; specify llm_provider and llm_model for a custom injected client.")
            extra = dict(getattr(source, "extra", {}) or {})
            extra.pop("parameters", None)
            extra.pop("base_url", None)
            return LLMProvider(
                provider=source.provider, model=source.model, api_key=getattr(source, "api_key", None),
                base_url=getattr(source, "base_url", None),
                parameters=resolve_parameters(self.config, "judge"), **extra,
            )
        self.logger.info("Reusing MuffakirRAG llm_provider for evaluation judges")
        return rag.llm_provider

    def evaluate(
        self,
        rag: Any,
        dataset: DatasetInput,
        save: bool = False,
        output_dir: Optional[str] = None,
    ) -> EvaluationReport:
        """
        Evaluate an injected MuffakirRAG on a QAPair-compatible dataset.

        Args:
            rag: Initialized MuffakirRAG instance
            dataset: path | DataFrame | list[dict] with QAPair columns
            save: if True, write report to output_dir
            output_dir: override config output_dir when saving
        """
        # Duck-type: accept any object exposing the RAG interface methods.
        if not (hasattr(rag, "ask") and callable(getattr(rag, "ask"))):
            raise TypeError(
                f"rag must expose an ask(question, k=...) method "
                f"(got {type(rag).__name__}). "
                "Build a MuffakirRAG (or compatible) first, then pass it to evaluate()."
            )

        metrics = self._resolve_metrics()
        from Muffakir.dependency_validation import validate_evaluation_dependencies

        validate_evaluation_dependencies(self.config, metrics)
        from Evaluation.dataset import load_evaluation_dataset
        from Evaluation.runner import EvalRunner

        rag_config = getattr(rag, "config", {}) or {}
        pipeline_mode = (
            rag_config.get("pipeline_mode", "full_rag")
            if isinstance(rag_config, dict)
            else "full_rag"
        )
        execute_generation = pipeline_mode != "retrieval_only"
        generation_metrics_requested = bool(set(metrics) & GENERATION_METRICS)
        if not execute_generation and generation_metrics_requested:
            raise ValueError(
                "pipeline_mode='retrieval_only' cannot score generation metrics "
                "because no answer is generated."
            )

        # The judge provider is evaluation overhead and is independent from
        # the RAG's own answer-generation LLM. Do not construct it unless a
        # selected metric actually needs a judge.
        llm_provider = (
            self._resolve_llm_provider(rag)
            if generation_metrics_requested
            else None
        )

        pairs = load_evaluation_dataset(
            dataset=dataset,
            fail_fast=bool(self.config.get("fail_fast_dataset", True)),
            max_samples=self.config.get("max_samples"),
        )
        self.logger.info(f"Loaded {len(pairs)} evaluation samples")

        runner = EvalRunner(
            rag=rag,
            llm_provider=llm_provider,
            prompt_manager=self.prompt_manager,
            metrics=metrics,
            k=int(self.config.get("k", 5)),
            execute_generation=execute_generation,
        )
        report = runner.run(pairs)

        if save:
            out = output_dir or self.config.get("output_dir") or "./muffakir_eval_results"
            report.save(out)
            self.logger.info(f"Evaluation report saved to {out}")

        return report

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()
