"""
Trial evaluation worker for MuffakirComposer.

This module contains the top-level (picklable) `evaluate_trial` function used by
GridSearch to evaluate a single pipeline configuration. Because it is a top-level
module attribute (not a closure) and `shared_state` is a plain picklable dict, it
can be shipped to ProcessPoolExecutor workers without pickling errors.

A per-process cache avoids re-loading the embedding model and re-opening the
VectorDB on every trial: heavy shared components are created once per worker
process and reused across all trials assigned to that worker.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from Muffakir.Enums import ProviderName, PROVIDER_MAPPING, resolve_provider_name
from Muffakir.exceptions import ConfigurationError, MuffakirError
from Trace.models import redact_config
from Trace.observability import observe_stage
from Evaluation.constants import GENERATION_METRICS

from .config_space import trial_config_to_rag_config
from .results.trial import TrialResult

logger = logging.getLogger(__name__)

# Providers supported by the downstream MuffakirRAG trial pipeline (imported from centralized Enums)
SUPPORTED_PROVIDERS: Dict[str, ProviderName] = PROVIDER_MAPPING



# Per-process cache of heavy shared components, keyed by the configuration that
# identifies them. Avoids re-loading the embedding model / re-opening the VectorDB
# for every trial within a single worker process.
_COMPONENT_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any]] = {}


def _get_shared_components(rag_config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Return a (embedding_provider, db_manager) pair for the given *resolved* trial
    config, creating and caching it once per process.

    ``rag_config`` must already be the output of trial_config_to_rag_config() —
    its db_path/collection_name are namespaced by compute_index_key() (see
    Composer/index_key.py), so this cache key naturally separates workers'
    components per distinct (chunking, embedding_model, vector_db_provider)
    combination without listing those fields separately here.
    """
    cache_key = (
        str(rag_config.get("db_path", "./muffakir_db")),
        str(rag_config.get("collection_name", "MuffakirComposer")),
        rag_config.get("embedding_model"),
    )
    cached = _COMPONENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from Embedding.EmbeddingProvider import EmbeddingProvider
    from VectorDB import create_vector_db

    embedding_provider = EmbeddingProvider(
        model_name=rag_config.get("embedding_model", "mohamed2811/Muffakir_Embedding"),
        provider=rag_config.get("embedding_provider", "sentence_transformers"),
        api_key=rag_config.get("api_key"),
        batch_size=rag_config.get("embedding_batch_size", 16),
        device=rag_config.get("device", "auto"),
    )

    vector_db_config = dict(rag_config.get("vector_db_config", {}) or {})
    vector_db_config.setdefault("path", rag_config.get("db_path", "./muffakir_db"))
    vector_db_config.setdefault("collection_name", rag_config.get("collection_name", "MuffakirComposer"))
    vector_db_config.setdefault("model_name", rag_config.get("embedding_model"))

    db_manager = create_vector_db(
        provider=rag_config.get("vector_db_provider", "chroma"),
        embedding_provider=embedding_provider,
        **vector_db_config,
    )

    _COMPONENT_CACHE[cache_key] = (embedding_provider, db_manager)
    logger.debug(f"Initialized shared components for process (key={cache_key})")
    return embedding_provider, db_manager


def evaluate_trial(
    trial_id: int,
    trial_config: Dict[str, Any],
    shared_state: Dict[str, Any],
) -> TrialResult:
    """
    Evaluate a single pipeline configuration against the eval dataset.

    This is a top-level function so it can be pickled and sent to ProcessPoolExecutor
    workers. `shared_state` must be picklable (no live DB/LLM objects).

    Args:
        trial_id: Unique trial identifier.
        trial_config: Trial-specific configuration from the search space
                      (e.g. {"query_expansion": "hyde", "retrieval": "hybrid", "k": 5}).
        shared_state: Picklable dict with keys:
            - base_config: base MuffakirRAG configuration (api_key, llm_*, db_path, ...)
            - eval_pairs: List[QAPair] to evaluate against
            - metrics: List[str] of metric names
            - metric_weights: Optional[Dict[str, float]] for composite scoring

    Returns:
        TrialResult with metrics + composite_score on success.

    Raises:
        MuffakirError: If the trial fails (provider, retrieval, generation,
            or configuration failure). Left to the caller (BaseSearch._run_with_retry)
            to decide whether to retry and to build the final failed TrialResult,
            so retry policy can be applied per error type instead of guessing
            from a flattened string.
    """
    from Evaluation.runner import EvalRunner
    from LLMProvider.LLMProvider import LLMProvider
    from PromptManager.PromptManager import MuffakirPrompt
    from Muffakir.Muffakir import MuffakirRAG
    from Muffakir.MuffakirSearch import MuffakirSearch
    from Muffakir.MuffakirRetrieval import MuffakirRetrieval
    from Pricing.price_map import PriceMap

    start_time = time.perf_counter()
    base_config: Dict[str, Any] = shared_state["base_config"]
    eval_pairs: List[Any] = shared_state["eval_pairs"]
    metrics: List[str] = shared_state.get("metrics", [])
    metric_weights: Dict[str, float] = shared_state.get("metric_weights", {}) or {}
    # Rehydrated from a plain dict -- pure in-memory work, no network call
    # (the price map was fetched once, up front, by Composer.fit()).
    price_map = PriceMap.from_dict(shared_state.get("pricing_snapshot") or {})
    prompt_config = shared_state.get("prompt_config") or {}
    prompt_manager = MuffakirPrompt(
        language=prompt_config.get("language", base_config.get("language", "ar")),
        overrides=prompt_config.get("overrides"),
    )

    try:
        # Build the trial's RAG config: base + trial overrides, in read-only mode
        rag_config = trial_config_to_rag_config(trial_config, base_config)
        rag_config["skip_document_ingestion"] = True

        if rag_config.get("retrieval_source") == "web_search_only":
            # No local corpus at all -- MuffakirSearch answers purely from a
            # live web search provider. Its .ask()/.get_similar_documents()
            # adapter (Muffakir/MuffakirSearch.py) makes it a drop-in `rag`
            # for EvalRunner with zero EvalRunner changes.
            with observe_stage("trial_initialization", "runtime"):
                rag = MuffakirSearch(config=rag_config, prompt_manager=prompt_manager)
        elif rag_config.get("pipeline_mode") == "retrieval_only":
            # Retrieval-only mode: exercise embedding/chunking/retrieval-method/
            # reranking/query-transformation without ever building a generation
            # LLM. Shares the same cached embedding_provider/db_manager as a
            # full_rag trial resolving to the same index.
            with observe_stage("trial_initialization", "runtime"):
                embedding_provider, db_manager = _get_shared_components(rag_config)
                rag = MuffakirRetrieval(
                    rag_config,
                    embedding_provider=embedding_provider,
                    db_manager=db_manager,
                    prompt_manager=prompt_manager,
                )
        else:
            # Reuse heavy shared components (embedding model + DB) across trials in this
            # process that resolve to the same index (see Composer/index_key.py)
            with observe_stage("trial_initialization", "runtime"):
                embedding_provider, db_manager = _get_shared_components(rag_config)

                rag = MuffakirRAG(
                    rag_config,
                    embedding_provider=embedding_provider,
                    db_manager=db_manager,
                    prompt_manager=prompt_manager,
                )

        # LLM for evaluation judging (separate from the RAG's internal LLM).
        # Only built when a generation metric was actually requested --
        # retrieval-only trials (either pipeline_mode="retrieval_only", or a
        # full_rag trial scored on retrieval metrics only) never need it, and
        # building it unconditionally would force every retrieval_only run to
        # configure a real llm_provider/api_key it will never use.
        generation_metrics_requested = bool(
            set(m.lower() for m in metrics) & GENERATION_METRICS
        )
        llm_provider = None
        if generation_metrics_requested:
            # The eval judge is intentionally built from base_config, never from
            # trial_config (even after the "llm" ConfigSpace dimension was added in
            # config_space.py) — comparing trials judged by different models would
            # silently corrupt every result in this search. See
            # test_evaluate_trial_judge_llm_ignores_trial_config_llm_override.
            # An optional judge_llm_provider/judge_llm_model/judge_api_key/
            # judge_base_url override is supported (same run-wide-constant rule
            # applies), falling back to the generation LLM's own config when
            # absent — mirrors Muffakir/MuffakirEvaluation.py::_resolve_llm_provider.
            provider_key = base_config.get("judge_llm_provider") or base_config["llm_provider"]
            if provider_key not in SUPPORTED_PROVIDERS:
                # composer.py's construction-time validation normally catches this
                # first, but evaluate_trial() can also be invoked directly/standalone
                # (e.g. a custom search strategy, or a test) — fail loudly instead of
                # silently misclassifying an unrecognized provider as "custom".
                raise ConfigurationError(
                    f"Unsupported judge_llm_provider/llm_provider: '{provider_key}'. "
                    f"Supported: {sorted(SUPPORTED_PROVIDERS.keys())}"
                )
            with observe_stage(
                "evaluation_judge_initialization",
                "llm",
                provider=str(provider_key),
                model=base_config.get("judge_llm_model") or base_config["llm_model"],
            ):
                from LLMProvider.parameters import resolve_parameters
                llm_provider = LLMProvider(
                    parameters=resolve_parameters(base_config, "judge"),
                    api_key=base_config.get("judge_api_key") or base_config.get("api_key"),
                    provider=SUPPORTED_PROVIDERS[provider_key],
                    model=base_config.get("judge_llm_model") or base_config["llm_model"],
                    temperature=base_config.get("llm_temperature", 0.0),
                    max_tokens=base_config.get("llm_max_tokens", 4096),
                    base_url=(
                        base_config.get("judge_base_url")
                        or base_config.get("llm_base_url")
                        or base_config.get("base_url")
                    ),
                )
        trace_queue = shared_state.get("trace_queue")
        runner = EvalRunner(
            rag=rag,
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            metrics=metrics,
            k=trial_config.get("k", base_config.get("k", 5)),
            trace_queue=trace_queue,
            trial_id=trial_id,
            price_map=price_map,
            execute_generation=(
                rag_config.get("pipeline_mode", "full_rag") != "retrieval_only"
            ),
            trace_attempt=shared_state.get("_trace_attempt"),
            observation_queue=shared_state.get("observation_queue"),
        )

        eval_report = runner.run(eval_pairs)

        # Flatten retrieval + generation metrics into a single dict
        result_metrics: Dict[str, float] = {}
        result_metrics.update(eval_report.retrieval or {})
        result_metrics.update(eval_report.generation or {})

        composite_score = _weighted_composite(result_metrics, metric_weights)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        token_usage, cost_usd = _compute_trial_cost(rag, llm_provider, price_map)
        sample_traces = getattr(runner, "collected_sample_traces", [])
        mean_pipeline_latency_ms = _mean_stage_timing(
            sample_traces, "pipeline_latency_ms"
        )
        mean_evaluation_overhead_ms = _mean_stage_timing(
            sample_traces, "evaluation_overhead_ms"
        )
        stage_timings_ms = {
            "query_transform": _mean_stage_timing(sample_traces, "query_transform_ms"),
            "query_embedding": _mean_stage_timing(sample_traces, "query_embedding_ms"),
            "vector_search": _mean_stage_timing(sample_traces, "vector_search_ms"),
            "rerank": _mean_stage_timing(sample_traces, "rerank_ms"),
            "relevance_check": _mean_stage_timing(sample_traces, "relevance_check_ms"),
            "web_search": _mean_stage_timing(sample_traces, "web_search_ms"),
            "generation": _mean_stage_timing(sample_traces, "generation_ms"),
            "hallucination_check": _mean_stage_timing(sample_traces, "hallucination_check_ms"),
        }

        if trace_queue is not None:
            _push_trial_trace(
                trace_queue, trial_id, rag_config, composite_score, result_metrics,
                latency_ms, token_usage, cost_usd, sample_traces,
            )

        return TrialResult(
            trial_id=trial_id,
            config=trial_config,
            metrics=result_metrics,
            composite_score=composite_score,
            latency_ms=latency_ms,
            token_usage=token_usage,
            cost_usd=cost_usd,
            resolved_config=redact_config(rag_config),
            mean_pipeline_latency_ms=mean_pipeline_latency_ms,
            mean_evaluation_overhead_ms=mean_evaluation_overhead_ms,
            stage_timings_ms=stage_timings_ms,
            answer_refusal_count=sum(1 for s in sample_traces if s.get("answer_refusal")),
            answer_refusal_rate=_answer_refusal_rate(sample_traces),
        )

    except MuffakirError:
        # Typed, known failure. Re-raise so BaseSearch._run_with_retry can apply
        # a type-aware retry policy and attach error_code/error_type to the
        # final TrialResult, instead of flattening it into a string here.
        raise
    except Exception as e:
        # Unexpected/unclassified failure (bug, 3rd-party quirk). Wrap into a
        # MuffakirError so the retry layer applies one uniform policy instead
        # of guessing from an arbitrary builtin exception type.
        logger.error(f"Trial {trial_id} raised an unexpected error: {e}", exc_info=True)
        raise MuffakirError(f"Unexpected trial failure: {e}", error_code="UNKNOWN_ERROR") from e


def _mean_stage_timing(sample_traces: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [s[key] for s in sample_traces if s.get(key) is not None]
    return sum(values) / len(values) if values else None


def _answer_refusal_rate(sample_traces: List[Dict[str, Any]]) -> Optional[float]:
    generated = [s for s in sample_traces if s.get("generation_attempted")]
    if not generated:
        return None
    return sum(1 for s in generated if s.get("answer_refusal")) / len(generated)


def _push_trial_trace(
    trace_queue: Any,
    trial_id: int,
    rag_config: Dict[str, Any],
    composite_score: float,
    metrics: Dict[str, float],
    latency_ms: float,
    token_usage: Dict[str, int],
    cost_usd: Optional[float],
    sample_traces: List[Dict[str, Any]],
) -> None:
    """Build a TrialRecord (mean per-stage timings from the samples EvalRunner
    already collected -- no second pass over eval_pairs) and push it."""
    from Trace.models import TrialRecord, redact_config

    record = TrialRecord(
        trial_id=trial_id,
        resolved_rag_config=redact_config(rag_config),
        composite_score=composite_score,
        metrics=metrics,
        latency_ms=latency_ms,
        mean_pipeline_latency_ms=_mean_stage_timing(sample_traces, "pipeline_latency_ms"),
        mean_evaluation_overhead_ms=_mean_stage_timing(sample_traces, "evaluation_overhead_ms"),
        token_usage=token_usage,
        cost_usd=cost_usd,
        status="success",
        mean_query_transform_ms=_mean_stage_timing(sample_traces, "query_transform_ms"),
        mean_query_embedding_ms=_mean_stage_timing(sample_traces, "query_embedding_ms"),
        mean_vector_search_ms=_mean_stage_timing(sample_traces, "vector_search_ms"),
        mean_rerank_ms=_mean_stage_timing(sample_traces, "rerank_ms"),
        mean_relevance_check_ms=_mean_stage_timing(sample_traces, "relevance_check_ms"),
        mean_web_search_ms=_mean_stage_timing(sample_traces, "web_search_ms"),
        mean_generation_ms=_mean_stage_timing(sample_traces, "generation_ms"),
        mean_hallucination_check_ms=_mean_stage_timing(sample_traces, "hallucination_check_ms"),
        web_search_fallback_count=sum(1 for s in sample_traces if s.get("web_search_used")),
        answer_refusal_count=sum(1 for s in sample_traces if s.get("answer_refusal")),
        answer_refusal_rate=_answer_refusal_rate(sample_traces),
    )
    trace_queue.put(("trial", record.to_dict()))


def _compute_trial_cost(
    rag: Any, judge_llm_provider: Any, price_map: Any
) -> Tuple[Dict[str, int], Optional[float]]:
    """
    Aggregate token usage and dollar cost across every distinct LLMProvider
    instance actually used by this trial (trial-level granularity -- see
    Trace.cost for the shared logic also used at sample-level granularity
    by Evaluation.runner.EvalRunner).

    A trial can involve up to three LLMProvider instances: `rag.llm_provider`
    (generation + hallucination-check, always present), `rag.query_transformer`'s
    provider (only distinct from the above when a per-task override is set --
    see MuffakirRAG's query_transform_llm_provider/query_transform_llm_model
    config), and the eval judge provider built in evaluate_trial(). Deduped by
    identity so a shared instance's usage is never double-counted.

    Returns (token_usage, cost_usd). token_usage is always populated when any
    usage was observed. cost_usd is the sum of whatever components could be
    priced; it is None only when NOT ONE component could be priced at all.
    """
    from Trace.cost import dedupe_llm_providers, aggregate_usage_and_cost

    providers = dedupe_llm_providers(rag, judge_llm_provider)

    def _usage_getter(provider: Any) -> Dict[str, int]:
        get_usage_totals = getattr(provider, "get_usage_totals", None)
        if get_usage_totals is None:
            # Not a real LLMProvider instance (e.g. a test double standing in
            # for one) -- no usage/cost to observe from it.
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return get_usage_totals()

    return aggregate_usage_and_cost(providers, _usage_getter, price_map)


def _weighted_composite(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Compute a weighted average of metric values, ignoring None entries.

    Unspecified weights default to 1.0. Returns 0.0 if no valid metric values.
    """
    valid = {k: v for k, v in metrics.items() if v is not None}
    if not valid:
        return 0.0
    total_weight = sum(weights.get(k, 1.0) for k in valid)
    if total_weight <= 0:
        return 0.0

    def composite_value(name: str, value: float) -> float:
        if name == "llm_judge_rating":
            # Persist/report the raw 1–5 rating, but compare it fairly with
            # existing 0–1 metrics when ranking Composer trials.
            return (float(value) - 1.0) / 4.0
        return float(value)

    weighted_sum = sum(
        composite_value(k, valid[k]) * weights.get(k, 1.0)
        for k in valid
    )
    return weighted_sum / total_weight
