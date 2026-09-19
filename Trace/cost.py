"""
Shared LLM-provider dedup + usage/cost aggregation, used at both trial-level
(Composer.evaluation._compute_trial_cost) and sample-level (Evaluation.runner)
granularity so both call the same logic instead of drifting apart.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

UsageGetter = Callable[[Any], Dict[str, int]]


def dedupe_llm_providers(rag: Any, judge_llm_provider: Optional[Any]) -> List[Any]:
    """
    Collect the distinct LLMProvider instances actually used for a trial/sample:
    rag.llm_provider (generation + hallucination-check, always present when rag
    is a real MuffakirRAG), rag.query_transformer's provider (only distinct from
    the above when a per-task override is set), and judge_llm_provider (the
    eval judge, built separately). Deduped by identity so a shared instance's
    usage is never double-counted.
    """
    providers: List[Any] = []

    rag_llm_provider = getattr(rag, "llm_provider", None)
    if rag_llm_provider is not None:
        providers.append(rag_llm_provider)

    query_transformer = getattr(rag, "query_transformer", None)
    qt_provider = getattr(query_transformer, "llm_provider", None) if query_transformer else None
    if qt_provider is not None and all(qt_provider is not p for p in providers):
        providers.append(qt_provider)

    if judge_llm_provider is not None and all(judge_llm_provider is not p for p in providers):
        providers.append(judge_llm_provider)

    return providers


def aggregate_usage_and_cost(
    providers: List[Any],
    usage_getter: UsageGetter,
    price_map: Optional[Any],
) -> Tuple[Dict[str, int], Optional[float]]:
    """
    Sum token usage (via usage_getter(provider) for each provider) and, when
    price_map can price a given provider's (provider, model), sum whatever
    dollar cost is computable.

    Returns (token_usage, cost_usd). cost_usd is None only when NOT ONE
    component could be priced at all -- token_usage is always populated when
    any usage was observed, even if pricing is entirely unknown.
    """
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_cost = 0.0
    any_priced = False

    for provider in providers:
        usage = usage_getter(provider)
        for key in total_usage:
            total_usage[key] += usage.get(key, 0)

        if price_map is None:
            continue
        provider_enum = getattr(provider, "provider", None)
        model_name = getattr(provider, "model", None)
        if provider_enum is None or model_name is None:
            continue
        provider_name = getattr(provider_enum, "value", str(provider_enum))
        cost = price_map.compute_cost(
            provider_name, model_name, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        )
        if cost is not None:
            any_priced = True
            total_cost += cost

    return total_usage, (total_cost if any_priced else None)
