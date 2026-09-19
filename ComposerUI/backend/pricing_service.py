"""Per-run custom-pricing normalization and inspection helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from Evaluation.constants import GENERATION_METRICS

if TYPE_CHECKING:
    from ComposerUI.backend.schemas import CreateRunRequest


_LLM_RERANKERS = {"llm", "llm_reranker", "llm-based", "llm_based"}
def _pair(provider: Optional[str], model: Optional[str]) -> Optional[Tuple[str, str]]:
    provider_value = str(provider or "").strip().lower()
    model_value = str(model or "").strip()
    if not provider_value or not model_value:
        return None
    return provider_value, model_value


def resolve_pricing_models(request: "CreateRunRequest") -> List[Dict[str, Any]]:
    """Return every token-priced provider/model pair and its run roles."""
    roles: Dict[Tuple[str, str], set[str]] = {}

    def add(provider: Optional[str], model: Optional[str], role: str) -> None:
        key = _pair(provider, model)
        if key is not None:
            roles.setdefault(key, set()).add(role)

    answer_models: List[Tuple[str, str]] = []
    if request.pipeline_mode == "full_rag":
        if request.search_space.llm:
            answer_models = [
                pair
                for entry in request.search_space.llm
                if (pair := _pair(entry.provider, entry.model)) is not None
            ]
        else:
            pair = _pair(request.llm_provider, request.llm_model)
            if pair is not None:
                answer_models = [pair]
        for provider, model in answer_models:
            add(provider, model, "Answer")

    query_values = request.search_space.query_expansion
    if query_values is None:
        query_values = ["none", "multi_query", "hyde", "step_back"]
    query_transform_enabled = (
        request.retrieval_source != "web_search_only"
        and any(str(value).lower() != "none" for value in query_values)
    )
    if query_transform_enabled:
        explicit_query_model = _pair(
            request.query_transform_llm_provider,
            request.query_transform_llm_model,
        )
        query_models = [explicit_query_model] if explicit_query_model else answer_models
        for provider, model in filter(None, query_models):
            add(provider, model, "Query Transform")

    reranking_values = request.search_space.reranking
    if reranking_values is None:
        reranking_values = ["none", "semantic_similarity", "cross_encoder"]
    llm_reranking_enabled = any(
        str(value).lower() in _LLM_RERANKERS for value in reranking_values
    )
    if llm_reranking_enabled:
        explicit_reranker_model = _pair(
            request.reranker_llm_provider,
            request.reranker_llm_model,
        )
        reranker_models = [explicit_reranker_model] if explicit_reranker_model else answer_models
        if request.pipeline_mode == "retrieval_only" and not explicit_reranker_model:
            explicit_query_model = _pair(
                request.query_transform_llm_provider,
                request.query_transform_llm_model,
            )
            reranker_models = [explicit_query_model] if explicit_query_model else []
        for provider, model in filter(None, reranker_models):
            add(provider, model, "LLM Reranker")

    metrics = request.metrics or ["recall", "faithfulness", "answer_correctness"]
    generation_judge_enabled = (
        request.pipeline_mode == "full_rag"
        and any(str(metric).lower() in GENERATION_METRICS for metric in metrics)
    )
    if generation_judge_enabled:
        add(
            request.judge_llm_provider or request.llm_provider,
            request.judge_llm_model or request.llm_model,
            "Judge",
        )

    custom_keys = {
        (entry.provider, entry.model) for entry in request.custom_pricing
    }
    for provider, model in custom_keys:
        roles.setdefault((provider, model), {"Manual Override"})

    return [
        {
            "provider": provider,
            "model": model,
            "roles": sorted(model_roles),
            "pricing_source": "custom" if (provider, model) in custom_keys else "default",
        }
        for (provider, model), model_roles in sorted(roles.items())
    ]


def read_pricing_snapshot(checkpoint_dir: Path) -> Dict[str, Any]:
    """Read a checkpoint pricing snapshot without failing run inspection."""
    checkpoint_file = checkpoint_dir / "composer_checkpoint.json"
    if not checkpoint_file.exists():
        return {}
    try:
        data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    snapshot = data.get("pricing_snapshot")
    return snapshot if isinstance(snapshot, dict) else {}


def build_pricing_summary(
    config: Dict[str, Any], snapshot: Dict[str, Any]
) -> Dict[str, Any]:
    """Combine persisted model roles with the exact price map used by a run."""
    from Pricing.price_map import PriceMap

    price_map = PriceMap.from_dict(snapshot)
    custom_map = price_map.custom_pricing
    rows = []
    for item in config.get("pricing_models") or []:
        provider = str(item.get("provider") or "").strip().lower()
        model = str(item.get("model") or "").strip()
        if not provider or not model:
            continue
        provider_model_key = f"{provider}/{model}"
        is_custom = provider_model_key in custom_map or model in custom_map
        rate = price_map.get_price(provider, model)
        rows.append(
            {
                "provider": provider,
                "model": model,
                "roles": list(item.get("roles") or []),
                "pricing_source": "custom" if is_custom else "default",
                "input_usd_per_million_tokens": (
                    float(rate.get("input_cost_per_token", 0.0)) * 1_000_000
                    if rate is not None
                    else None
                ),
                "output_usd_per_million_tokens": (
                    float(rate.get("output_cost_per_token", 0.0)) * 1_000_000
                    if rate is not None
                    else None
                ),
            }
        )

    return {
        "unit": "usd_per_million_tokens",
        "mode": "custom" if config.get("custom_pricing") else "default",
        "models": rows,
        "source_url": snapshot.get("source_url"),
        "fetched_at": snapshot.get("fetched_at"),
        "fetch_failed": snapshot.get("fetch_failed"),
    }
