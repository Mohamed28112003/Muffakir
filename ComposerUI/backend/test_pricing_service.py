"""Tests for ComposerUI per-run pricing resolution and inspection."""

import json

from ComposerUI.backend.pricing_service import (
    build_pricing_summary,
    read_pricing_snapshot,
    resolve_pricing_models,
)
from ComposerUI.backend.schemas import CreateRunRequest


def test_resolve_models_deduplicates_roles_and_keeps_provider_identity():
    request = CreateRunRequest(
        run_name="pricing",
        llm_provider="openai",
        llm_model="shared-model",
        query_transform_llm_provider="groq",
        query_transform_llm_model="shared-model",
        judge_llm_provider="anthropic",
        judge_llm_model="judge-model",
        search_space={
            "query_expansion": ["none", "rewrite"],
            "reranking": ["none", "llm"],
            "llm": [
                {"provider": "openai", "model": "shared-model"},
                {"provider": "groq", "model": "shared-model"},
            ],
        },
        metrics=["faithfulness"],
    )

    models = resolve_pricing_models(request)
    by_key = {(item["provider"], item["model"]): item for item in models}
    assert set(by_key) == {
        ("anthropic", "judge-model"),
        ("groq", "shared-model"),
        ("openai", "shared-model"),
    }
    assert by_key[("groq", "shared-model")]["roles"] == [
        "Answer",
        "LLM Reranker",
        "Query Transform",
    ]
    assert by_key[("anthropic", "judge-model")]["roles"] == ["Judge"]


def test_manual_custom_model_is_preserved_for_inspection():
    request = CreateRunRequest(
        run_name="manual",
        pipeline_mode="retrieval_only",
        search_space={"query_expansion": ["none"], "reranking": ["none"]},
        metrics=["recall"],
        custom_pricing=[
            {
                "provider": "custom",
                "model": "future-model",
                "input_usd_per_million_tokens": 0,
                "output_usd_per_million_tokens": 0,
            }
        ],
    )
    models = resolve_pricing_models(request)
    assert models == [
        {
            "provider": "custom",
            "model": "future-model",
            "roles": ["Manual Override"],
            "pricing_source": "custom",
        }
    ]


def test_llm_judge_rating_enables_judge_pricing_role():
    request = CreateRunRequest(
        run_name="rating-pricing",
        llm_provider="openai",
        llm_model="answer-model",
        judge_llm_provider="anthropic",
        judge_llm_model="rating-model",
        metrics=["llm_judge_rating"],
    )

    by_key = {
        (item["provider"], item["model"]): item
        for item in resolve_pricing_models(request)
    }
    assert by_key[("anthropic", "rating-model")]["roles"] == ["Judge"]


def test_dedicated_reranker_llm_is_priced_instead_of_answer_llm():
    request = CreateRunRequest(
        run_name="dedicated-reranker",
        llm_provider="openai",
        llm_model="answer-model",
        reranker_llm_provider="groq",
        reranker_llm_model="rerank-model",
        search_space={
            "query_expansion": ["none"],
            "reranking": ["llm"],
        },
        metrics=["recall"],
    )

    by_key = {
        (item["provider"], item["model"]): item
        for item in resolve_pricing_models(request)
    }
    assert by_key[("groq", "rerank-model")]["roles"] == ["LLM Reranker"]
    assert "LLM Reranker" not in by_key[("openai", "answer-model")]["roles"]


def test_pricing_summary_uses_exact_snapshot_and_preserves_zero():
    config = {
        "custom_pricing": [
            {
                "provider": "openai",
                "model": "free-model",
                "input_usd_per_million_tokens": 0,
                "output_usd_per_million_tokens": 0,
            }
        ],
        "pricing_models": [
            {"provider": "openai", "model": "free-model", "roles": ["Answer"]},
            {"provider": "groq", "model": "paid-model", "roles": ["Judge"]},
        ],
    }
    snapshot = {
        "custom_pricing": {
            "openai/free-model": {
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            }
        },
        "raw_map": {
            "groq/paid-model": {
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000002,
            }
        },
        "fetch_failed": False,
        "fetched_at": "2026-09-10T00:00:00",
    }
    summary = build_pricing_summary(config, snapshot)
    free, paid = summary["models"]
    assert free["pricing_source"] == "custom"
    assert free["input_usd_per_million_tokens"] == 0.0
    assert paid["pricing_source"] == "default"
    assert paid["input_usd_per_million_tokens"] == 1.0
    assert paid["output_usd_per_million_tokens"] == 2.0


def test_read_pricing_snapshot_handles_missing_and_valid_checkpoint(tmp_path):
    assert read_pricing_snapshot(tmp_path) == {}
    (tmp_path / "composer_checkpoint.json").write_text(
        json.dumps({"pricing_snapshot": {"raw_map": {"m": {}}}}),
        encoding="utf-8",
    )
    assert read_pricing_snapshot(tmp_path)["raw_map"] == {"m": {}}
