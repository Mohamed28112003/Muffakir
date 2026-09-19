"""Test suite for Trace.models (pytest)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Trace.models import RunManifest, TrialRecord, SampleTraceRecord, compute_config_hash


# ---------------------------------------------------------------------------
# compute_config_hash
# ---------------------------------------------------------------------------
def test_config_hash_deterministic():
    base_config = {"llm_provider": "openai", "llm_model": "gpt-4o-mini"}
    search_space = {"k": [3, 5]}
    h1 = compute_config_hash(base_config, search_space)
    h2 = compute_config_hash(dict(base_config), dict(search_space))
    assert h1 == h2
    assert len(h1) == 16


def test_config_hash_ignores_api_key():
    """Two runs differing only by api_key must hash identically -- the hash
    identifies the search configuration, not the credentials used."""
    search_space = {"k": [3]}
    h1 = compute_config_hash({"llm_provider": "openai", "api_key": "sk-aaa"}, search_space)
    h2 = compute_config_hash({"llm_provider": "openai", "api_key": "sk-bbb"}, search_space)
    assert h1 == h2


def test_config_hash_ignores_nested_secrets():
    search_space = {"k": [3]}
    h1 = compute_config_hash(
        {"search_provider_config": {"max_results": 5, "api_key": "first"}},
        search_space,
    )
    h2 = compute_config_hash(
        {"search_provider_config": {"max_results": 5, "api_key": "second"}},
        search_space,
    )
    assert h1 == h2


def test_config_hash_ignores_reranker_credentials():
    search_space = {"reranking": ["llm", "custom"]}
    first = {
        "reranker_api_key": "remote-first",
        "reranker_llm_api_key": "llm-first",
    }
    second = {
        "reranker_api_key": "remote-second",
        "reranker_llm_api_key": "llm-second",
    }
    assert compute_config_hash(first, search_space) == compute_config_hash(
        second, search_space
    )


def test_config_hash_differs_on_real_change():
    h1 = compute_config_hash({"llm_provider": "openai"}, {"k": [3]})
    h2 = compute_config_hash({"llm_provider": "openai"}, {"k": [3, 5]})
    assert h1 != h2


# ---------------------------------------------------------------------------
# RunManifest
# ---------------------------------------------------------------------------
def test_run_manifest_defaults():
    m = RunManifest()
    assert m.schema_version == "1.3"
    assert m.status == "running"
    assert m.run_id  # non-empty uuid string


def test_run_manifest_round_trip():
    m = RunManifest(config_hash="abc123", status="completed", total_trials=4, search_space={"k": [3, 5]})
    restored = RunManifest.from_dict(m.to_dict())
    assert restored == m


def test_run_manifest_from_dict_ignores_unknown_keys():
    RunManifest.from_dict({"schema_version": "1.0", "status": "running", "future_field": "x"})


# ---------------------------------------------------------------------------
# TrialRecord
# ---------------------------------------------------------------------------
def test_trial_record_round_trip():
    t = TrialRecord(
        trial_id=3,
        resolved_rag_config={"k": 5},
        composite_score=0.8,
        metrics={"recall": 0.9},
        latency_ms=120.0,
        token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        cost_usd=0.01,
        mean_query_embedding_ms=2.5,
        mean_vector_search_ms=8.1,
    )
    restored = TrialRecord.from_dict(t.to_dict())
    assert restored == t


def test_trial_record_defaults_are_success():
    t = TrialRecord(trial_id=0)
    assert t.status == "success"
    assert t.error is None
    assert t.cost_usd is None


# ---------------------------------------------------------------------------
# SampleTraceRecord
# ---------------------------------------------------------------------------
def test_sample_trace_record_round_trip():
    s = SampleTraceRecord(
        trial_id=1,
        sample_index=2,
        question="q?",
        transformed_query=["q?", "broader q?"],
        query_transform_strategy="step_back",
        gold_answer="a",
        gold_context="reference context",
        predicted_answer="a",
        context_source="web_search",
        query_transform_ms=1.0,
        query_embedding_ms=2.0,
        vector_search_ms=3.0,
        rerank_ms=0.5,
        generation_ms=40.0,
        hallucination_check_ms=5.0,
        retrieved_candidates=[{"page_content": "x", "metadata": {}}],
        metrics={"recall_at_k": 1.0},
        token_usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        cost_usd=0.001,
        generation_attempted=True,
        answer_refusal=True,
        answer_refusal_reason="insufficient_information",
    )
    restored = SampleTraceRecord.from_dict(s.to_dict())
    assert restored == s


def test_sample_trace_record_error_case_has_no_timings_required():
    s = SampleTraceRecord(trial_id=0, sample_index=0, error="boom")
    assert s.query_embedding_ms is None
    assert s.error == "boom"


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
