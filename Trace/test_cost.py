"""Test suite for Trace.cost (pytest)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Trace.cost import dedupe_llm_providers, aggregate_usage_and_cost


class _FakeProvider:
    def __init__(self, provider, model):
        self.provider = provider
        self.model = model


class _P:
    def __init__(self, value):
        self.value = value


# ---------------------------------------------------------------------------
# dedupe_llm_providers
# ---------------------------------------------------------------------------
def test_dedupe_includes_rag_llm_provider_only_when_no_query_transformer():
    rag_provider = _FakeProvider(_P("openai"), "gpt-4o-mini")

    class _Rag:
        llm_provider = rag_provider
        query_transformer = None

    providers = dedupe_llm_providers(_Rag(), None)
    assert providers == [rag_provider]


def test_dedupe_skips_query_transformer_sharing_same_provider():
    shared = _FakeProvider(_P("openai"), "gpt-4o-mini")

    class _QT:
        llm_provider = shared

    class _Rag:
        llm_provider = shared
        query_transformer = _QT()

    providers = dedupe_llm_providers(_Rag(), None)
    assert providers == [shared]


def test_dedupe_includes_distinct_query_transformer_provider():
    gen = _FakeProvider(_P("together"), "model-a")
    qt = _FakeProvider(_P("together"), "model-b")

    class _QT:
        llm_provider = qt

    class _Rag:
        llm_provider = gen
        query_transformer = _QT()

    providers = dedupe_llm_providers(_Rag(), None)
    assert providers == [gen, qt]


def test_dedupe_includes_judge_when_distinct():
    gen = _FakeProvider(_P("openai"), "gpt-4o-mini")
    judge = _FakeProvider(_P("openai"), "gpt-4o")

    class _Rag:
        llm_provider = gen
        query_transformer = None

    providers = dedupe_llm_providers(_Rag(), judge)
    assert providers == [gen, judge]


def test_dedupe_handles_missing_llm_provider_attribute():
    class _Rag:
        pass

    assert dedupe_llm_providers(_Rag(), None) == []


# ---------------------------------------------------------------------------
# aggregate_usage_and_cost
# ---------------------------------------------------------------------------
def test_aggregate_sums_usage_across_providers():
    p1 = _FakeProvider(_P("openai"), "gpt-4o-mini")
    p2 = _FakeProvider(_P("openai"), "gpt-4o")
    usages = {
        id(p1): {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        id(p2): {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }
    usage, cost = aggregate_usage_and_cost([p1, p2], lambda p: usages[id(p)], price_map=None)
    assert usage == {"prompt_tokens": 30, "completion_tokens": 13, "total_tokens": 43}
    assert cost is None  # no price_map -> nothing priceable


def test_aggregate_computes_cost_when_price_map_given():
    class _FakePriceMap:
        def compute_cost(self, provider, model, prompt_tokens, completion_tokens):
            return prompt_tokens * 0.001 + completion_tokens * 0.002

    p1 = _FakeProvider(_P("openai"), "gpt-4o-mini")
    usage, cost = aggregate_usage_and_cost(
        [p1], lambda p: {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}, price_map=_FakePriceMap()
    )
    assert cost == pytest.approx(100 * 0.001 + 50 * 0.002)


def test_aggregate_returns_none_cost_when_unpriceable():
    class _FakePriceMap:
        def compute_cost(self, *a, **k):
            return None

    p1 = _FakeProvider(_P("custom"), "unknown-model")
    usage, cost = aggregate_usage_and_cost(
        [p1], lambda p: {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}, price_map=_FakePriceMap()
    )
    assert usage == {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
    assert cost is None


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
