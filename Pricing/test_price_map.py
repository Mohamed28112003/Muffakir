"""
Test suite for PriceMap (pytest).

requests.get is monkeypatched in every test that would otherwise hit the
network -- no real HTTP calls are made.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Pricing.price_map import PriceMap, DEFAULT_URL


class _FakeResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._payload


def _patch_requests_get(monkeypatch, payload, status_ok=True):
    import requests

    def fake_get(url, timeout=None):
        return _FakeResponse(payload, status_ok=status_ok)

    monkeypatch.setattr(requests, "get", fake_get)


# ---------------------------------------------------------------------------
# load() / fetch semantics
# ---------------------------------------------------------------------------
def test_load_success_populates_raw_map(monkeypatch):
    _patch_requests_get(monkeypatch, {"gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001}})
    pm = PriceMap()
    pm.load()
    assert pm.fetch_failed is False
    assert pm.raw_map["gpt-4o"]["input_cost_per_token"] == 0.0000025
    assert pm.fetched_at is not None


def test_load_network_failure_sets_fetch_failed(monkeypatch):
    import requests

    def raising_get(url, timeout=None):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(requests, "get", raising_get)

    pm = PriceMap()
    pm.load()  # must not raise
    assert pm.fetch_failed is True
    assert pm.raw_map == {}


def test_load_http_error_sets_fetch_failed(monkeypatch):
    _patch_requests_get(monkeypatch, {"gpt-4o": {}}, status_ok=False)
    pm = PriceMap()
    pm.load()
    assert pm.fetch_failed is True


def test_load_non_dict_payload_sets_fetch_failed(monkeypatch):
    _patch_requests_get(monkeypatch, ["not", "a", "dict"])
    pm = PriceMap()
    pm.load()
    assert pm.fetch_failed is True
    assert pm.raw_map == {}


def test_load_uses_default_url(monkeypatch):
    captured = {}

    def fake_get(url, timeout=None):
        captured["url"] = url
        return _FakeResponse({})

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    PriceMap().load()
    assert captured["url"] == DEFAULT_URL


# ---------------------------------------------------------------------------
# get_price() lookup order
# ---------------------------------------------------------------------------
def test_get_price_finds_bare_model_key():
    pm = PriceMap()
    pm.raw_map = {"gpt-4o": {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2}}
    assert pm.get_price("openai", "gpt-4o") == {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2}


def test_get_price_falls_back_to_provider_prefixed_key():
    pm = PriceMap()
    pm.raw_map = {"groq/llama3-70b-8192": {"input_cost_per_token": 0.05, "output_cost_per_token": 0.08}}
    assert pm.get_price("groq", "llama3-70b-8192") == {"input_cost_per_token": 0.05, "output_cost_per_token": 0.08}


def test_provider_prefixed_builtin_price_wins_over_bare_model():
    pm = PriceMap()
    pm.raw_map = {
        "same-model": {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2},
        "groq/same-model": {"input_cost_per_token": 0.3, "output_cost_per_token": 0.4},
    }
    assert pm.get_price("groq", "same-model") == {
        "input_cost_per_token": 0.3,
        "output_cost_per_token": 0.4,
    }


def test_get_price_unknown_model_returns_none():
    pm = PriceMap()
    pm.raw_map = {"gpt-4o": {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2}}
    assert pm.get_price("openai", "some-unknown-model") is None


def test_custom_pricing_overrides_litellm_map():
    pm = PriceMap(custom_pricing={"gpt-4o": {"input_cost_per_token": 999.0, "output_cost_per_token": 999.0}})
    pm.raw_map = {"gpt-4o": {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2}}
    assert pm.get_price("openai", "gpt-4o") == {"input_cost_per_token": 999.0, "output_cost_per_token": 999.0}


def test_custom_pricing_covers_models_absent_from_litellm_map():
    pm = PriceMap(custom_pricing={"my-local-model": {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}})
    assert pm.get_price("custom", "my-local-model") == {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}


def test_provider_qualified_custom_prices_do_not_collide():
    pm = PriceMap(
        custom_pricing={
            "openai/shared": {"input_cost_per_token": 1.0, "output_cost_per_token": 2.0},
            "groq/shared": {"input_cost_per_token": 3.0, "output_cost_per_token": 4.0},
        }
    )
    assert pm.get_price("openai", "shared")["input_cost_per_token"] == 1.0
    assert pm.get_price("groq", "shared")["input_cost_per_token"] == 3.0


def test_provider_qualified_custom_price_wins_over_legacy_model_override():
    pm = PriceMap(
        custom_pricing={
            "shared": {"input_cost_per_token": 1.0, "output_cost_per_token": 2.0},
            "groq/shared": {"input_cost_per_token": 3.0, "output_cost_per_token": 4.0},
        }
    )
    assert pm.get_price("groq", "shared")["input_cost_per_token"] == 3.0
    assert pm.get_price("openai", "shared")["input_cost_per_token"] == 1.0


# ---------------------------------------------------------------------------
# compute_cost()
# ---------------------------------------------------------------------------
def test_compute_cost_matches_hand_computed_dollars():
    pm = PriceMap()
    pm.raw_map = {"gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001}}
    cost = pm.compute_cost("openai", "gpt-4o", prompt_tokens=1000, completion_tokens=500)
    assert cost == pytest.approx(1000 * 0.0000025 + 500 * 0.00001)


def test_compute_cost_unknown_model_returns_none():
    pm = PriceMap()
    assert pm.compute_cost("openai", "unknown-model", 100, 50) is None


def test_compute_cost_missing_output_cost_treated_as_zero():
    pm = PriceMap()
    pm.raw_map = {"m": {"input_cost_per_token": 0.01}}
    assert pm.compute_cost("p", "m", 100, 50) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# to_dict() / from_dict() round-trip
# ---------------------------------------------------------------------------
def test_to_dict_from_dict_round_trip():
    pm = PriceMap(custom_pricing={"x": {"input_cost_per_token": 1.0, "output_cost_per_token": 2.0}})
    pm.raw_map = {"gpt-4o": {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2}}
    pm.fetched_at = "2026-08-30T12:00:00"
    pm.fetch_failed = False

    snapshot = pm.to_dict()
    restored = PriceMap.from_dict(snapshot)

    assert restored.raw_map == pm.raw_map
    assert restored.custom_pricing == pm.custom_pricing
    assert restored.fetched_at == pm.fetched_at
    assert restored.fetch_failed == pm.fetch_failed
    assert restored.get_price("openai", "gpt-4o") == {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2}


def test_from_dict_never_makes_network_call(monkeypatch):
    import requests

    def raising_get(*a, **k):
        raise AssertionError("from_dict must not call requests.get")

    monkeypatch.setattr(requests, "get", raising_get)
    snapshot = {"raw_map": {"m": {"input_cost_per_token": 0.1}}, "custom_pricing": {}, "fetched_at": "t", "fetch_failed": False, "source_url": DEFAULT_URL}
    restored = PriceMap.from_dict(snapshot)
    assert restored.get_price("p", "m") is not None


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
