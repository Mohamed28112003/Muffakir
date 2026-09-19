"""
PriceMap - Static, per-run LLM pricing lookup backed by litellm's public
model_prices_and_context_window.json, with user-supplied overrides.

Design (agreed): pricing is fetched ONCE per Composer.fit() run (never inside
a per-trial hot path), and the exact same fetched snapshot is reused across a
resumed run (see Composer/checkpoint.py's pricing_snapshot storage) so a trial
run today and one resumed three days later are priced identically. A failed
fetch never raises -- pricing must never break a trial; it just means costs
are unknown (None) for anything not covered by a custom override.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


class PriceMap:
    """
    Looks up per-token input/output pricing for a (provider, model) pair.

    Lookup order in ``get_price``: user-supplied provider-qualified key,
    legacy user-supplied model-only key, then the same two forms in the
    LiteLLM map. Provider-qualified keys prevent collisions when two APIs use
    the same model identifier, while model-only overrides remain supported.
    """

    def __init__(self, custom_pricing: Optional[Dict[str, Dict[str, float]]] = None):
        self.custom_pricing: Dict[str, Dict[str, float]] = dict(custom_pricing or {})
        self.raw_map: Dict[str, Dict[str, Any]] = {}
        self.source_url: str = DEFAULT_URL
        self.fetched_at: Optional[str] = None
        self.fetch_failed: bool = False

    def load(self, url: str = DEFAULT_URL, timeout: float = 10.0) -> None:
        """
        Fetch the litellm price map once. Never raises: on any failure,
        ``fetch_failed`` is set True and ``get_price``/``compute_cost`` will
        return None for anything not covered by a custom override.
        """
        import requests

        self.source_url = url
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch litellm price map from {url}: {e}")
            self.fetch_failed = True
            self.raw_map = {}
            self.fetched_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            return

        if not isinstance(data, dict):
            logger.warning(f"litellm price map at {url} was not a JSON object; ignoring.")
            self.fetch_failed = True
            self.raw_map = {}
        else:
            self.fetch_failed = False
            self.raw_map = data
        self.fetched_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def get_price(self, provider: str, model: str) -> Optional[Dict[str, float]]:
        """Return {"input_cost_per_token": ..., "output_cost_per_token": ...} or None."""
        provider_key = str(provider).strip().lower()
        model_key = str(model).strip()

        for key in (f"{provider_key}/{model_key}", model_key):
            override = self.custom_pricing.get(key)
            if override is not None:
                return override

        for key in (f"{provider_key}/{model_key}", model_key):
            entry = self.raw_map.get(key)
            if isinstance(entry, dict) and (
                "input_cost_per_token" in entry or "output_cost_per_token" in entry
            ):
                return entry

        return None

    def compute_cost(
        self, provider: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> Optional[float]:
        """Return the dollar cost for the given token counts, or None if pricing is unknown."""
        price = self.get_price(provider, model)
        if price is None:
            return None
        input_cost = price.get("input_cost_per_token", 0.0) or 0.0
        output_cost = price.get("output_cost_per_token", 0.0) or 0.0
        return prompt_tokens * input_cost + completion_tokens * output_cost

    def to_dict(self) -> Dict[str, Any]:
        """Serialize a picklable/JSON-able snapshot (for checkpoint storage / worker processes)."""
        return {
            "source_url": self.source_url,
            "fetched_at": self.fetched_at,
            "fetch_failed": self.fetch_failed,
            "raw_map": self.raw_map,
            "custom_pricing": self.custom_pricing,
        }

    @classmethod
    def from_dict(cls, snapshot: Dict[str, Any]) -> "PriceMap":
        """Rehydrate a PriceMap from a snapshot produced by to_dict() -- no network call."""
        instance = cls(custom_pricing=snapshot.get("custom_pricing"))
        instance.raw_map = snapshot.get("raw_map") or {}
        instance.source_url = snapshot.get("source_url", DEFAULT_URL)
        instance.fetched_at = snapshot.get("fetched_at")
        instance.fetch_failed = bool(snapshot.get("fetch_failed", False))
        return instance
