"""Pricing public API with lazy network-client imports."""

from Muffakir._lazy import load_attribute

__all__ = ["PriceMap"]
_LAZY_EXPORTS = {"PriceMap": ("Pricing.price_map", "PriceMap")}


def __getattr__(name):
    return load_attribute(name, _LAZY_EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
