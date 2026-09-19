"""Automated architecture-search public API with lazy imports."""

from Muffakir._lazy import load_attribute

__all__ = ["MuffakirComposer", "ConfigSpace", "DEFAULT_SEARCH_SPACE", "TrialResult", "ComposerReport"]
_LAZY_EXPORTS = {
    "MuffakirComposer": ("Composer.composer", "MuffakirComposer"),
    "ConfigSpace": ("Composer.config_space", "ConfigSpace"),
    "DEFAULT_SEARCH_SPACE": ("Composer.config_space", "DEFAULT_SEARCH_SPACE"),
    "TrialResult": ("Composer.results.trial", "TrialResult"),
    "ComposerReport": ("Composer.results.report", "ComposerReport"),
}


def __getattr__(name):
    return load_attribute(name, _LAZY_EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
