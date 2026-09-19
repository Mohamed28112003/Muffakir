"""Query transformation public API with lazy strategy imports."""

from Muffakir._lazy import load_attribute
from .QueryTransformer import QueryTransformer

__all__ = [
    "BaseQueryTransformer", "QueryRewriter", "MultiQueryExpansion",
    "QueryExpansionOutput", "QueryDecomposition", "SubQueryDecomposition",
    "HyDEQueryTransformer", "StepBackQueryTransformer", "StepBackQueryOutput",
    "create_query_transformer", "QueryTransformer",
]
_LAZY_EXPORTS = {
    "BaseQueryTransformer": ("QueryTransformer.base", "BaseQueryTransformer"),
    "QueryRewriter": ("QueryTransformer.rewriter", "QueryRewriter"),
    "MultiQueryExpansion": ("QueryTransformer.multi_query", "MultiQueryExpansion"),
    "QueryExpansionOutput": ("QueryTransformer.multi_query", "QueryExpansionOutput"),
    "QueryDecomposition": ("QueryTransformer.query_decomposition", "QueryDecomposition"),
    "SubQueryDecomposition": ("QueryTransformer.query_decomposition", "SubQueryDecomposition"),
    "HyDEQueryTransformer": ("QueryTransformer.hyde", "HyDEQueryTransformer"),
    "StepBackQueryTransformer": ("QueryTransformer.step_back", "StepBackQueryTransformer"),
    "StepBackQueryOutput": ("QueryTransformer.step_back", "StepBackQueryOutput"),
    "create_query_transformer": ("QueryTransformer.factory", "create_query_transformer"),
}


def __getattr__(name):
    return load_attribute(name, _LAZY_EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
