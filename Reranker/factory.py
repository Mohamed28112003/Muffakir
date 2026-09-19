"""Extensible reranker registry and built-in strategy factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .base import BaseReranker
from Muffakir.optional_dependencies import require_optional_dependency


RerankerFactory = Callable[..., BaseReranker]


@dataclass(frozen=True)
class RerankerSpec:
    """Metadata for one selectable reranking strategy."""

    name: str
    label: str
    description: str
    factory: RerankerFactory
    aliases: Tuple[str, ...] = ()
    dependency: Optional[str] = None
    configuration: str = "none"


_REGISTRY: Dict[str, RerankerSpec] = {}
_ALIASES: Dict[str, str] = {}


def register_reranker(
    name: str,
    factory: RerankerFactory,
    *,
    label: Optional[str] = None,
    description: str = "",
    aliases: Tuple[str, ...] = (),
    dependency: Optional[str] = None,
    configuration: str = "none",
    replace: bool = False,
) -> None:
    """Register a reranker factory for SDK users and extensions.

    Factories receive the same keyword context as :func:`create_reranker`.
    Names and aliases are normalized to lowercase.
    """
    canonical = str(name).strip().lower()
    if not canonical:
        raise ValueError("Reranker name cannot be empty.")
    normalized_aliases = tuple(str(alias).strip().lower() for alias in aliases)
    occupied = [key for key in (canonical, *normalized_aliases) if key in _ALIASES]
    if occupied and not replace:
        raise ValueError(f"Reranker name or alias already registered: {occupied[0]}")

    if replace and canonical in _REGISTRY:
        old = _REGISTRY[canonical]
        for key in (canonical, *old.aliases):
            _ALIASES.pop(key, None)

    spec = RerankerSpec(
        name=canonical,
        label=label or canonical.replace("_", " ").title(),
        description=description,
        factory=factory,
        aliases=normalized_aliases,
        dependency=dependency,
        configuration=configuration,
    )
    _REGISTRY[canonical] = spec
    for key in (canonical, *normalized_aliases):
        _ALIASES[key] = canonical


def get_reranker_spec(name: str) -> RerankerSpec:
    normalized = str(name).strip().lower()
    canonical = _ALIASES.get(normalized)
    if canonical is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown reranking method: '{name}'. Available: {available}."
        )
    return _REGISTRY[canonical]


def list_reranker_specs() -> Tuple[RerankerSpec, ...]:
    """Return registered strategies in stable registration order."""
    return tuple(_REGISTRY.values())


def create_reranker(
    method: str = "semantic_similarity",
    embedding_provider: Optional[Any] = None,
    llm_provider: Optional[Any] = None,
    prompt_manager: Optional[Any] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> BaseReranker:
    """Instantiate a registered reranking strategy."""
    spec = get_reranker_spec(method)
    if spec.dependency:
        require_optional_dependency(spec.dependency)
    return spec.factory(
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        prompt_manager=prompt_manager,
        model_name=model_name,
        **kwargs,
    )


def _semantic_factory(**context: Any) -> BaseReranker:
    from .semantic_similarity import SemanticSimilarityReranker

    return SemanticSimilarityReranker(
        embedding_provider=context.get("embedding_provider"),
        model_name=context.get("model_name") or "mohamed2811/Muffakir_Embedding",
    )


def _bm25_factory(**_context: Any) -> BaseReranker:
    from .bm25 import BM25Reranker

    return BM25Reranker()


def _cross_encoder_factory(**context: Any) -> BaseReranker:
    from .cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(
        model_name=context.get("model_name") or "BAAI/bge-reranker-base",
        device=context.get("device", "auto"),
    )


def _pointwise_factory(**context: Any) -> BaseReranker:
    from .pointwise import PointwiseReranker

    return PointwiseReranker(
        model_name=context.get("model_name") or "BAAI/bge-reranker-base",
        relevance_threshold=context.get("relevance_threshold", 0.0),
        device=context.get("device", "auto"),
    )


def _llm_factory(**context: Any) -> BaseReranker:
    from .llm import LLMReranker

    return LLMReranker(
        llm_provider=context.get("llm_provider"),
        prompt_manager=context.get("prompt_manager"),
    )


def _remote_factory(**context: Any) -> BaseReranker:
    from .remote import RemoteReranker

    return RemoteReranker(
        base_url=context.get("remote_base_url"),
        api_key=context.get("remote_api_key"),
        model=context.get("remote_model"),
        timeout_seconds=context.get("remote_timeout", 30.0),
        options=context.get("remote_options"),
    )


register_reranker(
    "semantic_similarity", _semantic_factory,
    aliases=("cosine", "embedding"), dependency="local",
    label="Semantic similarity", configuration="model",
    description="Scores candidates with embedding similarity.",
)
register_reranker(
    "bm25", _bm25_factory, aliases=("bm25_reranker",), dependency="bm25",
    label="BM25", description="Uses lexical BM25 relevance scoring.",
)
register_reranker(
    "cross_encoder", _cross_encoder_factory, aliases=("crossencoder",),
    dependency="local", label="Cross encoder", configuration="model",
    description="Scores each query and document jointly with a local model.",
)
register_reranker(
    "pointwise", _pointwise_factory,
    aliases=("pointwise_ltr", "pointwise_l2r"), dependency="local",
    label="Pointwise", configuration="model",
    description="Scores candidates independently with a local relevance model.",
)
register_reranker(
    "llm", _llm_factory, aliases=("llm_reranker", "llm_based", "llm-based"),
    label="LLM reranker", configuration="llm",
    description="Uses an answer or dedicated LLM to judge relevance.",
)
register_reranker(
    "custom", _remote_factory, aliases=("remote", "http"),
    label="Custom endpoint", configuration="remote",
    description="Calls a user-provided HTTP reranking service.",
)
