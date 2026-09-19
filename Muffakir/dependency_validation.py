"""Resolve and validate optional capabilities required by a configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from Evaluation.constants import GENERATION_METRICS

from .optional_dependencies import require_optional_dependencies


_LLM_FEATURES = {
    "groq": "groq",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
    "ollama": "ollama",
    "openai": "openai",
    "azure": "openai",
    "azure_openai": "openai",
    "together": "openai",
    "openrouter": "openai",
    "custom": "openai",
    "deepseek": "openai",
    "vllm": "openai",
}

_VECTOR_FEATURES = {
    "chroma": "chroma",
    "faiss": "faiss",
    "qdrant": "qdrant",
    "milvus": "milvus",
    "pinecone": "pinecone",
}

_PARSER_FEATURES = {
    "azure": "azure",
    "docling": "docling",
    "llama_parse": "llamaparse",
    "llamaparse": "llamaparse",
    "llama-parse": "llamaparse",
}

_WEB_FEATURES = {
    "firecrawl": "firecrawl",
    "fire_crawl": "firecrawl",
    "fire-crawl": "firecrawl",
    "tavily": "tavily",
    "serpapi": "serpapi",
    "serp_api": "serpapi",
    "serp-api": "serpapi",
    "google": "serpapi",
}


def _values(search_space: Mapping[str, Any], key: str) -> List[Any]:
    value = search_space.get(key, [])
    return list(value) if isinstance(value, (list, tuple)) else []


def _add_llm(features: List[str], provider: Any) -> None:
    if not provider:
        return
    feature = _LLM_FEATURES.get(str(provider).lower().strip())
    if feature:
        features.append(feature)


def _corpus_contains_pdf(data_dir: Any) -> bool:
    if not data_dir:
        return False
    path = Path(str(data_dir)).expanduser()
    try:
        if path.is_file():
            return path.suffix.lower() == ".pdf"
        return path.is_dir() and any(path.rglob("*.pdf"))
    except OSError:
        return False


def required_features_for_composer(
    config: Mapping[str, Any],
    search_space: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Return the de-duplicated capability extras a Composer run can execute."""

    space: Mapping[str, Any] = search_space or {}
    features: List[str] = ["rag", "datasets"]
    web_only = config.get("retrieval_source") == "web_search_only"

    if not web_only:
        # Composer currently builds local/HuggingFace embeddings for every
        # local index.  Model IDs are a search dimension, not a provider.
        features.append("local")

        vector_values = _values(space, "vector_db_provider") or [
            config.get("vector_db_provider") or "chroma"
        ]
        for provider in vector_values:
            feature = _VECTOR_FEATURES.get(str(provider).lower().strip())
            if feature:
                features.append(feature)

        chunking_values = _values(space, "chunking")
        chunking_methods = [
            value.get("method") if isinstance(value, dict) else value
            for value in chunking_values
        ] or [config.get("chunking_method") or "recursive"]
        if any(str(method).lower() == "token" for method in chunking_methods):
            features.append("token")
        if any(str(method).lower() == "semantic" for method in chunking_methods):
            features.append("semantic")


        retrieval_values = _values(space, "retrieval") or [
            config.get("retrieval_method") or "similarity_search"
        ]
        if any(str(method).lower() in {"hybrid", "hybridrag", "bm25"} for method in retrieval_values):
            features.append("bm25")

        reranking_values = _values(space, "reranking") or [
            config.get("reranking_method") if config.get("reranking") else "none"
        ]
        if any(str(method).lower() in {"bm25", "bm25_reranker"} for method in reranking_values):
            features.append("bm25")

        parser = config.get("document_parser")
        if parser:
            feature = _PARSER_FEATURES.get(str(parser).lower().strip())
            if feature:
                features.append(feature)
        elif _corpus_contains_pdf(config.get("data_dir")):
            features.append("pdf")

    # Every possible selected LLM must be available, including trial-level
    # model sweeps and task-specific overrides.
    _add_llm(features, config.get("llm_provider"))
    _add_llm(features, config.get("dataset_llm_provider"))
    _add_llm(features, config.get("judge_llm_provider"))
    _add_llm(features, config.get("query_transform_llm_provider"))
    _add_llm(features, config.get("reranker_llm_provider"))
    for value in _values(space, "llm"):
        if isinstance(value, dict):
            _add_llm(features, value.get("provider"))

    if web_only or config.get("adaptive_web_search"):
        provider = config.get("search_provider")
        feature = _WEB_FEATURES.get(str(provider).lower().strip()) if provider else None
        if feature:
            features.append(feature)

    return list(dict.fromkeys(features))


def required_features_for_rag(
    config: Mapping[str, Any],
    *,
    injected_embedding: bool = False,
    injected_vector_db: bool = False,
    injected_llm: bool = False,
) -> List[str]:
    """Return capabilities required to construct one ``MuffakirRAG``."""

    features: List[str] = ["rag"]
    if not injected_embedding:
        embedding_provider = str(
            config.get("embedding_provider") or "sentence_transformers"
        ).lower().strip()
        features.append(
            {"openai": "openai", "langchain_openai": "openai", "cohere": "cohere"}.get(
                embedding_provider, "local"
            )
        )
    if not injected_vector_db:
        vector = str(config.get("vector_db_provider") or "chroma").lower().strip()
        if vector in _VECTOR_FEATURES:
            features.append(_VECTOR_FEATURES[vector])
    if not injected_llm:
        _add_llm(features, config.get("llm_provider"))

    chunking = str(config.get("chunking_method") or "recursive").lower().strip()
    if chunking == "token":
        features.append("token")
    elif chunking == "semantic":
        features.append("semantic")

    retrieval = str(config.get("retrieval_method") or "similarity_search").lower().strip()
    reranking = str(config.get("reranking_method") or "none").lower().strip()
    if retrieval in {"hybrid", "hybridrag", "bm25"} or (
        config.get("reranking") and reranking in {"bm25", "bm25_reranker"}
    ):
        features.append("bm25")
    if config.get("reranking") and reranking in {
        "semantic_similarity",
        "cross_encoder",
        "crossencoder",
        "pointwise",
        "pointwise_ltr",
        "pointwise_l2r",
    }:
        features.append("local")
    if config.get("reranking") and reranking in {
        "llm",
        "llm_reranker",
        "llm_based",
        "llm-based",
    }:
        _add_llm(
            features,
            config.get("reranker_llm_provider")
            or (None if injected_llm else config.get("llm_provider")),
        )

    parser = config.get("document_parser")
    if parser:
        feature = _PARSER_FEATURES.get(str(parser).lower().strip())
        if feature:
            features.append(feature)
    elif not config.get("skip_document_ingestion") and _corpus_contains_pdf(config.get("data_dir")):
        features.append("pdf")

    if config.get("adaptive_web_search"):
        provider = str(config.get("search_provider") or "tavily").lower().strip()
        if provider in _WEB_FEATURES:
            features.append(_WEB_FEATURES[provider])

    _add_llm(features, config.get("query_transform_llm_provider"))
    return list(dict.fromkeys(features))


def required_features_for_search(config: Mapping[str, Any]) -> List[str]:
    features: List[str] = []
    _add_llm(features, config.get("llm_provider"))
    provider = str(config.get("search_provider") or "firecrawl").lower().strip()
    if provider in _WEB_FEATURES:
        features.append(_WEB_FEATURES[provider])
    return list(dict.fromkeys(features))


def required_features_for_synthetic_data(config: Mapping[str, Any]) -> List[str]:
    """Return capabilities used by synthetic QA generation."""

    features: List[str] = ["rag", "datasets"]
    _add_llm(features, config.get("llm_provider"))
    parser = config.get("document_parser")
    if parser:
        feature = _PARSER_FEATURES.get(str(parser).lower().strip())
        if feature:
            features.append(feature)
    elif _corpus_contains_pdf(config.get("data_dir")):
        features.append("pdf")
    chunking = str(config.get("chunking_method") or "recursive").lower().strip()
    if chunking == "token":
        features.append("token")
    elif chunking == "semantic":
        features.append("semantic")
    return list(dict.fromkeys(features))


def required_features_for_evaluation(
    config: Mapping[str, Any], metrics: Iterable[str]
) -> List[str]:
    features = ["datasets"]
    if any(str(metric).lower() in GENERATION_METRICS for metric in metrics):
        _add_llm(features, config.get("judge_llm_provider") or config.get("llm_provider"))
    return list(dict.fromkeys(features))


def required_features_for_retrieval(config: Mapping[str, Any]) -> List[str]:
    features = ["rag"]
    retrieval = str(config.get("retrieval_method") or "similarity_search").lower().strip()
    reranking = str(config.get("reranking_method") or "none").lower().strip()
    if retrieval in {"hybrid", "hybridrag", "bm25"} or (
        config.get("reranking") and reranking in {"bm25", "bm25_reranker"}
    ):
        features.append("bm25")
    if config.get("reranking") and reranking in {
        "semantic_similarity", "cross_encoder", "pointwise"
    }:
        features.append("local")
    if config.get("query_transformer") or retrieval == "contextual":
        _add_llm(
            features,
            config.get("query_transform_llm_provider") or config.get("llm_provider"),
        )
    if config.get("reranking") and reranking == "llm":
        _add_llm(
            features,
            config.get("reranker_llm_provider")
            or config.get("query_transform_llm_provider")
            or config.get("llm_provider"),
        )
    return list(dict.fromkeys(features))


def validate_composer_dependencies(
    config: Mapping[str, Any],
    search_space: Optional[Mapping[str, Any]] = None,
) -> None:
    """Raise one typed error containing every unavailable configured feature."""

    require_optional_dependencies(required_features_for_composer(config, search_space))


def validate_rag_dependencies(
    config: Mapping[str, Any],
    *,
    injected_embedding: bool = False,
    injected_vector_db: bool = False,
    injected_llm: bool = False,
) -> None:
    """Validate a direct RAG configuration before component construction."""

    features = required_features_for_rag(
        config,
        injected_embedding=injected_embedding,
        injected_vector_db=injected_vector_db,
        injected_llm=injected_llm,
    )
    require_optional_dependencies(features)


def validate_search_dependencies(config: Mapping[str, Any]) -> None:
    require_optional_dependencies(required_features_for_search(config))


def validate_synthetic_data_dependencies(config: Mapping[str, Any]) -> None:
    require_optional_dependencies(required_features_for_synthetic_data(config))


def validate_evaluation_dependencies(
    config: Mapping[str, Any], metrics: Iterable[str]
) -> None:
    require_optional_dependencies(required_features_for_evaluation(config, metrics))


def validate_retrieval_dependencies(config: Mapping[str, Any]) -> None:
    require_optional_dependencies(required_features_for_retrieval(config))
