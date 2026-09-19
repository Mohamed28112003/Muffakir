"""
Provider catalog and preset definitions for ComposerUI.
"""

import importlib.util
from typing import Any, Dict, List, Optional

from Composer.config_space import DEFAULT_SEARCH_SPACE
from Evaluation.constants import ALL_METRICS, GENERATION_METRICS, RETRIEVAL_METRICS
from Muffakir.optional_dependencies import OPTIONAL_FEATURES


def _is_importable(module_name: str) -> bool:
    """True if `module_name` can be imported without actually importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


_PROVIDER_FEATURES: Dict[str, str] = {
    "groq": "groq",
    "together": "openai",
    "openrouter": "openai",
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "ollama": "ollama",
    "azure_openai": "openai",
    "custom": "openai",
    "deepseek": "openai",
}

_PROVIDER_NAMES: List[str] = [
    "groq",
    "together",
    "openrouter",
    "openai",
    "anthropic",
    "gemini",
    "ollama",
    "azure_openai",
    "custom",
    "deepseek",
]

_VECTOR_DB_FEATURES: Dict[str, str] = {
    "chroma": "chroma",
    "faiss": "faiss",
    "qdrant": "qdrant",
    "milvus": "milvus",
    "pinecone": "pinecone",
}

_VECTOR_DB_NAMES: List[str] = ["chroma", "faiss", "qdrant", "milvus", "pinecone"]

_PARSER_FEATURES: Dict[str, str] = {
    "docling": "docling",
    "llama_parse": "llamaparse",
    "azure": "azure",
}

_PARSER_FIELDS: Dict[str, List[str]] = {
    "docling": ["export_type"],
    "llama_parse": ["api_key", "language"],
    "azure": ["endpoint", "api_key"],
}


def _feature_entry(name: str, feature: str) -> Dict[str, Any]:
    """Return catalog metadata using the SDK's central feature registry."""
    spec = OPTIONAL_FEATURES[feature]
    missing = [module for module in spec.import_names if not _is_importable(module)]
    return {
        "name": name,
        "available": not missing,
        "requires": ", ".join(spec.import_names),
        "missing": missing,
        "extra": spec.extra,
        "install_command": spec.install_command,
    }


def _availability_entries(
    names: List[str], features: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Built fresh on every call (not cached at import time) so package
    availability is re-checked each time the catalog is fetched — cheap
    (importlib.util.find_spec does no actual import) and keeps this testable
    by monkeypatching `_is_importable`."""
    return [_feature_entry(name, features[name]) for name in names]


def _provider_options() -> List[Dict[str, Any]]:
    return _availability_entries(_PROVIDER_NAMES, _PROVIDER_FEATURES)


def _vector_db_options() -> List[Dict[str, Any]]:
    return _availability_entries(_VECTOR_DB_NAMES, _VECTOR_DB_FEATURES)


def _parser_options() -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            **_feature_entry(name, _PARSER_FEATURES[name]),
            "fields": _PARSER_FIELDS[name],
        }
        for name in ("docling", "llama_parse", "azure")
    ]


_WEB_SEARCH_PROVIDER_FEATURES: Dict[str, str] = {
    "firecrawl": "firecrawl",
    "tavily": "tavily",
    "serpapi": "serpapi",
}

_WEB_SEARCH_PROVIDER_NAMES: List[str] = ["firecrawl", "tavily", "serpapi"]

# Optional tuning knobs per provider (api_key is handled separately by the
# frontend's dedicated "Web Search API Key" field, never listed here).
_WEB_SEARCH_PROVIDER_FIELDS: Dict[str, List[str]] = {
    "firecrawl": ["max_depth", "time_limit", "max_urls"],
    "tavily": ["max_results"],
    "serpapi": ["max_results"],
}


def _web_search_provider_options() -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            **_feature_entry(name, _WEB_SEARCH_PROVIDER_FEATURES[name]),
            "fields": _WEB_SEARCH_PROVIDER_FIELDS[name],
        }
        for name in _WEB_SEARCH_PROVIDER_NAMES
    ]


def _cuda_available() -> bool:
    """True if torch reports a usable CUDA GPU. Checked fresh on every call
    (not cached at import time) so it stays testable via monkeypatch, same
    as `_is_importable`."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _device_options() -> List[Dict[str, Any]]:
    return [
        {"name": "auto", "available": True, "requires": None},
        {"name": "cpu", "available": True, "requires": None},
        {
            "name": "cuda",
            "available": _cuda_available(),
            "requires": "a CUDA-capable GPU + GPU-enabled torch build",
            "extra": "local",
            "install_command": OPTIONAL_FEATURES["local"].install_command,
        },
    ]


def _embedding_provider_options() -> List[Dict[str, Any]]:
    return [
        _feature_entry("local", "local"),
        _feature_entry("openai", "openai"),
        _feature_entry("cohere", "cohere"),
    ]


def _reranker_options() -> List[Dict[str, Any]]:
    """Expose the live reranker registry without importing strategy backends."""
    from Reranker.factory import list_reranker_specs

    entries: List[Dict[str, Any]] = [
        {
            "name": "none",
            "label": "None",
            "description": "Skip reranking for this trial.",
            "configuration": "none",
            "available": True,
            "requires": None,
            "missing": [],
            "extra": None,
            "install_command": None,
        }
    ]
    for spec in list_reranker_specs():
        if spec.dependency:
            availability = _feature_entry(spec.name, spec.dependency)
        else:
            availability = {
                "name": spec.name,
                "available": True,
                "requires": None,
                "missing": [],
                "extra": None,
                "install_command": None,
            }
        entries.append(
            {
                **availability,
                "label": spec.label,
                "description": spec.description,
                "configuration": spec.configuration,
            }
        )
    return entries

CHUNKING_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "recursive_600_100",
        "label": "Recursive (600 / 100)",
        "value": {"method": "recursive", "size": 600, "overlap": 100},
    },
    {
        "id": "character_500_50",
        "label": "Character (500 / 50)",
        "value": {"method": "character", "size": 500, "overlap": 50},
    },
    {
        "id": "token_400_50",
        "label": "Token (400 / 50)",
        "value": {"method": "token", "size": 400, "overlap": 50},
    },
    {
        "id": "contextual_recursive_600_100",
        "label": "Contextual Recursive (600 / 100)",
        "value": {"method": "contextual_recursive", "size": 600, "overlap": 100},
    },
]

# Composer-compatible chunking algorithms only (see examples/17_ultimate_composer.py
# for the "semantic"/"sliding_window" incompatibilities this deliberately excludes).
CHUNKING_METHODS: List[str] = ["recursive", "character", "token", "contextual_recursive"]

# Only the two real Muffakir embedding models — suggestions for the frontend's
# free-text input (any HuggingFace model id is accepted), not a restriction.
EMBEDDING_MODEL_OPTIONS: List[str] = [
    "mohamed2811/Muffakir_Embedding",
    "mohamed2811/Muffakir_Embedding_V2",
]

# Suggestions only: users may add any sentence-transformers CrossEncoder-
# compatible Hugging Face model ID from the Composer UI.
RERANKER_MODEL_OPTIONS: List[str] = [
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3",
]

METRICS: Dict[str, Any] = {
    "all": list(ALL_METRICS),
    "retrieval": sorted(RETRIEVAL_METRICS),
    "generation": sorted(GENERATION_METRICS),
    # Cheap retrieval-only metrics by default — generation metrics call the
    # LLM once per sample, opt-in only.
    "default": ["recall", "precision", "mrr"],
}


def get_catalog() -> Dict[str, Any]:
    """Return catalog data for the frontend (provider/vector-db/parser
    availability is re-checked on every call)."""
    from LLMProvider.parameters import parameter_capabilities
    return {
        "llm_parameter_capabilities": {name: parameter_capabilities(name) for name in _PROVIDER_NAMES},
        "capabilities": {
            key: _feature_entry(key, key)
            for key in OPTIONAL_FEATURES
        },
        "providers": _provider_options(),
        "chunking_presets": CHUNKING_PRESETS,
        "chunking_methods": CHUNKING_METHODS,
        "embedding_models": EMBEDDING_MODEL_OPTIONS,
        "reranker_models": RERANKER_MODEL_OPTIONS,
        "embedding_providers": _embedding_provider_options(),
        "rerankers": _reranker_options(),
        "vector_dbs": _vector_db_options(),
        "parsers": _parser_options(),
        "devices": _device_options(),
        "web_search_providers": _web_search_provider_options(),
        "metrics": METRICS,
        "default_search_space": DEFAULT_SEARCH_SPACE,
    }
