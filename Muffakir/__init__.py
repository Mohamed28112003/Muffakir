"""
Muffakir Arabic RAG — Public API.
"""

from importlib import import_module

__version__ = "0.3.0"
from .Enums import (
    ProviderName,
    RetrievalMethod,
    VectorDBProvider,
    DocumentParserProvider,
    WebSearchProvider,
    RerankerMethod,
    HallucinationMethod,
    QueryTransformerStrategy,
    ChunkingMethod,
    PROVIDER_MAPPING,
    RETRIEVAL_MAPPING,
    resolve_provider_name,
    resolve_retrieval_method,
)
from .constants import (
    DEFAULT_LANGUAGE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNKING_METHOD,
    DEFAULT_VECTOR_DB_PROVIDER,
    DEFAULT_DB_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_RETRIEVAL_METHOD,
    DEFAULT_K,
    DEFAULT_FETCH_K,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_HALLUCINATION_CHECK,
    DEFAULT_HALLUCINATION_METHOD,
    DEFAULT_QUERY_TRANSFORMER_STRATEGY,
    DEFAULT_RERANKING_METHOD,
    DEFAULT_RERANKING_MODEL,
    DEFAULT_SEARCH_PROVIDER,
    DEFAULT_RAG_CONFIG,
    DEFAULT_COMPOSER_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_SEARCH_CONFIG,
    DEFAULT_SYNTHETIC_DATA_CONFIG,
)
from .exceptions import (
    MuffakirError,
    ProviderError,
    ProviderAuthenticationError,
    ProviderTimeoutError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ConfigurationError,
    PromptValidationError,
    MissingOptionalDependencyError,
    RetrievalError,
    GenerationError,
    DocumentIndexError,
    PromptLoadError,
    HallucinationCheckError,
    CheckpointError,
    CorruptCheckpointError,
    DatasetError,
    ParsingError,
    UnsupportedDocumentError,
    DocumentTooLargeError,
    ParsingAuthenticationError,
    ParsingTimeoutError,
    ParsingRateLimitError,
    ParsingServiceUnavailableError,
)
__all__ = [
    "__version__",
    # Facades
    "MuffakirRAG",
    "MuffakirSearch",
    "MuffakirRetrieval",
    "MuffakirSyntheticData",
    "MuffakirEvaluation",
    "MuffakirComposer",
    "MuffakirPrompt",
    # Enums & Mappings
    "ProviderName",
    "RetrievalMethod",
    "VectorDBProvider",
    "DocumentParserProvider",
    "WebSearchProvider",
    "RerankerMethod",
    "HallucinationMethod",
    "QueryTransformerStrategy",
    "ChunkingMethod",
    "PROVIDER_MAPPING",
    "RETRIEVAL_MAPPING",
    "resolve_provider_name",
    "resolve_retrieval_method",
    # Configuration & Constants
    "DEFAULT_LANGUAGE",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_PROVIDER",
    "DEFAULT_EMBEDDING_BATCH_SIZE",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNKING_METHOD",
    "DEFAULT_VECTOR_DB_PROVIDER",
    "DEFAULT_DB_PATH",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_RETRIEVAL_METHOD",
    "DEFAULT_K",
    "DEFAULT_FETCH_K",
    "DEFAULT_LLM_TEMPERATURE",
    "DEFAULT_LLM_MAX_TOKENS",
    "DEFAULT_HALLUCINATION_CHECK",
    "DEFAULT_HALLUCINATION_METHOD",
    "DEFAULT_QUERY_TRANSFORMER_STRATEGY",
    "DEFAULT_RERANKING_METHOD",
    "DEFAULT_RERANKING_MODEL",
    "DEFAULT_SEARCH_PROVIDER",
    "DEFAULT_RAG_CONFIG",
    "DEFAULT_COMPOSER_CONFIG",
    "DEFAULT_EVALUATION_CONFIG",
    "DEFAULT_SEARCH_CONFIG",
    "DEFAULT_SYNTHETIC_DATA_CONFIG",
    # Exceptions
    "MuffakirError",
    "ProviderError",
    "ProviderAuthenticationError",
    "ProviderTimeoutError",
    "ProviderRateLimitError",
    "ProviderUnavailableError",
    "ConfigurationError",
    "PromptValidationError",
    "MissingOptionalDependencyError",
    "RetrievalError",
    "GenerationError",
    "DocumentIndexError",
    "PromptLoadError",
    "HallucinationCheckError",
    "CheckpointError",
    "CorruptCheckpointError",
    "DatasetError",
    "ParsingError",
    "UnsupportedDocumentError",
    "DocumentTooLargeError",
    "ParsingAuthenticationError",
    "ParsingTimeoutError",
    "ParsingRateLimitError",
    "ParsingServiceUnavailableError",
]


_LAZY_EXPORTS = {
    "MuffakirRAG": ("Muffakir.Muffakir", "MuffakirRAG"),
    "MuffakirSearch": ("Muffakir.MuffakirSearch", "MuffakirSearch"),
    "MuffakirRetrieval": ("Muffakir.MuffakirRetrieval", "MuffakirRetrieval"),
    "MuffakirSyntheticData": ("Muffakir.MuffakirSyntheticData", "MuffakirSyntheticData"),
    "MuffakirEvaluation": ("Muffakir.MuffakirEvaluation", "MuffakirEvaluation"),
    "MuffakirComposer": ("Composer.composer", "MuffakirComposer"),
    "MuffakirPrompt": ("PromptManager.PromptManager", "MuffakirPrompt"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        module_name, attribute = target
        value = getattr(import_module(module_name), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

