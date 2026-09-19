"""
Centralized Enums and Mappings for the Muffakir Arabic RAG Library.

Provides canonical typed Enums with string inheritance for maximum backward
compatibility, along with canonical alias mappings and resolver functions.
"""

from enum import Enum, unique
from typing import Dict, Union, Any


@unique
class ProviderName(str, Enum):
    """Supported LLM provider backends."""
    GROQ = "groq"
    TOGETHER = "together"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"
    CUSTOM = "custom"
    CUSTOM_OPENAI = "custom_openai"

    def __str__(self) -> str:
        return self.value


# Canonical mapping from strings / aliases to ProviderName
PROVIDER_MAPPING: Dict[str, ProviderName] = {
    "groq": ProviderName.GROQ,
    "together": ProviderName.TOGETHER,
    "openrouter": ProviderName.OPENROUTER,
    "open_router": ProviderName.OPENROUTER,
    "openai": ProviderName.OPENAI,
    "anthropic": ProviderName.ANTHROPIC,
    "gemini": ProviderName.GEMINI,
    "google": ProviderName.GEMINI,
    "ollama": ProviderName.OLLAMA,
    "azure_openai": ProviderName.AZURE_OPENAI,
    "azure": ProviderName.AZURE_OPENAI,
    "custom": ProviderName.CUSTOM,
    "custom_openai": ProviderName.CUSTOM_OPENAI,
    "openai_compatible": ProviderName.CUSTOM_OPENAI,
    "deepseek": ProviderName.CUSTOM,
    "vllm": ProviderName.CUSTOM,
}


def resolve_provider_name(provider: Union[ProviderName, str, Any]) -> ProviderName:
    """
    Resolve a ProviderName enum, string, or alias into a canonical ProviderName enum.

    Args:
        provider: ProviderName instance or provider string (e.g. 'together', 'open_router', 'deepseek')

    Returns:
        Canonical ProviderName enum instance

    Raises:
        ValueError: If provider name is unknown or unsupported
    """
    if isinstance(provider, ProviderName):
        return provider

    if isinstance(provider, str):
        key = provider.lower().strip()
        if key in PROVIDER_MAPPING:
            return PROVIDER_MAPPING[key]

    available = ", ".join(sorted(PROVIDER_MAPPING.keys()))
    valid_names = [p.name for p in ProviderName]
    raise ValueError(
        f"Unsupported provider: {provider!r}. "
        f"Expected one of ProviderName: {valid_names} (or aliases: {available})"
    )



@unique
class RetrievalMethod(str, Enum):
    """Retrieval strategies for vector search."""
    SIMILARITY_SEARCH = "similarity_search"
    MAX_MARGINAL_RELEVANCE = "max_marginal_relevance_search"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"

    def __str__(self) -> str:
        return self.value


# Canonical mapping for retrieval methods (including shorter aliases)
RETRIEVAL_MAPPING: Dict[str, RetrievalMethod] = {
    "similarity_search": RetrievalMethod.SIMILARITY_SEARCH,
    "similarity": RetrievalMethod.SIMILARITY_SEARCH,
    "max_marginal_relevance": RetrievalMethod.MAX_MARGINAL_RELEVANCE,
    "mmr": RetrievalMethod.MAX_MARGINAL_RELEVANCE,
    "hybrid": RetrievalMethod.HYBRID,
    "contextual": RetrievalMethod.CONTEXTUAL,
}


def resolve_retrieval_method(method: Union[RetrievalMethod, str, Any]) -> RetrievalMethod:
    """
    Resolve a RetrievalMethod enum or string into a canonical RetrievalMethod enum.
    """
    if isinstance(method, RetrievalMethod):
        return method

    if isinstance(method, str):
        key = method.lower().strip()
        if key in RETRIEVAL_MAPPING:
            return RETRIEVAL_MAPPING[key]

    available = ", ".join(sorted(RETRIEVAL_MAPPING.keys()))
    raise ValueError(f"Unsupported retrieval method: '{method}'. Available methods: {available}")


@unique
class VectorDBProvider(str, Enum):
    """Supported Vector Database engines."""
    CHROMA = "chroma"
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    FAISS = "faiss"
    MILVUS = "milvus"

    def __str__(self) -> str:
        return self.value


@unique
class DocumentParserProvider(str, Enum):
    """Supported Document Parser engines."""
    AZURE = "azure"
    DOCLING = "docling"
    LLAMA_PARSE = "llama_parse"
    PYPDF = "pypdf"
    UNSTRUCTURED = "unstructured"

    def __str__(self) -> str:
        return self.value


@unique
class WebSearchProvider(str, Enum):
    """Supported Web Search engines."""
    FIRECRAWL = "firecrawl"
    TAVILY = "tavily"
    SERPAPI = "serpapi"
    GOOGLE = "google"

    def __str__(self) -> str:
        return self.value


@unique
class RerankerMethod(str, Enum):
    """Supported Reranking algorithms."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    BM25 = "bm25"
    CROSS_ENCODER = "cross_encoder"
    POINTWISE = "pointwise"
    LLM = "llm"

    def __str__(self) -> str:
        return self.value


@unique
class HallucinationMethod(str, Enum):
    """Supported Hallucination Checking / Post-processing strategies."""
    TEXT_CLEANER = "text_cleaner"
    CONTEXT_GROUNDING = "context_grounding"
    NLI = "nli"
    SEMANTIC_SIMILARITY = "semantic_similarity"

    def __str__(self) -> str:
        return self.value


@unique
class QueryTransformerStrategy(str, Enum):
    """Supported Query Transformation strategies."""
    REWRITE = "rewrite"
    MULTI_QUERY = "multi_query"
    DECOMPOSITION = "decomposition"
    HYDE = "hyde"
    STEP_BACK = "step_back"

    def __str__(self) -> str:
        return self.value


@unique
class ChunkingMethod(str, Enum):
    """Supported Text Chunking strategies."""
    RECURSIVE = "recursive"
    CHARACTER = "character"
    TOKEN = "token"
    SPACY = "spacy"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"
    ARABIC_SENTENCE = "arabic_sentence"
    HIERARCHICAL = "hierarchical"

    def __str__(self) -> str:
        return self.value

