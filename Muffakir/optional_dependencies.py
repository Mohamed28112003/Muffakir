"""Central registry and validation helpers for optional Muffakir features.

This module deliberately uses only the Python standard library.  It is safe to
import from the package root, factories, the CLI, and ComposerUI without
pulling provider SDKs or heavyweight ML runtimes into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import sys
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class OptionalFeature:
    """Installation metadata for one independently installable capability."""

    key: str
    label: str
    extra: str
    import_names: Tuple[str, ...]

    @property
    def install_command(self) -> str:
        return f'pip install "Muffakir[{self.extra}]"'


OPTIONAL_FEATURES: Dict[str, OptionalFeature] = {
    "rag": OptionalFeature("rag", "RAG orchestration", "rag", ("langchain", "langchain_community", "langchain_text_splitters")),
    "local": OptionalFeature("local", "local embeddings and model inference", "local", ("sentence_transformers",)),
    "chroma": OptionalFeature("chroma", "Chroma vector database", "chroma", ("chromadb", "langchain_chroma")),
    "openai": OptionalFeature("openai", "OpenAI-compatible providers", "openai", ("langchain_openai",)),
    "groq": OptionalFeature("groq", "Groq provider", "groq", ("langchain_groq",)),
    "anthropic": OptionalFeature("anthropic", "Anthropic provider", "anthropic", ("langchain_anthropic",)),
    "gemini": OptionalFeature("gemini", "Google Gemini provider", "gemini", ("langchain_google_genai",)),
    "ollama": OptionalFeature("ollama", "Ollama provider", "ollama", ("langchain_ollama",)),
    "cohere": OptionalFeature("cohere", "Cohere embeddings", "cohere", ("langchain_cohere",)),
    "faiss": OptionalFeature("faiss", "FAISS vector database", "faiss", ("faiss",)),
    "qdrant": OptionalFeature("qdrant", "Qdrant vector database", "qdrant", ("qdrant_client", "langchain_qdrant")),
    "milvus": OptionalFeature("milvus", "Milvus vector database", "milvus", ("langchain_milvus", "pymilvus")),
    "pinecone": OptionalFeature("pinecone", "Pinecone vector database", "pinecone", ("langchain_pinecone",)),
    "datasets": OptionalFeature("datasets", "dataset and report support", "datasets", ("pandas", "openpyxl")),
    "pdf": OptionalFeature("pdf", "basic PDF loading", "pdf", ("pypdf",)),
    "token": OptionalFeature("token", "token-based chunking", "token", ("tiktoken",)),
    "semantic": OptionalFeature("semantic", "semantic chunking", "semantic", ("langchain_experimental",)),
    "bm25": OptionalFeature("bm25", "BM25 retrieval and reranking", "bm25", ("rank_bm25",)),
    "ui": OptionalFeature("ui", "ComposerUI", "ui", ("fastapi", "uvicorn", "platformdirs")),
    "azure": OptionalFeature("azure", "Azure Document Intelligence", "azure", ("azure.ai.formrecognizer",)),
    "docling": OptionalFeature("docling", "Docling document parser", "docling", ("langchain_docling",)),
    "llamaparse": OptionalFeature("llamaparse", "LlamaParse document parser", "llamaparse", ("llama_parse",)),
    "firecrawl": OptionalFeature("firecrawl", "Firecrawl web search", "firecrawl", ("firecrawl",)),
    "tavily": OptionalFeature("tavily", "Tavily web search", "tavily", ("langchain_tavily",)),
    "serpapi": OptionalFeature(
        "serpapi",
        "SerpAPI web search",
        "serpapi",
        ("serpapi", "langchain_community"),
    ),
}


def is_importable(module_name: str) -> bool:
    """Return whether *module_name* can be resolved without importing it."""

    # Respect dependency injection/test doubles already installed by callers.
    # ``find_spec`` raises for synthetic modules whose ``__spec__`` is unset.
    if module_name in sys.modules:
        return sys.modules[module_name] is not None
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return False


def missing_imports(feature: str) -> Tuple[str, ...]:
    """Return missing import names for a registered feature."""

    spec = OPTIONAL_FEATURES[feature]
    return tuple(name for name in spec.import_names if not is_importable(name))


def feature_status(feature: str) -> dict:
    """Return a JSON-safe availability record used by ComposerUI."""

    spec = OPTIONAL_FEATURES[feature]
    missing = missing_imports(feature)
    return {
        "available": not missing,
        "requires": ", ".join(spec.import_names),
        "missing": list(missing),
        "extra": spec.extra,
        "install_command": spec.install_command,
    }


def require_optional_dependency(feature: str) -> None:
    """Raise the public typed error when a feature is not installed."""

    missing = missing_imports(feature)
    if not missing:
        return
    from Muffakir.exceptions import MissingOptionalDependencyError

    spec = OPTIONAL_FEATURES[feature]
    raise MissingOptionalDependencyError(
        feature=spec.label,
        missing_imports=missing,
        extra=spec.extra,
    )


def collect_missing_features(features: Iterable[str]) -> List[dict]:
    """Return one availability record per unavailable feature, de-duplicated."""

    result: List[dict] = []
    for feature in dict.fromkeys(features):
        missing = missing_imports(feature)
        if not missing:
            continue
        spec = OPTIONAL_FEATURES[feature]
        result.append(
            {
                "feature": feature,
                "label": spec.label,
                "missing": list(missing),
                "extra": spec.extra,
                "install_command": spec.install_command,
            }
        )
    return result


def require_optional_dependencies(features: Iterable[str]) -> None:
    """Validate a capability set and report every missing extra together."""

    missing = collect_missing_features(features)
    if not missing:
        return

    from Muffakir.exceptions import MissingOptionalDependencyError

    if len(missing) == 1:
        item = missing[0]
        spec = OPTIONAL_FEATURES[item["feature"]]
        raise MissingOptionalDependencyError(
            feature=spec.label,
            missing_imports=item["missing"],
            extra=item["extra"],
        )

    labels = ", ".join(item["label"] for item in missing)
    imports = [name for item in missing for name in item["missing"]]
    commands = "; ".join(item["install_command"] for item in missing)
    error = MissingOptionalDependencyError(
        feature=f"Configured features ({labels})",
        missing_imports=imports,
        extra="standard",
    )
    error.context["missing_features"] = missing
    error.context["install_commands"] = [item["install_command"] for item in missing]
    error.message = (
        f"Missing optional dependencies for: {labels}. Install the required "
        f"capabilities with: {commands}"
    )
    error.args = (error.message,)
    raise error
