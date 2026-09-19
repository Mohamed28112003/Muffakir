from typing import Any, Dict, Type
from .base import BaseDocumentParser

# Registry of supported parser providers. Keys are accepted provider names
# (lowercased); values are the concrete class. Concrete classes are imported
# lazily inside ``create_document_parser`` to keep optional dependencies out
# of the import path of the core library.
_PROVIDER_REGISTRY: Dict[str, Type[BaseDocumentParser]] = {
    "azure": None,        # filled lazily
    "docling": None,
    "llama_parse": None,
    "llamaparse": None,
    "llama-parse": None,
}


def available_providers() -> str:
    """Human-readable, de-duplicated list of available provider names."""
    seen = []
    for key in ("azure", "docling", "llama_parse"):
        seen.append(f"'{key}'")
    return ", ".join(seen)


def create_document_parser(provider: str, **kwargs: Any) -> BaseDocumentParser:
    """
    Factory function to initialize the requested document parser provider.

    Args:
        provider (str): Provider name (e.g. 'azure', 'docling', 'llama_parse').
            Aliases 'llamaparse' and 'llama-parse' are also accepted.
        **kwargs: Provider-specific configuration arguments.

    Returns:
        BaseDocumentParser: Instantiated document parser.

    Raises:
        ConfigurationError: if ``provider`` is unknown or unsupported
            (also a ValueError subclass, for backward compatibility).
    """
    from Muffakir.exceptions import ConfigurationError

    if not isinstance(provider, str) or not provider.strip():
        raise ConfigurationError("provider must be a non-empty string.")

    provider = provider.lower().strip()

    if provider == "azure":
        from Muffakir.optional_dependencies import require_optional_dependency
        require_optional_dependency("azure")
        from .azure_parser import AzureDocumentParser
        return AzureDocumentParser(**kwargs)

    elif provider == "docling":
        from Muffakir.optional_dependencies import require_optional_dependency
        require_optional_dependency("docling")
        from .docling_parser import DoclingParser
        return DoclingParser(**kwargs)

    elif provider in ("llama_parse", "llamaparse", "llama-parse"):
        from Muffakir.optional_dependencies import require_optional_dependency
        require_optional_dependency("llamaparse")
        from .llama_parse_parser import LlamaParseParser
        return LlamaParseParser(**kwargs)

    raise ConfigurationError(
        f"Unknown document parser provider: '{provider}'. "
        f"Available providers: {available_providers()}"
    )
