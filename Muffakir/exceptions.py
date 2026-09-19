"""
Typed exception hierarchy for Muffakir.

Zero internal dependencies (mirrors Muffakir/Enums.py) so this module can be
imported eagerly from anywhere in the codebase, including modules that must
otherwise import Muffakir lazily to avoid circular imports (see
LLMProvider/LLMProvider.py, which imports this module lazily, inside function
bodies, for that reason).
"""

from typing import Any, Dict, Iterable, Optional, Type


def redact_secret(text: str, *secrets: Optional[str]) -> str:
    """Replace any occurrence of a known secret value (api_key, token, ...) in
    *text* with a redaction marker, so raw SDK error text that echoes a
    rejected credential never reaches a log line or exception message."""
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***REDACTED***")
    return text


class MuffakirError(Exception):
    """Base class for all typed Muffakir exceptions.

    Attributes:
        message: Human-readable description.
        error_code: Stable machine-readable code (e.g. "PROVIDER_TIMEOUT").
            Set as a class attribute on subclasses; can be overridden
            per-instance via the constructor.
        context: Optional structured diagnostic context (provider name,
            file path, trial_id, etc.) — never put raw secrets here.
        retryable: Class-level hint used by Composer/search/base_search.py to
            decide whether a transient failure is worth retrying.
    """

    error_code: str = "MUFFAKIR_ERROR"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if error_code is not None:
            self.error_code = error_code
        self.context = context or {}

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


# ---------------------------------------------------------------------------
# Provider (LLM call) failures
# ---------------------------------------------------------------------------
class ProviderError(MuffakirError):
    """An LLM provider call failed in an unclassified way."""

    error_code = "PROVIDER_ERROR"


class ProviderAuthenticationError(ProviderError):
    error_code = "PROVIDER_AUTH_FAILED"
    retryable = False


class ProviderTimeoutError(ProviderError):
    error_code = "PROVIDER_TIMEOUT"
    retryable = True


class ProviderRateLimitError(ProviderError):
    error_code = "PROVIDER_RATE_LIMITED"
    retryable = True


class ProviderUnavailableError(ProviderError):
    """Connection/server-side failure (5xx, connection refused, DNS, etc.)."""

    error_code = "PROVIDER_UNAVAILABLE"
    retryable = True


# ---------------------------------------------------------------------------
# Configuration / validation failures
# ---------------------------------------------------------------------------
class ConfigurationError(MuffakirError, ValueError):
    """Invalid configuration (bad provider name, missing API key, unsupported
    metric, ...). Deliberately also a ValueError subclass so existing call
    sites/tests that do `except ValueError` / `pytest.raises(ValueError)`
    keep working unchanged while new code can catch it specifically."""

    error_code = "CONFIGURATION_ERROR"
    retryable = False


class PromptValidationError(ConfigurationError):
    """A prompt override violates its declared formatting contract."""

    error_code = "PROMPT_VALIDATION_FAILED"


class MissingOptionalDependencyError(ConfigurationError, ImportError):
    """A selected capability needs an optional installation extra.

    Dual inheritance preserves compatibility with callers that historically
    caught ``ImportError`` while exposing a structured configuration failure.
    """

    error_code = "OPTIONAL_DEPENDENCY_MISSING"

    def __init__(
        self,
        *,
        feature: str,
        missing_imports: Iterable[str],
        extra: str,
    ) -> None:
        missing = tuple(str(name) for name in missing_imports)
        install_command = f'pip install "Muffakir[{extra}]"'
        super().__init__(
            f"{feature} requires optional dependencies: {', '.join(missing)}. "
            f"Install them with: {install_command}",
            context={
                "feature": feature,
                "missing_imports": list(missing),
                "extra": extra,
                "install_command": install_command,
            },
        )
        self.feature = feature
        self.missing_imports = missing
        self.extra = extra
        self.install_command = install_command


# ---------------------------------------------------------------------------
# Pipeline-stage failures
# ---------------------------------------------------------------------------
class RetrievalError(MuffakirError):
    error_code = "RETRIEVAL_FAILED"


class GenerationError(MuffakirError):
    error_code = "GENERATION_FAILED"


class DocumentIndexError(MuffakirError):
    error_code = "DOCUMENT_INDEX_FAILED"


class PromptLoadError(MuffakirError, RuntimeError):
    """A prompt YAML file exists but failed to load/parse. Dual-inherits
    RuntimeError for backward compatibility with existing `except RuntimeError`
    / `pytest.raises(RuntimeError)` call sites."""

    error_code = "PROMPT_LOAD_FAILED"


class HallucinationCheckError(MuffakirError):
    """A hallucination-checker strategy crashed while checking an answer.

    Raised instead of silently returning a fabricated "not a hallucination"
    verdict — a crashed safety check must never look like a passed one."""

    error_code = "HALLUCINATION_CHECK_FAILED"


# ---------------------------------------------------------------------------
# Composer-specific failures
# ---------------------------------------------------------------------------
class CheckpointError(MuffakirError):
    error_code = "CHECKPOINT_ERROR"


class CorruptCheckpointError(CheckpointError):
    error_code = "CHECKPOINT_CORRUPT"


class DatasetError(MuffakirError):
    error_code = "DATASET_LOAD_FAILED"


# ---------------------------------------------------------------------------
# Document parsing failures (Azure Document Intelligence / Docling / LlamaParse)
# ---------------------------------------------------------------------------
class ParsingError(MuffakirError):
    error_code = "DOCUMENT_PARSE_FAILED"


class UnsupportedDocumentError(ParsingError):
    error_code = "DOCUMENT_UNSUPPORTED"


class DocumentTooLargeError(ParsingError):
    error_code = "DOCUMENT_TOO_LARGE"


class ParsingAuthenticationError(ParsingError):
    error_code = "PARSING_AUTH_FAILED"
    retryable = False


class ParsingTimeoutError(ParsingError):
    error_code = "PARSING_TIMEOUT"
    retryable = True


class ParsingRateLimitError(ParsingError):
    error_code = "PARSING_RATE_LIMITED"
    retryable = True


class ParsingServiceUnavailableError(ParsingError):
    error_code = "PARSING_SERVICE_UNAVAILABLE"
    retryable = True


def classify_parsing_exception(
    exc: BaseException,
    *,
    backend: Optional[str] = None,
    secrets: Optional[Iterable[str]] = None,
) -> "ParsingError":
    """Map a raw exception from a document-parsing SDK (Azure Document
    Intelligence, Docling, LlamaParse) onto the right ParsingError subclass,
    based on its type name/attributes - no SDK-specific imports required.
    Mirrors classify_provider_exception's approach for LLM providers."""
    name = type(exc).__name__.lower()
    cls: Type[ParsingError]
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            code = int(status)
        except (TypeError, ValueError):
            code = None
        if code == 429:
            cls = ParsingRateLimitError
        elif code is not None and 500 <= code < 600:
            cls = ParsingServiceUnavailableError
        elif code in (401, 403):
            cls = ParsingAuthenticationError
        else:
            cls = ParsingError
    elif isinstance(exc, TimeoutError) or "timeout" in name:
        cls = ParsingTimeoutError
    elif "ratelimit" in name or "toomanyrequests" in name:
        cls = ParsingRateLimitError
    elif "authentication" in name or "permissiondenied" in name or "unauthorized" in name:
        cls = ParsingAuthenticationError
    elif isinstance(exc, ConnectionError) or "connection" in name or "unavailable" in name:
        cls = ParsingServiceUnavailableError
    else:
        cls = ParsingError
    prefix = f"{backend}: " if backend else ""
    exc_text = redact_secret(str(exc), *(secrets or ()))
    return cls(
        f"{prefix}{exc_text}",
        context={"backend": backend, "original_exception_type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Helper: classify a raw SDK/langchain exception into a typed ProviderError
# subclass without importing any specific SDK (keeps this file dependency-free).
# ---------------------------------------------------------------------------
def classify_provider_exception(
    exc: BaseException,
    *,
    provider: Optional[str] = None,
    secrets: Optional[Iterable[str]] = None,
) -> "ProviderError":
    """Map a raw exception from an LLM SDK/langchain client onto the right
    ProviderError subclass, based on its type name and MRO — no SDK-specific
    imports required."""
    name = type(exc).__name__.lower()
    cls: Type[ProviderError]
    if isinstance(exc, TimeoutError) or "timeout" in name:
        cls = ProviderTimeoutError
    elif "ratelimit" in name or "toomanyrequests" in name:
        cls = ProviderRateLimitError
    elif "authentication" in name or "permissiondenied" in name or "unauthorized" in name:
        cls = ProviderAuthenticationError
    elif (
        isinstance(exc, ConnectionError)
        or "connection" in name
        or "unavailable" in name
        or "internalserver" in name
    ):
        cls = ProviderUnavailableError
    else:
        cls = ProviderError
    prefix = f"{provider}: " if provider else ""
    exc_text = redact_secret(str(exc), *(secrets or ()))
    return cls(
        f"{prefix}{exc_text}",
        context={"provider": provider, "original_exception_type": type(exc).__name__},
    )
