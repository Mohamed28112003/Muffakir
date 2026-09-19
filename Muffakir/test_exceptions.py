import pytest

from Muffakir.exceptions import (
    redact_secret,
    classify_provider_exception,
    classify_parsing_exception,
    ProviderError,
)


# ---------------------------------------------------------------------------
# redact_secret
# ---------------------------------------------------------------------------

def test_redact_secret_masks_present_secret():
    assert redact_secret("Invalid key sk-abc123", "sk-abc123") == "Invalid key ***REDACTED***"


def test_redact_secret_masks_multiple_secrets():
    text = "key=sk-one other=sk-two"
    assert redact_secret(text, "sk-one", "sk-two") == "key=***REDACTED*** other=***REDACTED***"


def test_redact_secret_leaves_text_unchanged_when_secret_absent():
    assert redact_secret("Connection timed out", "sk-abc123") == "Connection timed out"


def test_redact_secret_ignores_none_and_short_secrets():
    # None and very short "secrets" (e.g. a 1-char test fixture) must not be
    # redacted — too easy to accidentally mangle unrelated text.
    assert redact_secret("value is k", None, "k") == "value is k"


# ---------------------------------------------------------------------------
# classify_provider_exception — secret redaction
# ---------------------------------------------------------------------------

def test_classify_provider_exception_redacts_known_secret():
    exc = Exception("AuthenticationError: invalid api key sk-real-secret-123")
    typed = classify_provider_exception(exc, provider="openai", secrets=["sk-real-secret-123"])
    assert isinstance(typed, ProviderError)
    assert "sk-real-secret-123" not in typed.message
    assert "***REDACTED***" in typed.message


def test_classify_provider_exception_without_secrets_is_unchanged():
    exc = Exception("connection refused")
    typed = classify_provider_exception(exc, provider="openai")
    assert "connection refused" in typed.message


# ---------------------------------------------------------------------------
# classify_parsing_exception — secret redaction
# ---------------------------------------------------------------------------

def test_classify_parsing_exception_redacts_known_secret():
    exc = Exception("401 Unauthorized: key az-real-secret-456 rejected")
    typed = classify_parsing_exception(exc, backend="azure", secrets=["az-real-secret-456"])
    assert "az-real-secret-456" not in typed.message
    assert "***REDACTED***" in typed.message
