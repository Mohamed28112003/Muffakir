"""
Test suite for the DocumentParser module (pytest).

No optional parsing SDKs (azure-ai-formrecognizer, langchain-docling, llama-parse)
are required: they are mocked via monkeypatching sys.modules. This validates the
contract, the factory, the BaseDocumentParser helpers, the ParsedDocument model,
input validation, batch failure aggregation, the LlamaParse API-key fast-fail,
and the P0 "no logging.basicConfig" regression.
"""

import logging
import os
import sys
import types
from pathlib import Path

import pytest

# Make the library importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from DocumentParser.base import BaseDocumentParser
from DocumentParser.models import ParsedDocument, PageContent
from DocumentParser.factory import create_document_parser, available_providers
from Muffakir.exceptions import (
    ParsingError,
    ParsingAuthenticationError,
    ParsingServiceUnavailableError,
    ConfigurationError,
)


# ---------------------------------------------------------------------------
# Fake concrete parser (no SDK needed) for base/contract tests
# ---------------------------------------------------------------------------
class _FakeParser(BaseDocumentParser):
    def parse_file(self, file_path: str) -> ParsedDocument:
        return ParsedDocument(text="hello", source_path=file_path, parser_name="fake")

    def parse_directory(self, directory_path: str):
        return []

    def supported_extensions(self) -> list:
        return [".pdf", ".txt"]


# ---------------------------------------------------------------------------
# BaseDocumentParser.supports()
# ---------------------------------------------------------------------------
def test_supports_matches_extensions_case_insensitive():
    p = _FakeParser()
    assert p.supports("/x/a.PDF")
    assert p.supports("/x/a.txt")
    assert not p.supports("/x/a.docx")


# ---------------------------------------------------------------------------
# ParsedDocument / PageContent model
# ---------------------------------------------------------------------------
def test_parsed_document_constructs():
    pd = ParsedDocument(text="t", source_path="/x.pdf", parser_name="fake", pages=[PageContent(1, "p")])
    assert pd.text == "t" and pd.pages[0].page_number == 1


def test_parsed_document_empty_text_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="DocumentParser.models"):
        ParsedDocument(text="", source_path="/x.pdf", parser_name="fake")
    assert any("contains empty text" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Factory dispatch + errors
# ---------------------------------------------------------------------------
def test_factory_unknown_provider_raises():
    with pytest.raises(ValueError):
        create_document_parser("not_a_provider")


def test_factory_empty_provider_raises():
    with pytest.raises(ValueError):
        create_document_parser("")


def test_available_providers_lists_all():
    s = available_providers()
    for name in ("azure", "docling", "llama_parse"):
        assert f"'{name}'" in s


# ---------------------------------------------------------------------------
# Mock-SDK helpers for provider tests
# ---------------------------------------------------------------------------
def _install_fake_azure_sdk(monkeypatch):
    """Inject a fake `azure.ai.formrecognizer` + `azure.core.credentials`."""
    azure_ai = types.ModuleType("azure.ai.formrecognizer")
    azure_core = types.ModuleType("azure.core.credentials")

    class _FakePage:
        def __init__(self, page_number):
            self.page_number = page_number
            self.lines = [types.SimpleNamespace(content=f"line{page_number}-1")]

    class _FakeResult:
        def __init__(self, n_pages=2):
            self.pages = [_FakePage(i + 1) for i in range(n_pages)]

    class _FakePoller:
        def __init__(self, result):
            self._result = result

        def result(self, timeout=None):
            return self._result

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.closed = False

        def begin_analyze_document(self, model_id, document):
            return _FakePoller(_FakeResult())

        def close(self):
            self.closed = True

    azure_ai.DocumentAnalysisClient = _FakeClient
    azure_core.AzureKeyCredential = lambda key: key
    monkeypatch.setitem(sys.modules, "azure", types.ModuleType("azure"))
    monkeypatch.setitem(sys.modules, "azure.ai", azure_ai)
    monkeypatch.setitem(sys.modules, "azure.ai.formrecognizer", azure_ai)
    monkeypatch.setitem(sys.modules, "azure.core", azure_core)
    monkeypatch.setitem(sys.modules, "azure.core.credentials", azure_core)


def _install_fake_llama_parse(monkeypatch):
    lp = types.ModuleType("llama_parse")

    class _FakeLlamaDoc:
        def __init__(self, text, page_number):
            self.text = text
            self.metadata = {"page_label": str(page_number)}

    class _FakeLlamaParse:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_data(self, file_path):
            return [_FakeLlamaDoc("page1 text", 1), _FakeLlamaDoc("page2 text", 2)]

    lp.LlamaParse = _FakeLlamaParse
    monkeypatch.setitem(sys.modules, "llama_parse", lp)


# ---------------------------------------------------------------------------
# Azure provider
# ---------------------------------------------------------------------------
def test_azure_parse_file_raises_on_missing_file(tmp_path, monkeypatch):
    _install_fake_azure_sdk(monkeypatch)
    from DocumentParser.azure_parser import AzureDocumentParser
    p = AzureDocumentParser(endpoint="https://x", api_key="k")
    with pytest.raises(FileNotFoundError):
        p.parse_file(str(tmp_path / "nope.pdf"))


def test_azure_parse_file_returns_document(tmp_path, monkeypatch):
    _install_fake_azure_sdk(monkeypatch)
    from DocumentParser.azure_parser import AzureDocumentParser
    p = AzureDocumentParser(endpoint="https://x", api_key="k", timeout_seconds=10)
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    pd = p.parse_file(str(f))
    assert pd.parser_name == "azure"
    assert "line1" in pd.text and "line2" in pd.text
    assert pd.metadata["ocr_processed"] is True  # .pdf is OCR'd


def test_azure_txt_metadata_ocr_false(tmp_path, monkeypatch):
    _install_fake_azure_sdk(monkeypatch)
    from DocumentParser.azure_parser import AzureDocumentParser
    p = AzureDocumentParser(endpoint="https://x", api_key="k")
    f = tmp_path / "a.txt"
    f.write_text("plain text", encoding="utf-8")
    pd = p.parse_file(str(f))
    assert pd.metadata["ocr_processed"] is False  # .txt is NOT OCR'd


def test_azure_parse_file_no_base_dir_unchanged(tmp_path, monkeypatch):
    """Regression guard: omitting base_dir (default None) behaves exactly as before."""
    _install_fake_azure_sdk(monkeypatch)
    from DocumentParser.azure_parser import AzureDocumentParser
    p = AzureDocumentParser(endpoint="https://x", api_key="k")
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    pd = p.parse_file(str(f))
    assert pd.parser_name == "azure"


def test_azure_parse_file_within_base_dir_succeeds(tmp_path, monkeypatch):
    _install_fake_azure_sdk(monkeypatch)
    from DocumentParser.azure_parser import AzureDocumentParser
    p = AzureDocumentParser(endpoint="https://x", api_key="k")
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    pd = p.parse_file(str(f), base_dir=str(tmp_path))
    assert pd.parser_name == "azure"


def test_azure_parse_file_outside_base_dir_raises(tmp_path, monkeypatch):
    _install_fake_azure_sdk(monkeypatch)
    from DocumentParser.azure_parser import AzureDocumentParser
    p = AzureDocumentParser(endpoint="https://x", api_key="k")
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside_file = tmp_path / "outside.pdf"
    outside_file.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(ConfigurationError, match="escapes"):
        p.parse_file(str(outside_file), base_dir=str(allowed_dir))


class _StatusCodeError(Exception):
    """Fake SDK exception exposing a status_code attribute (like azure-core)."""
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def test_azure_retries_transient_then_succeeds(tmp_path, monkeypatch):
    """A 503 (retryable) should be retried and eventually succeed."""
    _install_fake_azure_sdk(monkeypatch)
    from DocumentParser.azure_parser import AzureDocumentParser

    calls = {"n": 0}

    class _FakePage:
        def __init__(self, page_number):
            self.page_number = page_number
            self.lines = [types.SimpleNamespace(content=f"line{page_number}")]

    class _FakeResult:
        pages = [_FakePage(1)]

    class _FakePoller:
        def result(self, timeout=None):
            return _FakeResult()

    class _FlakyClient:
        def begin_analyze_document(self, model_id, document):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _StatusCodeError("service busy", 503)
            return _FakePoller()

    p = AzureDocumentParser(endpoint="https://x", api_key="k", max_retries=3)
    p._client = _FlakyClient()
    monkeypatch.setattr("time.sleep", lambda s: None)  # skip real backoff delay

    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF")
    pd = p.parse_file(str(f))
    assert calls["n"] == 3
    assert pd.parser_name == "azure"


def test_azure_non_retryable_fails_fast(tmp_path, monkeypatch):
    """A 401 (auth failure) must not be retried and raises ParsingAuthenticationError."""
    _install_fake_azure_sdk(monkeypatch)
    from DocumentParser.azure_parser import AzureDocumentParser

    calls = {"n": 0}

    class _AuthFailClient:
        def begin_analyze_document(self, model_id, document):
            calls["n"] += 1
            raise _StatusCodeError("bad credentials", 401)

    p = AzureDocumentParser(endpoint="https://x", api_key="k", max_retries=3)
    p._client = _AuthFailClient()

    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF")
    with pytest.raises(ParsingAuthenticationError):
        p.parse_file(str(f))
    assert calls["n"] == 1  # no retries wasted on a non-retryable failure


def test_azure_exhausts_retries_raises_typed_error(tmp_path, monkeypatch):
    """After exhausting retries on a persistent transient failure, the final
    exception must stay typed (not a generic RuntimeError) and chain the cause."""
    _install_fake_azure_sdk(monkeypatch)
    from DocumentParser.azure_parser import AzureDocumentParser

    class _AlwaysBusyClient:
        def begin_analyze_document(self, model_id, document):
            raise _StatusCodeError("service busy", 503)

    p = AzureDocumentParser(endpoint="https://x", api_key="k", max_retries=1)
    p._client = _AlwaysBusyClient()
    monkeypatch.setattr("time.sleep", lambda s: None)

    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF")
    with pytest.raises(ParsingServiceUnavailableError) as excinfo:
        p.parse_file(str(f))
    assert isinstance(excinfo.value.__cause__, ParsingServiceUnavailableError)
    assert "failed after 2 attempts" in str(excinfo.value)


def test_azure_directory_all_files_failed_raises(tmp_path, monkeypatch):
    """If every file in the batch fails, raise instead of silently returning []."""
    _install_fake_azure_sdk(monkeypatch)
    from DocumentParser.azure_parser import AzureDocumentParser
    p = AzureDocumentParser(endpoint="https://x", api_key="k")
    (tmp_path / "bad1.pdf").write_bytes(b"%PDF")
    (tmp_path / "bad2.pdf").write_bytes(b"%PDF")

    def always_fails(path):
        raise RuntimeError("boom")

    p.parse_file = always_fails
    with pytest.raises(ParsingError):
        p.parse_directory(str(tmp_path))


def test_azure_directory_aggregates_failures(tmp_path, monkeypatch, caplog):
    _install_fake_azure_sdk(monkeypatch)
    from DocumentParser.azure_parser import AzureDocumentParser
    p = AzureDocumentParser(endpoint="https://x", api_key="k")
    (tmp_path / "ok.pdf").write_bytes(b"%PDF")
    (tmp_path / "bad.pdf").write_bytes(b"%PDF")
    original = p.parse_file

    def flaky(path):
        if "bad" in path:
            raise RuntimeError("boom")
        return original(path)

    p.parse_file = flaky
    with caplog.at_level(logging.WARNING, logger="DocumentParser.azure_parser"):
        results = p.parse_directory(str(tmp_path))
    assert len(results) == 1
    assert any("1 parsed, 1 failed" in r.message and "bad.pdf" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# LlamaParse provider
# ---------------------------------------------------------------------------
def test_llama_parse_requires_api_key(monkeypatch):
    _install_fake_llama_parse(monkeypatch)
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY", raising=False)
    from DocumentParser.llama_parse_parser import LlamaParseParser
    with pytest.raises(ValueError):
        LlamaParseParser(api_key=None)


def test_llama_parse_parse_file(tmp_path, monkeypatch):
    _install_fake_llama_parse(monkeypatch)
    from DocumentParser.llama_parse_parser import LlamaParseParser
    p = LlamaParseParser(api_key="llx-xxx")
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF")
    pd = p.parse_file(str(f))
    assert "page1 text" in pd.text and "page2 text" in pd.text
    assert pd.parser_name == "llama_parse"


def test_llama_parse_missing_file_raises(tmp_path, monkeypatch):
    _install_fake_llama_parse(monkeypatch)
    from DocumentParser.llama_parse_parser import LlamaParseParser
    p = LlamaParseParser(api_key="llx-xxx")
    with pytest.raises(FileNotFoundError):
        p.parse_file(str(tmp_path / "nope.pdf"))


def test_llama_parse_directory_all_files_failed_raises(tmp_path, monkeypatch):
    _install_fake_llama_parse(monkeypatch)
    from DocumentParser.llama_parse_parser import LlamaParseParser
    p = LlamaParseParser(api_key="llx-xxx")
    (tmp_path / "bad1.pdf").write_bytes(b"%PDF")
    (tmp_path / "bad2.pdf").write_bytes(b"%PDF")

    def always_fails(path):
        raise RuntimeError("boom")

    p.parse_file = always_fails
    with pytest.raises(ParsingError):
        p.parse_directory(str(tmp_path))


def test_llama_parse_caches_client(tmp_path, monkeypatch):
    _install_fake_llama_parse(monkeypatch)
    from DocumentParser.llama_parse_parser import LlamaParseParser
    p = LlamaParseParser(api_key="llx-xxx")
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF")
    p.parse_file(str(f))
    client1 = p._get_parser()
    p.parse_file(str(f))
    client2 = p._get_parser()
    assert client1 is client2  # memoized, not recreated per call


# ---------------------------------------------------------------------------
# Docling provider
# ---------------------------------------------------------------------------
def _install_fake_docling(monkeypatch, docs_metadata):
    """Inject a fake `langchain_docling` whose DoclingLoader.load() returns
    documents with the given list of metadata dicts (one per doc)."""
    module = types.ModuleType("langchain_docling")

    class _FakeDoc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    class _FakeDoclingLoader:
        def __init__(self, file_path, export_type="markdown", **kwargs):
            self.file_path = file_path

        def load(self):
            return [_FakeDoc(f"text{i}", meta) for i, meta in enumerate(docs_metadata)]

    module.DoclingLoader = _FakeDoclingLoader
    monkeypatch.setitem(sys.modules, "langchain_docling", module)


def test_docling_batch_missing_source_metadata_raises(tmp_path, monkeypatch):
    """If Docling returns documents without a 'source' key, grouping by file
    is unsafe - must raise a typed ParsingError instead of mis-attributing content."""
    _install_fake_docling(monkeypatch, docs_metadata=[{}, {}])
    from DocumentParser.docling_parser import DoclingParser
    p = DoclingParser()
    (tmp_path / "a.pdf").write_bytes(b"%PDF")

    with pytest.raises(ParsingError):
        p.parse_directory(str(tmp_path))


def test_docling_parse_file_wraps_sdk_failure(tmp_path, monkeypatch):
    module = types.ModuleType("langchain_docling")

    class _FailingLoader:
        def __init__(self, file_path, export_type="markdown", **kwargs):
            pass

        def load(self):
            raise RuntimeError("docling backend crashed")

    module.DoclingLoader = _FailingLoader
    monkeypatch.setitem(sys.modules, "langchain_docling", module)

    from DocumentParser.docling_parser import DoclingParser
    p = DoclingParser()
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF")

    with pytest.raises(ParsingError) as excinfo:
        p.parse_file(str(f))
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# P0 regression: NO provider calls logging.basicConfig on instantiation
# ---------------------------------------------------------------------------
def test_no_basicconfig_called_on_instantiation(monkeypatch):
    root_handlers_before = list(logging.getLogger().handlers)
    # Azure (with fake SDK)
    _install_fake_azure_sdk(monkeypatch)
    from DocumentParser.azure_parser import AzureDocumentParser
    AzureDocumentParser(endpoint="https://x", api_key="k")
    # LlamaParse (with fake SDK + key)
    _install_fake_llama_parse(monkeypatch)
    from DocumentParser.llama_parse_parser import LlamaParseParser
    LlamaParseParser(api_key="llx-xxx")
    # Docling needs a fake module too
    monkeypatch.setitem(sys.modules, "langchain_docling", types.ModuleType("langchain_docling"))
    from DocumentParser.docling_parser import DoclingParser
    DoclingParser()
    # Root logger handlers must not have been replaced/added-to by the library
    assert list(logging.getLogger().handlers) == root_handlers_before


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))