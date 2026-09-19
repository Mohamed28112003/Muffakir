import os
import logging
import time
from typing import List, Optional, Tuple
from tqdm import tqdm
from .base import BaseDocumentParser
from .models import ParsedDocument, PageContent
from Muffakir.exceptions import (
    ConfigurationError,
    ParsingError,
    classify_parsing_exception,
    redact_secret,
)

logger = logging.getLogger(__name__)

# Extensions for which Azure actually performs OCR (layout/visual analysis).
# Plain-text files are read, not OCR'd, so their metadata must reflect that.
_AZURE_OCR_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class AzureDocumentParser(BaseDocumentParser):
    """
    Azure Document Intelligence provider for parsing documents.

    Args:
        endpoint (str): Azure Document Intelligence endpoint URL.
        api_key (str): Azure API key.
        model_id (str): Model ID to use for analysis. Default: ``"prebuilt-layout"``.
        timeout_seconds (float): Max seconds to wait for a single document
            analysis. Default: ``300`` (5 minutes).
        max_retries (int): Max retry attempts on transient Azure failures
            (HTTP 429 / 5xx, or network timeouts). Default: ``2``.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_id: str = "prebuilt-layout",
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
    ):
        if not endpoint or not api_key:
            raise ConfigurationError("Both endpoint and api_key are required for AzureDocumentParser.")

        self.endpoint = endpoint
        self.api_key = api_key
        self.model_id = model_id
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self._client = None  # lazily created Azure client
        self.logger = logging.getLogger(__name__)

    @property
    def client(self):
        """Lazily create the Azure ``DocumentAnalysisClient`` (deferred import + connection)."""
        if self._client is None:
            try:
                from azure.ai.formrecognizer import DocumentAnalysisClient
                from azure.core.credentials import AzureKeyCredential
            except ImportError as e:
                raise ImportError(
                    "The 'azure-ai-formrecognizer' package is required to use the AzureDocumentParser. "
                    "Please install it using: pip install azure-ai-formrecognizer"
                ) from e

            self._client = DocumentAnalysisClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
            )
        return self._client

    def close(self) -> None:
        """Close the underlying Azure client if it was created."""
        if self._client is not None:
            closer = getattr(self._client, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as e:  # defensive
                    self.logger.warning(f"Error closing Azure client: {e}")
            self._client = None

    def parse_file(self, file_path: str, base_dir: Optional[str] = None) -> ParsedDocument:
        """Perform OCR/analysis on a document using Azure Document Intelligence.

        Retries transient failures (HTTP 429 / 5xx, timeouts) up to ``max_retries``
        times with exponential backoff. Non-transient errors surface immediately.

        Args:
            base_dir: Optional containment root — if given, ``file_path`` must
                resolve within it or ``ConfigurationError`` is raised.
        """
        file_path = self._resolve_within_base(file_path, base_dir)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        last_error: Optional[ParsingError] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._analyze_once(file_path)
            except ParsingError as e:
                if not e.retryable:
                    # Non-transient errors must not be retried; surface immediately.
                    raise
                last_error = e
                if attempt >= self.max_retries:
                    break
                backoff = 2 ** attempt  # 1s, 2s, 4s, ...
                self.logger.warning(
                    f"Azure transient error on {file_path} (attempt {attempt + 1}/"
                    f"{self.max_retries + 1}): {e}. Retrying in {backoff}s…"
                )
                time.sleep(backoff)

        raise type(last_error)(
            f"Azure analysis of {file_path} failed after {self.max_retries + 1} attempts: {last_error}",
            error_code=last_error.error_code,
        ) from last_error

    def _analyze_once(self, file_path: str) -> ParsedDocument:
        """Single Azure analysis attempt (no retry logic)."""
        try:
            with open(file_path, "rb") as document:
                poller = self.client.begin_analyze_document(
                    model_id=self.model_id,
                    document=document,
                )
            # Cloud polling with a configurable timeout to avoid indefinite hangs.
            result = poller.result(timeout=self.timeout_seconds)

            pages: List[PageContent] = []
            for page in result.pages:
                page_text = ""
                for line in page.lines:
                    page_text += line.content + "\n"
                pages.append(PageContent(page_number=page.page_number, text=page_text))

            # Standardized separator contract: pages joined by "\n\n".
            extracted_text = "\n\n".join(p.text for p in pages)

            ext = os.path.splitext(file_path)[1].lower()
            is_ocr = ext in _AZURE_OCR_EXTENSIONS

            self.logger.info(
                f"Azure analysis completed for {file_path}. Total pages: {len(result.pages)}"
            )

            return ParsedDocument(
                text=extracted_text,
                source_path=file_path,
                parser_name="azure",
                pages=pages,
                metadata={
                    "ocr_processed": is_ocr,
                    "original_filename": os.path.basename(file_path),
                },
            )

        except ParsingError:
            raise
        except Exception as e:
            self.logger.error(
                f"Error performing Azure analysis on {file_path}: "
                f"{redact_secret(str(e), self.api_key)}"
            )
            raise classify_parsing_exception(e, backend="azure", secrets=[self.api_key]) from e

    def parse_directory(self, directory_path: str, base_dir: Optional[str] = None) -> List[ParsedDocument]:
        """
        Perform analysis on all supported files in the directory.

        Returns the list of successfully parsed documents. Per-file failures are
        logged individually and summarized at the end of the batch.

        Args:
            base_dir: Optional containment root — if given, ``directory_path``
                must resolve within it or ``ConfigurationError`` is raised.
        """
        directory_path = self._resolve_within_base(directory_path, base_dir)
        parsed_docs: List[ParsedDocument] = []
        failures: List[Tuple[str, str]] = []

        if not os.path.isdir(directory_path):
            self.logger.warning(f"Directory {directory_path} does not exist.")
            return parsed_docs

        supported_files = [
            f for f in os.listdir(directory_path) if self.supports(f)
        ]

        if not supported_files:
            self.logger.warning(f"No supported files found in {directory_path}")
            return parsed_docs

        with tqdm(
            supported_files,
            desc="Azure analysis",
            unit="file",
            colour="blue",
        ) as progress:
            for filename in progress:
                file_path = os.path.join(directory_path, filename)
                progress.set_postfix_str(filename)
                try:
                    parsed_docs.append(self.parse_file(file_path))
                    # file_path is already `directory_path`-joined and directory_path
                    # itself was validated against base_dir above, so no per-file
                    # re-check is needed (os.listdir() filenames can't contain "..").
                except Exception as e:
                    failures.append((filename, f"{type(e).__name__}: {e}"))
                    self.logger.error(f"Failed to parse {filename}: {e}")

        if failures:
            self.logger.warning(
                f"Azure analysis: {len(parsed_docs)} parsed, {len(failures)} failed: "
                + "; ".join(f"{fn} ({err})" for fn, err in failures)
            )
            if not parsed_docs:
                raise ParsingError(
                    f"All {len(failures)} file(s) in {directory_path} failed to parse.",
                    context={"directory": directory_path, "failures": failures},
                )
        else:
            self.logger.info(
                f"Azure analysis: {len(parsed_docs)} document(s) parsed from directory."
            )
        return parsed_docs

    def supported_extensions(self) -> List[str]:
        return [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".txt"]
