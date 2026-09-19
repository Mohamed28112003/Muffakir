import os
import logging
from typing import List, Optional
from tqdm import tqdm
from .base import BaseDocumentParser
from .models import ParsedDocument, PageContent
from Muffakir.exceptions import ConfigurationError, ParsingError, classify_parsing_exception


class LlamaParseParser(BaseDocumentParser):
    """
    LlamaParse document parser provider.

    Uses `llama-parse` under the hood for parsing PDFs, Word documents, 
    PowerPoints, spreadsheets, and scanned files into structured Markdown.

    Install the optional dependency:
        pip install Muffakir[llamaparse]
        # or directly:
        pip install llama-parse

    Args:
        api_key (str, optional): LlamaCloud API Key. If not provided, LlamaParse will look for 
            the ``LLAMA_CLOUD_API_KEY`` environment variable.
        result_type (str, optional): Format of output text ("markdown" or "text"). Default: "markdown".
        verbose (bool, optional): Verbosity flag for LlamaParse. Default: False.
        language (str, optional): Parsing language hint (e.g. "ar", "en"). Default: "ar".
        **kwargs: Any additional arguments forwarded directly to LlamaParse.

    Example::

        from DocumentParser import create_document_parser

        parser = create_document_parser("llama_parse", api_key="llx-...")
        result = parser.parse_file("report.pdf")
        print(result.text)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        result_type: str = "markdown",
        verbose: bool = False,
        language: str = "ar",
        **kwargs
    ):
        # Lazy-import guard — give a clear error if the package is missing
        try:
            import llama_parse  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'llama-parse' package is required to use LlamaParseParser. "
                "Install it with: pip install llama-parse"
            ) from e

        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "LlamaParse requires an API key. Pass `api_key=...` or set the "
                "LLAMA_CLOUD_API_KEY environment variable."
            )
        self.result_type = result_type
        self.verbose = verbose
        self.language = language
        self.parser_kwargs = kwargs
        self._client = None  # lazily created and memoized LlamaParse client

        # NOTE: libraries must not call logging.basicConfig (mutates host root logger).
        self.logger = logging.getLogger(__name__)

    def _get_parser(self):
        """Return the LlamaParse client, creating it once and caching it on the instance."""
        if self._client is None:
            from llama_parse import LlamaParse

            parser_args = {
                "result_type": self.result_type,
                "verbose": self.verbose,
                "language": self.language,
                "api_key": self.api_key,
                **self.parser_kwargs,
            }
            self._client = LlamaParse(**parser_args)
        return self._client

    def parse_file(self, file_path: str, base_dir: Optional[str] = None) -> ParsedDocument:
        """
        Parse a single file with LlamaParse and return a ``ParsedDocument``.

        Args:
            base_dir: Optional containment root — if given, ``file_path`` must
                resolve within it or ``ConfigurationError`` is raised.
        """
        file_path = self._resolve_within_base(file_path, base_dir)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.logger.info(f"LlamaParse: parsing {file_path} …")

        try:
            parser = self._get_parser()
            documents = parser.load_data(file_path)
        except Exception as e:
            from Muffakir.exceptions import redact_secret

            self.logger.error(
                f"LlamaParse failed on {file_path}: {redact_secret(str(e), self.api_key)}"
            )
            raise classify_parsing_exception(e, backend="llama_parse", secrets=[self.api_key]) from e

        pages: List[PageContent] = []
        full_text_parts: List[str] = []

        for idx, doc in enumerate(documents):
            text = getattr(doc, "text", "") or getattr(doc, "page_content", "") or ""
            full_text_parts.append(text)

            page_num = idx + 1
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                extracted_page = doc.metadata.get("page_label", doc.metadata.get("page_number", idx + 1))
                try:
                    page_num = int(extracted_page)
                except (ValueError, TypeError):
                    page_num = idx + 1

            pages.append(PageContent(page_number=page_num, text=text))

        full_text = "\n\n".join(full_text_parts)

        self.logger.info(
            f"LlamaParse: finished {file_path} — "
            f"{len(pages)} page(s)/section(s), {len(full_text)} chars"
        )

        return ParsedDocument(
            text=full_text,
            source_path=file_path,
            parser_name="llama_parse",
            pages=pages,
            metadata={
                "ocr_processed": True,
                "original_filename": os.path.basename(file_path),
                "result_type": self.result_type,
                "page_count": len(pages),
            },
        )

    def parse_directory(self, directory_path: str, base_dir: Optional[str] = None) -> List[ParsedDocument]:
        """
        Parse every supported file in *directory_path* using LlamaParse.

        Args:
            base_dir: Optional containment root — if given, ``directory_path``
                must resolve within it or ``ConfigurationError`` is raised.
        """
        directory_path = self._resolve_within_base(directory_path, base_dir)
        if not os.path.isdir(directory_path):
            self.logger.warning(f"Directory does not exist: {directory_path}")
            return []

        file_paths = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if self.supports(f)
        ]

        if not file_paths:
            self.logger.warning(f"No supported files found in {directory_path}")
            return []

        self.logger.info(
            f"LlamaParse: parsing {len(file_paths)} file(s) from {directory_path} …"
        )

        results: List[ParsedDocument] = []
        failures: List[str] = []
        for file_path in tqdm(file_paths, desc="LlamaParse: processing files", unit="file", colour="magenta"):
            try:
                parsed_doc = self.parse_file(file_path)
                results.append(parsed_doc)
            except Exception as e:
                failures.append(f"{os.path.basename(file_path)} ({type(e).__name__}: {e})")
                self.logger.error(f"LlamaParse skipping {file_path} due to error: {e}")

        if failures:
            self.logger.warning(
                f"LlamaParse: batch complete — {len(results)} parsed, {len(failures)} failed: "
                + "; ".join(failures)
            )
            if not results:
                raise ParsingError(
                    f"All {len(failures)} file(s) in {directory_path} failed to parse.",
                    context={"directory": directory_path, "failures": failures},
                )
        else:
            self.logger.info(
                f"LlamaParse: batch complete — {len(results)} document(s) parsed"
            )
        return results

    def supported_extensions(self) -> List[str]:
        """
        Supported file extensions for LlamaParse.
        """
        return [
            # Documents & Spreadsheets
            ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
            # Markup / E-books
            ".html", ".htm", ".epub", ".rtf", ".txt",
            # Images
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
        ]
