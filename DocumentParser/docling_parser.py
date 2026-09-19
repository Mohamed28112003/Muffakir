import os
import logging
from typing import List, Optional, Tuple
from tqdm import tqdm
from .base import BaseDocumentParser
from .models import ParsedDocument, PageContent
from Muffakir.exceptions import ParsingError, classify_parsing_exception


class DoclingParser(BaseDocumentParser):
    """
    Docling document parser provider.

    Uses langchain-docling (DoclingLoader) under the hood.
    Supports a wide variety of document formats: PDF, DOCX, PPTX,
    XLSX, HTML, Markdown, images, and more.

    Install the optional dependency:
        pip install Muffakir[docling]
        # or directly:
        pip install langchain-docling

    Args:
        export_type (str, optional): Controls how Docling exports document content.
            - ``"markdown"`` (default): rich structured text with headings and tables.
            - ``"text"``: plain text, no structure markers.
        **kwargs: Any additional keyword arguments accepted by ``DoclingLoader``
            (e.g. ``converter_kwargs``, ``export_kwargs``).

    Example::

        from DocumentParser import create_document_parser

        parser = create_document_parser("docling", export_type="markdown")
        result = parser.parse_file("report.pdf")
        print(result.text)
        print(result.pages[0].text)
    """

    def __init__(self, export_type: str = "markdown", **kwargs):
        # Lazy-import guard — give a clear error if the package is missing
        try:
            import langchain_docling  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'langchain-docling' package is required to use DoclingParser. "
                "Install it with: pip install langchain-docling"
            ) from e

        self.export_type = export_type
        self.loader_kwargs = kwargs  # forwarded verbatim to DoclingLoader

        # NOTE: libraries must not call logging.basicConfig (mutates host root logger).
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # BaseDocumentParser interface
    # ------------------------------------------------------------------

    def parse_file(self, file_path: str, base_dir: Optional[str] = None) -> ParsedDocument:
        """
        Parse a single file with Docling and return a ``ParsedDocument``.

        Each LangChain Document returned by DoclingLoader becomes one
        ``PageContent`` entry (Docling segments by semantic section, not
        necessarily by page number).

        Args:
            base_dir: Optional containment root — if given, ``file_path`` must
                resolve within it or ``ConfigurationError`` is raised.
        """
        file_path = self._resolve_within_base(file_path, base_dir)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        from langchain_docling import DoclingLoader

        self.logger.info(f"Docling: parsing {file_path} …")

        try:
            loader = DoclingLoader(
                file_path=file_path,
                export_type=self.export_type,
                **self.loader_kwargs,
            )
            lc_docs = loader.load()
        except Exception as e:
            self.logger.error(f"Docling failed on {file_path}: {e}")
            raise classify_parsing_exception(e, backend="docling") from e

        pages: List[PageContent] = []
        full_text_parts: List[str] = []

        for idx, doc in enumerate(lc_docs):
            text = doc.page_content or ""
            full_text_parts.append(text)

            # Docling stores rich metadata in doc.metadata["dl_meta"] when
            # available; fall back to the sequential index if absent.
            page_num = self._extract_page_number(doc.metadata, idx)
            pages.append(PageContent(page_number=page_num, text=text))

        # Standardized separator contract: sections joined by "\n\n".
        full_text = "\n\n".join(full_text_parts)

        self.logger.info(
            f"Docling: finished {file_path} — "
            f"{len(pages)} section(s), {len(full_text)} chars"
        )

        return ParsedDocument(
            text=full_text,
            source_path=file_path,
            parser_name="docling",
            pages=pages,
            metadata={
                "ocr_processed": True,
                "original_filename": os.path.basename(file_path),
                "export_type": self.export_type,
                "section_count": len(pages),
            },
        )

    def parse_directory(self, directory_path: str, base_dir: Optional[str] = None) -> List[ParsedDocument]:
        """
        Parse every supported file in *directory_path*.

        Docling natively accepts a list of paths so we call ``DoclingLoader``
        once for the whole directory and then group the resulting documents
        back by source file.

        Args:
            base_dir: Optional containment root — if given, ``directory_path``
                must resolve within it or ``ConfigurationError`` is raised.
        """
        from langchain_docling import DoclingLoader

        directory_path = self._resolve_within_base(directory_path, base_dir)
        if not os.path.isdir(directory_path):
            self.logger.warning(f"Directory does not exist: {directory_path}")
            return []

        # Collect all supported files
        file_paths = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if self.supports(f)
        ]

        if not file_paths:
            self.logger.warning(f"No supported files found in {directory_path}")
            return []

        self.logger.info(
            f"Docling: parsing {len(file_paths)} file(s) from {directory_path} …"
        )

        try:
            loader = DoclingLoader(
                file_path=file_paths,
                export_type=self.export_type,
                **self.loader_kwargs,
            )
            lc_docs = loader.load()
        except Exception as e:
            self.logger.error(f"Docling batch parse failed: {e}")
            raise classify_parsing_exception(e, backend="docling") from e

        # Group LangChain Documents by their source file.
        # DoclingLoader must set metadata["source"]; if any doc lacks it we fail
        # loudly rather than silently bucketing everything into file_paths[0]
        # (which would mis-attribute content to the wrong source document).
        grouped: dict = {}
        missing_source_count = 0
        for idx, doc in tqdm(
            enumerate(lc_docs),
            total=len(lc_docs),
            desc="Docling: grouping sections",
            unit="section",
            colour="cyan",
        ):
            src = doc.metadata.get("source") if isinstance(doc.metadata, dict) else None
            if not src:
                missing_source_count += 1
                continue
            if src not in grouped:
                grouped[src] = []
            grouped[src].append((idx, doc))

        if missing_source_count > 0:
            raise ParsingError(
                f"Docling returned {missing_source_count} document(s) without a "
                f"'source' metadata key; cannot safely group sections by file. "
                f"Please verify your langchain-docling version.",
                context={"directory": directory_path, "missing_source_count": missing_source_count},
            )

        results: List[ParsedDocument] = []
        for src_path, doc_items in tqdm(
            grouped.items(),
            desc="Docling: building documents",
            unit="file",
            colour="green",
        ):
            pages: List[PageContent] = []
            text_parts: List[str] = []

            for global_idx, doc in doc_items:
                text = doc.page_content or ""
                text_parts.append(text)
                page_num = self._extract_page_number(doc.metadata, global_idx)
                pages.append(PageContent(page_number=page_num, text=text))

            results.append(
                ParsedDocument(
                    # Standardized separator contract: sections joined by "\n\n".
                    text="\n\n".join(text_parts),
                    source_path=src_path,
                    parser_name="docling",
                    pages=pages,
                    metadata={
                        "ocr_processed": True,
                        "original_filename": os.path.basename(src_path),
                        "export_type": self.export_type,
                        "section_count": len(pages),
                    },
                )
            )

        self.logger.info(
            f"Docling: batch complete — {len(results)} document(s) parsed"
        )
        return results

    def supported_extensions(self) -> List[str]:
        """
        File types that Docling can handle.
        Ref: https://ds4sd.github.io/docling/supported_formats/
        """
        return [
            # Documents
            ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
            # Web / markup
            ".html", ".htm", ".md",
            # Images (Docling performs OCR on these)
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
            # Plain text
            ".txt",
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_page_number(metadata: dict, fallback: int) -> int:
        """
        Try to extract a page number from Docling's metadata dict.

        Docling stores structured info under ``dl_meta``:
            metadata["dl_meta"]["doc_items"][0]["prov"][0]["page_no"]
        We attempt a safe traversal and fall back to the document index.
        """
        try:
            dl_meta = metadata.get("dl_meta", {})
            doc_items = dl_meta.get("doc_items", [])
            if doc_items:
                prov = doc_items[0].get("prov", [])
                if prov:
                    return int(prov[0].get("page_no", fallback + 1))
        except (TypeError, KeyError, IndexError, ValueError):
            pass
        return fallback + 1  # make 1-indexed
