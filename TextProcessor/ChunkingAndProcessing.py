import glob
import logging
import os
from typing import List, Optional

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from DocumentParser import BaseDocumentParser
from .MuffakirChunking import MuffakirChunking
from .MuffakirTextCleaner import MuffakirTextCleaner

logger = logging.getLogger(__name__)

# Backward-compatible monkeypatch seam; populated lazily when text files exist.
TextLoader = None
# Kept as a lazy seam so applications and tests can supply a compatible PDF
# reader without importing PyPDF during ordinary module discovery.
PdfReader = None


class ChunkingAndProcessing:
    """
    Data Ingestion Pipeline.
    Responsible for loading documents from a directory (or via OCR parser) and
    delegating the chunking and text cleaning to MuffakirChunking.
    """

    def __init__(
        self,
        directory_path: str,
        chunk_size: int = 600,
        chunk_overlap: int = 200,
        document_parser: Optional[BaseDocumentParser] = None,
        language: Optional[str] = None,
        muffakir_chunking: Optional[MuffakirChunking] = None,
    ):
        """
        Initialize the Data Ingestion Pipeline.

        Args:
            directory_path: Path to directory containing documents.
            chunk_size: Backward compatibility (used only when muffakir_chunking is not provided).
            chunk_overlap: Backward compatibility (used only when muffakir_chunking is not provided).
            document_parser: Parser to use if OCR/extraction is needed.
            language: Backward compatibility language hint for the fallback chunker.
            muffakir_chunking: Injected MuffakirChunking orchestrator. Overrides other chunk parameters.
        """
        self.directory_path = directory_path
        self.document_parser = document_parser
        self.logger = logging.getLogger(__name__)

        # Set up the chunking orchestrator
        if muffakir_chunking:
            self.chunking_orchestrator = muffakir_chunking
        else:
            self.logger.info(
                "No MuffakirChunking provided. Falling back to default recursive chunker."
            )
            self.chunking_orchestrator = MuffakirChunking(
                chunker="recursive",
                chunker_config={
                    "size": chunk_size,
                    "overlap": chunk_overlap,
                },
                language=language or "auto",
            )

    def load_documents(self) -> List[Document]:
        """
        Load all .txt and .pdf documents from the specified directory.

        .txt files are loaded with LangChain's text loader.
        .pdf files are loaded page-by-page with PyPDF.  This deliberately
        avoids the much heavier Unstructured dependency tree.
        """
        documents: List[Document] = []

        # --- Load plain-text files using TextLoader (no unstructured dep) ---
        txt_files = glob.glob(
            os.path.join(self.directory_path, "**", "*.txt"), recursive=True
        )
        txt_failures: List[tuple] = []
        if txt_files:
            from Muffakir.optional_dependencies import require_optional_dependency

            require_optional_dependency("rag")
            loader_class = TextLoader
            if loader_class is None:
                from langchain_community.document_loaders import TextLoader as loader_class

        for txt_path in txt_files:
            try:
                loader = loader_class(txt_path, encoding="utf-8", autodetect_encoding=True)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                txt_failures.append((txt_path, str(e)))
                self.logger.warning("Could not load '%s': %s", txt_path, e)

        if txt_files and len(txt_failures) == len(txt_files):
            from Muffakir.exceptions import ParsingError

            raise ParsingError(
                f"Failed to load all {len(txt_files)} .txt file(s) in '{self.directory_path}'.",
                context={"directory": self.directory_path, "failures": txt_failures},
            )

        # --- Load text-based PDF files with lightweight PyPDF ---
        pdf_files = glob.glob(
            os.path.join(self.directory_path, "**", "*.pdf"), recursive=True
        )
        if pdf_files:
            from Muffakir.optional_dependencies import require_optional_dependency

            require_optional_dependency("pdf")
            reader_class = PdfReader
            if reader_class is None:
                from pypdf import PdfReader as reader_class

            pdf_failures: List[tuple] = []
            for pdf_path in pdf_files:
                try:
                    reader = reader_class(pdf_path)
                    for page_number, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        if not text.strip():
                            continue
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": pdf_path,
                                    "page": page_number,
                                    "page_label": str(page_number + 1),
                                },
                            )
                        )
                except Exception as e:
                    pdf_failures.append((pdf_path, str(e)))
                    self.logger.warning("Could not load '%s': %s", pdf_path, e)

            if len(pdf_failures) == len(pdf_files):
                from Muffakir.exceptions import ParsingError

                raise ParsingError(
                    f"Failed to load all {len(pdf_files)} .pdf file(s) in "
                    f"'{self.directory_path}'.",
                    context={"directory": self.directory_path, "failures": pdf_failures},
                )

        if not documents:
            self.logger.warning(
                "No .txt or .pdf documents found in '%s'.", self.directory_path
            )
        else:
            self.logger.info(
                "Loaded %d documents from '%s'.", len(documents), self.directory_path
            )

        return documents

    def process_all(
        self,
        chunking_method: Optional[str] = None,
        use_ocr: bool = False,
        ocr_output_dir: Optional[str] = None,
    ) -> List[Document]:
        """
        Complete processing pipeline: load, OCR (optional), clean, and chunk.

        Args:
            chunking_method: Backward-compat parameter (ignored when muffakir_chunking was injected).
            use_ocr: Whether to load documents via the document_parser (OCR mode).
            ocr_output_dir: Optional directory to persist raw OCR text output.

        Returns:
            List[Document]: Final processed and chunked documents.
        """
        try:
            # 1. Load documents
            if use_ocr:
                if not self.document_parser:
                    raise ValueError(
                        "A document_parser must be provided when use_ocr is True."
                    )
                parsed_docs = self.document_parser.parse_directory(self.directory_path)

                # Save OCR results if an output directory is specified
                if ocr_output_dir and parsed_docs:
                    os.makedirs(ocr_output_dir, exist_ok=True)
                    for pdoc in parsed_docs:
                        filename = os.path.basename(pdoc.source_path)
                        output_file = os.path.join(
                            ocr_output_dir,
                            f"{os.path.splitext(filename)[0]}_ocr.txt",
                        )
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(pdoc.text)

                # Convert ParsedDocument to LangChain Document
                documents = [
                    Document(
                        page_content=pdoc.text,
                        metadata={
                            "source": pdoc.source_path,
                            "ocr_processed": True,
                            "original_filename": os.path.basename(pdoc.source_path),
                            "parser_name": pdoc.parser_name,
                            **pdoc.metadata,
                        },
                    )
                    for pdoc in parsed_docs
                ]
            else:
                documents = self.load_documents()

            if not documents:
                self.logger.warning("No documents found to process.")
                return []

            # 2. Process using the orchestrator (Cleaning + Chunking)
            # If a legacy method was passed to process_all(), temporarily override the orchestrator
            if chunking_method and chunking_method != self.chunking_orchestrator.chunker.name:
                self.logger.warning(
                    "Legacy parameter 'chunking_method=%s' passed to process_all(). "
                    "Overriding injected chunker.",
                    chunking_method,
                )
                temp_orchestrator = MuffakirChunking(
                    chunker=chunking_method,
                    chunker_config={
                        "size": getattr(self.chunking_orchestrator.chunker, "size", 600),
                        "overlap": getattr(self.chunking_orchestrator.chunker, "overlap", 200),
                    },
                    text_cleaner=self.chunking_orchestrator.text_cleaner,
                )
                final_docs = temp_orchestrator.process(documents)
            else:
                final_docs = self.chunking_orchestrator.process(documents)

            # 3. Add chunk IDs and source_file metadata for backward compatibility
            for i, doc in enumerate(final_docs):
                doc.metadata["chunk_id"] = i + 1
                doc.metadata["chunk_size"] = len(doc.page_content)
                if "source" in doc.metadata and "source_file" not in doc.metadata:
                    filename = os.path.basename(doc.metadata["source"])
                    doc.metadata["source_file"] = os.path.splitext(filename)[0]

            self.logger.info(
                "Processing complete. Final document count: %d", len(final_docs)
            )
            return final_docs

        except Exception as e:
            self.logger.error("Error in processing pipeline: %s", str(e))
            raise

    # ------------------------------------------------------------------
    # Deprecated Aliases (kept for backward compatibility with older code)
    # ------------------------------------------------------------------

    @staticmethod
    def clean_text(text: str, language: Optional[str] = None) -> str:
        """Deprecated alias for MuffakirTextCleaner."""
        cleaner = MuffakirTextCleaner(language=language or "auto")
        return cleaner.clean(text)

    @staticmethod
    def clean_arabic_text(text: str) -> str:
        """Deprecated alias for clean_text(text, language='ar')."""
        return ChunkingAndProcessing.clean_text(text, language="ar")
