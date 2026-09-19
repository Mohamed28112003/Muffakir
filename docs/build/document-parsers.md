# Document parsers and OCR in the Python library

Real-world Arabic RAG applications deal with a wide spectrum of documents: clean digital text, scanned historical manuscripts, photographed legal contracts, dense financial spreadsheets, multi-column research reports, and presentation slides.

Muffakir provides a unified **`DocumentParser`** module that standardizes all document formats into a consistent document contract (`ParsedDocument`). You select a parser provider, pass provider-specific configuration or credentials, and Muffakir handles lazy import loading, file validation, batch resilience, and structured output.

For the visual interface, OCR toggling, and wizard workflow in Composer, see [Document parsing in ComposerUI](document-parsers-ui.md).

---

## Start with a RAG application

To enable document parsing in `MuffakirRAG`, set `document_parser`, `document_parser_config`, and optionally `use_ocr`.

=== "Docling (Local)"

    ```python
    from Muffakir import MuffakirRAG

    rag = MuffakirRAG(
        data_dir="./knowledge-base",
        document_parser="docling",
        document_parser_config={"export_type": "markdown"},
        use_ocr=True,
        chunking_method="recursive",
        chunk_size=600,
        chunk_overlap=150,
        language="ar",
    )

    result = rag.ask("ما هي المبادئ الأساسية المذكورة في التقرير؟")
    print(result["answer"])
    ```

=== "LlamaParse (Multimodal Cloud)"

    ```python
    import os
    from Muffakir import MuffakirRAG

    rag = MuffakirRAG(
        data_dir="./knowledge-base",
        document_parser="llama_parse",
        document_parser_config={
            "api_key": os.environ["LLAMA_CLOUD_API_KEY"],
            "language": "ar",
            "result_type": "markdown",
        },
        use_ocr=True,
        chunking_method="recursive",
        chunk_size=700,
        chunk_overlap=100,
        language="ar",
    )

    result = rag.ask("ما هي شروط العقد؟")
    print(result["answer"])
    ```

=== "Azure Document Intelligence (Enterprise)"

    ```python
    import os
    from Muffakir import MuffakirRAG

    rag = MuffakirRAG(
        data_dir="./knowledge-base",
        document_parser="azure",
        document_parser_config={
            "endpoint": os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
            "api_key": os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"],
            "model_id": "prebuilt-layout",
        },
        use_ocr=True,
        language="ar",
    )

    result = rag.ask("ما هو إجمالي الفاتورة؟")
    print(result["answer"])
    ```

---

## Built-in extraction vs. dedicated parsers

Before selecting a parser, understand the distinction between standard text extraction and OCR-driven document parsing:

| Feature | Built-in Default Extraction | `DocumentParser` Providers |
|---|---|---|
| **Best suited for** | Clean digital PDFs, `.txt`, `.md` | Scanned pages, photographed documents, dense tables, multi-column layouts |
| **OCR Capability** | None (reads digital text layer only) | Full OCR (Docling, LlamaParse, Azure) |
| **Dependencies** | None (standard library / core) | Optional extras (`[docling]`, `[llamaparse]`, `[azure]`) |
| **External API** | None (fully offline) | Offline (Docling) or Cloud API (LlamaParse, Azure) |
| **Output format** | Plain text string | Standardized `ParsedDocument` with page breakdown, confidence scores, and Markdown tables |

!!! tip "When to turn on `use_ocr`"
    If your documents are standard, text-selectable digital PDFs, the built-in extractor is much faster and uses fewer computing resources. Enable `use_ocr=True` and configure a `document_parser` when dealing with scanned documents, photographed papers, or complex tables where reading order matters.

---

## Supported parser providers

Muffakir supports three production-grade document parsers:

| Provider Key | Type | Best Used For | Supported Extensions | Install Extra |
|---|---|---|---|---|
| `docling` | Local / Open Source | Privacy-sensitive local environments, table extraction, zero API costs. | `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.md`, `.txt`, `.jpg`, `.png`, `.tiff`, `.webp` | `Muffakir[docling]` |
| `llama_parse` | Multimodal Cloud | Complex layout analysis, charts, dense tables, handwritten/scanned Arabic text. | `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.epub`, `.rtf`, `.txt`, `.jpg`, `.png`, `.tiff`, `.webp` | `Muffakir[llamaparse]` |
| `azure` | Enterprise Cloud | Microsoft Azure Document Intelligence, high-speed cloud OCR, page confidence scores. | `.pdf`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.txt` | `Muffakir[azure]` |

---

## Detailed provider guides

### 1. Docling (`docling`)

Docling runs locally and uses advanced layout models to convert PDF, Office, and image files into clean Markdown or text. It extracts reading order, headers, lists, and tables without external API calls.

```bash
pip install "Muffakir[docling]"
```

#### Standalone usage

```python
from DocumentParser import create_document_parser

parser = create_document_parser(
    "docling",
    export_type="markdown",  # "markdown" (default) or "text"
)

# Parse a single document
doc = parser.parse_file("./contracts/agreement.pdf")
print(f"Parser used: {doc.parser_name}")
print(f"Extracted pages: {len(doc.pages)}")
print(doc.text[:500])
```

#### Supported options (`document_parser_config`)

- `export_type` (*str*): Controls the export format. `"markdown"` (default) preserves table structure and headings; `"text"` emits raw plain text.
- Additional keyword arguments are forwarded directly to `DoclingLoader` (e.g., `converter_kwargs`, `export_kwargs`).

---

### 2. LlamaParse (`llama_parse`)

LlamaParse is a cloud-based multimodal parsing service specifically tuned for complex enterprise documents, embedded charts, dense spreadsheets, and multi-lingual text including Arabic.

```bash
pip install "Muffakir[llamaparse]"
```

#### Standalone usage

```python
import os
from DocumentParser import create_document_parser

parser = create_document_parser(
    "llama_parse",
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    language="ar",              # Optimizes character recognition for Arabic
    result_type="markdown",     # "markdown" (default) or "text"
    verbose=False,
)

doc = parser.parse_file("./scanned_archive/decree.pdf")
```

#### Supported options (`document_parser_config`)

- `api_key` (*str*, optional): LlamaCloud API Key. If omitted, reads from the `LLAMA_CLOUD_API_KEY` environment variable.
- `language` (*str*): Language hint passed to LlamaParse (e.g., `"ar"`, `"en"`). Default: `"ar"`.
- `result_type` (*str*): Output format (`"markdown"` or `"text"`). Default: `"markdown"`.
- `verbose` (*bool*): Enables verbose parsing logs. Default: `False`.
- Additional keyword arguments are forwarded directly to `LlamaParse`.

---

### 3. Azure Document Intelligence (`azure`)

Azure Document Intelligence (formerly Form Recognizer) is an enterprise cloud service with high-accuracy OCR, layout analysis, and per-page confidence scoring.

```bash
pip install "Muffakir[azure]"
```

#### Standalone usage

```python
import os
from DocumentParser import create_document_parser

parser = create_document_parser(
    "azure",
    endpoint=os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
    api_key=os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"],
    model_id="prebuilt-layout",
    timeout_seconds=300.0,
    max_retries=2,
)

doc = parser.parse_file("./finance/q3_report.pdf")
```

#### Supported options (`document_parser_config`)

- `endpoint` (*str*, required): The Azure Document Intelligence endpoint URL (e.g., `https://<service-name>.cognitiveservices.azure.com/`).
- `api_key` (*str*, required): Your Azure service key.
- `model_id` (*str*): Analysis model ID. Default: `"prebuilt-layout"`.
- `timeout_seconds` (*float*): Maximum seconds to wait for document analysis. Default: `300.0`.
- `max_retries` (*int*): Maximum automatic retries on transient network errors or HTTP 429 / 5xx responses with exponential backoff. Default: `2`.

---

## The `ParsedDocument` output contract

Regardless of which parser backend is used, every parser returns a standardized **`ParsedDocument`** dataclass. This guarantees consistent behavior across the rest of the Muffakir SDK.

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class PageContent:
    page_number: int                   # 1-indexed page or section number
    text: str                          # Content of this individual page
    confidence: Optional[float] = None # OCR confidence score (0.0 to 1.0) if provided

@dataclass
class ParsedDocument:
    text: str                          # Full extracted text (pages joined by "\n\n")
    source_path: str                   # Absolute or original path of the file
    parser_name: str                   # "docling", "llama_parse", or "azure"
    pages: List[PageContent]           # List of individual page contents
    metadata: Dict[str, Any]           # Backend metadata (file type, OCR flags, etc.)
```

### Text join contract

The full document `text` is formed by joining all pages with `\n\n`. Every concrete parser adheres strictly to this contract so downstream chunkers receive uniform paragraph and section breaks.

### Inspecting page content and OCR confidence

```python
doc = parser.parse_file("./handbook.pdf")

for page in doc.pages:
    print(f"--- Page {page.page_number} ---")
    print(f"Characters: {len(page.text)}")
    if page.confidence is not None:
        print(f"OCR Confidence: {page.confidence:.2%}")
```

---

## Batch directory parsing with resilience

Use `parse_directory()` to process entire folders containing mixed document formats.

```python
parser = create_document_parser("docling")
docs = parser.parse_directory("./corporate_docs")

print(f"Successfully parsed {len(docs)} documents.")
```

### Partial failure tolerance

Real-world directories often contain corrupted, password-protected, or unreadable files. Rather than aborting the entire ingestion process when one file fails:

1. The failed file is logged with an error explaining the cause.
2. The parser continues processing all remaining files.
3. At the end of the batch, a summary of successes and failures is logged.
4. If at least one file succeeded, the parsed documents are returned. Only if **all** files fail is a `ParsingError` raised.

---

## Security: Path-traversal containment (`base_dir`)

When exposing document parsing over an API, user uploads, or multi-tenant services, supply the `base_dir` parameter to prevent directory traversal attacks (such as `../../etc/passwd`):

```python
# Raises ConfigurationError if requested_file escapes /var/data/uploads
doc = parser.parse_file(
    file_path=user_submitted_path,
    base_dir="/var/data/uploads",
)
```

---

## Best practices for Arabic document parsing

1. **Enable Arabic language hints**: When using LlamaParse, always pass `language="ar"`. This informs the OCR engine to prioritize Arabic character recognition and diacritics.
2. **Combine with `MuffakirTextCleaner`**: Parsed text frequently contains erratic whitespace or broken diacritics. Muffakir automatically runs text cleaning on parsed content before chunking.
3. **Preserve source metadata**: `ParsedDocument.source_path` and `page_number` are attached to each generated chunk, allowing cited answers in your RAG app to point directly back to the original page.
