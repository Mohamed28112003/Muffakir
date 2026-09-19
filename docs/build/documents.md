# Documents and chunking

Good retrieval starts with a reliable representation of your source material. Muffakir separates parsing, OCR, text cleaning, and chunking so that you can tune and evaluate each stage independently.

For dedicated deep-dives, see:
- [Document parsers in the Python library](document-parsers.md) for backend recipes, parameters, and batching.
- [Document parsing in ComposerUI](document-parsers-ui.md) for visual workflow and preflight validation.

---

## Document parsing and OCR architecture

Documents in real-world Arabic RAG pipelines range from clean digital text to complex scanned PDFs, photographs of printed Arabic manuscripts, multi-column reports, and dense spreadsheets. Muffakir provides a unified **`DocumentParser`** layer that normalizes all formats into a standardized document contract.

### Built-in extraction vs. dedicated parsers

- **Built-in default**: Best for clean, digital-native text files (`.txt`, `.pdf`). It is lightweight, fast, and does not require heavy external libraries or cloud API keys.
- **`DocumentParser`**: Required when handling scanned pages, complex document layouts, embedded tables, presentations, spreadsheets, or when high-fidelity OCR is needed.

---

## Supported parser providers

Muffakir supports three production-grade parsing backends through the `DocumentParser` module:

| Parser | Type | Best Used For | Supported Formats | Required Dependency |
|---|---|---|---|---|
| **Docling** (`docling`) | Local / Open-Source | High-accuracy local parsing, layout analysis, table extraction, and clean Markdown export without sending data to external APIs. | `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.md`, `.txt`, `.jpg`, `.png`, `.tiff`, `.webp` | `pip install "Muffakir[docling]"` |
| **LlamaParse** (`llama_parse`) | Cloud API | State-of-the-art multimodal parsing for complex layouts, dense financial tables, charts, and Arabic handwritten/scanned documents. | `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.epub`, `.rtf`, `.txt`, `.jpg`, `.png`, `.tiff`, `.webp` | `pip install "Muffakir[llamaparse]"` |
| **Azure Document Intelligence** (`azure`) | Cloud API | Enterprise-grade cloud OCR and document layout analysis with page-level confidence scores and SLA guarantees. | `.pdf`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.txt` | `pip install "Muffakir[azure]"` |

### 1. Docling (Local & Privacy-Preserving)

Docling runs locally and converts complex documents into structured Markdown or plain text. It extracts reading order and tabular structures without external network calls.

```bash
pip install "Muffakir[docling]"
```

```python
from DocumentParser import create_document_parser

parser = create_document_parser(
    "docling",
    export_type="markdown",  # "markdown" (default) or "text"
)
doc = parser.parse_file("./knowledge-base/arabic_report.pdf")

print(f"Parsed {len(doc.pages)} pages via {doc.parser_name}")
print(doc.text[:500])
```

### 2. LlamaParse (Multimodal Cloud Parser)

LlamaParse uses multimodal AI to parse dense documents, complex tables, and scanned pages into clean Markdown, with dedicated Arabic language support.

```bash
pip install "Muffakir[llamaparse]"
```

```python
import os
from DocumentParser import create_document_parser

parser = create_document_parser(
    "llama_parse",
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    language="ar",              # Optimizes OCR for Arabic text
    result_type="markdown",     # "markdown" or "text"
    verbose=False,
)
doc = parser.parse_file("./knowledge-base/scanned_contract.pdf")
```

### 3. Azure Document Intelligence (Enterprise OCR)

Azure Document Intelligence (formerly Form Recognizer) provides enterprise OCR with field-level analysis and confidence scoring.

```bash
pip install "Muffakir[azure]"
```

```python
import os
from DocumentParser import create_document_parser

parser = create_document_parser(
    "azure",
    endpoint=os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
    api_key=os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"],
    model_id="prebuilt-layout",  # Default prebuilt layout model
    timeout_seconds=300.0,
    max_retries=2,
)
doc = parser.parse_file("./knowledge-base/invoice.pdf")
```

---

## The `ParsedDocument` output contract

Every parser in Muffakir returns a unified **`ParsedDocument`** instance, ensuring downstream chunking, cleaning, and embedding behave identically regardless of which parser produced the text.

```python
@dataclass
class ParsedDocument:
    text: str                          # Full extracted text (pages joined by "\n\n")
    source_path: str                   # Original file path
    parser_name: str                   # "docling", "llama_parse", or "azure"
    pages: List[PageContent]           # Page-by-page breakdown
    metadata: Dict[str, Any]           # Backend-specific extraction details
```

### Inspecting page-level text and confidence

When analyzing OCR results, you can inspect individual pages and their confidence scores:

```python
for page in doc.pages:
    print(f"Page {page.page_number}: {len(page.text)} characters")
    if page.confidence is not None:
        print(f"  OCR Confidence: {page.confidence:.2%}")
```

### Resilient batch directory parsing

Use `parse_directory()` to parse an entire folder of mixed formats. Batch operations include progress bars (`tqdm`) and **partial failure tolerance**: if a corrupt file fails to parse, it is logged and summarized without aborting the rest of the batch.

```python
parser = create_document_parser("docling", export_type="markdown")
docs = parser.parse_directory("./knowledge-base")

print(f"Successfully parsed {len(docs)} documents.")
```

### Path-traversal containment (`base_dir`)

When exposing document parsing over web APIs or user uploads, pass `base_dir` to ensure file paths cannot escape the designated root directory:

```python
# Raises ConfigurationError if file_path escapes /safe/storage
doc = parser.parse_file(user_provided_path, base_dir="/safe/storage")
```

---

## Integrating with `MuffakirRAG`

To use a document parser inside a full RAG pipeline, configure `document_parser`, `document_parser_config`, and `use_ocr` on `MuffakirRAG`:

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
    use_ocr=True,                  # Use the parser for OCR-driven extraction
    chunking_method="recursive",
    chunk_size=700,
    chunk_overlap=100,
    llm_provider="openai",
    llm_model="gpt-4o-mini",
)

response = rag.ask("ما هي شروط العقد المذكورة في الوثيقة؟")
print(response)
```

---

## Document processing in ComposerUI

In the visual **ComposerUI** workspace:

1. Open **Step 2: Document Processing** (`tab-panel-parser`).
2. Select your desired parser from the **Parser** dropdown (`None (default)`, `docling`, `llama_parse`, or `azure`).
3. Fill in backend-specific credentials (endpoint, API key, export format, or language hint).
4. Check **Use OCR for indexing** if your corpus contains scanned or image-based PDFs.
5. Composer validates credentials and dependencies during the **Dry Run Validate** step before starting architecture searches.

---

## Clean Arabic and mixed-language text

`MuffakirTextCleaner` normalizes Arabic text, strips tatweel (kashida), unifies diacritics, and handles mixed Arabic/English corpora:

- Use `language="ar"` for purely Arabic corpora.
- Use `language="auto"` for mixed datasets (Arabic and English).
- Original source metadata is preserved so that generated answers can always be traced back to exact source passages.

---

## Choose a chunking strategy

Once documents are parsed and cleaned, chunking segments text into manageable semantic blocks:

| Strategy | When to use |
|---|---|
| **Recursive** (`recursive`) | General-purpose default; respects paragraph and sentence boundaries. |
| **Fixed size** (`fixed_size`) | Content has uniform structure and strict chunk length constraints. |
| **Sliding window** (`sliding_window`) | Continuous narratives where overlap across chunk boundaries is critical. |
| **Token** (`token`) | When matching strict LLM context token budgets. |
| **Semantic** (`semantic`) | Breaks chunks at natural topic transitions rather than character counts. |
| **Contextual** (`contextual`) | Enriches each chunk with document-level context before embedding. |

Next: [embeddings](embeddings.md).
