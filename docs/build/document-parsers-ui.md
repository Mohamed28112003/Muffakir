# Document parsing in ComposerUI

ComposerUI lets you configure OCR and document parsing for your architecture search experiments directly in the visual workspace, without editing Python files or hardcoding credentials.

This page explains the visual setup in ComposerUI. For code-first configuration and Python SDK examples, see [Document parsers in the Python library](document-parsers.md).

---

## Where to configure document parsing

In the **New Search Run** interface (`http://127.0.0.1:2811/index.html`):

1. Configure your corpus path in **Step 1: Data & Pipeline** (`tab-panel-data`).
2. Move to **Step 2: Document Processing** (`tab-panel-parser`).
3. The **Document Parser (OCR)** card configures how source documents in your corpus directory are parsed and read during each trial.

---

## The Document Processing card explained

The **Document Parser (OCR)** card provides two primary controls:

### 1. Parser dropdown

Selects which parsing engine to use:

| Selection | Description | Credentials / Extra Fields |
|---|---|---|
| **None (default)** | Built-in lightweight extraction. Reads standard digital PDFs and text files quickly without OCR overhead. | None required. |
| **docling** | Local open-source document layout and table parser. Best for privacy and zero cloud cost. | Optional export type (`markdown` or `text`). |
| **llama_parse** | Cloud multimodal AI parser. Best for complex layouts, dense tables, and scanned Arabic text. | Requires LlamaCloud API Key (`LLAMA_CLOUD_API_KEY`). |
| **azure** | Enterprise Microsoft Azure Document Intelligence. High-speed OCR and structured forms analysis. | Requires Azure Endpoint URL and API Key. |

### 2. "Use OCR for indexing" toggle

A dedicated checkbox that determines whether files undergo deep OCR analysis:

- **Unchecked (default)**: The pipeline uses standard text extraction. Fast and lightweight. Recommended when documents are digital PDFs with an existing selectable text layer.
- **Checked**: The pipeline routes document ingestion through the selected parser's OCR engine. Required for scanned PDFs, photographed documents, or image files (`.png`, `.jpg`, `.tiff`).

!!! note "Parser selection required for OCR"
    Checking **Use OCR for indexing** requires selecting a parser other than `None (default)`. If left on `None`, Composer will use built-in extraction.

---

## Step-by-step provider configuration in the UI

### Configuring Docling (Local)

1. Select **docling** from the **Parser** dropdown.
2. Ensure you have installed the optional extra in your Python environment:
   ```bash
   pip install "Muffakir[docling]"
   ```
3. Docling processes documents entirely locally on your machine—no external endpoint or API key is needed.

### Configuring LlamaParse (Cloud)

1. Select **llama_parse** from the **Parser** dropdown.
2. In the dynamic fields that appear:
   - Enter your **LlamaCloud API Key** (or leave empty if `LLAMA_CLOUD_API_KEY` is set in your environment).
   - Verify the parsing language is set to `ar` for Arabic corpora.
3. Ensure the dependency is installed:
   ```bash
   pip install "Muffakir[llamaparse]"
   ```

### Configuring Azure Document Intelligence (Enterprise)

1. Select **azure** from the **Parser** dropdown.
2. Enter your **Azure Endpoint URL** (e.g., `https://my-rag-docs.cognitiveservices.azure.com/`).
3. Paste your **Azure API Key**. Credentials entered in the UI are used in-memory for the run process and are **never written to disk** in plain text.
4. Ensure the dependency is installed:
   ```bash
   pip install "Muffakir[azure]"
   ```

---

## Dry-run preflight validation

Before launching an architecture search that may take hours across many trials:

1. Navigate to **Step 7: Review & Launch** (`tab-panel-review`).
2. Click **Dry Run Validate**.
3. Composer verifies that:
   - The corpus directory exists and contains supported document formats.
   - The selected parser library is importable in the environment.
   - Any required API keys or endpoints are non-empty.
   - The combination count and configuration space are fully resolved.

If a dependency is missing, ComposerUI surfaces a clear error banner pointing to the exact `pip install` command needed.

---

## How document parsing affects Composer trials

When Composer evaluates multiple RAG architectures:

- **Cached Document Processing**: Ingestion and chunking are performed systematically for each chunking preset in the search space.
- **Trace Transparency**: The trial telemetry stream records document processing latency and chunk count, making it easy to see how OCR time compares against retrieval and answer-generation stages.
- **Failure Isolation**: If an individual document in a large corpus encounters a parsing error, the parser logs the warning and continues processing remaining documents, preventing a single unreadable page from aborting your entire search run.
