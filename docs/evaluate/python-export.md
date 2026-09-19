# Export a trial to Python

Turn a selected Composer trial into an application you can run and extend. You can
export **any saved trial with a usable configuration**, including a failed trial.
The score does not determine whether export is available.

## From ComposerUI to code

1. Open a run and select a trial from the trial table, or open the best result.
2. Select **Export Python** in the trial inspector.
3. Review the trial ID, original status, pipeline mode, requirements, and code.
4. Choose **Copy code**, **Download Python**, or **Download project ZIP**.

The ZIP contains `rag_app.py`, `requirements.txt`, `.env.example`, `.gitignore`, and
a setup `README.md`. The Python file is identical to the code shown in the preview.
You can export a saved trial while other trials continue running.

## LLM parameters in exported code

Exports retain the selected generation variant's `llm_parameters`, plus configured
`query_transform_llm_parameters` and `reranker_llm_parameters` for active runtime
stages. Temperature omission, token limits, stop sequences, and supported request
settings are preserved. Judge and synthetic-data settings are intentionally excluded
because the exported application answers questions rather than running an experiment.

## Run your exported project

Use Python 3.11 or later and activate a virtual environment. Extract the ZIP, then:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Supply the indicated credentials and connection
settings. Environment variables already set in your shell take precedence.

For corpus pipelines, set `MUFFAKIR_DOCUMENTS_DIR` to your document directory:

```dotenv
MUFFAKIR_DOCUMENTS_DIR=./knowledge-base
MUFFAKIR_INDEX_DIR=./index
```

Build the index once, then query it:

```bash
python rag_app.py --index
python rag_app.py --question "What does our documentation say about refunds?"
```

**Web Search Only** needs no local documents or indexing: configure the generation
and web-search providers, then run with `--question` directly.

You can also reuse the pipeline from Python:

```python
from rag_app import build_pipeline

pipeline = build_pipeline()
response = pipeline.ask("What does our documentation say about refunds?")
print(response)
```

For retrieval-only exports, use `pipeline.get_similar_documents(question)` to obtain
documents and metadata. Importing the generated module does not initialize models.

## What is preserved

The export retains the selected trial's chunking, embedding model, vector database,
retrieval method and depth, reranking, query transformation, generation settings,
language, runtime prompts, and Adaptive RAG settings. Separate stage providers remain
separate. Recorded runtime prompt snapshots are included as overrides when available.

Full RAG and Adaptive RAG use `MuffakirRAG`; retrieval-only uses `MuffakirRetrieval`;
web-search-only uses `MuffakirSearch`. Search execution, evaluation judges, dataset
generation, and ranking settings are not run by the exported application.

## Documents, indexes, and external services

Exports contain code and setup instructions. Supply your own documents and credentials;
Composer's indexes, sample answers, and corpus files are not included. The application
builds its own vector index and saves processed chunks for hybrid retrieval after restart.

Indexing requires a **fresh** index directory. Repeated or concurrent indexing into
the same directory is refused, including after a partial failure. To rebuild, choose
a new `MUFFAKIR_INDEX_DIR` and a dedicated collection name with `MUFFAKIR_COLLECTION`.
Querying uses the completed index without ingesting documents again.

The selected vector database provider stays the same. Qdrant and Milvus need connection
settings; Pinecone needs an existing empty index with dimensions matching your embedding
model. Configure a dedicated target for exported projects. Provider credentials and
custom endpoint/header values are supplied through the generated environment variables.

Hugging Face models use the standard cache and may download on first use. The saved
CPU/CUDA setting is retained. See [installation](../getting-started/installation.md)
for optional dependencies and [document parsing](../build/document-parsers.md) for parser setup.

## Failed and older trials

A failed trial can still be useful for debugging. Its export shows the original status;
running it may reproduce the same failure. An export is not evidence of a successful run.

If a saved trial lacks essential configuration, the inspector explains why export is
unavailable. It does not replace your selection with another trial. Older runs without
complete prompt snapshots use available overrides and identify missing templates that
will come from the installed version.

Dependencies pin the exporting Muffakir version. For unreleased changes, use the matching
source checkout as described in the generated README. Identical configuration does not
guarantee identical answers or scores when documents, models, or web results change.

## ComposerUI export endpoint

`GET /api/runs/{run_id}/trials/{trial_id}/export?format=preview|python|zip`

The default `preview` returns code, requirements, environment descriptions, metadata,
setup instructions, and notices. The other formats return downloadable artifacts.
Missing runs/trials return 404; unusable saved configurations return 422. Exporting
does not call providers, download models, index documents, or require saved credentials.
