# Build multilingual RAG applications with confidence

Muffakir is an open-source Python toolkit for building retrieval-augmented generation (RAG)
applications that work seamlessly across languages, with deep specialized support for Arabic and mixed-language content. It brings document
ingestion, chunking, retrieval, generation, evaluation, and architecture search into a
single composable workflow.

## Overview & Demo

<div style="position: relative; width: 100%; padding-bottom: 56.25%; margin: 1.5rem 0; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 14px rgba(0,0,0,0.15);">
  <iframe 
    src="https://www.youtube-nocookie.com/embed/SOXkpL4Q9PE" 
    title="Muffakir Framework Overview & Demo" 
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0;" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
    allowfullscreen>
  </iframe>
</div>

```python
import os

from Muffakir import MuffakirRAG

rag = MuffakirRAG(
    data_dir="./knowledge-base",
    llm_provider="openai",
    llm_model="gpt-4.1-mini",
    api_key=os.environ["OPENAI_API_KEY"],
)

result = rag.ask("ما هي سياسة الإجازات السنوية؟")
print(result["answer"])
```

!!! warning "Keep credentials out of code"

    The example shows the shape of a configuration. Read credentials from your
    environment or secret manager in production; never commit a real API key.

## Why Muffakir?

<div class="grid cards" markdown>

- :material-file-document-outline: **Bring your documents**

  Parse PDFs and office documents, preserve useful metadata, clean Arabic text, and
  choose a chunking strategy that suits your corpus.

- :material-magnify-scan: **Control retrieval**

  Combine vector stores, retrieval methods, query transformation, and rerankers without
  rewriting your pipeline.

- :material-chart-box-outline: **Measure quality**

  Score retrieval and answer quality, inspect traces, detect answer refusals, and use an
  LLM Judge Rating when a semantic 1–5 assessment is useful.

- :material-tune-variant: **Search for a better pipeline**

  Let Composer compare configurations, retain checkpoints, and report the winning setup.

</div>

## Start here

1. [Install Muffakir](getting-started/installation.md).
2. [Build your first RAG application](getting-started/quickstart.md).
3. Use [ComposerUI](getting-started/composer-ui.md) to explore and compare pipelines.
4. Configure an [LLM provider](build/llm-providers.md) in code or
   [ComposerUI](build/llm-providers-ui.md).

## Main capabilities

| Area | What you can do |
| --- | --- |
| Build | Parse, clean, chunk, embed, retrieve, rerank, and generate answers. |
| Evaluate | Measure Recall@k, Precision@k, MRR, nDCG, faithfulness, answer correctness, and LLM Judge Rating. |
| Optimize | Search retrieval, query transformation, reranking, embedding, and chunking choices with Composer. |
| Observe | Inspect per-sample traces, generated queries, execution timings, refusals, and reports. |

## Documentation map

- **Get started** explains installation, a minimal RAG workflow, and ComposerUI.
- **Build** covers the components you assemble into a production pipeline.
- **Evaluate and optimize** explains how to verify and improve quality.
- **Reference** is a compact guide to supported configuration, CLI, and public APIs.
