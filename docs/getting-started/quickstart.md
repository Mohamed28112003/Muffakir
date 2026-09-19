# Build your first RAG application

This example indexes a local document directory and answers an Arabic question. It uses
Chroma locally, OpenAI for generation, and a local embedding model.

## Prepare credentials

=== "PowerShell"

    ```powershell
    $env:OPENAI_API_KEY = "your-key"
    ```

=== "bash"

    ```bash
    export OPENAI_API_KEY="your-key"
    ```

## Create a knowledge base

Put PDFs, DOCX files, text files, or supported images under `./knowledge-base`, then
create `app.py`:

```python
import os

from Muffakir import MuffakirRAG

rag = MuffakirRAG(
    data_dir="./knowledge-base",
    llm_provider="openai",
    llm_model="gpt-4.1-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    vector_db_provider="chroma",
    db_path="./muffakir_db",
    language="ar",
)

result = rag.ask("ما هي الخطوات المطلوبة لتقديم طلب الإجازة؟")
print(result["answer"])
```

Run it with `python app.py`.

## What happens in the pipeline?

1. Muffakir parses and cleans the source documents.
2. It splits the content into chunks and stores embeddings in the configured vector store.
3. It retrieves relevant chunks for the question.
4. The configured LLM receives the retrieved context and returns an answer.

Use `rag.get_similar_documents(query, k=5)` when you only need the retrieved documents.
Use `rag.get_similar_documents_with_trace(...)` when you want retrieval details for
debugging.

## Next steps

- Configure [retrieval and vector stores](../build/retrieval.md).
- Improve retrieval with [query transformation](../build/query-transformers.md) and
  [reranking](../build/reranking.md).
- Measure quality with [evaluation](../evaluate/evaluation.md).
