# Query transformation

Query transformation rewrites or expands a user question before retrieval. It can help
when a conversational or short question does not match the terminology used in a corpus.

## Strategies

| Strategy | What it does |
| --- | --- |
| `rewrite` | Produces one retrieval-focused form of the question. |
| `multi_query` | Produces several query variants to broaden recall. |
| `decomposition` | Breaks a complex question into focused sub-questions. |
| `hyde` | Creates a hypothetical answer-like document for retrieval. |
| `step_back` | Creates a broader conceptual version of the question. |

```python
rag = MuffakirRAG(
    data_dir="./knowledge-base",
    query_transformer=True,
    query_transformer_strategy="rewrite",
    # ... LLM, embeddings, and vector-store configuration
)
```

## Inspect the generated query

Query transformation should be observable, not hidden. Composer traces and the run-detail
sample view retain the generated query for the selected strategy. Use it to answer three
questions:

1. Did the model preserve the user’s intent?
2. Did it add terminology that exists in the corpus?
3. Did it broaden the search so much that retrieval became less precise?

In retrieval-only mode, a query-transform LLM must be configured explicitly because no
main answer-generation LLM is available as a fallback.
