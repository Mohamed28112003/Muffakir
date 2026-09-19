# Query transformation — Python library

Query transformation rewrites or expands a user question **before** vector retrieval so
that the text sent to the embedding model more closely matches the language of the stored
corpus. It is particularly effective for:

- Conversational turn-by-turn questions that lack standalone context.
- Short or ambiguous queries that miss domain-specific vocabulary.
- Complex, multi-part questions that exceed what a single vector lookup can answer.

All strategies are bilingual and integrate directly with `MuffakirPrompt`, so prompt
templates are resolved by language without any extra configuration.

---

## Quick start — via `MuffakirRAG`

The simplest path is to enable transformation in the top-level facade:

```python
from Muffakir import MuffakirRAG

rag = MuffakirRAG(
    data_dir="./knowledge-base",
    query_transformer=True,
    query_transformer_strategy="rewrite",   # any of the five strategies
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    api_key="sk-...",
    embedding_provider="sentence_transformers",
    embedding_model="mohamed2811/Muffakir_Embedding",
    vector_db_provider="chroma",
)

answer = rag.ask("?? ?? ??????? ????????")
```

Set `query_transformer_strategy` to one of `"rewrite"`, `"multi_query"`,
`"decomposition"`, `"hyde"`, or `"step_back"`.

---

## Standalone `QueryTransformer`

Use the `QueryTransformer` orchestrator when you need transformation outside a full
pipeline or when you want to inspect the transformed query:

```python
from QueryTransformer import QueryTransformer
from LLMProvider import LLMProvider
from PromptManager import MuffakirPrompt

llm = LLMProvider(provider="openai", model="gpt-4o-mini", api_key="sk-...")
prompt = MuffakirPrompt(language="ar")

qt = QueryTransformer(
    llm_provider=llm,
    prompt_manager=prompt,
    strategy="multi_query",
)

result = qt.transform_query("??? ???? ??? ?????? ?????????")
print(result)
# ['??? ???? ??? ?????? ?????????',
#  '?? ?? ???? ??????? ?????? ?????????',
#  '????? ??????? ?? ????? ????????',
#  '????? ??????? ????? ????????']
```

The `transform_query` method returns a `str` for single-output strategies (`rewrite`,
`hyde`) and a `List[str]` for multi-output strategies (`multi_query`, `decomposition`,
`step_back`).

---

## Factory function

`create_query_transformer` builds any strategy directly without the orchestrator wrapper:

```python
from QueryTransformer import create_query_transformer
from LLMProvider import LLMProvider

llm = LLMProvider(provider="openai", model="gpt-4o-mini", api_key="sk-...")

transformer = create_query_transformer(
    strategy="hyde",
    llm_provider=llm,
)

hypothetical_doc = transformer.transform("?? ?? ????? ??????? ??? ????")
```

Accepted strategy aliases:

| Canonical name | Accepted aliases |
| --- | --- |
| `rewrite` | `query_rewrite` |
| `multi_query` | `multi_query_expansion`, `query_expansion` |
| `decomposition` | `query_decomposition`, `sub_query` |
| `hyde` | `hypothetical_document` |
| `step_back` | `stepback` |

---

## Strategy reference

### `rewrite` — Query rewriting

**Best for:** Conversational follow-up questions; vague or poorly phrased queries.

Rewrites the query into a clean, keyword-rich, standalone question optimised for
vector retrieval. Uses temperature `0.0` for fully deterministic output.

```python
from QueryTransformer import QueryRewriter
from LLMProvider import LLMProvider

rewriter = QueryRewriter(llm_provider=LLMProvider(...))

clean = rewriter.transform(
    query="??? ???? ????",
    conversation_history=[
        {"role": "user",      "content": "???? ?? ???? ???????"},
        {"role": "assistant", "content": "???? ??????? ?? ..."},
    ],
)
# '?? ??? ??????? ?? ???? ??????? ??? ??? ?????????'
```

**Key parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `prompt_key` | `"query_rewrite"` | Prompt template key in `MuffakirPrompt`. |

**Return type:** `str` — the single rewritten query (or original on failure).

---

### `multi_query` — Multi-query expansion

**Best for:** Short queries where a single phrasing may miss relevant documents.

Expands one query into 3–5 distinct paraphrases using synonyms and varied
phrasing. Uses temperature `0.2` for creative variance. Structured output via
Pydantic (`QueryExpansionOutput`) with a line-based fallback parser.

The original query is **always** the first element of the returned list.

```python
from QueryTransformer import MultiQueryExpansion
from LLMProvider import LLMProvider

expander = MultiQueryExpansion(llm_provider=LLMProvider(...), temperature=0.2)

queries = expander.transform("???? ?????? ??? ??? ?????")
# [
#   '???? ?????? ??? ??? ?????',
#   '??????? ?????? ????? ???????',
#   '?????? ??? ???? ????????',
#   '??? ???? ??? ????? ?????',
# ]
```

**Key parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `prompt_key` | `"multi_query_expansion"` | Prompt template key. |
| `temperature` | `0.2` | Sampling temperature for synonym variance. |

**Return type:** `List[str]`

---

### `decomposition` — Query decomposition

**Best for:** Complex, compound, or multi-step questions.

Splits a complex query into 2–4 focused, independently answerable sub-queries.
Each sub-query is retrieved separately and the results are merged.
Uses temperature `0.0` for deterministic splitting. Structured output via
`SubQueryDecomposition` with a line-based fallback.

Simple queries are preserved as single-item lists without over-decomposing.

```python
from QueryTransformer import QueryDecomposition
from LLMProvider import LLMProvider

decomposer = QueryDecomposition(llm_provider=LLMProvider(...))

sub_queries = decomposer.transform(
    "?? ????? ??? ????? ?????? ?????? ?????? ??? ???? ?? ??????"
)
# [
#   '?? ????? ????? ?????? ????????',
#   '?? ????? ????? ?????? ?????? ???????',
#   '?? ?????? ????????? ????? ???????',
#   '?? ?????? ????????? ????? ???????',
# ]
```

**Key parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `prompt_key` | `"query_decomposition"` | Prompt template key. |

**Return type:** `List[str]`

---

### `hyde` — Hypothetical Document Embeddings

**Best for:** Queries where the question vocabulary differs from answer-passage vocabulary.

Generates a short, plausible hypothetical document that *answers* the query. The
hypothetical document is then embedded and used as the retrieval query text, which
bridges the structural gap between question-style and answer-style embeddings.
Uses temperature `0.3` for natural prose variation.

```python
from QueryTransformer import HyDEQueryTransformer
from LLMProvider import LLMProvider

hyde = HyDEQueryTransformer(llm_provider=LLMProvider(...), temperature=0.3)

hypothetical = hyde.transform("?? ?? ???? ?????? ?????? ?? ????? ?????????")
# '????? ?? ?????? ?????? ?? ???? ???? ?? ?????? ??????? ????? ??? ??? ????? ...'
```

**Key parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `prompt_key` | `"hyde"` | Prompt template key. |
| `temperature` | `0.3` | Sampling temperature for natural prose. |

**Return type:** `str` — the hypothetical document text (or original query on failure).

---

### `step_back` — Step-back prompting

**Best for:** Highly specific or technical queries that need foundational context first.

Generates one broader, high-level "step-back" question derived from the specific
query. The retriever searches for **both** the original specific query and the
step-back question, providing both foundational concepts and specific facts.
Uses temperature `0.0` with Pydantic structured output (`StepBackQueryOutput`).

```python
from QueryTransformer import StepBackQueryTransformer
from LLMProvider import LLMProvider

step_back = StepBackQueryTransformer(llm_provider=LLMProvider(...))

queries = step_back.transform(
    "?? ?? ?????? 203 ?? ????? ??????? ??????? ????????"
)
# [
#   '?? ?? ?????? 203 ?? ????? ??????? ??????? ????????',
#   '?? ?? ????? ??????? ?? ????? ??????? ??????? ????????',
# ]
```

**Key parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `prompt_key` | `"step_back"` | Prompt template key. |

**Return type:** `List[str]` — `[original_query, step_back_query]`.

---

## Conversation history

All strategies accept an optional `conversation_history` list. Each element is a dict
with `"role"` and `"content"` keys (OpenAI-style). When provided, the base class
prefixes the formatted history before the latest query so the LLM can resolve
references and pronouns:

```python
history = [
    {"role": "user",      "content": "???? ?? ??? ???????"},
    {"role": "assistant", "content": "??? ??????? ?? ..."},
]

result = qt.transform_query("??? ???? ????????", conversation_history=history)
```

---

## Custom strategy

Subclass `BaseQueryTransformer` to implement your own strategy and inject it directly
into `QueryTransformer`:

```python
from QueryTransformer.base import BaseQueryTransformer
from QueryTransformer import QueryTransformer

class MyStrategy(BaseQueryTransformer):
    @property
    def name(self) -> str:
        return "my_strategy"

    def transform(self, query: str, conversation_history=None) -> str:
        return f"{query} [domain:legal]"   # example: append a domain tag

qt = QueryTransformer(
    llm_provider=llm,
    strategy=MyStrategy(),
)
```

---

## Choosing a strategy

| Scenario | Recommended strategy |
| --- | --- |
| Chatbot with multi-turn context | `rewrite` |
| Short, single-word or ambiguous query | `multi_query` |
| Complex question with several sub-topics | `decomposition` |
| Query vocabulary differs from document vocabulary | `hyde` |
| Narrow technical query needing background context | `step_back` |
| No transformation (default RAG) | Omit `query_transformer` |

---

## See also

- [Query transformation — ComposerUI](query-transformers-ui.md)
- [Prompt management](prompts-and-language.md) — customize the prompt templates used by each strategy
- [Configuration reference](../reference/configuration.md#query-transformation-and-web-search)
- [Public API](../reference/api.md#querytransformer)
