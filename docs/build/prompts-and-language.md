# Prompt management and language in the Python library

In Arabic RAG systems, prompt phrasing directly dictates generation quality, citation discipline, and hallucination rates. Standard English prompt patterns translated literally into Arabic often fail to convey nuance, resulting in awkward phrasing or evasive answers.

Muffakir provides a centralized **`PromptManager`** (`MuffakirPrompt`) with native, professionally crafted Arabic prompt templates (`ar.yaml`), an English counterpart (`en.yaml`), strict placeholder validation, and support for safe runtime overrides.

For the visual prompt editor and dynamic workflow filtering in Composer, see [Prompt management in ComposerUI](prompts-ui.md).

---

## Language selection and bilingual fallback

Muffakir defaults to Arabic (`language="ar"`) for all internal pipeline prompts. English (`language="en"`) is also supported.

```python
from Muffakir import MuffakirRAG

rag = MuffakirRAG(
    data_dir="./knowledge-base",
    language="ar",  # "ar" (default) or "en"
    llm_provider="openai",
    llm_model="gpt-4o-mini",
)
```

### Transparent fallback
When using English or a custom language file, if a specific prompt key is not present in that file, `MuffakirPrompt` automatically falls back to the Arabic version (`ar.yaml`) and logs the substitution. This ensures that experiments never crash due to a missing localized template.

---

## Core prompt catalog

Every prompt template in Muffakir is registered in `PROMPT_SPECS` with strict placeholder contracts:

| Prompt Key | Pipeline Stage | Category | Required Variables | Description |
|---|---|---|---|---|
| **`generation`** | Answer Generation | Generation | `{context}`, `{question}` | Final RAG answer generation with Arabic grounding, citations, and refusal discipline. |
| **`QA`** | Dataset Generation | Dataset | `{context}` | Generates grounded question-answer pairs from raw document chunks. |
| **`query_rewrite`** | Query Transform | Query | `{original_query}` | Rewrites ambiguous or poorly phrased user queries before retrieval. |
| **`hyde`** | Query Transform | Query | `{original_query}` | Generates a hypothetical document answer to capture semantic intent. |
| **`multi_query_expansion`** | Query Transform | Query | `{original_query}` | Expands the query into multiple alternative search phrases. |
| **`query_decomposition`** | Query Transform | Query | `{original_query}` | Breaks down complex multi-part questions into sequential sub-queries. |
| **`step_back`** | Query Transform | Query | `{original_query}` | Generates a higher-level, broader conceptual question. |
| **`reranker_scoring`** | Reranking | Retrieval | `{query}`, `{document}` | Pointwise relevance scoring using an LLM reranker. |
| **`context_relevance`** | Adaptive RAG | Generation | `{question}`, `{context}` | Decides whether local corpus context is sufficient or web fallback is needed. |
| **`hallucination_check_prompt`** | Hallucination Check | Generation | `{answer}` | Cleans and fact-checks generated answers against hallucinations. |
| **`answer_correctness`** | Evaluation Judge | Evaluation | `{question}`, `{gold_answer}`, `{predicted_answer}` | Semantic correctness judge comparing predicted vs reference answers. |
| **`llm_judge_rating`** | Evaluation Judge | Evaluation | `{gold_answer}`, `{predicted_answer}` | Rates semantic accuracy against the gold answer on a 1–5 scale. |
| **`context_grounding`** | Evaluation Judge | Evaluation | `{context}`, `{answer}` | Faithfulness judge checking if answer claims are grounded in retrieved context. |

---

## Overriding prompts safely

You can inspect, modify, and test prompt templates using `MuffakirPrompt`:

```python
from PromptManager import MuffakirPrompt

pm = MuffakirPrompt(language="ar")

# 1. Inspect existing template
current_prompt = pm.get_prompt("generation")
print(current_prompt)

# 2. Update with custom template
custom_template = (
    "أنت خبير توثيق قانوني. أجب باللغة العربية بناءً على النصوص التالية فقط:\n\n"
    "النصوص المرجعية:\n{context}\n\n"
    "السؤال القانوني: {question}\n\n"
    "الإجابة الموثقة:"
)

pm.update_prompt("generation", custom_template)
```

---

## Strict placeholder validation (`validate_prompt`)

To prevent runtime formatting errors (such as `KeyError: 'context'` during an expensive search run), Muffakir enforces strict syntax validation:

1. **Required Variables**: All placeholders in `PROMPT_SPECS[key].required_variables` must be present in the template.
2. **Allowed Variables**: The template cannot introduce arbitrary placeholders not recognized by that pipeline stage.
3. **Malformed Syntax**: Catches empty braces (`{}`), unclosed brackets (`{context`), and invalid format specifiers.

```python
from Muffakir.exceptions import PromptValidationError

try:
    # Fails because {question} is missing!
    pm.validate_prompt("generation", "السياق: {context}")
except PromptValidationError as e:
    print(f"Validation caught error: {e}")
```

---

## Using overrides in `MuffakirRAG` and `MuffakirComposer`

Pass `prompt_overrides` directly in your configuration dictionary. Both `MuffakirRAG` and `MuffakirComposer` validate overrides on initialization:

```python
from Muffakir import MuffakirRAG

rag = MuffakirRAG(
    data_dir="./knowledge-base",
    language="ar",
    prompt_overrides={
        "generation": (
            "أجب بدقة باللغة العربية مستندًا إلى السياق التالي فقط.\n"
            "السياق:\n{context}\n\n"
            "السؤال: {question}\n"
            "الإجابة:"
        ),
        "query_rewrite": (
            "أعد صياغة السؤال التالي ليكون أكثر دقة لمحرك البحث:\n"
            "السؤال الأصلي: {original_query}\n"
            "الصيغة المحسنة:"
        ),
    },
)
```

---

## Dynamic prompt resolution (`resolve_prompt_keys`)

When configuring multi-stage architecture searches in Composer, `resolve_prompt_keys()` determines which prompts are actually utilized by the active workflow:

```python
from PromptManager.registry import resolve_prompt_keys

active_keys = resolve_prompt_keys(
    pipeline_mode="full_rag",
    retrieval_source="vector_db",
    adaptive_web_search=True,
    search_space={"query_expansion": ["none", "hyde"]},
)

print(sorted(active_keys))
# Includes 'generation', 'hyde', 'context_relevance', etc.
```

This prevents developers and ComposerUI users from wasting time customizing prompt templates for stages that are not part of their current search space.

