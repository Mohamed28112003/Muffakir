# Synthetic data

`MuffakirSyntheticData` generates question-and-answer datasets directly from your
document corpus using an LLM. Use synthetic datasets to bootstrap an evaluation set
when human-labeled data is scarce, then **review and curate** examples before relying
on them for release decisions.

> **Important:** Keep a reviewed holdout dataset separate from the data used to tune
> Composer. If the same synthetic set is used for both generation and optimization,
> the search can overfit to its own benchmark.

---

## Quick start

```python
import os
from Muffakir import MuffakirSyntheticData

generator = MuffakirSyntheticData({
    "data_dir": "./knowledge-base",
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini",
    "api_key": os.environ["OPENAI_API_KEY"],
    "language": "ar",
    "output_dir": "./synthetic-evaluation",
})

df, stats = generator.generate_dataset(max_chunks=200)
print(f"Generated {stats.successful} Q&A pairs from {stats.total_chunks} chunks.")
print(df.head())
```

`generate_dataset` returns a **tuple** `(DataFrame, GenerationStats)`. The DataFrame
has one row per generated Q&A pair and the following columns:

| Column | Description |
| --- | --- |
| `question` | The generated question (Arabic or English). |
| `answer` | The grounded answer extracted from the chunk. |
| `context` | The exact source chunk text used for generation. |
| `chunk_id` | Integer ID of the source chunk within the document. |
| `source_file` | Path or filename of the originating document. |
| `generated_at` | ISO-8601 timestamp of when the pair was generated. |

---

## How it works

```
Documents → Parsing → Chunking → [filter] → LLM Q&A → Validation → Export
```

1. **Parse** — reads all files in `data_dir` using the built-in parser or an optional
   `DocumentParser` (Docling, LlamaParse, Azure).
2. **Chunk** — splits text using `MuffakirChunking` (default: recursive, 600-token
   chunks, 200-token overlap).
3. **Filter** — skips chunks shorter than `min_chunk_length` when
   `skip_empty_chunks=True`.
4. **Generate** — calls the LLM for each chunk with the `QA` prompt template.
   Prefers Pydantic structured output (`LLMQAOutput`); falls back to regex parsing
   for providers that do not support function calling.
5. **Validate** — checks minimum question and answer lengths and Pydantic schema
   constraints. Failed chunks are retried up to `max_retries` times.
6. **Checkpoint** — appends validated pairs to a rolling CSV every `save_frequency`
   pairs so progress is preserved on interruption.
7. **Export** — saves the final dataset in all requested `output_format` values
   (CSV, Excel, JSON).

---

## Configuration reference

Pass all options as a plain dictionary to `MuffakirSyntheticData`:

### Core settings

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `data_dir` | `str` | *Required* | Directory containing source documents. |
| `api_key` | `str` | *Required* | LLM provider credential. |
| `llm_provider` | `str` | *Required* | Provider name: `"openai"`, `"together"`, `"groq"`, `"anthropic"`, etc. |
| `llm_model` | `str` | *Required* | Model identifier string. |
| `language` | `str` | `"ar"` | Language for prompt resolution (`"ar"` or `"en"`). |
| `base_url` | `str` | `None` | OpenAI-compatible custom endpoint for self-hosted models. |

### Generation quality

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `llm_temperature` | `float` | `0.3` | Sampling temperature (0.0–1.0). |
| `llm_max_tokens` | `int` | `2000` | Maximum tokens per LLM response. |
| `max_retries` | `int` | `3` | Retry attempts per chunk on validation failure. |
| `validate_qa_pairs` | `bool` | `True` | Apply Pydantic schema validation on every pair. |
| `min_question_length` | `int` | `10` | Discard pairs whose question is shorter than this. |
| `min_answer_length` | `int` | `15` | Discard pairs whose answer is shorter than this. |

### Chunking

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `chunking_method` | `str` | `"recursive"` | Chunking strategy passed to `MuffakirChunking`. |
| `chunk_size` | `int` | `600` | Target chunk size in characters/tokens. |
| `chunk_overlap` | `int` | `200` | Overlap between consecutive chunks. |
| `skip_empty_chunks` | `bool` | `True` | Skip chunks shorter than `min_chunk_length`. |
| `min_chunk_length` | `int` | `50` | Minimum character length for a chunk to be processed. |
| `chunking` | `MuffakirChunking` | `None` | Inject a pre-configured chunking instance directly. |

### Document parsing

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `document_parser` | `str` | `None` | Parser backend: `"docling"`, `"llama_parse"`, `"azure"`, or `None` (built-in). |
| `document_parser_config` | `dict` | `{}` | Provider-specific parser options. |
| `use_ocr` | `bool` | `False` | Enable OCR for scanned documents. |

### Output

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `output_dir` | `str` | `"./muffakir_synthetic_data"` | Directory where datasets and checkpoints are saved. |
| `output_format` | `List[str]` | `["csv", "excel"]` | Export formats: `"csv"`, `"excel"`, `"json"`. |
| `save_frequency` | `int` | `10` | Append to checkpoint CSV every N successful pairs. |

---

## Output files

After a run, `output_dir` will contain:

```
output_dir/
  synthetic_qa_data.csv        ← rolling checkpoint (appended during run)
  synthetic_qa_final.csv       ← final complete dataset (CSV)
  synthetic_qa_final.xlsx      ← final complete dataset (Excel)
  synthetic_qa_final.json      ← final complete dataset (JSON, if requested)
  ocr_results/                 ← OCR output if use_ocr=True
```

The checkpoint CSV is safe to inspect during a long run — it is append-only and
written atomically.

---

## Custom prompt

Override the default Q&A generation prompt by passing a `custom_prompt` string to
`generate_dataset`. It must contain a `{context}` placeholder:

```python
custom = (
    "أنت خبير في إعداد مجموعات بيانات تقييم RAG. "
    "اقرأ المقطع التالي وأنشئ سؤالاً واحداً واضحاً وإجابته الدقيقة بالعربية.\n"
    "المقطع:\n{context}"
)

df, stats = generator.generate_dataset(
    max_chunks=100,
    custom_prompt=custom,
)
```

---

## Use with a custom provider

```python
generator = MuffakirSyntheticData({
    "data_dir": "./knowledge-base",
    "llm_provider": "custom_openai",
    "llm_model": "qwen2.5-7b-instruct",
    "api_key": "not-required",
    "base_url": "http://127.0.0.1:8000/v1",
    "language": "ar",
    "output_dir": "./synthetic-evaluation",
})

df, stats = generator.generate_dataset(max_chunks=50)
```

---

## Inspecting generation statistics

```python
df, stats = generator.generate_dataset(max_chunks=500)

print(f"Processed chunks : {stats.total_chunks}")
print(f"Successful pairs : {stats.successful}")
print(f"Failed chunks    : {stats.failed}")
print(f"Total retries    : {stats.total_retries}")
print(f"Dataset size     : {stats.dataset_size}")
print(f"Source files     : {stats.sources}")
```

`stats.failed` counts chunks that could not produce a valid Q&A pair after all
retries. A high failure rate usually means the `min_question_length` or
`min_answer_length` thresholds are too strict for the model, or the chunks are
too short.

---

## Loading the output as an evaluation dataset

The exported CSV matches the evaluation dataset schema expected by
`MuffakirEvaluation` and `MuffakirComposer`:

```python
import pandas as pd
from Muffakir import MuffakirEvaluation

eval_df = pd.read_csv("./synthetic-evaluation/synthetic_qa_final.csv")

evaluator = MuffakirEvaluation({
    "data_dir": "./knowledge-base",
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini",
    "api_key": "sk-...",
    "embedding_provider": "sentence_transformers",
    "embedding_model": "mohamed2811/Muffakir_Embedding",
    "vector_db_provider": "chroma",
    "metrics": ["faithfulness", "answer_relevance", "context_precision"],
})

results = evaluator.evaluate(eval_df)
print(results)
```

---

## Direct pipeline usage

Access the internal `SyntheticDataPipeline` directly for advanced control:

```python
from SyntheticData import SyntheticDataPipeline
from PromptManager import MuffakirPrompt

prompt = MuffakirPrompt(language="ar")

pipeline = SyntheticDataPipeline(
    config={
        "data_dir": "./knowledge-base",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "sk-...",
        "language": "ar",
        "output_dir": "./synthetic-evaluation",
        "output_format": ["csv", "json"],
        "max_retries": 5,
        "save_frequency": 25,
    },
    prompt_manager=prompt,
)

df, stats = pipeline.run(max_chunks=300)
```

---

## See also

- [Configuration reference](../reference/configuration.md)
- [Public API](../reference/api.md#syntheticdata)
- [Evaluation](../evaluate/evaluation.md)
- [Document parsers](../build/document-parsers.md)
