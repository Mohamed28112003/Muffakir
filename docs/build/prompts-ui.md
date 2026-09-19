# Prompt management in ComposerUI

Prompt engineering in Arabic RAG directly influences groundedness, refusal accuracy, and semantic alignment. ComposerUI features a dedicated, dynamic prompt editor that lets you review, customize, and validate prompt templates per experiment before dispatching architecture searches.

This page explains the visual prompt manager in ComposerUI. For programmatic prompt management, custom YAML directories, and Python SDK recipes, see [Prompt management in the Python library](prompts-and-language.md).

---

## Where to manage prompts in ComposerUI

In the **New Search Run** interface (`http://127.0.0.1:2811/index.html`):

1. Configure your data, search space dimensions, and evaluation metrics in earlier steps.
2. Navigate to **Step 6: Prompt Manager** (`tab-panel-prompts`).
3. ComposerUI dynamically calculates and displays the prompt templates relevant to your active pipeline configuration.

---

## Dynamic workflow relevance

Rather than cluttering the screen with dozens of irrelevant prompts, ComposerUI inspects your active search space and displays **only the prompts that will actually be called**:

| If your configuration includes... | ComposerUI displays prompt editors for... |
|---|---|
| **Full RAG mode** | **Answer Generation** (`generation`) |
| **Retrieval-Only mode** | *Answer Generation prompt is automatically hidden* |
| **HyDE in search space** | **HyDE Prompt** (`hyde`) |
| **Query Rewrite in search space** | **Query Rewrite Prompt** (`query_rewrite`) |
| **LLM Reranker in search space** | **LLM Reranker Scoring** (`reranker_scoring`) |
| **Adaptive Web Search** | **Adaptive Context Relevance** (`context_relevance`) |
| **Answer Correctness Metric** | **Answer Correctness Judge** (`answer_correctness`) |
| **LLM Judge Rating Metric** | **LLM Judge Rating** (`llm_judge_rating`) |

---

## The interactive prompt editor

Each relevant prompt card features:

1. **Stage & Purpose Badge**: Displays the pipeline component and a concise explanation of how the prompt affects inference.
2. **Required Variable Badges**: Interactive chips listing the placeholders required by the stage (e.g., `{context}`, `{question}`, `{original_query}`).
3. **Monospace Code Editor**: A formatted text area containing the active localized prompt template.
4. **Reset to Default Button**: Reverts the specific prompt back to the canonical Arabic or English default from `ar.yaml` / `en.yaml`.

---

## Real-time validation badges

ComposerUI validates prompt templates against `PROMPT_SPECS` as you type:

- **Missing Placeholders**: If you delete a required variable like `{question}`, the card displays an alert warning: `Missing required variable: {question}`.
- **Malformed Braces**: Detects unmatched brackets or syntax errors that would trigger runtime formatting exceptions.
- **Launch Protection**: The launch wizard prevents dispatch if any active prompt template has validation errors, protecting you from wasting compute time on failed runs.

---

## Experiment reproducibility

When you customize prompts in ComposerUI:

1. Customized templates are serialized into the run manifest (`config.json`) under `prompt_overrides`.
2. Every trial evaluated in the run uses these exact templates.
3. If the run is resumed or cloned, the exact custom prompts are restored, ensuring complete scientific reproducibility.
