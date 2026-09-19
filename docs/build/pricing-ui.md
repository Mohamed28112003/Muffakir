# Cost tracking in ComposerUI

ComposerUI provides real-time financial observability for RAG architecture searches. As trials evaluate different prompt strategies, LLM providers, and query expanders, ComposerUI tracks exact token expenditure and cumulative dollar costs.

This page explains how cost tracking works in ComposerUI. For programmatic pricing lookups and custom rate overrides in Python, see [Cost estimation and LLM pricing in the Python library](pricing.md).

---

## Where cost appears in ComposerUI

Cost metrics are surfaced across three primary views:

1. **Run Detail KPI Cards**: The top dashboard on `run_detail.html` features a dedicated **Total LLM Cost ($)** card showing total experiment spend.
2. **Live Trials Telemetry Stream**: The trials table displays the exact dollar cost incurred by each individual trial architecture.
3. **Cost Accumulation Chart**: A locally rendered SVG line chart plotting cumulative spending across sequential trials. Hover over points for exact values; no external chart script is required.

---

## Custom pricing in the New Run wizard

When using custom endpoints, private model deployments, or discounted enterprise contracts, you can provide custom token rates during run configuration:

### Entering rates per 1,000,000 tokens

Commercial providers quote pricing per million tokens (e.g., $0.15 per 1M input tokens). ComposerUI lets you input rates in this standard format:

| Field | Description | Example |
|---|---|---|
| **Model Identifier** | Exact provider-qualified or model name | `together/meta-llama/Llama-3-70b` |
| **Prompt Cost ($ / 1M tokens)** | Cost per million input/prompt tokens | `$0.90` |
| **Completion Cost ($ / 1M tokens)** | Cost per million output/completion tokens | `$0.90` |

ComposerUI automatically converts these values into per-token floating point numbers (`input_cost_per_token = rate / 1,000,000`) before submitting the run manifest to the backend.

### Setting local models to free

For local models hosted on Ollama, vLLM, or local GPUs, enter `$0.00` per 1M tokens. This ensures token counts are recorded in traces while dollar expenditure reflects zero cloud cost.

---

## Analyzing costs on the Run Detail page

On the Run Detail page (`run_detail.html`), financial metrics update live as trials complete:

### 1. Total LLM Cost KPI card
Displays the total dollar expenditure formatted to six decimal places (e.g., `$0.048210`). The badge indicates the active pricing status (`LiteLLM Live Catalog` or `Custom Pricing Active`).

### 2. Cost accumulation chart
The **Cost Accumulation Over Trials** chart plots running cumulative cost on the Y-axis against evaluated trials on the X-axis:

- **Flat slopes**: Represent lightweight retrieval-only trials or highly efficient models.
- **Steep slopes**: Highlight expensive architectures, such as multi-step query decomposition or high-sample LLM judge evaluations.

### 3. Per-trial cost breakdown
In the trials telemetry table:
- Look for trials with the highest **Cost-to-Score Efficiency** (maximum composite score at minimum dollar cost).
- Identify trials where costly query expanders or large models did not produce a proportional gain in retrieval accuracy.
