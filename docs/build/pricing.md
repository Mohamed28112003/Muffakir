# Cost estimation and LLM pricing in the Python library

In enterprise RAG applications, architecture decisions are not made on accuracy alone—operational cost is equally critical. Evaluating different LLMs, query transformers, and judge models can quickly incur substantial API expenditures if not measured precisely.

Muffakir includes a built-in **`Pricing`** module (`PriceMap`) that automatically calculates dollar costs for LLM calls across answer generation, synthetic-data creation, query expansion, and evaluation judges.

For visual cost tracking and interactive budgeting in Composer, see [Cost tracking in ComposerUI](pricing-ui.md).

---

## How pricing works in Muffakir

The pricing engine operates under four core principles:

1. **LiteLLM Community Catalog**: By default, Muffakir pulls per-token input and output rates from LiteLLM's public model catalog (`model_prices_and_context_window.json`), supporting hundreds of models across OpenAI, Anthropic, Google Gemini, Groq, Together, Cohere, DeepSeek, and Azure.
2. **One-Time Snapshot per Experiment**: Community rates are fetched **once** at the start of a `Composer.fit()` run. Pricing lookups are never performed over the network during per-trial or per-sample hot paths.
3. **Immutable Checkpoint Persistence**: The exact fetched price table is saved into `checkpoint.json` as a `pricing_snapshot`. If an architecture search is paused and resumed weeks later, trial costs remain 100% reproducible and comparable, immune to subsequent upstream price adjustments.
4. **Fault-Tolerant & Fail-Safe**: A network failure when fetching LiteLLM rates **never interrupts or crashes a trial**. If the catalog is unreachable, unmapped models simply report `cost = None` while evaluation proceeds normally.

---

## Standalone `PriceMap` usage

You can use `PriceMap` directly to estimate query costs or verify pricing rates:

```python
from Pricing import PriceMap

# 1. Initialize and load community rates
price_map = PriceMap()
price_map.load()

# 2. Look up input and output rates for a model
rates = price_map.get_price("openai", "gpt-4o-mini")
print(rates)
# Output: {'input_cost_per_token': 1.5e-07, 'output_cost_per_token': 6e-07}

# 3. Calculate exact dollar cost for a completed call
cost = price_map.compute_cost(
    provider="openai",
    model="gpt-4o-mini",
    prompt_tokens=2500,
    completion_tokens=450,
)
print(f"Call cost: ${cost:.6f}")
```

---

## Custom pricing overrides (`custom_pricing`)

When using self-hosted models (Ollama, vLLM), private cloud endpoints, discounted enterprise agreements, or newly released models not yet in the community catalog, pass custom rates via `custom_pricing`:

```python
from Muffakir import MuffakirComposer

composer = MuffakirComposer(
    config={
        "data_dir": "./knowledge-base",
        "llm_provider": "together",
        "llm_model": "meta-llama/Llama-3-70b",
        "custom_pricing": {
            # 1. Provider-qualified rate override
            "together/meta-llama/Llama-3-70b": {
                "input_cost_per_token": 0.0000009,
                "output_cost_per_token": 0.0000009,
            },
            # 2. Local model: explicitly zero cost
            "ollama/qwen2.5": {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
            # 3. Model-only rate (applies regardless of provider)
            "custom-fine-tuned-v1": {
                "input_cost_per_token": 0.000002,
                "output_cost_per_token": 0.000005,
            },
        },
    }
)
```

### Rate resolution hierarchy

When resolving the price of an LLM call for `(provider, model)`, `PriceMap` checks keys in strict hierarchical order:

$$\text{1. provider/model (custom)} \longrightarrow \text{2. model (custom)} \longrightarrow \text{3. provider/model (LiteLLM)} \longrightarrow \text{4. model (LiteLLM)} \longrightarrow \text{5. None}$$

This ensures provider-specific rates take precedence while generic model aliases remain supported.

---

## Snapshot serialization and offline reproduction

To serialize or rehydrate a `PriceMap` without performing any network requests:

```python
from Pricing import PriceMap

# Save snapshot to dictionary
snapshot = price_map.to_dict()

# Rehydrate instance with zero network calls
offline_price_map = PriceMap.from_dict(snapshot)
cost = offline_price_map.compute_cost("openai", "gpt-4o-mini", 1000, 200)
```

This serialization contract is what allows Composer parallel workers (`n_jobs > 1`) and checkpoints to maintain shared pricing state across distributed environments.

---

## Cost-aware architecture evaluation

By combining cost tracking with composite evaluation scores, Composer helps identify high-efficiency RAG architectures:

```python
report = composer.fit(...)

for trial in report.trials:
    print(f"Trial #{trial.trial_id}: "
          f"Score = {trial.composite_score:.4f}, "
          f"Cost = ${trial.cost_usd:.6f}")
```

Often, smaller models paired with rerankers achieve comparable Arabic accuracy to larger models at a fraction of the inference expenditure.
