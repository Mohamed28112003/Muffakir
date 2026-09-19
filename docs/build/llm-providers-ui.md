# LLM providers in ComposerUI

ComposerUI lets you configure the LLMs used by a Composer experiment without placing a
credential in a Python file. A run can use one LLM for final answers and optionally use
different LLMs for synthetic-data generation, query transformation, reranking, and
evaluation judges.

This page explains ComposerUI. For code-first configuration and provider-specific SDK
arguments, see [LLM providers in the Python library](llm-providers.md).

## Before you start

Install ComposerUI and the provider integration you plan to use. ComposerUI can show
which integrations are installed, but it does not install provider packages for you.

```bash
pip install "Muffakir[ui]"
pip install "Muffakir[openai]"  # OpenAI, Together, OpenRouter, Azure, or custom endpoints
muffakir serve
```

Open `http://127.0.0.1:2811`, create a run, and move through the guided workflow. The
provider picker is populated from the local provider catalog. Integrations that are not
installed appear unavailable and identify the missing package; install the displayed
`Muffakir[...]` extra, restart the server, and refresh the page.

## Configure the answer-generation LLM

In **Models & Providers**, the **Answer Generation LLM** card configures the model that
turns retrieved context into the final answer. It is required for a **Full RAG** run and
for web-search-only answering.

1. Select a value in **Answer LLM Provider**.
2. Enter the provider’s exact model ID in **Answer LLM Model**.
3. Paste the credential into **API key** when the provider needs one.
4. Set **Base URL** only for an endpoint override, compatible gateway, or local server.
5. Continue to the review page and run the dry-run validation before launch.

The model field is free text. ComposerUI does not restrict you to a stale built-in model
list; it passes the model identifier to the selected provider.

## Advanced LLM settings

Expand **Advanced LLM settings** on the answer, judge, query-transform, LLM-reranker,
or dataset-generation card. A role's controls appear only when that role is active.
You can use different parameters without selecting a different provider or model.

New UI runs start with temperature `0` and an output limit (`max_tokens`) of `4096` for each role.
Auxiliary-role settings are saved independently and remain fixed when generation
variants change. Reusing a model does not require reusing its sampling settings.

### Shared configuration and provider support

ComposerUI and the Python library share a unified, validated LLM-parameters object across all LLM roles (`llm_parameters`, `judge_llm_parameters`, `query_transform_llm_parameters`, `reranker_llm_parameters`, `dataset_llm_parameters`).

The shared parameter object contains:

- `temperature`: Sampling temperature controlling response variability (choose **Omit temperature from request** for models that do not accept it; zero is a valid value).
- `max_tokens`: Labeled **Maximum output tokens** in the UI. Sets an upper output token limit, not a target answer length.
- `top_p`: Nucleus sampling probability threshold (range: `0.0` to `1.0`).
- `stop`: List of stop sequences passed as a list of strings (`list[str]`, entered one sequence per line in the UI).
- `seed`: Integer random seed for best-effort reproducible sampling.
- `top_k`: Top-k token filtering limit (integer $\ge 1$).
- `frequency_penalty` and `presence_penalty`: Frequency and presence penalties adjusting token repetition likelihood (range: `-2.0` to `2.0`).
- `timeout_seconds`: Individual provider request timeout limit in seconds.
- `max_retries`: Explicitly labeled **Provider request retries** in the UI. Configures individual provider request retry attempts (does not affect Composer trial retries).

#### Provider capability catalog

Controls come from the provider capability catalog. Not every provider exposes every control, and individual models or custom endpoints may reject otherwise supported options:

| Parameter | UI Label | Allowed Format / Range | Supported Providers |
| --- | --- | --- | --- |
| `temperature` | Temperature | `0.0` – `2.0` (or `None`) | `openai`, `azure_openai`, `custom`, `together`, `openrouter`, `groq`, `anthropic`, `gemini`, `ollama` |
| `max_tokens` | **Maximum output tokens** | Integer $\ge 1$ | All providers |
| `top_p` | Top-p | `0.0` – `1.0` | All providers |
| `stop` | Stop sequences | `list[str]` (one per line) | All providers |
| `seed` | Seed | Integer | `openai`, `azure_openai`, `custom`, `together`, `openrouter`, `ollama` |
| `top_k` | Top-k | Integer $\ge 1$ | `anthropic`, `gemini`, `ollama` |
| `frequency_penalty` | Frequency penalty | `-2.0` – `2.0` | `openai`, `azure_openai`, `custom`, `together`, `openrouter` |
| `presence_penalty` | Presence penalty | `-2.0` – `2.0` | `openai`, `azure_openai`, `custom`, `together`, `openrouter` |
| `timeout_seconds` | Request timeout (seconds) | $> 0$ | All providers |
| `max_retries` | **Provider request retries** | Integer $\ge 0$ | `openai`, `azure_openai`, `custom`, `together`, `openrouter`, `groq`, `anthropic`, `gemini` |

When changing providers, clear any incompatible settings shown in red. **Reset settings** restores defaults.

## Compare generation variants

1. Configure the answer model and expand **Advanced LLM settings**.
2. Select **Add model variant**, review the model and parameters, then save it.
3. Select **Duplicate**, change a value (for example temperature `0.3`), and save.
4. Add another variant at `0.7`, or use **Edit** and **Remove** as needed.
5. Review the variants, expanded trial count, maximum trials, and pricing before launch.

With no variants, Composer uses the answer configuration once. With variants, it uses
only the listed choices. Three variants and two retrieval depths produce six trials;
individual parameter fields are not automatically multiplied. Exact duplicate variants
are rejected, but the same model with different settings is valid.

Credentials and the base endpoint are shared with the answer configuration; the
variant editor does not store separate credentials. Use compatible provider accounts
and endpoints. Evaluation judges and dataset generation continue using their configured
base/override models, while existing runtime-stage model-reuse behavior is preserved.

Web-search-only runs can compare generation variants without local index stages.
Retrieval-only runs do not expose generation variants. Changes to generation parameters
reuse compatible document indexes rather than rebuilding them.

The review screen, trial inspector, checkpoints, traces, and reports retain settings.
[Python exports](../evaluate/python-export.md) retain the selected runtime parameters.

### Provider choices in the UI

| UI choice | Credential | Base URL | How ComposerUI uses it |
| --- | --- | --- | --- |
| `openai` | Required | Optional | Default OpenAI endpoint or a compatible proxy. |
| `groq` | Required | Not normally needed | Groq’s native integration. |
| `together` | Required | Not normally needed | Together’s managed OpenAI-compatible endpoint. |
| `openrouter` | Required | Not normally needed | OpenRouter’s managed OpenAI-compatible endpoint. |
| `anthropic` | Required | Not normally needed | Anthropic’s native integration. |
| `gemini` | Required | Not normally needed | Google Gemini’s native integration. |
| `ollama` | Optional | Optional | Local Ollama; default server is `http://localhost:11434`. |
| `azure_openai` | Required | Do not use the generic field | Requires Azure endpoint/deployment settings; see the limitation below. |
| `custom` | Optional | Usually required | Any OpenAI-compatible endpoint. |
| `deepseek` | Usually required | Required | Convenience alias that resolves through the custom compatible provider. |

!!! note "Provider availability is local"

    The catalog checks whether the needed Python integration is installed on the computer
    running ComposerUI. It does not verify that an API key is valid or that your account
    can access a chosen model. The dry run validates the configuration before a run starts.

## Custom providers and local servers

Choose **custom** when your provider exposes an OpenAI-compatible chat-completions API.
This covers common self-hosted and gateway setups without adding a separate provider for
each service.

| Scenario | Provider | Base URL example | API key |
| --- | --- | --- | --- |
| Self-hosted vLLM | `custom` | `http://127.0.0.1:8000/v1` | Usually not needed |
| LM Studio compatible server | `custom` | Your server’s `/v1` URL | Usually not needed |
| Compatible hosted gateway | `custom` | `https://gateway.example/v1` | The gateway credential |
| DeepSeek compatible API | `deepseek` or `custom` | Provider-compatible `/v1` URL | Your provider credential |

Enter the URL including `http://` or `https://`. For hosted custom endpoints, provide the
key supplied by that service. For a local endpoint that accepts no key, leave the key
empty. ComposerUI preflight accepts a custom base URL in place of an API key for local or
compatible endpoints.

!!! tip "Use the provider’s documented OpenAI-compatible URL"

    The correct path is often `/v1`, but it is provider-specific. Copy the endpoint from
    that provider’s API documentation. An ordinary website URL is not necessarily an API
    endpoint.

### Azure OpenAI limitation

Azure OpenAI needs an Azure resource endpoint and deployment name. The current
ComposerUI answer-provider card has a generic **Base URL** field, not Azure-specific
`azure_endpoint` and deployment controls; that generic URL is not used by the Azure
provider. Configure Azure OpenAI through the [Python library guide](llm-providers.md#azure-openai)
until ComposerUI adds those Azure-specific fields.

## One run, several LLM roles

ComposerUI avoids making you configure the same account repeatedly. The answer LLM is the
default for all supported LLM roles. Turn on an override only when a role needs a different
provider, model, key, or compatible endpoint.

| UI area | When it appears | Default behavior | Override fields |
| --- | --- | --- | --- |
| **Answer Generation LLM** | Full RAG or answering with web search | Generates final answers | Provider, model, API key, base URL |
| **Dataset Generation LLM** | Automatic evaluation-data generation | Reuses answer LLM | Uncheck reuse, then choose provider, model, key, and base URL |
| **Query-Transform LLM** | A query transformation is selected | Reuses answer LLM in Full RAG | Enable its override and configure provider, model, key, and base URL |
| **LLM Reranker** | `llm` reranking is selected | Reuses answer LLM in Full RAG | Enable dedicated reranker LLM and configure its provider, model, key, and base URL |
| **Judge LLM** | A generation metric is selected | Reuses answer LLM | Enable judge override and configure provider, model, key, and base URL |

Examples of reasonable separation:

- Use a fast local Ollama model for final answers and a hosted model only for
  `LLM Judge Rating` or answer-correctness evaluation.
- Use a small, economical model for synthetic Q&A generation and a stronger answer model
  for the actual trial.
- Use a dedicated query-transform model for multilingual rewriting while leaving answer
  generation unchanged.

Each enabled role can make additional LLM calls. It may increase cost, latency, and the
number of credentials that need to be valid.

### Retrieval-only runs

**Retrieval Only** intentionally has no answer-generation LLM. It can still need an LLM
if you choose automatic dataset generation, query transformation, or LLM reranking. In
that case ComposerUI exposes the relevant dedicated fields and validates that a provider
and model have been supplied. Generation metrics are unavailable in retrieval-only mode.

## Credentials, saved runs, and the review screen

Credentials entered into the form are used to launch the current local run. ComposerUI
redacts credentials from the persisted run configuration, traces, reports, and review
summary. The review screen displays **Configured (hidden)** rather than the actual value.

The review page also lists each LLM role and base URL. Check it before launch, especially
when an override is active: it makes accidental reuse of the wrong provider much easier to
spot.

Treat the local run directory as experiment data, not a secret manager. Prefer temporary
environment variables or a managed secret store for long-lived credentials, and never put
keys in screenshots, exported configuration, or version-controlled files.

## Pricing and usage

ComposerUI shows the models involved in an experiment on the review page, including
answer, query-transform, reranker, and judge roles. Configure pricing for models that are
not recognized by the catalog using input and output cost per million tokens. Token and
cost values are best-effort because a provider or local model may omit usage metadata.

The resulting run details show resolved configurations, timings, cost information, and
per-sample traces. A trace reveals which generated query was used and whether the answer
was a refusal; it does not expose API keys.

## Validate before launching

The **Review & Run** step has a dry-run gate. Run it whenever provider settings, a model,
or a search dimension changes. Preflight checks the selected providers, installed optional
dependencies, dataset and metric requirements, and valid search-space combinations.

Common outcomes:

| Message | Meaning | Fix |
| --- | --- | --- |
| Provider is unavailable | The matching library integration is not installed. | Install the indicated `Muffakir[...]` extra and restart ComposerUI. |
| API key or custom base URL required | A hosted/compatible provider has neither credential nor endpoint. | Add its key or valid custom endpoint. |
| Enter the Answer LLM model | The provider is selected but model ID is blank. | Enter the exact model identifier. |
| Query-transform / reranker model required | An optional LLM role is enabled without a complete override. | Supply its model and provider settings, or disable the override. |
| Azure endpoint error | Azure-specific settings are not available in the generic UI card. | Configure that run through the Python library. |

See [Composer search](../evaluate/composer.md) for selecting dimensions and
[evaluation](../evaluate/evaluation.md) for judge-backed metrics such as **LLM Judge
Rating**.
