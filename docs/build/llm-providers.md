# LLM providers in the Python library

Muffakir uses one provider interface for answer generation, synthetic-data generation,
evaluation judges, query transformation, and LLM reranking. You choose a provider name,
model ID, credential, and—when needed—an endpoint URL. Muffakir constructs the matching
LangChain chat model lazily, so installing one provider integration does not require every
provider SDK.

This guide is about the Python library. For the visual setup, secret handling, and
provider roles in Composer, see [LLM providers in ComposerUI](llm-providers-ui.md).

## Start with a RAG application

Set credentials outside source code. This example uses OpenAI; only the provider, model,
and environment variable need to change for another hosted provider.

=== "PowerShell"

    ```powershell
    $env:OPENAI_API_KEY = "your-key"
    ```

=== "bash"

    ```bash
    export OPENAI_API_KEY="your-key"
    ```

```python
import os

from Muffakir import MuffakirRAG

rag = MuffakirRAG(
    data_dir="./knowledge-base",
    llm_provider="openai",
    llm_model="gpt-4.1-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    llm_temperature=0.2,
    llm_max_tokens=800,
    language="ar",
)

result = rag.ask("ما هي شروط التقديم؟")
print(result["answer"])
```

`llm_provider`, `llm_model`, `api_key`, `llm_temperature`, `llm_max_tokens`, and
`base_url` (or `llm_base_url`) are the RAG configuration keys. The RAG pipeline passes
them to the shared LLM provider facade.

!!! tip "Model IDs are provider-owned"

    Muffakir does not maintain a restrictive list of models. Enter a model ID supported
    by the provider and available to your account or local server. This keeps the library
    usable when providers add models.

## Supported providers

| Provider value | Install | API key | Endpoint behavior | Notes |
| --- | --- | --- | --- | --- |
| `openai` | `Muffakir[openai]` | Required | Provider default, or `base_url` | OpenAI through `ChatOpenAI`. |
| `groq` | `Muffakir[groq]` | Required | Groq managed endpoint | Uses `ChatGroq`. |
| `together` | `Muffakir[openai]` | Required | `https://api.together.xyz/v1` | OpenAI-compatible Together API. |
| `openrouter` | `Muffakir[openai]` | Required | `https://openrouter.ai/api/v1` | OpenAI-compatible OpenRouter API. |
| `anthropic` | `Muffakir[anthropic]` | Required | Anthropic managed endpoint | Uses `ChatAnthropic`. |
| `gemini` | `Muffakir[gemini]` | Required | Gemini managed endpoint | Uses `ChatGoogleGenerativeAI`. |
| `ollama` | `Muffakir[ollama]` | Optional | `http://localhost:11434` by default | Local Ollama server. |
| `azure_openai` | `Muffakir[openai]` | Required | Requires `azure_endpoint` | Uses an Azure deployment, not an ordinary base URL. |
| `custom` / `custom_openai` | `Muffakir[openai]` | Optional | Your OpenAI-compatible `base_url` | For gateways, hosted compatible APIs, and local servers. |

Install the base package plus the matching optional integration. For example:

```bash
pip install "Muffakir[openai]"
pip install "Muffakir[groq]"
pip install "Muffakir[ollama]"
```

The provider constructor validates the selected provider and key requirements before it
makes a request. Missing optional integrations result in an error that names the needed
`Muffakir[...]` extra.

### Accepted aliases

The library accepts several familiar aliases and resolves them to a canonical provider:

| Input | Resolved provider |
| --- | --- |
| `open_router` | `openrouter` |
| `google` | `gemini` |
| `azure` | `azure_openai` |
| `openai_compatible` | `custom_openai` |
| `deepseek` | `custom` |
| `vllm` | `custom` |

Aliases help existing applications migrate, but new applications should prefer the
canonical names in the first table.

## Use the provider facade directly

Use `LLMProvider` when you need a chat model without building a complete RAG pipeline.
It returns the underlying LangChain model through `get_llm()` and provides a convenient
`call()` method.

```python
import os

from LLMProvider import LLMProvider

provider = LLMProvider(
    provider="groq",
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.1,
    max_tokens=400,
)

message = provider.call("Give one short Arabic greeting.")
print(message.content)
```

The equivalent factory is `create_llm_provider(...)`. Both accept these common arguments:

| Argument | Meaning |
| --- | --- |
| `provider` | A supported provider name or accepted alias. Default: `openai`. |
| `model` | The provider’s model ID. Default: `gpt-4o-mini`. |
| `api_key` | Credential for hosted providers. Not required for Ollama and custom endpoints. |
| `temperature` | Sampling temperature. Default: `0.5`. |
| `max_tokens` | Maximum generated tokens. Default: `300`. Gemini receives this as `max_output_tokens`; Ollama as `num_predict`. |
| `base_url` | Optional endpoint override for OpenAI or custom providers; Ollama’s server URL. |
| `**kwargs` | Provider-specific options described below. |

`call()` uses the LangChain `invoke()` interface when it is available, and falls back to a
callable model for older clients. If you need LangChain composition, use
`provider.get_llm()` directly.

## Shared configuration and provider support

Muffakir uses a shared, validated LLM-parameters object across `LLMProvider` and ComposerUI (`llm_parameters`, `judge_llm_parameters`, `query_transform_llm_parameters`, `reranker_llm_parameters`, `dataset_llm_parameters`).

The shared parameters object validates and standardizes parameters across providers:

- `temperature`: Sampling temperature controlling response variability (nullable for models that omit temperature).
- `max_tokens`: Labeled **Maximum output tokens** in ComposerUI. Upper output limit on generated tokens.
- `top_p`: Nucleus sampling probability threshold (`0.0` to `1.0`).
- `stop`: List of stop sequences specified as a list of strings (`list[str]`).
- `seed`: Integer random seed for reproducible sampling.
- `top_k`: Top-k token filtering limit (integer $\ge 1$).
- `frequency_penalty` and `presence_penalty`: Frequency and presence penalties adjusting token repetition (`-2.0` to `2.0`).
- `timeout_seconds`: Individual request timeout in seconds.
- `max_retries`: Explicitly labeled **Provider request retries** in ComposerUI. Provider API request retry count.

```python
from LLMProvider.parameters import validate_parameters, provider_kwargs

# Validate parameters for a specific provider
params = {
    "temperature": 0.2,
    "max_tokens": 1024,
    "top_p": 0.95,
    "stop": ["\n\n", "User:"],
    "timeout_seconds": 30.0,
    "max_retries": 3,
}
validated = validate_parameters(params, provider="openai")
```

For the visual interface and provider capability catalog in ComposerUI, see [LLM providers in ComposerUI](llm-providers-ui.md).

## Provider recipes

### OpenAI

```python
LLMProvider(
    provider="openai",
    model="gpt-4.1-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.2,
    max_tokens=800,
)
```

For a compatible OpenAI proxy, `base_url`, `openai_api_base`, and `api_base` are accepted
endpoint names. Prefer `base_url` in new code. Extra non-conflicting keyword arguments
are forwarded to `ChatOpenAI`.

### Groq

```python
LLMProvider(
    provider="groq",
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.1,
)
```

Groq uses its native LangChain integration. Use a model identifier currently offered by
your Groq account.

### Together and OpenRouter

Both use OpenAI-compatible APIs, but Muffakir supplies their managed endpoint for you.
Their dedicated integrations use that endpoint; choose `custom` if you need a different
OpenAI-compatible gateway.

```python
together = LLMProvider(
    provider="together",
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    api_key=os.environ["TOGETHER_API_KEY"],
)

router = LLMProvider(
    provider="openrouter",
    model="your-provider-model-id",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

### Anthropic and Gemini

```python
anthropic = LLMProvider(
    provider="anthropic",
    model="claude-sonnet-4-5",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=800,
)

gemini = LLMProvider(
    provider="gemini",
    model="gemini-2.5-flash",
    api_key=os.environ["GOOGLE_API_KEY"],
    max_tokens=800,
)
```

The Gemini integration calls the provider’s output limit `max_output_tokens` using the
portable Muffakir name `max_tokens`.

### Ollama (local)

Start Ollama and pull the model separately, then point Muffakir at the local service.

```bash
ollama pull llama3.2
ollama serve
```

```python
local = LLMProvider(
    provider="ollama",
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.2,
    max_tokens=600,
)
```

No API key is required. If `base_url` is omitted, Muffakir uses
`http://localhost:11434`. `ollama_base_url` is also supported as a compatibility keyword.

### Azure OpenAI

Azure uses an Azure resource endpoint and deployment name. It is not configured through
the generic `base_url` field.

```python
azure = LLMProvider(
    provider="azure_openai",
    model="my-chat-deployment",  # used as deployment when no explicit value is supplied
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="my-chat-deployment",
    api_version="2024-02-15-preview",
    max_tokens=800,
)
```

Required values are `api_key`, `azure_endpoint` (or `endpoint`), and a deployment. The
deployment can be supplied as `azure_deployment`, `deployment_name`, or `model`. If no
`api_version` is provided, Muffakir uses `2024-02-15-preview`.

### Custom OpenAI-compatible endpoints

Use `custom` for an endpoint that implements the OpenAI chat-completions shape. It is the
right route for a self-hosted vLLM or LM Studio server, a gateway, and compatible services
such as DeepSeek, Fireworks, Anyscale, or SambaNova.

```python
compatible = LLMProvider(
    provider="custom",
    model="your-model-id",
    base_url="https://your-endpoint.example/v1",
    api_key=os.environ.get("COMPATIBLE_API_KEY"),
    temperature=0.2,
    max_tokens=800,
)
```

For a local server that has no credential, omit `api_key`:

```python
vllm = LLMProvider(
    provider="vllm",  # alias for custom
    model="your-local-model",
    base_url="http://127.0.0.1:8000/v1",
)
```

Muffakir supplies a harmless placeholder key to the underlying OpenAI-compatible client
when no key is provided. A `base_url` is strongly recommended for every `custom` setup;
without it, the underlying OpenAI client has no custom target to call. The custom adapter
also normalizes a non-standard `role: null` response to `assistant`, while preserving
structured-output metadata used by query transformation and reranking.

!!! warning "DeepSeek is an alias, not a separate SDK"

    `provider="deepseek"` resolves to `custom`. Supply DeepSeek’s compatible endpoint
    URL and the model ID supported by your account. This avoids a second provider-specific
    integration and also works for other compatible gateways.

## Portable generation parameters

Use `llm_parameters` in `MuffakirRAG`, `MuffakirSearch`, or Composer configuration.
For the low-level `LLMProvider`, pass the same dictionary as `parameters`:

```python
import os
from LLMProvider import LLMProvider

provider = LLMProvider(
    provider="openai", model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    parameters={"temperature": 0.2, "max_tokens": 800, "top_p": 0.9,
                "timeout_seconds": 60, "max_retries": 0},
)
```

| Parameter | Meaning and validation |
| --- | --- |
| `temperature` | Sampling temperature, 0–2 (0–1 for Anthropic). `None` omits the constructor argument. |
| `max_tokens` | Positive integer maximum output-token count. |
| `top_p` | Nucleus sampling, 0–1. |
| `stop` | List of non-empty stop strings; `[]` supplies no stop sequences. |
| `seed` | Integer sampling seed; no determinism guarantee. |
| `top_k` | Positive integer sampling candidate count, where supported. |
| `frequency_penalty`, `presence_penalty` | Values between -2 and 2, where supported. |
| `timeout_seconds` | Positive request timeout, in seconds. |
| `max_retries` | Nonnegative integer provider-request retry count; independent of trial and dataset retries. |

All adapters support the first four parameters and request timeout. The portable
adapter currently exposes seed and penalties for OpenAI-compatible adapters, seed and
top-k for Ollama, and top-k for Gemini and Anthropic. Provider request retries are
exposed for all adapters except Ollama. `GET /api/catalog` returns the exact
`llm_parameter_capabilities` used by ComposerUI. These describe adapter support;
the selected model or gateway may impose narrower limits.

Explicit dictionary values take precedence over existing `llm_temperature` and
`llm_max_tokens` fields (or low-level `temperature` and `max_tokens` arguments).
Omitted fields retain existing library defaults; new optional fields are not sent.
Only temperature accepts `None`; omit other optional keys instead of supplying null.
Credentials, endpoints, model IDs, and arbitrary provider kwargs do not belong in this
dictionary. Unsupported keys and invalid values raise configuration errors.

## Reuse or override an LLM by pipeline role

The primary RAG LLM generates final answers. Other features normally reuse that model and
credential, but can be intentionally separated:

| Feature | Default | Override configuration |
| --- | --- | --- |
| Query transformation | Answer LLM | `query_transform_llm_provider`, `query_transform_llm_model`, `query_transform_api_key`, `query_transform_base_url` |
| LLM reranking | Answer LLM | `reranker_llm_provider`, `reranker_llm_model`, `reranker_llm_api_key`, `reranker_llm_base_url` |
| Dataset generation | Answer LLM in Composer | `dataset_llm_provider`, `dataset_llm_model`, `dataset_api_key`, `dataset_base_url` |
| Generation evaluation judges | Answer LLM in Composer | `judge_llm_provider`, `judge_llm_model`, `judge_api_key`, `judge_base_url` |

This is useful when you want a fast, inexpensive answer model but a stronger judge, or a
local answer model with a hosted model used only for a specialized task. Each override is
an independent provider call and can have its own cost and rate limits.

Set `query_transform_llm_parameters`, `reranker_llm_parameters`,
`judge_llm_parameters`, or `dataset_llm_parameters` independently. A parameter-only
override creates a separate runtime client when needed without changing model identity
or mutating the answer client. For example:

```python
config = {
    "llm_parameters": {"temperature": 0.7, "max_tokens": 1200},
    "query_transform_llm_parameters": {"temperature": 0, "max_tokens": 300},
    "reranker_llm_parameters": {"temperature": 0, "max_tokens": 500},
    "judge_llm_parameters": {"temperature": 0, "max_tokens": 1500},
    "dataset_llm_parameters": {"temperature": 0.3, "max_tokens": 2000},
}
```

Role dictionaries replace inheritance from the generation dictionary; unspecified
temperature/token limits fall back to legacy fields and that workflow's defaults.
SDK configurations without role dictionaries retain legacy reuse/inheritance.
For standalone `MuffakirSyntheticData`, use `llm_parameters`; its existing defaults
remain temperature `0.3` and `2000` output tokens. Composer dataset generation instead
inherits Composer's legacy fields unless explicitly overridden.

## Token usage and cost observability

Every `LLMProvider` instance attaches a best-effort usage callback to the generated chat
model. Read the totals after one or more calls:

```python
usage = provider.get_usage_totals()
print(usage)  # prompt_tokens, completion_tokens, total_tokens

# For Composer-style per-sample accounting:
sample_usage = provider.get_sample_usage_totals(trial_id=3, sample_index=0)
```

Available methods are `get_usage_totals()`, `get_per_sample_usage_totals()`,
`get_sample_usage_totals(trial_id, sample_index)`, and `reset_usage()`. Tracking is
best-effort: providers or local models that do not return token metadata can legitimately
report zero or incomplete values.

## Errors and troubleshooting

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `An API key must be provided` | A hosted provider was initialized without a key. | Set the provider credential in the environment and pass it as `api_key`. |
| Missing optional dependency error | The integration extra is not installed. | Install the exact `Muffakir[...]` extra named in the error. |
| Unsupported provider | A misspelled provider name was supplied. | Use a value in the supported table or one of the accepted aliases. |
| Azure endpoint/deployment error | Azure-specific connection values are missing. | Pass `azure_endpoint` and deployment configuration; do not rely on `base_url`. |
| Connection refused from Ollama or a local server | The server is stopped, the URL is wrong, or the model is unavailable. | Start the server, confirm the URL, and pull/load the model. |
| Authentication, rate-limit, timeout, or service error during `call()` | The remote provider rejected or could not complete the request. | Check account access, model availability, quota, and network connectivity; Muffakir re-raises a typed provider error with sensitive keys redacted. |

For application-wide configuration keys, see the [configuration reference](../reference/configuration.md).
