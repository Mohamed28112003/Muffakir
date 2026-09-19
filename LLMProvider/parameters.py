"""Portable LLM settings, validation and provider capabilities (no SDK imports)."""
from copy import deepcopy
import math


FIELDS = {
    "temperature": {"label": "Temperature", "min": 0, "max": 2, "nullable": True},
    "max_tokens": {"label": "Maximum output tokens", "min": 1, "integer": True},
    "top_p": {"label": "Top-p", "min": 0, "max": 1},
    "stop": {"label": "Stop sequences", "type": "strings"},
    "seed": {"label": "Seed", "integer": True},
    "top_k": {"label": "Top-k", "min": 1, "integer": True},
    "frequency_penalty": {"label": "Frequency penalty", "min": -2, "max": 2},
    "presence_penalty": {"label": "Presence penalty", "min": -2, "max": 2},
    "timeout_seconds": {"label": "Request timeout (seconds)", "exclusive_min": 0},
    "max_retries": {"label": "Provider request retries", "min": 0, "integer": True},
}
_COMMON = {"temperature", "max_tokens", "top_p", "stop", "timeout_seconds"}
_OPENAI = _COMMON | {"seed", "frequency_penalty", "presence_penalty", "max_retries"}
SUPPORT = {
    **{p: _OPENAI for p in ("openai", "azure_openai", "custom", "together", "openrouter")},
    "groq": _COMMON | {"max_retries"},
    "anthropic": _COMMON | {"top_k", "max_retries"},
    "gemini": _COMMON | {"top_k", "max_retries"},
    "ollama": _COMMON | {"top_k", "seed"},
}
PARAMETER_CONFIG_KEYS = ("llm_parameters", "judge_llm_parameters", "query_transform_llm_parameters",
                         "reranker_llm_parameters", "dataset_llm_parameters")


def canonical_provider(provider):
    from Muffakir.Enums import resolve_provider_name
    name = resolve_provider_name(provider).value
    return "custom" if name == "custom_openai" else name


def parameter_capabilities(provider):
    name = canonical_provider(provider)
    result = {key: deepcopy(spec) for key, spec in FIELDS.items() if key in SUPPORT[name]}
    if name == "anthropic":
        result["temperature"]["max"] = 1
    return result


def validate_parameters(parameters, provider=None):
    """Return a defensive copy; reject unknown fields and coercion of user input."""
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise ValueError("LLM parameters must be an object")
    specs = parameter_capabilities(provider) if provider else FIELDS
    for key, value in parameters.items():
        if key not in specs:
            raise ValueError(f"LLM parameter '{key}' is not supported" + (f" by {provider}" if provider else ""))
        spec = specs[key]
        if value is None and spec.get("nullable"):
            continue
        if spec.get("type") == "strings":
            if not isinstance(value, list) or any(not isinstance(s, str) or not s for s in value):
                raise ValueError("stop must be a list of non-empty strings")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if spec.get("integer") and not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        if ("min" in spec and value < spec["min"]) or ("max" in spec and value > spec["max"]):
            raise ValueError(f"{key} must be between {spec.get('min', '-infinity')} and {spec.get('max', 'infinity')}")
        if "exclusive_min" in spec and value <= spec["exclusive_min"]:
            raise ValueError(f"{key} must be greater than {spec['exclusive_min']}")
    return {k: (float(v) if isinstance(v, (int, float)) and not FIELDS[k].get("integer") else deepcopy(v))
            for k, v in parameters.items()}


def resolve_parameters(config, role=None, *, temperature=0.0, max_tokens=4096):
    """Role objects are independent; absent role objects retain legacy inheritance."""
    values = {"temperature": config.get("llm_temperature", temperature),
              "max_tokens": config.get("llm_max_tokens", max_tokens)}
    role_key = f"{role}_llm_parameters" if role else "llm_parameters"
    selected = config.get(role_key)
    if selected is None:
        selected = config.get("llm_parameters")
    values.update(validate_parameters(selected))
    return validate_parameters(values)


def provider_kwargs(provider, parameters):
    """Translate only validated portable parameters to the installed adapter API."""
    values = validate_parameters(parameters, provider)
    name = canonical_provider(provider)
    if values.get("temperature", 0) is None:
        values.pop("temperature")
    aliases = {"timeout_seconds": "timeout"}
    if name == "gemini":
        aliases.update(max_tokens="max_output_tokens")
    elif name == "anthropic":
        aliases.update(stop="stop_sequences")
    elif name == "ollama":
        aliases.update(max_tokens="num_predict")
        if "timeout_seconds" in values:
            values["client_kwargs"] = {"timeout": values.pop("timeout_seconds")}
    return {aliases.get(k, k): v for k, v in values.items()}


def config_parameter_fields(config):
    """Copy only present settings so legacy requests retain their old behavior."""
    return {k: deepcopy(config[k]) for k in (*PARAMETER_CONFIG_KEYS, "llm_temperature", "llm_max_tokens")
            if config.get(k) is not None}


def validate_config_parameters(config, search_space=None):
    """Offline validation of every selected model and explicit auxiliary role."""
    for key in PARAMETER_CONFIG_KEYS:
        if config.get(key) is not None:
            validate_parameters(config[key])
    choices = (search_space or {}).get("llm") or []
    if config.get("pipeline_mode") == "retrieval_only" and choices:
        raise ValueError("Retrieval-only mode cannot tune generation LLM configurations")
    generation = choices or ([{"provider": config["llm_provider"]}] if config.get("llm_provider") else [])
    seen = set()
    import json
    for choice in generation:
        params = resolve_parameters(config)
        params.update(validate_parameters(choice.get("parameters")))
        validate_parameters(params, choice["provider"])
        identity = json.dumps([canonical_provider(choice["provider"]), choice.get("model"), params], sort_keys=True)
        if identity in seen:
            raise ValueError("Duplicate LLM variant: provider, model and effective parameters must differ")
        seen.add(identity)
    for role in ("judge", "dataset", "query_transform", "reranker"):
        if config.get(f"{role}_llm_parameters") is None:
            continue
        providers = [config.get(f"{role}_llm_provider")]
        if not providers[0]:
            if role == "reranker" and config.get("pipeline_mode") == "retrieval_only":
                providers = [config.get("query_transform_llm_provider")]
            elif role in {"judge", "dataset"}:
                providers = [config.get("llm_provider")]
            else:
                providers = [c["provider"] for c in generation]
        for provider in providers:
            validate_parameters(resolve_parameters(config, role), provider)
