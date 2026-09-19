# LLMProvider.py
from typing import Any, Union, Optional
import logging
from importlib import import_module


class LLMProvider:
    """
    Facade for building LangChain chat models from a unified configuration.

    Validates arguments, dispatches to the concrete provider builder
    (Groq / Together / OpenRouter / OpenAI / Anthropic / Gemini / Ollama /
    Azure OpenAI / Custom OpenAI-compatible), and exposes ``get_llm()`` / ``call()`` over the resulting
    model. Heavy SDK imports happen lazily inside the concrete builders, so
    optional dependencies stay optional.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Union["ProviderName", str] = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
        max_tokens: int = 300,
        base_url: Optional[str] = None,
        parameters: Optional[dict] = None,
        **kwargs: Any,
    ):
        from Muffakir.Enums import ProviderName, resolve_provider_name
        from Muffakir.exceptions import ConfigurationError

        # Convert string provider name to canonical ProviderName enum
        try:
            self.provider = resolve_provider_name(provider)
        except ValueError as e:
            raise ConfigurationError(str(e)) from e

        # Ollama and Custom local endpoints may not require an API key
        if not api_key and self.provider not in (ProviderName.OLLAMA, ProviderName.CUSTOM, ProviderName.CUSTOM_OPENAI):
            raise ConfigurationError("An API key must be provided.")


        self.api_key = api_key or ""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.extra = kwargs
        if parameters is not None:
            from .parameters import validate_parameters
            from Muffakir.exceptions import ConfigurationError
            try:
                self.extra["parameters"] = validate_parameters(parameters, self.provider)
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
        if self.base_url and "base_url" not in self.extra:
            self.extra["base_url"] = self.base_url

        self.logger = logging.getLogger(__name__)

        self.llm = self.initialize_llm()

        # Attach a usage-tracking callback directly onto the constructed model
        # instance (post-construction), rather than requiring every one of the
        # ~15 call sites across the codebase to pass callbacks= explicitly.
        # `.bind()`, `.with_structured_output()`, and prompt|llm chains all
        # wrap this same instance without resetting its `callbacks` field, so
        # LangChain's callback manager still fires `on_llm_end` on it no
        # matter how a caller composes the model afterward.
        from .usage import UsageCallbackHandler

        self._usage_callback = UsageCallbackHandler()
        try:
            existing = list(getattr(self.llm, "callbacks", None) or [])
            self.llm.callbacks = existing + [self._usage_callback]
        except Exception as e:
            self.logger.warning(
                f"Could not attach usage-tracking callback to {type(self.llm).__name__}: {e}. "
                "Token usage / cost tracking will be unavailable for this provider instance."
            )

    def initialize_llm(self) -> Any:
        """Dispatch to the concrete provider builder and return its model."""
        from Muffakir.Enums import ProviderName
        from Muffakir.optional_dependencies import require_optional_dependency

        provider_targets = {
            ProviderName.GROQ: ("groq", ".groq", "GroqLLMProvider"),
            ProviderName.TOGETHER: ("openai", ".together", "TogetherLLMProvider"),
            ProviderName.OPENROUTER: ("openai", ".openrouter", "OpenRouterLLMProvider"),
            ProviderName.OPENAI: ("openai", ".openai", "OpenAILLMProvider"),
            ProviderName.ANTHROPIC: ("anthropic", ".anthropic", "AnthropicLLMProvider"),
            ProviderName.GEMINI: ("gemini", ".gemini", "GeminiLLMProvider"),
            ProviderName.OLLAMA: ("ollama", ".ollama", "OllamaLLMProvider"),
            ProviderName.AZURE_OPENAI: ("openai", ".azure_openai", "AzureOpenAILLMProvider"),
            ProviderName.CUSTOM: ("openai", ".custom", "CustomOpenAILLMProvider"),
            ProviderName.CUSTOM_OPENAI: ("openai", ".custom", "CustomOpenAILLMProvider"),
        }

        target = provider_targets.get(self.provider)
        if target is None:
            from Muffakir.exceptions import ConfigurationError

            valid = [p.name for p in ProviderName]
            raise ConfigurationError(
                f"Unsupported provider: {self.provider!r}. "
                f"Expected one of ProviderName: {valid}"
            )

        feature, module_name, class_name = target
        require_optional_dependency(feature)
        provider_cls = getattr(import_module(module_name, __package__), class_name)

        return provider_cls(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            **self.extra,
        ).get_llm()

    def get_llm(self) -> Any:
        """
        Get the underlying LLM client.
        """
        return self.llm

    def get_usage_totals(self) -> dict:
        """Sum of prompt/completion/total tokens recorded across every call so far."""
        return self._usage_callback.get_totals()

    def get_per_sample_usage_totals(self) -> dict:
        """Sum of prompt/completion/total tokens per (trial_id, sample_index)."""
        return self._usage_callback.get_per_sample_totals()

    def get_sample_usage_totals(self, trial_id: int, sample_index: int) -> dict:
        """Convenience accessor: usage totals for one sample, or all-zero if unseen."""
        return self._usage_callback.get_sample_totals(trial_id, sample_index)

    def reset_usage(self) -> None:
        """Clear recorded token-usage history for this provider instance."""
        self._usage_callback.reset()

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """
        Proxy method to invoke the LLM.

        Prefers ``invoke`` (the LangChain standard interface) and falls back
        to calling the model object directly for legacy clients.
        """
        from Muffakir.exceptions import classify_provider_exception

        try:
            llm = self.get_llm()
            if hasattr(llm, "invoke"):
                return llm.invoke(*args, **kwargs)
            return llm(*args, **kwargs)
        except Exception as e:
            provider_name = getattr(self.provider, "value", str(self.provider))
            typed = classify_provider_exception(e, provider=provider_name, secrets=[self.api_key])
            self.logger.error(f"LLM call failed ({typed.error_code}): {typed.message}")
            raise typed from e


def create_llm_provider(
    provider: Union["ProviderName", str] = "openai",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    max_tokens: int = 300,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory function to easily instantiate an LLMProvider.
    """
    return LLMProvider(
        api_key=api_key,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        **kwargs,
    )
