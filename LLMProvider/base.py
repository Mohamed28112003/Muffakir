from abc import ABC, abstractmethod
from typing import Any, Optional
import logging


class BaseLLMProvider(ABC):
    """
    Abstract base for pluggable LLM providers in Muffakir.
    """
    PARAMETER_PROVIDER = None

    def __init__(
        self,
        model: str,
        temperature: float = 0.5,
        max_tokens: int = 300,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.extra = kwargs
        from .parameters import validate_parameters, provider_kwargs
        from Muffakir.exceptions import ConfigurationError
        provider = self.PARAMETER_PROVIDER
        values = {"temperature": temperature, "max_tokens": max_tokens}
        values.update(self.extra.pop("parameters", None) or {})
        try:
            self.parameters = validate_parameters(values, provider)
            self.generation_kwargs = (provider_kwargs(provider, self.parameters) if provider else
                                      {k: v for k, v in self.parameters.items() if v is not None})
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        self.logger = logging.getLogger(self.__class__.__name__)
        self._llm = self._build_llm()

    @abstractmethod
    def _build_llm(self) -> Any:
        """Construct and return the underlying LangChain chat model."""
        ...

    def get_llm(self) -> Any:
        return self._llm

    def call(self, *args: Any, **kwargs: Any) -> Any:
        try:
            llm = self.get_llm()
            if hasattr(llm, "invoke"):
                return llm.invoke(*args, **kwargs)
            return llm(*args, **kwargs)
        except Exception as e:
            from Muffakir.exceptions import redact_secret

            self.logger.error(f"LLM call failed: {redact_secret(str(e), self.api_key)}")
            raise
