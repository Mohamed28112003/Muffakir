from typing import Any
from .base import BaseLLMProvider


class AnthropicLLMProvider(BaseLLMProvider):
    PARAMETER_PROVIDER = "anthropic"

    def _build_llm(self) -> Any:
        if not self.api_key:
            from Muffakir.exceptions import ConfigurationError

            raise ConfigurationError("api_key is required for Anthropic.")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "The 'langchain-anthropic' package is required for Anthropic. "
                "Please install it using: pip install Muffakir[anthropic]"
            ) from e

        return ChatAnthropic(
            api_key=self.api_key,
            model=self.model,
            **self.generation_kwargs,
        )
