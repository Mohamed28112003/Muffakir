from typing import Any
from .base import BaseLLMProvider


class OpenRouterLLMProvider(BaseLLMProvider):
    PARAMETER_PROVIDER = "openrouter"

    def _build_llm(self) -> Any:
        if not self.api_key:
            from Muffakir.exceptions import ConfigurationError

            raise ConfigurationError("api_key is required for OpenRouter.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "The 'langchain-openai' package is required for OpenRouter. "
                "Please install it using: pip install langchain-openai"
            ) from e

        return ChatOpenAI(
            openai_api_key=self.api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model=self.model,
            **self.generation_kwargs,
        )
