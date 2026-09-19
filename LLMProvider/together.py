from typing import Any
from .base import BaseLLMProvider


class TogetherLLMProvider(BaseLLMProvider):
    PARAMETER_PROVIDER = "together"

    def _build_llm(self) -> Any:
        if not self.api_key:
            from Muffakir.exceptions import ConfigurationError

            raise ConfigurationError("api_key is required for Together.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "The 'langchain-openai' package is required for Together. "
                "Please install it using: pip install langchain-openai"
            ) from e

        return ChatOpenAI(
            openai_api_key=self.api_key,
            openai_api_base="https://api.together.xyz/v1",
            model=self.model,
            **self.generation_kwargs,
        )
