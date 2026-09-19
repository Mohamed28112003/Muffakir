from typing import Any
from .base import BaseLLMProvider


class GroqLLMProvider(BaseLLMProvider):
    PARAMETER_PROVIDER = "groq"

    def _build_llm(self) -> Any:
        if not self.api_key:
            from Muffakir.exceptions import ConfigurationError

            raise ConfigurationError("api_key is required for Groq.")
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:
            raise ImportError(
                "The 'langchain-groq' package is required for Groq. "
                "Please install it using: pip install langchain-groq"
            ) from e

        return ChatGroq(
            api_key=self.api_key,
            model=self.model,
            **self.generation_kwargs,
        )
