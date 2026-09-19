from typing import Any
from .base import BaseLLMProvider


class GeminiLLMProvider(BaseLLMProvider):
    PARAMETER_PROVIDER = "gemini"

    def _build_llm(self) -> Any:
        if not self.api_key:
            from Muffakir.exceptions import ConfigurationError

            raise ConfigurationError("api_key is required for Gemini.")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "The 'langchain-google-genai' package is required for Gemini. "
                "Please install it using: pip install Muffakir[gemini]"
            ) from e

        return ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model=self.model,
            **self.generation_kwargs,
        )
