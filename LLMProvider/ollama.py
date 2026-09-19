from typing import Any
from .base import BaseLLMProvider


class OllamaLLMProvider(BaseLLMProvider):
    """Local Ollama chat models. api_key is optional."""

    PARAMETER_PROVIDER = "ollama"

    def _build_llm(self) -> Any:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "The 'langchain-ollama' package is required for Ollama. "
                "Please install it using: pip install Muffakir[ollama]"
            ) from e

        base_url = (
            self.extra.get("base_url")
            or self.extra.get("ollama_base_url")
            or "http://localhost:11434"
        )
        model = self.model or "llama3.2"
        if not self.model:
            self.logger.debug(
                "No model specified for Ollama; defaulting to '%s'.", model
            )
        return ChatOllama(
            model=model,
            base_url=base_url,
            **self.generation_kwargs,
        )
