from typing import Any
from .base import BaseLLMProvider


class OpenAILLMProvider(BaseLLMProvider):
    PARAMETER_PROVIDER = "openai"

    def _build_llm(self) -> Any:
        if not self.api_key:
            from Muffakir.exceptions import ConfigurationError

            raise ConfigurationError("api_key is required for OpenAI.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "The 'langchain-openai' package is required for OpenAI. "
                "Please install it using: pip install langchain-openai"
            ) from e

        kwargs = {
            "openai_api_key": self.api_key,
            "model": self.model,
            **self.generation_kwargs,
        }

        base_url = self.extra.get("base_url") or self.extra.get("openai_api_base") or self.extra.get("api_base")
        if base_url:
            kwargs["base_url"] = base_url

        for k, v in self.extra.items():
            if k not in ("base_url", "openai_api_base", "api_base") and k not in kwargs:
                kwargs[k] = v

        return ChatOpenAI(**kwargs)
