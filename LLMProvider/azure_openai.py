from typing import Any
from .base import BaseLLMProvider


class AzureOpenAILLMProvider(BaseLLMProvider):
    """
    Azure OpenAI via LangChain AzureChatOpenAI.

    Required kwargs: azure_endpoint, azure_deployment (or model as deployment name).
    Optional: api_version
    """

    PARAMETER_PROVIDER = "azure_openai"

    def _build_llm(self) -> Any:
        from Muffakir.exceptions import ConfigurationError

        if not self.api_key:
            raise ConfigurationError("api_key is required for Azure OpenAI.")

        azure_endpoint = self.extra.get("azure_endpoint") or self.extra.get("endpoint")
        azure_deployment = (
            self.extra.get("azure_deployment")
            or self.extra.get("deployment_name")
            or self.model
        )
        api_version = self.extra.get("api_version") or "2024-02-15-preview"

        if not azure_endpoint:
            raise ConfigurationError(
                "azure_endpoint is required for Azure OpenAI "
                "(pass via llm_provider_config or kwargs)."
            )
        if not azure_deployment:
            raise ConfigurationError(
                "azure_deployment (or model) is required for Azure OpenAI."
            )

        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:
            raise ImportError(
                "The 'langchain-openai' package is required for Azure OpenAI. "
                "Please install it using: pip install Muffakir[azure_openai]"
            ) from e

        return AzureChatOpenAI(
            api_key=self.api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            **self.generation_kwargs,
        )
