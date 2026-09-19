"""LLM provider public API with lazy adapter imports."""

from importlib import import_module

from .LLMProvider import LLMProvider, create_llm_provider
from .base import BaseLLMProvider

__all__ = [
    "LLMProvider",
    "create_llm_provider",
    "BaseLLMProvider",
    "GroqLLMProvider",
    "TogetherLLMProvider",
    "OpenRouterLLMProvider",
    "OpenAILLMProvider",
    "AnthropicLLMProvider",
    "GeminiLLMProvider",
    "OllamaLLMProvider",
    "AzureOpenAILLMProvider",
    "CustomOpenAILLMProvider",
]

_LAZY_EXPORTS = {
    "GroqLLMProvider": ("LLMProvider.groq", "GroqLLMProvider"),
    "TogetherLLMProvider": ("LLMProvider.together", "TogetherLLMProvider"),
    "OpenRouterLLMProvider": ("LLMProvider.openrouter", "OpenRouterLLMProvider"),
    "OpenAILLMProvider": ("LLMProvider.openai", "OpenAILLMProvider"),
    "AnthropicLLMProvider": ("LLMProvider.anthropic", "AnthropicLLMProvider"),
    "GeminiLLMProvider": ("LLMProvider.gemini", "GeminiLLMProvider"),
    "OllamaLLMProvider": ("LLMProvider.ollama", "OllamaLLMProvider"),
    "AzureOpenAILLMProvider": ("LLMProvider.azure_openai", "AzureOpenAILLMProvider"),
    "CustomOpenAILLMProvider": ("LLMProvider.custom", "CustomOpenAILLMProvider"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
