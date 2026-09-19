import logging
from typing import Any, Optional

from .base import BaseLLMProvider


logger = logging.getLogger(__name__)


def _normalize_null_message_roles(response: Any) -> Any:
    """Return an OpenAI-style response with missing/null roles normalized.

    A few otherwise OpenAI-compatible gateways return ``role: null`` for a
    successful assistant message. LangChain interprets that as a generic
    ``ChatMessage`` and Pydantic then rejects its non-string role. Work on a
    shallow structural copy so SDK response objects and caller-owned dicts are
    never mutated. Valid responses are returned unchanged.
    """
    if isinstance(response, dict):
        response_dict = response
    elif hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    else:
        return response

    choices = response_dict.get("choices")
    if not isinstance(choices, list):
        return response

    normalized_choices = []
    changed = False
    for choice in choices:
        if not isinstance(choice, dict):
            normalized_choices.append(choice)
            continue
        message = choice.get("message")
        if isinstance(message, dict) and message.get("role") is None:
            normalized_message = dict(message)
            normalized_message["role"] = "assistant"
            normalized_choice = dict(choice)
            normalized_choice["message"] = normalized_message
            normalized_choices.append(normalized_choice)
            changed = True
        else:
            normalized_choices.append(choice)

    if not changed:
        return response

    normalized_response = dict(response_dict)
    normalized_response["choices"] = normalized_choices
    return normalized_response


class CustomOpenAILLMProvider(BaseLLMProvider):
    """
    Generic OpenAI-compatible LLM Provider.
    Supports any custom base_url / OpenAI-compatible endpoint (e.g., DeepSeek,
    Fireworks AI, vLLM, LM Studio, Anyscale, Sambanova, etc.).
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 300,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        self.base_url = base_url or kwargs.get("openai_api_base") or kwargs.get("api_base")
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or "EMPTY",
            base_url=self.base_url,
            **kwargs,
        )

    PARAMETER_PROVIDER = "custom"

    def _build_llm(self) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "The 'langchain-openai' package is required for custom OpenAI-compatible endpoints. "
                "Please install it using: pip install langchain-openai"
            ) from e

        kwargs = {
            "openai_api_key": self.api_key,
            "model": self.model,
            **self.generation_kwargs,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url

        for k, v in self.extra.items():
            if k not in ("base_url", "openai_api_base", "api_base") and k not in kwargs:
                kwargs[k] = v

        # Keep official OpenAI behavior untouched; this compatibility shim is
        # scoped to user-configured OpenAI-style gateways only. Test doubles
        # and older adapters without the conversion hook keep their original
        # behavior.
        if not hasattr(ChatOpenAI, "_create_chat_result"):
            return ChatOpenAI(**kwargs)

        class CompatibleChatOpenAI(ChatOpenAI):
            def _create_chat_result(self, response, generation_info=None):
                normalized = _normalize_null_message_roles(response)
                if normalized is not response:
                    logger.warning(
                        "Custom OpenAI-compatible endpoint returned a null/missing "
                        "message role; normalized it to 'assistant'"
                    )
                result = super()._create_chat_result(normalized, generation_info)

                # Passing a normalized dict to LangChain bypasses the small
                # BaseModel-only block that copies structured-output metadata.
                # Preserve it explicitly for query transforms/rerankers using
                # with_structured_output().
                if normalized is not response and not isinstance(response, dict):
                    original_choices = getattr(response, "choices", None) or []
                    for generation, choice in zip(result.generations, original_choices):
                        original_message = getattr(choice, "message", None)
                        if original_message is None:
                            continue
                        if hasattr(original_message, "parsed"):
                            generation.message.additional_kwargs["parsed"] = (
                                original_message.parsed
                            )
                        if hasattr(original_message, "refusal"):
                            generation.message.additional_kwargs["refusal"] = (
                                original_message.refusal
                            )
                return result

        return CompatibleChatOpenAI(**kwargs)
