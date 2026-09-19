import re
import logging
from typing import Optional
from .base import BaseHallucinationChecker, HallucinationResult
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt

logger = logging.getLogger(__name__)


class TextCleanerChecker(BaseHallucinationChecker):
    """
    Text Cleaner (Post-Processor) Checker.

    This is a direct port of the original HallucinationsCheck behavior.
    It uses the LLM to rewrite and clean the generated answer — removing
    control characters, noise, repetition, and obvious fabricated content.

    Unlike the other checkers, this strategy MODIFIES the answer (the
    `cleaned_answer` field in the result may differ from the input answer).
    It does not provide a binary hallucination verdict; `is_hallucination`
    is always False since it cannot detect unsupported claims.

    Use case: Post-processing to improve answer quality and readability.
    Prompt key: `hallucination_check_prompt` (existing bilingual prompt).
    """

    _CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[MuffakirPrompt] = None,
        prompt_key: str = "hallucination_check_prompt"
    ):
        if llm_provider is None:
            raise ValueError("llm_provider is required for TextCleanerChecker.")

        self.llm_provider = llm_provider
        # NOTE: If prompt_manager is not provided, a default Arabic-language
        # MuffakirPrompt is created implicitly. Pass an explicit prompt_manager
        # configured with your target language to avoid this fallback.
        self.prompt_manager = prompt_manager or MuffakirPrompt(language="ar")
        self.prompt_key = prompt_key

    @property
    def name(self) -> str:
        return "text_cleaner"

    def _strip_control_chars(self, text: str) -> str:
        """Remove control characters while preserving Arabic, Latin, and Unicode text."""
        return self._CONTROL_CHAR_RE.sub('', text).strip()

    @staticmethod
    def _extract_response_text(response) -> str:
        """Extract usable text from an LLM response, including thinking/reasoning models.

        Handles three formats:
        1. Normal: ``response.content`` is the answer.
        2. Thinking tags: content contains ``<think>…</think>`` — stripped out.
        3. Separate reasoning: ``response.additional_kwargs["reasoning_content"]``
           used as fallback when ``content`` is empty.
        """
        content = ""
        if hasattr(response, "content"):
            content = response.content or ""

        if content:
            import re as _re
            stripped = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
            if stripped:
                return stripped

        if hasattr(response, "additional_kwargs") and response.additional_kwargs:
            reasoning = response.additional_kwargs.get("reasoning_content", "")
            if reasoning and reasoning.strip():
                return reasoning.strip()

        if hasattr(response, "content"):
            return ""

        return str(response) if response is not None else ""

    def check(
        self,
        answer: str,
        context: str = "",
        query: str = ""
    ) -> HallucinationResult:
        if not answer or not answer.strip() or answer.startswith("❌"):
            return HallucinationResult(
                is_hallucination=False,
                confidence=0.0,
                cleaned_answer=answer,
                reasoning="Empty or error answer skipped by TextCleaner."
            )

        cleaned_answer = answer


        try:
            llm = self.llm_provider.get_llm()
            prompt_template = self.prompt_manager.get_prompt(self.prompt_key)
            prompt = prompt_template.format(answer=answer)
            response = llm.invoke(prompt)

            raw = self._extract_response_text(response)
            cleaned_answer = self._strip_control_chars(raw)

        except Exception as e:
            logger.warning(f"TextCleanerChecker LLM call failed: {e}. Returning original answer.")
            cleaned_answer = self._strip_control_chars(answer)

        return HallucinationResult(
            is_hallucination=False,
            confidence=0.0,
            cleaned_answer=cleaned_answer,
            reasoning="TextCleaner: LLM post-processed the answer for clarity and noise removal."
        )
