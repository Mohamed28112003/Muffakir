"""
Backward-compatible HallucinationsCheck entry point.
Wraps the factory and preserves the original check_answer(answer) interface.
"""
from typing import Optional, Any
from .base import BaseHallucinationChecker, HallucinationResult
from .factory import create_hallucination_checker
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt


class HallucinationsCheck:
    """
    Backward-compatible HallucinationsCheck wrapper.

    By default delegates to `TextCleanerChecker` (identical to original behavior).
    Pass `method=` to use a different strategy.

    Exposes both:
    - `check_answer(answer)` — original interface (returns str)
    - `check(answer, context, query)` — new interface (returns HallucinationResult)

    Note: ``llm_provider`` is only required for the ``context_grounding`` and
    ``text_cleaner`` methods; ``nli`` and ``semantic_similarity`` strategies
    do not use an LLM and accept ``None`` here.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        prompt_manager: Optional[MuffakirPrompt] = None,
        method: str = "text_cleaner",
        embedding_provider: Optional[Any] = None,
        **kwargs: Any
    ):
        self._checker: BaseHallucinationChecker = create_hallucination_checker(
            method=method,
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            embedding_provider=embedding_provider,
            **kwargs
        )

    def check(
        self,
        answer: str,
        context: str = "",
        query: str = ""
    ) -> HallucinationResult:
        """Full hallucination check — returns structured HallucinationResult."""
        return self._checker.check(answer=answer, context=context, query=query)

    def check_answer(self, answer: str, context: str = "", query: str = "") -> str:
        """Backward-compatible interface — returns only the cleaned answer string."""
        result = self._checker.check(answer=answer, context=context, query=query)
        return result.cleaned_answer
