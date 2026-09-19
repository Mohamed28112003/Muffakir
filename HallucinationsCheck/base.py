from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


class HallucinationResult(BaseModel):
    """Standardized result returned by all hallucination checker strategies."""

    is_hallucination: bool = Field(
        description="True if hallucination or unsupported content was detected."
    )
    confidence: float = Field(
        description="Confidence in the verdict, between 0.0 (uncertain) and 1.0 (certain).",
        ge=0.0,
        le=1.0
    )
    cleaned_answer: str = Field(
        description="The original answer, or a corrected/cleaned version if the checker modified it."
    )
    reasoning: str = Field(
        description="Brief explanation for the verdict."
    )


class BaseHallucinationChecker(ABC):
    """
    Abstract Base Class for Hallucination Checker strategies in Muffakir RAG.

    All strategies must implement `check(answer, context)` and return
    a standardized `HallucinationResult` object.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        pass

    @abstractmethod
    def check(
        self,
        answer: str,
        context: str = "",
        query: str = ""
    ) -> HallucinationResult:
        """
        Check whether the generated answer is grounded in the retrieved context.

        Args:
            answer (str): The generated LLM answer to verify.
            context (str): The retrieved context the answer should be grounded in.
            query (str): The original user query (optional, used by some strategies).

        Returns:
            HallucinationResult: Verdict, confidence, cleaned answer, and reasoning.
        """
        pass

    def check_answer(self, answer: str, context: str = "", query: str = "") -> str:
        """
        Backward-compatible interface. Returns only the cleaned answer string.
        """
        result = self.check(answer=answer, context=context, query=query)
        return result.cleaned_answer
