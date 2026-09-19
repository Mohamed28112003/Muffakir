import logging
from typing import Optional
from pydantic import BaseModel, Field

from .base import BaseHallucinationChecker, HallucinationResult
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


# ── Pydantic structured output schema ──────────────────────────────────────────

class GroundingVerdict(BaseModel):
    """Structured verdict for context grounding evaluation."""

    is_grounded: bool = Field(
        description=(
            "True if the answer is fully supported by the retrieved context. "
            "False if the answer contains information not present in the context "
            "(i.e., hallucinated or fabricated content)."
        )
    )
    confidence: float = Field(
        description="Confidence in the verdict from 0.0 (uncertain) to 1.0 (certain).",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="One or two sentence explanation of why the answer is or is not grounded."
    )


# ── Context Grounding Checker ───────────────────────────────────────────────────

class ContextGroundingChecker(BaseHallucinationChecker):
    """
    LLM-as-Judge Context Grounding Checker.

    Evaluates whether the generated answer is fully supported (grounded)
    by the retrieved context. Any claim in the answer that is not present
    in the context is considered hallucinated.

    Architecture:
    - Primary: Uses `llm.with_structured_output(GroundingVerdict)` for reliable
      JSON-enforced output (OpenAI, Anthropic, Groq, etc.).
    - Fallback: Plain text generation + keyword parsing if provider lacks
      function calling support.

    Temperature is forced to 0.0 for deterministic, consistent verdicts.
    """

    _GROUNDING_TEMPLATE = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert fact-checker for RAG (Retrieval-Augmented Generation) systems.\n"
                "Your task is to evaluate whether a generated answer is FULLY SUPPORTED by the retrieved context.\n\n"
                "Rules:\n"
                "1. The answer is GROUNDED if every factual claim in it can be directly found in or inferred from the context.\n"
                "2. The answer is NOT GROUNDED (hallucinated) if it contains any fact, name, number, or claim "
                "that is absent from the context.\n"
                "3. Be strict — even one unsupported claim makes the answer not grounded.\n"
                "4. Ignore stylistic differences; focus only on factual accuracy vs. the context."
            )
        ),
        (
            "human",
            "Retrieved Context:\n{context}\n\nGenerated Answer:\n{answer}"
        )
    ])

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[MuffakirPrompt] = None,
    ):
        if llm_provider is None:
            raise ValueError("llm_provider is required for ContextGroundingChecker.")

        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager or MuffakirPrompt(language="ar")
        self.prompt_template = self.prompt_manager.get_prompt("context_grounding")
        self._structured_chain = None

        # Try to build structured output chain at init
        try:
            llm = self.llm_provider.get_llm()
            # Use bind for temperature override (LangChain standard) — avoids
            # mutating shared state on a possibly-concurrent llm_provider.
            try:
                judge_llm = llm.bind(temperature=0.0)
            except (AttributeError, TypeError):
                judge_llm = llm
            structured_llm = judge_llm.with_structured_output(GroundingVerdict)
            managed_prompt = ChatPromptTemplate.from_messages([
                ("human", self.prompt_template)
            ])
            self._structured_chain = managed_prompt | structured_llm
            logger.info("ContextGroundingChecker: Using Pydantic structured output.")
        except Exception as e:
            logger.warning(
                f"ContextGroundingChecker: Structured output unavailable ({e}). "
                "Using fallback keyword parsing."
            )

    @property
    def name(self) -> str:
        return "context_grounding"

    def _check_structured(self, answer: str, context: str) -> Optional[GroundingVerdict]:
        """Check using structured output chain."""
        try:
            result: GroundingVerdict = self._structured_chain.invoke({
                "answer": answer,
                "context": context
            })
            return result
        except Exception as e:
            logger.debug(f"Structured grounding check failed: {e}")
            return None

    def _check_fallback(self, answer: str, context: str, llm) -> GroundingVerdict:
        """Fallback: plain text generation + keyword parsing."""
        prompt = self.prompt_template.format(context=context, answer=answer)
        try:
            response = llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            raw_lower = raw.strip().lower()
            is_grounded = raw_lower.startswith(("yes", "نعم")) or "grounded" in raw_lower
            return GroundingVerdict(
                is_grounded=is_grounded,
                confidence=0.7,
                reasoning=raw.strip()[:200]
            )
        except Exception as e:
            from Muffakir.exceptions import HallucinationCheckError

            logger.error(f"Fallback grounding check failed: {e}")
            raise HallucinationCheckError(f"Context grounding hallucination check failed: {e}") from e

    def check(
        self,
        answer: str,
        context: str = "",
        query: str = ""
    ) -> HallucinationResult:
        verdict = None

        if self._structured_chain is not None:
            verdict = self._check_structured(answer, context)

        if verdict is None:
            base_llm = self.llm_provider.get_llm()
            # Use bind for temperature override — avoids mutating shared state.
            try:
                fallback_llm = base_llm.bind(temperature=0.0)
            except (AttributeError, TypeError):
                fallback_llm = base_llm
            verdict = self._check_fallback(answer, context, fallback_llm)

        logger.debug(
            f"ContextGrounding verdict: grounded={verdict.is_grounded}, "
            f"confidence={verdict.confidence:.2f}, reasoning='{verdict.reasoning[:60]}...'"
        )

        return HallucinationResult(
            is_hallucination=not verdict.is_grounded,
            confidence=verdict.confidence,
            cleaned_answer=answer,  # This checker detects only; does not rewrite
            reasoning=verdict.reasoning
        )
