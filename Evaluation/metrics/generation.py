from __future__ import annotations

import logging
import re
from typing import Optional

from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt
from HallucinationsCheck.context_grounding import ContextGroundingChecker

logger = logging.getLogger(__name__)


class FaithfulnessMetric:
    """Faithfulness via ContextGroundingChecker (1.0 grounded, 0.0 otherwise)."""

    def __init__(self, llm_provider: LLMProvider, prompt_manager: Optional[MuffakirPrompt] = None):
        self.checker = ContextGroundingChecker(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
        )

    def score(self, answer: str, context: str, query: str = "") -> Optional[float]:
        """Return 1.0 if grounded, 0.0 if hallucinated.

        Raises whatever ``ContextGroundingChecker.check()`` raises on failure
        (e.g. ``HallucinationCheckError``) instead of swallowing it — the
        caller (``Evaluation/runner.py::_evaluate_one``) already distinguishes
        typed, systemic failures (abort the whole run) from per-sample bugs
        (exclude just this sample) and must see the real exception to do so.
        """
        result = self.checker.check(answer=answer, context=context, query=query)
        return 0.0 if result.is_hallucination else 1.0


class AnswerCorrectnessMetric:
    """LLM-as-judge answer correctness against gold answer (score 0–1)."""

    def __init__(self, llm_provider: LLMProvider, prompt_manager: MuffakirPrompt):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.prompt = self.prompt_manager.get_prompt("answer_correctness")

    def score(self, question: str, gold_answer: str, predicted_answer: str) -> Optional[float]:
        """Return correctness score (0.0-1.0).

        A prompt-formatting bug propagates untyped (per-sample exclusion via
        the caller's dispatcher). A judge LLM-call failure is classified into
        a typed ``ProviderError`` and raised — a live provider outage will
        fail identically for every remaining sample, so the caller
        (``Evaluation/runner.py::_evaluate_one``) aborts the whole run instead
        of silently excluding N samples one by one.
        """
        prompt = self.prompt.format(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )

        llm = self.llm_provider.get_llm()
        # Use bind for temperature override (LangChain standard, avoids
        # mutating shared state on a possibly-concurrent llm_provider).
        try:
            judge = llm.bind(temperature=0.0)
        except (AttributeError, TypeError):
            judge = llm

        try:
            if hasattr(judge, "invoke"):
                response = judge.invoke(prompt)
            else:
                response = judge(prompt)
        except Exception as e:
            from Muffakir.exceptions import classify_provider_exception

            logger.error(f"Answer correctness judge failed: {e}", exc_info=True)
            raise classify_provider_exception(e, provider="answer_correctness_judge") from e

        text = response.content if hasattr(response, "content") else str(response)
        return self._parse_score(text)

    @staticmethod
    def _parse_score(text: str) -> float:
        cleaned = (text or "").strip()
        # Prefer explicit score=0.85 / score=.85 patterns
        m = re.search(r"(?:score|درجة)\s*[:=]\s*(0?\.\d+|[01](?:\.\d+)?)", cleaned, re.IGNORECASE)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))

        # Last 0/1-prefixed float in the text, not the first — judges typically
        # reason first and state their final verdict last, so an incidental
        # number earlier in the reasoning ("rate this 0 out of ... but overall
        # 0.9 is fair") must not win over the actual concluding score.
        last_val = None
        for match in re.finditer(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", cleaned):
            val = float(match.group(1))
            if 0.0 <= val <= 1.0:
                last_val = val
        if last_val is not None:
            return last_val

        lower = cleaned.lower()
        # NOTE: negative keywords are checked FIRST — "incorrect" contains
        # "correct" as a substring, so positive-first ordering misclassified
        # negated answers as correct.
        if any(tok in lower for tok in ("incorrect", "خطأ", "no", "لا")):
            return 0.0
        if any(tok in lower for tok in ("correct", "صحيح", "yes", "نعم")):
            return 1.0
        logger.warning(
            f"AnswerCorrectnessMetric could not parse a score from judge response, "
            f"defaulting to 0.0: {cleaned[:200]!r}"
        )
        return 0.0


class LLMJudgeRatingMetric:
    """Semantic answer correctness against a reference answer (integer 1–5)."""

    def __init__(self, llm_provider: LLMProvider, prompt_manager: MuffakirPrompt):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.prompt = self.prompt_manager.get_prompt("llm_judge_rating")

    def score(self, gold_answer: str, predicted_answer: str) -> int:
        """Return an integer rating from 1 through 5.

        Malformed judge output raises ``ValueError`` so the runner records a
        per-sample evaluation error instead of inventing a valid-looking
        rating. Provider-call failures are promoted to ``ProviderError`` and
        abort the run, matching the other LLM-backed generation metrics.
        """
        prompt = self.prompt.format(
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )

        llm = self.llm_provider.get_llm()
        try:
            judge = llm.bind(temperature=0.0)
        except (AttributeError, TypeError):
            judge = llm

        try:
            if hasattr(judge, "invoke"):
                response = judge.invoke(prompt)
            else:
                response = judge(prompt)
        except Exception as e:
            from Muffakir.exceptions import classify_provider_exception

            logger.error(f"LLM judge rating call failed: {e}", exc_info=True)
            raise classify_provider_exception(e, provider="llm_judge_rating_judge") from e

        text = response.content if hasattr(response, "content") else str(response)
        return self._parse_rating(text)

    @staticmethod
    def _parse_rating(text: str) -> int:
        cleaned = re.sub(r"[*_`]", "", (text or "").strip())

        # Accept an explicitly labelled rating in either supported language.
        # Capture the complete numeric token first so values such as 4.5, 0,
        # and 6 are rejected rather than partially parsed into a valid rating.
        labelled = re.search(
            r"(?:llm\s+judge\s+rating|rating|التقييم|الدرجة)\s*[:=]\s*"
            r"([-+]?\d+(?:[.٫]\d+)?)(?![\d.٫])",
            cleaned,
            re.IGNORECASE,
        )
        if labelled:
            token = labelled.group(1)
            if len(token) == 1 and token.isdecimal() and 1 <= int(token) <= 5:
                return int(token)
            raise ValueError(f"LLM Judge Rating must be an integer from 1 to 5, got {token!r}")

        # A bare integer is also valid, but surrounding explanation without an
        # explicit label is deliberately rejected to keep the contract strict.
        if len(cleaned) == 1 and cleaned.isdecimal() and 1 <= int(cleaned) <= 5:
            return int(cleaned)

        raise ValueError(
            "Could not parse LLM Judge Rating; expected a standalone integer "
            "from 1 to 5 or a labelled 'rating: <integer>' response"
        )
