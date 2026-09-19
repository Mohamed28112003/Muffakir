"""Detection of explicit answer refusals for evaluation observability.

This is deliberately a lightweight, explainable signal rather than an LLM
judge: it identifies the standard no-answer responses emitted by Muffakir's
English and Arabic generation prompts, plus common close variants.
"""

from __future__ import annotations

import re
from typing import Optional


_REFUSAL_PATTERNS = (
    # The response required by the English generation prompt and close variants.
    r"i\s+(?:cannot|can'?t|am\s+unable\s+to)\s+(?:answer|respond)(?:\s+(?:this\s+)?question)?",
    r"i\s+(?:do\s+not|don'?t)\s+have\s+(?:enough|sufficient)\s+(?:information|context)",
    r"(?:there\s+is|insufficient)\s+(?:not\s+)?(?:enough|insufficient)\s+(?:information|context)",
    # The Arabic response required by the Arabic generation prompt and close variants.
    r"لا\s+(?:يمكنني|استطيع|أستطيع|يمكن)\s+(?:الإجابة|الاجابة|أن\s+أجيب)(?:\s+على\s+هذا\s+السؤال)?",
    r"لا\s+(?:تتوفر|توجد|أملك|املك)\s+(?:لدي\s+)?(?:معلومات|بيانات)\s+كافية",
)
_REFUSAL_RE = re.compile(r"^(?:" + "|".join(_REFUSAL_PATTERNS) + r")[\s.!؟،,:;\-]*$", re.IGNORECASE)


def detect_answer_refusal(answer: str) -> Optional[str]:
    """Return a stable reason for an explicit no-answer response, if any.

    We intentionally require the *whole* (short) answer to be a refusal. This
    avoids flagging a legitimate answer that merely quotes the phrase while
    discussing it.
    """
    normalized = " ".join(str(answer or "").strip().split())
    if not normalized or len(normalized) > 500:
        return None
    return "insufficient_information" if _REFUSAL_RE.fullmatch(normalized) else None
