import logging
import re
from typing import List
from .base import BaseHallucinationChecker, HallucinationResult

logger = logging.getLogger(__name__)


class NLIChecker(BaseHallucinationChecker):
    """
    NLI-Based Faithfulness Checker.

    Uses a Natural Language Inference (NLI) cross-encoder model to classify
    whether the retrieved context ENTAILS, CONTRADICTS, or is NEUTRAL to the
    generated answer.

    No LLM API calls required — fast, offline, and cost-free.

    Algorithm:
    1. Split the context into sentences (chunks).
    2. For each context sentence, score (sentence, answer) using the NLI model.
    3. Aggregate: if any sentence ENTAILS the answer → grounded.
       If the majority CONTRADICT → hallucination.

    Default model: 'cross-encoder/nli-deberta-v3-small'
    Multilingual model: 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli' (Arabic-compatible)
    Requires: sentence_transformers (`pip install sentence-transformers`).

    Label mapping (DeBERTa NLI):
      0 = contradiction
      1 = entailment
      2 = neutral

    ⚠️ Label-order caveat: this mapping is specific to DeBERTa-style NLI
    models (cross-encoder/nli-*).  Other NLI models may use different label
    orderings — passing such a model via ``model_name`` will silently produce
    wrong verdicts.  When available, the model's own ``label2id`` config is
    used instead of the hardcoded default.
    """

    LABEL_CONTRADICTION = 0
    LABEL_ENTAILMENT = 1
    LABEL_NEUTRAL = 2

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        entailment_threshold: float = 0.7,
        contradiction_threshold: float = 0.7
    ):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for NLI checking. "
                "Install via `pip install sentence-transformers`."
            )

        # Prefer the model's own label mapping when it exposes one.
        self.label_entailment = self.LABEL_ENTAILMENT
        self.label_contradiction = self.LABEL_CONTRADICTION
        config = getattr(self._model, "model", None)
        config = getattr(config, "config", None)
        label2id = getattr(config, "label2id", None)
        if isinstance(label2id, dict) and label2id:
            lowered = {str(k).lower(): v for k, v in label2id.items()}
            for key in ("entailment", "entail"):
                if key in lowered:
                    self.label_entailment = int(lowered[key])
            for key in ("contradiction", "contradict"):
                if key in lowered:
                    self.label_contradiction = int(lowered[key])

    @property
    def name(self) -> str:
        return "nli"

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for NLI scoring."""
        # Split on Arabic and Latin sentence boundaries
        sentences = re.split(r'(?<=[.!?؟\n])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def check(
        self,
        answer: str,
        context: str = "",
        query: str = ""
    ) -> HallucinationResult:
        if not context.strip():
            return HallucinationResult(
                is_hallucination=False,
                confidence=0.0,
                cleaned_answer=answer,
                reasoning="No context provided; NLI check skipped."
            )

        context_sentences = self._split_into_sentences(context)
        if not context_sentences:
            context_sentences = [context[:500]]  # Fallback: use whole context

        # Score each (context_sentence, answer) pair
        pairs = [(sent, answer) for sent in context_sentences]
        try:
            # scores shape: (n_pairs, 3) — [contradiction, entailment, neutral]
            raw_scores = self._model.predict(pairs, apply_softmax=True)
        except Exception as e:
            from Muffakir.exceptions import HallucinationCheckError

            logger.error(f"NLI model scoring failed: {e}")
            raise HallucinationCheckError(f"NLI hallucination check failed: {e}") from e

        # Aggregate: check if any sentence strongly ENTAILS the answer
        max_entailment = max(s[self.label_entailment] for s in raw_scores)
        max_contradiction = max(s[self.label_contradiction] for s in raw_scores)

        if max_entailment >= self.entailment_threshold:
            return HallucinationResult(
                is_hallucination=False,
                confidence=float(max_entailment),
                cleaned_answer=answer,
                reasoning=(
                    f"NLI: Context entails the answer with confidence {max_entailment:.2f}. "
                    "Answer appears to be grounded."
                )
            )
        elif max_contradiction >= self.contradiction_threshold:
            return HallucinationResult(
                is_hallucination=True,
                confidence=float(max_contradiction),
                cleaned_answer=answer,
                reasoning=(
                    f"NLI: Context contradicts the answer with confidence {max_contradiction:.2f}. "
                    "Possible hallucination detected."
                )
            )
        else:
            # Neutral — inconclusive, treat as uncertain (not hallucination)
            return HallucinationResult(
                is_hallucination=False,
                confidence=float(max_entailment),
                cleaned_answer=answer,
                reasoning=(
                    f"NLI: Verdict is neutral (entailment={max_entailment:.2f}, "
                    f"contradiction={max_contradiction:.2f}). Insufficient context signal."
                )
            )
