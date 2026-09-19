import logging
import numpy as np
from .base import BaseHallucinationChecker, HallucinationResult

logger = logging.getLogger(__name__)


class SemanticSimilarityChecker(BaseHallucinationChecker):
    """
    Semantic Similarity Hallucination Checker.

    Embeds both the generated answer and the retrieved context using the
    existing EmbeddingProvider, then computes cosine similarity between them.

    If similarity < threshold → the answer is semantically distant from the
    context, suggesting possible hallucination.

    No LLM API calls required — fast and uses the already-initialized
    EmbeddingProvider.

    Note: This is a weak signal — it catches obvious semantic drift (answer
    talks about completely different topics) but may miss subtle factual errors.
    Best used in combination with other strategies.
    """

    def __init__(
        self,
        embedding_provider=None,
        model_name: str = "mohamed2811/Muffakir_Embedding",
        similarity_threshold: float = 0.3
    ):
        self.similarity_threshold = similarity_threshold

        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            from Embedding import create_embedding_provider
            self.embedding_provider = create_embedding_provider(
                provider="sentence_transformers",
                model_name=model_name
            )

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def _cosine_similarity(self, vec_a: list, vec_b: list) -> float:
        """Compute cosine similarity between two embedding vectors."""
        a = np.array(vec_a, dtype=float)
        b = np.array(vec_b, dtype=float)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)

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
                reasoning="No context provided; semantic similarity check skipped."
            )

        try:
            answer_emb = self.embedding_provider.embed_query(answer)
            context_emb = self.embedding_provider.embed_query(context[:2000])  # Limit for perf
            similarity = self._cosine_similarity(answer_emb, context_emb)
        except Exception as e:
            from Muffakir.exceptions import HallucinationCheckError

            logger.error(f"SemanticSimilarityChecker embedding failed: {e}")
            raise HallucinationCheckError(f"Semantic similarity hallucination check failed: {e}") from e

        is_hallucination = similarity < self.similarity_threshold
        # Confidence measures distance from the decision threshold, normalized
        # so it stays within [0, 1] across threshold choices. A score far from
        # the threshold → high confidence; near the threshold → low confidence.
        denominator = max(self.similarity_threshold, 1 - self.similarity_threshold)
        confidence = abs(similarity - self.similarity_threshold) / denominator
        confidence = min(1.0, confidence)

        reasoning = (
            f"Cosine similarity between answer and context: {similarity:.3f} "
            f"(threshold={self.similarity_threshold}). "
            + ("Answer appears semantically distant from context — possible hallucination."
               if is_hallucination
               else "Answer appears semantically consistent with context.")
        )

        logger.debug(f"SemanticSimilarity: score={similarity:.3f}, hallucination={is_hallucination}")

        return HallucinationResult(
            is_hallucination=is_hallucination,
            confidence=confidence,
            cleaned_answer=answer,
            reasoning=reasoning
        )
