import logging
from typing import List, Tuple, Optional
from .base import BaseReranker

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


class PointwiseReranker(BaseReranker):
    """
    Pointwise Learning-to-Rank (L2R) Reranker.

    Each document is scored independently against the query using a Cross-Encoder
    as the scoring function. This is a practical, training-free implementation of
    pointwise L2R where the relevance score is the cross-encoder output probability.

    Optionally supports a relevance threshold to filter out irrelevant documents
    (score below threshold are treated as irrelevant and placed at the end).

    Default model: 'BAAI/bge-reranker-base' (backward compatible).
    Requires: sentence_transformers (`pip install sentence-transformers`).
    """

    def __init__(
        self,
        model_name: str = 'BAAI/bge-reranker-base',
        relevance_threshold: float = 0.0,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold

        try:
            from sentence_transformers import CrossEncoder
            import torch
            # Local import: Muffakir's package __init__ eagerly imports MuffakirRAG,
            # which pulls in this module transitively — a top-level import here
            # would be circular. See Muffakir/device.py.
            from Muffakir.device import resolve_device
            # apply_softmax=True normalizes scores to a probability distribution (0.0 to 1.0)
            self._cross_encoder = CrossEncoder(
                self.model_name,
                default_activation_function=torch.nn.Sigmoid(),
                device=resolve_device(device),
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers and torch are required for Pointwise reranking. "
                "Install via `pip install sentence-transformers torch`."
            )
        except Exception as exc:
            from Muffakir.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Failed to load Hugging Face reranker model '{self.model_name}' "
                "for pointwise. Use a sentence-transformers CrossEncoder-"
                f"compatible sequence-classification model. Original error: {exc}"
            ) from exc

    @property
    def name(self) -> str:
        return "pointwise"

    def score(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Score each document independently against the query.
        Sigmoid activation ensures scores are in [0.0, 1.0] probability range.
        """
        pairs = [(query, doc.page_content) for doc in documents]
        raw_scores = self._cross_encoder.predict(pairs)

        scored = [(doc, float(score)) for doc, score in zip(documents, raw_scores)]

        # Apply relevance threshold — docs below threshold are pushed to the end
        above = [(doc, s) for doc, s in scored if s >= self.relevance_threshold]
        below = [(doc, s) for doc, s in scored if s < self.relevance_threshold]

        above_sorted = sorted(above, key=lambda x: x[1], reverse=True)
        below_sorted = sorted(below, key=lambda x: x[1], reverse=True)

        return above_sorted + below_sorted
