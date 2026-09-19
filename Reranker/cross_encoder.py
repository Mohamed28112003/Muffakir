import logging
from typing import List, Tuple, Optional
from .base import BaseReranker

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder Reranker.
    Uses a bi-directional cross-encoder model to jointly score (query, document) pairs.
    This is the highest quality reranking method — the model reads both query and document
    together, producing a precise relevance score.

    Default model: 'BAAI/bge-reranker-base' (backward compatible).
    Requires: sentence_transformers (`pip install sentence-transformers`).
    """

    def __init__(
        self,
        model_name: str = 'BAAI/bge-reranker-base',
        device: str = "auto",
    ):
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
            # Local import: avoids a circular import with Muffakir's package
            # __init__ chain. See Muffakir/device.py.
            from Muffakir.device import resolve_device
            self._cross_encoder = CrossEncoder(self.model_name, device=resolve_device(device))
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for Cross-Encoder reranking. "
                "Install via `pip install sentence-transformers`."
            )
        except Exception as exc:
            from Muffakir.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Failed to load Hugging Face reranker model '{self.model_name}' "
                "for cross_encoder. Use a sentence-transformers CrossEncoder-"
                f"compatible sequence-classification model. Original error: {exc}"
            ) from exc

    @property
    def name(self) -> str:
        return "cross_encoder"

    def score(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        pairs = [(query, doc.page_content) for doc in documents]
        raw_scores = self._cross_encoder.predict(pairs)

        ranked = sorted(
            zip(documents, raw_scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [(doc, float(score)) for doc, score in ranked]
