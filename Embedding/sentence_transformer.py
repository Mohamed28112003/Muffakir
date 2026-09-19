import logging
from typing import List

import numpy as np

from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local SentenceTransformer embedding provider (e.g., 'mohamed2811/Muffakir_Embedding').

    Note on ``batch_size``: this controls the *outer* batching/caching loop in
    :class:`BaseEmbeddingProvider` (how many texts are sent to
    ``_embed_documents_raw`` per call). The SentenceTransformer model also has
    its own internal batching inside ``model.encode()``; the two are independent.
    """

    def __init__(
        self,
        model_name: str = 'mohamed2811/Muffakir_Embedding',
        cache_dir: str = '.embedding_cache',
        batch_size: int = 32,
        device: str = "auto",
    ):
        super().__init__(
            model_name=model_name,
            provider_name="sentence_transformers",
            cache_dir=cache_dir,
            batch_size=batch_size
        )

        try:
            from sentence_transformers import SentenceTransformer
            # Local import: Muffakir is a package whose __init__ eagerly imports
            # MuffakirRAG, which pulls in this module transitively (via
            # Reranker/Embedding facades) — a top-level import here would be
            # circular. See Muffakir/device.py.
            from Muffakir.device import resolve_device
            self.model = SentenceTransformer(
                self.model_name, device=resolve_device(device)
            )
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it via `pip install sentence-transformers`."
            ) from e

    def _embed_documents_raw(self, texts: List[str]) -> List[List[float]]:
        # encode() may return a numpy array, torch tensor, or plain list
        # depending on version/config — normalize before converting.
        return np.asarray(self.model.encode(texts)).tolist()

    def _embed_query_raw(self, text: str) -> List[float]:
        return np.asarray(self.model.encode(text)).tolist()
