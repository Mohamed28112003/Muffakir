import logging
from typing import List, Tuple
from .base import BaseReranker

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


class BM25Reranker(BaseReranker):
    """
    BM25 (Okapi BM25) Reranker.
    Ranks documents using sparse keyword frequency scoring.
    Requires: rank_bm25 (`pip install rank-bm25`).
    """

    def __init__(self):
        try:
            from rank_bm25 import BM25Okapi
            self._BM25Okapi = BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for BM25 reranking. "
                "Install via `pip install rank-bm25`."
            )

    @property
    def name(self) -> str:
        return "bm25"

    def score(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        tokenized_docs = [doc.page_content.split() for doc in documents]
        tokenized_query = query.split()

        bm25 = self._BM25Okapi(tokenized_docs)
        raw_scores = bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(documents, raw_scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [(doc, float(score)) for doc, score in ranked]
