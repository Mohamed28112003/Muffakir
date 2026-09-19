from __future__ import annotations

import re
import unicodedata
from typing import Any, List, Optional, Sequence

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


def normalize_text(text: str) -> str:
    """Normalize text for gold/retrieved chunk matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Arabic-specific normalization for consistent matching
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ة", "ه")
        .replace("ى", "ي")
    )
    return text


def _token_jaccard_ratio(a: str, b: str) -> float:
    """Token-level Jaccard similarity (with substring shortcut) for matching."""
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if shorter in longer:
        return 1.0
    # token Jaccard on whitespace tokens
    ta = set(shorter.split())
    tb = set(longer.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_relevant_document(
    retrieved: Document,
    gold_context: str,
    gold_chunk_id: Optional[int] = None,
    overlap_threshold: float = 0.55,
) -> bool:
    """
    Decide whether a retrieved document matches the gold evidence chunk.
    """
    meta = getattr(retrieved, "metadata", None) or {}
    if gold_chunk_id is not None:
        for key in ("chunk_id", "chunkId", "id"):
            if key in meta:
                try:
                    if int(meta[key]) == int(gold_chunk_id):
                        return True
                except (TypeError, ValueError):
                    pass

    retrieved_text = normalize_text(getattr(retrieved, "page_content", "") or "")
    gold = normalize_text(gold_context or "")
    if not retrieved_text or not gold:
        return False

    if gold in retrieved_text or retrieved_text in gold:
        return True

    return _token_jaccard_ratio(retrieved_text, gold) >= overlap_threshold


def build_relevance_list(
    retrieved_docs: Sequence[Document],
    gold_context: str,
    gold_chunk_id: Optional[int] = None,
    k: Optional[int] = None,
    overlap_threshold: float = 0.55,
) -> List[int]:
    """
    Return binary relevance list (1/0) of length k for retrieved docs.

    Args:
        overlap_threshold: Token Jaccard threshold for text-based matching
            (used when chunk_id matching is unavailable).  Default 0.55.
    """
    docs = list(retrieved_docs)
    if k is not None:
        docs = docs[:k]
        # pad if fewer than k retrieved
        while len(docs) < k:
            docs.append(Document(page_content="", metadata={}))

    return [
        1
        if (
            (getattr(doc, "page_content", None) or "").strip()
            and is_relevant_document(doc, gold_context, gold_chunk_id, overlap_threshold)
        )
        else 0
        for doc in docs
    ]
