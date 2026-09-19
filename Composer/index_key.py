"""Deterministic index key for Composer's index-time search dimensions.

Changing chunking, embedding_model, or vector_db_provider changes the indexed
corpus itself (unlike query-time dimensions such as retrieval/reranking/k/llm),
so each distinct combination needs its own on-disk VectorDB. This key identifies
that combination deterministically so a re-run over the same search space finds
and reuses an already-built index instead of rebuilding it, and so
_get_shared_components() (evaluation.py) always agrees with _index_documents()
(composer.py) on which index a given trial should use.
"""

import hashlib
import json
from typing import Any, Dict

INDEX_KEY_FIELDS = (
    "chunking_method",
    "chunk_size",
    "chunk_overlap",
    "embedding_model",
    "vector_db_provider",
)


def compute_index_key(rag_config: Dict[str, Any]) -> str:
    """Compute a short, deterministic key for a resolved MuffakirRAG config's
    index-time dimensions (chunking + embedding model + vector DB provider).
    """
    payload = {field: rag_config.get(field) for field in INDEX_KEY_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
