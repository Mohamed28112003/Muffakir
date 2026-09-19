import os
import json
import hashlib
import logging
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional
from langchain_core.embeddings import Embeddings

from .timing import EmbeddingTimingTracker

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(Embeddings, ABC):
    """
    Abstract Base Class for Embedding Providers.

    Provides two layers of caching across all providers:
      1. An in-memory LRU cache (bounded, thread-safe) for hot lookups.
      2. An on-disk JSON cache for persistence across runs.

    Concrete providers implement ``_embed_documents_raw`` and
    ``_embed_query_raw``; this base handles caching, batching, and the
    LangChain ``Embeddings`` interface.
    """

    # Maximum number of entries kept in the in-memory LRU cache.
    _MEMORY_CACHE_SIZE = 1024

    def __init__(
        self,
        model_name: str,
        provider_name: str,
        cache_dir: str = '.embedding_cache',
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.provider_name = provider_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size

        # Thread-safe in-memory LRU cache (key -> embedding).
        self._memory_cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._cache_lock = threading.Lock()

        # Per-sample query-embedding timing (see Trace.context / Embedding.timing).
        self.timing = EmbeddingTimingTracker()

        os.makedirs(cache_dir, exist_ok=True)

    # -- Pickling support (exclude the non-picklable threading.Lock) ----------
    def __getstate__(self):
        """Exclude the threading.Lock(s) so the provider can be pickled (e.g. multiprocessing).

        Timing data is process-local and meaningless across a process
        boundary anyway, so `timing` is reset to a fresh tracker on unpickle
        rather than round-tripped.
        """
        state = self.__dict__.copy()
        state["_cache_lock"] = None
        state["timing"] = None
        return state

    def __setstate__(self, state):
        """Restore the lock(s) after unpickling."""
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()
        self.timing = EmbeddingTimingTracker()

    def _get_cache_key(self, text: str) -> str:
        """Generate a unique cache key for the text, model, and provider combination."""
        combined = f"{self.provider_name}:{self.model_name}:{text}"
        # sha256 avoids SAST weak-hash flags (md5) and eliminates collision risk.
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _store_in_memory(self, cache_key: str, embedding: List[float]) -> None:
        """Insert into the in-memory LRU cache, evicting the oldest entry if full."""
        with self._cache_lock:
            self._memory_cache[cache_key] = embedding
            self._memory_cache.move_to_end(cache_key)
            while len(self._memory_cache) > self._MEMORY_CACHE_SIZE:
                self._memory_cache.popitem(last=False)

    def _check_cache(self, text: str) -> Optional[List[float]]:
        """Check if embedding exists in the in-memory or disk cache."""
        cache_key = self._get_cache_key(text)

        # 1. Hot in-memory cache (no disk I/O).
        with self._cache_lock:
            cached = self._memory_cache.get(cache_key)
            if cached is not None:
                self._memory_cache.move_to_end(cache_key)
                return cached

        # 2. Persistent disk cache.
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    embedding = json.load(f)
                self._store_in_memory(cache_key, embedding)
                return embedding
            except Exception as e:
                from Trace.observability import mark_current_stage_error

                mark_current_stage_error(e, "fallback")
                logger.debug(f"Cache read error: {e}")
        return None

    def _save_to_cache(self, text: str, embedding: List[float]):
        """Save embedding to both the in-memory and disk caches.

        The disk write is atomic: write to a temp file then ``os.replace``
        so concurrent processes cannot observe a half-written JSON file.
        """
        cache_key = self._get_cache_key(text)
        self._store_in_memory(cache_key, embedding)

        cache_path = self._get_cache_path(cache_key)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(embedding, f)
                os.replace(tmp_path, cache_path)
            except Exception:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            from Trace.observability import mark_current_stage_error

            mark_current_stage_error(e, "fallback")
            logger.debug(f"Cache write error: {e}")

    @abstractmethod
    def _embed_documents_raw(self, texts: List[str]) -> List[List[float]]:
        """Provider-specific implementation for batch document embedding."""
        pass

    @abstractmethod
    def _embed_query_raw(self, text: str) -> List[float]:
        """Provider-specific implementation for single query embedding."""
        pass

    def embed_single(self, text: str) -> List[float]:
        """Embed a single query with caching."""
        if text is None:
            raise ValueError("text must not be None.")
        cached = self._check_cache(text)
        if cached is not None:
            return cached

        embedding = self._embed_query_raw(text)
        if not isinstance(embedding, list):
            raise RuntimeError(
                f"{self.provider_name}._embed_query_raw did not return a list."
            )
        self._save_to_cache(text, embedding)
        return embedding

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with batching and caching.

        Uncached texts are batched (size = ``self.batch_size``) and sent to the
        provider's ``_embed_documents_raw``; results are cached and merged back
        into the output list preserving the original order.
        """
        if texts is None:
            raise ValueError("texts must not be None.")
        if len(texts) == 0:
            return []

        results = []
        batch = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text is None or not str(text).strip():
                logger.warning("embed(): empty/None text at index %d will be embedded as-is.", i)
            cached = self._check_cache(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                batch.append(text)
                uncached_indices.append(i)

        if batch:
            for i in range(0, len(batch), self.batch_size):
                sub_batch = batch[i:i + self.batch_size]
                sub_indices = uncached_indices[i:i + self.batch_size]

                embeddings = self._embed_documents_raw(sub_batch)

                if len(embeddings) != len(sub_batch):
                    raise RuntimeError(
                        f"{self.provider_name}._embed_documents_raw returned "
                        f"{len(embeddings)} embeddings for {len(sub_batch)} texts; "
                        f"cannot safely assign results."
                    )

                for j, (text, embedding) in enumerate(zip(sub_batch, embeddings)):
                    self._save_to_cache(text, embedding)
                    results[sub_indices[j]] = embedding

        return results

    # LangChain Embeddings Interface
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain protocol: embed a list of documents (delegates to :meth:`embed`)."""
        from Trace.observability import observe_stage

        with observe_stage(
            "document_embedding",
            "embedding",
            provider=self.provider_name,
            model=self.model_name,
        ):
            return self.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """LangChain protocol: embed a single query (delegates to :meth:`embed_single`).

        Every VectorDB backend's similarity_search() calls into this method
        (it's the `embedding_function=` handed to e.g. Chroma) -- timing it
        here, once, captures query-embedding time centrally without touching
        any of the call sites in RetrieveMethods.py or the concrete VectorDB
        providers. See Embedding.timing.EmbeddingTimingTracker.
        """
        start = time.perf_counter()
        try:
            from Trace.observability import observe_stage

            with observe_stage(
                "query_embedding",
                "embedding",
                provider=self.provider_name,
                model=self.model_name,
            ):
                return self.embed_single(text)
        finally:
            self.timing.record((time.perf_counter() - start) * 1000.0)
