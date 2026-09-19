"""HTTP-backed custom reranker implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from Muffakir.exceptions import ConfigurationError, ProviderError
from .base import BaseReranker


class RemoteReranker(BaseReranker):
    """Rerank through a Cohere-compatible, index-based HTTP response."""

    def __init__(
        self,
        base_url: Optional[str],
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 30.0,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        endpoint = str(base_url or "").strip()
        if not endpoint:
            raise ConfigurationError("Custom reranker requires reranker_base_url.")
        if not endpoint.lower().startswith(("http://", "https://")):
            raise ConfigurationError(
                "Custom reranker URL must start with http:// or https://."
            )
        if timeout_seconds <= 0:
            raise ConfigurationError(
                "Custom reranker timeout must be greater than zero."
            )
        self.base_url = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.options = dict(options or {})

    @property
    def name(self) -> str:
        return "custom"

    def score(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        if not documents:
            return []
        try:
            import requests

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload: Dict[str, Any] = {
                **self.options,
                "query": query,
                "documents": [document.page_content for document in documents],
                "top_n": len(documents),
            }
            if self.model:
                payload["model"] = self.model
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            scored = self._parse_scores(response.json(), documents)
            return sorted(scored, key=lambda item: item[1], reverse=True)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Custom reranker request failed: {exc}") from exc

    @staticmethod
    def _parse_scores(
        data: Any, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        if not isinstance(data, dict):
            raise ProviderError("Custom reranker response must be a JSON object.")
        results = data.get("results")
        if isinstance(results, list):
            parsed: List[Tuple[Document, float]] = []
            seen = set()
            for item in results:
                if not isinstance(item, dict):
                    raise ProviderError(
                        "Custom reranker results must contain objects."
                    )
                index = item.get("index")
                score = item.get("relevance_score", item.get("score"))
                if not isinstance(index, int) or index < 0 or index >= len(documents):
                    raise ProviderError(
                        "Custom reranker returned an invalid document index."
                    )
                if index in seen or not isinstance(score, (int, float)):
                    raise ProviderError(
                        "Custom reranker returned invalid or duplicate scores."
                    )
                seen.add(index)
                parsed.append((documents[index], float(score)))
            if len(parsed) != len(documents):
                raise ProviderError(
                    "Custom reranker must return one score per document."
                )
            return parsed
        scores = data.get("scores")
        if isinstance(scores, list) and len(scores) == len(documents) and all(
            isinstance(score, (int, float)) for score in scores
        ):
            return [
                (document, float(score))
                for document, score in zip(documents, scores)
            ]
        raise ProviderError(
            "Custom reranker response requires 'results' with index/score entries "
            "or a numeric 'scores' array."
        )
