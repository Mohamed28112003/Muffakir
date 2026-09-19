import logging
import os
from typing import Any, Dict, List, Optional

from .base import BaseWebSearchProvider
from .models import WebSearchResult

logger = logging.getLogger(__name__)


class SerpAPIWebSearchProvider(BaseWebSearchProvider):
    """
    SerpAPI Google search provider.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        **kwargs: Any,
    ):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "api_key is required for SerpAPIWebSearchProvider "
                "(pass api_key or set SERPAPI_API_KEY)."
            )

        self.max_results = max_results
        self.logger = logging.getLogger(__name__)

        try:
            from langchain_community.utilities import SerpAPIWrapper
        except ImportError as e:
            raise ImportError(
                "langchain-community and google-search-results are required to use SerpAPIWebSearchProvider. "
                "Please install them using: pip install google-search-results langchain-community"
            ) from e

        try:
            self._wrapper = SerpAPIWrapper(serpapi_api_key=self.api_key)
        except TypeError:
            # Older langchain-community wrapper fallback
            self._wrapper = SerpAPIWrapper()

    def _normalize_dict_results(self, data: Dict[str, Any]) -> WebSearchResult:
        sources: List[Dict[str, str]] = []
        content_parts: List[str] = []

        organic = data.get("organic_results") or data.get("organic") or []
        for item in organic[: self.max_results]:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            url = (item.get("link") or item.get("url") or "").strip()
            snippet = (item.get("snippet") or item.get("content") or "").strip()

            if snippet:
                label = title or url or "result"
                content_parts.append(f"{label}\n{snippet}")
            if title and url:
                sources.append({"title": title, "url": url})
            elif url:
                sources.append({"title": url, "url": url})

        answer_box = data.get("answer_box") or {}
        if isinstance(answer_box, dict):
            answer = (
                answer_box.get("answer")
                or answer_box.get("snippet")
                or answer_box.get("title")
                or ""
            ).strip()
            if answer:
                content_parts.insert(0, answer)

        return WebSearchResult(
            content="\n\n".join(content_parts).strip(),
            sources=sources,
        )

    def search(self, query: str) -> WebSearchResult:
        try:
            # Prefer structured results when available
            if hasattr(self._wrapper, "results"):
                raw = self._wrapper.results(query)
                if isinstance(raw, dict):
                    result = self._normalize_dict_results(raw)
                    self.logger.debug(f"SerpAPI returned {len(result.sources)} sources.")
                    return result

            text = self._wrapper.run(query)
            return WebSearchResult(content=str(text).strip(), sources=[])
        except Exception as e:
            self.logger.error(f"SerpAPI search failed: {e}", exc_info=True)
            raise
