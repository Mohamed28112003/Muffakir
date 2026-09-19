import logging
import os
from typing import Any, Dict, List, Optional

from .base import BaseWebSearchProvider
from .models import WebSearchResult

logger = logging.getLogger(__name__)


class TavilyWebSearchProvider(BaseWebSearchProvider):
    """
    Tavily AI-optimized web search provider.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        **kwargs: Any,
    ):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "api_key is required for TavilyWebSearchProvider "
                "(pass api_key or set TAVILY_API_KEY)."
            )

        self.max_results = max_results
        self.logger = logging.getLogger(__name__)

        try:
            from langchain_tavily import TavilySearchResults
        except ImportError as e:
            raise ImportError(
                "The 'langchain-tavily' package is required to use TavilyWebSearchProvider. "
                "Please install it using: pip install langchain-tavily"
            ) from e

        try:
            self._tool = TavilySearchResults(
                tavily_api_key=self.api_key,
                max_results=self.max_results,
            )
        except TypeError:
            # Fallback if specific version only expects max_results or api_key parameter name
            self._tool = TavilySearchResults(max_results=self.max_results)

    def _normalize(self, results: Any) -> WebSearchResult:
        sources: List[Dict[str, str]] = []
        content_parts: List[str] = []

        if isinstance(results, str):
            return WebSearchResult(content=results, sources=[])

        items = results if isinstance(results, list) else [results]
        for item in items:
            if not isinstance(item, dict):
                content_parts.append(str(item))
                continue

            title = (item.get("title") or "").strip()
            url = (item.get("url") or item.get("link") or "").strip()
            snippet = (
                item.get("content")
                or item.get("snippet")
                or item.get("raw_content")
                or ""
            ).strip()

            if snippet:
                label = title or url or "result"
                content_parts.append(f"{label}\n{snippet}")
            if title and url:
                sources.append({"title": title, "url": url})
            elif url:
                sources.append({"title": url, "url": url})

        return WebSearchResult(
            content="\n\n".join(content_parts).strip(),
            sources=sources,
        )

    def search(self, query: str) -> WebSearchResult:
        try:
            raw = self._tool.invoke({"query": query})
        except Exception as e:
            self.logger.error(f"Tavily search failed: {e}", exc_info=True)
            raise

        result = self._normalize(raw)
        self.logger.debug(f"Tavily returned {len(result.sources)} sources.")
        return result
