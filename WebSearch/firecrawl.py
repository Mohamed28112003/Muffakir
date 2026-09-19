import logging
from typing import Any, Dict, List

from .base import BaseWebSearchProvider
from .models import WebSearchResult

logger = logging.getLogger(__name__)


class FirecrawlWebSearchProvider(BaseWebSearchProvider):
    """
    Firecrawl deep-research provider.
    Uses FirecrawlApp.deep_research to gather web content and sources.
    """

    def __init__(
        self,
        api_key: str,
        max_depth: int = 2,
        time_limit: int = 30,
        max_urls: int = 5,
        **kwargs: Any,
    ):
        if not api_key:
            raise ValueError("api_key is required for FirecrawlWebSearchProvider.")

        self.api_key = api_key
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.max_urls = max_urls
        self.logger = logging.getLogger(__name__)

        try:
            from firecrawl import FirecrawlApp
        except ImportError as e:
            raise ImportError(
                "The 'firecrawl-py' package is required to use FirecrawlWebSearchProvider. "
                "Please install it using: pip install firecrawl-py"
            ) from e

        self.firecrawl = FirecrawlApp(api_key=self.api_key)

    def _parse_sources(self, raw_result: Dict[str, Any]) -> List[Dict[str, str]]:
        raw = raw_result.get("data", {}).get("sources", [])
        sources: List[Dict[str, str]] = []
        for s in raw:
            title = (s.get("title") or "").strip()
            url = (s.get("url") or "").strip()
            if title and url:
                sources.append({"title": title, "url": url})
        return sources

    def search(self, query: str) -> WebSearchResult:
        try:
            raw = self.firecrawl.deep_research(
                query=query,
                max_depth=self.max_depth,
                time_limit=self.time_limit,
                max_urls=self.max_urls,
            )
        except Exception as e:
            self.logger.error(f"Firecrawl deep_research failed: {e}", exc_info=True)
            raise

        if not isinstance(raw, dict):
            raw = {}

        content = raw.get("data", {}).get("finalAnalysis", "") or ""
        sources = self._parse_sources(raw)
        self.logger.debug(f"Firecrawl returned {len(sources)} sources.")
        return WebSearchResult(content=content, sources=sources)
