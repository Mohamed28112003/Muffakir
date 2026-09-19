from typing import Any

from .base import BaseWebSearchProvider


def create_web_search_provider(provider: str = "firecrawl", **kwargs: Any) -> BaseWebSearchProvider:
    """
    Factory function to instantiate pluggable Web Search providers.

    Args:
        provider (str): 'firecrawl', 'tavily', or 'serpapi'.
        **kwargs: Provider-specific configuration (api_key, max_results, max_depth, etc.).

    Returns:
        BaseWebSearchProvider: An instance inheriting from BaseWebSearchProvider.
    """
    provider_name = str(provider).lower().strip()

    if provider_name in ("firecrawl", "fire_crawl", "fire-crawl"):
        from Muffakir.optional_dependencies import require_optional_dependency
        require_optional_dependency("firecrawl")
        from .firecrawl import FirecrawlWebSearchProvider
        return FirecrawlWebSearchProvider(**kwargs)

    if provider_name in ("tavily",):
        from Muffakir.optional_dependencies import require_optional_dependency
        require_optional_dependency("tavily")
        from .tavily import TavilyWebSearchProvider
        return TavilyWebSearchProvider(**kwargs)

    if provider_name in ("serpapi", "serp_api", "serp-api", "google"):
        from Muffakir.optional_dependencies import require_optional_dependency
        require_optional_dependency("serpapi")
        from .serpapi import SerpAPIWebSearchProvider
        return SerpAPIWebSearchProvider(**kwargs)

    raise ValueError(
        f"Unknown web search provider: '{provider}'. "
        "Available options: 'firecrawl', 'tavily', 'serpapi'."
    )
