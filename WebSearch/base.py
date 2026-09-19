from abc import ABC, abstractmethod
from .models import WebSearchResult


class BaseWebSearchProvider(ABC):
    """Abstract base for all web search / scraping providers."""

    @abstractmethod
    def search(self, query: str) -> WebSearchResult:
        """Run a web search for the given query and return normalized results."""
        ...
