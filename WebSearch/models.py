from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class WebSearchResult:
    """
    Standardized output from any web search provider.
    This is the SINGLE contract between search providers and the Search orchestrator.
    """
    content: str
    sources: List[Dict[str, str]] = field(default_factory=list)
