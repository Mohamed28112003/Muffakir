from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Represents a single page of parsed content."""
    page_number: int
    text: str
    confidence: Optional[float] = None  # OCR confidence score if available


@dataclass
class ParsedDocument:
    """
    Standardized output from any document parser.
    This is the SINGLE contract between parsing and the rest of the SDK.

    Text contract: ``text`` is the concatenation of all page/section texts
    joined by ``"\\n\\n"``. Every concrete parser must comply with this so
    downstream chunking sees consistent whitespace regardless of provider.
    """
    text: str                          # Full extracted text (pages joined by "\n\n")
    source_path: str                   # Original file path
    parser_name: str                   # e.g. "azure", "docling", "llama_parse"
    pages: List[PageContent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Warn (do not raise) when the parsed text is empty — usually indicates a parse problem."""
        if not self.text or not self.text.strip():
            logger.warning(
                "ParsedDocument from '%s' (parser=%s) contains empty text.",
                self.source_path,
                self.parser_name,
            )
