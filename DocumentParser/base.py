import os
from abc import ABC, abstractmethod
from typing import List, Optional
from .models import ParsedDocument


class BaseDocumentParser(ABC):
    """Abstract base for all document parsing/OCR providers.

    Contract for concrete providers:

    - ``parse_file(file_path)`` must raise ``FileNotFoundError`` if the path
      does not exist, and return a fully-populated ``ParsedDocument`` otherwise.
    - ``parse_directory(directory_path)`` must return a list of successfully
      parsed ``ParsedDocument`` objects. Per-file failures must be logged and
      summarized at the end of the batch (the list must still be returned so
      partial progress is not lost).
    - ``supported_extensions()`` returns lowercase, dot-prefixed extensions
      (e.g. ``[".pdf", ".txt"]``).
    - The shared ``ParsedDocument.text`` contract joins pages/sections with
      ``"\\n\\n"``; concrete providers must comply.
    - Both methods accept an optional ``base_dir`` keyword: when given, the
      resolved path must stay within it or a ``ConfigurationError`` is raised
      (path-traversal containment). Defaults to ``None`` (no containment
      check), preserving existing behavior for callers that don't opt in —
      intended for callers that expose these parsers over an API boundary.

    Library note: concrete providers must NEVER call ``logging.basicConfig`` —
    that mutates the host application's root logger. Use ``getLogger(__name__)``
    only.
    """

    @abstractmethod
    def parse_file(self, file_path: str) -> ParsedDocument:
        """Parse a single file and return a :class:`ParsedDocument`.

        Raises:
            FileNotFoundError: if ``file_path`` does not exist.
        """
        ...

    @abstractmethod
    def parse_directory(self, directory_path: str) -> List[ParsedDocument]:
        """Parse all supported files in a directory.

        Returns the list of successfully parsed documents; per-file failures
        are logged and summarized but do not abort the batch.
        """
        ...

    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of (lowercase, dot-prefixed) file extensions this parser handles."""
        ...

    def supports(self, file_path: str) -> bool:
        """Return True if ``file_path`` has an extension this parser supports.

        Centralizes the extension check so concrete providers do not reimplement
        it in ``parse_directory``.
        """
        return os.path.splitext(file_path)[1].lower() in self.supported_extensions()

    def _resolve_within_base(self, path: str, base_dir: Optional[str]) -> str:
        """Resolve *path*; if ``base_dir`` is given, raise ``ConfigurationError``
        when the resolved path escapes it (path-traversal containment).

        Returns *path* unchanged when ``base_dir`` is None, preserving existing
        behavior for callers that don't opt in.
        """
        if base_dir is None:
            return path
        resolved_base = os.path.realpath(base_dir)
        resolved_path = os.path.realpath(path)
        try:
            within_base = os.path.commonpath([resolved_base, resolved_path]) == resolved_base
        except ValueError:
            # Different drives on Windows can't share a common path — never within base.
            within_base = False
        if not within_base:
            from Muffakir.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Path '{path}' escapes the allowed base directory '{base_dir}'."
            )
        return resolved_path
