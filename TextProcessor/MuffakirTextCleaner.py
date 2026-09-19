import logging
import re
from typing import Callable, List, Dict, Optional

logger = logging.getLogger(__name__)


class MuffakirTextCleaner:
    """
    Language-aware text cleaner that supports extensible custom steps.
    Developers can inject domain-specific cleaning logic via `add_cleaning_step()`.

    Example:
        cleaner = MuffakirTextCleaner(language="ar")
        cleaner.add_cleaning_step("strip_emails", lambda text: re.sub(r'\\S+@\\S+', '', text))
        clean_text = cleaner.clean("Raw text with noise")
    """

    def __init__(self, language: str = "auto"):
        """
        Initialize the text cleaner.

        Args:
            language: 'ar', 'en', 'auto', or 'none'.
                - 'auto': automatically detects per-document language.
                - 'none': passes through raw text (useful for AAS comparison).
        """
        self.language = language
        self._custom_steps: List[Dict[str, Callable[[str], str]]] = []

        # Map language choice to built-in strategy
        self._base_strategies = {
            "ar": self._arabic_clean,
            "en": self._english_clean,
            "auto": self._auto_detect_and_clean,
            "none": lambda t: t,
        }

        if self.language not in self._base_strategies:
            available = ", ".join(self._base_strategies.keys())
            raise ValueError(f"Unknown language '{self.language}'. Available: {available}")

    def add_cleaning_step(self, name: str, func: Callable[[str], str]) -> None:
        """
        Add a custom text cleaning function to the pipeline.
        Custom steps are executed IN ORDER, after the base language cleaning.

        Args:
            name: Identifier for the step (to allow removal/listing).
            func: A callable that takes a string and returns a string.
        """
        # Remove if exists to prevent duplicates
        self.remove_step(name)
        self._custom_steps.append({"name": name, "func": func})

    def remove_step(self, name: str) -> None:
        """Remove a previously added custom cleaning step by name."""
        self._custom_steps = [step for step in self._custom_steps if step["name"] != name]

    def list_steps(self) -> None:
        """Log the current sequence of cleaning steps."""
        logger.info("Cleaning steps: 1. Base language cleaner: %s", self.language)
        for i, step in enumerate(self._custom_steps, start=2):
            logger.info("  %d. Custom step: %s", i, step["name"])

    def clean(self, text: str) -> str:
        """
        Apply the full cleaning pipeline (base strategy + all custom steps).
        """
        if not text:
            return ""

        # 1. Strip common global artifacts first (e.g. standalone page numbers)
        if self.language != "none":
            text = re.sub(r'-\s*\d+\s*-', '', text)

        # 2. Base language strategy
        text = self._base_strategies[self.language](text)

        # 3. Custom steps
        for step in self._custom_steps:
            try:
                text = step["func"](text)
            except Exception as e:
                logger.warning("Error in custom cleaning step '%s': %s", step["name"], e)

        # 4. Normalize whitespace (unless raw mode)
        if self.language != "none":
            text = re.sub(r'\s+', ' ', text).strip()

        return text

    def get_search_space(self) -> dict:
        """
        Return the hyperparameter search space for Automated Architecture Search (AAS).
        """
        return {
            "text_cleaner_language": ["ar", "en", "auto", "none"]
        }

    # --- Built-in strategies ---

    @staticmethod
    def _arabic_clean(text: str) -> str:
        """
        Arabic mode: keep Arabic unicode block, Latin chars (for mixed docs),
        digits and common punctuation — NEVER strip Latin entirely.
        """
        return re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF'
                      r'a-zA-Z0-9\s\.\,\!\?\:\;\-\(\)\"\'\'\\u060C\u061B\u061F]',
                      '', text)

    @staticmethod
    def _english_clean(text: str) -> str:
        """English mode: keep Latin, digits, and common punctuation."""
        return re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\:\;\-\(\)\"\'\u2018\u2019\u201C\u201D]', '', text)

    def _auto_detect_and_clean(self, text: str) -> str:
        """Heuristically detect language and apply correct cleaner."""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))

        if arabic_chars >= latin_chars:
            return self._arabic_clean(text)
        return self._english_clean(text)
