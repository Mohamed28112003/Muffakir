"""
Structured JSON logging configuration for ComposerUI.
"""

import json
import logging
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Formats log records as compact JSON strings with optional run_id and exc_info."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        run_id = getattr(record, "run_id", None)
        if run_id is not None:
            payload["run_id"] = run_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure ComposerUI namespace logger with JSON formatting. Idempotent."""
    logger = logging.getLogger("ComposerUI")
    if logger.handlers:
        return  # idempotent — avoid duplicate handlers on repeated calls
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
