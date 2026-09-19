"""
Tests for ComposerUI structured JSON logging configuration.
"""

import json
import logging
import sys

from ComposerUI.backend.logging_config import JsonFormatter, configure_logging


def test_json_formatter_basic():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="ComposerUI.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="test message %s",
        args=("hello",),
        exc_info=None,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)
    assert data["level"] == "INFO"
    assert data["logger"] == "ComposerUI.test"
    assert data["message"] == "test message hello"
    assert "ts" in data
    assert "run_id" not in data


def test_json_formatter_with_run_id_extra():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="ComposerUI.run_manager",
        level=logging.ERROR,
        pathname=__file__,
        lineno=25,
        msg="run execution failed",
        args=(),
        exc_info=None,
    )
    record.run_id = "test-run-uuid-1234"
    formatted = formatter.format(record)
    data = json.loads(formatted)
    assert data["level"] == "ERROR"
    assert data["run_id"] == "test-run-uuid-1234"
    assert data["message"] == "run execution failed"


def test_json_formatter_with_exception():
    formatter = JsonFormatter()
    try:
        raise ValueError("synthetic error for logging test")
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="ComposerUI.crash",
        level=logging.ERROR,
        pathname=__file__,
        lineno=45,
        msg="unhandled exception",
        args=(),
        exc_info=exc_info,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)
    assert "exc_info" in data
    assert "synthetic error for logging test" in data["exc_info"]


def test_configure_logging_idempotent():
    logger = logging.getLogger("ComposerUI")
    logger.handlers.clear()

    configure_logging(level=logging.DEBUG)
    handler_count = len(logger.handlers)
    assert handler_count >= 1

    # Second call should not add duplicate handlers
    configure_logging(level=logging.DEBUG)
    assert len(logger.handlers) == handler_count
