"""
ReportGenerator - Generate structured JSON reports from architecture search results.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .results.report import ComposerReport

logger = logging.getLogger(__name__)


def _fmt_metric(value, precision: int = 4) -> str:
    """Format a metric value for display, returning 'N/A' for None."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_named_metric(name, value, precision: int = 4) -> str:
    """Format ratings on their native scale and other metrics as decimals."""
    if name == "llm_judge_rating" and value is not None:
        try:
            rendered = f"{float(value):.2f}".rstrip("0").rstrip(".")
            return f"{rendered} / 5"
        except (TypeError, ValueError):
            return str(value)
    return _fmt_metric(value, precision)


def _metric_label(name) -> str:
    return "LLM Judge Rating" if name == "llm_judge_rating" else str(name)


def _display_config(trial) -> Dict[str, Any]:
    """Add resolved local-reranker details to the compact trial config."""
    config = dict(trial.config)
    resolved = trial.resolved_config or {}
    for key in ("llm_parameters", "judge_llm_parameters", "query_transform_llm_parameters", "reranker_llm_parameters", "dataset_llm_parameters"):
        if resolved.get(key) is not None:
            config.setdefault(key, resolved[key])
    method = str(resolved.get("reranking_method") or "").strip().lower()
    if resolved.get("reranking") and method:
        config.setdefault("reranking_method", method)
    if method in {"cross_encoder", "pointwise"} and resolved.get("reranking_model"):
        config.setdefault("reranking_model", resolved["reranking_model"])
    return config


class ReportGenerator:
    """
    Generates reports from ComposerReport data.

    Supports:
    - JSON export for programmatic access and ComposerUI integration.
    """

    @staticmethod
    def generate_json(report: ComposerReport, output_path: str) -> None:
        """
        Generate a JSON report.

        Args:
            report: ComposerReport with trial results
            output_path: File path to save JSON report
        """
        report.to_json(output_path)
        logger.info(f"JSON report saved to: {output_path}")

    @staticmethod
    def generate_html(report: ComposerReport, output_path: str) -> None:
        """
        Deprecated: HTML report generation has been removed.

        Saves a structured JSON report to the equivalent path instead.
        """
        logger.warning(
            "ReportGenerator.generate_html is deprecated and removed; saving JSON report instead."
        )
        json_path = str(Path(output_path).with_suffix(".json"))
        ReportGenerator.generate_json(report, json_path)
