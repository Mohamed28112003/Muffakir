"""
Trace reader functions for ComposerUI.

Performs file-based reading of manifest.json, trials.jsonl, and error.txt sidecar.
Zero dependencies on FastAPI or MuffakirComposer.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STAGE_TIMING_FIELDS = {
    "query_transform": "mean_query_transform_ms",
    "query_embedding": "mean_query_embedding_ms",
    "vector_search": "mean_vector_search_ms",
    "rerank": "mean_rerank_ms",
    "relevance_check": "mean_relevance_check_ms",
    "web_search": "mean_web_search_ms",
    "generation": "mean_generation_ms",
    "hallucination_check": "mean_hallucination_check_ms",
}


def stage_timings_from_record(record: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Extract the normalized stage map from one trial record."""
    record = record or {}
    return {
        stage_key: record.get(timing_field)
        for stage_key, timing_field in STAGE_TIMING_FIELDS.items()
    }


def read_manifest(trace_dir: Path) -> Optional[Dict[str, Any]]:
    """Read manifest.json from trace_dir.

    Returns None if not yet created. Retries once on JSONDecodeError to guard
    against rare atomic replace collisions.
    """
    manifest_path = trace_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    for attempt in range(2):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            if attempt == 0:
                time.sleep(0.05)
                continue
            logger.warning(f"Failed to decode manifest JSON at {manifest_path}")
            return None
        except Exception as e:
            logger.warning(f"Error reading manifest at {manifest_path}: {e}")
            return None
    return None


def read_active_trials(trace_dir: Path) -> List[Dict[str, Any]]:
    """Read the writer-owned snapshot of trials currently occupying workers."""
    snapshot_path = trace_dir / "active_trials.json"
    if not snapshot_path.exists():
        return []

    for attempt in range(2):
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            rows = payload.get("trials", []) if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                return []
            valid_rows = [row for row in rows if isinstance(row, dict)]
            valid_rows.sort(key=lambda row: row.get("trial_id", 0))
            return valid_rows
        except json.JSONDecodeError:
            if attempt == 0:
                time.sleep(0.05)
                continue
            logger.warning(f"Failed to decode active trial snapshot at {snapshot_path}")
        except Exception as e:
            logger.warning(f"Error reading active trial snapshot at {snapshot_path}: {e}")
            break
    return []


def read_trials_after(trace_dir: Path, after: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    """Read trials from trials.jsonl after the given line count offset.

    Returns (new_records, new_line_count). Incomplete or malformed trailing lines
    (e.g., during active flush) are silently ignored and deferred to subsequent reads.
    """
    trials_path = trace_dir / "trials.jsonl"
    if not trials_path.exists():
        return [], after

    new_records: List[Dict[str, Any]] = []
    current_line_idx = 0

    try:
        with open(trials_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue

                if current_line_idx < after:
                    current_line_idx += 1
                    continue

                try:
                    record = json.loads(line_str)
                    new_records.append(record)
                    current_line_idx += 1
                except json.JSONDecodeError:
                    # Trailing partial line being written concurrently; skip and stop
                    break
    except Exception as e:
        logger.warning(f"Error reading trials at {trials_path}: {e}")

    return new_records, current_line_idx


def count_trial_lines(trace_dir: Path) -> int:
    """Quickly count non-empty lines in trials.jsonl without parsing JSON."""
    trials_path = trace_dir / "trials.jsonl"
    if not trials_path.exists():
        return 0

    count = 0
    try:
        with open(trials_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception as e:
        logger.warning(f"Error counting trial lines at {trials_path}: {e}")
    return count


def find_trial_record(trace_dir: Path, trial_id: int) -> Optional[Dict[str, Any]]:
    """Return one trial record by ID, tolerating a concurrently written tail."""
    records, _ = read_trials_after(trace_dir, after=0)
    for record in records:
        if record.get("trial_id") == trial_id:
            return record
    return None


def read_samples_for_trial(trace_dir: Path, trial_id: int) -> List[Dict[str, Any]]:
    """Read all currently available sample traces for one trial."""
    samples_path = trace_dir / "samples.jsonl"
    if not samples_path.exists():
        return []

    records: List[Dict[str, Any]] = []
    try:
        with open(samples_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                except json.JSONDecodeError:
                    # The writer may be flushing the final line. It will be
                    # available on the next inspector request.
                    break
                if record.get("trial_id") == trial_id:
                    records.append(record)
    except Exception as e:
        logger.warning(f"Error reading samples at {samples_path}: {e}")
        return []

    records.sort(key=lambda r: r.get("sample_index", 0))
    return records


def read_error_rates(trace_dir: Path) -> Optional[Dict[str, Any]]:
    """Read the retained aggregate error-rate snapshot for a traced run."""
    path = trace_dir / "error_rates.json"
    if not path.exists():
        return None
    for attempt in range(2):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            if attempt == 0:
                time.sleep(0.05)
                continue
            logger.warning(f"Failed to decode error-rate JSON at {path}")
        except PermissionError:
            if attempt == 0:
                time.sleep(0.05)
                continue
            logger.warning(f"Error-rate snapshot was temporarily locked at {path}")
        except Exception as e:
            logger.warning(f"Error reading error rates at {path}: {e}")
            break
    return None


def read_errors_after(trace_dir: Path, after: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    """Read sanitized error events after a JSONL line-count cursor."""
    path = trace_dir / "errors.jsonl"
    if not path.exists():
        return [], after

    records: List[Dict[str, Any]] = []
    current_line_idx = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if not value:
                    continue
                if current_line_idx < after:
                    current_line_idx += 1
                    continue
                try:
                    record = json.loads(value)
                except json.JSONDecodeError:
                    break
                if isinstance(record, dict):
                    records.append(record)
                current_line_idx += 1
    except Exception as e:
        logger.warning(f"Error reading error events at {path}: {e}")
    return records, current_line_idx


def cluster_failures_from_records(records: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Cluster non-successful trials by error code or type."""
    clusters: Dict[str, List[int]] = {}
    for rec in records:
        status = rec.get("status", "success")
        if status != "success":
            key = rec.get("error_code") or rec.get("error_type") or "unknown"
            clusters.setdefault(key, []).append(rec.get("trial_id", -1))
    return clusters


def has_zero_evaluation_samples(records: List[Dict[str, Any]]) -> bool:
    """Identify legacy trial searches that ran with no evaluation samples.

    Older trial records may lack total_samples entirely; do not reclassify
    those runs without explicit zero-sample evidence from every recorded trial.
    """
    return bool(records) and all(
        record.get("total_samples") == 0 for record in records
    )


def aggregate_stage_timings(records: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Compute mean of per-stage timings across the provided trial records."""
    result: Dict[str, Optional[float]] = {}
    for stage_key, timing_field in STAGE_TIMING_FIELDS.items():
        vals = [
            rec[timing_field]
            for rec in records
            if rec.get(timing_field) is not None
        ]
        if vals:
            result[stage_key] = round(sum(vals) / len(vals), 2)
        else:
            result[stage_key] = None
    return result


def total_web_search_fallbacks(records: List[Dict[str, Any]]) -> int:
    """Sum web-search fallback counts across all trial records in a run."""
    return sum(r.get("web_search_fallback_count", 0) or 0 for r in records)


def read_error_message(run_dir: Path) -> Optional[str]:
    """Read error.txt sidecar file if present."""
    error_path = run_dir / "error.txt"
    if error_path.exists():
        try:
            return error_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"Failed to read error file at {error_path}: {e}")
            return None
    return None
