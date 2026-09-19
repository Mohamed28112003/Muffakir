"""Persistent, cross-platform storage settings for ComposerUI runs."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from platformdirs import user_config_path, user_data_path


RUNS_ROOT_ENV = "MUFFAKIR_RUNS_ROOT"
SETTINGS_VERSION = 1
MAX_RECENT_ROOTS = 8


class StorageConfigurationError(ValueError):
    """Raised when a ComposerUI workspace cannot be used safely."""


def legacy_runs_root() -> Path:
    """Return the pre-0.3 package-local runs directory."""
    return Path(__file__).resolve().parent.parent / "runs"


def default_runs_root() -> Path:
    """Return the operating system's standard per-user Muffakir data path."""
    return user_data_path("Muffakir", appauthor=False) / "ComposerUI" / "runs"


def settings_file_path() -> Path:
    """Return the persistent ComposerUI settings file outside the runs root."""
    return user_config_path("Muffakir", appauthor=False) / "ComposerUI" / "settings.json"


def normalize_runs_root(value: Union[str, Path]) -> Path:
    """Expand a user-entered path and resolve it against the server cwd."""
    raw = str(value).strip()
    if not raw:
        raise StorageConfigurationError("The runs workspace path cannot be empty.")
    if "\x00" in raw:
        raise StorageConfigurationError("The runs workspace path contains invalid characters.")
    try:
        return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    except (OSError, ValueError) as exc:
        raise StorageConfigurationError(
            f"The runs workspace path is invalid: {raw}"
        ) from exc


def _read_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or settings_file_path()
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _is_run_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        uuid.UUID(path.name)
    except ValueError:
        return False
    return True


def has_legacy_user_data(path: Optional[Path] = None) -> bool:
    """Ignore the tracked .gitkeep but retain any real legacy user data."""
    root = path or legacy_runs_root()
    if not root.is_dir():
        return False
    try:
        children = list(root.iterdir())
    except OSError:
        return False
    if any(_is_run_directory(child) for child in children):
        return True
    datasets = root / "_datasets"
    try:
        return datasets.is_dir() and any(datasets.iterdir())
    except OSError:
        return False


def resolve_runs_root(
    *,
    settings_path: Optional[Path] = None,
    legacy_root: Optional[Path] = None,
    fallback_root: Optional[Path] = None,
) -> Tuple[Path, str, bool]:
    """Resolve root, source, and whether runtime UI changes are locked."""
    override = os.environ.get(RUNS_ROOT_ENV)
    if override:
        return normalize_runs_root(override), "environment", True

    settings = _read_settings(settings_path)
    saved_root = settings.get("runs_root")
    if isinstance(saved_root, str) and saved_root.strip():
        try:
            return normalize_runs_root(saved_root), "saved", False
        except StorageConfigurationError:
            # A manually edited/corrupt value must not prevent the UI from
            # starting; fall through to legacy/default discovery.
            pass

    candidate_legacy = legacy_root or legacy_runs_root()
    if has_legacy_user_data(candidate_legacy):
        return candidate_legacy.resolve(), "legacy", False

    return (fallback_root or default_runs_root()).resolve(), "default", False


def validate_runs_root(value: Union[str, Path]) -> Path:
    """Create the workspace when needed and verify it is a writable directory."""
    root = normalize_runs_root(value)
    if root.exists() and not root.is_dir():
        raise StorageConfigurationError(
            f"The runs workspace points to a file, not a directory: {root}"
        )
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=".muffakir-write-test-", dir=root, delete=True
        ) as probe:
            probe.write("ok")
            probe.flush()
    except OSError as exc:
        raise StorageConfigurationError(
            f"The runs workspace is not writable: {root}. {exc}"
        ) from exc
    return root


def recent_runs_roots(settings_path: Optional[Path] = None) -> List[Path]:
    """Read normalized recent roots, skipping malformed entries."""
    settings = _read_settings(settings_path)
    result: List[Path] = []
    for item in settings.get("recent_roots", []):
        if not isinstance(item, str) or not item.strip():
            continue
        try:
            normalized = normalize_runs_root(item)
        except StorageConfigurationError:
            continue
        if normalized not in result:
            result.append(normalized)
    return result


def _deduplicate_paths(paths: Iterable[Path]) -> List[Path]:
    result: List[Path] = []
    for path in paths:
        normalized = path.resolve()
        if normalized not in result:
            result.append(normalized)
    return result


def save_runs_root(
    new_root: Path,
    previous_root: Path,
    *,
    settings_path: Optional[Path] = None,
) -> None:
    """Atomically persist the selected root and bounded recent-root history."""
    target = settings_path or settings_file_path()
    existing_recent = recent_runs_roots(target)
    recent = _deduplicate_paths([previous_root, *existing_recent, new_root])
    recent = [path for path in recent if path != new_root][:MAX_RECENT_ROOTS]
    payload = {
        "version": SETTINGS_VERSION,
        "runs_root": str(new_root),
        "recent_roots": [str(path) for path in recent],
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, target)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise StorageConfigurationError(
            f"Could not save ComposerUI storage settings: {exc}"
        ) from exc
