"""Tests for persistent ComposerUI workspace resolution."""

import json
from pathlib import Path

import pytest

from ComposerUI.backend import storage


def test_resolution_precedence(monkeypatch, tmp_path):
    settings_path = tmp_path / "config" / "settings.json"
    legacy_root = tmp_path / "legacy"
    default_root = tmp_path / "default"
    saved_root = tmp_path / "saved"
    env_root = tmp_path / "environment"

    legacy_run = legacy_root / "12345678-1234-5678-1234-567812345678"
    legacy_run.mkdir(parents=True)
    settings_path.parent.mkdir()
    settings_path.write_text(json.dumps({"runs_root": str(saved_root)}), encoding="utf-8")

    monkeypatch.setenv(storage.RUNS_ROOT_ENV, str(env_root))
    assert storage.resolve_runs_root(
        settings_path=settings_path,
        legacy_root=legacy_root,
        fallback_root=default_root,
    ) == (env_root.resolve(), "environment", True)

    monkeypatch.delenv(storage.RUNS_ROOT_ENV)
    assert storage.resolve_runs_root(
        settings_path=settings_path,
        legacy_root=legacy_root,
        fallback_root=default_root,
    ) == (saved_root.resolve(), "saved", False)

    settings_path.unlink()
    assert storage.resolve_runs_root(
        settings_path=settings_path,
        legacy_root=legacy_root,
        fallback_root=default_root,
    ) == (legacy_root.resolve(), "legacy", False)

    legacy_run.rmdir()
    assert storage.resolve_runs_root(
        settings_path=settings_path,
        legacy_root=legacy_root,
        fallback_root=default_root,
    ) == (default_root.resolve(), "default", False)


def test_legacy_gitkeep_is_not_user_data(tmp_path):
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    (legacy_root / ".gitkeep").write_text("placeholder", encoding="utf-8")
    assert storage.has_legacy_user_data(legacy_root) is False


def test_validate_runs_root_creates_directory_and_rejects_file(tmp_path):
    new_root = tmp_path / "nested" / "runs"
    assert storage.validate_runs_root(new_root) == new_root.resolve()
    assert new_root.is_dir()

    file_path = tmp_path / "not-a-directory"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(storage.StorageConfigurationError, match="points to a file"):
        storage.validate_runs_root(file_path)


def test_save_runs_root_is_persistent_and_remembers_previous(tmp_path):
    settings_path = tmp_path / "config" / "settings.json"
    previous = tmp_path / "old-runs"
    selected = tmp_path / "new-runs"

    storage.save_runs_root(selected.resolve(), previous.resolve(), settings_path=settings_path)
    saved = json.loads(settings_path.read_text(encoding="utf-8"))

    assert saved["runs_root"] == str(selected.resolve())
    assert saved["recent_roots"] == [str(previous.resolve())]
    assert storage.resolve_runs_root(
        settings_path=settings_path,
        legacy_root=tmp_path / "missing-legacy",
        fallback_root=tmp_path / "fallback",
    ) == (selected.resolve(), "saved", False)


def test_invalid_saved_path_falls_back_safely(monkeypatch, tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"runs_root": "bad\u0000path"}), encoding="utf-8")
    fallback = tmp_path / "fallback"
    monkeypatch.delenv(storage.RUNS_ROOT_ENV, raising=False)

    resolved = storage.resolve_runs_root(
        settings_path=settings_path,
        legacy_root=tmp_path / "missing-legacy",
        fallback_root=fallback,
    )

    assert resolved == (fallback.resolve(), "default", False)


def test_default_root_is_outside_installed_package():
    assert storage.default_runs_root() != storage.legacy_runs_root()
    assert storage.legacy_runs_root() not in storage.default_runs_root().parents
