"""
Tests for the `muffakir` CLI (Muffakir/cli.py).
"""

import os

import pytest

from Muffakir import cli


def test_serve_defaults():
    args = cli._build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 2811
    assert args.reload is False
    assert args.open_browser is False
    assert args.log_level == "info"
    assert args.runs_dir is None


def test_serve_custom_options():
    args = cli._build_parser().parse_args(
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "5000",
            "--reload",
            "--open",
            "--log-level",
            "debug",
            "--runs-dir",
            "/tmp/muffakir-runs",
        ]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 5000
    assert args.reload is True
    assert args.open_browser is True
    assert args.log_level == "debug"
    assert args.runs_dir == "/tmp/muffakir-runs"


def test_serve_no_open_explicit():
    args = cli._build_parser().parse_args(["serve", "--no-open"])
    assert args.open_browser is False


def test_serve_dispatches_to_run_uvicorn(monkeypatch):
    captured = {}

    def _fake_run_uvicorn(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "_run_uvicorn", _fake_run_uvicorn)
    monkeypatch.delenv("MUFFAKIR_RUNS_ROOT", raising=False)

    cli.main(["serve", "--port", "5000", "--host", "0.0.0.0", "--log-level", "warning"])

    assert captured == {
        "host": "0.0.0.0",
        "port": 5000,
        "reload": False,
        "log_level": "warning",
    }


def test_serve_runs_dir_sets_env_var(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_run_uvicorn", lambda **kwargs: None)
    monkeypatch.delenv("MUFFAKIR_RUNS_ROOT", raising=False)

    runs_dir = tmp_path / "custom-runs"
    cli.main(["serve", "--runs-dir", str(runs_dir)])

    assert os.environ["MUFFAKIR_RUNS_ROOT"] == str(runs_dir.resolve())


def test_open_browser_schedules_timer(monkeypatch):
    monkeypatch.setattr(cli, "_run_uvicorn", lambda **kwargs: None)

    scheduled = {}

    class _FakeTimer:
        def __init__(self, delay, func, args=()):
            scheduled["delay"] = delay
            scheduled["func"] = func
            scheduled["args"] = args

        def start(self):
            scheduled["started"] = True

    monkeypatch.setattr(cli.threading, "Timer", _FakeTimer)

    cli.main(["serve", "--port", "1234", "--open"])

    assert scheduled["started"] is True
    assert scheduled["args"] == ("http://127.0.0.1:1234",)
    assert scheduled["func"] is cli.webbrowser.open


def test_no_open_does_not_schedule_timer(monkeypatch):
    monkeypatch.setattr(cli, "_run_uvicorn", lambda **kwargs: None)

    def _boom(*args, **kwargs):
        raise AssertionError("Timer should not be scheduled when --open is not passed")

    monkeypatch.setattr(cli.threading, "Timer", _boom)

    cli.main(["serve"])  # open_browser defaults to False


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0
    assert "muffakir" in capsys.readouterr().out


def test_no_command_prints_help_and_exits(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code == 1
    assert "usage" in capsys.readouterr().out.lower()
