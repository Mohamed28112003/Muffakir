"""
Muffakir command-line interface.

Usage:
    muffakir serve --port 5000
"""

import argparse
import os
import sys
import threading
import webbrowser
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from Muffakir.constants import DEFAULT_UI_HOST, DEFAULT_UI_PORT


def _version() -> str:
    try:
        return version("Muffakir")
    except PackageNotFoundError:
        return "unknown"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="muffakir", description="Muffakir multilingual RAG CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")

    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the ComposerUI web server")
    serve.add_argument("--host", default=DEFAULT_UI_HOST, help=f"Bind host (default: {DEFAULT_UI_HOST})")
    serve.add_argument("--port", type=int, default=DEFAULT_UI_PORT, help=f"Bind port (default: {DEFAULT_UI_PORT})")
    serve.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (dev only)")
    serve.add_argument(
        "--open", dest="open_browser", action="store_true", help="Open the UI in a browser tab on startup"
    )
    serve.add_argument(
        "--no-open", dest="open_browser", action="store_false", help="Do not open a browser tab (default)"
    )
    serve.set_defaults(open_browser=False)
    serve.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: info)",
    )
    serve.add_argument(
        "--runs-dir",
        default=None,
        help=(
            "Lock run storage to this directory for the server session "
            "(default: saved UI workspace or the OS user-data directory)"
        ),
    )

    return parser


def _run_uvicorn(**kwargs) -> None:
    from Muffakir.optional_dependencies import require_optional_dependency

    require_optional_dependency("ui")
    import uvicorn

    uvicorn.run("ComposerUI.backend.app:app", **kwargs)


def _maybe_open_browser(url: str, open_browser: bool) -> None:
    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()


def _serve(args: argparse.Namespace) -> None:
    if args.runs_dir:
        os.environ["MUFFAKIR_RUNS_ROOT"] = str(Path(args.runs_dir).resolve())

    url = f"http://{args.host}:{args.port}"
    print(f"Starting ComposerUI at {url}")

    _maybe_open_browser(url, args.open_browser)

    _run_uvicorn(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        _serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
