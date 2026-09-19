"""
Launcher script for ComposerUI server.

Usage:
    python -m ComposerUI.run_server
    python ComposerUI/run_server.py [PORT] [--runs-dir PATH]
"""

import argparse
import os
from pathlib import Path
from Muffakir.constants import DEFAULT_UI_HOST, DEFAULT_UI_PORT


def main():
    from Muffakir.optional_dependencies import require_optional_dependency

    require_optional_dependency("ui")
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the ComposerUI web server")
    parser.add_argument("port", nargs="?", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--host", default=DEFAULT_UI_HOST)
    parser.add_argument(
        "--runs-dir",
        help=(
            "Lock run storage to this directory for the server session "
            "(default: saved UI workspace or the OS user-data directory)"
        ),
    )
    args = parser.parse_args()
    if args.runs_dir:
        os.environ["MUFFAKIR_RUNS_ROOT"] = str(Path(args.runs_dir).resolve())
    print(f"Starting ComposerUI at http://{args.host}:{args.port}")
    uvicorn.run(
        "ComposerUI.backend.app:app", host=args.host, port=args.port, reload=False
    )


if __name__ == "__main__":
    main()
