"""Standalone script wrapper to execute the Multilingual ASR evaluation CLI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.cli import main  # noqa: E402

if __name__ == "__main__":
    # Forward CLI arguments, defaulting to `evaluate` if no subcommand is given.
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["evaluate", *argv]
    raise SystemExit(main(argv))
