#!/usr/bin/env python3
"""Regression gate before a strategy commit or restart."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
    ]
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = "vendor:."
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
