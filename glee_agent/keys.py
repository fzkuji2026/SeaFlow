"""Load API keys from the environment, .env, or OpenProgram config."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> None:
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_openprogram_keys() -> None:
    path = Path.home() / ".openprogram" / "config.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    keys = data.get("api_keys")
    if not isinstance(keys, dict):
        return
    for key, value in keys.items():
        if isinstance(value, str) and value:
            os.environ.setdefault(str(key), value)


def load_all() -> None:
    selected = os.environ.get("GLEE_ENV_FILE")
    load_dotenv(Path(selected).expanduser() if selected else None)
    load_openprogram_keys()
