"""Small cross-process, per-API-key request budget."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable


DEFAULT_REQUESTS_PER_MINUTE = 60
WINDOW_SECONDS = 60.0


def request_priority(method: str, url: str) -> int:
    """Return 0 for high priority and larger values for less urgent work."""
    path = url.casefold()
    if method.upper() == "POST" and "/move" in path:
        return 0
    if "/history" in path or "/stats" in path:
        return 2
    if "/queue" in path or "/games/" in path and "/move" not in path:
        return 1
    return 2


def _limit() -> int:
    for name in ("GLEE_REQUESTS_PER_MINUTE", "GLEE_RATE_LIMIT_PER_MINUTE"):
        raw = os.environ.get(name)
        if raw is not None:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    return DEFAULT_REQUESTS_PER_MINUTE


class KeyRequestBudget:
    """A sliding-window budget shared by processes using the same key."""

    def __init__(
        self,
        api_key: str,
        *,
        root: Path | None = None,
        limit: int | None = None,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ):
        self.limit = max(1, limit if limit is not None else _limit())
        self.clock = clock or (lambda: time.time())
        self.sleeper = sleeper or (lambda delay: time.sleep(delay))
        root = root or Path(
            os.environ.get("GLEE_KEY_COORDINATION_DIR", "~/.glee/coordination")
        ).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        key = api_key.strip()
        if key.casefold().startswith("bearer "):
            key = key[7:].strip()
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        self.path = root / f"rate-{digest}.json"
        self.lock_path = root / f"rate-{digest}.lock"

    @contextmanager
    def _locked(self):
        with self.lock_path.open("a+") as handle:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def acquire(self, priority: int = 1) -> None:
        return

    def _read(self) -> dict:
        try:
            value = json.loads(self.path.read_text())
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _write(self, state: dict) -> None:
        temp = self.path.with_name(f".{self.path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(state, separators=(",", ":")))
        os.replace(temp, self.path)
