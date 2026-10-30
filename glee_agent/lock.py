"""Refuse a second live agent on the same machine/key."""

from __future__ import annotations

import atexit
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "logs" / "agent.lock"
_FH = None


class AlreadyRunning(RuntimeError):
    pass


def exclusive(path: Path = LOCK_PATH) -> None:
    global _FH
    if os.environ.get("GLEE_ALLOW_MULTI"):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        fh.close()
        raise AlreadyRunning(f"another agent holds {path}") from exc
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    _FH = fh
    atexit.register(_release)


def _release() -> None:
    global _FH
    if _FH is None:
        return
    try:
        import fcntl

        fcntl.flock(_FH.fileno(), fcntl.LOCK_UN)
        _FH.close()
    except OSError:
        pass
    _FH = None
