"""Cross-process ownership of pending GLEE games."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class GameClaim:
    game_id: str
    family: str
    run_id: str
    pid: int
    claimed_at: float


class ClaimCoordinator:
    def __init__(self, root: Path, run_id: str, family: str, *, pid: int | None = None):
        self.root = Path(root)
        self.run_id = run_id
        self.family = family
        self.pid = pid or os.getpid()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, game_id: str) -> Path:
        return self.root / f"{game_id}.json"

    @staticmethod
    def _read(path: Path) -> GameClaim | None:
        try:
            return GameClaim(**json.loads(path.read_text()))
        except (OSError, TypeError, ValueError):
            return None

    def try_claim(self, game: dict) -> GameClaim | None:
        if str(game.get("game_family") or "") != self.family:
            return None
        game_id = str(game.get("game_id") or "")
        if not game_id:
            return None
        claim = GameClaim(game_id, self.family, self.run_id, self.pid, time.time())
        path = self._path(game_id)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            existing = self._read(path)
            return existing if existing and existing.run_id == self.run_id and existing.pid == self.pid else None
        with os.fdopen(fd, "w") as handle:
            json.dump(asdict(claim), handle, sort_keys=True)
            handle.write("\n")
        return claim

    def release(self, game_id: str) -> None:
        path = self._path(game_id)
        claim = self._read(path)
        if claim and claim.run_id == self.run_id and claim.pid == self.pid:
            path.unlink(missing_ok=True)

    def owns(self, game_id: str) -> bool:
        claim = self._read(self._path(game_id))
        return bool(
            claim
            and claim.run_id == self.run_id
            and claim.pid == self.pid
            and claim.family == self.family
        )

    def owned_games(self) -> list[GameClaim]:
        claims = []
        for path in self.root.glob("*.json"):
            claim = self._read(path)
            if claim and claim.run_id == self.run_id and claim.pid == self.pid:
                claims.append(claim)
        return claims

    def owned_incomplete_count(self) -> int:
        return len(self.owned_games())

    def reap_stale(self, is_finished: Callable[[str], bool]) -> list[str]:
        removed = []
        for path in self.root.glob("*.json"):
            claim = self._read(path)
            if not claim or self._pid_alive(claim.pid) or not is_finished(claim.game_id):
                continue
            path.unlink(missing_ok=True)
            removed.append(claim.game_id)
        return removed

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
