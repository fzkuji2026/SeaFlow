"""Atomic, run-scoped persistence for GLEE evidence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    def _atomic_json(self, relative: str | Path, value: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(self._json(value) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return path

    def _append(self, relative: str | Path, value: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(self._json(value) + "\n")
            handle.flush()
        return path

    def save_manifest(self, manifest: dict) -> Path:
        return self._atomic_json("manifest.json", manifest)

    def save_snapshot(self, snapshot: dict) -> Path:
        return self._atomic_json("state.json", snapshot)

    def append_claim(self, claim: dict) -> Path:
        return self._append("claims.jsonl", claim)

    def append_action(self, game_id: str, action: dict) -> Path:
        return self._append(Path("actions") / f"{game_id}.jsonl", action)

    def save_game(self, game_id: str, game: dict) -> Path:
        return self._atomic_json(Path("games") / f"{game_id}.json", game)

    def save_replay(self, game_id: str, replay: dict) -> Path:
        return self._atomic_json(Path("replays") / f"{game_id}.json", replay)

    def save_analysis(self, game_id: str, analysis: dict) -> Path:
        return self._atomic_json(Path("analyses") / f"{game_id}.json", analysis)

    def append_rating(self, rating: dict) -> Path:
        return self._append("ratings.jsonl", rating)

    def close(self) -> None:
        """Writes are synchronously flushed; no history-wide shutdown scan."""
