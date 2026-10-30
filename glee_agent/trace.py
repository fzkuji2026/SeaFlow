"""Client that dumps each finished game and feeds the live adapter."""

from __future__ import annotations

import json
import time
from pathlib import Path

from glee_agent.adapt import ingest_full
from glee_sdk import GleeClient

ROOT = Path(__file__).resolve().parents[1]
ACTION_TRACES = ROOT / "logs" / "action_traces"


class TracingClient(GleeClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch: list[dict] = []
        self.assigned: dict[str, dict] = {}
        self.ledger_path: Path | None = None

    @property
    def assigned_count(self) -> int:
        return len(self.assigned)

    def set_batch_ledger(self, path: Path) -> None:
        self.ledger_path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    def pending_games(self):
        games = super().pending_games()
        for game in games:
            game_id = str(game.get("game_id") or "")
            if not game_id or game_id in self.assigned:
                continue
            row = {
                "game_id": game_id,
                "game_family": game.get("game_family"),
                "your_player": game.get("your_player"),
                "first_seen_at": time.time(),
            }
            self.assigned[game_id] = row
            if self.ledger_path is not None:
                with self.ledger_path.open("a") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        return games

    def batch_summary(self) -> dict:
        family: dict[str, int] = {}
        role: dict[str, int] = {}
        for row in self.assigned.values():
            fam = str(row.get("game_family") or "unknown")
            player = str(row.get("your_player") or "unknown")
            family[fam] = family.get(fam, 0) + 1
            role[f"{fam}:{player}"] = role.get(f"{fam}:{player}", 0) + 1
        return {
            "assigned_games": self.assigned_count,
            "family": family,
            "role": role,
            "game_ids": sorted(self.assigned),
            "ledger": str(self.ledger_path) if self.ledger_path is not None else None,
        }

    def _record_action(self, game: dict, action: dict) -> None:
        """Persist the exact loaded-policy decision before submitting it."""
        from glee_agent.decision_trace import current_node, current_trace
        from glee_agent.diagnose import snapshot

        game_id = str(game.get("game_id") or "")
        if not game_id:
            return
        ACTION_TRACES.mkdir(parents=True, exist_ok=True)
        node = current_node()
        workflow_trace = current_trace()
        snap = snapshot(game, action)
        snap["decision_node"] = node
        row = {
            "ts": time.time(),
            "deployment_id": self.ledger_path.stem if self.ledger_path else None,
            "game_id": game_id,
            "game": game,
            "action": action,
            "decision_node": node,
            "workflow_trace": workflow_trace,
            "snapshot": snap,
        }
        with (ACTION_TRACES / f"{game_id}.jsonl").open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _handle_game(self, strategy_fn, game):
        def traced(current):
            action = strategy_fn(current)
            self._record_action(current, action)
            return action

        finished = super()._handle_game(traced, game)
        if not finished:
            return False
        full = dict(game)
        result = None
        try:
            fetched = self.game_state(game["game_id"])
            if isinstance(fetched, dict):
                full.update(fetched)
                result = fetched.get("result")
        except Exception:
            result = None
        full.setdefault("game_id", game.get("game_id"))
        full.setdefault("game_family", game.get("game_family"))
        full.setdefault("your_player", game.get("your_player"))
        if result is not None:
            full["result"] = result
        try:
            ingest_full(full)
        except Exception:
            pass
        self.batch.append(
            {
                "family": full.get("game_family"),
                "your_player": full.get("your_player"),
                "result": full.get("result"),
                "game_id": full.get("game_id"),
            }
        )
        return True
