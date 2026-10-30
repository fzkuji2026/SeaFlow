"""Canonical runtime state and configuration names."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

MAX_MOVE_WORKERS = 4


class RunState(StrEnum):
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    NETWORK_WAITING = "NETWORK_WAITING"
    COOLDOWN = "COOLDOWN"
    DRAINING = "DRAINING"
    ENDING = "ENDING"
    ENDED = "ENDED"
    # Backward-compatible spelling used by older tests and persisted callers.
    # It is an enum alias, so the canonical terminal state remains ENDED.
    STOPPED = ENDED


@dataclass(frozen=True)
class RunConfig:
    family: str
    # Zero means adaptive queueing; positive values retain the fixed target.
    game_concurrency: int = 0
    queue_target: int = 0
    move_workers: int = MAX_MOVE_WORKERS

    def __post_init__(self) -> None:
        if self.family not in {"bargaining", "negotiation", "persuasion"}:
            raise ValueError(f"invalid game family: {self.family}")
        if min(self.game_concurrency, self.queue_target, self.move_workers) < 0:
            raise ValueError("concurrency values must be nonnegative")
        if self.game_concurrency == 0:
            if self.queue_target != 0:
                raise ValueError("adaptive queue_target must be zero")
            if self.move_workers < 1:
                raise ValueError("adaptive move_workers must be positive")
            return
        if self.queue_target < 1 or self.move_workers < 1:
            raise ValueError("concurrency values must be positive")
        if self.move_workers > self.game_concurrency:
            raise ValueError("move_workers cannot exceed game_concurrency")


@dataclass(frozen=True)
class RunSnapshot:
    state: RunState
    owned_games: int = 0
    running_moves: int = 0
    queue_reservations: int = 0
    reason: str | None = None

    @property
    def drain_complete(self) -> bool:
        return (
            self.owned_games == 0
            and self.running_moves == 0
            and self.queue_reservations == 0
        )
