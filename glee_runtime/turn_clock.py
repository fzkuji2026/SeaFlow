"""Turn remaining-time helpers from 0303faf, used by GameRunner."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import time

_OPEN_TURN_S = 90.0
_KNOWN_TURN_S = 120.0
_URGENT_REMAINING_S = 20.0
_SOON_REMAINING_S = 45.0
_URGENT_POLL_S = 0.05
_SOON_POLL_S = 0.2


def _finite_seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number:
        return None
    return number


def _deadline_epoch(value: Any) -> float | None:
    number = _finite_seconds(value)
    if number is not None:
        return number / 1000.0 if number > 1e12 else number
    if not isinstance(value, str) or not value:
        return None
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.timestamp()


def turn_budget_seconds(game: dict) -> float:
    state = game.get("game_state")
    state = state if isinstance(state, dict) else {}
    if state.get("horizon_known") and state.get("max_rounds"):
        return _KNOWN_TURN_S
    return _OPEN_TURN_S


def remaining_seconds(
    game: dict,
    *,
    now: float | None = None,
    last_move_done: float | None = None,
) -> float:
    """Seconds left on this pending turn. Missing clock → full budget."""
    state = game.get("game_state")
    state = state if isinstance(state, dict) else {}
    for key in (
        "time_remaining",
        "seconds_remaining",
        "time_left",
        "remaining_time",
        "drop_time",
    ):
        value = _finite_seconds(game.get(key))
        if value is None:
            value = _finite_seconds(state.get(key))
        if value is not None:
            return value
    clock = time.time() if now is None else now
    for key in (
        "turn_deadline",
        "move_deadline",
        "deadline",
        "expires_at",
        "timeout_at",
    ):
        epoch = _deadline_epoch(game.get(key))
        if epoch is None:
            epoch = _deadline_epoch(state.get(key))
        if epoch is not None:
            return epoch - clock
    budget = turn_budget_seconds(game)
    if last_move_done is not None:
        # ponytail: opponent can answer in <1s; shrink from our last POST
        return budget - (clock - last_move_done)
    return budget


def poll_sleep_seconds(
    games: list[dict],
    poll_interval: float,
    *,
    now: float | None = None,
    last_done: dict[str, float] | None = None,
) -> float:
    if poll_interval <= 0:
        return 0.0
    if not games:
        return poll_interval
    done = last_done or {}
    shortest = min(
        remaining_seconds(
            game,
            now=now,
            last_move_done=done.get(str(game.get("game_id") or "")),
        )
        for game in games
    )
    if shortest <= _URGENT_REMAINING_S:
        return min(poll_interval, _URGENT_POLL_S)
    if shortest <= _SOON_REMAINING_S:
        return min(poll_interval, _SOON_POLL_S)
    return poll_interval
