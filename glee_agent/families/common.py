"""Shared helpers for family workflows. No game-family policy here."""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView


def round_money(value: float) -> float:
    return round(float(value), 2)


def progress(view: GameView) -> float:
    if view.max_rounds and view.max_rounds > 1 and view.horizon_known:
        return min(1.0, max(0.0, (view.round - 1) / (view.max_rounds - 1)))
    return min(1.0, max(0.0, (view.round - 1) / 9.0))


def with_message(view: GameView, action: dict[str, Any], text: str) -> dict[str, Any]:
    if view.messages_allowed:
        action["message"] = text
    return action
