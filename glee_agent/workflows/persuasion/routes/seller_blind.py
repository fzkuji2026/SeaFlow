"""Fallback for malformed seller states that omit current_quality."""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView


def handle(view: GameView) -> dict[str, Any]:
    if view.action_type == "seller_message":
        return {"message": "I recommend buying this unit."}
    return {"decision": "yes"}
