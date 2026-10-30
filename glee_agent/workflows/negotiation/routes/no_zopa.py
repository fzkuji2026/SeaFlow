"""Proven no surplus: WalkAway. Handler does not re-check other regimes."""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView, role_of, value_of
from glee_agent.workflows._common import round_money, with_message


def handle(view: GameView) -> dict[str, Any]:
    if view.action_type == "offer":
        return _offer(view)
    return _decide(view)


def _offer(view: GameView) -> dict[str, Any]:
    role = role_of(view)
    my_value = value_of(view) or 0.0
    price = round_money(my_value * (1.6 if role == "seller" else 0.4))
    return with_message(view, {"product_price": price}, "No surplus here.")


def _decide(_view: GameView) -> dict[str, Any]:
    return {"decision": "WalkAway"}
