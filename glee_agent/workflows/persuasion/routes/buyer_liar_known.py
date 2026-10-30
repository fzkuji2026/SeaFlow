"""Informed seller already sold a lemon: lemon_ev (posterior EV > price)."""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView
from glee_agent.workflows.persuasion._beliefs import compose_buy, execute_buy


def handle(view: GameView) -> dict[str, Any]:
    return execute_buy(view, compose_buy(view, "buyer_liar_known"))
