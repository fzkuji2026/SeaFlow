"""Untested seller: ev_ratio_092 unless skip/lemon/last rewrites the cell."""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView
from glee_agent.workflows.persuasion._beliefs import compose_buy, execute_buy


def handle(view: GameView) -> dict[str, Any]:
    return execute_buy(view, compose_buy(view, "buyer_prior"))
