"""Family map only. Parse already happened in strategy.py."""

from __future__ import annotations

from typing import Any, Callable

from glee_agent.rules import GameView
from glee_agent.workflows.bargaining import HANDLERS as BARGAINING
from glee_agent.workflows.bargaining import play as bargaining_play
from glee_agent.workflows.negotiation import HANDLERS as NEGOTIATION
from glee_agent.workflows.negotiation import play as negotiation_play
from glee_agent.workflows.persuasion import HANDLERS as PERSUASION
from glee_agent.workflows.persuasion import play as persuasion_play

FAMILIES: dict[str, Callable[[GameView], dict[str, Any]]] = {
    "bargaining": bargaining_play,
    "negotiation": negotiation_play,
    "persuasion": persuasion_play,
}

REGIMES: dict[str, dict[str, Callable[[GameView], dict[str, Any]]]] = {
    "bargaining": BARGAINING,
    "negotiation": NEGOTIATION,
    "persuasion": PERSUASION,
}


def play(view: GameView) -> dict[str, Any]:
    fn = FAMILIES.get(view.family)
    if fn is None:
        return {"decision": "reject"}
    return fn(view)


def handle_for(family: str, regime: str):
    """Return the handle(view) for this family/regime, or None if unknown."""
    return (REGIMES.get(family) or {}).get(regime)


def fail_closed(_view: GameView | None = None) -> dict[str, Any]:
    return {"decision": "reject"}
