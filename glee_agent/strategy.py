"""Dispatch a parsed GLEE state to the family workflow."""

from __future__ import annotations

from glee_agent.policy import refresh
from glee_agent.families.bargaining import play as bargaining_play
from glee_agent.families.negotiation import play as negotiation_play
from glee_agent.families.persuasion import play as persuasion_play
from glee_agent.rules import parse

# Names kept for tests and logs.
bargaining_strategy = bargaining_play
negotiation_strategy = negotiation_play
persuasion_strategy = persuasion_play

STRATEGIES = {
    "bargaining": bargaining_play,
    "negotiation": negotiation_play,
    "persuasion": persuasion_play,
}


def _with_opponent_identity(game: dict) -> dict:
    opponent = game.get("opponent")
    state = game.get("game_state")
    if not isinstance(opponent, dict) or not isinstance(state, dict):
        return game
    name = opponent.get("name") or opponent.get("type")
    if not name or state.get("opponent_name"):
        return game
    enriched = dict(game)
    enriched["game_state"] = {**state, "opponent_name": str(name)}
    return enriched


def strategy(game: dict) -> dict:
    refresh()
    game = _with_opponent_identity(game)
    view = parse(game)
    return STRATEGIES[view.family](view)
