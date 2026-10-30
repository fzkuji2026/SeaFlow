"""Hidden-value seller. Dead 1.5× types WalkAway; else Faratin."""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView
from glee_agent.workflows.negotiation._tactics import play_hidden


def handle(view: GameView) -> dict[str, Any]:
    return play_hidden(view, "incomplete_seller")
