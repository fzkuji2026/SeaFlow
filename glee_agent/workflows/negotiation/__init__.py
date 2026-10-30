"""Classify the trade game, then dispatch to one regime handler."""

from __future__ import annotations

from typing import Any, Callable

from glee_agent.rules import GameView

from .classify import classify
from .routes.complete import handle as complete
from .routes.complete_late import handle as complete_late
from .routes.complete_seller import handle as complete_seller
from .routes.incomplete import handle as incomplete
from .routes.incomplete_deadline import handle as incomplete_deadline
from .routes.incomplete_late import handle as incomplete_late
from .routes.incomplete_seller import handle as incomplete_seller
from .routes.no_zopa import handle as no_zopa
from .routes.one_shot import handle as one_shot
from .routes.open_horizon import handle as open_horizon
from .routes.ultimatum import handle as ultimatum

HANDLERS: dict[str, Callable[[GameView], dict[str, Any]]] = {
    "no_zopa": no_zopa,
    "ultimatum": ultimatum,
    "complete": complete,
    "complete_late": complete_late,
    "complete_seller": complete_seller,
    "one_shot": one_shot,
    "open_horizon": open_horizon,
    "incomplete": incomplete,
    "incomplete_deadline": incomplete_deadline,
    "incomplete_late": incomplete_late,
    "incomplete_seller": incomplete_seller,
}


def play(view: GameView) -> dict[str, Any]:
    return dispatch(view, classify(view))


def dispatch(view: GameView, regime: str) -> dict[str, Any]:
    handle = HANDLERS.get(regime)
    if handle is None:
        return {"decision": "reject"}
    return handle(view)
