"""Classify the cheap-talk game, then dispatch to one regime handler."""

from __future__ import annotations

from typing import Any, Callable

from glee_agent.rules import GameView

from .classify import classify
from .routes.buyer_honest import handle as buyer_honest
from .routes.buyer_liar import handle as buyer_liar
from .routes.buyer_known_untested import handle as buyer_known_untested
from .routes.buyer_liar_known import handle as buyer_liar_known
from .routes.buyer_p_high import handle as buyer_p_high
from .routes.buyer_prior import handle as buyer_prior
from .routes.seller_blind import handle as seller_blind
from .routes.seller_honest import handle as seller_honest
from .routes.seller_last import handle as seller_last

HANDLERS: dict[str, Callable[[GameView], dict[str, Any]]] = {
    "seller_honest": seller_honest,
    "seller_last": seller_last,
    "seller_blind": seller_blind,
    "buyer_liar": buyer_liar,
    "buyer_liar_known": buyer_liar_known,
    "buyer_known_untested": buyer_known_untested,
    "buyer_honest": buyer_honest,
    "buyer_prior": buyer_prior,
    "buyer_p_high": buyer_p_high,
}


def play(view: GameView) -> dict[str, Any]:
    return dispatch(view, classify(view))


def dispatch(view: GameView, regime: str) -> dict[str, Any]:
    handle = HANDLERS.get(regime)
    if handle is None:
        return {"decision": "reject"}
    return handle(view)
