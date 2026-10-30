"""Last known round + ZOPA: hold the late split."""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView, last_price, role_of, value_of
from glee_agent.workflows._common import with_message
from glee_agent.workflows.negotiation._tactics import acceptable_zopa, compose_ask, reject


def handle(view: GameView) -> dict[str, Any]:
    if view.action_type == "offer":
        return _offer(view)
    return _decide(view)


def _ask(view: GameView) -> float:
    role = role_of(view)
    my_value = value_of(view) or 0.0
    return compose_ask(view, "ultimatum", role, my_value).value


def _offer(view: GameView) -> dict[str, Any]:
    return with_message(view, {"product_price": _ask(view)}, "This is my number.")


def _decide(view: GameView) -> dict[str, Any]:
    role = role_of(view)
    my_value = value_of(view) or 0.0
    ask = compose_ask(view, "ultimatum", role, my_value).value
    price = last_price(view)
    if price is None:
        return reject(view, ask)
    if acceptable_zopa(view, role, my_value, price, ask):
        return {"decision": "AcceptOffer"}
    return reject(view, ask)
