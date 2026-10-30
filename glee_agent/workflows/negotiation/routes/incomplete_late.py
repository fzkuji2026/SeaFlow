"""Hidden values, one known round left. How we know — max_rounds-1, no ZOPA.

Procedure: post 1.10× / 0.90× (they can still accept this turn). Live types
take any non-negative profit. Strong types hold close_110 except last
(`4a906084` 149.43/150, `639a6735` 150/150). Last-round RejectOffer has no
counter — close now. Window: 54e5d9d1 / a47677e8 / 1eab2cab / 6fd2775a / 38a27997.
"""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView, last_price, role_of, value_of
from glee_agent.workflows._common import with_message
from glee_agent.workflows.negotiation._tactics import (
    acceptable_hidden,
    compose_ask,
    profit,
    reject,
)
from glee_agent.workflows.negotiation.classify import own_band


def handle(view: GameView) -> dict[str, Any]:
    role = role_of(view)
    my_value = value_of(view) or 0.0
    ask = compose_ask(view, "incomplete_late", role, my_value).value
    if view.action_type == "offer":
        return with_message(view, {"product_price": ask}, "This is my number.")
    price = last_price(view)
    if price is None:
        return reject(view, ask)
    if (
        own_band(view).startswith("strong")
        or (view.max_rounds is not None and view.round < view.max_rounds - 1)
    ):
        if acceptable_hidden(view, role, my_value, price, ask):
            return {"decision": "AcceptOffer"}
        return reject(view, ask)
    if profit(role, my_value, price) >= 0:
        return {"decision": "AcceptOffer"}
    return reject(view, ask)
