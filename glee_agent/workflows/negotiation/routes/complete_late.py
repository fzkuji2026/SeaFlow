"""Late complete-info: keep a high ZOPA ask; accept ≥40% of surplus."""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView, last_price, role_of, value_of
from glee_agent.workflows._common import with_message
from glee_agent.workflows.negotiation._tactics import acceptable_zopa, compose_ask, reject


def handle(view: GameView) -> dict[str, Any]:
    role = role_of(view)
    my_value = value_of(view) or 0.0
    ask = compose_ask(view, "complete_late", role, my_value).value
    if view.action_type == "offer":
        return with_message(view, {"product_price": ask}, "Last room to close.")
    price = last_price(view)
    if price is None:
        return reject(view, ask)
    if acceptable_zopa(view, role, my_value, price, ask):
        return {"decision": "AcceptOffer"}
    return reject(view, ask)
