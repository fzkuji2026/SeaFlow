"""Hidden one-round: how we know — max_rounds==1 and no ZOPA.

Live types use close_110 (1.10× / 0.90×). A strong 0.8× seller asks
1.50× cost, which is the public GLEE reference median of 1.2× base.
A known ZOPA one-shot is ultimatum.
"""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView, last_price, role_of, value_of
from glee_agent.workflows._common import with_message
from glee_agent.workflows.negotiation._tactics import compose_ask, profit, reject


def handle(view: GameView) -> dict[str, Any]:
    role = role_of(view)
    my_value = value_of(view) or 0.0
    ask = compose_ask(view, "one_shot", role, my_value).value
    if view.action_type == "offer":
        return with_message(view, {"product_price": ask}, "This is my number.")
    price = last_price(view)
    if price is None:
        return reject(view, ask)
    if profit(role, my_value, price) >= 0:
        return {"decision": "AcceptOffer"}
    return reject(view, ask)
