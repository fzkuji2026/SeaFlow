"""Decision-path labels. One segment per tree node; live play still uses family play()."""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView, parse, role_of, value_of


def path(game: dict[str, Any] | GameView) -> str:
    view = game if isinstance(game, GameView) else parse(game)
    if view.family == "bargaining":
        return _bargaining(view)
    if view.family == "negotiation":
        return _negotiation(view)
    if view.family == "persuasion":
        return _persuasion(view)
    return view.family


def _bargaining(view: GameView) -> str:
    from glee_agent.workflows.bargaining import classify
    from glee_agent.workflows.bargaining._opponent import style_of
    from glee_agent.workflows.bargaining._plan import compose

    regime = classify(view)
    plan = compose(view, regime, style_of(view))
    return f"bargaining/{regime}/{plan.procedure}/{view.action_type}"


def _negotiation(view: GameView) -> str:
    from glee_agent.workflows.negotiation import classify
    from glee_agent.workflows.negotiation._tactics import pick_procedure

    from glee_agent.workflows.negotiation.classify import own_band

    regime = classify(view)
    role = role_of(view)
    my_value = value_of(view) or 0.0
    proc = pick_procedure(view, regime, role, my_value)
    band = own_band(view)
    return f"negotiation/{regime}/{band}/{proc}/{view.action_type}"


def _persuasion(view: GameView) -> str:
    from glee_agent.workflows.persuasion import classify
    from glee_agent.workflows.persuasion._beliefs import compose_buy, compose_sell

    regime = classify(view)
    if view.action_type in ("seller_message", "seller_recommendation"):
        proc = compose_sell(view, regime)
    else:
        proc = compose_buy(view, regime)
    return f"persuasion/{regime}/{proc}/{view.action_type}"
