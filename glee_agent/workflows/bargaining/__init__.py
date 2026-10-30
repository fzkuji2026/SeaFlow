"""Classify the pie game, then execute the single current plan pipeline."""

from __future__ import annotations

from typing import Any, Callable

from glee_agent.rules import GameView

from .classify import classify
REGIMES = (
    "deadline",
    "penultimate",
    "powerless",
    "impatient_hot",
    "impatient_vs_hotter",
    "impatient",
    "patient",
    "patient_vs_impatient",
    "unknown_horizon",
    "incomplete",
    "symmetric",
)


def _make_handler(regime: str) -> Callable[[GameView], dict[str, Any]]:
    def handle(view: GameView) -> dict[str, Any]:
        from ._opponent import style_of
        from ._plan import compose, execute

        return execute(view, compose(view, regime, style_of(view)))

    return handle


HANDLERS = {regime: _make_handler(regime) for regime in REGIMES}


def play(view: GameView) -> dict[str, Any]:
    return dispatch(view, classify(view))


def dispatch(view: GameView, regime: str) -> dict[str, Any]:
    handle = HANDLERS.get(regime)
    if handle is None:
        return {"decision": "reject"}
    return handle(view)
