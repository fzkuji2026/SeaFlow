"""Lookup compact medians generated from the public GLEE corpus."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from glee_agent.rules import GameView, my_delta, role_of
from glee_agent.workflows.negotiation.classify import own_mult

PATH = Path(__file__).with_name("reference.json")


@lru_cache(maxsize=1)
def data() -> dict[str, Any]:
    try:
        loaded = json.loads(PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _number(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{float(value):.6g}"


def _horizon(view: GameView, family: str) -> str:
    if family == "bargaining":
        if not view.horizon_known:
            return "inf"
        if view.max_rounds is not None and view.max_rounds <= 12:
            return "12"
        return "25"
    if view.horizon_known and view.max_rounds == 1:
        return "1"
    if view.horizon_known and view.max_rounds is not None and view.max_rounds <= 10:
        return "10"
    return "25"


def bargaining(view: GameView) -> dict[str, Any] | None:
    delta = _number(my_delta(view))
    if delta is None:
        return None
    role = "Alice" if view.you == "player_1" else "Bob"
    key = f"{role}|{delta}|{_horizon(view, 'bargaining')}|{view.complete_information}"
    row = (data().get("bargaining") or {}).get(key)
    return row if isinstance(row, dict) else None


def negotiation(view: GameView) -> dict[str, Any] | None:
    mult = _number(own_mult(view))
    role = role_of(view)
    if mult is None or role not in ("seller", "buyer"):
        return None
    key = f"{role}|{mult}|{_horizon(view, 'negotiation')}|{view.complete_information}"
    row = (data().get("negotiation") or {}).get(key)
    return row if isinstance(row, dict) else None


def persuasion(view: GameView) -> dict[str, Any] | None:
    state = view.state
    price = float(state.get("product_price") or 0.0)
    if price <= 0:
        return None
    p = _number(float(state.get("p") or 0.5))
    value_mult = _number(float(state.get("v") or 0.0) / price)
    known = bool(state.get("is_seller_know_cv"))
    message_type = str(state.get("seller_message_type") or "").lower()
    if message_type not in ("text", "binary"):
        message_type = (
            "binary"
            if view.action_type in ("seller_recommendation", "buyer_decision")
            and isinstance(state.get("seller_message"), dict)
            else "text"
        )
    myopic = bool(state.get("is_myopic", False))
    key = f"{p}|{value_mult}|{known}|{message_type}|{myopic}"
    row = (data().get("persuasion") or {}).get(key)
    return row if isinstance(row, dict) else None
