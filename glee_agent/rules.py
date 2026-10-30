"""Official GLEE visible-state view and legal-action contract.

Source: https://glee-competition.com/llms.txt (game rules + API).
Nothing here is copied from other participants' agents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

MAX_MESSAGE_LEN = 2000
FAMILIES = ("bargaining", "negotiation", "persuasion")


class ActionError(ValueError):
    """Action does not match the current GLEE contract."""


def _num(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_nonneg(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionError(f"{field} must be a number")
    if not math.isfinite(value) or value < 0:
        raise ActionError(f"{field} must be finite and nonnegative")
    return float(value)


def opponent_id(player: str) -> str:
    if player == "player_1":
        return "player_2"
    if player == "player_2":
        return "player_1"
    if player == "alice":
        return "bob"
    if player == "bob":
        return "alice"
    raise ActionError(f"unknown player {player!r}")


def as_player_id(name: str) -> str:
    if name in ("alice", "player_1"):
        return "player_1"
    if name in ("bob", "player_2"):
        return "player_2"
    return name


@dataclass(frozen=True)
class GameView:
    family: str
    action_type: str
    you: str
    state: dict[str, Any]
    valid_actions: dict[str, Any]
    messages_allowed: bool
    complete_information: bool
    horizon_known: bool
    round: int
    max_rounds: int | None
    is_last_round: bool

    @property
    def fields(self) -> dict[str, Any]:
        fields = self.valid_actions.get("fields")
        return fields if isinstance(fields, dict) else {}


def parse(game: dict[str, Any]) -> GameView:
    family = str(game.get("game_family") or "")
    if family not in FAMILIES:
        raise ActionError(f"unknown game family: {family!r}")
    state = game.get("game_state")
    if not isinstance(state, dict):
        raise ActionError("game_state must be a dict")
    valid = game.get("valid_actions")
    if not isinstance(valid, dict) or not valid.get("type"):
        raise ActionError("valid_actions.type is required")
    you = as_player_id(str(game.get("your_player") or state.get("current_player") or "player_1"))
    max_rounds_raw = state.get("max_rounds")
    if max_rounds_raw is None and family == "persuasion":
        max_rounds_raw = state.get("total_rounds")
    max_rounds = int(max_rounds_raw) if max_rounds_raw is not None else None
    if "horizon_known" in state:
        horizon_known = bool(state.get("horizon_known"))
    else:
        horizon_known = max_rounds is not None
    round_n = int(_num(state.get("round"), 1.0) or 1)
    is_last = bool(
        horizon_known and max_rounds is not None and round_n >= max_rounds
    )
    messages = bool(state.get("messages_allowed", False))
    fields = valid.get("fields")
    if (
        str(valid["type"]) == "offer"
        and isinstance(fields, dict)
        and fields
        and "message" not in fields
    ):
        messages = False
    return GameView(
        family=family,
        action_type=str(valid["type"]),
        you=you,
        state=state,
        valid_actions=valid,
        messages_allowed=messages,
        complete_information=bool(state.get("complete_information", False)),
        horizon_known=horizon_known,
        round=round_n,
        max_rounds=max_rounds,
        is_last_round=is_last,
    )


def money(view: GameView) -> float:
    return float(_num(view.state.get("money_to_divide"), 0.0) or 0.0)


def _player_delta(state: dict[str, Any], pid: str) -> float | None:
    """Visible discount factor in (0, 1]. Accepts delta_* or inflation_* (percent or rate)."""
    idx = "1" if pid == "player_1" else "2"
    names = ("alice",) if pid == "player_1" else ("bob",)
    for key in (f"delta_{idx}", f"{pid}_delta", f"{names[0]}_delta"):
        if key in state:
            value = _num(state.get(key), None)
            if value is None:
                return None
            return min(max(value, 0.0), 1.0)
    for key in (
        f"inflation_{idx}",
        f"{pid}_inflation",
        f"{names[0]}_inflation",
        f"{pid}_inflation_rate",
        f"inflation_{names[0]}",
    ):
        if key in state:
            value = _num(state.get(key), None)
            if value is None:
                return None
            if value > 1.0:
                value = value / 100.0
            return min(max(1.0 - value, 0.0), 1.0)
    return None


def my_delta(view: GameView) -> float | None:
    return _player_delta(view.state, view.you)


def opp_delta(view: GameView) -> float | None:
    try:
        other = opponent_id(view.you)
    except ActionError:
        other = "player_2" if view.you == "player_1" else "player_1"
    return _player_delta(view.state, other)


def offer_gain(offer: dict[str, Any] | None, player: str) -> float | None:
    if not isinstance(offer, dict):
        return None
    pid = as_player_id(player)
    keys = {
        "player_1": ("player_1_gain", "alice_gain"),
        "player_2": ("player_2_gain", "bob_gain"),
    }[pid]
    for key in keys:
        if key in offer:
            return _num(offer[key], None)
    return None


def last_price(view: GameView) -> float | None:
    last = view.state.get("last_offer")
    if not isinstance(last, dict):
        return None
    if "price" in last:
        return _num(last.get("price"), None)
    if "product_price" in last:
        return _num(last.get("product_price"), None)
    return None


def role_of(view: GameView, player: str | None = None) -> str:
    pid = as_player_id(player or view.you)
    return str(view.state.get(f"{pid}_role") or "")


def value_of(view: GameView, player: str | None = None) -> float | None:
    pid = as_player_id(player or view.you)
    key = f"{pid}_value"
    if key not in view.state:
        return None
    return _num(view.state.get(key), None)


def seller_buyer_values(view: GameView) -> tuple[float | None, float | None]:
    """(seller_value, buyer_value); missing side is None under incomplete info."""
    seller = buyer = None
    for pid in ("player_1", "player_2"):
        role = role_of(view, pid)
        val = value_of(view, pid)
        if role == "seller":
            seller = val
        elif role == "buyer":
            buyer = val
    return seller, buyer


def zopa(view: GameView) -> tuple[float, float] | None:
    """Inclusive [seller_value, buyer_value] when both values are visible and surplus >= 0."""
    seller, buyer = seller_buyer_values(view)
    if seller is None or buyer is None:
        return None
    if buyer < seller:
        return None
    return (seller, buyer)


def split_pot(total: float, mine: float, you: str) -> tuple[float, float]:
    """Return (alice_gain, bob_gain) that sum exactly to total."""
    mine = round(float(mine), 2)
    mine = min(max(mine, 0.0), round(float(total), 2))
    theirs = float(total) - mine
    if you == "player_1":
        return mine, theirs
    return theirs, mine


def rubinstein_share_raw(d_me: float, d_opp: float) -> float:
    """Unclamped Rubinstein 1982 proposer SPE: (1-δ_opp)/(1-δ_me·δ_opp)."""
    d_me = min(max(float(d_me), 0.0), 1.0)
    d_opp = min(max(float(d_opp), 0.0), 1.0)
    denom = 1.0 - d_me * d_opp
    if denom <= 1e-9:
        return 0.5
    return (1.0 - d_opp) / denom


def rubinstein_proposer_share(d_me: float, d_opp: float) -> float:
    """Stationary SPE ask, capped [0.40, 0.70] so we neither crumb nor grab 80/20."""
    return min(max(rubinstein_share_raw(d_me, d_opp), 0.40), 0.70)


def _check_keys(action: dict[str, Any], required: set[str], optional: set[str]) -> None:
    actual = set(action)
    missing = required - actual
    if missing:
        raise ActionError(f"missing field(s): {', '.join(sorted(missing))}")
    extra = actual - required - optional
    if extra:
        raise ActionError(f"unexpected field(s): {', '.join(sorted(extra))}")


def _check_message(view: GameView, action: dict[str, Any]) -> None:
    if "message" not in action:
        return
    if not view.messages_allowed:
        raise ActionError("message is not allowed in this game")
    msg = action["message"]
    if not isinstance(msg, str):
        raise ActionError("message must be a string")
    if len(msg) > MAX_MESSAGE_LEN:
        raise ActionError(f"message exceeds {MAX_MESSAGE_LEN} characters")


def validate(game: dict[str, Any], action: Any) -> dict[str, Any]:
    """Raise ActionError if `action` is illegal for this turn. Returns a shallow copy."""
    if not isinstance(action, dict):
        raise ActionError("action must be a dict")
    view = parse(game)
    family, kind = view.family, view.action_type
    action = dict(action)

    if family == "bargaining" and kind == "offer":
        _check_keys(action, {"alice_gain", "bob_gain"}, {"message"})
        a = _finite_nonneg(action["alice_gain"], "alice_gain")
        b = _finite_nonneg(action["bob_gain"], "bob_gain")
        pot = money(view)
        if abs(a + b - pot) > 1e-6:
            raise ActionError("alice_gain + bob_gain must equal money_to_divide")
        _check_message(view, action)
        return action
    if family == "bargaining" and kind == "decision":
        _check_keys(action, {"decision"}, set())
        if action["decision"] not in {"accept", "reject", "walkaway"}:
            raise ActionError("bargaining decision must be accept/reject/walkaway")
        return action

    if family == "negotiation" and kind == "offer":
        _check_keys(action, {"product_price"}, {"message"})
        _finite_nonneg(action["product_price"], "product_price")
        _check_message(view, action)
        return action
    if family == "negotiation" and kind == "decision":
        if "decision" not in action:
            raise ActionError("missing field(s): decision")
        decision = action["decision"]
        if decision in {"AcceptOffer", "WalkAway"}:
            _check_keys(action, {"decision"}, set())
            return action
        if decision != "RejectOffer":
            raise ActionError("invalid negotiation decision")
        if view.is_last_round:
            _check_keys(action, {"decision"}, set())
            return action
        _check_keys(action, {"decision", "product_price"}, {"message"})
        _finite_nonneg(action["product_price"], "product_price")
        _check_message(view, action)
        return action

    if family == "persuasion" and kind == "seller_message":
        _check_keys(action, {"message"}, set())
        _check_message(replace(view, messages_allowed=True), action)
        return action
    if family == "persuasion" and kind in {"seller_recommendation", "buyer_decision"}:
        _check_keys(action, {"decision"}, set())
        if action["decision"] not in {"yes", "no"}:
            raise ActionError("persuasion decision must be yes or no")
        return action

    raise ActionError(f"unknown action type {family}/{kind}")


def looks_legal(game: dict[str, Any], action: Any) -> bool:
    try:
        validate(game, action)
        return True
    except ActionError:
        return False


def sanitize(game: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Drop fields the contract forbids; do not invent numbers."""
    view = parse(game)
    out = dict(action)
    if "message" in out:
        if not view.messages_allowed and view.action_type != "seller_message":
            out.pop("message")
        elif isinstance(out["message"], str) and len(out["message"]) > MAX_MESSAGE_LEN:
            out["message"] = out["message"][:MAX_MESSAGE_LEN]
    if (
        view.family == "negotiation"
        and view.action_type == "decision"
        and out.get("decision") == "RejectOffer"
        and view.is_last_round
    ):
        out.pop("product_price", None)
        out.pop("message", None)
    if view.action_type in {"decision", "buyer_decision", "seller_recommendation"}:
        if out.get("decision") in {"AcceptOffer", "WalkAway", "accept", "reject", "walkaway", "yes", "no"}:
            if view.family == "bargaining" or out.get("decision") in {"AcceptOffer", "WalkAway", "yes", "no"}:
                out = {k: v for k, v in out.items() if k == "decision" or (
                    k in {"product_price", "message"} and out.get("decision") == "RejectOffer"
                )}
    return out
