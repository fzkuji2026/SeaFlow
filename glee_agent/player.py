"""Numbers come from the tree. Grok picks among precomputed legal alternatives."""

from __future__ import annotations

import logging
from typing import Any

from glee_agent.brief import brief_game
from glee_agent.judge import apply as apply_judge
from glee_agent.rules import (
    ActionError,
    last_price,
    looks_legal as contract_legal,
    money,
    parse,
    sanitize,
    validate,
    value_of,
)
from glee_agent.strategy import strategy as heuristic
from glee_agent.talk import fill_message as agentic_fill_message

logger = logging.getLogger("glee_agent.player")


def user_prompt(game: dict[str, Any], action: dict[str, Any]) -> str:
    """Compact brief for the message LLM. Not a full state dump."""
    return brief_game(game, action)


def unwrap_action(action: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(action, dict):
        return action
    fields = action.get("fields")
    if isinstance(fields, dict) and fields:
        return fields
    return action


def _canon_token(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


_DECISION_MAP = {
    "accept": "accept",
    "yes": "yes",
    "buy": "yes",
    "recommend": "yes",
    "reject": "reject",
    "no": "no",
    "skip": "no",
    "pass": "no",
    "walkaway": "walkaway",
    "walk_away": "walkaway",
    "walk away": "walkaway",
    "acceptoffer": "AcceptOffer",
    "rejectoffer": "RejectOffer",
    "walkawayoffer": "WalkAway",
}


def normalize_action(game: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    action = dict(unwrap_action(action))
    wanted = (game.get("valid_actions") or {}).get("type")
    family = game.get("game_family")
    if wanted in ("decision", "buyer_decision", "seller_recommendation") and "decision" in action:
        token = _canon_token(action.get("decision"))
        if wanted == "decision" and family == "negotiation":
            mapped = {
                "accept": "AcceptOffer",
                "yes": "AcceptOffer",
                "acceptoffer": "AcceptOffer",
                "reject": "RejectOffer",
                "no": "RejectOffer",
                "rejectoffer": "RejectOffer",
                "walkaway": "WalkAway",
                "walkawayoffer": "WalkAway",
            }.get(token)
        elif wanted == "decision" and family == "bargaining":
            mapped = {
                "accept": "accept",
                "yes": "accept",
                "acceptoffer": "accept",
                "reject": "reject",
                "no": "reject",
                "rejectoffer": "reject",
                "walkaway": "walkaway",
                "walkawayoffer": "walkaway",
            }.get(token)
        else:
            mapped = _DECISION_MAP.get(token)
        if mapped:
            action["decision"] = mapped
    return action


def looks_legal(game: dict[str, Any], action: dict[str, Any]) -> bool:
    action = normalize_action(game, action)
    return contract_legal(game, action)


def _wants_message(game: dict[str, Any], action: dict[str, Any]) -> bool:
    wanted = (game.get("valid_actions") or {}).get("type")
    if wanted == "seller_message":
        return True
    view = parse(game)
    if not view.messages_allowed:
        return False
    if wanted == "offer":
        return True
    if wanted == "decision" and game.get("game_family") == "negotiation" and action.get("decision") == "RejectOffer":
        return True
    return False


def _fill_message(game: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    if not _wants_message(game, action):
        return action
    return agentic_fill_message(game, action)


def fallback_action(game: dict[str, Any]) -> dict[str, Any]:
    """Return one legal action when any primary workflow raises."""
    view = parse(game)
    if view.family == "bargaining":
        if view.action_type == "decision":
            return {"decision": "reject"}
        half = money(view) / 2.0
        return {"alice_gain": half, "bob_gain": money(view) - half}
    if view.family == "negotiation":
        if view.action_type == "decision" and view.is_last_round:
            return {"decision": "RejectOffer"}
        price = value_of(view)
        if price is None:
            price = last_price(view) or 0.0
        if view.action_type == "decision":
            return {"decision": "RejectOffer", "product_price": price}
        return {"product_price": price}
    if view.action_type == "seller_message":
        return {"message": "skip"}
    return {"decision": "no"}


def _primary_strategy(game: dict[str, Any]) -> dict[str, Any]:
    action = sanitize(game, normalize_action(game, heuristic(game)))
    action = sanitize(game, apply_judge(game, action))
    action = sanitize(game, _fill_message(game, action))
    try:
        action = validate(game, action)
    except ActionError as exc:
        logger.warning("primary action illegal (%s); retry heuristic", exc)
        action = sanitize(game, normalize_action(game, heuristic(game)))
        action = validate(game, action)
    try:
        from glee_agent.workflows.tree import path as tree_path

        leaf = tree_path(game)
    except Exception:
        leaf = str(game.get("game_family") or "?")
    logger.info(
        "[%s] %s %s",
        leaf,
        (game.get("valid_actions") or {}).get("type"),
        action,
    )
    return action


def strategy(game: dict[str, Any]) -> dict[str, Any]:
    from glee_agent.decision_trace import set_node

    set_node(None)
    try:
        return _primary_strategy(game)
    except Exception:
        logger.exception("primary workflow failed; submitting legal fallback")
        set_node("player.fallback.legal_action")
        action = validate(game, sanitize(game, fallback_action(game)))
        logger.info(
            "[fallback] %s %s",
            (game.get("valid_actions") or {}).get("type"),
            action,
        )
        return action
