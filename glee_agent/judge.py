"""Bounded Grok pick among precomputed legal alternatives.

Numbers stay in the tree / diagnose calc. This module never asks the model
to do arithmetic. If picking is off, times out, or the pick is illegal, the
heuristic stands. Do not import player.strategy (snapshot would recurse).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from glee_agent.agentic import agentic_function
from glee_agent.brief import decision_packet
from glee_agent.diagnose import snapshot
from glee_agent.llm import cooling_down
from glee_agent.rules import (
    GameView,
    last_price,
    looks_legal,
    parse,
    role_of,
    sanitize,
    value_of,
    zopa,
)

logger = logging.getLogger("glee_agent.judge")

MAX_ALTS = 4
TIMEOUT_S = 4.0
MIN_INTERVAL_S = 8.0
_SLOT = threading.Lock()
_LAST_CALL = 0.0


@agentic_function(expose="io", render_range={"callers": 0})
def pick_alternative(brief: str, names: tuple[str, ...] = (), runtime=None) -> dict:
    """You are playing a live GLEE game. The packet has path, calc, history, and
    legal alternatives with numbers already computed by code.

    Maximize the current player's own final payoff percentile. Do not optimize
    fairness, social welfare, or the opponent's payoff. In persuasion, a seller
    earns the price on every sale even when quality is low; buyer EV is not the
    seller's utility.

    Pick exactly one alternative name. You may override the heuristic when calc
    and history make another listed option better. Never invent a price, split,
    share, or EV. Never pick an option that the notes forbid.

    Reply with JSON only: {"pick": "<name>", "why": "<one short sentence>"}.
    """
    if not names:
        return {"pick": "", "why": "empty"}
    data = runtime.exec(
        content=[{"type": "text", "text": brief}],
        response_format={"type": "json_object"},
        timeout_s=TIMEOUT_S,
        max_attempts=1,
    )
    if isinstance(data, dict):
        return data
    return {"pick": str(data or ""), "why": ""}


def _payload(action: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in action.items() if k != "message"}


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ka, kb = _payload(a), _payload(b)
    if set(ka) != set(kb):
        return False
    for key, val in ka.items():
        other = kb[key]
        if isinstance(val, (int, float)) and isinstance(other, (int, float)):
            if abs(float(val) - float(other)) > 1e-6:
                return False
        elif val != other:
            return False
    return True


def _legal(game: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    # Lazy: player imports this module. Normalize tokens the same way live play does.
    from glee_agent.player import normalize_action

    cleaned = sanitize(game, normalize_action(game, dict(action)))
    if looks_legal(game, cleaned):
        return cleaned
    return None


def _add(
    alts: list[dict[str, Any]],
    game: dict[str, Any],
    name: str,
    action: dict[str, Any],
    why: str,
) -> None:
    if len(alts) >= MAX_ALTS:
        return
    cleaned = _legal(game, action)
    if cleaned is None:
        return
    if any(_same(cleaned, existing["action"]) for existing in alts):
        return
    alts.append({"name": name, "action": cleaned, "why": why})


def alternatives(
    game: dict[str, Any],
    heuristic: dict[str, Any],
    snap: dict[str, Any],
) -> list[dict[str, Any]]:
    view = parse(game)
    alts: list[dict[str, Any]] = []
    _add(alts, game, "heuristic", heuristic, "tree")
    if view.family == "bargaining":
        _bargain_alts(game, view, heuristic, snap, alts)
    elif view.family == "negotiation":
        _deal_alts(game, view, heuristic, snap, alts)
    else:
        _talk_alts(game, view, heuristic, snap, alts)
    return alts


def _bargain_alts(
    game: dict[str, Any],
    view: GameView,
    heuristic: dict[str, Any],
    snap: dict[str, Any],
    alts: list[dict[str, Any]],
) -> None:
    from glee_agent.workflows.bargaining._helpers import DEFAULT_MSG, make_offer

    calc = snap.get("calc") or {}
    if view.action_type == "offer":
        msg = str(heuristic.get("message") or DEFAULT_MSG)
        _add(alts, game, "even", make_offer(view, 0.50, msg), "50/50")
        _add(alts, game, "ask_52", make_offer(view, 0.52, msg), "slight premium")
        return
    last_share = calc.get("last_share")
    last_gain = calc.get("last_gain")
    posted = bool(calc.get("we_posted"))
    accept_blocked = False
    if last_gain is not None and last_gain <= 0:
        accept_blocked = True
    if posted and view.round <= 6 and last_share is not None and last_share + 1e-9 < 0.50:
        accept_blocked = True
    if view.is_last_round:
        if last_gain is not None and last_gain > 0 and not accept_blocked:
            _add(alts, game, "accept", {"decision": "accept"}, "last round positive")
        _add(alts, game, "walkaway", {"decision": "walkaway"}, "last round walk")
        return
    if not accept_blocked:
        _add(alts, game, "accept", {"decision": "accept"}, "take the incoming split")
    _add(alts, game, "reject", {"decision": "reject"}, "hold")


def _deal_alts(
    game: dict[str, Any],
    view: GameView,
    heuristic: dict[str, Any],
    snap: dict[str, Any],
    alts: list[dict[str, Any]],
) -> None:
    from glee_agent.workflows.negotiation._tactics import (
        nash_price,
        profit,
        reject,
    )

    calc = snap.get("calc") or {}
    role = role_of(view)
    mine = value_of(view)
    zone = zopa(view)
    ask = calc.get("ask")
    close = calc.get("close_110")
    if view.action_type == "offer":
        if close is not None:
            _add(alts, game, "close_110", {"product_price": close}, "1.10× / 0.90×")
        if mine is not None:
            _add(alts, game, "reservation", {"product_price": mine}, "own value")
        if zone is not None and role in ("seller", "buyer"):
            _add(
                alts,
                game,
                "nash",
                {"product_price": nash_price(role, zone[0], zone[1])},
                "half ZOPA",
            )
        return
    price = last_price(view)
    gain = None
    if mine is not None and price is not None and role in ("seller", "buyer"):
        gain = profit(role, mine, price)
    if gain is not None and gain > 0:
        _add(alts, game, "accept", {"decision": "AcceptOffer"}, f"profit={gain}")
    counter = ask if ask is not None else close
    if counter is not None:
        _add(alts, game, "reject", reject(view, float(counter)), "counter")
    if view.is_last_round or (gain is not None and gain <= 0) or (calc.get("own_band") or "").startswith("dead"):
        _add(alts, game, "walk", {"decision": "WalkAway"}, "leave")


def _talk_alts(
    game: dict[str, Any],
    view: GameView,
    heuristic: dict[str, Any],
    snap: dict[str, Any],
    alts: list[dict[str, Any]],
) -> None:
    calc = snap.get("calc") or {}
    proc = str(calc.get("procedure") or "")
    if view.action_type == "seller_message":
        return
    if proc == "skip_block":
        return
    current_decision = str(heuristic.get("decision") or "")
    other = "no" if current_decision == "yes" else "yes"
    if view.action_type == "buyer_decision":
        ev = calc.get("ev")
        price = calc.get("price")
        if other == "yes" and ev is not None and price is not None and ev + 1e-9 < float(price):
            return
        if other == "yes" and calc.get("rec") == "skip":
            return
        _add(alts, game, other, {"decision": other}, "other legal buy/skip")
        return
    if view.action_type == "seller_recommendation":
        quality = str(calc.get("quality") or "").lower()
        if quality == "high" or view.is_last_round:
            return
        _add(alts, game, other, {"decision": other}, "other legal recommend")


def _needs_judge(snap: dict[str, Any], alts: list[dict[str, Any]]) -> bool:
    """Use the model only where history and continuation create a real choice."""
    calc = snap.get("calc") or {}
    decisions = {
        str((alt.get("action") or {}).get("decision"))
        for alt in alts
        if (alt.get("action") or {}).get("decision") is not None
    }
    family = snap.get("family")
    action_type = snap.get("action_type")
    if family == "bargaining":
        # The two live overrides totaled -2.1 (+2.7 / -4.8), and the current
        # tree already encodes replay-backed continuation stops. Do not let a
        # model replace deterministic discounted-payoff comparisons.
        return False
    if family == "negotiation":
        if action_type != "decision" or snap.get("is_last_round"):
            return False
        # Live evidence supports model overrides only for open-horizon
        # take-profit decisions (+7.4 across two changes). The other five
        # negotiation overrides totaled -22.2 by replacing deterministic
        # counters with premature acceptance.
        if "/open_horizon/live/take_profit/" not in str(snap.get("path") or ""):
            return False
        if not {"AcceptOffer", "RejectOffer"}.issubset(decisions):
            return False
        share = calc.get("surplus_share")
        if share is not None:
            return 0.48 <= float(share) <= 0.65
        band = str(calc.get("own_band") or "")
        if band.startswith("strong") or band.startswith("dead"):
            return False
        gain = calc.get("profit")
        ask_gain = calc.get("planned_ask_profit")
        if gain is None or ask_gain is None or gain <= 0:
            return False
        return abs(float(gain) - float(ask_gain)) <= 0.10 * max(abs(float(ask_gain)), 1.0)
    if family == "persuasion":
        # Signal semantics, public iid p, seller credibility, buyer response,
        # and EV are all computed in code. Live logs showed no useful seller
        # override and incorrect claims that mid-game rounds had no future.
        return False
    return False


def apply(game: dict[str, Any], heuristic: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("GLEE_LLM_JUDGE", "0") != "1":
        return heuristic
    if cooling_down():
        return heuristic
    if not _SLOT.acquire(blocking=False):
        return heuristic
    try:
        return _pick(game, heuristic)
    finally:
        _SLOT.release()


def _pick(game: dict[str, Any], heuristic: dict[str, Any]) -> dict[str, Any]:
    global _LAST_CALL
    try:
        snap = snapshot(game, action=heuristic)
        alts = alternatives(game, heuristic, snap)
    except Exception as exc:
        logger.warning("judge packet failed (%s); heuristic", exc)
        return heuristic
    if len(alts) < 2:
        return heuristic
    if not _needs_judge(snap, alts):
        return heuristic
    now = time.monotonic()
    if now - _LAST_CALL < MIN_INTERVAL_S:
        return heuristic
    _LAST_CALL = now
    by_name = {str(alt["name"]): alt for alt in alts}
    names = tuple(by_name)
    brief = decision_packet(game, snap, alts)
    try:
        picked = pick_alternative(brief, names=names)
    except Exception as exc:
        logger.warning("judge pick failed (%s); heuristic", exc)
        return heuristic
    name = str((picked or {}).get("pick") or "").strip()
    if name not in by_name:
        compact = "".join(ch for ch in name.lower() if ch.isalnum())
        for key in by_name:
            if "".join(ch for ch in key.lower() if ch.isalnum()) == compact:
                name = key
                break
    alt = by_name.get(name)
    if alt is None:
        logger.info("judge ignored unknown pick %r; heuristic", name)
        return heuristic
    action = alt["action"]
    if not looks_legal(game, action):
        return heuristic
    logger.info(
        "[judge] game=%s path=%s picked=%s changed=%s why=%s",
        str(game.get("game_id") or "?")[:8],
        snap.get("path"),
        name,
        name != "heuristic",
        (picked or {}).get("why"),
    )
    return action
