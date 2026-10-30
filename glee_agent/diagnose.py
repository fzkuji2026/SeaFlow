"""Decision-time numbers. Inspectors read this JSON; they do not multiply."""

from __future__ import annotations

from typing import Any

from glee_agent.rules import (
    GameView,
    as_player_id,
    last_price,
    money,
    my_delta,
    offer_gain,
    opponent_id,
    opp_delta,
    parse,
    role_of,
    value_of,
    zopa,
)
from glee_agent.strategy import strategy
from glee_agent.workflows._common import round_money
from glee_agent.workflows.tree import path as tree_path


def _r(value: float | None, n: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), n)


def _share(part: float | None, whole: float | None) -> float | None:
    if part is None or whole is None or abs(whole) < 1e-12:
        return None
    return round(float(part) / float(whole), 6)


def _last_posted(view: GameView, player: str) -> dict[str, Any] | None:
    """Most recent offer posted by `player` (history, then last_offer)."""
    want = as_player_id(player)
    found: dict[str, Any] | None = None
    for row in view.state.get("history") or []:
        if not isinstance(row, dict):
            continue
        offer = row.get("offer")
        if not isinstance(offer, dict):
            continue
        who = str(row.get("proposer") or offer.get("proposer") or offer.get("from_player") or "")
        if who and as_player_id(who) == want:
            found = offer
    last = view.state.get("last_offer")
    if isinstance(last, dict):
        who = str(last.get("proposer") or last.get("from_player") or "")
        if who and as_player_id(who) == want:
            found = last
        elif (
            not who
            and view.action_type == "decision"
            and want == as_player_id(opponent_id(view.you))
        ):
            found = last
    return found


def snapshot(game: dict[str, Any], action: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify, compute, and replay the heuristic on this decision-time game.

    Pass `action` from `glee_agent.strategy` (never `player.strategy`) so a
    live judge can reuse the packet without calling itself through snapshot.
    """
    view = parse(game)
    if action is None:
        action = strategy(game)
    out: dict[str, Any] = {
        "family": view.family,
        "you": view.you,
        "action_type": view.action_type,
        "round": view.round,
        "max_rounds": view.max_rounds,
        "is_last_round": view.is_last_round,
        "complete_information": view.complete_information,
        "horizon_known": view.horizon_known,
        "path": tree_path(view),
        "action": {k: v for k, v in action.items() if k != "message"},
        "message": action.get("message"),
    }
    if view.family == "bargaining":
        out["calc"] = _bargaining(view, action)
    elif view.family == "negotiation":
        out["calc"] = _negotiation(view)
    else:
        out["calc"] = _persuasion(view)
    from glee_agent.segments import segment

    seg = segment(view, str((out.get("calc") or {}).get("procedure") or ""))
    out["segment"] = {
        "scenario_id": seg.scenario_id,
        "state_id": seg.state_id,
        "parents": seg.parents,
        "indicators": seg.indicators,
    }
    return out


def _bargaining(view: GameView, action: dict[str, Any]) -> dict[str, Any]:
    from glee_agent.workflows.bargaining.offer_shadow import predict as offer_shadow
    from glee_agent.rating_surrogate import predict as rating_shadow
    from glee_agent.reference import bargaining as public_reference
    from glee_agent.workflows.bargaining import classify
    from glee_agent.workflows.bargaining._helpers import (
        ask_share,
        continuation_probability,
        expected_even_continuation,
        fitted_offer_curve,
        fitted_response_curve,
        response_value_model,
        same_split_repeats,
        we_posted_offer,
    )
    from glee_agent.workflows.bargaining._opponent import style_of
    from glee_agent.workflows.bargaining._opponent import state_of
    from glee_agent.workflows.bargaining._plan import compose

    pot = money(view)
    gain = offer_gain(view.state.get("last_offer"), view.you)
    regime = classify(view)
    plan = compose(view, regime, style_of(view))
    asked = None
    if view.action_type == "offer":
        asked = offer_gain(action, view.you)
    last_share = _share(gain, pot)
    try:
        opp = opponent_id(view.you)
    except Exception:
        opp = "player_2" if view.you == "player_1" else "player_1"
    opp_gain = offer_gain(_last_posted(view, opp), view.you)
    delta = my_delta(view)
    discount = None if delta is None else delta ** max(0, view.round - 1)
    discounted = None if last_share is None or discount is None else last_share * discount
    next_even = None if delta is None else 0.50 * (delta ** view.round)
    next_plan = None if delta is None else float(plan.ask) * (delta ** view.round)
    rating_candidate = (
        action if view.action_type == "offer" else view.state.get("last_offer")
    )
    return {
        "opponent_name": view.state.get("opponent_name"),
        "pot": _r(pot, 2),
        "my_delta": _r(my_delta(view)),
        "opp_delta": _r(opp_delta(view)),
        "regime": regime,
        "style": style_of(view),
        "opponent_state": state_of(view).__dict__,
        "procedure": plan.procedure,
        "spe_share": _r(ask_share(view, fade=False)),
        "plan_ask_share": _r(plan.ask),
        "last_gain": _r(gain, 2),
        "last_share": last_share,
        "discounted_last_share": _r(discounted),
        "next_even_if_accepted": _r(next_even),
        "next_plan_if_accepted": _r(next_plan),
        "current_dominates_next_plan": (
            None
            if discounted is None or next_plan is None
            else discounted + 1e-9 >= next_plan
        ),
        "continuation_probability": _r(continuation_probability(view)),
        "expected_next_even": _r(expected_even_continuation(view)),
        "same_offer_repeats": same_split_repeats(view),
        "last_meets_plan": (
            None if last_share is None else last_share + 1e-9 >= float(plan.ask)
        ),
        "opp_last_gain": _r(opp_gain, 2),
        "opp_last_share": _share(opp_gain, pot),
        "our_offer_gain": _r(asked, 2),
        "our_offer_share": _share(asked, pot),
        "we_posted": we_posted_offer(view),
        "posted_share": _share(offer_gain(_last_posted(view, view.you), view.you), pot),
        "public_reference": public_reference(view),
        "rating_shadow": (
            rating_shadow(
                view,
                rating_candidate,
                response=view.action_type == "decision",
            )
            if isinstance(rating_candidate, dict)
            else None
        ),
        "offer_route_shadow": offer_shadow(view),
        "fitted_offer_curve": (
            fitted_offer_curve(view, plan.ask, plan.procedure)
            if view.action_type == "offer"
            else None
        ),
        "fitted_response_curve": (
            fitted_response_curve(view, plan.procedure)
            if view.action_type == "decision"
            else None
        ),
        "response_value_model": (
            response_value_model(view) if view.action_type == "decision" else None
        ),
    }


def _negotiation(view: GameView) -> dict[str, Any]:
    from glee_agent.reference import negotiation as public_reference
    from glee_agent.workflows.negotiation import classify
    from glee_agent.workflows.negotiation._tactics import (
        acceptable_hidden,
        acceptable_zopa,
        best_offered_share,
        compose_ask,
        fitted_offer_features,
        pick_procedure,
        profit,
        slice_of_surplus,
        state_of,
        style_of,
    )
    from glee_agent.workflows.negotiation.classify import own_band, own_mult

    role = role_of(view)
    mine = value_of(view)
    zone = zopa(view)
    price = last_price(view)
    surplus = None
    slice_ = None
    gain = None
    if zone is not None:
        surplus = zone[1] - zone[0]
        if price is not None:
            slice_ = slice_of_surplus(role, price, zone[0], zone[1])
    if mine is not None and price is not None and role in ("seller", "buyer"):
        gain = profit(role, mine, price)
    regime = classify(view)
    proc = pick_procedure(view, regime, role, mine or 0.0)
    ask = compose_ask(view, regime, role, mine or 0.0).value
    fitted_features = None
    fitted_accept = fitted_continue = None
    fitted_accept_level = fitted_continue_level = None
    fitted_accept_n = fitted_continue_n = 0
    if mine is not None and role in ("seller", "buyer"):
        fitted_features = fitted_offer_features(view, role, mine, ask)
        if fitted_features is not None:
            from glee_agent.fitted import predict_segmented
            from glee_agent.segments import segment

            parents = segment(view, proc).parents
            accepted = predict_segmented(
                f"negotiation_{role}_offer_accept", parents, fitted_features
            )
            continued = predict_segmented(
                f"negotiation_{role}_offer_continue", parents, fitted_features
            )
            if accepted is not None:
                fitted_accept, fitted_accept_level, row = accepted
                fitted_accept_n = int(row.get("n") or 0)
            if continued is not None:
                fitted_continue, fitted_continue_level, row = continued
                fitted_continue_n = int(row.get("n") or 0)
    ask_gain = None
    if mine is not None and role in ("seller", "buyer"):
        ask_gain = profit(role, mine, ask)
    close = None
    open_seed = None
    if mine is not None and role in ("seller", "buyer"):
        close = round_money(mine * (1.10 if role == "seller" else 0.90))
        open_seed = round_money(mine * (1.50 if role == "seller" else 0.75))
    would_accept = None
    if mine is not None and price is not None and role in ("seller", "buyer"):
        if zone is not None:
            would_accept = acceptable_zopa(view, role, mine, price, ask)
        else:
            would_accept = acceptable_hidden(view, role, mine, price, ask)
    return {
        "role": role,
        "my_value": _r(mine, 2),
        "own_mult": _r(own_mult(view), 4),
        "own_band": own_band(view),
        "regime": regime,
        "procedure": proc,
        "zopa": [_r(zone[0], 2), _r(zone[1], 2)] if zone else None,
        "surplus": _r(surplus, 2),
        "last_price": _r(price, 2),
        "profit": _r(gain, 2),
        "planned_ask_profit": _r(ask_gain, 2),
        "current_beats_planned_ask": (
            None if gain is None or ask_gain is None else gain + 1e-9 >= ask_gain
        ),
        "opponent_style": style_of(view, role, mine or 0.0),
        "opponent_state": state_of(view, role, mine or 0.0).__dict__,
        "best_offered_share": _r(best_offered_share(view, role)),
        "last_profitable": None if gain is None else gain > 0,
        "surplus_share": _r(slice_),
        "close_110": _r(close, 2),
        "open_seed": _r(open_seed, 2),
        "ask": _r(ask, 2),
        "fitted_offer": {
            "features": fitted_features,
            "accept_probability": _r(fitted_accept),
            "continue_value_over_base": _r(fitted_continue),
            "accept_model_level": fitted_accept_level,
            "continue_model_level": fitted_continue_level,
            "accept_model_n": fitted_accept_n,
            "continue_model_n": fitted_continue_n,
        },
        "would_accept": would_accept,
        "public_reference": public_reference(view),
    }


def _persuasion(view: GameView) -> dict[str, Any]:
    from glee_agent.reference import persuasion as public_reference
    from glee_agent.workflows.persuasion import classify
    from glee_agent.workflows.persuasion._beliefs import (
        compose_buy,
        compose_sell,
        current_rec,
        posterior,
        seller_said_buy,
        signal_response,
        state_of,
    )

    state = view.state
    p = float(state.get("p") or 0.5)
    v = float(state.get("v") or 0.0)
    u = float(state.get("u") or 0.0)
    price = float(state.get("product_price") or 0.0)
    post, seller = posterior(state)
    ev = post * v + (1.0 - post) * u
    rec = current_rec(state)
    regime = classify(view)
    if view.action_type in ("seller_message", "seller_recommendation"):
        proc = compose_sell(view, regime)
    else:
        proc = compose_buy(view, regime)
    rec_s = None if rec is None else ("yes" if rec else "skip")
    stated = p * v + (1.0 - p) * u
    bought_high = bought_low = 0
    for row in state.get("history") or []:
        if not isinstance(row, dict):
            continue
        bought = row.get("bought") or str(row.get("buyer_decision") or "").lower() in (
            "yes",
            "buy",
        )
        if not bought:
            continue
        quality = str(row.get("quality") or "").lower()
        bought_high += quality == "high"
        bought_low += quality == "low"
    bought_total = bought_high + bought_low
    response = signal_response(state)
    return {
        "p": _r(p),
        "v": _r(v, 2),
        "u": _r(u, 2),
        "price": _r(price, 2),
        "stated_ev": _r(stated, 2),
        "stated_ev_gt_price": stated > price,
        "posterior": _r(post),
        "seller_type": seller,
        "ev": _r(ev, 2),
        "ev_minus_price": _r(ev - price, 2),
        "ev_gt_price": ev > price,
        "rec": rec_s,
        "regime": regime,
        "procedure": proc,
        "quality": state.get("current_quality"),
        "known": bool(state.get("is_seller_know_cv")),
        "bought_high": bought_high,
        "bought_low": bought_low,
        "observed_high_rate": _r(bought_high / bought_total) if bought_total else None,
        "signal_response": response,
        "opponent_state": state_of(
            state,
            role=(
                "buyer"
                if view.action_type in ("seller_message", "seller_recommendation")
                else "seller"
            ),
        ).__dict__,
        "message_is_buy": seller_said_buy(state.get("seller_message")),
        "public_reference": public_reference(view),
    }


def replay_to_game(
    replay: dict[str, Any],
    *,
    round_n: int | None = None,
    action_type: str | None = None,
) -> dict[str, Any]:
    """Turn a History modal `/replay` payload into a decision-time `game` dict."""
    family = str(replay.get("game_family") or "")
    you = str(replay.get("your_player") or "player_1")
    config = replay.get("config") if isinstance(replay.get("config"), dict) else {}
    source = replay.get("state")
    if not isinstance(source, dict) or not source:
        source = replay.get("game_state")
    state = dict(source) if isinstance(source, dict) else {}
    hist = list(state.get("history") or [])
    if round_n is None:
        round_n = int(hist[-1].get("round") or state.get("round") or 1) if hist else 1
    before: list[dict[str, Any]] = []
    row_at: dict[str, Any] | None = None
    for row in hist:
        if not isinstance(row, dict):
            continue
        r = int(row.get("round") or 0)
        if r < round_n:
            before.append(row)
        elif r == round_n:
            row_at = row
            break
    state["history"] = before
    state["round"] = round_n
    if family == "bargaining":
        offer = (row_at or {}).get("offer") or state.get("last_offer")
        if isinstance(offer, dict):
            state["last_offer"] = offer
        if action_type is None:
            proposer = str((row_at or {}).get("proposer") or "")
            action_type = "offer" if proposer == you or not row_at else "decision"
        if action_type == "offer":
            state["last_offer"] = None
    elif family == "negotiation":
        offer = (row_at or {}).get("offer") or state.get("last_offer")
        if isinstance(offer, dict):
            state["last_offer"] = offer
        if action_type is None:
            frm = ""
            if isinstance(offer, dict):
                frm = str(offer.get("from_player") or "")
            action_type = "offer" if frm == you or not row_at else "decision"
        if action_type == "offer":
            state["last_offer"] = None
    else:
        seller_turn = you == "player_1"
        seller_knows_cv = bool(
            state.get("is_seller_know_cv") or config.get("is_seller_know_cv")
        )
        state["is_seller_know_cv"] = seller_knows_cv
        if action_type is None:
            if seller_turn:
                signal_type = str(
                    state.get("seller_message_type")
                    or config.get("seller_message_type")
                    or "binary"
                ).lower()
                action_type = (
                    "seller_message" if signal_type == "text" else "seller_recommendation"
                )
            else:
                action_type = "buyer_decision"
        # Every seller sees current quality; is_seller_know_cv only controls
        # whether it also sees the buyer's v/u. A buyer sees the signal only.
        state.pop("current_quality", None)
        state.pop("quality", None)
        if row_at:
            if seller_turn and row_at.get("quality") is not None:
                state["current_quality"] = row_at.get("quality")
            if not seller_turn and row_at.get("seller_message") is not None:
                state["seller_message"] = row_at.get("seller_message")
    fields: dict[str, Any] = {}
    if action_type == "offer":
        fields = (
            {"alice_gain": {}, "bob_gain": {}}
            if family == "bargaining"
            else {"product_price": {}}
        )
        # Live offer schemas include the optional message field when messages
        # are enabled. Omitting it makes rules.parse() correctly infer that a
        # live schema forbids messages, but incorrectly relabels replayed
        # message-enabled offers as messages=0 during offline fitting.
        if state.get("messages_allowed"):
            fields["message"] = {}
    return {
        "game_family": family,
        "your_player": you,
        "valid_actions": {"type": action_type, "fields": fields},
        "game_state": state,
    }


def snapshot_replay(
    replay: dict[str, Any],
    *,
    round_n: int | None = None,
    action_type: str | None = None,
) -> dict[str, Any]:
    game = replay_to_game(replay, round_n=round_n, action_type=action_type)
    snap = snapshot(game)
    snap["game_id"] = replay.get("game_id")
    return snap
