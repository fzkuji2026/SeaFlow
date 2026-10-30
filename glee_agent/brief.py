"""Compact visible-state briefs for LLM calls. No full history dump."""

from __future__ import annotations

import json
from typing import Any

from glee_agent.rules import GameView, last_price, money, my_delta, offer_gain, opp_delta, parse, role_of, value_of, zopa


def _strip_messages(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_messages(v) for k, v in value.items() if k != "message"}
    if isinstance(value, list):
        return [_strip_messages(v) for v in value]
    return value


def _hist_tail(state: dict[str, Any], n: int = 3) -> list[Any]:
    history = state.get("history")
    if not isinstance(history, list):
        return []
    return [_strip_messages(row) for row in history[-n:]]


def _path_of(view: GameView) -> str:
    try:
        from glee_agent.workflows.tree import path as tree_path

        return tree_path(view)
    except Exception:
        return view.family


def brief_game(game: dict[str, Any], action: dict[str, Any] | None = None) -> str:
    view = parse(game)
    return brief_view(view, action)


def brief_view(view: GameView, action: dict[str, Any] | None = None) -> str:
    lines = [
        f"path={_path_of(view)}",
        f"family={view.family} you={view.you} phase={view.action_type}",
        f"round={view.round} max={view.max_rounds} last={view.is_last_round} "
        f"ci={view.complete_information} msg={view.messages_allowed}",
    ]
    state = view.state
    if view.family == "bargaining":
        pot = money(view)
        lines.append(f"pot={pot} my_delta={my_delta(view)} opp_delta={opp_delta(view)}")
        gain = offer_gain(state.get("last_offer"), view.you)
        if gain is not None:
            share = (gain / pot) if pot else None
            lines.append(f"last_offer_my_gain={gain} last_share={share}")
    elif view.family == "negotiation":
        role = role_of(view)
        mine = value_of(view, view.you)
        zone = zopa(view)
        lines.append(f"role={role} my_value={mine} zopa={zone} last_price={last_price(view)}")
    else:
        lines.append(
            f"price={state.get('product_price')} p={state.get('p')} "
            f"v={state.get('v')} u={state.get('u')} quality={state.get('current_quality')}"
        )
    tail = _hist_tail(state, 3)
    if tail:
        lines.append(f"history_tail={tail}")
    if action:
        numeric = {k: v for k, v in action.items() if k != "message"}
        lines.append(f"decided={numeric}")
    return "\n".join(lines)


def _situation_notes(snap: dict[str, Any]) -> list[str]:
    calc = snap.get("calc") or {}
    family = snap.get("family")
    notes = [
        "Numbers in calc and alternatives were computed by code. Do not recompute.",
        "Pick one alternative name. Do not invent a price, split, or EV.",
    ]
    if family == "bargaining":
        reference = calc.get("public_reference") or {}
        notes.append(
            f"procedure={calc.get('procedure')} plan_ask_share={calc.get('plan_ask_share')} "
            f"spe_share={calc.get('spe_share')} last_share={calc.get('last_share')} "
            f"discounted_now={calc.get('discounted_last_share')} "
            f"next_even_if_accepted={calc.get('next_even_if_accepted')} "
            f"continuation_probability={calc.get('continuation_probability')} "
            f"expected_next_even={calc.get('expected_next_even')} "
            f"response_value_model={calc.get('response_value_model')} "
            f"repeats={calc.get('same_offer_repeats')} we_posted={calc.get('we_posted')}"
        )
        notes.append(
            "Compare discounted payoff now with continuation value, not nominal shares. "
            "expected_next_even discounts a next-round 50/50 by observed rejection history. "
            "Repeated rejection reduces the case for waiting."
        )
        notes.append(
            "If own delta is 1 and the opponent's delta is lower, waiting does not "
            "discount own payoff. Use observed opponent concessions before accepting equality."
        )
        notes.append(
            "On the known final round, any positive payoff dominates rejection. "
            "Public GLEE games usually agree near round 2, so delay needs compensation."
        )
        if reference:
            notes.append(
                f"Public matched-config reference: median discounted payoff share="
                f"{reference.get('median_payoff_share')}, median agreement round="
                f"{reference.get('median_round')}, deal rate={reference.get('deal_rate')}."
            )
            notes.append(
                "This public aggregate matches own role/delta, horizon, and information "
                "structure, but not a fixed opponent delta. Use it as context, not a target."
            )
    elif family == "negotiation":
        reference = calc.get("public_reference") or {}
        notes.append(
            f"role={calc.get('role')} own_band={calc.get('own_band')} "
            f"profit={calc.get('profit')} surplus_share={calc.get('surplus_share')} "
            f"ask={calc.get('ask')} ask_profit={calc.get('planned_ask_profit')} "
            f"current_beats_ask={calc.get('current_beats_planned_ask')} "
            f"opponent_style={calc.get('opponent_style')} "
            f"best_share={calc.get('best_offered_share')}"
        )
        notes.append(
            "Use an AC-next comparison: current utility at least as high as the planned "
            "counter supports acceptance, but discount it for the chance that the counter "
            "is rejected. One-shot games have no continuation. Never accept negative profit."
        )
        notes.append(
            "For complete information, reason in surplus share. For hidden games, infer "
            "reservation pressure from the opponent's concessions and repeated prices."
        )
        if reference:
            notes.append(
                f"Public matched-config reference: median payoff/base="
                f"{reference.get('median_payoff_over_base')}, median surplus share="
                f"{reference.get('median_surplus_share')}, first seller claim="
                f"{reference.get('median_first_seller_claim')}."
            )
            notes.append(
                "This public aggregate matches own role/type, horizon, and information "
                "structure, but not a fixed opponent value. Use it as context, not a target."
            )
    else:
        reference = calc.get("public_reference") or {}
        notes.append(
            f"ev={calc.get('ev')} price={calc.get('price')} rec={calc.get('rec')} "
            f"quality={calc.get('quality')} procedure={calc.get('procedure')} "
            f"ev_minus_price={calc.get('ev_minus_price')} bought_high={calc.get('bought_high')} "
            f"bought_low={calc.get('bought_low')} observed_high_rate={calc.get('observed_high_rate')} "
            f"signal_response={calc.get('signal_response')}"
        )
        if snap.get("action_type") == "buyer_decision":
            notes.append(
                "Buyer objective: treat unverified recommendations as cheap talk, update "
                "credibility from revealed quality, then compare posterior EV with price."
            )
            notes.append(
                "Each round's quality prior remains the public p. Earlier revealed qualities "
                "inform seller signal reliability, not the next round's base quality rate."
            )
        else:
            notes.append(
                "Seller objective: every sale earns price regardless of quality. Do not use "
                "buyer EV or buyer losses as seller utility. Choose the signal most likely "
                "to induce purchase, while accounting for future response rates."
            )
            notes.append(
                "A low-quality sale can reduce future trust; on the final round there is no "
                "future reputation cost. If the buyer responds more to skip than yes, use "
                "that empirical behavior rather than the literal signal meaning."
            )
        if reference:
            notes.append(
                f"Public same-scenario reference: median sale rate="
                f"{reference.get('median_sale_rate')}, high buy rate="
                f"{reference.get('median_high_buy_rate')}, low buy rate="
                f"{reference.get('median_low_buy_rate')}."
            )
    return notes


def decision_packet(
    game: dict[str, Any],
    snap: dict[str, Any],
    alts: list[dict[str, Any]],
) -> str:
    """Full decision-time context: path, calc, heuristic, legal alternatives, history."""
    view = parse(game)
    lines = [
        f"path={snap.get('path') or _path_of(view)}",
        f"family={view.family} you={view.you} phase={view.action_type}",
        f"round={view.round} max={view.max_rounds} last={view.is_last_round} "
        f"ci={view.complete_information} horizon={view.horizon_known} "
        f"msg={view.messages_allowed}",
        f"calc={json.dumps(snap.get('calc') or {}, default=str, sort_keys=True)}",
        f"segment={json.dumps(snap.get('segment') or {}, default=str, sort_keys=True)}",
        f"heuristic={json.dumps({k: v for k, v in (snap.get('action') or {}).items() if k != 'message'}, default=str, sort_keys=True)}",
        "alternatives (pick one name; numbers already computed):",
    ]
    for alt in alts:
        numeric = {k: v for k, v in (alt.get("action") or {}).items() if k != "message"}
        why = str(alt.get("why") or "").strip()
        extra = f" {why}" if why else ""
        lines.append(f"  {alt.get('name')}: {json.dumps(numeric, default=str)}{extra}")
    lines.append("notes:")
    for note in _situation_notes(snap):
        lines.append(f"  - {note}")
    tail = _hist_tail(view.state, 8)
    if tail:
        lines.append(f"history_tail={tail}")
    return "\n".join(lines)
