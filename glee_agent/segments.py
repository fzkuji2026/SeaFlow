"""Canonical, explainable scenario and opponent-state segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from glee_agent.rules import GameView, my_delta, opp_delta, role_of, value_of


@dataclass(frozen=True)
class Segment:
    scenario_id: str
    state_id: str
    parents: tuple[str, ...]
    indicators: dict[str, Any]


MODEL_SCHEMA = {
    "bargaining": {
        "static": ("role", "own_delta", "visible_opp_delta", "horizon", "complete_information", "messages_allowed"),
        "dynamic": ("action_type", "procedure", "opponent_style", "opponent_samples", "rejections"),
        "targets": ("accept_probability", "offer_reject_value", "response_continue_value"),
        "candidates": ("legal_split_curve", "accept", "reject"),
        "objective": "discounted_own_payoff",
        "constraints": ("legality", "terminal_dominance", "observed_action_support", "evidence_gate"),
        "strategy_map": {
            "offer": {
                "uses": ("accept_probability", "offer_reject_value"),
                "scores": "p_accept * discounted_offer + (1 - p_accept) * reject_value",
                "selects": "legal_split_curve",
            },
            "response": {
                "uses": ("discounted_offer", "next_planned_value", "response_continue_value"),
                "scores": "discounted_offer - response_continue_value - uncertainty_guard",
                "selects": ("accept", "reject"),
            },
        },
        "fallback": "deterministic_bargaining_workflow",
    },
    "negotiation": {
        "static": ("role", "own_mult", "visible_opp_mult", "horizon", "complete_information", "messages_allowed"),
        "dynamic": ("action_type", "procedure", "opponent_style", "opponent_samples", "rejections", "zopa_probability"),
        "targets": ("accept_probability", "offer_reject_value", "response_continue_value"),
        "candidates": ("legal_price_curve", "accept", "reject", "walk_away"),
        "objective": "own_profit",
        "constraints": ("legality", "terminal_dominance", "positive_profit_dominance", "observed_action_support", "evidence_gate"),
        "strategy_map": {
            "offer": {
                "uses": ("accept_probability", "offer_reject_value"),
                "scores": "p_accept * offer_profit + (1 - p_accept) * reject_value",
                "selects": "legal_price_curve",
            },
            "response": {
                "uses": ("immediate_profit", "response_continue_value", "zopa_probability"),
                "scores": "immediate_profit - response_continue_value - uncertainty_guard",
                "selects": ("accept", "reject", "walk_away"),
            },
        },
        "fallback": "deterministic_negotiation_workflow",
    },
    "persuasion": {
        "static": ("role", "p", "high_mult", "low_mult", "seller_knows_values", "message_type", "myopic"),
        "dynamic": ("action_type", "procedure", "opponent_style", "opponent_samples", "signal", "revealed_signal_accuracy"),
        "targets": ("quality_posterior", "yes_buy_probability", "no_buy_probability"),
        "candidates": ("buy", "pass", "legal_signal_set"),
        "objective": "buyer_expected_value_or_seller_expected_sales",
        "constraints": ("legality", "posterior_consistency", "observed_action_support", "evidence_gate"),
        "strategy_map": {
            "buyer": {
                "uses": ("quality_posterior",),
                "scores": "posterior * high_value + (1 - posterior) * low_value - price",
                "selects": ("buy", "pass"),
            },
            "seller": {
                "uses": ("yes_buy_probability", "no_buy_probability"),
                "scores": "current_sale_probability + remaining_expected_sales",
                "selects": "legal_signal_set",
            },
        },
        "fallback": "deterministic_persuasion_workflow",
    },
}


def _f(value: float | None) -> str:
    return "hidden" if value is None else f"{float(value):.6g}"


def _horizon(view: GameView) -> str:
    if not view.horizon_known:
        return "open"
    return str(view.max_rounds or "unknown")


def _sample_band(samples: int) -> str:
    if samples <= 0:
        return "n0"
    if samples == 1:
        return "n1"
    if samples <= 3:
        return "n2_3"
    return "n4p"


def _bargaining(view: GameView, procedure: str) -> Segment:
    from glee_agent.rules import money
    from glee_agent.workflows.bargaining._opponent import state_of

    opponent = state_of(view)
    slope_direction = (
        "up"
        if opponent.concession_rate > 0.005
        else "down" if opponent.concession_rate < -0.005 else "flat"
    )
    plateau_band = (
        "unknown"
        if opponent.plateau_span is None
        else "stable" if opponent.plateau_span < 0.015 else "moving"
    )
    role = "alice" if view.you == "player_1" else "bob"
    static = (
        f"role={role}",
        f"own_delta={_f(my_delta(view))}",
        f"opp_delta={_f(opp_delta(view))}",
        f"horizon={_horizon(view)}",
        f"info={int(view.complete_information)}",
        f"messages={int(view.messages_allowed)}",
    )
    scenario = "bargaining|" + "|".join(static)
    pooled_scenario = "bargaining|" + "|".join(static[:-1]) + "|messages=pooled"
    action = f"action={view.action_type}"
    proc = f"procedure={procedure or 'unknown'}"
    style = f"style={opponent.style}"
    evidence = f"samples={_sample_band(opponent.samples)}"
    state_id = "|".join((scenario, action, proc, style, evidence))
    return Segment(
        scenario_id=scenario,
        state_id=state_id,
        parents=(
            state_id,
            "|".join((scenario, action, proc)),
            "|".join((scenario, action)),
            "|".join((pooled_scenario, action, proc)),
            "|".join((pooled_scenario, action)),
        ),
        indicators={
            "role": role,
            "own_delta": my_delta(view),
            "visible_opp_delta": opp_delta(view),
            "horizon": _horizon(view),
            "complete_information": view.complete_information,
            "messages_allowed": view.messages_allowed,
            "amount": money(view),
            "action_type": view.action_type,
            "procedure": procedure,
            "opponent_style": opponent.style,
            "opponent_samples": opponent.samples,
            "rejections": opponent.own_rejections,
            "opponent_slope": opponent.concession_rate,
            "slope_uncertainty": opponent.slope_uncertainty,
            "slope_direction": slope_direction,
            "prediction_confidence": opponent.prediction_confidence,
            "plateau_span": opponent.plateau_span,
            "plateau_band": plateau_band,
            "post_rejection_gain": opponent.post_rejection_gain,
            "patience_confidence": opponent.patience_confidence,
        },
    )


def _negotiation(view: GameView, procedure: str) -> Segment:
    from glee_agent.rules import opponent_id
    from glee_agent.workflows.negotiation._tactics import state_of
    from glee_agent.workflows.negotiation.classify import own_mult

    role = role_of(view)
    mine = value_of(view) or 0.0
    mult = own_mult(view)
    base = mine / mult if mult else 0.0
    other = value_of(view, opponent_id(view.you))
    other_mult = other / base if other is not None and base else None
    opponent = state_of(view, role, mine)
    static = (
        f"role={role}",
        f"own_mult={_f(mult)}",
        f"opp_mult={_f(other_mult)}",
        f"horizon={_horizon(view)}",
        f"info={int(view.complete_information)}",
        f"messages={int(view.messages_allowed)}",
    )
    scenario = "negotiation|" + "|".join(static)
    pooled_scenario = "negotiation|" + "|".join(static[:-1]) + "|messages=pooled"
    action = f"action={view.action_type}"
    proc = f"procedure={procedure or 'unknown'}"
    style = f"style={opponent.style}"
    evidence = f"samples={_sample_band(opponent.samples)}"
    state_id = "|".join((scenario, action, proc, style, evidence))
    return Segment(
        scenario_id=scenario,
        state_id=state_id,
        parents=(
            state_id,
            "|".join((scenario, action, proc)),
            "|".join((scenario, action)),
            "|".join((pooled_scenario, action, proc)),
            "|".join((pooled_scenario, action)),
        ),
        indicators={
            "role": role,
            "own_mult": mult,
            "visible_opp_mult": other_mult,
            "horizon": _horizon(view),
            "complete_information": view.complete_information,
            "messages_allowed": view.messages_allowed,
            "amount_scale": base,
            "action_type": view.action_type,
            "procedure": procedure,
            "opponent_style": opponent.style,
            "opponent_samples": opponent.samples,
            "rejections": opponent.own_rejections,
            "zopa_probability": opponent.zopa_probability,
            "zopa_model": opponent.zopa_model,
        },
    )


def _persuasion(view: GameView, procedure: str) -> Segment:
    from glee_agent.workflows.persuasion._beliefs import current_rec, state_of

    state = view.state
    price = float(state.get("product_price") or 0.0)
    high = state.get("v")
    low = state.get("u")
    high_mult = float(high) / price if isinstance(high, (int, float)) and price else None
    low_mult = float(low) / price if isinstance(low, (int, float)) and price else None
    seller_turn = view.action_type in ("seller_message", "seller_recommendation")
    role = "seller" if seller_turn else "buyer"
    opponent = state_of(state, role="buyer" if seller_turn else "seller")
    signal = current_rec(state)
    static = (
        f"role={role}",
        f"p={_f(float(state.get('p') or 0.5))}",
        f"high_mult={_f(high_mult)}",
        f"low_mult={_f(low_mult)}",
        f"known={int(bool(state.get('is_seller_know_cv')))}",
        f"message_type={str(state.get('seller_message_type') or 'text').lower()}",
        f"myopic={int(bool(state.get('is_myopic', False)))}",
    )
    scenario = "persuasion|" + "|".join(static)
    action = f"action={view.action_type}"
    proc = f"procedure={procedure or 'unknown'}"
    style_value = opponent.buyer_style if seller_turn else opponent.seller_type
    style = f"style={style_value}"
    evidence = f"samples={_sample_band(opponent.samples)}"
    state_id = "|".join((scenario, action, proc, style, evidence))
    return Segment(
        scenario_id=scenario,
        state_id=state_id,
        parents=(
            state_id,
            "|".join((scenario, action, proc)),
            "|".join((scenario, action)),
        ),
        indicators={
            "role": role,
            "p": state.get("p"),
            "high_mult": high_mult,
            "low_mult": low_mult,
            "seller_knows_values": bool(state.get("is_seller_know_cv")),
            "message_type": state.get("seller_message_type"),
            "myopic": bool(state.get("is_myopic", False)),
            "action_type": view.action_type,
            "procedure": procedure,
            "opponent_style": style_value,
            "opponent_samples": opponent.samples,
            "signal": signal,
            "revealed_signal_accuracy": opponent.revealed_signal_accuracy,
            "revealed_signal_samples": opponent.revealed_samples,
        },
    )


def segment(view: GameView, procedure: str = "") -> Segment:
    if view.family == "bargaining":
        return _bargaining(view, procedure)
    if view.family == "negotiation":
        return _negotiation(view, procedure)
    return _persuasion(view, procedure)
