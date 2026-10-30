"""Regime × style × inflation × horizon → a named procedure, then execute."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from glee_agent.rules import (
    GameView,
    money,
    my_delta,
    offer_gain,
    opponent_id,
    opp_delta,
    rubinstein_proposer_share,
)
from glee_agent.reference import bargaining as public_bargaining_reference
from glee_agent.rating_surrogate import predict as predict_rating
from glee_agent.workflows.bargaining._helpers import (
    DEFAULT_MSG,
    MIN_CONTINUATION_REJECTIONS,
    POWERLESS_FLOOR,
    expected_even_continuation,
    fitted_offer_curve,
    fitted_response_curve,
    last_gain,
    make_offer,
    ask_share,
    opp_share,
    same_split_repeats,
    rejected_own_offers,
    response_value_model,
    we_posted_offer,
)
from glee_agent.workflows.bargaining.classify import (
    HOT,
    PATIENT,
    horizon_phase,
    inflation_band,
    info_band,
    we_hotter,
)
from glee_agent.workflows._common import below_share_floor


@dataclass(frozen=True)
class Plan:
    ask: float
    procedure: str
    message: str = DEFAULT_MSG


def cell_of(view: GameView, regime: str, style: str) -> tuple[str, str, str, str, str]:
    return (regime, style, inflation_band(view), horizon_phase(view), info_band(view))


def compose(view: GameView, regime: str, style: str) -> Plan:
    band = inflation_band(view)
    phase = horizon_phase(view)
    info = info_band(view)
    return _matrix(view, regime, style, band, phase, info)


def _matrix(
    view: GameView,
    regime: str,
    style: str,
    band: str,
    phase: str,
    info: str,
) -> Plan:
    d_me = my_delta(view)
    d_opp = opp_delta(view)
    if (
        view.action_type == "offer"
        and view.round == 1
        and view.complete_information
        and d_me is not None
        and d_opp is not None
        and 0.90 - 1e-9 <= d_me < PATIENT
        and abs(d_me - d_opp) <= 1e-9
    ):
        # Equal-discount play is symmetric. Public same-configuration games
        # accept 50/50 in 89–100% of openings, while our Rubinstein asks
        # reached rounds 2–4 and lowered discounted payoff.
        return Plan(0.50, "symmetric_open")
    if regime == "deadline":
        return _deadline_cell(view, band, regime)
    if regime == "penultimate":
        return _penultimate_cell(view, regime, style, band)
    if regime == "powerless":
        # Fixed 50% was rejected in 34/39 complete-info proposer games, which
        # then settled after heavy 0.8 discounting. A 40% opening stays above
        # the 35.86% public payoff median while giving the patient responder 60%.
        return Plan(0.40, "take_positive")
    if regime == "impatient_hot":
        return _hot_cell(view, style, band, phase)
    if regime == "impatient_vs_hotter":
        return _vs_hotter_cell(view, style, band, phase, info)
    if regime == "impatient":
        return _impatient_cell(view, style, band, phase)
    if regime in ("patient", "patient_vs_impatient"):
        return _patient_cell(view, regime, style, band, phase, info)
    if regime == "unknown_horizon":
        return _unknown_horizon_cell(view, style, band, phase, info)
    if regime == "incomplete":
        return _incomplete_cell(view, style, band, phase)
    return Plan(ask_share(view, fade=False, floor=0.53), "mild")


def _deadline_cell(view: GameView, band: str, regime: str) -> Plan:
    # Re-offer the opponent's own latest split when available. 447e5540:
    # they offered us 45.2% through r98, while our fixed 99% r99 claim was
    # rejected for no deal. Their revealed split is the strongest acceptance
    # evidence available when there is no continuation round.
    if view.action_type == "offer":
        # With an explicitly known T=12 deadline, final-round 50% offers were
        # accepted 14/14 and 99% offers 7/7, even after sub-half counters.
        # Unknown-horizon round 99 differs: 50% was rejected 4/4 there, so it
        # still reoffers the opponent's revealed boundary below.
        if view.horizon_known:
            return Plan(
                0.99,
                "deadline_claim",
                "This is the final round; a positive share is better than no agreement.",
            )
        from glee_agent.workflows.bargaining._opponent import their_shares

        seen = their_shares(view)
        ask = min(max(seen[-1], 0.01), 0.99) if seen else 0.99
        return Plan(
            ask,
            "deadline_reoffer" if seen else "deadline_claim",
            "This is the final round; a positive share is better than no agreement.",
        )
    d = my_delta(view)
    if band == "high" or regime == "powerless" or (d is not None and d <= HOT):
        return Plan(0.50, "take_positive")
    return Plan(0.50, "late")


def _penultimate_cell(view: GameView, regime: str, style: str, band: str) -> Plan:
    ask = min(ask_share(view, fade=True, floor=0.50), 0.52)
    d = my_delta(view)
    d_opp = opp_delta(view)
    # A patient proposer whose offers have repeatedly failed should close at
    # even in the penultimate round. The 51.8% branch was accepted in only
    # 3/8 matching games and averaged -5.25 rating; a final lowball followed
    # in the rejected games.
    if (
        d is not None
        and d >= PATIENT
        and d_opp is not None
        and d_opp < PATIENT
        and view.complete_information
        and rejected_own_offers(view) >= 3
    ):
        return Plan(0.50, "hold_first_even")
    if d is not None and d >= PATIENT:
        return Plan(ask, "hold_first_even")
    # 27293183: we do not inflate, took 43% on r11 (−7.3). Hold for half.
    if band == "none" and style == "conceder":
        return Plan(ask, "late")
    if style == "stubborn" and (band == "none" or (d is not None and d >= PATIENT)):
        return Plan(ask, "late")
    return Plan(ask, "late")


def _hot_cell(view: GameView, style: str, band: str, phase: str) -> Plan:
    # e5b8cdd4 / cd373867: we posted 50%, they parked 15% r2 and r4.
    # Keep this procedure so style cannot flip to take_positive; _run applies
    # the regular hot 43% early / 35–40% late floors.
    if we_posted_offer(view):
        return Plan(0.50, "hot_posted")
    if (
        view.action_type == "offer"
        and view.round == 1
        and not view.complete_information
    ):
        # Recent same-cell replays favor 52% for hidden delta-.8 Alice;
        # other open/no-message hot cells retain the supported 52% probe.
        ask = (
            0.52
            if (
                view.you == "player_1"
                and abs((my_delta(view) or 0.0) - 0.80) <= 1e-9
            )
            or (not view.horizon_known and not view.messages_allowed)
            else 0.50
        )
        return Plan(ask, "hot")
    if style == "lowballer":
        # b960847f: Bob still posted 12% they took. Proposer 50%.
        return Plan(0.50, "take_positive")
    if style == "stubborn" and phase in ("late", "penultimate"):
        return Plan(min(ask_share(view, fade=True), 0.46), "late")
    # Recent 50% immediate agreements still rank below the short-window field;
    # 55% was rejected. Start at the intermediate 52%, then hot_posted uses 50%.
    return Plan(0.52, "hot")


def _vs_hotter_cell(
    view: GameView, style: str, band: str, phase: str, info: str
) -> Plan:
    ask = ask_share(view, fade=True)
    # Complete-info 0.95 vs 0.8 opens at 55% because the generic mild cap
    # overrides its higher Rubinstein share. All 28 scored first-round closes
    # at 55% were negative; ask the existing 60% global maximum in this cell.
    if (
        view.action_type == "offer"
        and view.round == 1
        and info == "complete"
        and abs((my_delta(view) or 0.0) - 0.95) <= 1e-9
        and abs((opp_delta(view) or 0.0) - 0.80) <= 1e-9
    ):
        return Plan(0.60, "hold_first_even")
    if we_hotter(view):
        return Plan(min(ask, 0.46), "take_positive")
    if opp_delta(view) is None:
        return Plan(min(ask, 0.55), "mild")
    if phase == "last":
        return Plan(0.50, "take_positive" if band == "high" else "late")
    if info != "complete":
        return Plan(min(ask, 0.55), "mild")
    if style == "lowballer" and band == "high":
        return Plan(0.50, "take_positive")
    if style == "stubborn":
        return Plan(ask, "stall_half")
    if style == "conceder":
        # 2993bab5 / d53c74ec: conceder late 0.42 in mid took 42–46% (−7.1 / −6.65).
        # late is penultimate / unknown r≥12, not r4–r8.
        if phase in ("late", "penultimate", "last"):
            return Plan(ask, "late")
        return Plan(ask, "hold_first_even")
    # half_bot / mixed / unknown on complete: first-even is theirs to repeat
    return Plan(ask, "hold_first_even")


def _impatient_cell(view: GameView, style: str, band: str, phase: str) -> Plan:
    ask = ask_share(view, fade=True)
    if not view.complete_information:
        # The public-curve 50% opening is already the accepted-payoff optimum;
        # a later hidden-value offer must not increase above that opening.
        ask = min(ask, 0.50)
    offered = opp_share(view)
    d_me = my_delta(view)
    d_opp = opp_delta(view)
    if (
        view.action_type == "offer"
        and view.round == 1
        and view.complete_information
        and not view.horizon_known
        and d_me is not None
        and 0.90 - 1e-9 <= d_me <= 0.95 + 1e-9
        and d_opp is not None
        and d_opp >= PATIENT
    ):
        # The current 50% opening was rejected 5/5 before 15–40% counters.
        # Public same-opponent 47% openings were accepted 3/3.
        return Plan(0.47, "mild")
    if (
        view.action_type == "offer"
        and view.you == "player_2"
        and view.round == 2
        and not view.complete_information
        and view.horizon_known
        and view.max_rounds == 12
        and d_me is not None
        and abs(d_me - 0.90) <= 1e-9
    ):
        # 419791e6 and 35 matching games: 50% counters were accepted 31/36
        # but averaged -1.91 rating. Probe the smallest higher grid point.
        return Plan(0.52, "mild")
    if (
        view.action_type == "offer"
        and view.you == "player_2"
        and view.round == 2
        and not view.complete_information
        and view.horizon_known
        and view.max_rounds == 12
        and d_me is not None
        and abs(d_me - 0.95) <= 1e-9
    ):
        # Public response curve: 53% was accepted in 11/15 matching games;
        # 50% was accepted in 63/135, giving much lower immediate EV.
        return Plan(0.53, "mild")
    # 359f3aa7 / 710ee941: with no deadline, a fully patient opponent can
    # delay our 50% agreement until discounting makes an early 35%+ counter
    # strictly better. Only apply this after they have rejected our offer.
    if (
        view.action_type == "decision"
        and view.complete_information
        and not view.horizon_known
        and d_me is not None
        and d_me <= 0.90 + 1e-9
        and d_opp is not None
        and d_opp >= PATIENT
        and rejected_own_offers(view) >= 1
        and offered is not None
        and offered >= 0.35
    ):
        return Plan(ask, "patient_counter")
    if view.action_type == "offer" and view.round == 1 and not view.complete_information:
        # Alice delta-.9 uses the observed 52% improvement in all four
        # horizon/message cells; other hidden impatient openings keep 50%.
        if view.you == "player_1" and abs((d_me or 0.0) - 0.90) <= 1e-9:
            return Plan(0.52, "mild")
        return Plan(0.50, "mild")
    if not view.horizon_known and view.round >= 12:
        return Plan(ask, "late")
    if style == "conceder" and phase in ("late", "penultimate"):
        return Plan(ask, "late")
    if style == "stubborn" and phase == "early":
        return Plan(ask, "stall_half")
    return Plan(ask, "mild")


def _patient_cell(
    view: GameView,
    regime: str,
    style: str,
    band: str,
    phase: str,
    info: str,
) -> Plan:
    ask = ask_share(view, fade=False, floor=0.53)
    offered = opp_share(view)
    d_me = my_delta(view)
    d_opp = opp_delta(view)
    # With our delta at one, a rejected probe costs no payoff. Ask 60% once
    # to separate impatient hidden opponents, then return to the 55% field
    # fallback. Historical 55% closes score poorly against delta-0.8 agents,
    # while 57–64% closes improve the 0.8/0.9/0.95 branches.
    if (
        regime == "patient"
        and view.action_type == "offer"
        and not view.complete_information
        and view.round <= 3
        and rejected_own_offers(view) == 0
    ):
        return Plan(0.60, "patient_probe")
    if (
        regime == "patient_vs_impatient"
        and view.action_type == "offer"
        and view.round == 1
        and d_me is not None
        and d_me >= PATIENT
        and d_opp is not None
        and d_opp < PATIENT
        and rejected_own_offers(view) == 0
    ):
        ask = max(ask, 0.65)
    if (
        view.action_type == "offer"
        and view.horizon_known
        and view.max_rounds is not None
        and view.max_rounds - view.round <= 4
        and d_me is not None
        and d_me >= PATIENT
        and d_opp is not None
        and d_opp < PATIENT
        and rejected_own_offers(view) >= 3
    ):
        from glee_agent.workflows.bargaining._opponent import their_shares

        seen = their_shares(view)
        if len(seen) >= 3 and all(0.49 <= share <= 0.51 for share in seen[-3:]):
            return Plan(0.52, "hold_first_even")
    # dd8dc13a: δ=1 accepted the opponent's repeated 50% at r6 (−5.0),
    # while the matching public field final-share median is 52%.
    if (
        d_me is not None
        and d_me >= PATIENT
        and d_opp is not None
        and d_opp < PATIENT
        and offered is not None
        and offered >= 0.50
        and same_split_repeats(view) >= 2
    ):
        return Plan(0.52, "hold_first_even")
    # A fully patient proposer can wait without losing payoff, so conceding
    # below 60% only lowers our field percentile. The two latest 52% closes
    # scored -3.5/-2.9; matching 60% closes stay positive even at round 41.
    # Preserve the round-50 even-split fallback for opponents that never close.
    if (
        regime == "patient_vs_impatient"
        and view.action_type == "offer"
        and view.complete_information
        and not view.horizon_known
        and view.round < 50
        and d_me is not None
        and d_me >= PATIENT
        and d_opp is not None
        and d_opp < PATIENT
        and rejected_own_offers(view) >= 1
    ):
        return Plan(0.60, "patient_search")
    # 322126f3 / cb84b406: unknown-horizon patient hold_first_even parked
    # 45–48% for 99 rounds (−7.6 / −7.47). Impatient already uses late at r≥12.
    if not view.horizon_known and view.round >= 50 and view.action_type == "offer":
        return Plan(0.50, "hold_first_even")
    if not view.horizon_known and view.round >= 12:
        if offered is not None and offered < 0.50:
            return Plan(0.50, "hold_first_even")
        return Plan(ask, "late")
    if style == "lowballer" and band == "high":
        return Plan(0.50, "take_positive")
    if style == "lowballer" and phase == "early":
        return Plan(ask, "stall_half")
    if style == "stubborn":
        if phase in ("late", "penultimate", "last"):
            return Plan(ask, "late")
        return Plan(ask, "stall_half")
    if style == "conceder" and band == "none" and phase == "penultimate":
        return Plan(min(ask, 0.52), "late")
    near = band in ("none", "mild")
    if near and style in ("half_bot", "mixed", "unknown"):
        return Plan(ask, "hold_first_even")
    if near:
        return Plan(ask, "hold_first_even")
    return Plan(ask, "hold_first_even")


def _unknown_horizon_cell(
    view: GameView, style: str, band: str, phase: str, info: str
) -> Plan:
    # unknown horizon + incomplete + δ_me=1 must not ask 70% (capped 0.55)
    share = min(ask_share(view, fade=True), 0.55)
    if view.round >= 12 or phase == "late":
        return Plan(min(share, 0.52), "late")
    if style == "half_bot":
        return Plan(min(share, 0.52), "stall_half")
    if style == "stubborn":
        return Plan(min(share, 0.52), "stall_half")
    if style == "lowballer":
        return Plan(share, "mild")
    if info == "incomplete" and band == "none":
        return Plan(share, "mild")
    return Plan(share, "mild")


def _incomplete_cell(view: GameView, style: str, band: str, phase: str) -> Plan:
    ask = min(ask_share(view, fade=True), 0.55)
    if style == "half_bot":
        return Plan(ask, "stall_half")
    if style == "stubborn" and phase != "early":
        return Plan(ask, "late")
    return Plan(ask, "mild")


def execute(view: GameView, plan: Plan) -> dict[str, Any]:
    if view.action_type == "offer":
        fitted = (
            None
            if plan.procedure == "symmetric_open"
            else fitted_offer_curve(view, plan.ask, plan.procedure)
        )
        share = float(fitted["share"]) if fitted is not None else plan.ask
        return make_offer(view, share, plan.message)
    return _run(view, plan)


def _run(view: GameView, plan: Plan) -> dict[str, Any]:
    from glee_agent.decision_trace import set_node

    procedure = plan.procedure
    # This scenario node covers every response leaf; safety/model nodes below
    # replace it when a cross-scenario rule owns the final decision.
    set_node(f"bargaining.response.scenario.{procedure}")
    pot = money(view)
    my_gain = last_gain(view)
    if my_gain is None:
        set_node("bargaining.response.safety.missing_offer")
        return {"decision": "walkaway" if view.is_last_round else "reject"}
    share = my_gain / pot if pot else 0.0
    if view.is_last_round:
        if my_gain <= 0:
            set_node("bargaining.response.safety.terminal_nonpositive")
            return {"decision": "walkaway"}
        # c88bf3b2 / 9cb9241b: walking 20% / 10% produced zero payoff and
        # bottom-percentile scores. With no future round, any positive dominates.
        set_node("bargaining.response.safety.terminal_positive")
        return {"decision": "accept"}
    # Bottom-field offers must never pass a later empirical "positive offer"
    # rule. All 58 recorded nonterminal closes at <=10% scored non-positive
    # (rating sum -104.6), including four floating-point-zero agreements.
    if pot > 0 and my_gain / pot <= 0.10 + 1e-9:
        set_node("bargaining.response.safety.nonterminal_bottom_offer")
        return {"decision": "reject"}
    delta = my_delta(view)
    rejections = rejected_own_offers(view)
    from glee_agent.workflows.bargaining._opponent import state_of, their_shares

    opponent = state_of(view)
    seen_opponent_shares = their_shares(view)
    other_gain = offer_gain(
        view.state.get("last_offer"), opponent_id(view.you)
    )
    d_opp_now = opp_delta(view)
    if (
        view.complete_information
        and delta is not None
        and d_opp_now is not None
        and d_opp_now > delta + 1e-9
        and not view.horizon_known
        and opponent.samples <= 3
        and (
            (
                delta <= 0.80 + 1e-9
                and view.round <= 5
                and rejections >= 2
                and share < 0.45
            )
            or (
                abs(delta - 0.90) <= 1e-9
                and d_opp_now >= 1.0 - 1e-9
                and view.round <= 3
                and rejections >= 1
                and share < 0.20
            )
        )
        and other_gain is not None
        and pot > 0
        and share * delta ** max(0, view.round - 1)
        < (0.50 if delta <= 0.80 + 1e-9 else 0.25)
        * (other_gain / pot)
        * d_opp_now ** max(0, view.round - 1)
    ):
        set_node("bargaining.response.safety.extreme_relative_hold")
        return {"decision": "reject"}
    if (
        view.you == "player_2"
        and delta is not None
        and abs(delta - 0.95) <= 1e-9
        and not view.complete_information
        and view.horizon_known
        and view.max_rounds == 12
        and view.round <= 5
        and rejections >= 2
        and share < 0.30
    ):
        set_node("bargaining.response.safety.hidden_extreme_hold")
        return {"decision": "reject"}
    # a0575327 / 15f9e7c0 / 725b99a2: a fully patient opponent can reject our
    # 40/60 offer and answer with 0-31.36%, producing -8.02, -7.8, and -7.1.
    # Keep the profitable 39.09% response band, but do not reward an extreme
    # first undercut. Re-offer 40% once; if it repeats after two rejected own
    # offers, leave so the opponent cannot retain the large asymmetric payoff.
    if (
        view.you == "player_1"
        and delta is not None
        and delta <= 0.80 + 1e-9
        and d_opp_now is not None
        and d_opp_now >= 1.0 - 1e-9
        and view.complete_information
        and view.horizon_known
        and view.max_rounds == 12
        and share < 0.38
        and rejections >= 1
    ):
        if rejections >= 2:
            set_node("bargaining.response.safety.powerless_repeat_undercut_hold")
            return {"decision": "reject"}
        set_node("bargaining.response.safety.powerless_first_undercut_hold")
        return {"decision": "reject"}
    # e0d63d63: accepting 46% scored -2.6. Keep a narrow rating floor ahead
    # of the economic next-plan check; 8f821041 remains an observed exception
    # at 47.37%, where waiting produced only 45% after discounting.
    if (
        view.you == "player_2"
        and delta is not None
        and abs(delta - 0.90) <= 1e-9
        and not view.complete_information
        and view.horizon_known
        and view.max_rounds == 12
        and view.round == 1
        and plan.ask <= 0.50 + 1e-9
        and share < 0.47
    ):
        set_node("bargaining.response.safety.hidden_t12_opening_rating_floor")
        return {"decision": "reject"}
    # 1f2530b7 / a5a5d616: hidden open-horizon hot Bob openings below
    # 45% scored negative; the three exact 45% closes all scored positive.
    if (
        view.you == "player_2"
        and delta is not None
        and abs(delta - 0.80) <= 1e-9
        and not view.complete_information
        and not view.horizon_known
        and view.round == 1
        and plan.ask <= 0.52 + 1e-9
        and 0.43 <= share < 0.45
    ):
        set_node("bargaining.response.safety.hidden_hot_opening_rating_floor")
        return {"decision": "reject"}
    # e2c0cc7d and 13 matching replays: after four failed offers, a late
    # 45-48% counter that already dominates modeled continuation closed with
    # no observed replay regressions. Earlier rounds retain the 48% floor.
    if (
        delta is not None
        and abs(delta - 0.95) <= 1e-9
        and not view.complete_information
        and view.horizon_known
        and view.max_rounds == 12
        and 8 <= view.round < view.max_rounds - 1
        and rejections >= 4
        and 0.45 <= share < 0.48
    ):
        late_model = response_value_model(view)
        if (
            late_model is not None
            and float(late_model.get("accept_value") or 0.0)
            >= float(late_model.get("continuation_value") or 0.0)
            + float(late_model.get("guard") or 0.005)
        ):
            set_node("bargaining.response.model.hidden_t12_late_value_stop")
            return {"decision": "accept"}
    # 0a0ee944 / e0f23995 / 8731e985 / d4cf975c: in hidden T=12
    # delta-.95 games, model acceptance below 48% repeatedly produced severe
    # losses. This observed scenario floor owns the decision before learned
    # continuation models; matching rejections later closed our 53% counter.
    if (
        delta is not None
        and abs(delta - 0.95) <= 1e-9
        and not view.complete_information
        and view.horizon_known
        and view.max_rounds is not None
        and view.round < view.max_rounds - 1
        and share < 0.48
    ):
        set_node("bargaining.response.safety.hidden_t12_patient_floor")
        return {"decision": "reject"}
    # 80db7561 and nine matching replays: a learned accept in the
    # penultimate round must not replace a certain next-round half claim.
    discounted_now = (
        share * (delta ** max(0, view.round - 1))
        if delta is not None
        else None
    )
    if (
        delta is not None
        and delta < 0.99
        and view.horizon_known
        and view.max_rounds is not None
        and view.round == view.max_rounds - 1
        and discounted_now is not None
        and discounted_now + 1e-9 < 0.50 * (delta**view.round)
    ):
        set_node("bargaining.response.safety.penultimate_half_floor")
        return {"decision": "reject"}
    if (
        view.you == "player_2"
        and delta is not None
        and delta >= 1.0 - 1e-9
        and view.horizon_known
        and view.max_rounds == 12
        and view.round == view.max_rounds - 1
        and share < 0.99
    ):
        # 56c80261 and six matching closes: patient Bob owns the final offer
        # and the 99% claim was accepted in every observed matching replay.
        set_node("bargaining.response.safety.patient_bob_final_claim")
        return {"decision": "reject"}
    # Compare like-for-like discounted payoffs. If accepting now is at least as
    # valuable as our own next planned claim even when that claim is accepted
    # immediately, rejecting contradicts the workflow's declared continuation.
    if delta is not None and share + 1e-9 >= delta * plan.ask:
        set_node("bargaining.response.model.planned_continuation")
        return {"decision": "accept"}
    fitted = fitted_response_curve(view, procedure)
    if fitted is not None and fitted["decision"] == "accept":
        set_node("bargaining.response.model.fitted_curve")
        return {"decision": "accept"}
    modeled = response_value_model(view)
    if modeled is not None and modeled["decision"] == "accept":
        set_node("bargaining.response.model.continuation_value")
        return {"decision": "accept"}
    if (
        modeled is not None
        and float(modeled.get("accept_value") or 0.0) <= 0.20
        and float(modeled.get("accept_value") or 0.0)
        + float(modeled.get("guard") or 0.005)
        < float(modeled.get("continuation_value") or 0.0)
    ):
        set_node("bargaining.response.safety.low_value_dominated")
        return {"decision": "reject"}
    # 99a70288: 25 -> 40 -> 38 -> 40 used one old jump to inflate the
    # continuation forecast, then returned to the same anchor while our 10%
    # discount compounded. This observed pattern owns the decision before a
    # continuation model that still contains the stale 15-point jump.
    if (
        delta is not None
        and abs(delta - 0.90) <= 1e-9
        and not view.complete_information
        and opponent.samples >= 4
        and rejections >= 3
        and opponent.current_share is not None
        and opponent.best_share is not None
        and opponent.current_share >= opponent.best_share - 0.005
        and max(seen_opponent_shares) - min(seen_opponent_shares) >= 0.10
        and opponent.slope_uncertainty >= 0.03
        and 0.35 <= share <= 0.45
    ):
        set_node("bargaining.response.model.oscillating_anchor_stop")
        return {"decision": "accept"}
    field_floor = (public_bargaining_reference(view) or {}).get(
        "median_payoff_share"
    )
    if (
        modeled is not None
        and isinstance(field_floor, (int, float))
        and float(modeled.get("accept_value") or 0.0) <= float(field_floor) - 0.10
        and float(modeled.get("accept_value") or 0.0)
        + float(modeled.get("guard") or 0.005)
        < float(modeled.get("continuation_value") or 0.0)
    ):
        set_node("bargaining.response.safety.rating_floor_dominated")
        return {"decision": "reject"}
    rating_risk = predict_rating(
        view,
        view.state.get("last_offer") or {},
        response=True,
    )
    if (
        modeled is not None
        and float(modeled.get("accept_value") or 0.0)
        + float(modeled.get("guard") or 0.005)
        < float(modeled.get("continuation_value") or 0.0)
        and rating_risk is not None
        and rating_risk["positive_probability"] < 0.30
        and rating_risk["expected_rating_delta"] < 0.0
    ):
        set_node("bargaining.response.risk.rating_surrogate_veto")
        return {"decision": "reject"}
    d_opp = opp_delta(view)
    continuation = expected_even_continuation(view)
    repeats = same_split_repeats(view)

    first_opponent_offer = len(their_shares(view)) == 1
    # 1789e730: a fully patient responder accepted the opponent's first 50%
    # immediately after a 30% plateau. The 20-point jump is fresh concession
    # evidence, not a stable boundary; hold 55% once before the repeat-based
    # close rules are allowed to end the negotiation.
    if (
        delta is not None
        and delta >= 1.0 - 1e-9
        and not view.horizon_known
        and opponent.samples >= 2
        and opponent.plateau_repeats == 1
        and opponent.post_rejection_gain is not None
        and opponent.post_rejection_gain >= 0.05
        and 0.50 <= share < 0.55
    ):
        set_node("bargaining.response.safety.patient_fresh_concession_hold")
        return {"decision": "reject"}
    # a8a4ba3b: with zero discount and no deadline, the opponent was still
    # conceding from 39% toward our 55% ask, but late accepted 50.66% on
    # round 35. A moving boundary is not an observed reservation point.
    if (
        delta is not None
        and delta >= 1.0 - 1e-9
        and not view.horizon_known
        and opponent.samples >= 3
        and rejections >= 2
        and opponent.prediction_confidence >= 0.66
        and opponent.concession_rate > 0.005
        and share < 0.55
    ):
        set_node("bargaining.response.safety.patient_moving_boundary_hold")
        return {"decision": "reject"}
    # 97081023 / 6a9509e5: a slow 0-1.6% concession did not repay our 10%
    # per-round discount. Once two opponent offers and one rejected own offer
    # expose that trajectory, take the current offer when it already exceeds
    # the optimistic trajectory continuation.
    if (
        delta is not None
        and 0.80 - 1e-9 <= delta <= 0.90 + 1e-9
        and (delta > 0.80 + 1e-9 or not view.horizon_known)
        and opponent.samples >= 2
        and rejections >= 1
        and 0.0 <= opponent.concession_rate <= 0.016 + 1e-9
        and 0.35 <= share <= 0.42
        and (delta > 0.80 + 1e-9 or share > 0.38 + 1e-9)
        and modeled is not None
        and float(modeled.get("accept_value") or 0.0)
        >= float(modeled.get("continuation_value") or 0.0)
        + float(modeled.get("guard") or 0.005)
    ):
        set_node("bargaining.response.model.slow_concession_discount_stop")
        return {"decision": "accept"}
    # 4acdefb0: equal hot players, our 52% counter was rejected, then accepting
    # the opponent's 38.87% round-3 counter yielded only 24.88% discounted
    # value and -7.4 rating. Do not let hot_posted close this severe undercut.
    if (
        delta is not None
        and d_opp is not None
        and abs(delta - 0.80) <= 1e-9
        and abs(d_opp - 0.80) <= 1e-9
        and view.complete_information
        and view.horizon_known
        and view.max_rounds == 12
        and view.round == 3
        and rejections >= 1
        and share < 0.45
    ):
        set_node("bargaining.response.safety.equal_hot_round3_undercut")
        return {"decision": "reject"}
    # a7020739 / 6a50714f: with known 0.9 vs 0.8 discounting and no horizon,
    # accepting the opponent's slow 39.6 -> 41.2 -> 42.4% concession after
    # two rejected 55% offers scored -7.3 twice. The less patient opponent
    # must cross a competitive floor instead of winning through delay.
    if (
        delta is not None
        and d_opp is not None
        and abs(delta - 0.90) <= 1e-9
        and abs(d_opp - 0.80) <= 1e-9
        and view.complete_information
        and not view.horizon_known
        and rejections >= 2
        and share < 0.45
    ):
        set_node("bargaining.response.safety.hotter_opponent_delay_floor")
        return {"decision": "reject"}
    # Bob sees the opponent's first offer in round 1, whereas Alice normally
    # sees it in round 2 after proposing. Route on observed state instead of
    # an absolute round number. Against a fully patient known opponent, 53/59
    # historical 30-40% closes beat continuation across both horizon types.
    early_patient_close = bool(
        view.you == "player_2"
        and first_opponent_offer
        and view.complete_information
        and delta is not None
        and 0.90 - 1e-9 <= delta <= 0.95 + 1e-9
        and d_opp is not None
        and abs(d_opp - 1.0) <= 1e-9
        and 0.30 <= share < 0.40
    )
    # A hot Bob facing a known more-patient opponent has a separate narrow
    # opening band: five of six local/public cross-checks improved by closing
    # at 28-30%, while nearby 32% includes a counterexample.
    early_hot_close = bool(
        view.you == "player_2"
        and first_opponent_offer
        and view.complete_information
        and not view.horizon_known
        and delta is not None
        and delta <= HOT
        and d_opp is not None
        and d_opp > delta + 1e-9
        and 0.28 <= share <= 0.30
    )
    if early_patient_close or early_hot_close:
        return {"decision": "accept"}
    if (
        procedure == "mild"
        and view.complete_information
        and delta is not None
        and d_opp is not None
        and delta < d_opp
        and rejections >= 1
    ):
        responder_spe = 1.0 - rubinstein_proposer_share(d_opp, delta)
        if abs(share - responder_spe) <= 0.011:
            return {"decision": "accept"}
    if procedure == "patient_counter":
        return {"decision": "accept" if share >= 0.35 else "reject"}
    # Complete hidden-horizon delta-.95 play had a gap between the existing
    # 10-30% lowball stop and 39%+ patient-counter stop. Across nine first
    # counters in the 30-39% band, accepting improved five and tied four
    # (+.168 normalized); 242d6341 otherwise waited from 38% to 34.3% payoff.
    if (
        delta is not None
        and 0.94 <= delta <= 0.95 + 1e-9
        and d_opp is not None
        and d_opp >= PATIENT
        and view.complete_information
        and not view.horizon_known
        and view.round == 2
        and rejections >= 1
        and 0.30 <= share < 0.39
    ):
        return {"decision": "accept"}
    # A fully patient pair can otherwise stall at the platform's hidden cap.
    # Close only after the visible opponent model observes a stable 45–48%
    # boundary and repeated rejection of our own offers. Visible impatient
    # opponents remain excluded because their bids can improve without cost.
    if (
        delta is not None
        and delta >= PATIENT
        and not view.horizon_known
        and opponent.estimated_patience >= PATIENT
        and opponent.plateau_repeats >= 4
        and opponent.own_rejections >= 4
        and 0.45 <= share < 0.48
    ):
        return {"decision": "accept"}
    # A fully patient known opponent does not need to improve after rejecting
    # our opening, while our 0.8 discount compounds immediately. Across 22
    # hidden-horizon first counters at 11%+, accepting improved 11 and tied 11
    # (+1.076 normalized); e5005e3a otherwise fell from 9.24% to 3.37% payoff.
    if (
        delta is not None
        and delta <= HOT
        and d_opp is not None
        and d_opp >= PATIENT
        and view.complete_information
        and not view.horizon_known
        and view.round == 2
        and rejections >= 1
        and share >= 0.11
    ):
        return {"decision": "accept"}
    # A more patient known opponent does not improve enough to repay hot 0.8
    # discounting after rejecting our opening. Taking its first positive
    # counter gained +2.167 across 45 games. Delta 0.9 stays restricted to a
    # fully patient opponent (+1.314 across 19 games).
    if (
        delta is not None
        and d_opp is not None
        and (
            (delta <= HOT and d_opp > delta + 1e-9)
            or (delta <= 0.90 + 1e-9 and d_opp >= PATIENT)
        )
        and view.complete_information
        and view.horizon_known
        and view.round == 2
        and rejections >= 1
        and share > 0
    ):
        return {"decision": "accept"}
    # Hidden T=12 opponents that reject our opening require a rating-aware
    # floor. Delta-.9 closes below 39% were consistently bottom-field despite
    # sometimes beating the eventual raw payoff; 39-40% is near-neutral and
    # public 40% continuations favor closing. The hotter .8 cell keeps its
    # separate payoff-supported stop.
    if (
        delta is not None
        and not view.complete_information
        and view.horizon_known
        and view.max_rounds == 12
        and view.round == 2
        and rejections >= 1
        and (
            (delta <= HOT and share >= 0.10)
            or (abs(delta - 0.90) <= 1e-9 and 0.39 <= share <= 0.40)
        )
    ):
        return {"decision": "accept"}
    # With no posted deadline, an equally or more patient known opponent does
    # not justify paying one round of our discount for the same half split.
    # Across 84 opening responses, taking a share that already beats next
    # round's discounted 50% gained +1.414 total; worse-patience opponents are
    # excluded because that direction regressed.
    if (
        delta is not None
        and delta < 1.0
        and d_opp is not None
        and delta <= d_opp + 1e-9
        and view.complete_information
        and not view.horizon_known
        and view.round == 1
        and share + 1e-9 >= 0.50 * delta
    ):
        return {"decision": "accept"}
    # A hot responder facing a more patient known opponent should not wait for
    # a nominally larger split that loses more to discounting. Across 35 T=12
    # opening responses, taking at least the Rubinstein responder share with a
    # 30% field floor improved 17, tied 10, and regressed 8 (+2.360 total).
    if (
        delta is not None
        and delta <= HOT
        and d_opp is not None
        and d_opp > delta + 1e-9
        and view.complete_information
        and view.horizon_known
        and view.round == 1
    ):
        responder_spe = 1.0 - rubinstein_proposer_share(d_opp, delta)
        if share + 1e-9 >= max(0.30, responder_spe):
            return {"decision": "accept"}
    # With a known deadline but hidden opponent discount, take an opening that
    # already beats next round's discounted half: 26/106 improved, 78 tied,
    # and two regressed (+1.225 total, worst -.016). Delta 0.95 additionally
    # holds for 50%: all ten scored 48–49.9% closes were negative, while 50%+
    # averaged positive. Opening 38% is excluded: four hidden T=12 delta-0.95
    # acceptances scored -7.1, -6.1, -3.5, and -3.1; nearby 36.8-40% rejections
    # recovered 47.5% in round 2 with materially smaller losses.
    if (
        delta is not None
        and delta < 1.0
        and view.horizon_known
        and not view.complete_information
        and rejections == 0
        and (
            share + 1e-9 >= max(0.50 * delta, 0.50 if delta >= 0.94 else 0.0)
            or (delta <= 0.80 + 1e-9 and abs(share - 0.38) <= 1e-9)
        )
    ):
        return {"decision": "accept"}
    # An unknown posted horizon still has the platform cap. On the first
    # hidden opponent offer, taking 48%+ never regressed across 67 cached
    # delta-0.8/0.9/0.95 games (12 improved, 55 tied, +.315 normalized).
    # Delta 0.8 additionally takes a 10–38% first counter. All four cached
    # rejections in that band ended with a lower discounted payoff; nearby
    # 39% and zero-value crumbs remain excluded.
    if (
        delta is not None
        and delta <= 0.95 + 1e-9
        and not view.horizon_known
        and not view.complete_information
    ):
        first_hidden_offer = len(their_shares(view)) == 1
        exact_hot_stop = abs(delta - 0.80) <= 1e-9 and 0.38 <= share < 0.39
        low_counter_stop = (
            abs(delta - 0.80) <= 1e-9
            and rejections >= 1
            and 0.10 <= share < 0.38
        )
        if first_hidden_offer and (share >= 0.48 or exact_hot_stop or low_counter_stop):
            return {"decision": "accept"}
    # Against a fully patient known opponent, the exact 38% opening is worth
    # taking at delta 0.9. All six matching T=12 replays improved, adding .726
    # normalized payoff; the nearby 39.2% cell has a counterexample.
    if (
        delta is not None
        and abs(delta - 0.90) <= 1e-9
        and d_opp is not None
        and d_opp >= PATIENT
        and view.horizon_known
        and view.complete_information
        and abs(share - 0.38) <= 1e-9
    ):
        from glee_agent.workflows.bargaining._opponent import their_shares

        if len(their_shares(view)) == 1:
            return {"decision": "accept"}
    # With a known deadline and a strictly more patient opponent, rejecting
    # 42% can expose an impatient responder to repeated lower counters. Across
    # 11 matching cached replays this stop gained .751 normalized payoff in
    # total; the four regressions were at most .030 each.
    if (
        delta is not None
        and d_opp is not None
        and delta <= 0.90 + 1e-9
        and d_opp > delta + 1e-9
        and view.horizon_known
        and view.complete_information
        and rejections == 0
        and share >= 0.42
    ):
        return {"decision": "accept"}
    # Equal 0.95 discounting makes a 48% opening preferable to a long deadline
    # search. After our opener is rejected, take the opponent's first 45%+
    # counter: all seven cached matches improved or tied (+.536 normalized).
    # Fully patient play stays excluded because waiting does not discount it.
    if (
        delta is not None
        and d_opp is not None
        and 0.94 <= delta < 0.99
        and abs(d_opp - delta) <= 1e-9
        and view.complete_information
        and (
            (view.horizon_known and rejections == 0 and share >= 0.48)
            or (view.round == 2 and rejections == 1 and share >= 0.45)
        )
    ):
        return {"decision": "accept"}
    # With complete information and no discount on either side, waiting does
    # not reduce our payoff. Three observed 45.2% acceptances scored -6.2,
    # -5.2, and -2.15, while matching rejections later obtained 47–53%.
    if (
        delta is not None
        and d_opp is not None
        and delta >= PATIENT
        and d_opp >= PATIENT
        and share < 0.50
    ):
        return {"decision": "reject"}
    # Once a more patient complete-info opponent has rejected our offer, its
    # 39%+ counter is preferable to further discounting. Delta 0.9 improved
    # seven of eight matches (+.856 normalized); delta 0.95 improved or tied
    # all six matches (+.201 normalized).
    if (
        delta is not None
        and d_opp is not None
        and 0.90 - 1e-9 <= delta <= 0.95 + 1e-9
        and d_opp > delta + 1e-9
        and view.horizon_known
        and view.complete_information
        and rejections == 1
        and share >= 0.39
    ):
        return {"decision": "accept"}
    # A hidden opponent that rejects our first offer and immediately counters
    # at 39–41% behaves like the patient complete-info branch above. Across 22
    # known/unknown-horizon replays, accepting improved 17 and tied five, for
    # +1.248 normalized payoff.
    if (
        delta is not None
        and abs(delta - 0.90) <= 1e-9
        and not view.complete_information
        and view.round <= 3
        and rejections == 1
        and 0.39 <= share <= 0.41
    ):
        return {"decision": "accept"}
    # After one rejected offer, take a counter that already beats a certain
    # next-round 50/50 agreement. Historical replay: 13 improvements, one
    # tie, and no regressions across the 14 triggered games.
    if (
        delta is not None
        and delta < 0.99
        and rejections >= 1
        and discounted_now is not None
        and discounted_now + 1e-9 >= 0.50 * (delta**view.round)
    ):
        return {"decision": "accept"}
    # After our hot opening was rejected, a known-deadline counter can still
    # dominate the behavior-weighted continuation even below the fixed 42%
    # early floor. 04b30fce / 3ad00fa4 both lost over .22 normalized payoff by
    # postponing a 36.7–39% counter until round 8.
    if (
        procedure == "hot_posted"
        and view.horizon_known
        and delta is not None
        and delta <= HOT
        and rejections >= 1
        and share >= 0.35
        and discounted_now is not None
        and continuation is not None
        and discounted_now + 1e-9 >= continuation
    ):
        return {"decision": "accept"}
    # High discount loss makes a fixed 40% nominal floor incorrect after the
    # opponent has rejected two of our offers. Across 55 historical games,
    # taking 20%+ when it beats behavior-weighted continuation improved 41,
    # tied five, and had one negligible payoff regression.
    if (
        delta is not None
        and delta <= HOT
        and rejections >= 2
        and share >= 0.20
        and discounted_now is not None
        and continuation is not None
        and discounted_now + 1e-9 >= continuation
    ):
        return {"decision": "accept"}
    # Once two of our offers have been rejected, compare the certain offer
    # with the behavior-weighted continuation instead of assuming the next
    # 50/50 counter will be accepted. Across cached replays, the 40% stop
    # improved all 23 triggered outcomes; 42% covered only 18.
    if (
        delta is not None
        and delta < 0.99
        and rejections >= 2
        and share >= 0.40
        and discounted_now is not None
        and continuation is not None
        and discounted_now + 1e-9 >= continuation
    ):
        return {"decision": "accept"}
    # An unknown horizon still has a finite platform cap. After eight identical
    # counters and eight rejected offers, the opponent's boundary is observed;
    # e09dcd1a / 87b9a2da otherwise reached the cap with zero payoff.
    if (
        delta is not None
        and delta < 0.99
        and not view.horizon_known
        and rejections >= 8
        and repeats >= 8
        and share >= 0.30
        and discounted_now is not None
        and continuation is not None
        and discounted_now + 1e-9 >= continuation
    ):
        return {"decision": "accept"}
    if (
        procedure == "hot_posted"
        and not view.horizon_known
        and delta is not None
        and rejections >= 1
        and discounted_now is not None
        and discounted_now + 1e-9 >= 0.50 * (delta**view.round)
    ):
        return {"decision": "accept"}
    if (
        delta is not None
        and delta <= HOT
        and procedure != "hot_posted"
        and not (procedure == "hot" and view.horizon_known)
        and (
            rejections >= 1
            or (
                procedure in ("hot", "hot_posted")
                and (view.horizon_known or view.complete_information or share >= 0.40)
            )
        )
        and continuation is not None
        and (
            share >= (0.20 if procedure == "take_positive" else 0.30)
            or procedure in ("hot", "hot_posted")
        )
    ):
        if discounted_now is not None and discounted_now + 1e-9 >= continuation:
            return {"decision": "accept"}
    if procedure == "take_positive":
        if my_gain <= 0:
            return {"decision": "reject"}
        d_me = my_delta(view)
        d_opp = opp_delta(view)
        powerless = (
            d_me is not None and d_opp is not None and d_me <= HOT and d_opp >= PATIENT
        )
        # 66c9656e: 5% r1 reject. Powerless can eventually accept ≥8%; zero
        # crumbs (90ac85a2 / d914fe8d) remain rejects. Last already returned.
        if powerless:
            # A fully patient opponent accepted our round-2 40% counter only
            # once in nine matching T=12 games. Taking its initial 25–29%
            # offer improved 8/9 cross-agent replays (+.479 normalized).
            # The adjacent 20% and 22% cells both benefited from rejecting.
            if (
                view.round == 1
                and view.complete_information
                and view.horizon_known
                and view.max_rounds == 12
                and 0.25 <= share < 0.30
            ):
                return {"decision": "accept"}
            # 36c90b36 / 2853bda6: one or two rejected even-split offers do
            # not establish that 16–20% is the best attainable agreement.
            if rejections < MIN_CONTINUATION_REJECTIONS and share < 0.35:
                return {"decision": "reject"}
            if delta is not None and continuation is not None:
                discounted_now = share * (delta ** max(0, view.round - 1))
                if discounted_now + 1e-9 < continuation:
                    return {"decision": "reject"}
            if below_share_floor(my_gain, pot, POWERLESS_FLOOR):
                return {"decision": "reject"}
            return {"decision": "accept"}
        if view.round <= 3 and below_share_floor(my_gain, pot, 0.50):
            return {"decision": "reject"}
        return {"decision": "accept"}
    if procedure in ("hot", "hot_posted"):
        if repeats >= 2 and share >= 0.50:
            return {"decision": "accept"}
        # 904ce545 / a0eb38eb: known-horizon r1 39% / 38.1% scored poorly.
        # 42% matches the discounted payoff of a next-round 52% agreement.
        # f6bfe3d6: r4 40% after we had posted 50% (−5.9). 5bf1c208 /
        # 81d0a4d1: r1 45%. d197c013 / 31f24dd6: r1 44% reject, same
        # 44% taken at r7 under δ=0.8 (115k vs 440k).
        hot_floor = (
            0.40
            if (not view.horizon_known and not view.complete_information)
            else 0.35
        )
        if view.round <= 6:
            hot_floor = 0.42
        if below_share_floor(my_gain, pot, hot_floor):
            return {"decision": "reject"}
        return {"decision": "accept"}
    if procedure == "late":
        if (
            not view.horizon_known
            and delta is not None
            and delta >= 1.0 - 1e-9
            and d_opp is not None
            and d_opp < delta - 1e-9
            and share < 0.60
        ):
            return {"decision": "reject"}
        if below_share_floor(my_gain, pot, 0.42):
            return {"decision": "reject"}
        return {"decision": "accept"}
    if procedure == "stall_half":
        if (
            delta is not None
            and delta >= PATIENT
            and d_opp is None
            and not view.complete_information
            and not view.horizon_known
            and share < 0.50
        ):
            return {"decision": "reject"}
        if repeats >= 2 and share >= 0.45:
            return {"decision": "accept"}
        if below_share_floor(my_gain, pot, 0.50):
            return {"decision": "reject"}
        return {"decision": "accept"}
    if procedure == "hold_first_even":
        if (
            view.you == "player_1"
            and repeats >= 2
            and share >= 0.50
            and view.complete_information
            and view.horizon_known
            and delta is not None
            and delta >= 1.0 - 1e-9
            and d_opp is not None
            and d_opp < 1.0 - 1e-9
        ):
            # 21329ccc: two observed 50% offers are a stronger boundary signal
            # than the speculative 52% patient edge; waiting fell to 25%.
            set_node("bargaining.response.scenario.patient_repeated_even_stop")
            return {"decision": "accept"}
        if (
            view.you == "player_2"
            and delta is not None
            and delta >= 1.0 - 1e-9
            and not view.horizon_known
            and opponent.plateau_repeats >= 8
            and opponent.own_rejections >= 8
            and abs(opponent.concession_rate) <= 0.001
            and share >= 0.45
        ):
            # 079e8a58: visible delta-.9 did not predict behavior; eight
            # repeated 47.368% offers and eight rejected 60% claims did.
            set_node("bargaining.response.scenario.patient_bob_fixed_boundary")
            return {"decision": "accept"}
        patient_unknown_plateau = bool(
            delta is not None
            and delta >= 0.99
            and d_opp is None
            and not view.horizon_known
            and opponent.plateau_repeats >= 4
            and opponent.own_rejections >= 4
        )
        if patient_unknown_plateau and share >= 0.50:
            set_node("bargaining.response.scenario.patient_stable_near_even")
            return {"decision": "accept"}
        if patient_unknown_plateau and share >= 0.49:
            set_node("bargaining.response.scenario.patient_stable_near_even_hold")
            return {"decision": "reject"}
        # A fully patient responder can claim 60% from a known 0.8 opponent
        # without discounting its own payoff. Across 49 matching rejections at
        # 50% or below, every game closed at 60% on round 2. Accepting 52-55%
        # instead produced uniformly negative ratings (e090294d / 7e91e7b2).
        if (
            view.round == 1
            and delta is not None
            and delta >= PATIENT
            and d_opp is not None
            and d_opp <= HOT
            and view.complete_information
            and share < 0.60
        ):
            return {"decision": "reject"}
        # 35e9ccd9: first 50/50 as Bob scored −6.18. Field median is above half.
        if repeats >= 2 and share >= 0.48:
            patient_edge = (
                delta is not None
                and delta >= PATIENT
                and d_opp is not None
                and d_opp < PATIENT
            )
            if patient_edge and share < 0.52:
                return {"decision": "reject"}
            if share >= 0.50 or (
                delta is not None
                and delta >= PATIENT
                and d_opp is None
                and view.horizon_known
            ):
                return {"decision": "accept"}
        if 0.49 <= share <= 0.51 and repeats < 2:
            return {"decision": "reject"}
        if below_share_floor(my_gain, pot, 0.50):
            return {"decision": "reject"}
        return {"decision": "accept"}
    # mild
    if repeats >= 2 and share >= 0.45:
        return {"decision": "accept"}
    if below_share_floor(my_gain, pot, 0.50):
        return {"decision": "reject"}
    return {"decision": "accept" if my_gain > 0 else "reject"}
