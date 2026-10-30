"""Rubinstein SPE ask + RUFL-lite reservation. Routes pick fade / last-round only."""

from __future__ import annotations

import math
from typing import Any

from glee_agent.fitted import policy_segmented
from glee_agent.fitted import predict_pair_segmented
from glee_agent.policy import current
from glee_agent.rules import (
    GameView,
    as_player_id,
    money,
    my_delta,
    offer_gain,
    opp_delta,
    rubinstein_proposer_share,
    split_pot,
)
from glee_agent.workflows._common import progress, round_money, with_message

DEFAULT_MSG = "A split that still leaves a deal worth taking now."
DEFAULT_OPP_DELTA = 0.97
ASK_MIN = 0.40
ASK_MAX = 0.60
MILD_ASK_CAP = 0.55
MILD_FLOOR = 0.50
HOT_DELTA = 0.85
POWERLESS_FLOOR = 0.08
CONTINUATION_PRIOR = 0.85
REJECTION_DECAY = 0.50
MIN_CONTINUATION_REJECTIONS = 4
TRAJECTORY_GUARD = 0.005
STABLE_BOUNDARY_BAND = 0.015


def opp_share(view: GameView) -> float | None:
    pot = money(view)
    if pot <= 0:
        return None
    gain = offer_gain(view.state.get("last_offer"), view.you)
    if gain is None:
        return None
    return gain / pot


def same_split_repeats(view: GameView) -> int:
    from glee_agent.workflows.bargaining._opponent import their_shares

    share = opp_share(view)
    if share is None:
        return 0
    repeats = 0
    for seen in reversed(their_shares(view)):
        if abs(seen - share) >= 0.01:
            break
        repeats += 1
    return repeats


def resolved_deltas(view: GameView) -> tuple[float | None, float | None]:
    d_me = my_delta(view)
    d_opp = opp_delta(view)
    if d_me is not None and d_opp is None:
        d_opp = DEFAULT_OPP_DELTA
    return d_me, d_opp


def is_hot(view: GameView) -> bool:
    d_me = my_delta(view)
    return d_me is not None and d_me <= HOT_DELTA


def spe_or_policy_share(view: GameView) -> float:
    known_opp = opp_delta(view) is not None
    d_me, d_opp = resolved_deltas(view)
    if d_me is not None and d_opp is not None:
        share = rubinstein_proposer_share(d_me, d_opp)
        # Defaulted opp + our δ=1 was SPE 70% (05d707fd −7). Policy knobs stay.
        if not known_opp:
            share = min(share, MILD_ASK_CAP)
            # Default opp δ=0.97 vs our 0.95 is SPE 0.40. Alice r1 then
            # opened 4000/6000 and Bob accepted (−5.3). Field holds ~50%.
            if not is_hot(view):
                share = max(share, MILD_FLOOR)
        return share
    return float(current()["bargaining"]["offer_share"])


def apply_opp_anchor(view: GameView, share: float) -> float:
    best = opp_share(view)
    if best is not None:
        share = max(share, min(best + 0.03, 0.62))
        if best >= 0.48:
            share = min(share, best + 0.08)
    return share


def finish_ask(share: float, floor: float) -> float:
    return min(max(share, floor), ASK_MAX)


def we_posted_offer(view: GameView) -> bool:
    """True if history already has an offer we proposed. Ignores last_offer."""
    want = as_player_id(view.you)
    for row in view.state.get("history") or []:
        if not isinstance(row, dict):
            continue
        proposer = str(row.get("proposer") or "")
        offer = row.get("offer")
        if isinstance(offer, dict):
            proposer = str(offer.get("proposer") or proposer)
        if proposer and as_player_id(proposer) == want:
            return True
    return False


def rejected_own_offers(view: GameView) -> int:
    want = as_player_id(view.you)
    count = 0
    for row in view.state.get("history") or []:
        if not isinstance(row, dict):
            continue
        offer = row.get("offer")
        if not isinstance(offer, dict):
            continue
        proposer = str(row.get("proposer") or offer.get("proposer") or "")
        decision = str(row.get("decision") or "").lower()
        if proposer and as_player_id(proposer) == want and "reject" in decision:
            count += 1
    return count


def continuation_probability(view: GameView) -> float:
    """Behavior-dependent chance that our next even split is accepted."""
    return CONTINUATION_PRIOR * (REJECTION_DECAY ** rejected_own_offers(view))


def expected_even_continuation(view: GameView) -> float | None:
    """Normalized payoff of a next-round 50/50 offer, weighted by acceptance."""
    delta = my_delta(view)
    if delta is None or view.is_last_round:
        return None
    return 0.50 * (delta ** view.round) * continuation_probability(view)


def response_value_model(view: GameView) -> dict[str, Any] | None:
    """Compare accepting now with an optimistic visible-state continuation."""
    delta = my_delta(view)
    share = opp_share(view)
    if delta is None or share is None or delta >= 0.99 or view.is_last_round:
        return None

    from glee_agent.workflows.bargaining._opponent import state_of, their_shares

    opponent = state_of(view)
    own_offer = expected_even_continuation(view) or 0.0
    seen = their_shares(view)
    recent_rates = [right - left for left, right in zip(seen, seen[1:])][-3:]
    optimistic_rate = max([0.0, *recent_rates])
    recent_boundary = seen[-4:]
    boundary_range = (
        max(recent_boundary) - min(recent_boundary)
        if len(recent_boundary) == 4
        else None
    )
    stable_boundary = (
        boundary_range is not None and boundary_range < STABLE_BOUNDARY_BAND
    )
    last_round = view.max_rounds if view.horizon_known and view.max_rounds else 99
    future_counters = max(0, (last_round - view.round) // 2)
    counter_offer = max(
        [0.0]
        + [
            min(1.0, share + optimistic_rate * step)
            * (delta ** (view.round - 1 + 2 * step))
            for step in range(1, future_counters + 1)
        ]
    )
    accept_value = share * (delta ** max(0, view.round - 1))
    continuation_value = max(own_offer, counter_offer)
    eligible = bool(
        not view.horizon_known
        and delta <= HOT_DELTA + 1e-9
        and opponent.own_rejections >= 2
        and stable_boundary
    )
    accept = eligible and accept_value + 1e-9 >= continuation_value + TRAJECTORY_GUARD
    return {
        "decision": "accept" if accept else "fallback",
        "reason": (
            "stable_boundary_dominates_continuation"
            if accept
            else "insufficient_evidence_or_value"
        ),
        "accept_value": accept_value,
        "continuation_value": continuation_value,
        "own_offer_continuation": own_offer,
        "counter_offer_upper_bound": counter_offer,
        "optimistic_concession_rate": optimistic_rate,
        "guard": TRAJECTORY_GUARD,
        "eligible": eligible,
        "stable_boundary": stable_boundary,
        "boundary_range": boundary_range,
        "opponent_rejections": opponent.own_rejections,
        "plateau_repeats": opponent.plateau_repeats,
    }


def fitted_progress(view: GameView) -> float:
    """Continuous time feature; an open horizon saturates without a round cutoff."""
    if view.horizon_known and view.max_rounds and view.max_rounds > 1:
        return min(1.0, max(0.0, (view.round - 1) / (view.max_rounds - 1)))
    return 1.0 - math.exp(-max(0, view.round - 1) / 16.0)


def fitted_offer_features(view: GameView, share: float) -> list[float] | None:
    delta = my_delta(view)
    if delta is None:
        return None
    from glee_agent.workflows.bargaining._opponent import state_of

    opponent = state_of(view)
    return [
        float(share),
        fitted_progress(view),
        float(delta),
        float(opp_delta(view) if opp_delta(view) is not None else 0.95),
        min(opponent.own_rejections, 8) / 8.0,
        float(opponent.current_share if opponent.current_share is not None else 0.50),
        float(view.you == "player_1"),
    ]


def fitted_offer_curve(
    view: GameView,
    fallback: float,
    procedure: str = "",
) -> dict[str, Any] | None:
    """Maximize fitted accept/continue EV over a supported continuous share curve."""
    delta = my_delta(view)
    if delta is None:
        return None
    from glee_agent.segments import segment

    seg = segment(view, procedure)
    candidates = {round(0.35 + 0.01 * index, 2) for index in range(26)}
    candidates.add(round(float(fallback), 6))
    rows: list[tuple[float, float, float, float]] = []
    for share in sorted(candidates):
        features = fitted_offer_features(view, share)
        if features is None:
            continue
        pair = predict_pair_segmented(
            "bargaining_offer_accept",
            "bargaining_offer_continue",
            seg.parents,
            features,
        )
        if pair is None:
            return None
        accepted, continued, level, accept_model, continue_model = pair
        if view.is_last_round:
            continued = 0.0
        payoff = share * (delta ** max(0, view.round - 1))
        expected = accepted * payoff + (1.0 - accepted) * continued
        rows.append((expected, share, accepted, continued))
    if not rows:
        return None
    best = max(rows, key=lambda row: (row[0], row[1]))
    current = min(rows, key=lambda row: abs(row[1] - fallback))
    edge = best[0] - current[0]
    threshold = max(
        float(accept_model.get("guard", 0.015)),
        float(continue_model.get("guard", 0.015)),
    )
    chosen = best if edge > threshold else current
    return {
        "share": chosen[1],
        "expected_value": chosen[0],
        "accept_probability": chosen[2],
        "continue_value": chosen[3],
        "fallback_share": current[1],
        "fallback_expected_value": current[0],
        "model_edge": edge,
        "guard": threshold,
        "scenario_id": seg.scenario_id,
        "accept_model_level": level,
        "continue_model_level": level,
        "accept_model_n": int(accept_model.get("n") or 0),
        "continue_model_n": int(continue_model.get("n") or 0),
    }


def fitted_response_features(view: GameView) -> list[float] | None:
    delta = my_delta(view)
    share = opp_share(view)
    if delta is None or share is None:
        return None
    from glee_agent.workflows.bargaining._opponent import state_of, their_shares

    state = state_of(view)
    accept_now = share * (delta ** max(0, view.round - 1))
    old = expected_even_continuation(view)
    if old is None:
        return None
    ratio = math.log(max(accept_now, 1e-6) / max(old, 1e-6))
    seen = their_shares(view)
    previous = seen[-2] if len(seen) >= 2 else 0.50
    rate = share - previous if len(seen) >= 2 else 0.0
    return [
        accept_now,
        old,
        ratio,
        fitted_progress(view),
        float(delta),
        float(opp_delta(view) if opp_delta(view) is not None else 0.95),
        min(state.own_rejections, 8) / 8.0,
        share,
        previous,
        rate,
        min(state.plateau_repeats, 8) / 8.0,
        float(view.you == "player_1"),
    ]


def fitted_response_curve(
    view: GameView,
    procedure: str = "",
) -> dict[str, Any] | None:
    features = fitted_response_features(view)
    if features is None:
        return None
    from glee_agent.segments import segment

    seg = segment(view, procedure)
    def deployable(policy: dict[str, Any], level_n: int) -> bool:
        return bool(
            level_n <= 4
            and float(policy.get("effective_games") or 0.0) >= 8.0
            and int(policy.get("train_triggered") or 0) >= 3
            and int(policy.get("holdout_improved") or 0) >= 1
            and int(policy.get("holdout_improved") or 0)
            >= int(policy.get("holdout_regressed") or 0)
            and float(policy.get("train_gain") or 0.0) > 0.0
            and float(policy.get("holdout_gain") or 0.0) > 0.0
            and int(policy.get("external_triggered") or 0) >= 3
            and int(policy.get("external_improved") or 0) >= 3
            and int(policy.get("external_regressed") or 0) == 0
            and float(policy.get("external_gain") or 0.0) > 0.0
        )

    selected = policy_segmented(
        "bargaining_response_accept",
        seg.parents,
        predicate=deployable,
    )
    if selected is None:
        return None
    policy, level = selected
    accepted, continued = features[0], features[1]
    margin = accepted - continued
    threshold = float(policy.get("threshold") or 0.0)
    decision = "accept" if margin + 1e-9 >= threshold else "fallback"
    return {
        "decision": decision,
        "accept_value": accepted,
        "continue_value": continued,
        "margin": margin,
        "guard": threshold,
        "scenario_id": seg.scenario_id,
        "model_level": level,
        "sample_n": int(policy.get("n") or 0),
        "effective_games": float(policy.get("effective_games") or 0.0),
        "train_gain": float(policy.get("train_gain") or 0.0),
        "holdout_gain": float(policy.get("holdout_gain") or 0.0),
        "external_triggered": int(policy.get("external_triggered") or 0),
        "external_improved": int(policy.get("external_improved") or 0),
        "external_regressed": int(policy.get("external_regressed") or 0),
        "external_gain": float(policy.get("external_gain") or 0.0),
    }


def make_offer(view: GameView, share: float, message: str) -> dict[str, Any]:
    pot = money(view)
    mine = round_money(pot * share)
    alice, bob = split_pot(pot, mine, view.you)
    return with_message(view, {"alice_gain": alice, "bob_gain": bob}, message)


def last_gain(view: GameView) -> float | None:
    return offer_gain(view.state.get("last_offer"), view.you)


def ask_share(view: GameView, *, fade: bool = False, floor: float = ASK_MIN) -> float:
    share = spe_or_policy_share(view)
    if fade:
        t = progress(view)
        share = 0.50 + (share - 0.50) * max(0.0, 1.0 - t)
        if view.round >= 8:
            share = min(share, 0.52)
        if is_hot(view):
            share = min(share, 0.46)
        else:
            share = min(share, MILD_ASK_CAP)
    if not is_hot(view):
        floor = max(floor, MILD_FLOOR)
    return finish_ask(apply_opp_anchor(view, share), floor)
