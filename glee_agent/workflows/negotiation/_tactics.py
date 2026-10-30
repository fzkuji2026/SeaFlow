"""Faratin 1998 + SAC-lite bidding, Nash ZOPA, NegMAS-style accept."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import median
from typing import Any

from glee_agent.compose import Trace, pipeline
from glee_agent.fitted import model as fitted_model
from glee_agent.fitted import predict as fitted_predict
from glee_agent.fitted import predict_pair_segmented
from glee_agent.policy import current
from glee_agent.rules import (
    GameView,
    as_player_id,
    last_price,
    opponent_id,
    role_of,
    value_of,
    zopa,
)
from glee_agent.workflows._common import progress, round_money, with_message

STRONG_SELLER_FLOOR = 1.20


@dataclass(frozen=True)
class OpponentState:
    style: str
    samples: int
    current_price: float | None
    best_price: float | None
    concession_rate: float
    predicted_next_price: float | None
    plateau_repeats: int
    own_rejections: int
    rejected_price_boundary: float | None
    revealed_reservation_bound: float | None
    current_profitable: bool | None
    current_surplus_share: float | None
    estimated_patience: float
    stall_probability: float
    platform_stop_risk: float
    zopa_probability: float | None
    zopa_model: str | None
    zopa_model_auc: float | None


def _priced_offers(state: dict):
    rows = [(row, False) for row in list(state.get("history") or [])]
    last = state.get("last_offer")
    if isinstance(last, dict):
        rows.append((last, True))
    seen: set[tuple] = set()
    previous: tuple[str, float] | None = None
    for row, is_last in rows:
        if not isinstance(row, dict):
            continue
        offer = row.get("offer") or row.get("last_offer") or row
        if not isinstance(offer, dict):
            continue
        for key in ("price", "product_price"):
            if key in offer:
                try:
                    price = float(offer[key])
                except (TypeError, ValueError):
                    break
                sender = str(
                    offer.get("from_player")
                    or row.get("from_player")
                    or offer.get("proposer")
                    or row.get("proposer")
                    or ""
                )
                if not sender and row.get("decided_by"):
                    sender = opponent_id(str(row["decided_by"]))
                round_n = offer.get("round") or row.get("round")
                identity = (round_n, sender, price) if round_n is not None else None
                if identity is not None and identity in seen:
                    break
                if is_last and identity is None and previous == (sender, price):
                    break
                if identity is not None:
                    seen.add(identity)
                previous = (sender, price)
                yield sender, price
                break


def prices(state: dict) -> list[float]:
    return [price for _sender, price in _priced_offers(state)]


def opponent_prices(view: GameView) -> list[float]:
    mine = as_player_id(view.you)
    return [
        price
        for sender, price in _priced_offers(view.state)
        if not sender or as_player_id(sender) != mine
    ]


def own_prices(view: GameView) -> list[float]:
    mine = as_player_id(view.you)
    return [
        price
        for sender, price in _priced_offers(view.state)
        if sender and as_player_id(sender) == mine
    ]


def time_t(view: GameView) -> float:
    if view.is_last_round:
        return 1.0
    return min(1.0, max(0.0, progress(view)))


def faratin_e(view: GameView) -> float:
    """Faratin β. β<1 Boulware (hold); β>1 Conceder.

    Early hold: the live field opens at ~0.90 of ZOPA and hardens. Conceder
    β=1.2 was the official-starter close-any-deal schedule and scored below
    the even-split attractor. Last rounds still concede so a thin ZOPA can close.
    """
    t = time_t(view)
    rounds_left = None
    if view.horizon_known and view.max_rounds is not None:
        rounds_left = view.max_rounds - view.round
    if rounds_left is not None and rounds_left <= 1:
        return 2.0
    if not view.horizon_known and t >= 0.50:
        return 2.0
    if t >= 0.50:
        return 1.2
    return 0.6


def faratin_alpha(view: GameView) -> float:
    """Faratin 1998: α(t) = t^(1/β). β<1 Boulware (hold), β>1 Conceder."""
    t = time_t(view)
    e = faratin_e(view)
    if t <= 0:
        alpha = 0.0
    elif t >= 1:
        alpha = 1.0
    else:
        alpha = t ** (1.0 / e)
    p = current()["negotiation"]
    if progress(view) >= float(p.get("walk_progress", 0.45)):
        alpha = max(alpha, 0.75)
    return min(1.0, max(0.0, alpha))


def aspiration(view: GameView, role: str, my_value: float) -> float:
    p = current()["negotiation"]
    if role == "seller":
        open_ask = max(my_value, my_value * float(p["seller_open"]))
    else:
        open_ask = min(my_value, my_value * float(p["buyer_open"]))
    opp = last_price(view)
    # An unprofitable bid is not evidence for conceding to reservation. Use
    # the official 1.30×/0.80× counter, then let Faratin handle later rounds.
    if opp is not None and not in_the_money(role, my_value, opp):
        if role == "seller":
            open_ask = min(open_ask, my_value * 1.30)
        else:
            open_ask = max(open_ask, my_value * 0.80)
    return open_ask


def faratin(view: GameView, role: str, my_value: float, ask: float) -> float:
    res = my_value
    alpha = faratin_alpha(view)
    price = res + (1.0 - alpha) * (ask - res)
    if role == "seller":
        return round_money(max(res, price))
    return round_money(min(res, price))


def belief_cap(view: GameView, role: str, my_value: float, ask: float) -> float:
    seen = opponent_prices(view)
    if not seen:
        return ask
    if role == "seller":
        their_best = max(seen)
        if their_best >= ask:
            return round_money(max(my_value, min(ask, their_best)))
        return ask
    their_best = min(seen)
    if their_best <= ask:
        return round_money(min(my_value, max(ask, their_best)))
    return ask


def nash_price(role: str, lo: float, hi: float) -> float:
    """Exact 50/50 split of a known ZOPA."""
    surplus = max(hi - lo, 0.0)
    if surplus <= 0:
        return round_money(lo if role == "seller" else hi)
    if role == "seller":
        return round_money(lo + 0.50 * surplus)
    return round_money(hi - 0.50 * surplus)


def slice_of_surplus(role: str, price: float, lo: float, hi: float) -> float:
    surplus = max(hi - lo, 0.0)
    if surplus <= 0:
        return 0.0
    if role == "seller":
        return (price - lo) / surplus
    return (hi - price) / surplus


def best_offered_share(view: GameView, role: str) -> float | None:
    zone = zopa(view)
    if zone is None:
        return None
    lo, hi = zone
    best: float | None = None
    for price in opponent_prices(view):
        share = slice_of_surplus(role, price, lo, hi)
        if best is None or share > best:
            best = share
    return best


def split_zone(view: GameView, role: str, lo: float, hi: float) -> float:
    """Complete-info ask: field-level open, Boulware fade, deadline claim.

    One-shot and last-round *offer* are the same leaf (ultimatum as proposer):
    claim most surplus. Last-round *respond* is not this function. Mid-game still
    opens at the field prior and fades to 70%. Accept is a separate node.
    """
    surplus = max(hi - lo, 0.0)
    if surplus <= 0:
        return round_money(lo if role == "seller" else hi)
    one_shot = bool(view.horizon_known and view.max_rounds == 1)
    # All 73 observed complete-info one-shot 90% claims were accepted. Leave a
    # positive 1% surplus to the responder instead of giving up another 9%.
    if one_shot:
        share = 0.99
    elif view.is_last_round:
        share = 0.80
    else:
        last = last_price(view)
        lowball = (
            role == "seller"
            and last is not None
            and slice_of_surplus(role, last, lo, hi) <= 0.20 + 1e-9
            and same_price_repeats(view, last) >= 2
        )
        # In 21 complete-info T=10 seller games, holding 80%+ against a
        # repeated 0–20% counter averaged -5.65. Offer the even split once the
        # opponent has demonstrated that fixed boundary twice. The fifth
        # frozen batch then showed the even split rejected to the deadline in
        # both matching seller games. Use the same-scenario public own-payoff
        # median instead of another fixed surplus share.
        if lowball:
            target = field_payoff_price(view, role, lo)
            return round_money(min(hi, max(lo, target if target is not None else lo + 0.50 * surplus)))
        from glee_agent.workflows.negotiation._field import field_open_claim

        open_s = field_open_claim()
        floor_s = 0.70
        t = time_t(view)
        mine = value_of(view) or (lo if role == "seller" else hi)
        opponent = state_of(view, role, mine)
        stalled = bool(
            not view.horizon_known
            and opponent.style == "stubborn"
            and opponent.plateau_repeats >= 4
        )
        if stalled:
            # Six cached settlements below 30% all rated -5.9 or worse;
            # comparable equal-split settlements rated +3.3 to +3.5.
            share = 0.50
        else:
            expo = 3.0 if opponent.style == "stubborn" else 2.0
            share = open_s + (floor_s - open_s) * (t ** expo)
    if role == "seller":
        return round_money(lo + share * surplus)
    return round_money(hi - share * surplus)


def tit_for_tat(role: str, my_value: float, ask: float, seen: list[float]) -> float:
    if len(seen) < 2:
        return ask
    prev, now = seen[-2], seen[-1]
    if role == "seller":
        their_concede = now - prev
        if their_concede <= 0:
            return ask
        return round_money(max(my_value, ask - 0.8 * their_concede))
    their_concede = prev - now
    if their_concede <= 0:
        return ask
    return round_money(min(my_value, ask + 0.8 * their_concede))


def toward_offer(role: str, my_value: float, ask: float, opp: float | None) -> float:
    if opp is None:
        return ask
    mid = (my_value + ask) / 2.0
    if role == "seller" and my_value <= opp < ask and opp + 1e-9 >= mid:
        return round_money(max(my_value, (ask + opp) / 2.0))
    if role == "buyer" and ask < opp <= my_value and opp - 1e-9 <= mid:
        return round_money(min(my_value, (ask + opp) / 2.0))
    return ask


def pipeline_ask(view: GameView, role: str, my_value: float) -> Trace:
    seed = aspiration(view, role, my_value)
    seen = opponent_prices(view)
    opp = last_price(view)
    return pipeline(
        seed,
        [
            ("faratin", lambda x: faratin(view, role, my_value, x)),
            ("tit_for_tat", lambda x: tit_for_tat(role, my_value, x, seen)),
            ("meet", lambda x: toward_offer(role, my_value, x, opp)),
            ("belief_cap", lambda x: belief_cap(view, role, my_value, x)),
        ],
    )


def _style(seen: list[float], role: str, my_value: float) -> str:
    if not seen:
        return "unknown"
    recent = seen[-4:]
    span = max(0.5, 0.02 * max(abs(recent[0]), 1.0))
    stall = max(0.5, 0.015 * max(abs(recent[0]), 1.0))
    if len(recent) >= 2:
        if len(recent) >= 3 and max(recent[-3:]) - min(recent[-3:]) < stall:
            return "stubborn"
        if role == "seller" and recent[-1] > recent[0] + span:
            return "conceder"
        if role == "buyer" and recent[-1] < recent[0] - span:
            return "conceder"
        if abs(recent[-1] - recent[0]) < stall:
            return "stubborn"
    if role == "seller" and max(seen) + 1e-9 < my_value:
        return "lowballer"
    if role == "buyer" and min(seen) - 1e-9 > my_value:
        return "lowballer"
    return "mixed"


def _own_rejected_prices(view: GameView) -> list[float]:
    mine = as_player_id(view.you)
    out: list[float] = []
    for row in view.state.get("history") or []:
        if not isinstance(row, dict) or "reject" not in str(row.get("decision") or "").lower():
            continue
        offer = row.get("offer") or row.get("last_offer") or row
        if not isinstance(offer, dict):
            continue
        sender = str(offer.get("from_player") or row.get("from_player") or "")
        if sender:
            if as_player_id(sender) != mine:
                continue
        else:
            decider = str(row.get("decided_by") or "")
            if not decider or as_player_id(decider) == mine:
                continue
        for key in ("price", "product_price"):
            try:
                out.append(float(offer[key]))
            except (KeyError, TypeError, ValueError):
                continue
            break
    return out


def state_of(view: GameView, role: str, my_value: float) -> OpponentState:
    """Estimate the opponent from visible prices and rejected offers."""
    seen = opponent_prices(view)
    current = seen[-1] if seen else None
    scale = max(abs(my_value), abs(current or 0.0), 1.0)
    direction = 1.0 if role == "seller" else -1.0
    diffs = [direction * (right - left) / scale for left, right in zip(seen, seen[1:])]
    rate = median(diffs[-3:]) if diffs else 0.0
    predicted = None if current is None else current + direction * rate * scale
    repeats = same_price_repeats(view, current) if current is not None else 0
    rejected = _own_rejected_prices(view)
    boundary = None
    if rejected:
        boundary = min(rejected) if role == "seller" else max(rejected)
    best = None
    if seen:
        best = max(seen) if role == "seller" else min(seen)
    zone = zopa(view)
    share = None
    if zone is not None and current is not None:
        share = slice_of_surplus(role, current, zone[0], zone[1])
    profitable = None
    if current is not None:
        profitable = profit(role, my_value, current) >= 0
    evidence = max(0, repeats - 1) + min(len(rejected), 4)
    stop_risk = 0.0
    if not view.horizon_known:
        stop_risk = min(1.0, max(0.0, (view.round - 40) / 33.0))
    zopa_probability, zopa_model, zopa_auc = fitted_zopa_probability(
        view, role, my_value
    )
    return OpponentState(
        style=_style(seen, role, my_value),
        samples=len(seen),
        current_price=current,
        best_price=best,
        concession_rate=rate,
        predicted_next_price=predicted,
        plateau_repeats=repeats,
        own_rejections=len(rejected),
        rejected_price_boundary=boundary,
        revealed_reservation_bound=best,
        current_profitable=profitable,
        current_surplus_share=share,
        estimated_patience=min(1.0, evidence / 6.0),
        stall_probability=1.0 - 0.5 ** min(evidence, 8),
        platform_stop_risk=stop_risk,
        zopa_probability=zopa_probability,
        zopa_model=zopa_model,
        zopa_model_auc=zopa_auc,
    )


def style_of(view: GameView, role: str, my_value: float) -> str:
    return state_of(view, role, my_value).style


def fitted_zopa_probability(
    view: GameView,
    role: str,
    my_value: float,
) -> tuple[float | None, str | None, float | None]:
    """Estimate only whether hidden values overlap; never infer an exact value."""
    if view.complete_information:
        return (1.0 if zopa(view) is not None else 0.0), "visible", 1.0
    from glee_agent.workflows.negotiation.classify import own_mult

    mult = own_mult(view)
    if not mult or my_value <= 0:
        return None, None, None
    base = my_value / mult
    seen = [price / base for price in opponent_prices(view)]
    if not seen:
        return None, None, None
    first = seen[0]
    second = seen[1] if len(seen) >= 2 else first
    horizon = "open" if not view.horizon_known else str(view.max_rounds or "unknown")
    name = f"negotiation_zopa_{role}_{horizon}"
    features = [
        float(mult),
        float(view.messages_allowed),
        first,
        second,
        second - first,
        min(len(seen), 6) / 6.0,
    ]
    probability = fitted_predict(name, features)
    row = fitted_model(name)
    if probability is None or row is None:
        return None, None, None
    return probability, name, float(row.get("holdout_auc") or 0.0)


def pick_procedure(view: GameView, regime: str, role: str, my_value: float) -> str:
    """One tree node: empty zone / complete clock / hidden type / hidden clock."""
    if regime == "no_zopa":
        return "walk"
    if regime in ("ultimatum", "complete", "complete_seller", "complete_late"):
        return _complete_procedure(view, regime)
    return _hidden_procedure(view, regime, role, my_value)


def _complete_procedure(view: GameView, regime: str) -> str:
    if regime == "ultimatum":
        return "nash_half"
    return "zopa_boulware"


def _hidden_procedure(view: GameView, regime: str, role: str, my_value: float) -> str:
    from glee_agent.workflows.negotiation.classify import own_band

    band = own_band(view)
    if band.startswith("dead") and regime not in ("one_shot", "incomplete_deadline"):
        return _hidden_dead(view, role, my_value)
    if band == "strong_buyer" and regime not in ("one_shot", "incomplete_deadline"):
        return _hidden_strong_buyer()
    if band == "strong_seller" and regime not in ("one_shot", "incomplete_deadline"):
        return _hidden_strong_seller(view, regime, role, my_value)
    return _hidden_clock(view, regime, role, my_value)


def _hidden_dead(view: GameView, role: str, my_value: float) -> str:
    """Seller 1.5× / buyer 0.8×: no ZOPA on the official grid."""
    from glee_agent.workflows.negotiation.classify import horizon_phase

    phase = horizon_phase(view)
    opp = last_price(view)
    repeats = same_price_repeats(view, opp) if opp is not None else 0
    if opp is not None and not in_the_money(role, my_value, opp):
        if phase in ("mid", "late") or repeats >= 1:
            return "walk"
    return "close_110"


def _hidden_strong_buyer() -> str:
    """Buyer 1.5×: bid from 0.75×, while acceptance separately holds 0.90×."""
    return "faratin_boulware"


def _hidden_strong_seller(view: GameView, regime: str, role: str, my_value: float) -> str:
    """Seller 0.8×: Faratin from 1.50×. Do not open 1.10×."""
    if regime == "open_horizon":
        seen = len(opponent_prices(view))
        if 1 <= seen <= 2:
            return "probe_103"
        if seen >= 3:
            return "probe_096"
    return _hidden_clock(view, regime, role, my_value)


def _hidden_clock(view: GameView, regime: str, role: str, my_value: float) -> str:
    from glee_agent.workflows.negotiation.classify import (
        horizon_phase,
        own_band,
        own_mult,
        zopa_band,
    )

    style = style_of(view, role, my_value)
    phase = horizon_phase(view)
    zb = zopa_band(view)
    opp = last_price(view)
    repeats = same_price_repeats(view, opp) if opp is not None else 0
    if regime == "one_shot":
        if zb != "hidden":
            return "nash_half"
        if role == "seller" and own_mult(view) == 1.2:
            # Only the hidden 1.5x buyer has positive surplus. Leave it a
            # strictly positive gain while claiming the rest in the sole bid.
            return "close_125"
        if role == "seller" and own_band(view) == "strong_seller":
            # In 47 hidden one-shot games at cost 0.8× base, the four-type
            # equal-weight rating was +.586 near 1.2× base versus -.720 at
            # 1.0×. Stay just below the 1.2× buyer type so accepting leaves
            # that responder a strictly positive gain.
            return "close_150"
        # Seller 1.0x: claim just below the 1.2x hidden buyer value. Exact
        # 1.20x was rejected in four of six matching games because it leaves
        # the buyer zero gain; a strict positive gain preserves that type.
        if role == "seller" and own_mult(view) == 1.0:
            return "close_120"
        return "close_110"
    if regime == "incomplete_late":
        if style == "conceder":
            return "close_110" if role == "seller" else "take_profit"
        if style == "stubborn" or repeats >= 2:
            # 82bb7150: at r8/10 a stubborn 1.004x-cost buyer induced a
            # reservation close worth only 2% of the hidden surplus (-5.9),
            # while matching 1.09-1.10x closes score +3.9 to +5.3. Keep the
            # close target while two response opportunities remain.
            if view.max_rounds is not None and view.round < view.max_rounds - 1:
                return "close_110"
            return "reservation"
        return "close_110"
    if regime == "incomplete_deadline":
        return "faratin_reservation"
    if regime == "open_horizon":
        if style == "conceder" and phase in ("mid", "late"):
            return "take_profit"
        if phase == "late":
            return "faratin_conceder"
        return "faratin_boulware"
    if regime in ("incomplete", "incomplete_seller"):
        if style == "conceder" and phase == "late":
            return "faratin_conceder"
        return "faratin_boulware"
    return "faratin_boulware"


def take_profit_ask(view: GameView, role: str, my_value: float) -> float:
    opp = last_price(view)
    if role == "seller":
        close = my_value * 1.05
        if opp is not None and profit(role, my_value, opp) > 0:
            return round_money(max(my_value, min(close, (opp + close) / 2.0)))
        return round_money(close)
    close = my_value * 0.95
    if opp is not None and profit(role, my_value, opp) > 0:
        return round_money(min(my_value, max(close, (opp + close) / 2.0)))
    return round_money(close)


def pipeline_ask_e(view: GameView, role: str, my_value: float, e: float) -> Trace:
    seed = aspiration(view, role, my_value)
    seen = opponent_prices(view)
    opp = last_price(view)

    def _faratin(x: float) -> float:
        t = time_t(view)
        if t <= 0:
            alpha = 0.0
        elif t >= 1:
            alpha = 1.0
        else:
            alpha = t ** (1.0 / e)
        price = my_value + (1.0 - alpha) * (x - my_value)
        if role == "seller":
            return round_money(max(my_value, price))
        return round_money(min(my_value, price))

    return pipeline(
        seed,
        [
            ("faratin", _faratin),
            ("tit_for_tat", lambda x: tit_for_tat(role, my_value, x, seen)),
            ("meet", lambda x: toward_offer(role, my_value, x, opp)),
            ("belief_cap", lambda x: belief_cap(view, role, my_value, x)),
        ],
    )


def _clamp_strong(view: GameView, role: str, my_value: float, price: float) -> float:
    from glee_agent.workflows.negotiation.classify import own_band

    # Known-ZOPA asks are already expressed as a surplus share. Reservation
    # multipliers are only a fallback for hidden opponent values.
    if zopa(view) is not None:
        return round_money(price)
    band = own_band(view)
    if band == "strong_seller" and role == "seller":
        # A bid above 1.25x cost rules out the 0.8x and 1.0x buyer types.
        # Keep targeting just below the remaining 1.2x buyer value on every
        # later turn; 44a2d5c2 / 0047011b / 1c2777dd otherwise closed at only
        # 31-40% of the hidden surplus (-4.1 to -4.9).
        if any(seen > my_value * 1.25 for seen in opponent_prices(view)):
            return round_money(max(price, my_value * 1.487125))
        field = field_payoff_price(view, role, my_value)
        floor = max(my_value * STRONG_SELLER_FLOOR, field or my_value)
        return round_money(max(price, floor))
    outgoing_round = view.round + (1 if view.action_type == "decision" else 0)
    live_penultimate = bool(
        view.horizon_known
        and view.max_rounds
        and outgoing_round >= view.max_rounds - 1
    )
    if (
        band == "live"
        and role == "seller"
        and not view.is_last_round
        and not live_penultimate
    ):
        opp = last_price(view)
        late_open_stall = bool(
            not view.horizon_known
            and view.round >= 20
            and opp is not None
            and opp < my_value
            and style_of(view, role, my_value) in ("stubborn", "lowballer")
        )
        floor = 1.05 if late_open_stall else 1.20
        return round_money(max(price, my_value * floor))
    if (
        band == "live"
        and role == "buyer"
        and not view.is_last_round
        and not live_penultimate
    ):
        return round_money(min(price, my_value * 0.90))
    if band == "strong_buyer" and role == "buyer":
        price = min(price, my_value * 0.90)
        # ed00091e / 2cc88d25 / 938525b3: every seller ask remained above
        # our value, so raising our own bid revealed no acceptance boundary
        # and closed at the same low-payoff price. Hold the best prior bid
        # until the seller posts a feasible offer.
        if opponent_prices(view) and all(
            seen > my_value + 1e-9 for seen in opponent_prices(view)
        ):
            prior = own_prices(view)
            if prior:
                boundary = max(prior)
                plateau = max(prior) - min(prior) <= 0.01 * max(my_value, 1.0)
                state = state_of(view, role, my_value)
                if (
                    view.horizon_known
                    and view.max_rounds == 10
                    and state.own_rejections >= 2
                    and plateau
                ):
                    # A repeated exact bid has already failed. Probe one
                    # percent of the remaining reservation interval; this can
                    # cross a strict-positive boundary while preserving 99%
                    # of the still-unclaimed buyer margin.
                    boundary += 0.01 * (my_value - boundary)
                    price = min(price, boundary)
                else:
                    price = min(price, min(prior))
        return round_money(price)
    return round_money(price)


def compose_ask(view: GameView, regime: str, role: str, my_value: float) -> Trace:
    proc = pick_procedure(view, regime, role, my_value)
    zone = zopa(view)
    if proc == "walk":
        out = Trace(steps=("walk",), value=round_money(my_value))
    elif proc == "nash_half" and zone is not None:
        out = Trace(
            steps=("ultimatum_claim",),
            value=split_zone(view, role, zone[0], zone[1]),
        )
    elif proc == "zopa_boulware" and zone is not None:
        price = split_zone(view, role, zone[0], zone[1])
        if regime in ("complete", "complete_late", "ultimatum"):
            out = Trace(steps=("zopa",), value=price)
        else:
            out = Trace(steps=(proc,), value=price)
    elif proc == "open_125":
        p = current()["negotiation"]
        if role == "seller":
            seed = my_value * float(p.get("seller_open", 1.50))
        else:
            seed = my_value * float(p.get("buyer_open", 0.75))
        out = Trace(steps=("seed",), value=round_money(seed))
    elif proc == "close_110":
        close = my_value * (1.10 if role == "seller" else 0.90)
        out = Trace(steps=("close_seed",), value=round_money(close))
    elif proc == "close_120":
        out = Trace(steps=("close_seed",), value=round_money(my_value * 1.19875))
    elif proc == "close_125":
        out = Trace(steps=("close_seed",), value=round_money(my_value * 1.24875))
    elif proc == "close_150":
        out = Trace(steps=("close_seed",), value=round_money(my_value * 1.49875))
    elif proc == "probe_103":
        out = Trace(steps=("type_probe",), value=round_money(my_value * 1.2875))
    elif proc == "probe_096":
        out = Trace(steps=("type_probe",), value=round_money(my_value * 1.20))
    elif proc == "reservation":
        value = my_value
        from glee_agent.workflows.negotiation.classify import own_band

        if role == "seller" and own_band(view) == "live" and not view.is_last_round:
            value = my_value * 1.001
        out = Trace(steps=("reservation",), value=round_money(value))
    elif proc == "take_profit":
        out = Trace(steps=("take_profit",), value=take_profit_ask(view, role, my_value))
    elif proc == "faratin_conceder":
        out = pipeline_ask_e(view, role, my_value, 1.5)
    elif proc == "faratin_reservation":
        out = pipeline_ask(view, role, my_value)
    else:
        out = pipeline_ask(view, role, my_value)
    if proc == "walk":
        return out
    # 72c297b9 / 8bf62f78: Faratin to cost, profit 0. Strong seller floor 1.10×.
    return Trace(steps=out.steps, value=_clamp_strong(view, role, my_value, out.value))


def profit(role: str, my_value: float, price: float) -> float:
    if role == "seller":
        return price - my_value
    return my_value - price


def field_payoff_price(
    view: GameView,
    role: str,
    my_value: float,
) -> float | None:
    """Price that gives us the public same-scenario median normalized payoff."""
    from glee_agent.reference import negotiation as public_reference
    from glee_agent.workflows.negotiation.classify import own_mult

    mult = own_mult(view)
    median_payoff = (public_reference(view) or {}).get("median_payoff_over_base")
    if not mult or not isinstance(median_payoff, (int, float)):
        return None
    base = my_value / mult
    if role == "seller":
        return round_money(my_value + float(median_payoff) * base)
    if role == "buyer":
        return round_money(my_value - float(median_payoff) * base)
    return None


def fitted_progress(view: GameView) -> float:
    """Continuous offer-time feature; counters describe the following round."""
    round_n = view.round + (1 if view.action_type == "decision" else 0)
    if view.horizon_known and view.max_rounds and view.max_rounds > 1:
        return min(1.0, max(0.0, (round_n - 1) / (view.max_rounds - 1)))
    return 1.0 - math.exp(-max(0, round_n - 1) / 4.0)


def fitted_offer_features(
    view: GameView,
    role: str,
    my_value: float,
    price: float,
) -> list[float] | None:
    from glee_agent.workflows.negotiation.classify import own_mult

    mult = own_mult(view)
    if not mult or role not in ("seller", "buyer"):
        return None
    base = my_value / mult
    if base <= 0:
        return None
    other = value_of(view, opponent_id(view.you))
    other_mult = float(other / base) if other is not None else 1.10
    state = state_of(view, role, my_value)
    seen = [profit(role, my_value, value) / base for value in opponent_prices(view)]
    last = seen[-1] if seen else 0.0
    rate = seen[-1] - seen[-2] if len(seen) >= 2 else 0.0
    repeated = min(state.plateau_repeats, 8) / 8.0
    features = [
        profit(role, my_value, price) / base,
        fitted_progress(view),
        float(mult),
        other_mult,
        min(state.own_rejections, 8) / 8.0,
        last,
        rate,
        repeated,
    ]
    if role == "seller":
        features.append(float(view.horizon_known))
    return features


def fitted_offer_curve(
    view: GameView,
    role: str,
    my_value: float,
    fallback: float,
    procedure: str = "",
) -> dict[str, Any] | None:
    """Choose a price from fitted accept and reject-continuation curves."""
    from glee_agent.workflows.negotiation.classify import own_mult

    mult = own_mult(view)
    if not mult or role not in ("seller", "buyer"):
        return None
    base = my_value / mult
    if base <= 0:
        return None
    from glee_agent.segments import segment

    seg = segment(view, procedure)
    zone = zopa(view)
    max_claim = 0.70
    if zone is not None:
        max_claim = max(0.001, (zone[1] - zone[0]) / base)
    fallback_claim = profit(role, my_value, fallback) / base
    accept_model = f"negotiation_{role}_offer_accept"
    continue_model = f"negotiation_{role}_offer_continue"
    fallback_features = fitted_offer_features(view, role, my_value, fallback)
    if fallback_features is None:
        return None
    fallback_pair = predict_pair_segmented(
        accept_model,
        continue_model,
        seg.parents,
        fallback_features,
    )
    if fallback_pair is None:
        return None
    _, _, level, accept_row, continue_row = fallback_pair
    if (
        int(level.rsplit("_", 1)[-1]) > 1
    ):
        return None
    support = accept_row.get("support") or []
    if not support or len(support[0]) != 2:
        return None
    support_lo, support_hi = (float(value) for value in support[0])
    if not support_lo - 1e-9 <= fallback_claim <= support_hi + 1e-9:
        return None
    claim_lo = max(0.001, support_lo)
    claim_hi = min(max_claim, support_hi)
    if claim_lo > claim_hi:
        return None
    claims = {
        round(claim_lo + index * 0.01, 6)
        for index in range(int((claim_hi - claim_lo) / 0.01) + 1)
        if claim_lo + index * 0.01 <= claim_hi + 1e-9
    }
    if fallback_claim >= 0:
        claims.add(min(max_claim, fallback_claim))
    rows: list[tuple[float, float, float, float, float]] = []
    for claim in sorted(claims):
        price = my_value + claim * base if role == "seller" else my_value - claim * base
        price = round_money(max(0.0, price))
        features = fitted_offer_features(view, role, my_value, price)
        if features is None:
            continue
        pair = predict_pair_segmented(
            accept_model,
            continue_model,
            seg.parents,
            features,
        )
        if pair is None:
            return None
        accepted, continued, candidate_level, candidate_accept, candidate_continue = pair
        if (
            candidate_level != level
            or candidate_accept is not accept_row
            or candidate_continue is not continue_row
        ):
            return None
        if view.is_last_round or (view.horizon_known and view.max_rounds == 1):
            continued = 0.0
        realized_claim = max(0.0, profit(role, my_value, price) / base)
        expected = accepted * realized_claim + (1.0 - accepted) * continued
        rows.append((expected, price, accepted, continued, realized_claim))
    if not rows:
        return None
    best = max(rows, key=lambda row: (row[0], row[4]))
    current = min(rows, key=lambda row: abs(row[1] - fallback))
    edge = best[0] - current[0]
    threshold = max(
        float(accept_row.get("guard", 0.015)),
        float(continue_row.get("guard", 0.015)),
    )
    chosen = best if edge > threshold else current
    return {
        "price": chosen[1],
        "expected_value_over_base": chosen[0],
        "accept_probability": chosen[2],
        "continue_value_over_base": chosen[3],
        "claim_over_base": chosen[4],
        "fallback_price": current[1],
        "fallback_expected_value_over_base": current[0],
        "model_edge": edge,
        "guard": threshold,
        "scenario_id": seg.scenario_id,
        "accept_model_level": level,
        "continue_model_level": level,
        "accept_model_n": int(accept_row.get("n") or 0),
        "continue_model_n": int(continue_row.get("n") or 0),
    }


def late(view: GameView) -> bool:
    """True from two rounds before a known deadline (matches classify _late)."""
    if view.is_last_round:
        return True
    if view.horizon_known and view.max_rounds:
        return view.round >= max(1, view.max_rounds - 2)
    return view.round >= 7 or progress(view) >= 0.70


def play_hidden(view: GameView, regime: str) -> dict[str, Any]:
    """Shared hidden offer/decide, including WalkAway on the dead-type leaf."""
    role = role_of(view)
    my_value = value_of(view) or 0.0
    proc = pick_procedure(view, regime, role, my_value)
    ask = compose_ask(view, regime, role, my_value).value
    if proc == "walk":
        if view.action_type == "offer":
            return with_message(view, {"product_price": ask}, "No deal here.")
        return walk()
    if view.action_type == "offer":
        return with_message(view, {"product_price": ask}, "This is my number.")
    price = last_price(view)
    if price is None:
        return reject(view, ask)
    if acceptable_hidden(view, role, my_value, price, ask):
        return {"decision": "AcceptOffer"}
    return reject(view, ask)


def reject(view: GameView, ask: float) -> dict[str, Any]:
    if view.is_last_round:
        return {"decision": "RejectOffer"}
    message = "Too far from a deal for me."
    last = view.state.get("last_offer")
    if view.messages_allowed and isinstance(last, dict):
        match = re.search(
            r"\bwrite\s+([A-Za-z0-9_-]{1,32})\s+in\s+(?:your\s+)?(?:text|message)\b",
            str(last.get("message") or ""),
            flags=re.IGNORECASE,
        )
        if match:
            message = match.group(1)
    return with_message(
        view,
        {"decision": "RejectOffer", "product_price": ask},
        message,
    )


def walk() -> dict[str, Any]:
    return {"decision": "WalkAway"}


def acceptable_zopa(view: GameView, role: str, my_value: float, price: float, ask: float) -> bool:
    gain = profit(role, my_value, price)
    if gain <= 0:
        return False
    zone = zopa(view)
    surplus = 0.0
    if zone is not None:
        surplus = max(zone[1] - zone[0], 0.0)
    if surplus <= 0 or zone is None:
        return gain > 0
    one_shot = bool(view.horizon_known and view.max_rounds == 1)
    # One-shot and multi-round last have no continuation: any positive gain
    # strictly dominates the zero payoff from rejection.
    if one_shot:
        return gain > 0
    if view.is_last_round:
        return gain > 0
    # The platform's nominally open horizon ends at round 99. A responder at
    # that cap has no continuation action, so positive profit dominates zero.
    if not view.horizon_known and view.round >= 99:
        return gain > 0
    # f8f129df: do not reject 1,001,600 and counter with 975,000 once the
    # current offer already reaches the public field's median payoff. The
    # field guard preserves bc9816ce, whose sub-median 28.8% ZOPA slice was
    # still worth rejecting despite a lower dynamic probe.
    from glee_agent.reference import negotiation as public_reference
    from glee_agent.workflows.negotiation.classify import own_mult

    mult = own_mult(view)
    field = public_reference(view) or {}
    median = field.get("median_payoff_over_base")
    current_beats_ask = (role == "seller" and price + 1e-9 >= ask) or (
        role == "buyer" and price - 1e-9 <= ask
    )
    stop_risk = state_of(view, role, my_value).platform_stop_risk
    if (current_beats_ask or stop_risk >= 0.8) and mult and isinstance(median, (int, float)):
        base = my_value / mult
        if gain + 1e-9 >= float(median) * base:
            return True
    share = gain / surplus
    if (
        not view.horizon_known
        and view.round >= 95
        and share > 0
        and style_of(view, role, my_value) == "stubborn"
        and same_price_repeats(view, price) >= 4
    ):
        return True
    from glee_agent.workflows.negotiation.classify import horizon_phase

    phase = horizon_phase(view)
    # The field's late attractor is still an even split, but accepting an early
    # half split scored poorly: six of seven scored samples at 49.5–50% were
    # negative (mean -3.16). Keep 50% late and require 55% while counters remain.
    # Last-round responses above already keep the any-positive exception.
    if phase == "early":
        need = 0.55
    else:
        need = 0.50
    style = style_of(view, role, my_value)
    best = best_offered_share(view, role)
    if style == "conceder" and best is not None:
        need = max(need, min(best, 0.70))
    if share + 1e-9 >= need:
        return True
    return False


def same_price_repeats(view: GameView, price: float) -> int:
    tolerance = max(1e-6, 0.015 * max(abs(price), 1.0))
    # Some game states omit sender ids, so the recent window can contain our
    # alternating counteroffers. Count the current plateau inside that window
    # instead of requiring adjacent rows or accumulating the full history.
    return sum(
        abs(seen - price) < tolerance for seen in opponent_prices(view)[-6:]
    )


def in_the_money(role: str, my_value: float, price: float) -> bool:
    """Official starter: seller accepts price >= value, buyer price <= value."""
    if role == "seller":
        return price + 1e-9 >= my_value
    return price - 1e-9 <= my_value


def acceptable_hidden(view: GameView, role: str, my_value: float, price: float, ask: float) -> bool:
    """Hidden: last round any non-neg; earlier, compare live bids with AC-next.

    528bd908: r1 accept at value (gain 0) is payoff-identical to no-deal and
    −5.08. Thin positive still closes a *live* cell (c0ef9a90 / 6c3e3e6c).
    Strong types (buyer 1.5× / seller 0.8×) follow their composed bid except last.
    """
    gain = profit(role, my_value, price)
    if view.is_last_round:
        # 0b8bf595: a hidden T=10 buyer repeated near-cost bids, then raised
        # only to 1.01x cost. Accepting transferred nearly the whole surplus.
        # Restrict refusal to the observed strong-seller plateau cell.
        if (
            role == "seller"
            and my_value > 0
            and gain / my_value <= 0.02 + 1e-9
            and same_price_repeats(view, price) >= 3
        ):
            from glee_agent.workflows.negotiation.classify import own_mult

            if own_mult(view) == 0.8:
                return False
        return gain >= 0
    # Six earlier cap-99 responses and two new holdout cases all improved by
    # accepting positive profit. A hidden-value seller responds on round 98
    # before its final counter; that restricted cell is also 2/2 with no
    # historical regression. Complete-information round 98 stays modelled.
    if (
        not view.horizon_known
        and (
            view.round >= 99
            or (not view.complete_information and view.round >= 98)
        )
    ):
        return gain > 0
    from glee_agent.workflows.negotiation.classify import own_band

    band = own_band(view)
    if band.startswith("strong"):
        if (
            band == "strong_seller"
            and role == "seller"
            and any(seen > my_value * 1.25 for seen in opponent_prices(view))
        ):
            return price + 1e-9 >= ask
        # A 1.23x-cost first counter identifies the hidden 1.2 buyer in all
        # six observed T=10 games. Accepting it scored -4.7 to -5.1; the
        # existing counter is 1.487x cost and has closed this cell at +7.5.
        if (
            band == "strong_seller"
            and role == "seller"
            and view.horizon_known
            and view.round == 2
            and (
                abs(price - my_value * 1.23) <= max(1e-6, my_value * 1e-6)
                or (my_value * 1.25 < price < ask)
            )
        ):
            return False
        # The strong buyer opens at 0.75× value; accepting at the former 0.90×
        # ceiling skipped a profitable counter and scored poorly against low-
        # cost sellers. Use the same adaptive price for bidding and accepting.
        if band == "strong_buyer" and role == "buyer":
            return price - 1e-9 <= ask
        if band == "strong_seller" and role == "seller" and not view.horizon_known:
            return price + 1e-9 >= ask
        close = my_value * (STRONG_SELLER_FLOOR if role == "seller" else 0.90)
        if role == "seller":
            return price + 1e-9 >= close
        return price - 1e-9 <= close
    if band == "live":
        if not view.horizon_known and same_price_repeats(view, price) >= 2:
            close = my_value * (1.10 if role == "seller" else 0.90)
            if role == "seller" and price + 1e-9 >= close:
                return True
            if role == "buyer" and price - 1e-9 <= close:
                return True
        if role == "seller":
            return price + 1e-9 >= ask
        return price - 1e-9 <= ask
    return gain > 0
