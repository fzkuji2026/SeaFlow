"""Infer how the opponent bids from THEIR offers only."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from glee_agent.rules import GameView, as_player_id, money, offer_gain, opp_delta


@dataclass(frozen=True)
class OpponentState:
    style: str
    samples: int
    current_share: float | None
    best_share: float | None
    concession_rate: float
    slope_uncertainty: float
    predicted_next_share: float | None
    prediction_confidence: float
    plateau_span: float | None
    plateau_repeats: int
    own_rejections: int
    rejected_share_floor: float | None
    post_rejection_gain: float | None
    estimated_patience: float
    patience_confidence: float
    stall_probability: float


def their_shares(view: GameView) -> list[float]:
    pot = money(view)
    if pot <= 0:
        return []
    out: list[float] = []
    seen: set[tuple[object, str, float]] = set()
    rows = list(view.state.get("history") or [])
    last = view.state.get("last_offer")
    if isinstance(last, dict):
        rows = rows + [last]
    for row in rows:
        if not isinstance(row, dict):
            continue
        proposer = str(row.get("proposer") or "")
        offer = row.get("offer") or row.get("last_offer") or row
        if isinstance(offer, dict):
            proposer = str(offer.get("proposer") or proposer)
        if proposer and as_player_id(proposer) == as_player_id(view.you):
            continue
        gain = offer_gain(offer if isinstance(offer, dict) else None, view.you)
        if gain is None:
            continue
        round_n = offer.get("round") or row.get("round")
        identity = (round_n, proposer, gain)
        if round_n is not None:
            if identity in seen:
                continue
            seen.add(identity)
        out.append(gain / pot)
    return out


def _style(xs: list[float]) -> str:
    if not xs:
        return "unknown"
    if max(xs) <= 0.22:
        return "lowballer"
    if all(0.44 <= x <= 0.53 for x in xs):
        return "half_bot"
    recent = xs[-4:]
    if len(recent) >= 3 and max(recent[-3:]) - min(recent[-3:]) < 0.015:
        return "stubborn"
    if len(recent) >= 2 and abs(recent[-1] - recent[0]) < 0.015:
        return "stubborn"
    if len(recent) >= 2 and recent[-1] > recent[0] + 0.02:
        return "conceder"
    return "mixed"


def _our_rejected_shares(view: GameView) -> list[float]:
    pot = money(view)
    if pot <= 0:
        return []
    mine = as_player_id(view.you)
    out: list[float] = []
    for row in view.state.get("history") or []:
        if not isinstance(row, dict) or "reject" not in str(row.get("decision") or "").lower():
            continue
        offer = row.get("offer") or row
        if not isinstance(offer, dict):
            continue
        proposer = str(offer.get("proposer") or row.get("proposer") or "")
        if proposer and as_player_id(proposer) != mine:
            continue
        gain = offer_gain(offer, view.you)
        if gain is not None:
            out.append(gain / pot)
    return out


def state_of(view: GameView) -> OpponentState:
    """Estimate opponent patience and next bid from visible offers only."""
    xs = their_shares(view)
    current = xs[-1] if xs else None
    recent = xs[-4:]
    slopes = [right - left for left, right in zip(recent, recent[1:])]
    rate = median(slopes) if slopes else 0.0
    uncertainty = median([abs(slope - rate) for slope in slopes]) if slopes else 0.0
    confidence = min(1.0, len(slopes) / 3.0)
    shrunk_rate = rate * confidence
    predicted = (
        None
        if current is None
        else min(1.0, max(0.0, current + shrunk_rate))
    )
    plateau_recent = xs[-3:]
    plateau_span = (
        max(plateau_recent) - min(plateau_recent)
        if len(plateau_recent) >= 2
        else None
    )
    repeats = 0
    if current is not None:
        for seen in reversed(xs):
            if abs(seen - current) >= 0.01:
                break
            repeats += 1
    rejected = _our_rejected_shares(view)
    post_rejection_gain = (
        current - xs[-2]
        if current is not None and len(xs) >= 2 and rejected
        else None
    )
    known_patience = opp_delta(view)
    if known_patience is not None:
        patience = known_patience
        patience_confidence = 1.0
    elif repeats >= 3:
        patience = 1.0
        patience_confidence = min(0.9, 0.45 + 0.1 * repeats)
    elif rate >= 0.02:
        patience = 0.90
        patience_confidence = min(0.8, 0.25 + 0.1 * len(xs))
    else:
        patience = 0.95
        patience_confidence = min(0.7, 0.15 + 0.1 * len(xs))
    stable = (
        0.0
        if plateau_span is None
        else max(0.0, 1.0 - plateau_span / 0.04)
    )
    sample_evidence = min(1.0, max(0, len(recent) - 1) / 3.0)
    rejection_evidence = (
        1.0
        if post_rejection_gain is not None and post_rejection_gain <= 0.01
        else 0.5 if rejected else 0.0
    )
    stall = stable * sample_evidence * (0.5 + 0.5 * rejection_evidence)
    return OpponentState(
        style=_style(xs),
        samples=len(xs),
        current_share=current,
        best_share=max(xs) if xs else None,
        concession_rate=rate,
        slope_uncertainty=uncertainty,
        predicted_next_share=predicted,
        prediction_confidence=confidence,
        plateau_span=plateau_span,
        plateau_repeats=repeats,
        own_rejections=len(rejected),
        rejected_share_floor=min(rejected) if rejected else None,
        post_rejection_gain=post_rejection_gain,
        estimated_patience=patience,
        patience_confidence=patience_confidence,
        stall_probability=stall,
    )


def style_of(view: GameView) -> str:
    return state_of(view).style
