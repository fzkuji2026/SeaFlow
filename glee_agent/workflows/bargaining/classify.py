"""Pick one bargaining regime, then the cell axes compose uses."""

from __future__ import annotations

from glee_agent.rules import GameView, my_delta, opp_delta

PATIENT = 0.99
IMPATIENT = 0.97
HOT = 0.85


def inflation_band(view: GameView) -> str:
    d = my_delta(view)
    if d is None:
        return "unknown"
    if d >= PATIENT:
        return "none"
    if d >= IMPATIENT:
        return "mild"
    if d > HOT:
        return "mid"
    return "high"


def horizon_phase(view: GameView) -> str:
    if view.is_last_round:
        return "last"
    if view.horizon_known and view.max_rounds:
        left = view.max_rounds - view.round
        if left <= 1:
            return "penultimate"
        if view.round <= 3:
            return "early"
        if left <= 2:
            return "late"
        return "mid"
    if not view.horizon_known:
        if view.round >= 12:
            return "late"
        if view.round <= 3:
            return "early"
        return "mid"
    return "unknown"


def info_band(view: GameView) -> str:
    return "complete" if view.complete_information else "incomplete"


def we_hotter(view: GameView) -> bool:
    d_me = my_delta(view)
    d_opp = opp_delta(view)
    return d_me is not None and d_opp is not None and d_me < d_opp


def classify(view: GameView) -> str:
    d_me = my_delta(view)
    d_opp = opp_delta(view)
    if view.is_last_round:
        return "deadline"
    if view.horizon_known and view.max_rounds and view.round >= view.max_rounds - 1:
        return "penultimate"
    if d_me is not None and d_opp is not None and d_me <= HOT and d_opp >= PATIENT:
        return "powerless"
    if d_me is not None and d_me <= HOT:
        return "impatient_hot"
    # They inflate faster than we do: first even-split is theirs to repeat.
    if (
        d_me is not None
        and d_opp is not None
        and d_me < IMPATIENT
        and d_opp < d_me
    ):
        return "impatient_vs_hotter"
    if d_me is not None and d_me < IMPATIENT:
        return "impatient"
    if d_me is not None and d_me >= PATIENT and d_opp is not None and d_opp < PATIENT:
        return "patient_vs_impatient"
    if d_me is not None and d_me >= PATIENT:
        return "patient"
    if not view.horizon_known and d_me is not None:
        return "unknown_horizon"
    if not view.complete_information and d_me is not None:
        return "incomplete"
    return "symmetric"


def situation(view: GameView) -> tuple[str, str, str, str]:
    """(regime, info, inflation_band, horizon_phase)."""
    return (classify(view), info_band(view), inflation_band(view), horizon_phase(view))
