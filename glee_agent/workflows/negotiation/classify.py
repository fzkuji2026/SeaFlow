"""How to recognize each negotiation situation. One name → one handler."""

from __future__ import annotations

from glee_agent.rules import GameView, role_of, value_of, zopa


def _late(view: GameView) -> bool:
    if view.horizon_known and view.max_rounds:
        return view.round >= max(1, view.max_rounds - 2)
    return view.round >= 7


def classify(view: GameView) -> str:
    zone = zopa(view)
    both = value_of(view, "player_1") is not None and value_of(view, "player_2") is not None
    role = role_of(view)
    # no overlap: both values known and the interval is empty
    if view.complete_information and both and zone is None:
        return "no_zopa"
    one = bool(view.horizon_known and view.max_rounds == 1)
    # one round, both values: Nash split (not the hidden opener)
    if one and zone is not None:
        return "ultimatum"
    # one round, only our value: 24/89 recent games died here
    if one:
        return "one_shot"
    if zone is not None and view.is_last_round:
        return "ultimatum"
    if zone is not None and _late(view):
        return "complete_late"
    if zone is not None and role == "seller":
        return "complete_seller"
    if zone is not None:
        return "complete"
    if view.is_last_round:
        return "incomplete_deadline"
    # A rejected decision counter is posted on the following round. Close when
    # that outgoing offer is penultimate; the last response cannot counter.
    outgoing_round = view.round + (1 if view.action_type == "decision" else 0)
    if view.horizon_known and view.max_rounds and outgoing_round >= view.max_rounds - 1:
        return "incomplete_late"
    # no deadline posted — 13 recent hidden no-deals
    if not view.horizon_known:
        return "open_horizon"
    if role == "seller":
        return "incomplete_seller"
    return "incomplete"



def info_band(view: GameView) -> str:
    from glee_agent.rules import zopa

    if zopa(view) is not None:
        return "complete"
    if view.complete_information:
        return "complete"
    return "hidden"


def horizon_phase(view: GameView) -> str:
    if view.horizon_known and view.max_rounds == 1:
        return "one_shot"
    if view.is_last_round:
        return "last"
    if not view.horizon_known:
        if view.round <= 3:
            return "early"
        if view.round >= 7:
            return "late"
        return "mid"
    if view.horizon_known and view.max_rounds:
        left = view.max_rounds - view.round
        if left <= 1:
            return "penultimate"
        if view.round <= 3:
            return "early"
        if left <= 2:
            return "late"
        return "mid"
    return "unknown"


_BASES = (100.0, 10_000.0, 1_000_000.0)
_MULTS = (0.8, 1.0, 1.2, 1.5)


def own_mult(view: GameView) -> float | None:
    """Map our hidden value onto the official {0.8,1.0,1.2,1.5}×{100,10k,1M} grid."""
    val = value_of(view)
    if val is None or val <= 0:
        return None
    best: float | None = None
    best_err = 1e9
    for base in _BASES:
        ratio = val / base
        for target in _MULTS:
            err = abs(ratio - target)
            if err < best_err:
                best_err = err
                best = target
    if best is None or best_err > 0.05:
        return None
    return best


def own_band(view: GameView) -> str:
    """One hidden type. Seller and buyer extremes are different leaves."""
    role = role_of(view)
    mult = own_mult(view)
    if mult is None:
        return "unknown"
    if role == "seller" and mult >= 1.45:
        return "dead_seller"
    if role == "buyer" and mult <= 0.85:
        return "dead_buyer"
    if role == "seller" and mult <= 0.85:
        return "strong_seller"
    if role == "buyer" and mult >= 1.45:
        return "strong_buyer"
    return "live"


def zopa_band(view: GameView) -> str:
    from glee_agent.rules import zopa

    zone = zopa(view)
    if zone is None:
        return "hidden"
    surplus = max(zone[1] - zone[0], 0.0)
    if surplus <= 0:
        return "none"
    mid = abs((zone[0] + zone[1]) / 2.0) or 1.0
    if surplus / mid < 0.15:
        return "narrow"
    return "wide"
