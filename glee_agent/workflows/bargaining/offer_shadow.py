"""Read-only challenger for context-specific bargaining offers."""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView, as_player_id, money, my_delta, offer_gain


_TABLE: dict[str, list[dict[str, Any]]] = {
    "bob_d1_hidden_t12_nomsg_first_counter": [
        {"share": 0.60, "train": (110, 0.564, 0.419, 0.338), "holdout": (63, 0.587, 2.498, 0.352)},
        {"share": 0.55, "train": (49, 0.163, -0.961, 0.090), "holdout": (34, 0.206, 3.376, 0.113)},
    ],
    "bob_d1_hidden_open_nomsg_first_counter": [
        {"share": 0.60, "train": (116, 0.509, 1.121, 0.308), "holdout": (58, 0.586, 2.050, 0.352)},
        {"share": 0.55, "train": (77, 0.260, -0.330, 0.143), "holdout": (37, 0.135, -0.116, 0.074)},
    ],
    "bob_d095_hidden_t12_nomsg_first_counter": [
        {"share": 0.525, "train": (25, 0.760, 0.176, 0.384), "holdout": (19, 0.737, 0.411, 0.372)},
        {"share": 0.50, "train": (73, 0.781, -2.975, 0.319), "holdout": (17, 0.824, -3.300, 0.391)},
    ],
    "alice_d1_hidden_t12_repeat_60": [
        {"share": 0.55, "train": (60, 0.330, -0.360, 0.182), "holdout": (53, 0.340, 1.130, 0.187)},
    ],
    "bob_d1_hidden_t12_deadline": [
        {"share": 1.00, "train": (9, 1.000, 7.089, 1.000), "holdout": (14, 1.000, 7.200, 1.000)},
    ],
}


def _prior_own_shares(view: GameView) -> list[float]:
    pot = money(view)
    if pot <= 0:
        return []
    mine = as_player_id(view.you)
    out: list[float] = []
    for row in view.state.get("history") or []:
        if not isinstance(row, dict):
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


def _route(view: GameView, prior: list[float]) -> tuple[str, str | None]:
    if view.is_last_round:
        route = "deadline_claim"
    elif not prior and view.round > 1:
        route = "first_counter"
    elif prior:
        route = "repeat_after_reject"
    else:
        return "ordinary_offer", None

    hidden = not view.complete_information
    no_messages = not view.messages_allowed
    delta = my_delta(view)
    bob = as_player_id(view.you) == "player_2"
    t12 = view.horizon_known and view.max_rounds == 12
    opened = not view.horizon_known
    if route == "deadline_claim" and bob and delta == 1.0 and hidden and t12:
        return route, "bob_d1_hidden_t12_deadline"
    if route == "repeat_after_reject" and not bob and delta == 1.0 and hidden and t12 and abs(prior[-1] - 0.60) <= 0.0125:
        return route, "alice_d1_hidden_t12_repeat_60"
    if route != "first_counter" or not bob or not hidden or not no_messages:
        return route, None
    if delta == 1.0 and t12:
        return route, "bob_d1_hidden_t12_nomsg_first_counter"
    if delta == 1.0 and opened:
        return route, "bob_d1_hidden_open_nomsg_first_counter"
    if delta == 0.95 and t12:
        return route, "bob_d095_hidden_t12_nomsg_first_counter"
    return route, None


def predict(view: GameView) -> dict[str, Any] | None:
    """Return evidence-backed candidates without influencing the live action."""
    if view.family != "bargaining" or view.action_type != "offer":
        return None
    prior = _prior_own_shares(view)
    route, context = _route(view, prior)
    rows = _TABLE.get(context or "", [])

    def evidence(values: tuple[int, float, float, float]) -> dict[str, Any]:
        n, acceptance, rating, discounted_ev = values
        return {
            "n": n,
            "acceptance_probability": acceptance,
            "mean_rating_delta": rating,
            "discounted_immediate_ev": discounted_ev,
        }

    candidates = [
        {
            "share": row["share"],
            "train": evidence(row["train"]),
            "holdout": evidence(row["holdout"]),
        }
        for row in rows
    ]
    return {
        "shadow_only": True,
        "route": route,
        "context": context,
        "prior_own_share": prior[-1] if prior else None,
        "candidates": candidates,
        "recommended_share": candidates[0]["share"] if candidates else None,
    }
