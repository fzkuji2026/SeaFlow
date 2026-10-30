"""Stdlib-only shadow inference for the offline bargaining rating model."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from glee_agent.rules import GameView, money, my_delta, offer_gain, opp_delta

PATH = Path(__file__).with_name("rating_surrogate.json")

FEATURES = (
    "discounted_share",
    "share",
    "progress",
    "role_alice",
    "own_delta",
    "visible_opp_delta",
    "opp_delta_known",
    "complete_information",
    "horizon_known",
    "messages_allowed",
    "is_response",
    "log_round",
    "share_x_delta",
    "share_x_progress",
    "share_x_role",
    "response_x_progress",
)


def features(
    view: GameView, candidate: dict[str, Any], *, response: bool
) -> list[float] | None:
    """Return only information available before the candidate action is chosen."""
    pot = money(view)
    gain = offer_gain(candidate, view.you)
    own = my_delta(view)
    if not pot or gain is None or own is None:
        return None
    share = max(0.0, min(1.0, float(gain) / float(pot)))
    own = float(own)
    if view.horizon_known and view.max_rounds:
        progress = min(1.0, max(0.0, (view.round - 1) / max(1, view.max_rounds - 1)))
    else:
        progress = 1.0 - 1.0 / max(1, view.round)
    # Replays retain the hidden value in config; it was not visible at decision time.
    other = opp_delta(view) if view.complete_information else None
    other_known = float(other is not None)
    visible_other = float(other) if other is not None else 0.0
    role_alice = float(view.you == "player_1")
    is_response = float(response)
    return [
        share * own ** max(0, view.round - 1),
        share,
        progress,
        role_alice,
        own,
        visible_other,
        other_known,
        float(view.complete_information),
        float(view.horizon_known),
        float(view.messages_allowed),
        is_response,
        math.log1p(max(0, view.round - 1)),
        share * own,
        share * progress,
        share * role_alice,
        is_response * progress,
    ]


@lru_cache(maxsize=1)
def data() -> dict[str, Any]:
    try:
        value = json.loads(PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _predict(row: dict[str, Any], values: list[float]) -> float | None:
    mean, scale, coef = (
        row.get("mean") or [],
        row.get("scale") or [],
        row.get("coef") or [],
    )
    if not (len(values) == len(mean) == len(scale) == len(coef)):
        return None
    score = float(row.get("intercept") or 0.0)
    for value, center, width, weight in zip(values, mean, scale, coef):
        score += ((float(value) - float(center)) / (float(width) or 1.0)) * float(
            weight
        )
    if row.get("kind") == "logistic":
        score = max(-40.0, min(40.0, score))
        return 1.0 / (1.0 + math.exp(-score))
    return score


def predict(
    view: GameView, candidate: dict[str, Any], *, response: bool
) -> dict[str, float] | None:
    """Return shadow predictions; callers must not use them to choose an action."""
    values = features(view, candidate, response=response)
    models = data().get("models") or {}
    if values is None or not isinstance(models, dict):
        return None
    positive = _predict(models.get("positive") or {}, values)
    delta = _predict(models.get("delta") or {}, values)
    if positive is None or delta is None:
        return None
    return {
        "positive_probability": round(positive, 6),
        "expected_rating_delta": round(delta, 6),
    }


def clear_cache() -> None:
    data.cache_clear()
