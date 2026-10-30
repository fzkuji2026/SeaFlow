"""Runtime for offline-fitted decision curves (stdlib only)."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

PATH = Path(__file__).with_name("fitted_models.json")


@lru_cache(maxsize=1)
def data() -> dict[str, Any]:
    try:
        loaded = json.loads(PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def model(name: str) -> dict[str, Any] | None:
    row = (data().get("models") or {}).get(name)
    return row if isinstance(row, dict) else None


def predict(name: str, features: list[float]) -> float | None:
    row = model(name)
    if row is None:
        return None
    mean = row.get("mean") or []
    scale = row.get("scale") or []
    coef = row.get("coef") or []
    if not (len(features) == len(mean) == len(scale) == len(coef)):
        return None
    score = float(row.get("intercept") or 0.0)
    for value, center, width, weight in zip(features, mean, scale, coef):
        width = float(width) or 1.0
        score += ((float(value) - float(center)) / width) * float(weight)
    if row.get("kind") == "logistic":
        score = max(-40.0, min(40.0, score))
        return 1.0 / (1.0 + math.exp(-score))
    lo, hi = row.get("clip") or [None, None]
    if lo is not None:
        score = max(float(lo), score)
    if hi is not None:
        score = min(float(hi), score)
    return score


def predict_segmented(
    name: str,
    parents: tuple[str, ...],
    features: list[float],
) -> tuple[float, str, dict[str, Any]] | None:
    table = ((data().get("segmented_models") or {}).get(name) or {})
    for level, key in enumerate(parents):
        row = table.get(key)
        if not isinstance(row, dict):
            continue
        score = _predict_row(row, features)
        if score is None:
            continue
        return score, f"level_{level}", row
    return None


def predict_pair_segmented(
    first: str,
    second: str,
    parents: tuple[str, ...],
    features: list[float],
) -> tuple[float, float, str, dict[str, Any], dict[str, Any]] | None:
    """Predict two targets from the same segmentation parent."""
    tables = data().get("segmented_models") or {}
    first_table = tables.get(first) or {}
    second_table = tables.get(second) or {}
    for level, key in enumerate(parents):
        first_row = first_table.get(key)
        second_row = second_table.get(key)
        if not isinstance(first_row, dict) or not isinstance(second_row, dict):
            continue
        first_value = _predict_row(first_row, features)
        second_value = _predict_row(second_row, features)
        if first_value is None or second_value is None:
            continue
        return first_value, second_value, f"level_{level}", first_row, second_row
    return None


def _predict_row(row: dict[str, Any], features: list[float]) -> float | None:
    mean = row.get("mean") or []
    scale = row.get("scale") or []
    coef = row.get("coef") or []
    if not (len(features) == len(mean) == len(scale) == len(coef)):
        return None
    score = float(row.get("intercept") or 0.0)
    for value, center, width, weight in zip(features, mean, scale, coef):
        width = float(width) or 1.0
        score += ((float(value) - float(center)) / width) * float(weight)
    if row.get("kind") == "logistic":
        score = max(-40.0, min(40.0, score))
        return 1.0 / (1.0 + math.exp(-score))
    lo, hi = row.get("clip") or [None, None]
    if lo is not None:
        score = max(float(lo), score)
    if hi is not None:
        score = min(float(hi), score)
    return score


def policy_segmented(
    name: str,
    parents: tuple[str, ...],
    predicate: Callable[[dict[str, Any], int], bool] | None = None,
) -> tuple[dict[str, Any], str] | None:
    table = ((data().get("segmented_policies") or {}).get(name) or {})
    for level, key in enumerate(parents):
        row = table.get(key)
        if isinstance(row, dict) and (
            predicate is None or predicate(row, level)
        ):
            return row, f"level_{level}"
    return None


def guard(name: str, default: float = 0.0) -> float:
    row = model(name) or {}
    try:
        return max(0.0, float(row.get("guard", default)))
    except (TypeError, ValueError):
        return default


def prior(group: str, key: str, signal: str) -> tuple[float, float]:
    """Return (mean, strength) for an empirical-Bayes response prior."""
    table = ((data().get("priors") or {}).get(group) or {})
    row = table.get(f"{key}|{signal}") or table.get(f"*|{signal}") or {}
    try:
        return float(row.get("mean", 0.5)), float(row.get("strength", 1.0))
    except (TypeError, ValueError):
        return 0.5, 1.0


def clear_cache() -> None:
    data.cache_clear()
