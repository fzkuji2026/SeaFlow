#!/usr/bin/env python3
"""Fit a chronological, recency-weighted bargaining rating surrogate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT)]

from glee_agent.diagnose import replay_to_game  # noqa: E402
from glee_agent.rating_surrogate import FEATURES, features  # noqa: E402
from glee_agent.rules import as_player_id, parse  # noqa: E402


def _timestamp(game: dict) -> float | None:
    try:
        return datetime.fromisoformat(str(game["completed_at"])).timestamp()
    except (KeyError, TypeError, ValueError):
        return None


def samples(game: dict) -> list[tuple[list[float], float, float]]:
    """Return (features, rating_delta, completed_at) for our visible choices."""
    if game.get("game_family") != "bargaining" or not isinstance(
        game.get("rating_delta"), (int, float)
    ):
        return []
    timestamp = _timestamp(game)
    if timestamp is None:
        return []
    you = as_player_id(str(game.get("your_player") or "player_1"))
    rows = []
    for row in (game.get("state") or game.get("game_state") or {}).get("history") or []:
        if not isinstance(row, dict) or not isinstance(row.get("offer"), dict):
            continue
        offer = row["offer"]
        proposer = as_player_id(
            str(
                row.get("proposer")
                or offer.get("proposer")
                or offer.get("from_player")
                or ""
            )
        )
        round_n = int(row.get("round") or 1)
        response = proposer != you
        if not response and proposer != you:
            continue
        if response:
            decider = row.get("decided_by")
            if decider and as_player_id(str(decider)) != you:
                continue
            action_type = "decision"
        else:
            action_type = "offer"
        try:
            view = parse(replay_to_game(game, round_n=round_n, action_type=action_type))
        except (KeyError, TypeError, ValueError):
            continue
        values = features(view, offer, response=response)
        if values is not None:
            rows.append((values, float(game["rating_delta"]), timestamp))
    return rows


def _fit(
    kind: str, rows: list[tuple[list[float], float, float, float]], weights: list[float]
) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray(
        [row[1] > 0 if kind == "logistic" else row[1] for row in rows], dtype=float
    )
    scaler = StandardScaler().fit(x)
    if kind == "logistic":
        estimator = LogisticRegression(C=0.2, max_iter=2000).fit(
            scaler.transform(x), y.astype(int), sample_weight=weights
        )
    else:
        estimator = Ridge(alpha=20.0).fit(scaler.transform(x), y, sample_weight=weights)
    return {
        "kind": kind,
        "features": list(FEATURES),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": estimator.coef_.ravel().tolist(),
        "intercept": float(
            estimator.intercept_.ravel()[0]
            if hasattr(estimator.intercept_, "ravel")
            else estimator.intercept_
        ),
    }


def _predict(model: dict, values: list[float]) -> float:
    score = float(model["intercept"])
    for value, center, width, coef in zip(
        values, model["mean"], model["scale"], model["coef"]
    ):
        score += ((value - center) / (width or 1.0)) * coef
    return (
        1 / (1 + math.exp(-max(-40.0, min(40.0, score))))
        if model["kind"] == "logistic"
        else score
    )


def train(
    rows: list[tuple[list[float], float, float, float]], half_life_hours: float = 3.0
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        mean_absolute_error,
        mean_squared_error,
        roc_auc_score,
    )

    rows = sorted(rows, key=lambda row: row[2])
    cut = max(1, min(len(rows) - 1, int(len(rows) * 0.8)))
    holdout_start = rows[cut][2]
    train_rows = [row for row in rows if row[2] < holdout_start]
    holdout = [row for row in rows if row[2] >= holdout_start]
    if not train_rows or not holdout:
        raise ValueError("chronological split needs distinct timestamps")
    latest = max(row[2] for row in train_rows)
    half_life = max(1.0, half_life_hours * 3600.0)
    weights = [
        row[3] * 0.5 ** (max(0.0, latest - row[2]) / half_life) for row in train_rows
    ]
    positive = _fit("logistic", train_rows, weights)
    delta = _fit("linear", train_rows, weights)
    truth_positive = [int(row[1] > 0) for row in holdout]
    pred_positive = [_predict(positive, row[0]) for row in holdout]
    truth_delta = [row[1] for row in holdout]
    pred_delta = [_predict(delta, row[0]) for row in holdout]
    baseline_probability = sum(
        int(row[1] > 0) * weight for row, weight in zip(train_rows, weights)
    ) / sum(weights)
    baseline_delta = sum(
        row[1] * weight for row, weight in zip(train_rows, weights)
    ) / sum(weights)
    metrics = {
        "train_n": len(train_rows),
        "holdout_n": len(holdout),
        "holdout_start": holdout[0][2],
        "positive_auc": (
            round(float(roc_auc_score(truth_positive, pred_positive)), 6)
            if len(set(truth_positive)) == 2
            else None
        ),
        "positive_brier": round(
            float(brier_score_loss(truth_positive, pred_positive)), 6
        ),
        "positive_baseline_brier": round(
            sum((baseline_probability - y) ** 2 for y in truth_positive) / len(holdout),
            6,
        ),
        "positive_accuracy": round(
            float(accuracy_score(truth_positive, [p >= 0.5 for p in pred_positive])), 6
        ),
        "delta_mae": round(float(mean_absolute_error(truth_delta, pred_delta)), 6),
        "delta_baseline_mae": round(
            sum(abs(y - baseline_delta) for y in truth_delta) / len(holdout), 6
        ),
        "delta_rmse": round(
            math.sqrt(float(mean_squared_error(truth_delta, pred_delta))), 6
        ),
    }
    all_latest = max(row[2] for row in rows)
    all_weights = [
        row[3] * 0.5 ** (max(0.0, all_latest - row[2]) / half_life) for row in rows
    ]
    return {
        "schema_version": 1,
        "shadow_only": True,
        "half_life_hours": half_life_hours,
        "metrics": metrics,
        "models": {
            "positive": _fit("logistic", rows, all_weights),
            "delta": _fit("linear", rows, all_weights),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=Path, default=ROOT / "logs" / "games")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "glee_agent" / "rating_surrogate.json"
    )
    parser.add_argument("--agent")
    parser.add_argument("--half-life-hours", type=float, default=3.0)
    args = parser.parse_args()
    rows = []
    games = 0
    for path in args.games.glob("*.json"):
        try:
            game = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if args.agent and game.get("your_player_id") != args.agent:
            continue
        game_rows = samples(game)
        if game_rows:
            weight = 1.0 / len(game_rows)
            rows.extend(
                (values, delta, timestamp, weight)
                for values, delta, timestamp in game_rows
            )
            games += 1
    if len(rows) < 50:
        raise SystemExit("need at least 50 bargaining decision samples")
    fitted = train(rows, args.half_life_hours)
    fitted["games"] = games
    fitted["samples"] = len(rows)
    args.output.write_text(json.dumps(fitted, indent=2, sort_keys=True) + "\n")
    print(json.dumps(fitted["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
