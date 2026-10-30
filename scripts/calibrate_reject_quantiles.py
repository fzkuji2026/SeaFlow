#!/usr/bin/env python3
"""Fit and calibrate role/horizon payoff quantiles after bargaining rejects."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT)]

from glee_agent.diagnose import replay_to_game  # noqa: E402
from glee_agent.rating_surrogate import FEATURES as BASE_FEATURES  # noqa: E402
from glee_agent.rating_surrogate import features as base_features  # noqa: E402
from glee_agent.rules import as_player_id, offer_gain, parse  # noqa: E402

QUANTILES = (0.1, 0.5, 0.9)
FEATURES = BASE_FEATURES + (
    "prior_opponent_offers",
    "same_offer_repeats",
    "opponent_concession",
)


def _timestamp(game: dict) -> float | None:
    try:
        return datetime.fromisoformat(str(game["completed_at"])).timestamp()
    except (KeyError, TypeError, ValueError):
        return None


def _samples(game: dict) -> list[tuple]:
    if game.get("game_family") != "bargaining" or _timestamp(game) is None:
        return []
    you = as_player_id(str(game.get("your_player") or "player_1"))
    state = game.get("state") or game.get("game_state") or {}
    history = [row for row in state.get("history") or [] if isinstance(row, dict)]
    config = game.get("config") or {}
    pot = float(config.get("money_to_divide") or state.get("money_to_divide") or 0)
    if pot <= 0:
        return []
    payoff = float((game.get("result") or {}).get(f"{you}_payoff") or 0) / pot
    out = []
    previous: list[float] = []
    for row in history:
        offer = row.get("offer") or {}
        proposer = as_player_id(
            str(
                row.get("proposer")
                or offer.get("proposer")
                or offer.get("from_player")
                or ""
            )
        )
        if proposer == you:
            continue
        gain = offer_gain(offer, you)
        if gain is None:
            continue
        share = float(gain) / pot
        rejected = "reject" in str(row.get("decision") or "").lower()
        decider = row.get("decided_by")
        ours = not decider or as_player_id(str(decider)) == you
        if rejected and ours:
            round_n = int(row.get("round") or 1)
            try:
                view = parse(
                    replay_to_game(game, round_n=round_n, action_type="decision")
                )
            except (KeyError, TypeError, ValueError):
                previous.append(share)
                continue
            values = base_features(view, offer, response=True)
            if values is not None:
                repeats = sum(abs(old - share) < 1e-6 for old in previous)
                concession = share - previous[-1] if previous else 0.0
                values.extend([min(10, len(previous)), min(10, repeats), concession])
                group = f"{'alice' if you == 'player_1' else 'bob'}|{'known' if view.horizon_known else 'open'}"
                out.append(
                    (
                        values,
                        payoff,
                        _timestamp(game),
                        group,
                        str(game.get("game_id") or ""),
                    )
                )
        previous.append(share)
    weight = 1.0 / max(1, len(out))
    return [(*row, weight) for row in out]


def load(folder: Path, agents: set[str]) -> list[tuple]:
    rows = []
    for path in folder.glob("*.json"):
        try:
            game = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if agents and str(game.get("your_player_id") or "") not in agents:
            continue
        rows.extend(_samples(game))
    return sorted(rows, key=lambda row: row[2])


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _coverage(rows: list[tuple], predictions: list[list[float]]) -> dict:
    actual = [row[1] for row in rows]
    coverages = [
        sum(y <= p[i] for y, p in zip(actual, predictions)) / len(rows)
        for i in range(3)
    ]
    interval = sum(p[0] <= y <= p[2] for y, p in zip(actual, predictions)) / len(rows)
    return {
        "n": len(rows),
        "q10": round(coverages[0], 6),
        "q50": round(coverages[1], 6),
        "q90": round(coverages[2], 6),
        "interval": round(interval, 6),
        "crossing": round(
            sum(not (p[0] <= p[1] <= p[2]) for p in predictions) / len(rows), 6
        ),
    }


def _fit_group(rows: list[tuple], half_life_hours: float) -> dict:
    import numpy as np
    from sklearn.linear_model import QuantileRegressor
    from sklearn.preprocessing import StandardScaler

    timestamps = sorted({row[2] for row in rows})
    train_end, calibration_end = (
        timestamps[int(len(timestamps) * 0.6)],
        timestamps[int(len(timestamps) * 0.8)],
    )
    training = [row for row in rows if row[2] < train_end]
    calibration = [row for row in rows if train_end <= row[2] < calibration_end]
    test = [row for row in rows if row[2] >= calibration_end]
    if min(map(len, (training, calibration, test))) < 30:
        return {
            "deployable": False,
            "reason": "insufficient_time_split_support",
            "n": len(rows),
        }
    scaler = StandardScaler().fit(np.asarray([row[0] for row in training]))
    latest = max(row[2] for row in training)
    weights = [
        row[5] * 0.5 ** ((latest - row[2]) / max(1, half_life_hours * 3600))
        for row in training
    ]
    x = scaler.transform(np.asarray([row[0] for row in training]))
    y = np.asarray([row[1] for row in training])
    estimators = []
    for quantile in QUANTILES:
        model = QuantileRegressor(quantile=quantile, alpha=0.002, solver="highs")
        model.fit(x, y, sample_weight=weights)
        estimators.append(model)

    def raw(part: list[tuple]) -> list[list[float]]:
        x_part = scaler.transform(np.asarray([row[0] for row in part]))
        return [
            [float(model.predict(sample.reshape(1, -1))[0]) for model in estimators]
            for sample in x_part
        ]

    calibration_raw = raw(calibration)
    offsets = [
        _quantile(
            [
                row[1] - prediction[index]
                for row, prediction in zip(calibration, calibration_raw)
            ],
            quantile,
        )
        for index, quantile in enumerate(QUANTILES)
    ]

    def calibrated(part: list[tuple]) -> list[list[float]]:
        return [
            sorted(value + offset for value, offset in zip(prediction, offsets))
            for prediction in raw(part)
        ]

    predictions = calibrated(test)
    overall = _coverage(test, predictions)
    blocks = []
    for block in range(4):
        lo, hi = len(test) * block // 4, len(test) * (block + 1) // 4
        if hi - lo >= 20:
            blocks.append(_coverage(test[lo:hi], predictions[lo:hi]))
    overall_ok = (
        abs(overall["q10"] - 0.1) <= 0.05
        and abs(overall["q90"] - 0.9) <= 0.05
        and abs(overall["interval"] - 0.8) <= 0.08
    )
    blocks_ok = all(
        abs(row["q10"] - 0.1) <= 0.12
        and abs(row["q90"] - 0.9) <= 0.12
        and abs(row["interval"] - 0.8) <= 0.18
        for row in blocks
    )
    return {
        "deployable": overall_ok and blocks_ok and overall["crossing"] == 0,
        "reason": "calibrated_and_stable"
        if overall_ok and blocks_ok
        else "coverage_or_time_block_instability",
        "n": len(rows),
        "train_n": len(training),
        "calibration_n": len(calibration),
        "test_n": len(test),
        "test": overall,
        "time_blocks": blocks,
        "model": {
            "kind": "role_horizon_linear_quantile_conformal",
            "features": list(FEATURES),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "offsets": {str(q): offset for q, offset in zip(QUANTILES, offsets)},
            "quantiles": {
                str(q): {
                    "coef": model.coef_.tolist(),
                    "intercept": float(model.intercept_),
                }
                for q, model in zip(QUANTILES, estimators)
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=Path, default=ROOT / "logs" / "games")
    parser.add_argument("--agent", action="append", default=[])
    parser.add_argument("--half-life-hours", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped = defaultdict(list)
    for row in load(args.games, set(args.agent)):
        grouped[row[3]].append(row)
    result = {
        "schema_version": 1,
        "shadow_only": True,
        "no_hidden_opponent_delta": True,
        "half_life_hours": args.half_life_hours,
        "groups": {
            key: _fit_group(rows, args.half_life_hours)
            for key, rows in sorted(grouped.items())
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: {k: value.get(k) for k in ("deployable", "reason", "n", "test")}
                for key, value in result["groups"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
