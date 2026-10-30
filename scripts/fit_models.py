#!/usr/bin/env python3
"""Fit compact decision curves from complete local replays."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT)]

AGENT = "210086e2-f599-485f-915c-6b12df76cc6a"


def _fit(
    kind: str,
    features: list[str],
    rows: list[tuple[list[float], float, float]],
    *,
    regularization: float,
    clip: tuple[float, float] | None,
    guard: float,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray([row[1] for row in rows], dtype=float)
    weights = np.asarray([row[2] for row in rows], dtype=float)
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    if kind == "logistic":
        estimator = LogisticRegression(C=regularization, max_iter=2000)
        estimator.fit(scaled, y.astype(int), sample_weight=weights)
    else:
        estimator = Ridge(alpha=regularization)
        estimator.fit(scaled, y, sample_weight=weights)
    out = {
        "kind": kind,
        "features": features,
        "n": len(rows),
        "effective_games": round(float(weights.sum()), 3),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": estimator.coef_.ravel().tolist(),
        "intercept": float(estimator.intercept_.ravel()[0] if hasattr(estimator.intercept_, "ravel") else estimator.intercept_),
        "guard": guard,
        "support": [
            [float(x[:, index].min()), float(x[:, index].max())]
            for index in range(x.shape[1])
        ],
    }
    if clip is not None:
        out["clip"] = list(clip)
    return out


def _weight(
    rows: list[tuple[tuple[str, ...], list[float], float]],
    timestamp: float,
    game_id: str,
) -> list[tuple[tuple[str, ...], list[float], float, float, float, str]]:
    weight = 1.0 / max(1, len(rows))
    return [
        (parents, features, target, weight, timestamp, game_id)
        for parents, features, target in rows
    ]


def _plain(rows: list[tuple]) -> list[tuple[list[float], float, float, float]]:
    return [(row[1], row[2], row[3], row[4]) for row in rows]


def _time_decay(rows: list[tuple], half_life_seconds: float) -> list[tuple]:
    """Continuously favor newer primary examples without discarding old ones."""
    if not rows or half_life_seconds <= 0:
        return rows
    latest = max(float(row[3]) for row in rows)
    out = []
    for row in rows:
        age = max(0.0, latest - float(row[3]))
        weight = float(row[2]) * (0.5 ** (age / half_life_seconds))
        out.append((*row[:2], weight, *row[3:]))
    return out


def _player_offer(row: dict, you: str) -> bool:
    from glee_agent.rules import as_player_id

    offer = row.get("offer") or {}
    sender = row.get("proposer") or offer.get("proposer") or offer.get("from_player")
    return as_player_id(str(sender or "")) == as_player_id(you)


def _bargaining(game: dict) -> tuple[list, list, list]:
    from glee_agent.diagnose import replay_to_game
    from glee_agent.rules import offer_gain, parse
    from glee_agent.workflows.bargaining._helpers import (
        fitted_offer_features,
        fitted_response_features,
    )
    from glee_agent.segments import segment
    from glee_agent.workflows.bargaining import classify
    from glee_agent.workflows.bargaining._opponent import style_of
    from glee_agent.workflows.bargaining._plan import compose

    state = game.get("state") or {}
    history = [row for row in state.get("history") or [] if isinstance(row, dict)]
    you = str(game.get("your_player") or "player_1")
    pot = float((game.get("config") or {}).get("money_to_divide") or 0.0)
    final = float((game.get("result") or {}).get(f"{you}_payoff") or 0.0) / pot if pot else 0.0
    timestamp = datetime.fromisoformat(str(game.get("completed_at"))).timestamp()
    offers: list[tuple[tuple[str, ...], list[float], float]] = []
    offer_continues: list[tuple[tuple[str, ...], list[float], float]] = []
    response_continues: list[tuple[tuple[str, ...], list[float], float]] = []
    for row in history:
        round_n = int(row.get("round") or 1)
        if _player_offer(row, you):
            view = parse(replay_to_game(game, round_n=round_n, action_type="offer"))
            share = offer_gain(row.get("offer"), you)
            if share is None or pot <= 0:
                continue
            features = fitted_offer_features(view, share / pot)
            if features is None:
                continue
            procedure = compose(view, classify(view), style_of(view)).procedure
            parents = segment(view, procedure).parents
            accepted = "accept" in str(row.get("decision") or "").lower()
            offers.append((parents, features, float(accepted)))
            if not accepted:
                offer_continues.append((parents, features, final))
        elif "reject" in str(row.get("decision") or "").lower():
            view = parse(replay_to_game(game, round_n=round_n, action_type="decision"))
            features = fitted_response_features(view)
            if features is not None:
                procedure = compose(view, classify(view), style_of(view)).procedure
                parents = segment(view, procedure).parents
                response_continues.append((parents, features, final))
    return (
        _weight(offers, timestamp, str(game.get("game_id") or "")),
        _weight(offer_continues, timestamp, str(game.get("game_id") or "")),
        _weight(response_continues, timestamp, str(game.get("game_id") or "")),
    )


def _negotiation(game: dict) -> tuple[str, list, list]:
    from glee_agent.diagnose import replay_to_game
    from glee_agent.rules import parse, role_of, value_of
    from glee_agent.workflows.negotiation._tactics import fitted_offer_features
    from glee_agent.workflows.negotiation._tactics import pick_procedure
    from glee_agent.workflows.negotiation import classify
    from glee_agent.workflows.negotiation.classify import own_mult
    from glee_agent.segments import segment

    state = game.get("state") or {}
    history = [row for row in state.get("history") or [] if isinstance(row, dict)]
    you = str(game.get("your_player") or "player_1")
    role = "seller" if you == "player_1" else "buyer"
    config = game.get("config") or {}
    base = float(config.get("product_price_order") or 0.0)
    final = float((game.get("result") or {}).get(f"{you}_payoff") or 0.0) / base if base else 0.0
    timestamp = datetime.fromisoformat(str(game.get("completed_at"))).timestamp()
    offers: list[tuple[tuple[str, ...], list[float], float]] = []
    continues: list[tuple[tuple[str, ...], list[float], float]] = []
    for row in history:
        if not _player_offer(row, you):
            continue
        offer = row.get("offer") or {}
        price = offer.get("price", offer.get("product_price"))
        if price is None:
            continue
        round_n = int(row.get("round") or 1)
        view = parse(replay_to_game(game, round_n=round_n, action_type="offer"))
        mine = value_of(view)
        actual_role = role_of(view)
        if mine is None or actual_role not in ("seller", "buyer") or not own_mult(view):
            continue
        features = fitted_offer_features(view, actual_role, mine, float(price))
        if features is None:
            continue
        procedure = pick_procedure(view, classify(view), actual_role, mine)
        parents = segment(view, procedure).parents
        accepted = "accept" in str(row.get("decision") or "").lower()
        offers.append((parents, features, float(accepted)))
        if not accepted:
            continues.append((parents, features, final))
    game_id = str(game.get("game_id") or "")
    return role, _weight(offers, timestamp, game_id), _weight(
        continues, timestamp, game_id
    )


def _prior_key(config: dict) -> str:
    values = (
        float(config.get("p") or 0.5),
        float(config.get("v") or 0.0),
        float(config.get("c") or 0.0),
        bool(config.get("is_seller_know_cv")),
        str(config.get("seller_message_type") or "text").lower(),
        bool(config.get("is_myopic", False)),
    )
    return "|".join(str(value) for value in values)


def _zopa_sample(game: dict) -> tuple[str, list[float], float, float] | None:
    from glee_agent.rules import as_player_id

    config = game.get("config") or {}
    if config.get("complete_information") is not False:
        return None
    you = as_player_id(str(game.get("your_player") or "player_1"))
    role = "seller" if you == "player_1" else "buyer"
    own = config.get("seller_value" if role == "seller" else "buyer_value")
    other = config.get("buyer_value" if role == "seller" else "seller_value")
    base = config.get("product_price_order")
    horizon = config.get("max_rounds")
    if not all(isinstance(value, (int, float)) for value in (own, other, base, horizon)):
        return None
    seen: list[float] = []
    for row in ((game.get("state") or {}).get("history") or []):
        if not isinstance(row, dict):
            continue
        offer = row.get("offer") or {}
        price = offer.get("price", offer.get("product_price"))
        sender = as_player_id(str(offer.get("from_player") or ""))
        if isinstance(price, (int, float)) and sender and sender != you:
            seen.append(float(price) / float(base))
    if not seen:
        return None
    first = seen[0]
    second = seen[1] if len(seen) >= 2 else first
    exists = float(other) > float(own) if role == "seller" else float(other) < float(own)
    horizon_key = "open" if int(horizon) == 99 else str(int(horizon))
    name = f"negotiation_zopa_{role}_{horizon_key}"
    features = [
        float(own),
        float(bool(config.get("messages_allowed"))),
        first,
        second,
        second - first,
        min(len(seen), 6) / 6.0,
    ]
    timestamp = datetime.fromisoformat(str(game.get("completed_at"))).timestamp()
    return name, features, float(exists), timestamp


def _serialized_predict(row: dict[str, Any], features: list[float]) -> float:
    score = float(row["intercept"])
    for value, center, scale, coef in zip(
        features, row["mean"], row["scale"], row["coef"]
    ):
        score += ((float(value) - float(center)) / (float(scale) or 1.0)) * float(coef)
    if row["kind"] == "logistic":
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, score))))
    lo, hi = row.get("clip") or [None, None]
    if lo is not None:
        score = max(float(lo), score)
    if hi is not None:
        score = min(float(hi), score)
    return score


def _fit_segment(
    kind: str,
    features: list[str],
    rows: list[tuple[list[float], float, float, float]],
    *,
    regularization: float,
    clip: tuple[float, float] | None,
    guard: float,
    holdout_ts: float,
) -> dict[str, Any] | None:
    rows = sorted(rows, key=lambda row: row[3])
    effective = sum(row[2] for row in rows)
    if len(rows) < max(16, len(features) * 2) or effective < 8.0:
        return None
    train = [row for row in rows if row[3] < holdout_ts]
    holdout = [row for row in rows if row[3] >= holdout_ts]
    if not train or not holdout:
        return None
    if sum(row[2] for row in train) < 4.0 or sum(row[2] for row in holdout) < 2.0:
        return None
    if kind == "logistic":
        if len({row[1] for row in train}) < 2 or len({row[1] for row in holdout}) < 2:
            return None
    probe = _fit(
        kind,
        features,
        train,
        regularization=regularization,
        clip=clip,
        guard=guard,
    )
    total_weight = sum(row[2] for row in holdout)
    baseline = sum(row[1] * row[2] for row in train) / sum(row[2] for row in train)
    if kind == "logistic":
        model_loss = sum(
            row[2] * (_serialized_predict(probe, row[0]) - row[1]) ** 2
            for row in holdout
        ) / total_weight
        base_loss = sum(row[2] * (baseline - row[1]) ** 2 for row in holdout) / total_weight
    else:
        model_loss = sum(
            row[2] * abs(_serialized_predict(probe, row[0]) - row[1])
            for row in holdout
        ) / total_weight
        base_loss = sum(row[2] * abs(baseline - row[1]) for row in holdout) / total_weight
    if model_loss >= base_loss * 0.98:
        return None
    final = _fit(
        kind,
        features,
        rows,
        regularization=regularization,
        clip=clip,
        guard=guard,
    )
    final["holdout_loss"] = round(model_loss, 6)
    final["holdout_baseline_loss"] = round(base_loss, 6)
    # Claim/share is feature zero. Higher own claims cannot causally make an
    # unchanged opponent more willing to accept; a positive slope is policy
    # selection bias and is diagnostic-only.
    if kind == "logistic" and float(final["coef"][0]) >= 0.0:
        return None
    return final


def _fit_accept_threshold(
    rows: list[tuple[list[float], float, float, float, str]],
    holdout_ts: float,
) -> dict[str, Any] | None:
    """Fit a segment-local margin using the first triggered reject per game."""
    rows = sorted(rows, key=lambda row: row[3])
    games = {row[4] for row in rows if row[4]}
    if len(rows) < 12 or len(games) < 8:
        return None
    train = [row for row in rows if row[3] < holdout_ts]
    holdout = [row for row in rows if row[3] >= holdout_ts]
    if not train or not holdout:
        return None
    train_games = {row[4] for row in train if row[4]}
    holdout_games = {row[4] for row in holdout if row[4]}
    if len(train_games) < 5 or len(holdout_games) < 2:
        return None

    margins = sorted({round(row[0][0] - row[0][1], 6) for row in train})
    candidates: list[tuple[float, float, int]] = []
    for threshold in margins:
        triggered = _first_triggered(train, threshold)
        if len(triggered) < 3:
            continue
        gain = sum((row[0][0] - row[1]) * row[2] for row in triggered)
        if gain > 0:
            candidates.append((gain, -threshold, len(triggered)))
    for best in sorted(candidates, reverse=True):
        threshold = -best[1]
        triggered = _first_triggered(holdout, threshold)
        if not triggered:
            continue
        holdout_gain = sum(row[0][0] - row[1] for row in triggered)
        improved = sum(row[0][0] > row[1] + 1e-9 for row in triggered)
        regressed = sum(row[0][0] < row[1] - 1e-9 for row in triggered)
        if holdout_gain <= 0 or improved < regressed:
            continue
        return {
            "kind": "accept_margin",
            "threshold": threshold,
            "n": len(rows),
            "effective_games": len(games),
            "train_gain": round(best[0], 6),
            "train_triggered": best[2],
            "holdout_gain": round(holdout_gain, 6),
            "holdout_triggered": len(triggered),
            "holdout_improved": improved,
            "holdout_regressed": regressed,
        }
    return None


def _first_triggered(
    rows: list[tuple[list[float], float, float, float, str]],
    threshold: float,
) -> list[tuple[list[float], float, float, float, str]]:
    selected: dict[str, tuple[list[float], float, float, float, str]] = {}
    for row in sorted(rows, key=lambda item: item[3]):
        game_id = row[4]
        if not game_id or game_id in selected:
            continue
        if row[0][0] - row[0][1] >= threshold - 1e-9:
            selected[game_id] = row
    return list(selected.values())


def _external_accept_validation(
    rows: list[tuple[list[float], float, float, float, str]],
    threshold: float,
) -> dict[str, Any]:
    triggered = _first_triggered(rows, threshold)
    gains = [row[0][0] - row[1] for row in triggered]
    out = {
        "external_triggered": len(triggered),
        "external_gain": round(sum(gains), 6),
        "external_improved": sum(gain > 1e-9 for gain in gains),
        "external_tied": sum(abs(gain) <= 1e-9 for gain in gains),
        "external_regressed": sum(gain < -1e-9 for gain in gains),
    }
    return out


def _passes_external_validation(row: dict[str, Any]) -> bool:
    overall = bool(
        int(row.get("external_triggered") or 0) >= 3
        and int(row.get("external_improved") or 0) >= 3
        and int(row.get("external_regressed") or 0) == 0
        and float(row.get("external_gain") or 0.0) > 0.0
    )
    return overall


def _fit_zopa(
    rows: list[tuple[list[float], float, float, float]],
    holdout_ts: float,
) -> dict[str, Any] | None:
    from sklearn.metrics import roc_auc_score

    rows = sorted(rows, key=lambda row: row[3])
    if len(rows) < 30:
        return None
    train = [row for row in rows if row[3] < holdout_ts]
    holdout = [row for row in rows if row[3] >= holdout_ts]
    if not train or not holdout:
        return None
    if len({row[1] for row in train}) < 2 or len({row[1] for row in holdout}) < 2:
        return None
    probe = _fit(
        "logistic",
        ["own_mult", "messages", "first_price", "second_price", "price_change", "samples"],
        train,
        regularization=0.20,
        clip=None,
        guard=0.0,
    )
    actual = [int(row[1]) for row in holdout]
    predicted = [_serialized_predict(probe, row[0]) for row in holdout]
    auc = roc_auc_score(actual, predicted)
    baseline = sum(row[1] for row in train) / len(train)
    model_brier = sum((p - y) ** 2 for p, y in zip(predicted, actual)) / len(actual)
    base_brier = sum((baseline - y) ** 2 for y in actual) / len(actual)
    if auc < 0.75 or model_brier >= base_brier:
        return None
    final = _fit(
        "logistic",
        ["own_mult", "messages", "first_price", "second_price", "price_change", "samples"],
        rows,
        regularization=0.20,
        clip=None,
        guard=0.0,
    )
    final["holdout_auc"] = round(float(auc), 6)
    final["holdout_brier"] = round(model_brier, 6)
    final["holdout_baseline_brier"] = round(base_brier, 6)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default=AGENT)
    parser.add_argument("--external-agent")
    parser.add_argument("--external-after-ts", type=float, default=0.0)
    parser.add_argument("--recency-half-life-hours", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=ROOT / "glee_agent" / "fitted_models.json")
    parser.add_argument(
        "--holdout-ts",
        type=float,
        required=True,
        help="Unix timestamp separating training history from frozen chronological holdout",
    )
    args = parser.parse_args()

    from glee_agent.workflows.persuasion._beliefs import seller_said_buy

    model_rows: dict[str, list[tuple[list[float], float, float, float]]] = defaultdict(list)
    segmented_rows: dict[
        str, dict[str, list[tuple[list[float], float, float, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    response_policy_rows: dict[
        str, list[tuple[list[float], float, float, float, str]]
    ] = defaultdict(list)
    external_response_policy_rows: dict[
        str, list[tuple[list[float], float, float, float, str]]
    ] = defaultdict(list)
    zopa_rows: dict[str, list[tuple[list[float], float, float, float]]] = defaultdict(list)
    priors: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    games = 0
    for path in (ROOT / "logs" / "games").glob("*.json"):
        try:
            game = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        player_id = game.get("your_player_id")
        primary = player_id == args.agent
        external = bool(args.external_agent) and player_id == args.external_agent
        if not primary and not external:
            continue
        family = game.get("game_family")
        if family == "bargaining":
            accept, offer_continue, response_continue = _bargaining(game)
            if primary:
                for name, samples in (
                    ("bargaining_offer_accept", accept),
                    ("bargaining_offer_continue", offer_continue),
                    ("bargaining_response_continue", response_continue),
                ):
                    model_rows[name].extend(_plain(samples))
                    for parents, features, target, weight, timestamp, game_id in samples:
                        for parent in parents:
                            segmented_rows[name][parent].append(
                                (features, target, weight, timestamp)
                            )
                            if name == "bargaining_response_continue":
                                response_policy_rows[parent].append(
                                    (features, target, weight, timestamp, game_id)
                                )
            elif external:
                for parents, features, target, weight, timestamp, game_id in response_continue:
                    if timestamp < args.external_after_ts:
                        continue
                    for parent in parents:
                        external_response_policy_rows[parent].append(
                            (features, target, weight, timestamp, game_id)
                        )
        elif primary and family == "negotiation":
            role, accept, continued = _negotiation(game)
            for name, samples in (
                (f"negotiation_{role}_offer_accept", accept),
                (f"negotiation_{role}_offer_continue", continued),
            ):
                model_rows[name].extend(_plain(samples))
                for parents, features, target, weight, timestamp, _game_id in samples:
                    for parent in parents:
                        segmented_rows[name][parent].append(
                            (features, target, weight, timestamp)
                        )
            zopa_sample = _zopa_sample(game)
            if zopa_sample is not None:
                name, features, target, timestamp = zopa_sample
                zopa_rows[name].append((features, target, 1.0, timestamp))
        elif primary and family == "persuasion" and game.get("your_player") == "player_1":
            key = _prior_key(game.get("config") or {})
            for row in ((game.get("state") or {}).get("history") or []):
                if not isinstance(row, dict):
                    continue
                recommendation = seller_said_buy(row.get("seller_message"))
                if recommendation is None:
                    continue
                signal = "yes" if recommendation else "no"
                bought = bool(
                    row.get("bought")
                    or str(row.get("buyer_decision") or "").lower() in ("yes", "buy")
                )
                priors[(key, signal)][0] += bought
                priors[(key, signal)][1] += 1
                priors[("*", signal)][0] += bought
                priors[("*", signal)][1] += 1
        if primary:
            games += 1

    half_life = max(0.0, args.recency_half_life_hours) * 3600.0
    for name, rows in model_rows.items():
        model_rows[name] = _time_decay(rows, half_life)
    for tables in segmented_rows.values():
        for key, rows in tables.items():
            tables[key] = _time_decay(rows, half_life)
    for key, rows in response_policy_rows.items():
        response_policy_rows[key] = _time_decay(rows, half_life)
    for name, rows in zopa_rows.items():
        zopa_rows[name] = _time_decay(rows, half_life)

    specs = {
        "bargaining_offer_accept": (
            "logistic",
            ["share", "progress", "own_delta", "visible_opp_delta", "own_rejections", "last_opp_share", "role_alice"],
            0.10,
            None,
            0.015,
        ),
        "bargaining_offer_continue": (
            "linear",
            ["share", "progress", "own_delta", "visible_opp_delta", "own_rejections", "last_opp_share", "role_alice"],
            10.0,
            (0.0, 1.0),
            0.015,
        ),
        "bargaining_response_continue": (
            "linear",
            ["accept_now", "old_continue", "log_ratio", "progress", "own_delta", "visible_opp_delta", "own_rejections", "share", "previous_share", "concession", "repeats", "role_alice"],
            10.0,
            (0.0, 1.0),
            0.025,
        ),
        "negotiation_seller_offer_accept": (
            "logistic",
            ["claim", "progress", "own_mult", "visible_opp_mult", "own_rejections", "last_opp_profit", "opp_concession", "repeats", "known_horizon"],
            0.20,
            None,
            0.015,
        ),
        "negotiation_seller_offer_continue": (
            "linear",
            ["claim", "progress", "own_mult", "visible_opp_mult", "own_rejections", "last_opp_profit", "opp_concession", "repeats", "known_horizon"],
            10.0,
            (0.0, 0.70),
            0.015,
        ),
        "negotiation_buyer_offer_accept": (
            "logistic",
            ["claim", "progress", "own_mult", "visible_opp_mult", "own_rejections", "last_opp_profit", "opp_concession", "repeats"],
            0.20,
            None,
            0.015,
        ),
        "negotiation_buyer_offer_continue": (
            "linear",
            ["claim", "progress", "own_mult", "visible_opp_mult", "own_rejections", "last_opp_profit", "opp_concession", "repeats"],
            10.0,
            (0.0, 0.70),
            0.015,
        ),
    }
    fitted = {}
    for name, (kind, features, regularization, clip, guard) in specs.items():
        rows = model_rows[name]
        if len(rows) < len(features) * 3 or (kind == "logistic" and len({row[1] for row in rows}) < 2):
            raise SystemExit(f"insufficient rows for {name}: {len(rows)}")
        fitted[name] = _fit(
            kind,
            features,
            rows,
            regularization=regularization,
            clip=clip,
            guard=guard,
        )
        print(name, fitted[name]["n"], fitted[name]["effective_games"])
    for name, rows in sorted(zopa_rows.items()):
        row = _fit_zopa(rows, args.holdout_ts)
        if row is not None:
            fitted[name] = row
            print(name, row["n"], "holdout_auc", row["holdout_auc"])

    segmented = {}
    for name, (kind, features, regularization, clip, guard) in specs.items():
        table = {}
        for key, rows in segmented_rows[name].items():
            row = _fit_segment(
                kind,
                features,
                rows,
                regularization=regularization,
                clip=clip,
                guard=guard,
                holdout_ts=args.holdout_ts,
            )
            if row is not None:
                table[key] = row
        segmented[name] = table
        print(name, "stable segments", len(table))

    response_policies = {}
    for key, rows in response_policy_rows.items():
        row = _fit_accept_threshold(rows, args.holdout_ts)
        if row is None:
            continue
        if args.external_agent:
            row.update(
                _external_accept_validation(
                    external_response_policy_rows[key], float(row["threshold"])
                )
            )
            if not _passes_external_validation(row):
                continue
        response_policies[key] = row
    print("bargaining_response_accept stable policies", len(response_policies))

    prior_out = {}
    for (key, signal), (bought, total) in priors.items():
        prior_out[f"{key}|{signal}"] = {
            "mean": (bought + 0.5) / (total + 1.0),
            "strength": 1.0,
            "n": total,
        }
    output = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent_id": args.agent,
        "external_agent_id": args.external_agent,
        "external_after_ts": args.external_after_ts if args.external_agent else None,
        "holdout_ts": args.holdout_ts,
        "recency_half_life_hours": args.recency_half_life_hours,
        "games": games,
        "models": fitted,
        "segmented_models": segmented,
        "segmented_policies": {
            "bargaining_response_accept": response_policies,
        },
        "priors": {"persuasion_buy": prior_out},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
