#!/usr/bin/env python3
"""Audit a candidate Bargaining policy against factual historical continuations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT)]

from glee_agent.diagnose import replay_to_game  # noqa: E402
from glee_agent.decision_trace import current_node  # noqa: E402
from glee_agent.rules import as_player_id, money, my_delta, offer_gain, parse  # noqa: E402
from glee_agent.strategy import strategy  # noqa: E402


def _timestamp(game: dict[str, Any]) -> float | None:
    try:
        return datetime.fromisoformat(str(game["completed_at"])).timestamp()
    except (KeyError, TypeError, ValueError):
        return None


def _our_response(game: dict[str, Any], row: dict[str, Any]) -> bool:
    you = as_player_id(str(game.get("your_player") or "player_1"))
    offer = row.get("offer") or row
    proposer = as_player_id(
        str(row.get("proposer") or offer.get("proposer") or offer.get("from_player") or "")
    )
    return bool(proposer and proposer != you)


def audit_game(game: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidate changes with known or unknown counterfactual value."""
    if game.get("game_family") != "bargaining":
        return []
    state = game.get("state") or game.get("game_state") or {}
    result = game.get("result") or {}
    you = as_player_id(str(game.get("your_player") or "player_1"))
    rows = []
    for row in state.get("history") or []:
        if not isinstance(row, dict) or not _our_response(game, row):
            continue
        actual = str(row.get("decision") or "").lower()
        if "accept" not in actual and "reject" not in actual:
            continue
        round_n = int(row.get("round") or 1)
        try:
            decision_game = replay_to_game(game, round_n=round_n, action_type="decision")
            candidate = str(strategy(decision_game).get("decision") or "").lower()
            node = current_node() or "unknown"
            view = parse(decision_game)
        except (KeyError, TypeError, ValueError):
            continue
        actual = "accept" if "accept" in actual else "reject"
        if candidate == actual:
            continue
        item: dict[str, Any] = {
            "game_id": game.get("game_id"),
            "round": round_n,
            "actual": actual,
            "candidate": candidate,
            "decision_node": node,
            "known": False,
            "gain": None,
        }
        # If history rejected and the candidate accepts, both outcomes are known:
        # immediate discounted offer value versus the factual final continuation.
        if actual == "reject" and candidate == "accept":
            pot = money(view)
            gain = offer_gain(row.get("offer") or row, you)
            delta = my_delta(view)
            final = result.get(f"{you}_payoff")
            if pot > 0 and gain is not None and delta is not None and isinstance(final, (int, float)):
                accept_value = gain * delta ** max(0, round_n - 1) / pot
                factual_value = float(final) / pot
                item.update(
                    known=True,
                    gain=accept_value - factual_value,
                    accept_value=accept_value,
                    factual_value=factual_value,
                )
        rows.append(item)
        # After the first changed decision, the factual later trajectory is no
        # longer a valid counterfactual for this candidate policy.
        break
    return rows


def summarize(
    games: list[dict[str, Any]], *, half_life_hours: float = 3.0
) -> dict[str, Any]:
    timed = [(game, _timestamp(game)) for game in games]
    latest = max((ts for _, ts in timed if ts is not None), default=0.0)
    known: list[tuple[float, float]] = []
    by_node: dict[str, list[float]] = {}
    unknown = 0
    changed = 0
    for game, ts in timed:
        for row in audit_game(game):
            changed += 1
            if row["known"] and isinstance(row["gain"], (int, float)):
                age = max(0.0, latest - float(ts or latest))
                weight = 0.5 ** (age / max(1.0, half_life_hours * 3600.0))
                known.append((float(row["gain"]), weight))
                by_node.setdefault(str(row["decision_node"]), []).append(float(row["gain"]))
            else:
                unknown += 1
    weighted_gain = sum(gain * weight for gain, weight in known)
    weight_sum = sum(weight for _, weight in known)
    losses = sorted(gain for gain, _ in known if gain < 0.0)
    tail_n = max(1, math.ceil(len(known) * 0.10)) if known else 0
    return {
        "games": len(games),
        "changed_responses": changed,
        "known_counterfactuals": len(known),
        "unknown_counterfactuals": unknown,
        "known_improved": sum(gain > 1e-9 for gain, _ in known),
        "known_regressed": sum(gain < -1e-9 for gain, _ in known),
        "weighted_gain": weighted_gain,
        "weighted_mean_gain": weighted_gain / weight_sum if weight_sum else 0.0,
        "worst_10pct_mean_gain": (
            sum(gain for gain, _ in sorted(known)[:tail_n]) / tail_n if tail_n else 0.0
        ),
        "worst_known_gain": losses[0] if losses else 0.0,
        "known_by_node": {
            node: {
                "n": len(values),
                "improved": sum(value > 1e-9 for value in values),
                "regressed": sum(value < -1e-9 for value in values),
                "gain": sum(values),
            }
            for node, values in sorted(by_node.items(), key=lambda item: sum(item[1]))
        },
        "pass": bool(
            known
            and weighted_gain > 0.0
            and sum(gain > 1e-9 for gain, _ in known)
            > sum(gain < -1e-9 for gain, _ in known)
            and unknown == 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=Path, default=ROOT / "logs" / "games")
    parser.add_argument("--agent")
    parser.add_argument("--half-life-hours", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    games = []
    for path in args.games.glob("*.json"):
        try:
            game = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if args.agent and game.get("your_player_id") != args.agent:
            continue
        if game.get("game_family") == "bargaining":
            games.append(game)
    result = summarize(games, half_life_hours=args.half_life_hours)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
