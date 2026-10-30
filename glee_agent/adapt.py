"""Turn a finished game (result + history) into a live policy nudge."""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import os

from glee_agent.policy import BOUNDS, current, load_policy, save_policy, set_policy

logger = logging.getLogger("glee_agent.adapt")

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "logs" / "adapt_state.json"
GAMES_DIR = ROOT / "logs" / "games"
WATCH_LOG = ROOT / "logs" / "watch.jsonl"

WINDOW = 8
NUDGE_EVERY = 4
SEEN_KEEP = 2500
_FAIL_OUTCOMES = {"walked_away", "no_deal", "timeout", "invalid"}


def is_fail(family: str, result: dict[str, Any] | None) -> bool:
    result = result or {}
    both0 = result.get("player_1_payoff") == 0 and result.get("player_2_payoff") == 0
    if family == "persuasion":
        return result.get("rounds_bought") == 0 or both0
    return result.get("outcome") in _FAIL_OUTCOMES or both0


def my_role(family: str, your_player: str | None, game: dict[str, Any] | None) -> str:
    state = (game or {}).get("game_state") or {}
    me = your_player or state.get("current_player") or ""
    if family == "negotiation":
        role = state.get(f"{me}_role")
        if role in ("seller", "buyer"):
            return role
        if me in ("player_1", "alice"):
            return "seller"
        if me in ("player_2", "bob"):
            return "buyer"
    if family == "bargaining":
        return "alice" if me in ("player_1", "alice") else "bob"
    if family == "persuasion":
        return "seller" if me in ("player_1", "seller") else "buyer"
    return me or "?"


def _clip(family: str, key: str, value: float) -> float:
    lo, hi = BOUNDS[family][key]
    return round(min(hi, max(lo, value)), 4)


def _load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text())
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {"seen": [], "window": {}, "since_nudge": {}}


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state) + "\n")


def _append_watch(row: dict[str, Any]) -> None:
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WATCH_LOG.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def dump_game(game_id: str, payload: dict[str, Any]) -> Path:
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    path = GAMES_DIR / f"{game_id}.json"
    merged: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if isinstance(existing, dict):
                merged.update(existing)
        except (OSError, json.JSONDecodeError):
            pass
    merged.update({key: value for key, value in payload.items() if value is not None})
    path.write_text(json.dumps(merged, default=str) + "\n")
    return path


def pressure_from_points(mean_delta: float | None, fail_rate: float) -> float:
    """0..1 pressure. Official per-game rating_delta wins over deal/fail."""
    if mean_delta is not None:
        if mean_delta >= 0.4:
            return 0.0
        if mean_delta >= -0.2:
            return 0.2
        if mean_delta >= -2.0:
            return 0.45
        return 0.7
    return fail_rate


def stance_from_window(mean_delta: float | None, fail_rate: float) -> str | None:
    """Soften only when games die. Tighten when deals close but lose points."""
    if fail_rate >= 0.35:
        return "soften"
    if mean_delta is not None and mean_delta < -0.3:
        return "tighten"
    return None


def planned_nudge(family: str, role: str, fail_rate: float, stance: str | None = None) -> dict[str, float]:
    if stance is None:
        if fail_rate < 0.35:
            return {}
        stance = "soften"
    step = 0.012 if fail_rate < 0.55 else 0.02
    if stance == "tighten":
        step = 0.016
        if family == "negotiation":
            # Higher walk_progress delays the α≥0.75 jump — hold longer.
            out = {"walk_progress": step}
            if role == "seller":
                out["seller_open"] = step
                out["seller_floor"] = step * 0.2
            else:
                out["buyer_open"] = -step
                out["buyer_floor"] = -step * 0.15
            return out
        if family == "bargaining":
            return {"offer_share": step}
        if family == "persuasion":
            if role == "buyer":
                return {"untested_ev_ratio": step}
            return {}
        return {}
    if family == "negotiation":
        # Soften = concede earlier. The old sign raised walk_progress on
        # no-deals and parked it at the 0.80 cap, so sellers held 1.3× to r9.
        out = {"walk_progress": -step}
        if role == "seller":
            out["seller_open"] = -step
            out["seller_floor"] = -step * 0.25
        else:
            out["buyer_open"] = step
            out["buyer_floor"] = step * 0.15
        return out
    if family == "bargaining":
        return {"offer_share": -step * 0.6}
    if family == "persuasion":
        if role == "buyer":
            return {"untested_ev_ratio": -step}
        return {}
    return {}


def apply_deltas(policy: dict[str, Any], family: str, deltas: dict[str, float]) -> dict[str, Any]:
    out = deepcopy(policy)
    fam = dict(out.get(family) or {})
    for key, delta in deltas.items():
        if key not in BOUNDS.get(family, {}):
            continue
        fam[key] = _clip(family, key, float(fam.get(key, 0)) + delta)
    if family == "negotiation":
        fam["seller_floor"] = max(fam.get("seller_floor", 1.0), 1.0)
        fam["seller_open"] = max(fam.get("seller_open", 1.0), fam["seller_floor"])
        fam["buyer_floor"] = min(fam.get("buyer_floor", 1.0), 1.0)
        fam["buyer_open"] = min(fam.get("buyer_open", 0.8), fam["buyer_floor"])
    out[family] = fam
    return out


def ingest(
    *,
    game_id: str | None,
    family: str,
    your_player: str | None,
    result: dict[str, Any] | None,
    game: dict[str, Any] | None = None,
    persist: bool = True,
    rating_delta: float | None = None,
) -> dict[str, Any]:
    family = (family or "").lower()
    if family not in ("bargaining", "negotiation", "persuasion"):
        return {"skipped": "family"}
    state = _load_state()
    seen: list[str] = list(state.get("seen") or [])
    already = bool(game_id and game_id in seen)
    if game_id and not already:
        seen.append(game_id)
        state["seen"] = seen[-SEEN_KEEP:]

    failed = is_fail(family, result)
    role = my_role(family, your_player, game)
    points: dict[str, list] = dict(state.get("points") or {})
    if rating_delta is not None:
        bucket_p = [row for row in (points.get(family) or []) if row.get("game_id") != game_id]
        bucket_p.append({"game_id": game_id, "delta": float(rating_delta), "role": role})
        points[family] = bucket_p[-WINDOW:]
        state["points"] = points
    scored = points.get(family) or []
    mean_delta = (sum(float(row["delta"]) for row in scored) / len(scored)) if scored else None

    windows: dict[str, list] = dict(state.get("window") or {})
    if not already:
        bucket = list(windows.get(family) or [])
        bucket.append({"fail": failed, "role": role, "outcome": (result or {}).get("outcome"), "rating_delta": rating_delta})
        windows[family] = bucket[-WINDOW:]
        state["window"] = windows
        since = dict(state.get("since_nudge") or {})
        since[family] = int(since.get(family, 0)) + 1
    else:
        since = dict(state.get("since_nudge") or {})

    n = len(windows.get(family) or [])
    fails = sum(1 for item in (windows.get(family) or []) if item.get("fail"))
    rate = fails / n if n else 0.0
    stance = stance_from_window(mean_delta, rate)
    deltas: dict[str, float] = {}
    applied = False
    frozen = os.environ.get("GLEE_FREEZE_POLICY") == "1"
    if (
        not frozen
        and stance
        and (n >= 5 or len(scored) >= 5)
        and int(since.get(family, 0)) >= NUDGE_EVERY
    ):
        deltas = planned_nudge(family, role, rate, stance=stance)
        if deltas:
            load_policy()
            child = apply_deltas(current(), family, deltas)
            set_policy(child)
            if persist:
                save_policy(child)
            since[family] = 0
            applied = True
            logger.info(
                "adapt %s role=%s stance=%s mean_delta=%s fail_rate=%.2f nudge=%s",
                family, role, stance, None if mean_delta is None else round(mean_delta, 3), rate, deltas,
            )
    state["since_nudge"] = since
    if persist:
        _save_state(state)
        _append_watch(
            {
                "ts": time.time(),
                "game_id": game_id,
                "family": family,
                "role": role,
                "fail": failed,
                "rating_delta": rating_delta,
                "mean_delta": None if mean_delta is None else round(mean_delta, 3),
                "window_fail": round(rate, 3),
                "n": n,
                "nudged": applied,
                "deltas": deltas,
            }
        )
    return {
        "family": family,
        "role": role,
        "fail": failed,
        "rating_delta": rating_delta,
        "mean_delta": mean_delta,
        "window_fail": rate,
        "n": n,
        "nudged": applied,
        "deltas": deltas,
        "already": already,
    }


def ingest_full(full: dict[str, Any], persist: bool = True) -> dict[str, Any]:
    game_id = full.get("game_id") or full.get("id")
    if persist and game_id:
        dump_game(str(game_id), full)
    delta = None
    for key in ("rating_delta", "ratingDelta"):
        if full.get(key) is not None:
            try:
                delta = float(full[key])
            except (TypeError, ValueError):
                delta = None
            break
    return ingest(
        game_id=str(game_id) if game_id else None,
        family=full.get("game_family") or "",
        your_player=full.get("your_player"),
        result=full.get("result"),
        game=full,
        persist=persist,
        rating_delta=delta,
    )
