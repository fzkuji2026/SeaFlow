"""Slow field prior for complete-info opening claim.

Reads recent finished transcripts (hours, not an 8-game Elo window).
Disabled under pytest so tests stay deterministic.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
GAMES_DIR = ROOT / "logs" / "games"
CACHE_PATH = ROOT / "logs" / "field_prior.json"

DEFAULT_OPEN = 0.90
# Do not track the conceding median (clipping to 0.70 opened below leaders'
# close). Leaders open 0.92–1.05.
OPEN_MIN = 0.90
OPEN_MAX = 0.95
WINDOW_S = 6 * 3600
MIN_N = 12
TTL_S = 600

_MEM: dict[str, Any] = {"ts": 0.0, "open": DEFAULT_OPEN}


def field_open_claim() -> float:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return DEFAULT_OPEN
    if os.environ.get("GLEE_FIELD_PRIOR", "1") != "1":
        return DEFAULT_OPEN
    if os.environ.get("GLEE_FREEZE_POLICY") == "1" and _MEM["ts"]:
        return float(_MEM["open"])
    now = time.time()
    if now - float(_MEM.get("ts") or 0) < TTL_S:
        return float(_MEM["open"])
    cached = _read_cache(now)
    if cached is not None:
        _MEM["ts"] = now
        _MEM["open"] = cached
        return cached
    value = _scan(now)
    _MEM["ts"] = now
    _MEM["open"] = value
    _write_cache(value, now)
    return value


def _read_cache(now: float) -> float | None:
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    ts = float(data.get("ts") or 0)
    if now - ts > TTL_S:
        return None
    try:
        value = float(data["open"])
    except (KeyError, TypeError, ValueError):
        return None
    return _clip(value)


def _write_cache(value: float, now: float) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": now, "open": value}
    try:
        CACHE_PATH.write_text(json.dumps(payload) + "\n")
    except OSError:
        pass


def _scan(now: float) -> float:
    if not GAMES_DIR.is_dir():
        return DEFAULT_OPEN
    claims: list[float] = []
    cutoff = now - WINDOW_S
    try:
        paths = list(GAMES_DIR.glob("*.json"))
    except OSError:
        return DEFAULT_OPEN
    for path in paths:
        try:
            if path.stat().st_mtime < cutoff:
                continue
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        claim = _opponent_first_claim(raw)
        if claim is not None:
            claims.append(claim)
    if len(claims) < MIN_N:
        return DEFAULT_OPEN
    claims.sort()
    mid = claims[len(claims) // 2]
    return _clip(mid)


def _opponent_first_claim(raw: dict[str, Any]) -> float | None:
    if raw.get("game_family") != "negotiation":
        return None
    state = raw.get("game_state")
    if not isinstance(state, dict):
        return None
    you = str(raw.get("your_player") or "")
    p1 = state.get("player_1_value")
    p2 = state.get("player_2_value")
    if p1 is None or p2 is None:
        return None
    seller_v = float(p1 if state.get("player_1_role") == "seller" else p2)
    buyer_v = float(p1 if state.get("player_1_role") == "buyer" else p2)
    surplus = buyer_v - seller_v
    if surplus <= 1e-9:
        return None
    role = state.get(f"{you}_role")
    for row in state.get("history") or []:
        if not isinstance(row, dict):
            continue
        offer = row.get("offer") or {}
        if not isinstance(offer, dict):
            continue
        who = offer.get("from_player")
        if who is None or who == you:
            continue
        try:
            price = float(offer["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if role == "buyer":
            return (price - seller_v) / surplus
        if role == "seller":
            return (buyer_v - price) / surplus
        return None
    return None


def _clip(value: float) -> float:
    return min(OPEN_MAX, max(OPEN_MIN, float(value)))
