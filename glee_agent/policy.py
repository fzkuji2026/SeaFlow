"""Mutable numeric policy. One JSON file is the live genome."""

from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "logs" / "policy.json"

DEFAULT: dict[str, Any] = {
    "version": 2,
    "bargaining": {
        "offer_share": 0.55,
    },
    "negotiation": {
        "seller_open": 1.50,
        "seller_floor": 1.00,
        "buyer_open": 0.75,
        "buyer_floor": 1.00,
        "walk_progress": 0.65,
    },
    "persuasion": {
        "seller_always_yes": False,
        "untested_ev_ratio": 1.0,
    },
}

BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "bargaining": {
        "offer_share": (0.50, 0.68),
    },
    "negotiation": {
        "seller_open": (1.05, 1.50),
        "seller_floor": (1.00, 1.10),
        "buyer_open": (0.55, 0.90),
        "buyer_floor": (0.95, 1.00),
        "walk_progress": (0.20, 0.80),
    },
    "persuasion": {
        "untested_ev_ratio": (0.50, 1.10),
    },
}

_CURRENT: dict[str, Any] = deepcopy(DEFAULT)
_MEM_TS = time.time()


def current() -> dict[str, Any]:
    return _CURRENT


def refresh() -> dict[str, Any]:
    """Reload from disk if another process wrote a newer policy."""
    if not POLICY_PATH.exists():
        return _CURRENT
    try:
        mtime = POLICY_PATH.stat().st_mtime
    except OSError:
        return _CURRENT
    if mtime <= _MEM_TS:
        return _CURRENT
    return load_policy()


def set_policy(policy: dict[str, Any]) -> None:
    global _CURRENT, _MEM_TS
    _CURRENT = deepcopy(policy)
    _MEM_TS = time.time()


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            merged = deepcopy(DEFAULT)
            for fam, params in data.items():
                if fam in merged and isinstance(params, dict):
                    merged[fam].update(
                        {key: value for key, value in params.items() if key in merged[fam]}
                    )
                elif fam == "version":
                    merged["version"] = params
            set_policy(merged)
            _touch_disk()
            return current()
        except (OSError, json.JSONDecodeError):
            pass
    set_policy(DEFAULT)
    return current()


def _touch_disk() -> None:
    global _MEM_TS
    try:
        _MEM_TS = POLICY_PATH.stat().st_mtime
    except OSError:
        _MEM_TS = time.time()


def save_policy(policy: dict[str, Any] | None = None, path: Path = POLICY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = policy or current()
    path.write_text(json.dumps(payload, indent=2) + "\n")
    set_policy(payload)
    _touch_disk()


def mutate(policy: dict[str, Any], family: str | None = None, rng: random.Random | None = None) -> tuple[dict[str, Any], str, str]:
    """Perturb one numeric knob (or flip seller_always_yes). Returns (new, family, key)."""
    rng = rng or random.Random()
    out = deepcopy(policy)
    family = family or rng.choice(["bargaining", "negotiation", "persuasion"])
    keys = list(BOUNDS[family])
    if family == "persuasion" and rng.random() < 0.2:
        key = "seller_always_yes"
        out[family][key] = not bool(out[family].get(key, True))
        return out, family, key
    key = rng.choice(keys)
    lo, hi = BOUNDS[family][key]
    span = hi - lo
    value = float(out[family][key]) + rng.gauss(0, 0.12 * span)
    out[family][key] = round(min(hi, max(lo, value)), 4)
    if family == "negotiation":
        out[family]["seller_floor"] = max(out[family]["seller_floor"], 1.0)
        out[family]["seller_open"] = max(out[family]["seller_open"], out[family]["seller_floor"])
        out[family]["buyer_floor"] = min(out[family]["buyer_floor"], 1.0)
        out[family]["buyer_open"] = min(out[family]["buyer_open"], out[family]["buyer_floor"])
    return out, family, key


def keep_mutation(
    before_rating: float,
    after_rating: float,
    before_fail: float,
    after_fail: float,
) -> bool:
    if after_rating >= before_rating + 3:
        return True
    # Fail-rate is only a tie-break when rating is flat. Unknown
    # before_fail (0) must not auto-keep a mutation.
    if after_rating >= before_rating - 3 and before_fail > 0 and after_fail + 0.05 <= before_fail:
        return True
    return False
