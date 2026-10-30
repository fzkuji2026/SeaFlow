"""Per-family notes the LLM player reads and rewrites after batches."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "logs" / "playbook.json"

DEFAULT: dict[str, str] = {
    "bargaining": (
        "Maximize your own payoff. Fair 50/50 closes most deals; ask a bit more "
        "only if inflation is slow. Reject insulting splits early, but accept any "
        "positive share on the last round — delay destroys the pie. Never walk away "
        "from a positive offer."
    ),
    "negotiation": (
        "Trade only if buyer value > seller value. Open away from your value, then "
        "concede. Do not walk away from a profitable price; last round take any "
        "positive surplus. About a third of games currently die as walked_away — "
        "that is usually worse than a thin deal."
    ),
    "persuasion": (
        "Seller: reputation is the product. Always recommending gets you ignored "
        "(many 20-round games with 0 buys). Recommend high, warn off some lows. "
        "Buyer: prior EV vs price is the baseline; follow a buy signal once, then "
        "trust honest sellers and ignore caught liars. Passing every round scores 0."
    ),
}

_lock = threading.Lock()
_CURRENT = deepcopy(DEFAULT)


def current() -> dict[str, str]:
    return _CURRENT


def set_playbook(notes: dict[str, str]) -> None:
    global _CURRENT
    merged = deepcopy(DEFAULT)
    for family, text in notes.items():
        if family in merged and isinstance(text, str) and text.strip():
            merged[family] = text.strip()[:1200]
    _CURRENT = merged


def load(path: Path = PATH) -> dict[str, str]:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                set_playbook({k: str(v) for k, v in data.items()})
                return current()
        except (OSError, json.JSONDecodeError):
            pass
    set_playbook(DEFAULT)
    return current()


def save(notes: dict[str, str] | None = None, path: Path = PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        payload: dict[str, Any] = notes or current()
        path.write_text(json.dumps(payload, indent=2) + "\n")
        set_playbook({k: str(v) for k, v in payload.items()})


def note_for(family: str) -> str:
    return current().get(family) or DEFAULT.get(family, "")
