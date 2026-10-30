"""Known last round: may recommend low."""

from __future__ import annotations

from typing import Any

from glee_agent.policy import current
from glee_agent.rules import GameView
from glee_agent.workflows.persuasion._beliefs import buyer_prefers_skip_signal


def handle(view: GameView) -> dict[str, Any]:
    if view.action_type == "seller_message":
        return _message(view)
    return _recommend(view)


def _should(view: GameView) -> bool:
    if buyer_prefers_skip_signal(view.state):
        return False
    p = current()["persuasion"]
    quality = str(view.state.get("current_quality") or view.state.get("quality") or "").lower()
    return quality != "low" or bool(p.get("seller_always_yes")) or True


def _message(view: GameView) -> dict[str, Any]:
    if _should(view):
        return {"message": "This unit is high quality. I recommend buying it."}
    return {"message": "Honestly, I would skip this one — it is not my best unit."}


def _recommend(view: GameView) -> dict[str, Any]:
    return {"decision": "yes" if _should(view) else "no"}
