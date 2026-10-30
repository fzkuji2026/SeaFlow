"""Isolated LLM helpers. Numbers are already decided; these only write text."""

from __future__ import annotations

import os
from typing import Any

from glee_agent.agentic import agentic_function
from glee_agent.brief import brief_game
from glee_agent.llm import cooling_down
from glee_agent.rules import MAX_MESSAGE_LEN


@agentic_function(expose="io", render_range={"callers": 0})
def write_message(brief: str, fallback: str = "", runtime=None) -> str:
    """Write the chat message for a GLEE action that is already decided.

    One short sentence. Do not mention hidden values, system tags, or numbers.
    Do not claim the platform suggested anything. Reply with text only, not JSON.
    """
    text = runtime.exec(
        content=[{"type": "text", "text": brief}],
        timeout_s=6.0,
    )
    msg = str(text or "").strip().strip('"')
    if msg.startswith("{") or len(msg) > MAX_MESSAGE_LEN:
        return (fallback or "")[:MAX_MESSAGE_LEN]
    return (msg or fallback)[:MAX_MESSAGE_LEN]


def fill_message(game: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    fallback = str(action.get("message") or "").strip()
    if os.environ.get("GLEE_LLM_MESSAGES", "0") != "1":
        return action
    # One xAI call budget: decision selection affects payoff; wording does not.
    if os.environ.get("GLEE_LLM_JUDGE", "0") == "1":
        return action
    if cooling_down():
        return action
    try:
        msg = write_message(brief_game(game, action), fallback or "Let's close a fair deal.")
    except Exception:
        return action
    if not msg:
        return action
    out = dict(action)
    out["message"] = msg[:MAX_MESSAGE_LEN]
    return out
