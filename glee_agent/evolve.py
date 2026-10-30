"""After a batch of games, the same LLM rewrites its own playbook."""

from __future__ import annotations

import json
import logging
from typing import Any

from glee_agent import playbook
from glee_agent.llm import LLMError, complete, parse_json_object

logger = logging.getLogger("glee_agent.evolve")

SYSTEM = """You coach a GLEE agent. Ranking is self-gain only.
Rewrite the playbook note for ONE family so the next batch scores higher.
Keep the note under 800 characters, concrete, and about this family's moves.
Reply JSON only: {"family": "...", "playbook": "...", "why": "..."}."""


def compact_results(rows: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    out = []
    for row in rows[-limit:]:
        result = row.get("result") or {}
        out.append(
            {
                "family": row.get("family"),
                "role": row.get("your_player"),
                "outcome": result.get("outcome"),
                "p1": result.get("player_1_payoff"),
                "p2": result.get("player_2_payoff"),
                "bought": result.get("rounds_bought"),
            }
        )
    return out


def fail_rate(family: str, rows: list[dict[str, Any]]) -> float:
    n = 0
    fail = 0
    for row in rows:
        if row.get("family") != family:
            continue
        n += 1
        result = row.get("result") or {}
        both0 = result.get("player_1_payoff") == 0 and result.get("player_2_payoff") == 0
        if family == "persuasion":
            if result.get("rounds_bought") == 0 or both0:
                fail += 1
        elif result.get("outcome") in ("walked_away", "no_deal") or both0:
            fail += 1
    return (fail / n) if n else 1.0


def reflect(family: str, batch: list[dict[str, Any]], ratings_before: dict, ratings_after: dict) -> dict[str, str] | None:
    parent = playbook.current()
    prompt = (
        f"Family to improve: {family}\n"
        f"Ratings before: {json.dumps(ratings_before)}\n"
        f"Ratings after this batch: {json.dumps(ratings_after)}\n"
        f"Fail rate this batch: {fail_rate(family, batch):.2f}\n"
        f"Current playbook:\n{json.dumps(parent, indent=2)}\n\n"
        f"Recent results:\n{json.dumps(compact_results(batch), indent=2)}\n"
    )
    try:
        text = complete(prompt, system=SYSTEM, timeout=40.0, temperature=0.4, max_tokens=700)
        data = parse_json_object(text)
    except (LLMError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("reflect failed: %s", exc)
        return None
    note = str(data.get("playbook") or "").strip()
    target = str(data.get("family") or family)
    if target not in parent or not note:
        logger.warning("reflect ignored: %s", data)
        return None
    child = dict(parent)
    child[target] = note[:1200]
    logger.info("playbook update %s: %s", target, data.get("why"))
    return child
