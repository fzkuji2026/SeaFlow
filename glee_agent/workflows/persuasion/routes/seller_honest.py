"""Honor policy.seller_always_yes (live True at the 1800 peak).
Otherwise skip lows only after a sale."""

from __future__ import annotations

from typing import Any

from glee_agent.policy import current
from glee_agent.rules import GameView
from glee_agent.workflows.persuasion._beliefs import (
    buyer_prefers_skip_signal,
    buyer_prior_ev_needs_signal,
    seller_has_skipped,
    seller_should_probe_skip,
)


def handle(view: GameView) -> dict[str, Any]:
    if view.action_type == "seller_message":
        return _message(view)
    return _recommend(view)


def _has_sale(state: dict) -> bool:
    for row in state.get("history") or []:
        if not isinstance(row, dict):
            continue
        bought = row.get("bought") or str(row.get("buyer_decision") or "").lower() in (
            "yes",
            "buy",
        )
        if bought:
            return True
    return False


def _should(view: GameView) -> bool:
    if buyer_prefers_skip_signal(view.state):
        return False
    quality = str(view.state.get("current_quality") or view.state.get("quality") or "").lower()
    if quality != "low":
        return True
    if view.is_last_round:
        return True
    if view.state.get("is_seller_know_cv") is False:
        return not (
            seller_has_skipped(view.state) or seller_should_probe_skip(view.state)
        )
    if view.state.get("is_seller_know_cv") is True and (
        seller_has_skipped(view.state) or seller_should_probe_skip(view.state)
    ):
        return False
    # f6b28640 / 2cadd9df: when public prior EV is at or below price, lying on
    # the first low poisons the signal and eliminates later high-quality sales.
    if buyer_prior_ev_needs_signal(view.state):
        return False
    # 2026-08-22 evolve peak persuasion 1875: seller_always_yes True.
    # logs/policy.json still has it; skip_lows was ignoring the knob.
    if bool(current()["persuasion"].get("seller_always_yes")):
        return True
    # 122a77fa / e925a022 / 29a3f2d1: skip on r1 low, +EV buyers
    # skip_block forever. 979a2211: skip lows after they had bought highs.
    if not _has_sale(view.state):
        return True
    return False


def _message(view: GameView) -> dict[str, Any]:
    if _should(view):
        return {"message": "This unit is high quality. I recommend buying it."}
    return {"message": "Honestly, I would skip this one — it is not my best unit."}


def _recommend(view: GameView) -> dict[str, Any]:
    return {"decision": "yes" if _should(view) else "no"}
