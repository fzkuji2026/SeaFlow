"""Known seller, no skip and no lemon yet.

Procedure: raw_ev (talk is cheap). After a skip: after_skip_trust.
Last-round: last_raw_ev. Lemon: lemon_ev (not a hard block).
"""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView
from glee_agent.workflows.persuasion._beliefs import compose_buy, execute_buy


def handle(view: GameView) -> dict[str, Any]:
    return execute_buy(view, compose_buy(view, "buyer_known_untested"))
