"""Hidden values, no max_rounds.

Dead own-types WalkAway an unprofitable stall. Strong types (buyer 1.5× /
seller 0.8×) keep their public payoff floor. Live types continue legal
Faratin counters because an early voluntary exit fixes payoff at zero; their
incoming offers are compared with the planned counter.
"""

from __future__ import annotations

from typing import Any

from glee_agent.rules import GameView
from glee_agent.workflows.negotiation._tactics import play_hidden


def handle(view: GameView) -> dict[str, Any]:
    return play_hidden(view, "open_horizon")
