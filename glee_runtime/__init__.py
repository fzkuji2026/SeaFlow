"""Shared operational runtime for GLEE agents.

Strategy code belongs in :mod:`glee_agent`; this package owns transport,
per-game recovery, cross-process claims, lifecycle state, and persistence.
"""

from .claim_coordinator import ClaimCoordinator
from .models import RunConfig, RunSnapshot, RunState
from .per_game_recovery import GameMoveRecovery
from .persistence import RunStore
from .queue_control import NetworkGate
from .runner import GameRunner
from .strategy_registry import StrategyRegistry
from .transport import ApiTransport

__all__ = [
    "GameRunner",
    "RunConfig",
    "RunSnapshot",
    "RunState",
    "ApiTransport",
    "GameMoveRecovery",
    "ClaimCoordinator",
    "NetworkGate",
    "RunStore",
    "StrategyRegistry",
]
