"""Shared numbers and tiny helpers. No family policy branches here."""

from __future__ import annotations

from glee_agent.families.common import progress, round_money, with_message

__all__ = ["progress", "round_money", "with_message", "below_share_floor"]


def below_share_floor(my_gain: float, pot: float, share: float) -> bool:
    """True when my_gain is strictly below share * pot (1e-9 slack)."""
    return my_gain + 1e-9 < share * pot
