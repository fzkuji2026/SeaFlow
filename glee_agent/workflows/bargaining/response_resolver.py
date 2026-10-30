"""Stage-ordered response resolution independent of scenario procedures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValueInterval:
    lower: float
    expected: float
    upper: float
    reliable: bool = True

    def __post_init__(self) -> None:
        if not self.lower <= self.expected <= self.upper:
            raise ValueError("value interval must satisfy lower <= expected <= upper")


@dataclass(frozen=True)
class CandidateEstimate:
    action: str
    economic: ValueInterval
    rating: ValueInterval | None = None
    actionable: bool = True


@dataclass(frozen=True)
class Resolution:
    action: str
    reason: str
    eliminated: tuple[str, ...]
    margin: float


def _dominates(left: CandidateEstimate, right: CandidateEstimate) -> bool:
    economic = left.economic.lower >= right.economic.upper
    strict = left.economic.lower > right.economic.upper
    if not economic:
        return False
    if left.rating is None or right.rating is None:
        return strict
    if not left.rating.reliable or not right.rating.reliable:
        return strict
    rating = left.rating.lower >= right.rating.upper
    return rating and (strict or left.rating.lower > right.rating.upper)


def resolve_strict(
    candidates: tuple[CandidateEstimate, ...], *, terminal: bool = False
) -> Resolution | None:
    """Resolve only proven dominance; leave uncertain states to later stages."""
    active = tuple(candidate for candidate in candidates if candidate.actionable)
    if not active:
        return None
    accept = next((c for c in active if c.action == "accept"), None)
    if terminal and accept is not None and accept.economic.lower > 0.0:
        return Resolution("accept", "terminal_positive", (), accept.economic.lower)

    eliminated = {
        candidate.action
        for candidate in active
        if any(
            other.action != candidate.action and _dominates(other, candidate)
            for other in active
        )
    }
    survivors = tuple(c for c in active if c.action not in eliminated)
    if len(survivors) != 1:
        return None
    winner = survivors[0]
    runner_upper = max(
        (c.economic.upper for c in active if c.action != winner.action),
        default=winner.economic.lower,
    )
    return Resolution(
        winner.action,
        "strict_dominance",
        tuple(sorted(eliminated)),
        winner.economic.lower - runner_upper,
    )


def tail_recovery_required(
    rating_prediction: dict[str, float] | None,
    *,
    cutoff: float = -1.5,
    positive_probability_max: float = 0.30,
) -> bool:
    """Use the chronologically validated severe-loss gate for terminal actions."""
    return bool(
        rating_prediction is not None
        and rating_prediction.get("expected_rating_delta", 0.0) <= cutoff
        and rating_prediction.get("positive_probability", 1.0)
        < positive_probability_max
    )
