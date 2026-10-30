import pytest

from glee_agent.workflows.bargaining.response_resolver import (
    CandidateEstimate,
    ValueInterval,
    resolve_strict,
    tail_recovery_required,
)


def candidate(action, lower, expected=None, upper=None):
    expected = lower if expected is None else expected
    upper = expected if upper is None else upper
    return CandidateEstimate(action, ValueInterval(lower, expected, upper))


def test_positive_terminal_accept_dominates_zero_exit():
    result = resolve_strict(
        (candidate("accept", 0.01), candidate("reject", 0), candidate("walkaway", 0)),
        terminal=True,
    )
    assert result is not None and result.action == "accept"


def test_accept_wins_only_above_reject_upper_bound():
    result = resolve_strict(
        (
            candidate("accept", 0.128),
            candidate("reject", 0, 0.06, 0.08192),
            candidate("walkaway", 0),
        )
    )
    assert result is not None and result.action == "accept"


def test_uncertain_overlap_defers_to_later_workflow_stages():
    result = resolve_strict(
        (
            candidate("accept", 0.1467),
            candidate("reject", 0, 0.15, 0.172125),
            candidate("walkaway", 0),
        )
    )
    assert result is None


def test_candidate_order_and_positive_scale_do_not_change_resolution():
    base = (
        candidate("accept", 0.4),
        candidate("reject", 0, 0.2, 0.3),
        candidate("walkaway", 0),
    )
    assert resolve_strict(base).action == "accept"
    assert resolve_strict(tuple(reversed(base))).action == "accept"
    scaled = tuple(
        CandidateEstimate(
            row.action,
            ValueInterval(
                row.economic.lower * 100,
                row.economic.expected * 100,
                row.economic.upper * 100,
            ),
        )
        for row in base
    )
    assert resolve_strict(scaled).action == "accept"


def test_invalid_interval_fails_closed():
    with pytest.raises(ValueError):
        ValueInterval(0.5, 0.4, 0.6)


def test_tail_recovery_uses_fitted_rating_cutoff():
    assert tail_recovery_required(
        {"expected_rating_delta": -1.5, "positive_probability": 0.29}
    )
    assert not tail_recovery_required(
        {"expected_rating_delta": -1.49, "positive_probability": 0.29}
    )
    assert not tail_recovery_required(
        {"expected_rating_delta": -2.0, "positive_probability": 0.40}
    )
    assert not tail_recovery_required(None)
