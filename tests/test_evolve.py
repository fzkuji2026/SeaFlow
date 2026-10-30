from datetime import datetime, timedelta, timezone

from glee_sdk.client import GleeAPIError

from scripts.evolve import _pause_seconds


def test_pause_seconds_reads_unlock_timestamp():
    when = datetime.now(timezone.utc) + timedelta(seconds=120)
    stamp = when.strftime("2026-08-22T22:04:30.373723Z")
    # Use a far-future stamp so the wait is deterministic.
    exc = GleeAPIError(403, f"Queue joins are paused; try again after {stamp}")
    wait = _pause_seconds(exc)
    assert wait >= 5.0


def test_pause_seconds_defaults_without_timestamp():
    assert _pause_seconds(GleeAPIError(403, "paused")) == 60.0
