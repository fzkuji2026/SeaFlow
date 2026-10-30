"""Queue reservations and temporary network/cooldown state."""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .models import RunState

# A queue join is counted as concurrent until the platform has had a full
# matching window to assign it.  Keep this equal to the platform window so a
# lost join response cannot immediately cause a duplicate POST.
QUEUE_RESERVATION_TTL = 30.0
HEALTH_RETRY_INTERVAL = 2.0
MAX_TOPUP_PER_PASS = 12
TOPUP_INTERVAL = 2.0
NETWORK_BREAKER_TTL = 120.0

_NETWORK_GATES: dict[str, "NetworkGate"] = {}
_NETWORK_GATES_LOCK = threading.Lock()


class QueueReservations:
    def __init__(
        self,
        ttl: float = QUEUE_RESERVATION_TTL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl = ttl
        self.clock = clock
        self._by_family: dict[str, deque[tuple[float, str | None, bool]]] = defaultdict(
            deque
        )
        self._lock = threading.Lock()

    def _purge_locked(self, now: float) -> None:
        for family, values in tuple(self._by_family.items()):
            while values and now - values[0][0] >= self.ttl:
                values.popleft()
            if not values:
                self._by_family.pop(family, None)

    def reserve(self, family: str, key: str | None = None) -> str | None:
        """Reserve one queue slot, optionally deduplicated by an idempotency key.

        The reservation is created before the queue request is sent.  A caller
        retrying the same request can pass the same key and will not create a
        second outstanding slot.  ``None`` means that key is already reserved.
        """
        now = self.clock()
        with self._lock:
            self._purge_locked(now)
            values = self._by_family[family]
            if key is not None and any(item_key == key for _, item_key, _ in values):
                return None
            token = key if key is not None else uuid.uuid4().hex
            values.append((now, token, False))
            return token

    def add(self, family: str) -> None:
        # Compatibility entry point for existing callers that do not have an
        # idempotency key.  Each call intentionally represents a new join.
        now = self.clock()
        with self._lock:
            self._purge_locked(now)
            self._by_family[family].append((now, None, False))

    def consume(self, family: str, token: str | None = None) -> None:
        now = self.clock()
        with self._lock:
            self._purge_locked(now)
            values = self._by_family.get(family)
            if values and token is None:
                values.popleft()
            elif values and token is not None:
                for index, (_, item_token, _) in enumerate(values):
                    if item_token == token:
                        del values[index]
                        break
            if not values:
                self._by_family.pop(family, None)

    def mark_uncertain(self, family: str, token: str | None = None) -> None:
        """Keep a lost queue response reserved until its TTL expires."""
        now = self.clock()
        with self._lock:
            self._purge_locked(now)
            values = self._by_family.get(family)
            if not values:
                return
            for index, (created, item_token, uncertain) in enumerate(values):
                if token is None or item_token == token:
                    values[index] = (created, item_token, True)
                    if token is not None:
                        break

    def clear_confirmed(self, family: str | None = None) -> None:
        """Clear joins known to have been cancelled; retain uncertain joins."""
        with self._lock:
            families = (family,) if family is not None else tuple(self._by_family)
            for name in families:
                values = self._by_family.get(name)
                if values is None:
                    continue
                kept = deque(item for item in values if item[2])
                if kept:
                    self._by_family[name] = kept
                else:
                    self._by_family.pop(name, None)

    def clear(self, family: str | None = None) -> None:
        with self._lock:
            if family is None:
                self._by_family.clear()
            else:
                self._by_family.pop(family, None)

    def count(self) -> int:
        now = self.clock()
        with self._lock:
            self._purge_locked(now)
            return sum(map(len, self._by_family.values()))

    def has_uncertain(self, family: str) -> bool:
        now = self.clock()
        with self._lock:
            self._purge_locked(now)
            return any(uncertain for _, _, uncertain in self._by_family.get(family, ()))


@dataclass
class NetworkGate:
    state: RunState | None = None
    reason: str | None = None
    resume_at: float = 0.0
    queues_cancelled: bool = False
    health_successes: int = 0
    paused_at: float = 0.0

    @property
    def paused(self) -> bool:
        return self.state in {RunState.NETWORK_WAITING, RunState.COOLDOWN}

    def expired(self, now: float) -> bool:
        return (
            self.state is RunState.NETWORK_WAITING
            and self.paused_at > 0
            and now - self.paused_at >= NETWORK_BREAKER_TTL
        )

    @property
    def accepting_reservations(self) -> bool:
        """Whether new queue reservations may be created.

        Pausing or draining only affects new queue work; already-owned games
        and outstanding reservations remain tracked until matched or expired.
        """
        return self.state not in {
            RunState.NETWORK_WAITING,
            RunState.COOLDOWN,
            RunState.DRAINING,
            RunState.ENDING,
            RunState.ENDED,
        }

    def pause(self, state: RunState, reason: str, delay: float, now: float) -> None:
        self.state = state
        self.reason = reason
        self.resume_at = max(self.resume_at, now + max(0.0, delay))
        self.paused_at = now
        self.queues_cancelled = False
        self.health_successes = 0

    def resume(self) -> None:
        self.state = None
        self.reason = None
        self.resume_at = 0.0
        self.paused_at = 0.0
        self.queues_cancelled = False
        self.health_successes = 0


def network_gate(key: str) -> NetworkGate:
    """Return the breaker shared by runners using the same API key."""
    with _NETWORK_GATES_LOCK:
        gate = _NETWORK_GATES.get(key)
        if gate is None or gate.expired(time.monotonic()):
            gate = NetworkGate()
            _NETWORK_GATES[key] = gate
        return gate


def is_queue_cooldown(error: Exception) -> bool:
    if getattr(error, "status_code", None) != 403:
        return False
    text = " ".join(
        str(value)
        for value in (
            getattr(error, "code", None),
            getattr(error, "message", None),
            getattr(error, "detail", None),
        )
        if value is not None
    ).casefold()
    return "queue joins are paused" in text or (
        "last 3 games" in text and "timed out" in text
    )


def cooldown_delay(error: Exception, default: float = 30.0) -> float:
    message = str(getattr(error, "message", error))
    match = re.search(r"after\s+([0-9T:+.-]+Z?)", message, re.IGNORECASE)
    if match is None:
        return default
    try:
        when = datetime.fromisoformat(
            match.group(1).rstrip(".").replace("Z", "+00:00")
        )
    except ValueError:
        return default
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max((when - datetime.now(timezone.utc)).total_seconds(), 0.0)
