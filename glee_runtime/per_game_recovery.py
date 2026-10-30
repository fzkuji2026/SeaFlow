"""Safely reconcile one ambiguous move without blocking other games."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import requests

from .turn_clock import remaining_seconds

_RECONNECT_GAP_S = 0.2
_RECONNECT_MIN_LEFT_S = 0.2
_POST_READ_TIMEOUT_SECONDS = 8.0
_RECONCILE_GET_TIMEOUT_SECONDS = 2.0
_FINISHED = frozenset({"completed", "finished", "timeout", "cancelled"})
_MISSING = object()


def _is_dead_turn_error(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) != 400:
        return False
    msg = str(getattr(exc, "message", exc)).casefold()
    return "not your turn" in msg or "not active" in msg


def _is_transient_network(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) in (502, 503, 504):
        return True
    return isinstance(exc, (requests.Timeout, requests.ConnectionError, OSError, EOFError))


def _state(game: dict) -> dict:
    value = game.get("game_state")
    return value if isinstance(value, dict) else {}


def _value(game: dict, *keys: str, default: Any = _MISSING) -> Any:
    state = _state(game)
    for key in keys:
        if key in state:
            return state[key]
        if key in game:
            return game[key]
    return default


def _turn_fingerprint(game: dict) -> tuple[str, ...] | None:
    """Return round/phase/history/actions/player/offer/version identity."""
    if not isinstance(game, dict):
        return None
    values = (
        _value(game, "round"),
        _value(game, "phase"),
        _value(game, "history"),
        _value(game, "valid_actions"),
        _value(game, "current_player"),
        _value(game, "last_offer", default=None),
        _value(game, "turn_version", "state_version", "version", default=None),
    )
    # Missing last_offer/version is a valid representation; the other fields
    # are required before an unchanged turn can be established.
    if any(value is _MISSING for value in values[:5]):
        return None
    return tuple(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        for value in values
    )


def _game_finished(game: dict) -> bool:
    if game.get("game_over"):
        return True
    for source in (game, _state(game)):
        for key in ("status", "phase"):
            if str(source.get(key) or "").casefold() in _FINISHED:
                return True
    return False


def _our_turn_and_alive(game: dict, you: Any = None) -> bool:
    if _game_finished(game):
        return False
    current = _value(game, "current_player", default=None)
    who = you or game.get("your_player")
    if who and current:
        return who == current
    return bool(game.get("your_turn")) if "your_turn" in game else False


@dataclass(frozen=True)
class MoveResult:
    payload: dict
    reconciled: bool = False


class MoveUncertain(requests.RequestException):
    def __init__(self, game_id: str, cause: Exception):
        self.game_id = game_id
        self.cause = cause
        super().__init__(f"move outcome is uncertain for {game_id}: {cause}")


@dataclass
class _LockEntry:
    lock: threading.Lock
    users: int = 0


class GameMoveRecovery:
    """POST once; after ambiguity, resend once only after two matching GETs."""

    def __init__(
        self,
        post_move: Callable[[str, dict, tuple[float, float]], dict],
        get_state: Callable[[str, tuple[float, float]], dict],
        *,
        budget_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        already_applied_error: Callable[[Exception], bool] | None = None,
        on_network_error: Callable[[Exception], None] | None = None,
        gap: float = _RECONNECT_GAP_S,
    ) -> None:
        self._post_move = post_move
        self._get_state = get_state
        self._budget_seconds = budget_seconds
        self._clock = clock
        self._sleep = sleep
        self._gap = gap
        self._already_applied_error = already_applied_error
        self._on_network_error = on_network_error
        self._locks: dict[str, _LockEntry] = {}
        self._uncertain: dict[str, tuple[tuple[str, ...], Exception]] = {}
        self._locks_guard = threading.Lock()

    @contextmanager
    def _locked(self, game_id: str) -> Iterator[None]:
        with self._locks_guard:
            entry = self._locks.setdefault(game_id, _LockEntry(threading.Lock()))
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._locks_guard:
                entry.users -= 1
                if entry.users == 0 and self._locks.get(game_id) is entry:
                    self._locks.pop(game_id, None)

    def forget(self, game_id: str) -> None:
        """Drop recovery state after the game lifecycle has ended."""
        with self._locks_guard:
            entry = self._locks.get(str(game_id))
            if entry is not None and entry.users == 0:
                self._locks.pop(str(game_id), None)
            self._uncertain.pop(str(game_id), None)

    def release(self, game_id: str) -> None:
        """Backward-compatible alias for lifecycle cleanup."""
        self.forget(game_id)

    def set_network_error_callback(self, callback: Callable[[Exception], None]) -> None:
        self._on_network_error = callback

    def _timeout(self, deadline: float, read_cap: float) -> tuple[float, float]:
        remaining = min(max(0.05, deadline - self._clock()), read_cap)
        return min(2.0, remaining), remaining

    def _deadline(self, game: dict) -> float:
        remaining = max(0.0, remaining_seconds(game))
        if self._budget_seconds is not None:
            remaining = min(remaining, self._budget_seconds)
        return self._clock() + remaining

    @staticmethod
    def _reconciled(state: dict) -> MoveResult:
        payload = {
            "valid": True,
            "game_over": _game_finished(state),
            "reconciled": True,
            "applied": True,
        }
        result = state.get("result", _state(state).get("result"))
        if result is not None:
            payload["result"] = result
        return MoveResult(payload, reconciled=True)

    def _remember_uncertain(
        self, game_id: str, token: tuple[str, ...] | None, error: Exception
    ) -> None:
        if token is not None:
            self._uncertain[game_id] = (token, error)

    def _wait_past_uncertain(
        self,
        game_id: str,
        token: tuple[str, ...],
        you: Any,
        deadline: float,
        last_err: Exception,
    ) -> MoveResult:
        """Never POST a token whose resend outcome was also ambiguous."""
        while self._clock() < deadline - _RECONNECT_MIN_LEFT_S:
            try:
                state = self._get_state(
                    game_id, self._timeout(deadline, _RECONCILE_GET_TIMEOUT_SECONDS)
                )
            except Exception as exc:
                if _is_dead_turn_error(exc) or not _is_transient_network(exc):
                    raise
                last_err = exc
            else:
                if isinstance(state, dict):
                    current = _turn_fingerprint(state)
                    if _game_finished(state) or not _our_turn_and_alive(state, you) or (
                        current is not None and current != token
                    ):
                        self._uncertain.pop(game_id, None)
                        return self._reconciled(state)
                    deadline = min(
                        deadline,
                        self._clock() + max(0.0, remaining_seconds(state)),
                    )
            leftover = deadline - self._clock()
            if leftover <= _RECONNECT_MIN_LEFT_S:
                break
            self._sleep(min(self._gap, leftover / 2))
        raise MoveUncertain(game_id, last_err) from last_err

    def _reconcile_dead_turn(
        self,
        game_id: str,
        token: tuple[str, ...] | None,
        you: Any,
        deadline: float,
        error: Exception,
    ) -> MoveResult:
        """A server dead-turn rejection is definitive: GET, but never resend."""
        while self._clock() < deadline - _RECONNECT_MIN_LEFT_S:
            try:
                state = self._get_state(
                    game_id, self._timeout(deadline, _RECONCILE_GET_TIMEOUT_SECONDS)
                )
            except Exception as exc:
                if not _is_transient_network(exc):
                    raise
            else:
                if isinstance(state, dict):
                    current = _turn_fingerprint(state)
                    if _game_finished(state) or not _our_turn_and_alive(state, you) or (
                        token is not None and current is not None and current != token
                    ):
                        self._uncertain.pop(game_id, None)
                        return self._reconciled(state)
                    if current is not None and current == token:
                        self._remember_uncertain(game_id, token, error)
                        return self._reconciled(state)
            leftover = deadline - self._clock()
            if leftover <= _RECONNECT_MIN_LEFT_S:
                break
            self._sleep(min(self._gap, leftover / 2))
        self._remember_uncertain(game_id, token, error)
        return self._reconciled({})

    def _read_until_definitive(
        self,
        game_id: str,
        original: tuple[str, ...] | None,
        you: Any,
        deadline: float,
        last_err: Exception,
    ) -> tuple[dict | None, Exception]:
        previous: tuple[str, ...] | None = None
        while self._clock() < deadline - _RECONNECT_MIN_LEFT_S:
            try:
                state = self._get_state(
                    game_id, self._timeout(deadline, _RECONCILE_GET_TIMEOUT_SECONDS)
                )
            except Exception as exc:
                if _is_dead_turn_error(exc) or not _is_transient_network(exc):
                    raise
                last_err = exc
                previous = None
            else:
                if not isinstance(state, dict):
                    previous = None
                elif not _our_turn_and_alive(state, you):
                    return state, last_err
                else:
                    current = _turn_fingerprint(state)
                    if original is not None and current is not None and current != original:
                        return state, last_err
                    if current is not None and current == original and current == previous:
                        return None, last_err
                    previous = current if current == original else None
                    deadline = min(
                        deadline,
                        self._clock() + max(0.0, remaining_seconds(state)),
                    )
            leftover = deadline - self._clock()
            if leftover <= _RECONNECT_MIN_LEFT_S:
                break
            self._sleep(min(self._gap, leftover / 2))
        raise MoveUncertain(game_id, last_err) from last_err

    def submit(self, game: dict, action: dict) -> MoveResult:
        game_id = str(game["game_id"])
        with self._locked(game_id):
            deadline = self._deadline(game)
            token = _turn_fingerprint(game)
            uncertain = self._uncertain.get(game_id)
            if uncertain is not None:
                uncertain_token, uncertain_error = uncertain
                if token is not None and token != uncertain_token:
                    self._uncertain.pop(game_id, None)
                else:
                    return self._wait_past_uncertain(
                        game_id,
                        uncertain_token,
                        game.get("your_player"),
                        deadline,
                        uncertain_error,
                    )
            try:
                return MoveResult(
                    self._post_move(
                        game_id,
                        action,
                        self._timeout(deadline, _POST_READ_TIMEOUT_SECONDS),
                    )
                )
            except Exception as exc:
                if _is_dead_turn_error(exc) or (
                    self._already_applied_error and self._already_applied_error(exc)
                ):
                    return self._reconcile_dead_turn(
                        game_id,
                        token,
                        game.get("your_player"),
                        deadline,
                        exc,
                    )
                if not _is_transient_network(exc):
                    raise
                if self._on_network_error is not None:
                    self._on_network_error(exc)
                last_err = exc

            changed, last_err = self._read_until_definitive(
                game_id,
                token,
                game.get("your_player"),
                deadline,
                last_err,
            )
            if changed is not None:
                return self._reconciled(changed)

            # Exactly one resend. Its outcome is never followed by another POST.
            try:
                return MoveResult(
                    self._post_move(
                        game_id,
                        action,
                        self._timeout(deadline, _POST_READ_TIMEOUT_SECONDS),
                    )
                )
            except Exception as exc:
                if self._already_applied_error and self._already_applied_error(exc):
                    self._remember_uncertain(game_id, token, exc)
                    return self._reconciled({})
                if _is_transient_network(exc):
                    self._remember_uncertain(game_id, token, exc)
                    raise MoveUncertain(game_id, exc) from exc
                raise
