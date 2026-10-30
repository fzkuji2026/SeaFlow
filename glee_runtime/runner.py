"""Task-local GLEE queue, move scheduling, pause, and drain loop."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from inspect import signature
from typing import Any, Callable

import requests

from .models import RunSnapshot, RunState
from .per_game_recovery import MoveUncertain
from .transport import NetworkUnavailable
from .queue_control import (
    HEALTH_RETRY_INTERVAL,
    MAX_TOPUP_PER_PASS,
    TOPUP_INTERVAL,
    NetworkGate,
    QueueReservations,
    cooldown_delay,
    is_queue_cooldown,
    network_gate,
)
from .turn_clock import poll_sleep_seconds, remaining_seconds

logger = logging.getLogger("glee_runtime.runner")
DRAIN_EMPTY_CONFIRMATIONS = 3


class GameRunner:
    """Run one task without using account-wide active game counts."""

    def __init__(self, client: Any):
        self.client = client
        self.state = RunState.WAITING
        self.snapshot = RunSnapshot(self.state)
        self._gate = network_gate(self._network_key())
        self._queues_cancelled = False
        self._draining = False
        self._reservations = QueueReservations()
        self._owned_ids: set[str] = set()
        self._seen_ids: set[str] = set()
        self._last_move_done: dict[str, float] = {}
        self._last_pending_ok = False
        self._owned_games: dict[str, dict] = {}
        self._uncertain_turns: dict[str, str] = {}
        self._queue_index = 0

    def _network_key(self) -> str:
        api_key = getattr(self.client, "api_key", None)
        if api_key:
            return str(api_key)
        transport = getattr(self.client, "transport", None)
        authorization = getattr(transport, "authorization", None)
        if authorization:
            return str(authorization)
        session = getattr(self.client, "session", None)
        headers = getattr(session, "headers", None)
        authorization = headers.get("Authorization") if headers is not None else None
        return str(authorization) if authorization else f"client:{id(self.client)}"

    def _owned_count(self) -> int:
        active_owned_count = getattr(self.client, "active_owned_count", None)
        if callable(active_owned_count):
            try:
                return max(0, int(active_owned_count()))
            except Exception as exc:
                logger.warning("Could not count active owned games: %s", exc)
        return len(self._owned_ids)

    def _client_active_ids(self) -> set[str] | None:
        assigned = getattr(self.client, "assigned", None)
        if not isinstance(assigned, dict):
            return None
        completed = getattr(self.client, "completed", ())
        return {str(game_id) for game_id in assigned if game_id not in completed}

    def _close_game_resources(self, game_id: str) -> None:
        closer = getattr(self.client, "close_game_resources", None)
        if not callable(closer):
            closer = getattr(self.client, "close_game_session", None)
        if not callable(closer):
            closer = getattr(
                getattr(self.client, "transport", None),
                "close_game_session",
                None,
            )
        if callable(closer):
            try:
                closer(game_id)
            except Exception as exc:
                logger.warning("Could not close game %s resources: %s", game_id, exc)

    def _forget_game(self, game_id: str) -> None:
        self._owned_ids.discard(game_id)
        self._seen_ids.discard(game_id)
        self._owned_games.pop(game_id, None)
        self._uncertain_turns.pop(game_id, None)
        self._close_game_resources(game_id)

    def _set_state(
        self,
        state: RunState,
        *,
        running_moves: int = 0,
        reason: str | None = None,
    ) -> None:
        self.state = state
        self.snapshot = RunSnapshot(
            state,
            owned_games=self._owned_count(),
            running_moves=running_moves,
            queue_reservations=self._reservations.count(),
            reason=reason,
        )
        store = getattr(self.client, "run_store", None)
        if store is not None:
            try:
                store.save_snapshot(asdict(self.snapshot))
            except Exception as exc:
                logger.warning("Could not publish run snapshot: %s", exc)

    def _pause(self, state: RunState, reason: str, delay: float) -> None:
        self._queues_cancelled = False
        self._gate.pause(state, reason, delay, time.monotonic())
        self._set_state(state, reason=reason)

    def _pause_for_error(self, error: Exception) -> None:
        if is_queue_cooldown(error):
            self._pause(RunState.COOLDOWN, str(error), cooldown_delay(error))
        elif (
            isinstance(
                error,
                (requests.RequestException, NetworkUnavailable, OSError, EOFError),
            )
            or getattr(error, "status_code", None) in {502, 503, 504}
        ):
            self._pause(RunState.NETWORK_WAITING, str(error), HEALTH_RETRY_INTERVAL)

    @staticmethod
    def _turn_token(game: dict) -> str:
        state = game.get("game_state")
        state = state if isinstance(state, dict) else {}
        values = {
            key: state.get(key, game.get(key))
            for key in (
                "round",
                "phase",
                "history",
                "valid_actions",
                "current_player",
                "last_offer",
                "turn_version",
                "state_version",
                "version",
            )
        }
        return json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)

    def _queue_one(self, family: str) -> None:
        if self._reservations.has_uncertain(family):
            return
        token = self._reservations.reserve(family)
        try:
            self.client.queue(family)
        except (requests.RequestException, NetworkUnavailable) as exc:
            self._reservations.mark_uncertain(family, token)
            self._pause_for_error(exc)
            raise
        except Exception:
            self._reservations.consume(family, token)
            raise
        # A successful join is still unassigned.  Keep it reserved until a
        # matching game appears (or the platform matching window expires).

    def _leave_families(self, families: tuple[str, ...]) -> None:
        for family in families:
            try:
                self.client.leave_queue(family)
            except Exception as exc:
                logger.warning("Could not leave %s queue: %s", family, exc)
            else:
                self._reservations.clear_confirmed(family)

    def _pending_games(self) -> list[dict]:
        pending = self.client.pending_games
        try:
            games = (
                pending(short=self._gate.paused)
                if "short" in signature(pending).parameters
                else pending()
            )
        except Exception:
            self._last_pending_ok = False
            raise
        self._last_pending_ok = True
        return games

    def _network_healthy(self) -> bool:
        health = getattr(self.client, "network_healthy", None)
        if callable(health):
            return bool(health())
        stats = self.client.stats
        stats(short=True) if "short" in signature(stats).parameters else stats()
        return True

    def _maybe_resume(self, families: tuple[str, ...]) -> None:
        if self._draining or not self._gate.paused:
            return
        now = time.monotonic()
        if self._gate.expired(now):
            self._gate.resume()
            return
        if not self._queues_cancelled:
            self._leave_families(families)
            self._queues_cancelled = True
        if now < self._gate.resume_at:
            return
        try:
            if not self._network_healthy():
                raise requests.ConnectionError("network health check failed")
        except Exception as exc:
            self._gate.resume_at = time.monotonic() + HEALTH_RETRY_INTERVAL
            self._gate.health_successes = 0
            logger.warning("Network health check failed: %s", exc)
            return
        self._gate.health_successes += 1
        if self._gate.health_successes < 2:
            self._gate.resume_at = time.monotonic() + HEALTH_RETRY_INTERVAL
            return
        self._gate.resume()
        self._set_state(RunState.WAITING)
        logger.info("Network healthy; queue claiming resumed")

    def _topup(
        self,
        families: tuple[str, ...],
        game_concurrency: int,
        queue_target: int,
        active_limit: Callable[[], int] | None,
        *,
        adaptive: bool = False,
        deadline: float | None = None,
    ) -> None:
        if self._draining or self._gate.paused:
            return
        requested = active_limit() if active_limit is not None else game_concurrency
        if adaptive:
            # Adaptive mode deliberately has no local queue target.  A single
            # confirmed response permits the next join on the next paced pass;
            # the shared transport budget remains the request-level ceiling.
            if requested is not None and int(requested) <= 0:
                return
            if deadline is not None and time.monotonic() >= deadline:
                return
            try:
                self._queue_one(families[self._queue_index % len(families)])
                self._queue_index += 1
                logger.info(
                    "adaptive queue confirmed; active=%d reservations=%d",
                    self._owned_count(),
                    self._reservations.count(),
                )
            except Exception as exc:
                self._pause_for_error(exc)
                if is_queue_cooldown(exc) or isinstance(
                    exc, requests.RequestException
                ):
                    return
                logger.warning("Queue request failed: %s", exc)
            return
        target = min(game_concurrency, queue_target, max(0, int(requested)))
        used = self._owned_count() + self._reservations.count()
        joins = min(max(target - used, 0), MAX_TOPUP_PER_PASS)
        for index in range(joins):
            if self._gate.paused:
                return
            if deadline is not None and time.monotonic() >= deadline:
                return
            try:
                self._queue_one(families[index % len(families)])
            except Exception as exc:
                self._pause_for_error(exc)
                if is_queue_cooldown(exc) or isinstance(
                    exc, requests.RequestException
                ):
                    return
                logger.warning("Queue request failed: %s", exc)

    def _reap_future(self, future: Future, game_id: str) -> bool:
        finished = False
        failed = False
        uncertain = False
        try:
            finished = bool(future.result())
        except BaseException as exc:
            failed = True
            uncertain = isinstance(exc, MoveUncertain)
            logger.warning("Game %s error: %s. Continuing...", game_id, exc)
        finally:
            # MoveUncertain retains the per-game token and session so the next
            # worker can reconcile the same turn instead of silently abandoning
            # it until the platform times it out.
            if not uncertain:
                self._close_game_resources(game_id)
        self._last_move_done[game_id] = time.time()
        if finished or (failed and not uncertain):
            self._owned_ids.discard(game_id)
        if finished:
            self._owned_games.pop(game_id, None)
            self._uncertain_turns.pop(game_id, None)
        return finished

    def _reconcile_assigned(self, pending_ids: set[str]) -> int:
        reconcile = getattr(self.client, "reconcile_assigned", None)
        if not callable(reconcile):
            return 0
        before = self._client_active_ids()
        try:
            released = int(reconcile(pending_ids) or 0)
        except Exception as exc:
            logger.warning("Could not reconcile assigned games: %s", exc)
            return 0
        after = self._client_active_ids()
        if before is not None and after is not None:
            for game_id in before - after:
                self._forget_game(game_id)
        return released

    def _adopt_platform_games(self, families: tuple[str, ...]) -> None:
        pending = getattr(self.client, "pending_games", None)
        if not callable(pending):
            return
        try:
            games = self._pending_games()
        except Exception:
            return
        for game in games:
            if game.get("game_family") not in families:
                continue
            game_id = str(game.get("game_id") or "")
            if game_id:
                self._seen_ids.add(game_id)
                self._owned_ids.add(game_id)
                self._owned_games[game_id] = game

    def _install_network_pause(self) -> None:
        recovery = getattr(self.client, "move_recovery", None)
        setter = getattr(recovery, "set_network_error_callback", None)
        if callable(setter):
            setter(self._pause_for_error)

    def _family_still_on_table(self, families: tuple[str, ...]) -> bool:
        pending = getattr(self.client, "pending_games", None)
        if not callable(pending):
            return False
        try:
            games = self._pending_games()
        except Exception:
            return True
        return any(game.get("game_family") in families for game in games)

    def _drain_complete(self, inflight: dict[Future, str]) -> bool:
        pending_confirmed = self._last_pending_ok or not callable(
            getattr(self.client, "pending_games", None)
        )
        return (
            not inflight
            and self._owned_count() == 0
            and self._reservations.count() == 0
            and pending_confirmed
        )

    def run(
        self,
        strategy: Callable[[dict], dict],
        game_families: list[str] | None = None,
        poll_interval: float = 2.0,
        max_games: int | None = None,
        max_time: float | None = None,
        requeue: bool = True,
        concurrency: int | None = None,
        should_stop: Callable[[], bool] | None = None,
        start_games: int | None = None,
        queue_depth: int | None = None,
        move_concurrency: int | None = None,
        active_limit: Callable[[], int] | None = None,
    ) -> None:
        del start_games
        families = tuple(
            game_families or ("bargaining", "negotiation", "persuasion")
        )
        valid = {"bargaining", "negotiation", "persuasion"}
        if not families or any(family not in valid for family in families):
            raise ValueError("invalid game families")
        if concurrency is not None and concurrency < 0:
            raise ValueError("concurrency must be >= 0")
        adaptive = concurrency in (None, 0)
        self.adaptive = adaptive
        game_limit = concurrency if not adaptive else 1
        if game_limit < 1:
            raise ValueError("concurrency must be >= 1")
        if queue_depth is not None and queue_depth < 0:
            raise ValueError("queue_depth must be >= 0")
        if adaptive:
            queue_target = 0
        else:
            queue_target = min(
                concurrency,
                queue_depth if queue_depth is not None else concurrency,
            )
        workers = move_concurrency if move_concurrency is not None else game_limit
        if workers < 1:
            raise ValueError("move_concurrency must be >= 1")
        workers = min(workers, game_limit)
        self.client._allowed_game_families = frozenset(families)
        self.client.game_families = frozenset(families)
        self._install_network_pause()
        initial_assigned = int(getattr(self.client, "assigned_count", 0))
        self._adopt_platform_games(families)
        draining = bool(
            (should_stop is not None and should_stop())
            or (max_games is not None and initial_assigned >= max_games)
        )
        started = time.monotonic()
        deadline = started + max_time if max_time is not None else None
        if draining or (deadline is not None and started >= deadline):
            draining = True
            self._start_drain(families, 0)
        else:
            self._set_state(RunState.RUNNING)
            self._topup(
                families,
                game_limit,
                queue_target,
                active_limit,
                adaptive=adaptive,
                deadline=deadline,
            )
        completed = 0
        drain_empty_polls = 0
        last_topup = time.monotonic()
        inflight: dict[Future, str] = {}
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                while True:
                    for future in tuple(inflight):
                        if not future.done():
                            continue
                        game_id = inflight.pop(future)
                        if self._reap_future(future, game_id):
                            completed += 1

                    now = time.monotonic()
                    assigned = int(getattr(self.client, "assigned_count", completed))
                    if not draining and (
                        bool(getattr(self.client, "drain_requested", False))
                        or (should_stop is not None and should_stop())
                        or (max_games is not None and assigned >= max_games)
                        or (max_time is not None and now - started >= max_time)
                    ):
                        draining = True
                        self._start_drain(families, len(inflight))

                    if draining:
                        if self._drain_complete(inflight):
                            drain_empty_polls += 1
                        else:
                            drain_empty_polls = 0
                        if drain_empty_polls >= DRAIN_EMPTY_CONFIRMATIONS:
                            self._set_state(RunState.ENDING)
                            self._set_state(RunState.ENDED)
                            return

                    if not draining:
                        self._maybe_resume(families)
                    if (
                        requeue
                        and not draining
                        and not self._gate.paused
                        and now - last_topup >= TOPUP_INTERVAL
                    ):
                        last_topup = now
                        self._topup(
                            families,
                            game_limit,
                            queue_target,
                            active_limit,
                            adaptive=adaptive,
                            deadline=deadline,
                        )

                    try:
                        games = [
                            game
                            for game in self._pending_games()
                            if game.get("game_family") in families
                        ]
                        for game in games:
                            game_id = str(game.get("game_id") or "")
                            if game_id:
                                self._owned_games[game_id] = game
                    except Exception as exc:
                        self._pause_for_error(exc)
                        games = [
                            game
                            for game in self._owned_games.values()
                            if game.get("game_family") in families
                        ]

                    pending_ids = {
                        str(game.get("game_id") or "") for game in games
                    }
                    released = self._reconcile_assigned(pending_ids)
                    if released and requeue and not draining and not self._gate.paused:
                        self._topup(
                            families,
                            game_limit,
                            queue_target,
                            active_limit,
                            adaptive=adaptive,
                            deadline=deadline,
                        )
                        last_topup = now

                    now_wall = time.time()
                    games.sort(
                        key=lambda game: remaining_seconds(
                            game,
                            now=now_wall,
                            last_move_done=self._last_move_done.get(
                                str(game.get("game_id") or "")
                            ),
                        )
                    )
                    submitted = self._submit_games(
                        pool, inflight, games, strategy, workers
                    )
                    self._publish_loop_state(draining, inflight)
                    if not submitted:
                        leftover = [
                            game
                            for game in games
                            if str(game.get("game_id") or "")
                            not in inflight.values()
                        ]
                        sleep_s = poll_sleep_seconds(
                            leftover,
                            poll_interval,
                            now=now_wall,
                            last_done=self._last_move_done,
                        )
                        if inflight:
                            wait(
                                tuple(inflight),
                                timeout=sleep_s,
                                return_when=FIRST_COMPLETED,
                            )
                        elif sleep_s > 0:
                            time.sleep(sleep_s)
        except KeyboardInterrupt:
            self._set_state(RunState.ENDING)
        except BaseException:
            self._set_state(RunState.ENDING)
            raise
        finally:
            self._leave_families(families)

    def _start_drain(self, families: tuple[str, ...], running_moves: int) -> None:
        self._draining = True
        stop_claiming = getattr(self.client, "stop_claiming", None)
        if callable(stop_claiming):
            stop_claiming()
        self._leave_families(families)
        self._set_state(RunState.DRAINING, running_moves=running_moves)

    def _submit_games(
        self,
        pool: ThreadPoolExecutor,
        inflight: dict[Future, str],
        games: list[dict],
        strategy: Callable[[dict], dict],
        move_workers: int,
    ) -> bool:
        inflight_ids = set(inflight.values())
        submitted = False
        for game in games:
            game_id = str(game.get("game_id") or "")
            if not game_id:
                continue
            uncertain = self._uncertain_turns.get(game_id)
            current_token = self._turn_token(game)
            if uncertain == current_token:
                continue
            if uncertain is not None:
                self._uncertain_turns.pop(game_id, None)
            if game_id not in self._seen_ids:
                self._seen_ids.add(game_id)
                self._owned_ids.add(game_id)
                self._reservations.consume(str(game.get("game_family") or ""))
            self._owned_games[game_id] = game
            if game_id in inflight_ids or len(inflight) >= move_workers:
                continue
            future = pool.submit(self.client._handle_game, strategy, game)
            inflight[future] = game_id
            inflight_ids.add(game_id)
            submitted = True
        return submitted

    def _publish_loop_state(
        self, draining: bool, inflight: dict[Future, str]
    ) -> None:
        if draining:
            self._set_state(RunState.DRAINING, running_moves=len(inflight))
        elif self._gate.paused:
            self._set_state(
                self._gate.state or RunState.NETWORK_WAITING,
                running_moves=len(inflight),
                reason=self._gate.reason,
            )
        elif self._owned_count() or inflight:
            self._set_state(RunState.RUNNING, running_moves=len(inflight))
        else:
            self._set_state(RunState.WAITING)
