"""GLEE Competition SDK client (vendored from glee-sdk 0.0.5)."""

from __future__ import annotations

import email.utils
import logging
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from glee_runtime.lifecycle import GameRunner
from glee_runtime.per_game_recovery import GameMoveRecovery
from glee_runtime.transport import ApiTransport

logger = logging.getLogger("glee_sdk")

MAX_MESSAGE_LEN = 2000


class GleeAPIError(Exception):
    """Raised when the GLEE API returns a non-success response."""

    def __init__(
        self,
        status_code: int,
        message: str,
        code: str | None = None,
        detail: Any = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail
        super().__init__(f"[{status_code}] {message}")


class CompetitionNotOpenError(GleeAPIError):
    """The competition hasn't opened yet. Check ``competition_open_at``."""

    def __init__(self, message: str, detail: dict):
        super().__init__(403, message, code="competition_not_open", detail=detail)
        self.competition_open_at: str | None = detail.get("competition_open_at")
        self.competition_close_at: str | None = detail.get("competition_close_at")


class CompetitionClosedError(GleeAPIError):
    """The competition has closed."""

    def __init__(self, message: str, detail: dict):
        super().__init__(403, message, code="competition_closed", detail=detail)
        self.competition_open_at: str | None = detail.get("competition_open_at")
        self.competition_close_at: str | None = detail.get("competition_close_at")


def _raise_for_response(resp: requests.Response) -> None:
    if resp.ok:
        return
    body: Any = None
    try:
        body = resp.json()
    except ValueError:
        body = resp.text

    detail = body.get("detail") if isinstance(body, dict) else body
    if isinstance(detail, dict):
        code = detail.get("code")
        message = detail.get("message") or str(detail)
        if code == "competition_not_open":
            raise CompetitionNotOpenError(message, detail)
        if code == "competition_closed":
            raise CompetitionClosedError(message, detail)
        raise GleeAPIError(resp.status_code, message, code=code, detail=detail)
    raise GleeAPIError(resp.status_code, str(detail) if detail else resp.reason)


def _parse_retry_after(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return default
    if when is None:
        return default
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max((when - datetime.now(timezone.utc)).total_seconds(), 0.0)


class GleeClient:
    """Client for the GLEE Competition API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://glee-competition.com",
        timeout: float | tuple[float, float] = (2.0, 5.0),
    ):
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/agent"
        self.timeout = timeout
        self.transport = ApiTransport(
            f"Bearer {api_key}",
            timeout=timeout,
            trust_env=False,
        )
        # Public compatibility: callers historically mounted adapters on
        # ``client.session``.
        self.session = self.transport.default_session
        self.move_recovery = GameMoveRecovery(
            self._post_move_once,
            self._get_game_once,
            already_applied_error=self._not_your_turn,
        )

    def _request(self, method: str, path: str, **kwargs) -> dict | list:
        url = f"{self.api_url}{path}"
        attempts = int(kwargs.pop("_attempts", 3))
        retry_post = bool(kwargs.pop("_retry_post_connection_errors", True))
        request_session = kwargs.pop("_session", None)
        transport = getattr(self, "transport", None)
        if transport is None:
            # Subclasses constructed without ``super().__init__`` remain
            # supported during migration.
            transport = ApiTransport(
                self.session.headers.get("Authorization", ""),
                timeout=getattr(self, "timeout", (2.0, 5.0)),
                trust_env=self.session.trust_env,
                session_factory=lambda: self.session,
            )
        try:
            return transport.request_json(
                method,
                url,
                raise_for_response=_raise_for_response,
                attempts=attempts,
                retry_post_connection_errors=retry_post,
                session=request_session,
                **kwargs,
            )
        finally:
            # Keep the historical public alias current after a route rebind.
            self.session = transport.default_session

    def queue(self, game_family: str) -> dict:
        return self._request(
            "POST",
            "/queue",
            json={"game_family": game_family},
            _attempts=1,
            _retry_post_connection_errors=False,
        )

    def leave_queue(self, family: str | None = None) -> dict:
        params = {"game_family": family} if family else None
        try:
            return self._request("DELETE", "/queue", params=params)
        except GleeAPIError as e:
            if e.status_code in (404, 501):
                return {}
            raise

    def _leave_queue_quietly(self, families: list[str] | None = None) -> None:
        for family in families or [None]:
            try:
                self.leave_queue(family)
            except Exception as e:
                logger.warning(f"Could not leave the matchmaking queue: {e}")

    def pending_games(self, *, short: bool = False) -> list[dict]:
        kwargs = {"timeout": (1.0, 1.5), "_attempts": 1} if short else {}
        return self._request("GET", "/games/pending", **kwargs)

    def move(self, game_id: str, action: dict) -> dict:
        return self._request(
            "POST",
            f"/games/{game_id}/move",
            json={"action": action},
            _attempts=1,
            _retry_post_connection_errors=False,
        )

    @staticmethod
    def _not_your_turn(error: Exception) -> bool:
        if not isinstance(error, GleeAPIError) or error.status_code != 400:
            return False
        msg = error.message.casefold()
        return "not your turn" in msg or "not active" in msg

    def _post_move_once(
        self,
        game_id: str,
        action: dict,
        timeout: tuple[float, float],
    ) -> dict:
        return self._request(
            "POST",
            f"/games/{game_id}/move",
            json={"action": action},
            timeout=timeout,
            _attempts=1,
            _retry_post_connection_errors=False,
            _session=self.transport.game_session(game_id),
        )

    def _get_game_once(self, game_id: str, timeout: tuple[float, float]) -> dict:
        return self._request(
            "GET",
            f"/games/{game_id}",
            timeout=timeout,
            _attempts=1,
            _session=self.transport.game_session(game_id),
        )

    def close_game_resources(self, game_id: str) -> None:
        """Release per-game recovery and HTTP resources after lifecycle exit."""
        self.move_recovery.forget(game_id)
        self.transport.close_game_session(game_id)

    def game_state(self, game_id: str) -> dict:
        return self._request("GET", f"/games/{game_id}")

    def stats(self, *, short: bool = False) -> dict:
        kwargs = {"timeout": (1.0, 1.5), "_attempts": 1} if short else {}
        return self._request("GET", "/stats", **kwargs)

    def network_healthy(self) -> bool:
        self.stats(short=True)
        return True

    def _handle_game(self, strategy: Callable[[dict], dict], game: dict) -> bool:
        game_id = game["game_id"]
        family = game["game_family"]
        logger.info(
            f"Game {game_id} ({family}) - "
            f"round {game['game_state'].get('round', '?')}, "
            f"phase: {game['phase']}, your turn as {game['your_player']}"
        )
        try:
            action = strategy(game)
        except Exception:
            logger.exception(f"Strategy error for game {game_id}")
            return False

        result = self.move_recovery.submit(game, action).payload
        if result.get("game_over"):
            self.close_game_resources(str(game_id))
        if result.get("valid") is False:
            attempts = result.get("attempts_left")
            suffix = f" ({attempts} attempts left)" if attempts is not None else ""
            logger.warning(
                f"Game {game_id}: move rejected as invalid — "
                f"{result.get('error')!r}{suffix}"
            )
        else:
            logger.info(f"Move result: {result}")
        if result.get("lost_turn"):
            # The server has already advanced (or rejected) this turn.  The
            # caller must release this local assignment instead of retrying
            # the same action from a stale pending snapshot.
            return True
        if result.get("game_over"):
            logger.info(f"Game {game_id} finished! Result: {result.get('result')}")
            return True
        return False

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
        """Compatibility entrypoint delegated to the shared lifecycle."""
        return GameRunner(self).run(
            strategy,
            game_families=game_families,
            poll_interval=poll_interval,
            max_games=max_games,
            max_time=max_time,
            requeue=requeue,
            concurrency=concurrency,
            should_stop=should_stop,
            start_games=start_games,
            queue_depth=queue_depth,
            move_concurrency=move_concurrency,
            active_limit=active_limit,
        )
