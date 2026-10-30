"""HTTP transport shared by all GLEE game families."""

from __future__ import annotations

import email.utils
import ipaddress
import logging
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import requests
from requests.adapters import HTTPAdapter

from glee_agent.net import HOST_IPS, en0_ipv4
from .rate_limit import KeyRequestBudget, request_priority

logger = logging.getLogger("glee_runtime.transport")

_FAILOVER_INSTALLED = False


class TransportError(RuntimeError):
    """Base class for transport-level failures."""


class NetworkUnavailable(requests.ConnectionError, TransportError):
    """The API could not be reached through the configured route."""


def connect_with_failover(address, connect):
    """Try every pinned IP, then report that the network must be waited on."""
    host, port = address
    ips = HOST_IPS.get(str(host).lower())
    if not ips:
        return connect(address)
    last_error: OSError | None = None
    for ip in ips:
        try:
            return connect((ip, port))
        except OSError as exc:
            last_error = exc
    assert last_error is not None
    raise NetworkUnavailable(
        f"network unavailable: all configured addresses for {host!r} failed; "
        "waiting for en0 connectivity"
    ) from last_error


def _install_connect_failover() -> None:
    global _FAILOVER_INSTALLED
    if _FAILOVER_INSTALLED:
        return
    import urllib3.util.connection as uconn

    original = uconn.create_connection

    def create_connection(address, *args, **kwargs):
        return connect_with_failover(
            address, lambda addr: original(addr, *args, **kwargs)
        )

    uconn.create_connection = create_connection  # type: ignore[method-assign]
    _FAILOVER_INSTALLED = True


# Competition clients use the operating system's default route and DNS.


def _retry_after(value: str | None, default: float) -> float:
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


def interface_ipv4(interface: str) -> str | None:
    """Return the macOS interface IPv4 without consulting proxy settings."""
    try:
        value = subprocess.check_output(
            ["ipconfig", "getifaddr", interface],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        return str(ipaddress.IPv4Address(value))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _interface_socket_options(interface: str) -> tuple[tuple[int, int, int], ...]:
    """Bind new macOS sockets to en0 before urllib3 connects them."""
    if sys.platform != "darwin" or interface != "en0":
        return ()
    try:
        index = socket.if_nametoindex(interface)
    except OSError:
        return ()
    return ((socket.IPPROTO_IP, getattr(socket, "IP_BOUND_IF", 25), index),)


class DirectHTTPAdapter(HTTPAdapter):
    def __init__(
        self,
        interface: str,
        source_ip: str | None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.interface = interface
        self.source_ip = source_ip
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        if self.source_ip is not None:
            pool_kwargs["source_address"] = (self.source_ip, 0)
        socket_options = list(pool_kwargs.get("socket_options") or ())
        socket_options.extend(_interface_socket_options(self.interface))
        if socket_options:
            pool_kwargs["socket_options"] = socket_options
        return super().init_poolmanager(connections, maxsize, block, **pool_kwargs)

    def send(self, request, *args, **kwargs):
        if self.source_ip is None:
            raise NetworkUnavailable(
                f"direct interface {self.interface!r} has no IPv4 address"
            )
        return super().send(request, *args, **kwargs)


def attach_direct_adapter(
    session: requests.Session,
    *,
    authorization: str = "",
    trust_env: bool = False,
    interface: str | None = None,
    source_ip: str | None = None,
    pool_size: int = 10,
    _source_ip_explicit: bool = False,
) -> requests.Session:
    """Same en0 / Cloudflare-IP path as competition. Never enable env proxies."""
    session.trust_env = trust_env
    if authorization:
        session.headers["Authorization"] = authorization
    # Do not let an environment variable route direct traffic through utun.
    iface = interface or "en0"
    if not _source_ip_explicit and source_ip is None:
        local_ip = en0_ipv4() if iface == "en0" else interface_ipv4(iface)
    else:
        local_ip = source_ip
    adapter = DirectHTTPAdapter(
        iface,
        local_ip,
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@dataclass(frozen=True)
class RequestTimeouts:
    connect: float = 2.0
    read: float = 5.0

    def as_requests(self) -> tuple[float, float]:
        return self.connect, self.read


@dataclass(frozen=True)
class RequestPolicy:
    attempts: int = 3
    backoff_seconds: tuple[float, ...] = (2.0, 4.0)

    def delay(self, attempt: int) -> float:
        if not self.backoff_seconds:
            return 0.0
        return self.backoff_seconds[min(attempt, len(self.backoff_seconds) - 1)]


class ApiTransport:
    """Thread-local Sessions with explicit proxy and retry policy.

    The compatibility client may retain its historical retry policy. New
    move recovery passes ``attempts=1`` because a POST response can be lost
    after the server accepted the action.
    """

    def __init__(
        self,
        authorization: str,
        *,
        timeout: float | tuple[float, float] = RequestTimeouts().as_requests(),
        pool_size: int = 10,
        trust_env: bool = False,
        direct_interface: str | None = None,
        session_factory: Callable[[], requests.Session] = requests.Session,
        request_budget: KeyRequestBudget | None = None,
    ) -> None:
        self.authorization = authorization
        self.timeout = timeout if isinstance(timeout, tuple) else (timeout, timeout)
        self.pool_size = max(1, pool_size)
        self.trust_env = False
        # Kept in the signature for callers from older strategy checkouts.
        # Competition traffic always uses the operating system's default route.
        self.direct_interface = None
        self.direct_source_ip = None
        self._had_source_ip = False
        self._session_factory = session_factory
        self._local = threading.local()
        self._sessions_lock = threading.RLock()
        self._thread_sessions: dict[int, requests.Session] = {}
        self._game_sessions: dict[str, requests.Session] = {}
        self.request_budget = request_budget or (
            KeyRequestBudget(authorization) if authorization else None
        )
        with self._sessions_lock:
            self.default_session = self._new_session()
            self._remember_thread_session_locked(self.default_session)

    def _new_session(self) -> requests.Session:
        session = self._session_factory()
        session.trust_env = False
        if self.authorization:
            session.headers["Authorization"] = self.authorization
        adapter = HTTPAdapter(
            pool_connections=self.pool_size,
            pool_maxsize=self.pool_size,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _remember_thread_session_locked(self, session: requests.Session) -> None:
        self._local.session = session
        self._thread_sessions[threading.get_ident()] = session

    def _all_sessions_locked(self) -> list[requests.Session]:
        sessions: dict[int, requests.Session] = {}
        for session in (
            self.default_session,
            *self._thread_sessions.values(),
            *self._game_sessions.values(),
        ):
            sessions[id(session)] = session
        return list(sessions.values())

    def _has_stale_adapter_locked(self, source_ip: str | None) -> bool:
        for session in self._all_sessions_locked():
            adapter = session.get_adapter("https://")
            if hasattr(adapter, "source_ip") and adapter.source_ip != source_ip:
                return True
        return False

    def _rebind_locked(self, source_ip: str | None) -> None:
        old_sessions = self._all_sessions_locked()
        self.direct_source_ip = source_ip
        self._had_source_ip = self._had_source_ip or source_ip is not None
        self._thread_sessions.clear()
        self._game_sessions.clear()
        for session in old_sessions:
            session.close()
        self.default_session = self._new_session()
        self._remember_thread_session_locked(self.default_session)

    def _refresh_source(self) -> str | None:
        if self.direct_interface is None:
            return None
        with self._sessions_lock:
            source_ip = interface_ipv4(self.direct_interface)
            if source_ip != self.direct_source_ip or self._has_stale_adapter_locked(
                source_ip
            ):
                self._rebind_locked(source_ip)
        return source_ip

    def _session_for_request(
        self, explicit_session: requests.Session | None
    ) -> requests.Session:
        source_ip = self._refresh_source()
        with self._sessions_lock:
            if self.direct_interface is not None and source_ip is None and self._had_source_ip:
                raise NetworkUnavailable(
                    "network unavailable: en0 has no IPv4 address; "
                    "waiting for en0 connectivity"
                )

            if explicit_session is not None:
                active = any(
                    explicit_session is session
                    for session in self._all_sessions_locked()
                )
                adapter = explicit_session.get_adapter("https://")
                if active and getattr(adapter, "source_ip", source_ip) == source_ip:
                    return explicit_session

            thread_id = threading.get_ident()
            session = self._thread_sessions.get(thread_id)
            if session is None:
                session = self._new_session()
                self._remember_thread_session_locked(session)
            return session

    def session(self) -> requests.Session:
        self._refresh_source()
        with self._sessions_lock:
            session = self._thread_sessions.get(threading.get_ident())
            if session is None:
                session = self._new_session()
                self._remember_thread_session_locked(session)
            return session

    def game_session(self, game_id: str) -> requests.Session:
        if self.direct_interface is not None:
            self._refresh_source()
        with self._sessions_lock:
            return self._game_sessions.setdefault(game_id, self._new_session())

    def close_game_session(self, game_id: str) -> None:
        with self._sessions_lock:
            session = self._game_sessions.pop(game_id, None)
        if session is not None:
            session.close()

    def request_json(
        self,
        method: str,
        url: str,
        *,
        raise_for_response: Callable[[requests.Response], None],
        attempts: int = 3,
        retry_post_connection_errors: bool = True,
        session: requests.Session | None = None,
        **kwargs: Any,
    ) -> dict | list:
        kwargs.setdefault("timeout", self.timeout)
        response: requests.Response | None = None
        policy = RequestPolicy(attempts=max(1, attempts))
        verb = method.upper()
        for attempt in range(policy.attempts):
            try:
                request_session = self._session_for_request(session)
            except NetworkUnavailable:
                # The route cannot recover within this HTTP attempt. Let the
                # runner enter NETWORK_WAITING without spending request budget.
                raise
            if self.request_budget is not None:
                self.request_budget.acquire(request_priority(verb, url))
            try:
                response = request_session.request(verb, url, **kwargs)
            except requests.RequestException as exc:
                if isinstance(exc, NetworkUnavailable):
                    raise
                safe = verb in {"GET", "DELETE"} or (
                    retry_post_connection_errors
                    and isinstance(exc, requests.ConnectionError)
                )
                if not safe or attempt + 1 >= policy.attempts:
                    raise
                delay = policy.delay(attempt)
                logger.warning("Request error: %s. Retrying in %ss...", exc, delay)
                time.sleep(delay)
                continue
            if response.status_code == 429 and attempt + 1 < policy.attempts:
                delay = min(
                    _retry_after(response.headers.get("Retry-After"), policy.delay(attempt)),
                    30.0,
                )
                logger.warning("Rate limited. Waiting %ss...", delay)
                time.sleep(delay)
                continue
            break
        assert response is not None
        raise_for_response(response)
        if response.status_code == 204:
            return {}
        return response.json()
