"""DNS overrides for sandboxed / filtered resolvers."""

from __future__ import annotations

import ipaddress
import os
import socket
import subprocess
import threading
import time

HOST_IPS: dict[str, tuple[str, ...]] = {
    # Keep both Cloudflare edges as ordered candidates for connection-level
    # failover. DNS must not collapse this list to a single address.
    "glee-competition.com": ("104.21.59.229", "172.67.184.212"),
    "www.glee-competition.com": ("104.21.59.229", "172.67.184.212"),
    "api.x.ai": ("104.18.18.80", "104.18.19.80"),
    "auth.x.ai": ("104.18.18.80", "104.18.19.80"),
}

_patched = False
_en0_lock = threading.Lock()
_en0_cached: str | None = None
_en0_cached_at = 0.0
_EN0_CACHE_SECONDS = 5.0
_EN0_STALE_SECONDS = 300.0


def en0_ipv4() -> str | None:
    """Return the local en0 IPv4 address without proxy or DNS lookup."""
    global _en0_cached, _en0_cached_at
    now = time.monotonic()
    if _en0_cached and now - _en0_cached_at < _EN0_CACHE_SECONDS:
        return _en0_cached
    with _en0_lock:
        now = time.monotonic()
        if _en0_cached and now - _en0_cached_at < _EN0_CACHE_SECONDS:
            return _en0_cached
        try:
            value = subprocess.check_output(
                ["ipconfig", "getifaddr", "en0"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
            _en0_cached = str(ipaddress.IPv4Address(value))
            _en0_cached_at = now
            return _en0_cached
        except (OSError, ValueError, subprocess.SubprocessError):
            if _en0_cached and now - _en0_cached_at < _EN0_STALE_SECONDS:
                return _en0_cached
            return None


def patch_dns() -> None:
    global _patched
    if _patched or os.environ.get("GLEE_PATCH_DNS") != "1":
        return
    original = socket.getaddrinfo

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        ips = HOST_IPS.get(str(host).lower())
        if not ips:
            return original(host, port, family, type, proto, flags)
        answers = []
        last_error: OSError | None = None
        for ip in ips:
            try:
                answers.extend(original(ip, port, family, type, proto, flags))
            except OSError as exc:
                last_error = exc
        if answers:
            return answers
        if last_error is not None:
            raise last_error
        return original(host, port, family, type, proto, flags)

    socket.getaddrinfo = getaddrinfo  # type: ignore[method-assign]
    _patched = True
