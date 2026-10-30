import socket
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from glee_agent import net
from glee_runtime import transport
from glee_runtime.transport import (
    DirectHTTPAdapter,
    NetworkUnavailable,
    ApiTransport,
    attach_direct_adapter,
)


def test_patch_dns_keeps_both_cloudflare_addresses(monkeypatch):
    calls = []

    def fake(host, port, family=0, type=0, proto=0, flags=0):
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    monkeypatch.setattr(net, "_patched", False)
    monkeypatch.setenv("GLEE_PATCH_DNS", "1")
    net.patch_dns()

    answers = socket.getaddrinfo("glee-competition.com", 443)
    assert calls == ["104.21.59.229", "172.67.184.212"]
    assert [row[4][0] for row in answers] == list(net.HOST_IPS["glee-competition.com"])


def test_en0_ipv4_caches_and_survives_transient_lookup_failure(monkeypatch):
    now = [10.0]
    calls = []
    monkeypatch.setattr(net, "_en0_cached", None)
    monkeypatch.setattr(net, "_en0_cached_at", 0.0)
    monkeypatch.setattr(net.time, "monotonic", lambda: now[0])

    def lookup(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return "192.168.0.101\n"
        raise subprocess.TimeoutExpired("ipconfig", 2)

    import subprocess
    monkeypatch.setattr(net.subprocess, "check_output", lookup)
    assert net.en0_ipv4() == "192.168.0.101"
    assert net.en0_ipv4() == "192.168.0.101"
    assert len(calls) == 1
    now[0] += 6
    assert net.en0_ipv4() == "192.168.0.101"
    assert len(calls) == 2


def test_direct_adapter_ignores_utun_environment_override(monkeypatch):
    monkeypatch.setenv("GLEE_DIRECT_INTERFACE", "utun0")
    monkeypatch.setattr(transport, "en0_ipv4", lambda: "192.0.2.10")

    session = attach_direct_adapter(requests.Session())
    adapter = session.get_adapter("https://example.test")

    assert isinstance(adapter, DirectHTTPAdapter)
    assert adapter.interface == "en0"
    assert adapter.source_ip == "192.0.2.10"
    assert adapter.poolmanager.connection_pool_kw["source_address"] == (
        "192.0.2.10",
        0,
    )


def test_macos_adapter_binds_each_new_socket_to_en0(monkeypatch):
    monkeypatch.setattr(transport.sys, "platform", "darwin")
    monkeypatch.setattr(transport.socket, "if_nametoindex", lambda name: 25)
    monkeypatch.setattr(transport.socket, "IP_BOUND_IF", 25, raising=False)
    monkeypatch.setattr(transport, "en0_ipv4", lambda: "192.0.2.10")

    session = attach_direct_adapter(requests.Session())
    adapter = session.get_adapter("https://example.test")
    options = adapter.poolmanager.connection_pool_kw["socket_options"]
    assert options == [(transport.socket.IPPROTO_IP, 25, 25)]

    events = []

    class FakeSocket:
        def __init__(self, *_args):
            pass

        def setsockopt(self, *args):
            events.append(("setsockopt", *args))

        def settimeout(self, _timeout):
            pass

        def connect(self, address):
            events.append(("connect", address))

        def close(self):
            pass

    monkeypatch.setattr(
        transport.socket,
        "getaddrinfo",
        lambda host, port, family, socktype: [
            (transport.socket.AF_INET, socktype, 6, "", (host, port))
        ],
    )
    monkeypatch.setattr(transport.socket, "socket", FakeSocket)

    import urllib3.util.connection as uconn

    uconn.create_connection(("192.0.2.20", 443), socket_options=options)
    uconn.create_connection(("192.0.2.21", 443), socket_options=options)
    assert events == [
        ("setsockopt", transport.socket.IPPROTO_IP, 25, 25),
        ("connect", ("192.0.2.20", 443)),
        ("setsockopt", transport.socket.IPPROTO_IP, 25, 25),
        ("connect", ("192.0.2.21", 443)),
    ]


def test_linux_adapter_keeps_existing_socket_behavior(monkeypatch):
    monkeypatch.setattr(transport.sys, "platform", "linux")
    monkeypatch.setattr(transport, "en0_ipv4", lambda: "192.0.2.10")
    session = attach_direct_adapter(requests.Session())
    adapter = session.get_adapter("https://example.test")
    assert "socket_options" not in adapter.poolmanager.connection_pool_kw


class _ResponseSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.closed_calls = 0
        self.requests = 0

    def close(self):
        self.closed_calls += 1
        super().close()

    def request(self, method, url, **kwargs):
        del method, url, kwargs
        self.requests += 1
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.headers = {}
        return response
