import asyncio
import socket

import aiohttp
import pytest

from scripts.alert_router.model import AlertError
from scripts.alert_router.security import HTTPS


def test_transport_pins_checked_dns_keeps_tls_hostname_and_refuses_redirects(monkeypatch):
    resolved = []
    connections = []

    async def dns(host):
        resolved.append(host)
        return ["93.184.216.34"]

    class Connector:
        def __init__(self, *, resolver, use_dns_cache):
            self.resolver = resolver
            assert use_dns_cache is False

    class Response:
        status = 302
        headers = {"Location": "http://169.254.169.254/latest/meta-data"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session:
        def __init__(self, *, connector, trust_env, timeout):
            self.connector = connector
            assert trust_env is False
            assert timeout.total == 5

        async def __aenter__(self):
            pins = await self.connector.resolver.resolve("example.org", 443, socket.AF_UNSPEC)
            connections.extend(pins)
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, url, *, data, headers, allow_redirects):
            assert url == "https://example.org/hook"
            assert allow_redirects is False
            return Response()

    monkeypatch.setattr(aiohttp, "TCPConnector", Connector)
    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    receipt = asyncio.run(HTTPS(resolver=dns).post("https://example.org/hook", b"{}", {}))
    assert receipt.status == 302
    assert resolved == ["example.org"]
    assert connections[0]["host"] == "93.184.216.34"
    assert connections[0]["hostname"] == "example.org"


def test_rebinding_is_rechecked_on_every_attempt_before_any_connection(monkeypatch):
    answers = iter([["93.184.216.34"], ["127.0.0.1"]])

    async def dns(host):
        return next(answers)

    # A resolver's first public answer cannot authorize a later private one.
    from scripts.alert_router.security import SafeURL

    endpoint = SafeURL.parse("https://example.org/hook")
    endpoint.validate_addresses(asyncio.run(dns(endpoint.host)))
    connected = []
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: connected.append(kw))
    with pytest.raises(AlertError, match="SSRF_BLOCKED"):
        asyncio.run(HTTPS(resolver=dns).post(endpoint.url, b"{}", {}))
    assert connected == []


def test_literal_metadata_ip_blocked():
    async def dns(host):
        return [host]

    with pytest.raises(AlertError, match="SSRF_BLOCKED"):
        asyncio.run(HTTPS(resolver=dns).post("https://169.254.169.254/latest", b"{}", {}))
