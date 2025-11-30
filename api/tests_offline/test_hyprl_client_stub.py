from __future__ import annotations

import json

import pytest

from bot.config import BotSettings
from bot.hyprl_client import HyprlAPIError, HyprlClient


class DummyResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.headers = {"content-type": "application/json"}
        self.content = json.dumps(payload).encode()

    def json(self):
        return self._payload

    @property
    def text(self):
        return json.dumps(self._payload)


class DummyAsyncClient:
    def __init__(self, *, base_url: str, headers: dict, timeout: float):
        self.base_url = base_url
        self.headers = headers
        self.routes: dict[tuple[str, str], DummyResponse] = {}

    async def request(self, method: str, path: str, json=None, params=None):
        key = (method.upper(), path)
        if key not in self.routes:
            raise RuntimeError(f"Route {key} not stubbed")
        return self.routes[key]

    async def aclose(self):  # pragma: no cover - nothing to close
        return None


def test_hyprl_client_usage(monkeypatch):
    dummy = DummyAsyncClient(base_url="http://test", headers={}, timeout=10)
    dummy.routes[("GET", "/v2/usage")] = DummyResponse(200, {"credits_remaining": 10, "credits_total": 20, "by_endpoint": {}})

    monkeypatch.setattr("bot.hyprl_client.httpx.AsyncClient", lambda **kwargs: dummy)
    client = HyprlClient(BotSettings(discord_token="d", hyprl_api_base="http://test", hyprl_api_token="x"))

    usage = asyncio_run(client.get_usage())
    assert usage["credits_remaining"] == 10
    asyncio_run(client.close())


def test_hyprl_client_predict_error(monkeypatch):
    dummy = DummyAsyncClient(base_url="http://test", headers={}, timeout=10)
    dummy.routes[("POST", "/v2/predict")] = DummyResponse(402, {"error": "insufficient_credits"})

    monkeypatch.setattr("bot.hyprl_client.httpx.AsyncClient", lambda **kwargs: dummy)
    client = HyprlClient(BotSettings(discord_token="d", hyprl_api_base="http://test", hyprl_api_token="x"))

    with pytest.raises(HyprlAPIError) as exc:
        asyncio_run(client.post_predict(["AAPL"]))
    assert exc.value.status_code == 402
    asyncio_run(client.close())


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
