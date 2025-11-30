from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from portal.config import PortalSettings
from portal.hyprl_client import HyprlAPIError, PortalHyprlClient


class FakeResponse:
    def __init__(self, *, status_code: int, json_data: Any | None = None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        if json_data is not None:
            self.text = json.dumps(json_data)
            self.content = self.text.encode("utf-8")
            self.headers = {"content-type": "application/json"}
        else:
            self.content = text.encode("utf-8")
            self.headers = {"content-type": "text/plain"}

    def json(self) -> Any:
        if self._json_data is None:
            raise ValueError("No JSON payload")
        return self._json_data


def build_client(monkeypatch: pytest.MonkeyPatch, handler: Callable[[str, str], FakeResponse]) -> PortalHyprlClient:
    def fake_request(self: httpx.Client, method: str, path: str, json: Any | None = None, params: dict | None = None):
        return handler(method, path)

    monkeypatch.setattr(httpx.Client, "request", fake_request, raising=False)
    settings = PortalSettings(api_base="http://hyprl.local", api_token="token", title="Portal")
    return PortalHyprlClient(settings)


def test_client_get_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(method: str, path: str) -> FakeResponse:
        assert method == "GET" and path == "/v2/usage"
        return FakeResponse(status_code=200, json_data={"credits_total": 100, "credits_remaining": 80})

    client = build_client(monkeypatch, handler)
    usage = client.get_usage()
    assert usage["credits_total"] == 100
    assert usage["credits_remaining"] == 80
    client.close()


def test_client_session_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(method: str, path: str) -> FakeResponse:
        assert method == "GET"
        assert path == "/v2/sessions/missing"
        return FakeResponse(status_code=404, json_data={"detail": "not found"})

    client = build_client(monkeypatch, handler)
    with pytest.raises(HyprlAPIError) as exc:
        client.get_session_status("missing")
    assert exc.value.status_code == 404
    assert exc.value.payload == {"detail": "not found"}
    client.close()
