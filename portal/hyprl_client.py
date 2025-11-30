"""Synchronous HTTP client for the HyprL V2 REST API."""

from __future__ import annotations

from typing import Any, Iterable

import httpx

from portal.config import PortalSettings


class HyprlAPIError(RuntimeError):
    """Raised when the HyprL API responds with an error status."""

    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"HyprL API error ({status_code})")


class PortalHyprlClient:
    """Thin wrapper over httpx.Client with HyprL-specific helpers."""

    def __init__(self, settings: PortalSettings, *, timeout: float = 10.0):
        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.api_base,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {settings.api_token}",
                "X-HYPRL-Client": "portal",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _request(self, method: str, path: str, *, json: Any | None = None, params: dict | None = None) -> Any:
        if not path.startswith("/"):
            path = f"/{path}"
        response = self._client.request(method, path, json=json, params=params)
        if response.status_code >= 400:
            try:
                payload = response.json()
            except ValueError:
                payload = response.text
            raise HyprlAPIError(response.status_code, payload)
        if not response.content:
            return None
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        return response.text

    def get_usage(self) -> dict:
        return self._request("GET", "/v2/usage")

    def predict(self, symbols: list[str], threshold: float | None = None, risk_pct: float | None = None) -> dict:
        payload: dict[str, Any] = {"symbols": [sym.upper() for sym in symbols]}
        if threshold is not None:
            payload["threshold"] = threshold
        if risk_pct is not None:
            payload["risk_pct"] = risk_pct
        return self._request("POST", "/v2/predict", json=payload)

    def get_predict_summary(self) -> dict:
        return self._request("GET", "/v2/predict/summary")

    def post_predict(self, symbols: Iterable[str]) -> dict:
        """Backward compatible shim around predict()."""
        return self.predict([sym.upper() for sym in symbols])

    def start_session(self, payload: dict) -> dict:
        return self._request("POST", "/v2/sessions", json=payload)

    def get_session_status(self, session_id: str) -> dict:
        return self._request("GET", f"/v2/sessions/{session_id}")

    def get_session_report(self, session_id: str) -> dict:
        return self._request("GET", f"/v2/sessions/{session_id}/report")

    def start_autorank(self, payload: dict) -> dict:
        return self._request("POST", "/v2/autorank/start", json=payload)

    def get_autorank_status(self, autorank_id: str) -> dict:
        return self._request("GET", f"/v2/autorank/{autorank_id}")

    def create_token(self, request: dict) -> dict:
        return self._request("POST", "/v2/tokens", json=request)

    def revoke_token(self, token_id: str) -> None:
        self._request("DELETE", f"/v2/tokens/{token_id}")


__all__ = ["PortalHyprlClient", "HyprlAPIError"]
