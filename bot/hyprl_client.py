"""HTTP client for interacting with the HyprL V2 API."""

from __future__ import annotations

from typing import Any, Iterable, List

import httpx

from bot.config import BotSettings


class HyprlAPIError(RuntimeError):
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self.payload = payload
        message = f"HyprL API error ({status_code}): {payload}"
        super().__init__(message)


class HyprlClient:
    def __init__(self, settings: BotSettings, *, timeout: float = 30.0):
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.hyprl_api_base,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {settings.hyprl_api_token}",
                "X-HYPRL-Client": "discord-bot",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, *, json: Any | None = None, params: dict | None = None) -> Any:
        try:
            response = await self._client.request(method, path, json=json, params=params)
        except httpx.HTTPError as exc:
            raise HyprlAPIError(-1, f"HTTP client error: {exc}") from exc
        if response.status_code >= 400:
            payload: Any
            try:
                payload = response.json()
            except ValueError:
                payload = response.text
            raise HyprlAPIError(response.status_code, payload)
        if response.content:
            if "application/json" in response.headers.get("content-type", ""):
                return response.json()
            return response.text
        return None

    async def get_health(self) -> dict:
        return await self._request("GET", "/health")

    async def get_usage(self) -> dict:
        return await self._request("GET", "/v2/usage")

    async def get_predict_summary(self) -> dict:
        return await self._request("GET", "/v2/predict/summary")

    async def post_predict(self, symbols: Iterable[str]) -> dict:
        payload = {
            "symbols": [sym.upper() for sym in symbols],
            "interval": "1m",
            "features": [],
            "threshold": 0.55,
            "risk_pct": 0.1,
        }
        return await self._request("POST", "/v2/predict", json=payload)

    async def start_session(
        self,
        *,
        symbols: List[str],
        interval: str,
        threshold: float,
        risk_pct: float,
        kill_switch_dd: float = 0.30,
        enable_paper: bool = False,
    ) -> dict:
        payload = {
            "symbols": [sym.upper() for sym in symbols],
            "interval": interval,
            "threshold": threshold,
            "risk_pct": risk_pct,
            "kill_switch_dd": kill_switch_dd,
            "resume_session": None,
            "enable_paper": enable_paper,
        }
        return await self._request("POST", "/v2/sessions", json=payload)

    async def get_session_status(self, session_id: str) -> dict:
        return await self._request("GET", f"/v2/sessions/{session_id}")

    async def get_session_report(self, session_id: str) -> dict:
        return await self._request("GET", f"/v2/sessions/{session_id}/report")

    async def list_sessions(self, session_ids: Iterable[str]) -> List[dict]:
        sessions: List[dict] = []
        for session_id in session_ids:
            session_id = session_id.strip()
            if not session_id:
                continue
            try:
                sessions.append(await self.get_session_status(session_id))
            except HyprlAPIError as exc:
                sessions.append({"session_id": session_id, "error": exc.payload, "status": "unknown"})
        return sessions

    async def start_autorank(self, payload: dict) -> dict:
        return await self._request("POST", "/v2/autorank/start", json=payload)

    @property
    def api_base(self) -> str:
        return self._settings.hyprl_api_base


__all__ = ["HyprlClient", "HyprlAPIError"]
