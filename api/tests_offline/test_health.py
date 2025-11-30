import asyncio

import httpx

from api.app import app


def test_health_endpoint_ok() -> None:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get("/health")

    response = asyncio.run(_run())
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers.get("X-HYPRL-Version") == "2"
