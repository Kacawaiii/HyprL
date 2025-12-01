from __future__ import annotations

from fastapi import FastAPI

from hyprl_api.config import get_settings
from hyprl_api.db import init_db
from hyprl_api.v2 import router as v2_router


settings = get_settings()
app = FastAPI(title=settings.api_title, version=settings.api_version)
init_db()


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    """Simple readiness probe."""

    return {"status": "ok"}


app.include_router(v2_router)
