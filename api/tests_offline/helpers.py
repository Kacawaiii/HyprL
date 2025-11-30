"""Testing utilities for invoking FastAPI dependencies without HTTP clients."""

from __future__ import annotations

import os
from typing import Dict

from starlette.requests import Request

os.environ.setdefault("HYPRL_DEV_TOKEN", "hyprl_dev_token_123")


def build_request(
    *,
    token: str | None = None,
    headers: Dict[str, str] | None = None,
    include_auth: bool = True,
    method: str = "POST",
    path: str = "/v2/predict",
) -> Request:
    merged: Dict[str, str] = {}
    if include_auth:
        used_token = token or os.environ["HYPRL_DEV_TOKEN"]
        merged["authorization"] = f"Bearer {used_token}"
    if headers:
        merged.update({k.lower(): v for k, v in headers.items()})
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(key.encode(), value.encode()) for key, value in merged.items()],
    }
    return Request(scope)
