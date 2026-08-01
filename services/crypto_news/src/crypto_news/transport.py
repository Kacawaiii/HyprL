"""Bounded HTTPS transport; the only Phase 2 module with network capability."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from crypto_news.collection import CollectionValidationError, FetchResult
from crypto_news.egress import DEFAULT_EGRESS_POLICY, EgressPolicy


class _Response(Protocol):
    status: int
    headers: object

    def __enter__(self) -> _Response: ...

    def __exit__(self, *args: object) -> None: ...

    def read(self, size: int) -> bytes: ...

    def geturl(self) -> str: ...


class _Opener(Protocol):
    def open(self, request: Request, timeout: float) -> _Response: ...


class _AuthorizedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, policy: EgressPolicy):
        super().__init__()
        self._policy = policy

    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> Request | None:
        self._policy.authorize(newurl)
        return None


class HttpTransport:
    """Fetch allowlisted HTTPS responses with redirects forbidden and size bounded."""

    def __init__(
        self,
        *,
        policy: EgressPolicy = DEFAULT_EGRESS_POLICY,
        user_agent: str,
        opener: _Opener | None = None,
        clock: Callable[[], datetime] | None = None,
        timeout_seconds: float = 10.0,
        max_response_bytes: int = 2 * 1024 * 1024,
    ):
        if not isinstance(user_agent, str) or not user_agent.strip():
            raise ValueError("an explicit collection user_agent is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 1 <= max_response_bytes <= 10 * 1024 * 1024:
            raise ValueError("max_response_bytes must be between 1 byte and 10 MiB")
        self.policy = policy
        self.user_agent = user_agent.strip()
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = max_response_bytes
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.opener = opener or build_opener(
            ProxyHandler({}),
            _AuthorizedRedirectHandler(policy),
        )

    def fetch(
        self,
        *,
        source_id: str,
        endpoint_url: str,
        first_seen_at: datetime,
    ) -> FetchResult:
        authorized_url = self.policy.authorize(endpoint_url)
        request = Request(
            authorized_url,
            headers={
                "Accept": "application/rss+xml, application/atom+xml, application/json",
                "User-Agent": self.user_agent,
            },
            method="GET",
        )
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                final_url = self.policy.authorize(response.geturl())
                if final_url != authorized_url:
                    raise CollectionValidationError("source redirects are forbidden")
                status = int(response.status)
                if status != 200:
                    raise CollectionValidationError(
                        f"source returned unexpected HTTP status: {status}"
                    )
                body = response.read(self.max_response_bytes + 1)
                if len(body) > self.max_response_bytes:
                    raise CollectionValidationError("response exceeds maximum size")
                try:
                    media_type = response.headers.get_content_type()
                except AttributeError:
                    media_type = "application/octet-stream"
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise CollectionValidationError("source fetch failed") from exc
        return FetchResult(
            source_id=source_id,
            endpoint_url=final_url,
            media_type=media_type,
            first_seen_at=first_seen_at,
            retrieved_at=self.clock(),
            body=body,
        )
