from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.message import Message
from urllib.request import Request

import pytest

from crypto_news.collection import CollectionValidationError
from crypto_news.egress import DEFAULT_EGRESS_POLICY, EgressDeniedError
from crypto_news.transport import HttpTransport
from crypto_news.transport import _AuthorizedRedirectHandler


UTC = timezone.utc
BASE = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)


def test_default_redirect_handler_never_creates_a_follow_up_request() -> None:
    handler = _AuthorizedRedirectHandler(DEFAULT_EGRESS_POLICY)
    request = Request("https://www.sec.gov/news/pressreleases.rss")

    assert handler.redirect_request(
        request,
        object(),
        302,
        "Found",
        {},
        "https://api.coinbase.com/redirect-target",
    ) is None
    with pytest.raises(EgressDeniedError):
        handler.redirect_request(
            request,
            object(),
            302,
            "Found",
            {},
            "https://evil.example/redirect-target",
        )


class FakeResponse:
    def __init__(self, body: bytes, url: str, content_type: str = "application/rss+xml"):
        self._body = body
        self._url = url
        self.status = 200
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        return self._body[:size]

    def geturl(self) -> str:
        return self._url


class FakeOpener:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls: list[tuple[object, float]] = []

    def open(self, request: object, timeout: float) -> FakeResponse:
        self.calls.append((request, timeout))
        return self.response


def test_transport_authorizes_before_fetch_and_returns_bounded_result() -> None:
    url = "https://www.sec.gov/news/pressreleases.rss"
    opener = FakeOpener(FakeResponse(b"<rss/>", url))
    transport = HttpTransport(
        policy=DEFAULT_EGRESS_POLICY,
        user_agent="HyprL research collector contact@example.invalid",
        opener=opener,
        clock=lambda: BASE + timedelta(seconds=1),
    )
    result = transport.fetch(
        source_id="sec_press_releases",
        endpoint_url=url,
        first_seen_at=BASE,
    )
    assert result.body == b"<rss/>"
    assert result.retrieved_at == BASE + timedelta(seconds=1)
    request, timeout = opener.calls[0]
    assert timeout == 10.0
    assert request.get_header("User-agent") is not None


def test_transport_rejects_unallowlisted_url_without_opening_it() -> None:
    opener = FakeOpener(FakeResponse(b"ignored", "https://evil.example/feed"))
    transport = HttpTransport(
        policy=DEFAULT_EGRESS_POLICY,
        user_agent="research@example.invalid",
        opener=opener,
    )
    with pytest.raises(EgressDeniedError):
        transport.fetch(
            source_id="bad",
            endpoint_url="https://evil.example/feed",
            first_seen_at=BASE,
        )
    assert opener.calls == []


def test_transport_rejects_oversized_or_redirected_payloads() -> None:
    url = "https://www.sec.gov/news/pressreleases.rss"
    oversized = HttpTransport(
        policy=DEFAULT_EGRESS_POLICY,
        user_agent="research@example.invalid",
        opener=FakeOpener(FakeResponse(b"123456", url)),
        max_response_bytes=5,
    )
    with pytest.raises(CollectionValidationError, match="maximum size"):
        oversized.fetch(
            source_id="sec_press_releases",
            endpoint_url=url,
            first_seen_at=BASE,
        )

    redirected = HttpTransport(
        policy=DEFAULT_EGRESS_POLICY,
        user_agent="research@example.invalid",
        opener=FakeOpener(FakeResponse(b"<rss/>", "https://evil.example/feed")),
    )
    with pytest.raises(EgressDeniedError):
        redirected.fetch(
            source_id="sec_press_releases",
            endpoint_url=url,
            first_seen_at=BASE,
        )

    same_host_redirect = HttpTransport(
        policy=DEFAULT_EGRESS_POLICY,
        user_agent="research@example.invalid",
        opener=FakeOpener(
            FakeResponse(b"<rss/>", "https://www.sec.gov/news/another-feed")
        ),
    )
    with pytest.raises(CollectionValidationError, match="redirect"):
        same_host_redirect.fetch(
            source_id="sec_press_releases",
            endpoint_url=url,
            first_seen_at=BASE,
        )
