"""The real HTTP transport, exercised without a network.

`urlopen` is replaced throughout. That is not a compromise: the behaviour worth
testing here is the *policy* -- which hosts are reachable, which failures are
retried, which are not, how a rate limit is honoured, whether the credential
can reach a URL -- and none of it needs a socket to be true.

The distinction the retry tests are really about: a timeout means the request
was never answered, so asking again is reasonable. A 400 means it *was*
answered, and asking again produces the same answer more slowly while hiding a
bug behind a delay.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from scripts.trading_lab.massive_http_transport import (
    ALLOWED_HOSTS, CAPTURE_USER_AGENT, HostNotAllowedError, MAX_RETRY_AFTER_SECONDS,
    MassiveHTTPTransport, RateLimitedError, TransportError, require_allowed_url)
from scripts.trading_lab.massive_provider import MassiveProviderError

SENTINEL = "sentinel-massive-key-do-not-use-1a2b3c4d5e6f"


class _Response(io.BytesIO):
    """Enough of an HTTP response for the transport to read."""

    def __init__(self, payload, status=200):
        raw = (payload if isinstance(payload, bytes)
               else json.dumps(payload, sort_keys=True).encode("utf-8"))
        super().__init__(raw)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, *, headers=None):
    return urllib.error.HTTPError(
        "https://api.massive.com/v1/stocks/bars", code, "boom",
        headers or {}, None)


def _transport(responses, **kwargs):
    """A transport whose opener yields the given responses or raises them."""
    calls = []
    queue = list(responses)

    def opener(request, timeout=None):
        calls.append({"url": request.full_url,
                      "method": request.get_method(),
                      "headers": dict(request.header_items()),
                      "timeout": timeout})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    sleeps = []
    transport = MassiveHTTPTransport(opener=opener, sleep=sleeps.append,
                                     **kwargs)
    transport.calls = calls
    transport.sleeps = sleeps
    return transport


# --- the host allowlist ----------------------------------------------------


def test_only_allowlisted_hosts_are_reachable():
    assert ALLOWED_HOSTS == ("api.massive.com",)
    require_allowed_url("https://api.massive.com/v1/stocks/bars")
    for url in ("https://evil.example.com/v1/stocks/bars",
                "https://api.massive.com.evil.example.com/v1",
                "https://massive.com/v1/stocks/bars",
                "https://api-massive.com/v1"):
        with pytest.raises(HostNotAllowedError):
            require_allowed_url(url)


def test_plain_http_is_refused():
    """https in the URL is not a guarantee, but its absence is a disqualifier."""
    with pytest.raises(HostNotAllowedError):
        require_allowed_url("http://api.massive.com/v1/stocks/bars")


def test_a_credential_shaped_query_string_is_refused():
    """A key in a query string lands in every access log on the path."""
    for query in ("apikey=abc", "api_key=abc", "token=abc", "key=abc",
                  "secret=abc"):
        with pytest.raises(TransportError) as error:
            require_allowed_url(f"https://api.massive.com/v1/stocks/bars?{query}")
        assert "Authorization header" in str(error.value)


def test_the_transport_only_ever_issues_a_get():
    transport = _transport([_Response({"bars": []})])
    transport.fetch("/v1/stocks/bars", {"symbol": "AAPL"}, {})
    assert transport.calls[0]["method"] == "GET"
    assert not hasattr(MassiveHTTPTransport, "post")
    assert not hasattr(MassiveHTTPTransport, "put")
    assert not hasattr(MassiveHTTPTransport, "delete")


def test_the_url_is_built_deterministically_from_sorted_parameters():
    transport = _transport([_Response({"bars": []}), _Response({"bars": []})])
    first = transport.build_url("/v1/stocks/bars", {"b": 2, "a": 1})
    second = transport.build_url("/v1/stocks/bars", {"a": 1, "b": 2})
    assert first == second
    assert first.endswith("?a=1&b=2")


# --- the credential --------------------------------------------------------


def test_the_credential_goes_in_a_header_and_never_in_the_url():
    transport = _transport([_Response({"bars": []})])
    transport.fetch("/v1/stocks/bars", {"symbol": "AAPL"},
                    {"Authorization": f"Bearer {SENTINEL}"})
    call = transport.calls[0]
    assert SENTINEL not in call["url"]
    # urllib title-cases header names.
    assert SENTINEL in call["headers"]["Authorization"]
    assert call["headers"]["User-agent"] == CAPTURE_USER_AGENT


def test_an_auth_failure_never_echoes_the_response_body():
    """A vendor may quote the Authorization header it just rejected."""
    for status in (401, 403):
        transport = _transport([_http_error(status)])
        with pytest.raises(MassiveProviderError) as error:
            transport.fetch("/v1/stocks/bars", {},
                            {"Authorization": f"Bearer {SENTINEL}"})
        assert SENTINEL not in str(error.value)
        assert SENTINEL not in repr(error.value)
        assert "HYPRL_MASSIVE_API_KEY" in str(error.value)


# --- retries ---------------------------------------------------------------


def test_a_network_failure_is_retried_and_then_succeeds():
    transport = _transport([
        urllib.error.URLError("connection reset"),
        _Response({"bars": [], "ok": True}),
    ])
    payload = transport.fetch("/v1/stocks/bars", {}, {}).payload
    assert payload["ok"] is True
    assert transport.stats.retries == 1
    assert transport.stats.requests == 1


def test_a_transient_status_is_retried():
    transport = _transport([_http_error(503), _Response({"bars": []})])
    transport.fetch("/v1/stocks/bars", {}, {})
    assert transport.stats.retries == 1


@pytest.mark.parametrize("status", [400, 404, 422])
def test_a_client_error_is_never_retried(status):
    """Asking again produces the same answer while hiding the bug in a delay."""
    transport = _transport([_http_error(status)])
    with pytest.raises(MassiveProviderError):
        transport.fetch("/v1/stocks/bars", {}, {})
    assert transport.stats.retries == 0


def test_a_body_that_is_not_json_is_never_retried():
    """A non-JSON body is an answer, not a missing one."""
    transport = _transport([_Response(b"<html>maintenance</html>")])
    with pytest.raises(MassiveProviderError) as error:
        transport.fetch("/v1/stocks/bars", {}, {})
    assert "not JSON" in str(error.value)
    assert transport.stats.retries == 0


def test_retries_are_bounded_and_then_the_request_fails():
    transport = _transport([urllib.error.URLError("down")] * 4)
    with pytest.raises(TransportError) as error:
        transport.fetch("/v1/stocks/bars", {}, {})
    assert "after 4 attempts" in str(error.value)
    assert transport.stats.retries == 3


# --- rate limits -----------------------------------------------------------


def test_a_rate_limit_is_counted_and_honoured():
    transport = _transport([
        _http_error(429, headers={"Retry-After": "2"}),
        _Response({"bars": []}),
    ])
    transport.fetch("/v1/stocks/bars", {}, {})
    assert transport.stats.rate_limits == 1
    assert 2 in transport.sleeps


def test_a_rate_limit_is_typed_so_a_capture_can_report_it():
    """Never confused with a data problem, and never parsed from a string."""
    transport = _transport([_http_error(429)] * 4)
    with pytest.raises(TransportError):
        transport.fetch("/v1/stocks/bars", {}, {})
    assert transport.stats.rate_limits == 4
    assert issubclass(RateLimitedError, TransportError)


def test_an_unreasonable_retry_after_stops_the_run_rather_than_sleeping():
    transport = _transport([
        _http_error(429, headers={"Retry-After": str(MAX_RETRY_AFTER_SECONDS + 1)}),
    ])
    with pytest.raises(TransportError) as error:
        transport.fetch("/v1/stocks/bars", {}, {})
    assert "rerun the capture later" in str(error.value)


def test_a_non_numeric_retry_after_falls_back_to_the_standard_backoff():
    transport = _transport([
        _http_error(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        _Response({"bars": []}),
    ])
    transport.fetch("/v1/stocks/bars", {}, {})
    assert transport.stats.rate_limits == 1


# --- raw bytes -------------------------------------------------------------


def test_the_caller_receives_the_bytes_before_anything_parsed_them():
    """A canonicalisation bug is only recoverable if the source survives."""
    payload = {"bars": [{"open": "1.00"}], "adjustment": "SPLIT_ADJUSTED"}
    transport = _transport([_Response(payload)])
    response = transport.fetch("/v1/stocks/bars", {}, {})
    assert response.raw == json.dumps(payload, sort_keys=True).encode("utf-8")
    assert response.payload == payload
    assert response.status == 200


def test_a_non_object_json_body_is_refused():
    transport = _transport([_Response(b"[1, 2, 3]")])
    with pytest.raises(MassiveProviderError):
        transport.fetch("/v1/stocks/bars", {}, {})


# --- the reported state ----------------------------------------------------


def test_the_transport_payload_says_what_it_is_and_never_what_it_carries():
    transport = _transport([_Response({"bars": []})])
    transport.fetch("/v1/stocks/bars", {}, {"Authorization": f"Bearer {SENTINEL}"})
    payload = transport.payload()
    assert payload["methods"] == ["GET"]
    assert payload["allowed_hosts"] == list(ALLOWED_HOSTS)
    assert payload["stats"]["requests"] == 1
    rendered = json.dumps(payload)
    assert SENTINEL not in rendered
    for banned in ("authorization", "api_key", "bearer"):
        assert banned not in rendered.lower()


def test_the_provider_contract_still_works_through_the_real_transport():
    """6D's interface is unchanged: request() returns the parsed payload."""
    transport = _transport([_Response({"bars": [], "adjustment": "RAW"})])
    assert transport.request("/v1/stocks/bars", {}, {}) == {
        "bars": [], "adjustment": "RAW"}


def test_the_no_network_default_is_untouched_by_this_module():
    """6D's offline provider must not have quietly acquired a socket."""
    from scripts.trading_lab.instrument_registry import PROVIDERS_V1
    from scripts.trading_lab.massive_provider import NoNetworkTransport

    provider = PROVIDERS_V1.resolve("massive-stocks-historical-v1")
    assert isinstance(provider.transport, NoNetworkTransport)
    assert provider.payload()["network_enabled"] is False
