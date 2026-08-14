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
    ALLOWED_HOSTS, CAPTURE_USER_AGENT, HostNotAllowedError,
    MAX_RETRY_AFTER_SECONDS, MassiveHTTPTransport, RateLimitedError,
    TransportError, require_allowed_host, require_allowed_url)
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


# --- documented query auth, kept out of everything recorded ---------------
#
# Some endpoints require the key as a query parameter rather than a header.
# That is the vendor's documented scheme, so the transport supports it -- but
# a key in a query string is exactly what lands in access logs and, worse, in
# committed raw metadata. So the URL that goes on the wire and the URL that
# gets recorded are built separately, and only one of them ever carries a key.


def test_query_auth_reaches_the_wire():
    """Not a vacuous check: the request really must carry the credential."""
    transport = _transport([_Response({"results": []})])
    transport.fetch("/stocks/v1/splits", {"ticker": "AAPL"}, {},
                    auth_query={"apiKey": SENTINEL})
    assert SENTINEL in transport.calls[0]["url"]


def test_query_auth_never_reaches_the_recorded_url():
    """The recorded URL is what gets stored in raw metadata forever."""
    transport = _transport([_Response({"results": []})])
    response = transport.fetch("/stocks/v1/splits", {"ticker": "AAPL"}, {},
                               auth_query={"apiKey": SENTINEL})
    assert SENTINEL not in response.url
    assert "apikey" not in response.url.lower()
    assert response.url.endswith("?ticker=AAPL")


def test_query_auth_never_reaches_an_exception_or_the_stats():
    transport = _transport([_http_error(500)] * 4)
    with pytest.raises(TransportError) as error:
        transport.fetch("/stocks/v1/splits", {"ticker": "AAPL"}, {},
                        auth_query={"apiKey": SENTINEL})
    assert SENTINEL not in str(error.value)
    assert SENTINEL not in repr(error.value)
    assert SENTINEL not in json.dumps(transport.payload())


def test_a_recorded_url_carrying_a_key_is_still_refused():
    """The strict check stays strict; query auth does not relax it."""
    with pytest.raises(TransportError):
        require_allowed_url(
            "https://api.massive.com/stocks/v1/splits?apiKey=abc")
    # The host-only check is what the outbound URL uses, and it permits it.
    assert require_allowed_host(
        "https://api.massive.com/stocks/v1/splits?apiKey=abc")


def test_query_auth_still_cannot_reach_another_host():
    transport = _transport([_Response({"results": []})])
    with pytest.raises(HostNotAllowedError):
        transport.build_url("/stocks/v1/splits", {})
        require_allowed_host("https://evil.example.com/x?apiKey=abc")


def test_bearer_remains_the_default_and_sends_no_query_key():
    transport = _transport([_Response({"results": []})])
    response = transport.fetch("/stocks/v1/splits", {"ticker": "AAPL"},
                               {"Authorization": f"Bearer {SENTINEL}"})
    assert SENTINEL not in transport.calls[0]["url"]
    assert SENTINEL not in response.url
    assert SENTINEL in transport.calls[0]["headers"]["Authorization"]


# --- redirects (Phase 6E-V2-FIX2) -----------------------------------------
#
# urllib follows 3xx automatically and its redirect_request copies every
# header except Content-Length and Content-Type onto the new request --
# Authorization among them. A vendor redirect to another host would hand that
# host the API key, bypassing in one step every credential control in this
# project, because they all guard the value everywhere *except* the moment it
# leaves the process.
#
# These tests watch the requests the opener actually builds. Calling the
# validator directly would prove only that the validator works; it would not
# prove the credential never reaches the wire.


class _RedirectingOpener:
    """A urllib handler chain with one scripted redirect hop.

    Drives the real AllowlistedRedirectHandler, so the policy under test is
    the production one rather than a re-implementation.
    """

    def __init__(self, location, *, final_payload=None, hops=1):
        self.location = location
        self.final_payload = final_payload if final_payload is not None else {
            "results": []}
        self.hops = hops
        self.requests = []
        from scripts.trading_lab.massive_http_transport import (
            AllowlistedRedirectHandler)
        self._handler = AllowlistedRedirectHandler()

    def open(self, request, timeout=None):
        self.requests.append({"url": request.full_url,
                              "headers": dict(request.header_items())})
        if len(self.requests) <= self.hops:
            following = self._handler.redirect_request(
                request, io.BytesIO(b""), 302, "Found", {}, self.location)
            if following is None:                    # pragma: no cover
                raise AssertionError("handler returned None")
            return self.open(following, timeout=timeout)
        response = _Response(self.final_payload)
        response.geturl = lambda: self.requests[-1]["url"]
        return response


def _redirect_transport(location, **kwargs):
    opener = _RedirectingOpener(location, **kwargs)
    transport = MassiveHTTPTransport(opener=opener.open, sleep=lambda _: None)
    transport.opener = opener
    return transport


def test_R6E_1_a_same_host_redirect_is_followed():
    transport = _redirect_transport(
        "https://api.massive.com/stocks/v1/splits?ticker=AAPL")
    response = transport.fetch("/stocks/v1/splits", {"ticker": "AAPL"}, {})
    assert response.payload == {"results": []}
    assert len(transport.opener.requests) == 2


def test_R6E_2_a_cross_host_redirect_is_refused():
    from scripts.trading_lab.massive_http_transport import HostNotAllowedError

    transport = _redirect_transport("https://evil.example.com/steal")
    with pytest.raises(HostNotAllowedError) as error:
        transport.fetch("/stocks/v1/splits", {"ticker": "AAPL"}, {})
    assert "allowlist" in str(error.value)


def test_R6E_3_authorization_never_reaches_a_cross_host_request():
    """The whole point. The validator passing is not the same as the key
    staying home."""
    from scripts.trading_lab.massive_http_transport import HostNotAllowedError

    transport = _redirect_transport("https://evil.example.com/steal")
    with pytest.raises(HostNotAllowedError):
        transport.fetch("/stocks/v1/splits", {"ticker": "AAPL"},
                        {"Authorization": f"Bearer {SENTINEL}"})

    hostile = [call for call in transport.opener.requests
               if "evil.example.com" in call["url"]]
    assert hostile == [], "a request was built for the hostile host"
    for call in transport.opener.requests:
        assert "api.massive.com" in call["url"]
    # And nowhere in anything the opener saw did the key leave the allowlist.
    for call in transport.opener.requests:
        if SENTINEL in json.dumps(call["headers"]):
            assert "api.massive.com" in call["url"]


def test_R6E_4_the_final_url_is_recorded_not_the_one_we_asked_for():
    transport = _redirect_transport(
        "https://api.massive.com/stocks/v1/splits?ticker=AAPL&page=2")
    response = transport.fetch("/stocks/v1/splits", {"ticker": "AAPL"}, {})
    assert "page=2" in response.url, "the redirect was not reflected"


def test_R6E_5_the_final_url_is_revalidated_even_if_a_handler_lets_one_slip():
    """Belt and braces: an injected opener must not widen the boundary."""
    from scripts.trading_lab.massive_http_transport import HostNotAllowedError

    def sneaky(request, timeout=None):
        response = _Response({"results": []})
        response.geturl = lambda: "https://evil.example.com/served"
        return response

    transport = MassiveHTTPTransport(opener=sneaky, sleep=lambda _: None)
    with pytest.raises(HostNotAllowedError):
        transport.fetch("/stocks/v1/splits", {"ticker": "AAPL"}, {})


@pytest.mark.parametrize("location,label", [
    ("http://api.massive.com/downgrade", "https->http"),
    ("https://api.massive.com:4444/altport", "alternate port"),
    ("https://api.massive.com.evil.example/sub", "hostile subdomain"),
    ("https://evil.example.com@api.massive.com/userinfo", "userinfo"),
    ("https://user:pass@api.massive.com/userinfo", "userinfo with password"),
])
def test_R6E_6_7_8_scheme_port_and_authority_attacks_are_refused(location,
                                                                 label):
    from scripts.trading_lab.massive_http_transport import HostNotAllowedError

    transport = _redirect_transport(location)
    with pytest.raises(HostNotAllowedError):
        transport.fetch("/stocks/v1/splits", {"ticker": "AAPL"},
                        {"Authorization": f"Bearer {SENTINEL}"})
    assert all("api.massive.com" in call["url"] and ":4444" not in call["url"]
               for call in transport.opener.requests)


def test_R6E_9_query_auth_never_survives_into_the_recorded_final_url():
    transport = _redirect_transport(
        f"https://api.massive.com/stocks/v1/splits?ticker=AAPL&apiKey={SENTINEL}")
    response = transport.fetch("/stocks/v1/splits", {"ticker": "AAPL"}, {},
                               auth_query={"apiKey": SENTINEL})
    assert SENTINEL not in response.url
    assert "apikey" not in response.url.lower()
    assert "ticker=AAPL" in response.url


def test_R6E_10_a_redirect_loop_stays_bounded_by_urllib():
    """The hardened handler must not have removed urllib's own bound.

    Asserting on a scripted opener would test the harness. What matters is
    that the real handler still carries the inherited redirect ceiling and
    the loop-detection machinery, so a vendor bouncing us between two of its
    own paths terminates instead of spinning.
    """
    import urllib.request

    from scripts.trading_lab.massive_http_transport import (
        AllowlistedRedirectHandler)

    handler = AllowlistedRedirectHandler()
    assert isinstance(handler, urllib.request.HTTPRedirectHandler)
    assert handler.max_redirections == \
        urllib.request.HTTPRedirectHandler.max_redirections
    assert handler.max_redirections < 100


def test_a_refused_redirect_is_not_retried():
    """It is a refusal, not a timeout. Retrying buries the real reason."""
    from scripts.trading_lab.massive_http_transport import HostNotAllowedError

    transport = _redirect_transport("https://evil.example.com/steal")
    with pytest.raises(HostNotAllowedError):
        transport.fetch("/stocks/v1/splits", {"ticker": "AAPL"}, {})
    assert transport.stats.retries == 0
    assert len(transport.opener.requests) == 1


def test_the_production_default_is_the_hardened_opener_not_bare_urlopen():
    import urllib.request

    from scripts.trading_lab.massive_http_transport import (
        ALLOWED_HOSTS, build_hardened_opener)
    from scripts.trading_lab.safe_http import AllowlistedRedirectHandler

    transport = MassiveHTTPTransport()
    assert transport._opener is not urllib.request.urlopen

    opener = build_hardened_opener()
    guards = [h for h in opener.handlers
              if isinstance(h, AllowlistedRedirectHandler)]
    assert guards, "the production opener carries no allowlisted redirect guard"
    # Stronger than merely present: bound to this provider's allowlist, and
    # raising this provider's error type rather than the shared one.
    assert guards[0].allowed_hosts == ALLOWED_HOSTS
    assert guards[0].error_class is HostNotAllowedError
    # urllib's own permissive handler must not also be in the chain.
    plain = [h for h in opener.handlers
             if type(h) is urllib.request.HTTPRedirectHandler]
    assert plain == []
