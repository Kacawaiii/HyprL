import pytest

from crypto_news.egress import DEFAULT_EGRESS_POLICY, EgressDeniedError


@pytest.mark.parametrize(
    "url",
    [
        "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD",
        "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
        "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
        "https://www.sec.gov/news/press-release/example",
        "https://www.cftc.gov/PressRoom/PressReleases/example",
        "https://www.federalreserve.gov/newsevents/pressreleases.htm",
        "https://ethereum.org/en/roadmap/",
        "https://bitcoincore.org/en/releases/",
    ],
)
def test_explicit_https_hosts_are_authorized_without_connecting(url: str) -> None:
    assert DEFAULT_EGRESS_POLICY.authorize(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://api.coinbase.com/",
        "https://user:password@api.coinbase.com/",
        "https://api.coinbase.com:444/",
        "https://127.0.0.1/",
        "https://localhost/",
        "https://www.sec.gov.evil.example/",
        "https://ｗｗｗ.sec.gov/news",
        "https://evil.sec.gov/",
        "https://example.com/",
        "https://www.sec.gov/news#fragment",
    ],
)
def test_non_allowlisted_or_ambiguous_destinations_are_denied(url: str) -> None:
    with pytest.raises(EgressDeniedError):
        DEFAULT_EGRESS_POLICY.authorize(url)


def test_allowlist_is_immutable_and_exact() -> None:
    assert isinstance(DEFAULT_EGRESS_POLICY.allowed_hosts, frozenset)
    assert "api.coinbase.com" in DEFAULT_EGRESS_POLICY.allowed_hosts
    assert "coinbase.com" not in DEFAULT_EGRESS_POLICY.allowed_hosts
    assert all(host == host.lower() for host in DEFAULT_EGRESS_POLICY.allowed_hosts)
