import hashlib
import json
from contextlib import contextmanager

import pytest

from scripts.trading_lab.trader_agent.config import Authorization, TraderError
from scripts.trading_lab.trader_agent.data import PublicData, build_context
from scripts.trading_lab.trader_agent.news_queries import QUERIES, news_query
from scripts.trading_lab.trader_agent.synthetic import SyntheticData

REAL_UNIVERSE = {
    "stocks": "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD INTC JPM GS XOM CVX LLY UNH PFE WMT COST NKE BA CAT GM NFLX DIS".split(),
    "sector_etfs": "XLK XLF XLE XLI XLV".split(), "crypto": ["BTC-USD", "ETH-USD"]}
TEXT = b"The specified phrase is too short."


def test_every_universe_asset_has_a_non_trivial_query():
    for asset in [a for group in REAL_UNIVERSE.values() for a in group] + ["macro", "politics", "trade"]:
        query = news_query(asset)
        bare = query.strip('"')
        assert bare != asset and len(bare) >= 4, asset
        if " " in bare and not query.startswith("("):
            assert query.startswith('"') and query.endswith('"')
        if " " not in bare:
            assert '"' not in query


def test_unmapped_asset_is_a_typed_error():
    with pytest.raises(TraderError) as error:
        news_query("ZZZZ")
    assert error.value.code == "NEWS_QUERY_UNMAPPED"
    assert set(QUERIES) >= {"AAPL", "macro"}


class Response:
    def __init__(self, body):
        self.body = body

    def read(self, _):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_network(monkeypatch, body):
    class Opener:
        def open(self, request, timeout):
            return Response(body)
    monkeypatch.setattr("scripts.trading_lab.trader_agent.data.build_opener", lambda *a: Opener())


def test_text_answer_is_source_not_json_with_evidence_digest(ledger, clock, monkeypatch):
    fake_network(monkeypatch, TEXT)
    data = PublicData(ledger, clock=clock, sleep=lambda s: clock.__setattr__("at", clock.at + __import__("datetime").timedelta(seconds=s)))
    with pytest.raises(TraderError) as error:
        data.headlines("Apple", clock())
    assert error.value.code == "SOURCE_NOT_JSON"
    assert error.value.evidence == hashlib.sha256(TEXT).hexdigest()
    assert any(p.name.endswith(error.value.evidence + ".json") for p in (ledger.root / "raw").iterdir())


def test_headline_failure_marks_asset_and_run_continues(ledger, clock):
    class TextNews(SyntheticData):
        def headlines(self, query, before):
            if query == "Apple":
                raise TraderError("SOURCE_NOT_JSON", "abc")
            return super().headlines(query, before)
    context, _ = build_context(ledger.grant, TextNews(ledger, clock), at=clock())
    assert context["headlines"]["AAPL"] == {"state": "SOURCE_NOT_JSON", "items": [], "evidence_digest": "abc"}
    assert context["headlines"]["MSFT"]["state"] == "AVAILABLE"
    assert "AAPL" in context["universe"]


def test_price_text_answer_stays_strict(ledger, clock, monkeypatch):
    fake_network(monkeypatch, TEXT)
    with pytest.raises(TraderError) as error:
        PublicData(ledger, clock=clock).yahoo("SPY", clock() - __import__("datetime").timedelta(days=5), clock())
    assert error.value.code == "SOURCE_NOT_JSON"
