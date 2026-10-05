"""Clearly synthetic native-shape providers and deterministic model fakes."""
from datetime import timedelta
import json
import re

from scripts.trading_lab.coinbase_candles import adapt_coinbase_candles
from scripts.trading_lab.equity_calendar import USEquityRegularCalendar
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.yahoo_chart_provider import adapt_chart_rows

from .config import instant, iso
from .data import equity_calendar
from .runners import ModelRunner


class Clock:
    def __init__(self, at):
        self.at = at

    def __call__(self):
        return self.at


class SyntheticData:
    synthetic = True

    def __init__(self, ledger, clock):
        self.ledger, self.clock = ledger, clock

    def source(self, source, payload):
        # Use the same durable accounting, including GDELT's persistent spacing.
        if source == "gdelt_doc_api":
            self.clock.at += timedelta(seconds=self.ledger.grant.payload["data_sources"][source]["min_spacing_seconds"])
        seq = self.ledger.reserve(source)
        return {"url": "https://example.invalid/synthetic/" + source, "received_at": iso(self.clock()),
                "digest": sha256_canonical(payload), "dispatch": seq}

    def yahoo(self, asset, start, end):
        sessions = equity_calendar(end.year).sessions_between(start, end)
        base = 90 + sum(ord(c) for c in asset) % 50
        values = [base + (s.open_at.toordinal() % 29) * .1 for s in sessions]
        payload = {"chart": {"error": None, "result": [{"meta": {"symbol": asset, "currency": "USD",
            "exchangeTimezoneName": "America/New_York", "exchangeName": "NMS", "instrumentType": "EQUITY"},
            "timestamp": [int(s.open_at.timestamp()) for s in sessions], "indicators": {"quote": [{
                "open": values, "close": [v + .2 for v in values], "high": [v + 1 for v in values],
                "low": [v - 1 for v in values], "volume": [1000 for _ in values]}]}}]}}
        return adapt_chart_rows(payload, instrument_id="xnas:" + asset), self.source("yahoo_chart", payload)

    def crypto(self, asset, start, end, *, minute=False):
        if minute:
            payload = [[int(start.timestamp()), 99, 103, 100 + start.day * .01, 101, 20]]
            return payload[0][3], self.source("coinbase_exchange_public", payload)
        opening = start.replace(hour=0, minute=0, second=0, microsecond=0)
        payload = []
        while opening + timedelta(days=1) <= end:
            close = 100 + opening.day * .01
            payload.append([int(opening.timestamp()), 99, 103, 100, close, 20])
            opening += timedelta(days=1)
        source = self.source("coinbase_exchange_public", payload)
        rows = adapt_coinbase_candles(json.dumps(payload).encode(), product_id=asset, timeframe="1d",
            available_at=source["received_at"], ingested_at=source["received_at"])
        return [{"bar_open_at": instant(r["bar_open_at"]), "close": float(r["close"])} for r in rows], source

    def headlines(self, query, before):
        payload = {"articles": [{"url": "https://example.invalid/synthetic-news", "title": "Synthetic catalyst",
                   "seendate": (before - timedelta(hours=1)).strftime("%Y%m%dT%H%M%SZ"), "domain": "example.invalid"}]}
        a = payload["articles"][0]
        digest = [{"title": a["title"], "source": a["domain"], "published_at": iso(before - timedelta(hours=1)),
                   "url": a["url"], "publication_clock": "SYNTHETIC"}]
        return digest, self.source("gdelt_doc_api", payload)


class SyntheticRunner(ModelRunner):
    synthetic = True

    def once(self, role, text):
        self.ledger.reserve(role)
        self.metadata[role] = {"model": "SYNTHETIC_FAKE", "cli_version": "synthetic-v1", "reported_version": "synthetic-v1"}
        context = json.loads(text.split("\nCONTEXT_DATA:\n")[1].split("\nINDEPENDENT_ANALYST_DATA:\n")[0])
        if role == "reviewer":
            analysts = json.loads(text.split("\nINDEPENDENT_ANALYST_DATA:\n")[1])
            return {"verdicts": [{"analyst": role, "asset": v["asset"], "horizon": v["horizon"], "verdict": "KEEP",
                                  "reason_code": "supported", "adjusted_p": None, "note": "Synthetic verified fake."}
                                 for role, output in analysts.items() for v in output["views"]]}
        return {"regime": ["Synthetic demo; no real evidence."], "views": [{
            "asset": a, "horizon": h, "view": "UP", "p_outperform": .56,
            "confidence_reason": "Synthetic only", "catalysts": [{"url": "https://example.invalid/synthetic",
                "published_at": iso(instant(context["decision_time"]) - timedelta(hours=1)), "fact": "Synthetic fact"}],
            "priced_in_assessment": "Synthetic", "second_order": "Synthetic", "counter_thesis": "Synthetic",
            "falsifier": "Synthetic", "event_risk": []} for a in context["universe"] for h in ("1d", "5d")]}
