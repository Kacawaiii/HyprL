"""Bounded public prices/news and causal read-only archive summaries."""
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import math
import statistics
import time
from urllib.parse import urlencode, urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler

from scripts.trading_lab.coinbase_candles import adapt_coinbase_candles
from scripts.trading_lab.equity_calendar import USEquityRegularCalendar
from scripts.trading_lab.research_protection import crypto_interval, equity_interval
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.yahoo_chart_provider import adapt_chart_rows, chart_path, parse_chart_meta

from .config import HOSTS, TraderError, instant, iso, now, strict_json


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TraderError("REDIRECT_REFUSED")


class PublicData:
    def __init__(self, ledger, *, clock=now, sleep=time.sleep):
        self.ledger, self.clock, self.sleep = ledger, clock, sleep
        self.synthetic = False

    def fetch(self, source, path, params):
        host = HOSTS[source]
        if source == "yahoo_chart":
            allowed = {chart_path(a) for a in self.ledger.grant.universe + self.ledger.grant.payload["universe"]["benchmarks_not_predicted"]}
        elif source == "coinbase_exchange_public":
            allowed = {f"/products/{a}/candles" for a in self.ledger.grant.payload["universe"]["crypto"]}
        else:
            allowed = {"/api/v2/doc/doc"}
        if path not in allowed:
            raise TraderError("UNAUTHORIZED_PATH")
        while True:
            try:
                seq = self.ledger.reserve(source)
                break
            except TraderError as error:
                if error.code != "SOURCE_SPACING":
                    raise
                self.sleep(min(6, self.ledger.grant.payload["data_sources"][source]["min_spacing_seconds"]))
        url = "https://" + host + path + "?" + urlencode(params)
        self.ledger.grant.check(self.clock())
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "HyprL-paper-shadow/1"})
        try:
            with build_opener(NoRedirect()).open(request, timeout=30) as response:
                raw = response.read(1_000_001)
                if len(raw) > 1_000_000:
                    raise TraderError("SOURCE_TOO_LARGE")
        except TraderError:
            raise
        except Exception:
            raise TraderError("SOURCE_UNAVAILABLE") from None
        received = self.clock()
        folder = self.ledger.root / "raw"
        folder.mkdir(mode=0o700, exist_ok=True)
        digest = hashlib.sha256(raw).hexdigest()
        (folder / f"{seq}-{digest}.json").write_bytes(raw)
        return strict_json(raw), {"url": url, "received_at": iso(received), "digest": digest, "dispatch": seq}

    def yahoo(self, asset, start, end):
        payload, source = self.fetch("yahoo_chart", chart_path(asset), {
            "period1": int(start.timestamp()), "period2": int(end.timestamp()), "interval": "1d", "events": "div,split"})
        parse_chart_meta(payload)
        rows = adapt_chart_rows(payload, instrument_id=asset)
        for row in rows:
            values = [float(row[k]) for k in ('open', 'high', 'low', 'close', 'volume')]
            op, high, low, close, volume = values
            if not all(math.isfinite(v) for v in values) or not 0 < low <= min(op, close) <= max(op, close) <= high or volume < 0:
                raise TraderError('INVALID_PRICE')
        return rows, source

    def crypto(self, asset, start, end, *, minute=False):
        payload, source = self.fetch("coinbase_exchange_public", f"/products/{asset}/candles", {
            "start": iso(start), "end": iso(end), "granularity": 60 if minute else 86400})
        if minute:
            # Preserve Coinbase's six-field native shape; open at the exact anchor, never interpolate.
            if not isinstance(payload, list) or len(payload) > 300:
                raise TraderError("CRYPTO_SHAPE_INVALID")
            matches = [r for r in payload if isinstance(r, list) and len(r) == 6 and r[0] == int(start.timestamp())]
            if len(matches) != 1:
                raise TraderError("MISSING_ANCHOR_PRICE")
            row = matches[0]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row):
                raise TraderError("CRYPTO_SHAPE_INVALID")
            if not 0 < row[1] <= row[3] <= row[2] or not row[1] <= row[4] <= row[2] or row[5] < 0:
                raise TraderError("CRYPTO_SHAPE_INVALID")
            return float(row[3]), source
        if not isinstance(payload, list) or any(not isinstance(r, list) or len(r) != 6 for r in payload):
            raise TraderError('CRYPTO_SHAPE_INVALID')
        for row in payload:
            if isinstance(row[0], bool) or not isinstance(row[0], (int, float)) or not math.isfinite(row[0]):
                raise TraderError('CRYPTO_SHAPE_INVALID')
        # Coinbase may include a forming current-day candle. Preserve the raw response privately;
        # pass only completed native-shape rows to the proven completed-bar adapter.
        completed = [r for r in payload if datetime.fromtimestamp(r[0], tz=timezone.utc) + timedelta(days=1) <= end]
        rows = adapt_coinbase_candles(json.dumps(completed).encode(), product_id=asset, timeframe="1d",
                                     available_at=source["received_at"], ingested_at=source["received_at"])
        return [{"bar_open_at": instant(r["bar_open_at"]), "close": float(r["close"])} for r in rows], source

    def headlines(self, query, before):
        payload, source = self.fetch("gdelt_doc_api", "/api/v2/doc/doc", {
            "query": query, "mode": "ArtList", "format": "json", "maxrecords": 10,
            "sort": "DateDesc", "startdatetime": (before - timedelta(hours=72)).strftime("%Y%m%d%H%M%S"),
            "enddatetime": before.strftime("%Y%m%d%H%M%S")})
        if not isinstance(payload, dict) or not isinstance(payload.get("articles"), list):
            raise TraderError("NEWS_SHAPE_INVALID")
        output, seen = [], set()
        for article in payload["articles"]:
            url = article["url"]
            published = datetime.strptime(article["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            if published <= before and url not in seen and urlparse(url).scheme == "https":
                output.append({"title": article["title"][:500], "source": article["domain"],
                    "published_at": iso(published), "publication_clock": "GDELT_SEEN_NOT_PUBLISHER_ATTESTED", "url": url})
                seen.add(url)
        return output, source


@lru_cache(maxsize=2048)
def calendar_session(day):
    start = instant(day + "T00:00:00Z")
    return equity_calendar(start.year).session_on(day)


@lru_cache(maxsize=4)
def equity_calendar(year):
    calendar = USEquityRegularCalendar()
    calendar.sessions_between(f"{year - 1}-01-01T00:00:00Z", f"{year + 1}-12-31T23:59:59Z")
    return calendar


def label_window(asset, day, horizon, crypto_assets):
    n = 1 if horizon == "1d" else 5
    if asset in crypto_assets:
        entry = instant(day + "T13:30:00Z")
        return entry, entry + timedelta(days=n)
    start = instant(day + "T00:00:00Z")
    sessions = equity_calendar(start.year).sessions_between(start, start + timedelta(days=20))
    if not sessions or sessions[0].session_date != day:
        raise TraderError("HOLIDAY")
    return sessions[0].open_at, sessions[n - 1].close_at


def protected(asset, first, last):
    canonical = {"BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD"}.get(asset, "equity:XNAS:" + asset)
    # Registries use BTC-USD / ETH-USD for some contracts and canonical IDs for others.
    candidates = {asset, canonical, "equity:XNYS:" + asset, "xnas:" + asset, "xnys:" + asset}
    for interval in (crypto_interval(), equity_interval()):
        if candidates.intersection(interval.products) and interval.touches(first, last):
            return interval.identity_hash
    return None


def price_summary(rows, asset, before, crypto=False):
    calendar = equity_calendar(before.year) if not crypto else None
    completed = []
    for row in rows:
        opening = row["bar_open_at"]
        if crypto:
            close_at = opening + timedelta(days=1)
        else:
            session = calendar_session(opening.date().isoformat())
            if session is None or session.open_at != opening:
                raise TraderError("PRICE_OFF_CALENDAR")
            close_at = session.close_at
        close = float(row["close"])
        if not math.isfinite(close) or close <= 0:
            raise TraderError("INVALID_PRICE")
        if close_at <= before:
            completed.append((opening, close_at, close))
    completed.sort()
    if len(completed) < 21:
        raise TraderError("MISSING_PRICES")
    completed = completed[-21:]
    if len({r[0] for r in completed}) != 21:
        raise TraderError("DUPLICATE_PRICE")
    if crypto:
        expected = [completed[0][0] + timedelta(days=i) for i in range(21)]
        latest_due = before.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        expected = [s.open_at for s in calendar.sessions_between(completed[0][0], before)
                    if s.close_at <= before]
        latest_due = expected[-1] if expected else None
    if [r[0] for r in completed] != expected or (crypto and completed[-1][1] != latest_due):
        raise TraderError("PRICE_GAP_OR_STALE")
    closes = [r[2] for r in completed]
    daily = [closes[i] / closes[i - 1] - 1 for i in range(1, 21)]
    return {"asset": asset, "state": "AVAILABLE", "recent_closes": closes,
            "returns": {str(n): closes[-1] / closes[-1 - n] - 1 for n in (1, 5, 20)},
            "vol_20d": statistics.stdev(daily), "last_price_at": iso(completed[-1][1]),
            "vol_method": "sample_std_daily_raw_returns_20", "adjustment": "RAW"}


def archived(root, kind, at):
    if not root:
        return {"state": "NOT_CONFIGURED", "items": []}
    from scripts.trading_lab.app_api.sources import FomcViews, EdgarViews
    views = (FomcViews if kind == "fomc" else EdgarViews)(root)
    try:
        status = views.status()
        if status.get('status') != 'AVAILABLE':
            return {'state':'INTEGRITY_ERROR' if status.get('status') == 'REJECTED' else 'UNAVAILABLE', 'items':[]}
        cutoff = status.get("suggested_as_of")
        if not cutoff:
            return {"state": "UNRESOLVED", "items": []}
        cutoff = min(instant(cutoff), at)
        snapshot = views.snapshot(as_of=iso(cutoff), limit="1000")
        items = list(snapshot.get("items", snapshot.get("filings", [])))
        pagination = snapshot.get('pagination', {})
        cursor = pagination.get('next_cursor')
        while cursor and len(items) < 10000:
            page = views.snapshot(as_of=iso(cutoff), limit='1000', cursor=cursor)
            items.extend(page.get('items', page.get('filings', [])))
            cursor = page.get('pagination', {}).get('next_cursor')
        total = len(items)
        items.sort(key=lambda item: (item.get('filing_date') or item.get('declared_release_at') or '',
                                     item.get('first_available_at') or ''), reverse=True)
        return {"state": snapshot["snapshot"]["read_state"], "archive_as_of": iso(cutoff),
            "age_seconds": (at - cutoff).total_seconds(), "binding": snapshot["snapshot"], "items": items[:100],
            "archived_items_scanned": total, 'scan_truncated': bool(cursor),
            "coverage": "ARCHIVED_NOT_CURRENT" if not cursor else 'ARCHIVED_TRUNCATED_NOT_CURRENT'}
    finally:
        views.close()


def build_context(grant, data, *, at, fomc=None, edgar=None):
    start = at.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=70)
    crypto_assets = grant.payload["universe"]["crypto"]
    prices, sources, headlines, exclusions = {}, [], {}, []
    for asset in grant.universe + grant.payload["universe"]["benchmarks_not_predicted"]:
        _, last = label_window(asset, at.date().isoformat(), "5d", crypto_assets)
        binding = None if data.synthetic else protected(asset, start, last)
        if binding:
            prices[asset] = {"asset": asset, "state": "PROTECTED", "protection_hash": binding}
            exclusions.append({"asset": asset, "reason": "PROTECTED", "binding": binding})
            continue
        try:
            rows, source = (data.crypto(asset, start, at) if asset in crypto_assets else data.yahoo(asset, start, at))
            # Never admit a partially observed current daily candle.
            available = instant(source["received_at"])
            prices[asset] = price_summary(rows, asset, min(at, available), asset in crypto_assets)
            sources.append(source)
        except TraderError as error:
            if error.code in {"BUDGET_EXHAUSTED", "AUTHORIZATION_EXPIRED_OR_NOT_STARTED"}:
                raise
            prices[asset] = {"asset": asset, "state": error.code}
            exclusions.append({"asset": asset, "reason": error.code})
    eligible = [a for a in grant.universe if prices[a]["state"] == "AVAILABLE"]
    if prices["SPY"]["state"] != "AVAILABLE" or not eligible:
        raise TraderError("MISSING_PRICES")
    for asset in eligible + ["macro", "politics", "trade"]:
        query = {"macro": '(inflation OR "interest rates" OR employment)',
                 "politics": '(election OR sanctions OR geopolitics)',
                 "trade": '(tariff OR "trade agreement")'}.get(asset, f'"{asset}"')
        try:
            digest, source = data.headlines(query, at)
            headlines[asset] = {"state": "AVAILABLE", "items": digest}
            sources.append(source)
        except TraderError as error:
            if error.code in {"BUDGET_EXHAUSTED", "AUTHORIZATION_EXPIRED_OR_NOT_STARTED"}:
                raise
            headlines[asset] = {"state": error.code, "items": []}
    # The decision clock is after every durable input acquisition. The price cut stays at run start.
    decision_at = max([at] + [instant(s["received_at"]) for s in sources])
    context = {"schema": "trader-context-v1", "session": at.date().isoformat(), "decision_time": iso(decision_at),
               "price_cutoff": iso(at), "universe": eligible, "prices": prices, "headlines": headlines,
               "archives": {k: archived(root, k, decision_at) for k, root in (("fomc", fomc), ("edgar", edgar))},
               "sources": sources, "exclusions": exclusions, "synthetic": data.synthetic,
               "limitations": ["GDELT seen time is not attested publisher time", "Raw OHLC; corporate actions may distort returns",
                               "Probabilities are uncalibrated analyst judgments", "Archive coverage is stale and source specific"]}
    return context, sha256_canonical(context)
