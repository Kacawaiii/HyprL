"""Bounded public GETs, persistent request budget, validated external cache.

Only this module can access the network, and only with an operator grant.
Cached replay never opens a socket. Raw prices and catalogs never enter Git.
"""

import fcntl
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd

from .protocol import CUTOFF, FIRST_DAY, LAST_DAY, outside_repo, utc, validate_bars

COINBASE = "https://api.exchange.coinbase.com"
YAHOO = "https://query1.finance.yahoo.com"
BAR_COLUMNS = ["low", "high", "open", "close", "volume"]
EMPTY = lambda: pd.DataFrame(columns=BAR_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))


class Blocked(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Blocked("Redirect outside the scoped request is forbidden")


class Grant:
    """Exact origins, methods and path templates; retries consume the grant too."""

    def __init__(self, path, cache):
        self.path = outside_repo(path)
        auth_root = Path.home() / "authorizations"
        if self.path.parent != auth_root.resolve():
            raise Blocked("Operator authorization must be in ~/authorizations/")
        self.raw = self.path.read_bytes()
        self.digest = hashlib.sha256(self.raw).hexdigest()
        self.spec = json.loads(self.raw)
        if self.spec.get("schema") != "crypto-carre-research-authorization-v1":
            raise Blocked("Wrong authorization schema/scope for this research")
        self.expiry = utc(self.spec["not_after"])
        self.start = utc(self.spec.get("not_before", self.spec.get("granted_at")))
        self.state = outside_repo(cache) / ("grant-" + self.digest + ".json")
        self.lock = self.state.with_suffix(".lock")
        self.state.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def matches(template, path):
        escaped = re.escape(template)
        escaped = re.sub(r"\\\{(?:product_id|id)\\\}", r"[A-Z0-9]+-USD", escaped)
        escaped = re.sub(r"\\\{symbol\\\}", r"(?:BTC|ETH|SOL|LINK|AVAX|XRP)-USD", escaped)
        return re.fullmatch(escaped, path) is not None

    def reserve(self, url):
        now = pd.Timestamp(datetime.now(timezone.utc))
        if not self.start <= now < self.expiry:
            raise Blocked("Operator authorization is not effective or has expired")
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.digest:
            raise Blocked("Authorization changed during capture; restart explicitly")
        parsed = urlparse(url)
        origin = parsed.scheme + "://" + parsed.netloc
        if origin not in (COINBASE, YAHOO) or parsed.username or parsed.fragment:
            raise Blocked("Forbidden origin or URL")
        scope = self.spec.get("scope", {}).get(origin, {})
        if "GET" not in scope.get("methods", []):
            raise Blocked("GET not authorized")
        if not any(self.matches(p, parsed.path) for p in scope.get("paths", [])):
            raise Blocked("Request path outside authorization")
        budget = scope.get("max_requests", 0)
        if type(budget) is not int or budget <= 0:
            raise Blocked("Explicit positive request budget required")
        with self.lock.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            used = json.loads(self.state.read_text()) if self.state.exists() else {}
            if used.get(origin, 0) >= budget:
                raise Blocked("Request budget exhausted (including failed attempts)")
            # Persist before request: crashes cannot refund an attempted request.
            used[origin] = used.get(origin, 0) + 1
            atomic_json(self.state, used)


class PublicClient:
    def __init__(self, grant):
        self.grant = grant
        self.last = 0.0
        self.opener = build_opener(NoRedirect())

    def get(self, origin, path, params=None):
        url = origin + path + ("?" + urlencode(params) if params else "")
        for attempt in range(3):
            time.sleep(max(0, 0.35 - (time.monotonic() - self.last)))
            self.grant.reserve(url)
            self.last = time.monotonic()
            request = Request(url, headers={"Accept": "application/json", "User-Agent": "crypto-carre-research/1"})
            try:
                with self.opener.open(request, timeout=30) as response:
                    return json.load(response)
            except HTTPError as exc:
                # Error bodies deliberately neither read nor printed.
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise Blocked(f"Public endpoint HTTP {exc.code}") from None
                time.sleep(2 ** attempt)
        raise Blocked("Retries exhausted")


def coinbase_bars(payload, start, end):
    if not isinstance(payload, list):
        raise ValueError("Coinbase candles must be an array")
    rows = []
    for row in payload:
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError("Coinbase candle shape must be [time, low, high, open, close, volume]")
        timestamp = pd.to_datetime(row[0], unit="s", utc=True)
        if timestamp > CUTOFF:
            raise ValueError("Forbidden Coinbase price after cutoff")
        if utc(start) <= timestamp <= utc(end):
            rows.append([timestamp, *row[1:]])
    frame = pd.DataFrame(rows, columns=["day", *BAR_COLUMNS]).set_index("day")
    frame.index = pd.DatetimeIndex(frame.index, tz="UTC")
    frame = frame.sort_index().astype(float)
    validate_bars(frame)
    return frame


def yahoo_bars(payload, before):
    if utc(before) > LAST_DAY + pd.Timedelta(days=1):
        raise ValueError("Yahoo extension bound after cutoff")
    chart = payload["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError("Yahoo returned no chart result")
    result = chart["result"][0]
    if result.get("meta", {}).get("currency") != "USD":
        raise ValueError("Yahoo chart must be denominated in USD")
    quote = result["indicators"]["quote"][0]
    rows = []
    for i, raw_time in enumerate(result.get("timestamp", [])):
        stamp = pd.to_datetime(raw_time, unit="s", utc=True)
        if stamp > CUTOFF:
            raise ValueError("Forbidden Yahoo observation after cutoff")
        if stamp >= utc(before):
            raise ValueError("Yahoo is permitted only before first Coinbase candle")
        row = [quote[col][i] for col in BAR_COLUMNS]
        if any(x is None for x in row):
            continue
        rows.append([stamp.normalize(), *row])
    frame = pd.DataFrame(rows, columns=["day", *BAR_COLUMNS]).set_index("day")
    frame.index = pd.DatetimeIndex(frame.index, tz="UTC")
    frame = frame.sort_index().astype(float)
    validate_bars(frame)
    return frame


def encode_bars(frame):
    validate_bars(frame)
    return [[int(date.timestamp()), *list(row)] for date, row in frame.iterrows()]


def decode_bars(rows):
    return coinbase_bars(rows, FIRST_DAY, CUTOFF)


class Cache:
    def __init__(self, root, client=None):
        self.root = outside_repo(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = client

    def products(self):
        path = self.root / "products.json"
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("source") != COINBASE + "/products":
                raise ValueError("Unrecognized cached product catalog")
            return value
        if not self.client:
            raise Blocked("No cached catalog and no research authorization")
        payload = self.client.get(COINBASE, "/products")
        if not isinstance(payload, list) or not payload:
            raise Blocked("Product endpoint returned no usable array")
        products = []
        for item in payload:
            if not isinstance(item, dict) or not all(k in item for k in ("id", "base_currency", "quote_currency", "status")):
                raise ValueError("Unexpected Coinbase product shape")
            if item["quote_currency"] == "USD" and not re.fullmatch(r"[A-Z0-9]+-USD", item["id"]):
                raise ValueError("Unexpected USD product ID")
            products.append({k: item.get(k) for k in (
                "id", "base_currency", "quote_currency", "status", "trading_disabled",
                "cancel_only", "post_only", "auction_mode",
            )})
        if len({p["id"] for p in products}) != len(products):
            raise ValueError("Duplicate Coinbase product IDs")
        value = {
            "source": COINBASE + "/products", "captured_at": datetime.now(timezone.utc).isoformat(),
            "raw_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "returned_count": len(products), "products": products,
            "status_counts": dict(Counter(str(p["status"]) for p in products)),
        }
        atomic_json(path, value)
        return value

    def history(self, product):
        if not re.fullmatch(r"[A-Z0-9]+-USD", product):
            raise ValueError("Only Coinbase USD products allowed")
        combined = self.root / "coinbase" / product / "complete.json"
        if combined.exists():
            value = json.loads(combined.read_text())
            if value.get("cutoff") != CUTOFF.isoformat() or value.get("scan_start") != FIRST_DAY.isoformat():
                raise ValueError("Cache coverage mismatch")
            return decode_bars(value["rows"])
        chunks = []
        start = FIRST_DAY
        while start <= LAST_DAY:
            # 299 days, safely below Coinbase's 300 candle limit, including boundaries.
            end = min(start + pd.Timedelta(days=298, hours=23, minutes=59), CUTOFF)
            path = self.root / "coinbase" / product / f"{start.date()}.json"
            if path.exists():
                value = json.loads(path.read_text())
                if value["start"] != start.isoformat() or value["end"] != end.isoformat():
                    raise ValueError("Chunk coverage mismatch")
                frame = coinbase_bars(value["rows"], start, end)
            else:
                if not self.client:
                    raise Blocked(f"Incomplete external cache for {product}")
                payload = self.client.get(COINBASE, f"/products/{product}/candles", {
                    "start": start.isoformat(), "end": end.isoformat(), "granularity": 86400,
                })
                frame = coinbase_bars(payload, start, end)
                atomic_json(path, {"start": start.isoformat(), "end": end.isoformat(), "rows": encode_bars(frame)})
            chunks.append(frame)
            start += pd.Timedelta(days=299)
        frame = pd.concat(chunks).sort_index()
        validate_bars(frame)
        atomic_json(combined, {"cutoff": CUTOFF.isoformat(), "scan_start": FIRST_DAY.isoformat(), "rows": encode_bars(frame)})
        return frame

    def extension(self, product, first_coinbase):
        if product not in ("BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "AVAX-USD", "XRP-USD"):
            raise ValueError("Yahoo allowed only for named Part A prelisting extension")
        path = self.root / "yahoo" / (product + ".json")
        if path.exists():
            value = json.loads(path.read_text())
            if value["before"] != first_coinbase.isoformat():
                raise ValueError("Yahoo cache listing bound mismatch")
            result = decode_bars(value["rows"])
            if len(result) and result.index.max() >= first_coinbase:
                raise ValueError("Yahoo overlaps Coinbase")
            return result
        if not self.client:
            raise Blocked(f"Yahoo extension missing for {product}")
        payload = self.client.get(YAHOO, f"/v8/finance/chart/{product}", {
            "period1": int(FIRST_DAY.timestamp()), "period2": int(first_coinbase.timestamp()), "interval": "1d",
        })
        frame = yahoo_bars(payload, first_coinbase)
        atomic_json(path, {"before": first_coinbase.isoformat(), "rows": encode_bars(frame)})
        return frame
