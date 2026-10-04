"""Offline stand-ins for the SEC: a simulated clock and a fetcher answering from synthetic submissions
listings (columnar like filings.recent), with a server Date taken from the simulated true time. Nothing
here opens a socket."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json

from scripts.trading_lab.edgar.listing import cik10
from scripts.trading_lab.edgar.transport import FetchResult, classify_status

START = datetime(2026, 6, 17, 13, 0, 0, tzinfo=timezone.utc)
CIK_A, CIK_B = "320193", "789019"
COLUMNS = ("accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "act", "form", "fileNumber",
           "filmNumber", "items", "size", "isXBRL", "isInlineXBRL", "primaryDocument", "primaryDocDescription")


class SimClock:
    def __init__(self, start: datetime = START, *, wall_offset_s: float = 0.0):
        self.true, self._mono, self.wall_offset = start, 1000.0, timedelta(seconds=wall_offset_s)

    def wall(self) -> datetime:
        return self.true + self.wall_offset

    def mono(self) -> float:
        return self._mono

    def sleep(self, seconds: float) -> None:
        seconds = max(float(seconds), 0.0)
        self.true += timedelta(seconds=seconds)
        self._mono += seconds


def filing(accession: str, *, form: str = "8-K", filed: str = "2026-06-16", items: str = "2.02,9.01",
           acceptance: str = "2026-06-16T16:31:05.000Z", primary: str = "doc8k.htm", size: int = 23456) -> dict:
    return {"accessionNumber": accession, "filingDate": filed, "reportDate": filed, "acceptanceDateTime": acceptance,
            "act": "34", "form": form, "fileNumber": "001-36743", "filmNumber": "26412345", "items": items,
            "size": size, "isXBRL": 1, "isInlineXBRL": 1, "primaryDocument": primary,
            "primaryDocDescription": form}


def listing(cik: str, filings: list[dict], *, name: str = "Example Corp", files: bool = False) -> bytes:
    recent = {column: [f[column] for f in filings] for column in COLUMNS}
    doc = {"cik": cik10(cik).lstrip("0") or "0", "entityType": "operating", "name": name, "tickers": ["EXM"],
           "filings": {"recent": recent, "files": [{"name": f"CIK{cik10(cik)}-submissions-001.json"}] if files else []}}
    return json.dumps(doc).encode("utf-8")


class Reply:
    def __init__(self, body: bytes | None = None, *, status: int = 200, date: str | None = "auto",
                 content_type: str = "application/json", extra: list[tuple[str, str]] | None = None):
        self.body, self.status, self.date, self.content_type, self.extra = body, status, date, content_type, extra or []


class FakeFetcher:
    """routes: cik10 -> Reply, or a callable(count) -> Reply; an absent route is a connection failure."""

    def __init__(self, clock: SimClock):
        self.clock, self.routes, self.requests = clock, {}, []

    def fetch(self, url: str) -> FetchResult:
        self.requests.append(url)
        cik = url.rsplit("CIK", 1)[1].split(".", 1)[0]
        route = self.routes.get(cik)
        count = sum(1 for u in self.requests if u == url) - 1
        reply = route(count) if callable(route) else route
        self.clock.sleep(0.4)  # the network time of one request
        if reply is None:
            return FetchResult("SOURCE_UNAVAILABLE", reason="ConnectionRefusedError: synthetic")
        headers = [("Content-Type", reply.content_type), *reply.extra]
        if reply.date == "auto":
            headers.append(("Date", format_datetime(self.clock.true.replace(microsecond=0), usegmt=True)))
        elif reply.date is not None:
            headers.append(("Date", reply.date))
        kind, reason = classify_status(reply.status)
        return FetchResult(kind, reply.status, headers, reply.body if kind == "RESPONSE" else None, self.clock.wall(), reason)
