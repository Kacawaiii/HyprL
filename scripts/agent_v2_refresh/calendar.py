"""Daily one-pass calendar acquisition preserving source receipts and conservative exclusions."""

from datetime import date, datetime, timedelta
from hashlib import sha256
from html.parser import HTMLParser
import json, re, time
from pathlib import Path

from scripts.trading_lab.agent_v2_quality.contracts import digest, instant, iso, utc
from scripts.trading_lab.agent_v2_quality.producer import evaluation_end, atomic_json, transport


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data.strip())


def plain(raw):
    parser = Text()
    parser.feed(raw)
    return " ".join(parser.parts)


def acquire(output, grantpath, entry, *, clock=utc, send=transport, pause=time.sleep):
    now = clock()
    entry = date.fromisoformat(entry)
    grantpath = Path(grantpath)
    if now.date() != entry:
        raise RuntimeError("RUN_ONLY_ON_ENTRY_SESSION_DATE_WITH_REAL_CLOCK")
    target = instant(entry.isoformat() + "T12:45:00Z")
    end = evaluation_end(target)
    grant = json.loads(grantpath.read_text())
    if grant["id"] != "agent-v2-calendar-sources-v1" or now.date() > date.fromisoformat(
        grant["expires"]
    ):
        raise RuntimeError("CALENDAR_GRANT_INVALID")
    if (
        "api.nasdaq.com/api/calendar/earnings" not in grant["scope"]
        or "www.bls.gov" not in grant["scope"]
        or "www.federalreserve.gov" not in grant["scope"]
    ):
        raise RuntimeError("CALENDAR_SCOPE_INVALID")
    root = Path(output)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    sources = root / "calendar-sources"
    sources.mkdir(mode=0o700)
    months = sorted(
        {
            (d.year, d.month)
            for d in (entry + timedelta(days=i) for i in range((end.date() - entry).days + 1))
        }
    )
    urls = [
        (f"bls-{y}-{m:02}", f"https://www.bls.gov/schedule/{y}/{m:02}_sched.htm") for y, m in months
    ]
    urls += [
        ("bls-cpi", "https://www.bls.gov/schedule/news_release/cpi.htm"),
        ("bls-nfp", "https://www.bls.gov/schedule/news_release/empsit.htm"),
        ("fed-fomc", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"),
    ]
    days = [entry + timedelta(days=i) for i in range((end.date() - entry).days + 1)]
    urls += [
        (
            "nasdaq-" + d.isoformat(),
            "https://api.nasdaq.com/api/calendar/earnings?date=" + d.isoformat(),
        )
        for d in days
    ]
    receipts = []
    bodies = {}
    for name, url in urls:
        record = sources / (name + ".receipt.json")
        r = {
            "url": url,
            "method": "GET",
            "started_at": iso(clock()),
            "status": "RESERVED",
            "authorization_sha256": digest(grant),
            "authorization_file_sha256": sha256(grantpath.read_bytes()).hexdigest(),
        }
        atomic_json(record, r)
        # Never retry/redirect. Any failure halts the entire refresh, preserving reservation.
        status, raw = send(
            url,
            {"User-Agent": "HyprL-agent-v2-calendar/1.0", "Accept": "application/json,text/html"},
        )
        r.update(
            {
                "status": status,
                "received_at": iso(clock()),
                "body_sha256": sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
        atomic_json(record, r)
        if status != 200 or len(raw) > 12_000_000:
            raise RuntimeError("CALENDAR_HTTP_" + str(status) + ": " + url)
        (sources / (name + ".body")).write_bytes(raw)
        atomic_json(record, r)
        receipts.append(r)
        bodies[name] = raw
        print(name, status, r["body_sha256"], flush=True)
        pause(1.05)
    evidence = []
    events = []
    for d in days:
        name = "nasdaq-" + d.isoformat()
        payload = json.loads(bodies[name])
        receipt = next(r for r in receipts if r["url"].endswith("date=" + d.isoformat()))
        if (
            payload["status"]["rCode"] != 200
            or datetime.strptime(payload["data"]["asOf"], "%a, %b %d, %Y").date() != d
        ):
            raise RuntimeError("EARNINGS_SOURCE_DATE_INVALID")
        for row in payload["data"]["rows"] or []:
            evidence.append(
                {
                    "symbol": row["symbol"],
                    "date": d.isoformat(),
                    "time_label": row.get("time"),
                    "source_url": receipt["url"],
                    "source_hash": receipt["body_sha256"],
                    "received_at": receipt["received_at"],
                    "wire_row": row,
                }
            )
            events.append(
                {"kind": "EARNINGS", "at": d.isoformat() + "T23:59:59Z", "symbol": row["symbol"]}
            )
    for d in days:
        if d.weekday() >= 5:
            continue  # no NYSE entry/exit session; BLS weekday cells carry IDs
        body = bodies[f"bls-{d.year}-{d.month:02}"].decode("utf-8")
        cell = re.search(
            r'<td\b[^>]*id="d' + f"{d.month:02}{d.day:02}" + r'"[^>]*>(.*?)</td>', body, re.S
        )
        if cell is None:
            raise RuntimeError("BLS_COMPLETE_DATE_COVERAGE_REQUIRED")
        text = plain(cell.group(1))
        for label, kind in [("Consumer Price Index", "CPI"), ("Employment Situation", "NFP")]:
            if label in text:
                events.append({"kind": kind, "at": d.isoformat() + "T00:00:00Z", "symbol": None})
    fed = plain(bodies["fed-fomc"].decode("utf-8"))
    events += fomc_events(fed, entry, end.date())
    cal = {
        "schema": "agent-v2-quality-calendar-v2",
        "received_at": max(r["received_at"] for r in receipts),
        "source_hashes": [r["body_sha256"] for r in receipts],
        "covered_macro": ["CPI", "FOMC", "NFP"],
        "coverage_start": entry.isoformat() + "T00:00:00Z",
        "coverage_end": iso(end),
        "covered_symbols": sorted({e["symbol"] for e in evidence} | {"SPY"}),
        "events": events,
        "ready": False,
        "missing_earnings_dates": [],
        "not_applicable_symbols": {"SPY": "Broad-index ETF corporate earnings not applicable"},
        "coverage_notes": [
            "Granted Nasdaq date/session labels; issuer-confirmed versus estimated is unasserted.",
            "23:59:59Z earnings and 00:00:00Z macro markers are conservative civil-date exclusions, not release-hour estimates.",
            "No dated source row means UNKNOWN and exclusion; BLS complete monthly weekday cells cover every NYSE session; weekend sessions do not exist.",
        ],
    }
    atomic_json(root / "earnings-evidence.json", evidence)
    atomic_json(root / "calendar.json", cal)
    atomic_json(root / "calendar-source-manifest.json", receipts)
    return cal


def fomc_events(text, start, end):
    """Require a complete published annual schedule; never infer a meeting time."""
    from calendar import month_name, month_abbr
    months = {name.lower(): i for i in range(1, 13) for name in (month_name[i], month_abbr[i])}
    names = '|'.join(sorted(months, key=len, reverse=True))
    pattern = re.compile(r'\b(' + names + r')(?:/(' + names + r'))?\s+(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?', re.I)
    result = []
    for year in range(start.year, end.year + 1):
        heading = str(year) + ' FOMC Meetings'
        if heading not in text:
            raise RuntimeError('FOMC_YEAR_COVERAGE_REQUIRED')
        section = re.split(r'\b20\d{2} FOMC Meetings\b', text.split(heading, 1)[1], maxsplit=1)[0]
        meetings = set()
        for first_month, second_month, first, last in pattern.findall(section):
            first_day = date(year, months[first_month.lower()], int(first))
            last_day = date(year, months[(second_month or first_month).lower()], int(last or first))
            if not 0 <= (last_day - first_day).days <= 3:
                raise RuntimeError('FOMC_DATE_RANGE_INVALID')
            meetings.add((first_day, last_day))
        if len(meetings) < 8:
            raise RuntimeError('FOMC_COMPLETE_SCHEDULE_REQUIRED')
        for first_day, last_day in sorted(meetings):
            for offset in range((last_day-first_day).days+1):
                day = first_day + timedelta(days=offset)
                if start <= day <= end:
                    result.append({'kind': 'FOMC', 'at': day.isoformat()+'T00:00:00Z', 'symbol': None})
    return result
