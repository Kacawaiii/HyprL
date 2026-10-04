"""The EDGAR capture runner. It sends nothing without an operator authorization file that names this spec,
the CIKs, a request budget, an expiry and the declared User-Agent; it stops at the first of: the budget, the
expiry, a stop signal. `--check` validates the spec binding, the store and the authorization without
writing a byte or opening a socket.

    python -m scripts.trading_lab.edgar.service --store DIR --authorization FILE --check
    python -m scripts.trading_lab.edgar.service --store DIR --authorization FILE

Authorization file (JSON): {"authorizes": "sec_edgar_submissions_v1", "spec_hash": "<edgar spec hash>",
"ciks": ["320193"], "max_requests": 6, "not_after": "2026-10-05T18:00:00+00:00",
"user_agent": "Organization contact@example.org", "granted_by": "operator", "granted_at": "..."}
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import threading
import time

from scripts.trading_lab.edgar import spec
from scripts.trading_lab.edgar.collector import EdgarCollector, RequestCancelled
from scripts.trading_lab.edgar.listing import cik10
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.edgar.transport import HttpsFetcher
from scripts.trading_lab.sources.httpclock import iso, parse_iso
from scripts.trading_lab.sources.store import admit_existing

MAX_AUTHORIZED_REQUESTS = 50


class CaptureRefused(RuntimeError):
    """No valid authorization: nothing is sent."""


class RealClock:
    def __init__(self, stop: threading.Event):
        self._stop = stop

    def wall(self) -> datetime:
        return datetime.now(timezone.utc)

    def mono(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        self._stop.wait(max(seconds, 0.0))


def load_authorization(path: Path, now: datetime) -> dict:
    try:
        auth = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureRefused(f"no readable authorization at {path}: {exc}") from exc
    problems = []
    if auth.get("authorizes") != spec.PROVIDER_ID:
        problems.append(f"it authorizes {auth.get('authorizes')!r}, not {spec.PROVIDER_ID}")
    if auth.get("spec_hash") != spec.SPEC_HASH:
        problems.append("it names another EDGAR spec revision")
    ciks = auth.get("ciks")
    try:
        padded = [cik10(c) for c in ciks] if isinstance(ciks, list) else None
    except ValueError as exc:
        padded = None
        problems.append(str(exc))
    if not padded or not 1 <= len(padded) <= spec.WATCHLIST_MAX:
        problems.append(f"it must name 1 to {spec.WATCHLIST_MAX} CIKs")
    budget = auth.get("max_requests")
    if not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= MAX_AUTHORIZED_REQUESTS:
        problems.append(f"max_requests must be an integer from 1 to {MAX_AUTHORIZED_REQUESTS}")
    try:
        not_after = parse_iso(auth["not_after"])
        if not_after <= now:
            problems.append(f"it expired at {auth['not_after']}")
    except (KeyError, TypeError, ValueError):
        problems.append("not_after must be an ISO-8601 instant with its offset")
    if not isinstance(auth.get("user_agent"), str) or not spec.USER_AGENT.match(auth["user_agent"].strip()):
        problems.append("user_agent must name an organization and a contact e-mail")
    if not auth.get("granted_by"):
        problems.append("granted_by is required")
    if problems:
        raise CaptureRefused("authorization refused: " + "; ".join(problems))
    return dict(auth, ciks=padded)


def check(store_dir: Path, authorization: Path, *, now: datetime | None = None) -> dict:
    """Preflight without network or write: spec binding, store opening rule, authorization."""
    out = {"spec": [spec.SPEC_REVISION, spec.verify_spec_binding()]}
    admit_existing(Path(store_dir) / EdgarStore.DB_NAME, schema_version=spec.SCHEMA_VERSION, spec_hash=spec.SPEC_HASH)
    auth = load_authorization(authorization, now or datetime.now(timezone.utc))
    out.update(store=str(store_dir), ciks=auth["ciks"], max_requests=auth["max_requests"], not_after=auth["not_after"])
    return out


def run(store_dir: Path, authorization: Path, *, fetcher=None, clock=None, stop: threading.Event | None = None,
        log=print) -> dict:
    stop = stop or threading.Event()
    clock = clock or RealClock(stop)
    auth = load_authorization(authorization, clock.wall())
    not_after = parse_iso(auth["not_after"])
    store = EdgarStore(Path(store_dir), wall_clock=clock.wall)
    fetcher = fetcher or HttpsFetcher(auth["user_agent"], wall=clock.wall, mono=clock.mono)
    collector = EdgarCollector(store, fetcher, clock, boot_id=f"boot-{int(time.time())}")
    try:
        collector.submit_watchlist(auth["ciks"])
        log(f"EDGAR capture: epoch {collector.epoch}, reconciled {collector.reconciled}, "
            f"budget {auth['max_requests']} requests until {auth['not_after']}")

        def sent() -> int:
            return sum(1 for t in store.rows("TRANSPORT_INVOKED") if t.body["epoch"] == collector.epoch)

        def permit_request() -> None:
            if stop.is_set():
                raise RequestCancelled("stopped")
            if clock.wall() >= not_after:
                raise RequestCancelled("authorization expired")

        reason = None
        while reason is None:
            for cik in collector.watchlist():
                if stop.is_set():
                    reason = "stopped"
                elif sent() >= auth["max_requests"]:
                    reason = "request budget spent"
                elif clock.wall() >= not_after:
                    reason = "authorization expired"
                if reason:
                    break
                try:
                    result = collector.poll(cik, before_request=permit_request)
                except RequestCancelled as exc:
                    reason = str(exc)
                    break
                log(f"{iso(clock.wall())} {cik} {result['status']} {result.get('outcome', '')}")
                if result["status"] == "INTERRUPTED":
                    reason = result["reason"]
                    break
                if result["status"] == "SOURCE_THROTTLED":
                    reason = "throttled (403/429): the trial stops, no further request"
                    break
            if reason is None:
                clock.sleep(spec.POLL_INTERVAL_S)
        summary = {"reason": reason, "requests": sent(), "records": len(store.rows("RESPONSE")),
                   "revisions": len(store.rows("FILING_REVISION")), "epoch": collector.epoch}
        log(f"EDGAR capture ended: {summary}")
        return summary
    finally:
        collector.close()
        store.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="validate only: no network, no write")
    args = parser.parse_args(argv)
    try:
        if args.check:
            print(json.dumps(check(args.store, args.authorization), sort_keys=True))
            return 0
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
        signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
        run(args.store, args.authorization, stop=stop)
        return 0
    except CaptureRefused as exc:
        print(f"EDGAR capture refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
