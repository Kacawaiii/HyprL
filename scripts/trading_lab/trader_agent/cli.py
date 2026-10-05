"""Private runtime entry points. Dry-run never constructs a network/model runner."""
import argparse
from datetime import timedelta
import json
import os

from scripts.trading_lab.research.store import ResearchStore

from .config import Authorization, TraderError, instant, iso, now, private_root
from .data import PublicData, calendar_session
from .ledger import Ledger
from .runners import ModelRunner
from .scoring import realize, scorecard, weekly_report
from .service import TraderService
from .synthetic import Clock, SyntheticData, SyntheticRunner


def health(ledger):
    at = ledger.clock()
    try:
        ledger.grant.check(at)
        with ResearchStore(ledger.root / "evidence", read_only=True).connect() as db:
            rows = db.execute("SELECT payload FROM records WHERE kind='replay-summary' ORDER BY sequence DESC LIMIT 100").fetchall()
        runs = [json.loads(r[0]) for r in rows if json.loads(r[0]).get("schema") == "trader-run-v1"]
        today = [r for r in runs if r["at"][:10] == at.date().isoformat() and not r["synthetic"]]
        session = calendar_session(at.date().isoformat())
        started = at >= instant('2026-10-06T12:00:00Z')
        state = "PAUSED" if (ledger.root / "PAUSED").exists() else "HEALTHY"
        if started and session and at >= session.open_at and not any(r["status"] == "COMPLETE" for r in today):
            state = "MISSING_DAILY_RUN"
        if any(r["status"] in {"FAILED", "SKIPPED_QUOTA"} for r in today):
            state = "FAILED_DAILY_RUN"
        label_path = ledger.root / 'last-label.json'
        if started and session and at >= at.replace(hour=22, minute=0, second=0, microsecond=0):
            if not label_path.is_file() or json.loads(label_path.read_text()).get('at', '')[:10] != at.date().isoformat():
                state = 'MISSING_LABEL_JOB'
    except (TraderError, FileNotFoundError):
        state = "EXPIRED_OR_UNAVAILABLE"
    if state != "HEALTHY":
        ledger.alert(state)
    payload = {"schema": "trader-health-v1", "at": iso(at), "state": state, "budget_counts": ledger.counts()}
    temp = ledger.root / "health.tmp"
    temp.write_text(json.dumps(payload))
    temp.replace(ledger.root / "health.json")
    return payload


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "label", "health", "status", "pause", "resume"))
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--fomc-store")
    parser.add_argument("--edgar-store")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-time", default="2026-10-06T12:00:00Z")
    args = parser.parse_args(argv)
    grant = Authorization.load(args.authorization)
    clock = Clock(instant(args.synthetic_time)) if args.dry_run else now
    runtime = private_root(args.runtime)
    if args.dry_run:
        runtime = private_root(runtime / "synthetic")
    ledger = Ledger(runtime, grant, clock=clock)
    if args.action in {"pause", "resume"}:
        marker = runtime / "PAUSED"
        if args.action == "pause":
            marker.touch(mode=0o600)
        else:
            marker.unlink(missing_ok=True)
        result = {"state": "PAUSED" if args.action == "pause" else "RESUMED"}
    elif args.action in {"health", "status"}:
        result = health(ledger)
    else:
        data = SyntheticData(ledger, clock) if args.dry_run else PublicData(ledger)
        runner = SyntheticRunner(ledger, clock=clock) if args.dry_run else ModelRunner(ledger)
        service = TraderService(ledger, data, runner, clock=clock,
                                fomc=None if args.dry_run else args.fomc_store,
                                edgar=None if args.dry_run else args.edgar_store)
        if args.action == "run":
            result = service.run()
            if args.dry_run and result["status"] == "COMPLETE":
                initial_labels = service.store.records("label")
                # Advance to after every 5d endpoint: all are future at issue time.
                clock.at += timedelta(days=9, hours=10)
                result = {"synthetic": True, "run_status": result["status"], "views": len(result["decision"]["views"]),
                    "initial_labels": len(initial_labels), "label_job": realize(service.store, ledger, data, at=clock()),
                    "chain": service.store.verify(), "scorecard": scorecard(service.store, synthetic=True)}
        else:
            result = realize(service.store, ledger, data, at=clock())
            result["scorecard"] = scorecard(service.store, synthetic=args.dry_run)
            if clock().weekday() == 4:
                result["weekly"] = weekly_report(service.store, runtime, at=clock(), synthetic=args.dry_run)["identity"]
        if args.action == "label":
            (runtime / "last-label.json").write_text(json.dumps({"at": iso(clock()), "state": result["state"]}))
    print(json.dumps(result, indent=2))
    if result.get("status") == "FAILED" or result.get("state") == "BLOCKED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
