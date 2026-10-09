"""Private runtime entry points. Dry-run never constructs a network/model runner."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

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
        crypto_day = session is None and ledger.grant.weekend_decision is not None
        if crypto_day:
            from .weekend import check
            check(ledger.grant, at)
        started = at >= instant('2026-10-06T12:00:00Z')
        state = "PAUSED" if ledger.paused else "HEALTHY"
        deadline = session.open_at if session else at.replace(hour=13, minute=30, second=0, microsecond=0)
        if not ledger.paused and started and (session or crypto_day) and at >= deadline and not any(r["status"] in {"COMPLETE", "DEGRADED"} for r in today):
            state = "MISSING_DAILY_RUN"
        if (not ledger.paused and any(r["status"] in {"FAILED", "SKIPPED_QUOTA"} for r in today)
                and not any(r['status'] in {'COMPLETE', 'DEGRADED'} for r in today)):
            state = "FAILED_DAILY_RUN"
        label_path = ledger.root / 'last-label.json'
        if not ledger.paused and started and (session or crypto_day) and at >= at.replace(hour=22, minute=0, second=0, microsecond=0):
            if not label_path.is_file() or json.loads(label_path.read_text()).get('at', '')[:10] != at.date().isoformat():
                state = 'MISSING_LABEL_JOB'
    except (TraderError, FileNotFoundError):
        state = "EXPIRED_OR_UNAVAILABLE"
    ledger.alert_state('health', state, initial=state not in {'HEALTHY', 'PAUSED'})
    payload = {"schema": "trader-health-v1", "at": iso(at), "state": state, "budget_counts": ledger.counts()}
    temp = ledger.root / "health.tmp"
    temp.write_text(json.dumps(payload))
    temp.replace(ledger.root / "health.json")
    return payload


def paper_executor(args, grant, runtime):
    from .alpaca_paper import PaperAuthorization, PaperExecutor, PaperLedger
    from .alpaca_data import QuoteFeed
    paper_grant = PaperAuthorization.load(args.paper_authorization, grant)
    # Shared production bank: runtime paths never reset order caps or peak equity.
    paper = PaperLedger(Path.home() / '.local/share/hyprl/trader-alpaca-paper', paper_grant)
    evidence = ResearchStore(runtime / 'evidence', read_only=True)
    return PaperExecutor(paper, evidence, quotes=QuoteFeed(paper_grant, args.data_authorization, args.paper_quotes),
                         pause_paths=(runtime / 'PAUSED', Path.home() / '.local/share/hyprl/trader-agent-budget/PAUSED'))


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "recover", "catchup", "catchup-after-hours", "label", "health", "status", "pause", "resume", 'gpt-preflight',
                                          'paper-execute', 'paper-exit', 'paper-report', 'paper-status', 'paper-rebind', 'paper-register-weekend'))
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--fomc-store")
    parser.add_argument("--edgar-store")
    parser.add_argument('--paper-authorization')
    parser.add_argument('--paper-account', choices=('ia_actions', 'ia_crypto'))
    parser.add_argument('--paper-quotes')
    parser.add_argument('--operator-decision')
    parser.add_argument('--paper-replay-at')
    parser.add_argument('--data-authorization')
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-time", default="2026-10-06T12:00:00Z")
    args = parser.parse_args(argv)
    grant = Authorization.load(args.authorization)
    if args.action.startswith('paper-'):
        if not args.paper_authorization:
            parser.error('--paper-authorization is required for paper actions')
        runtime = private_root(args.runtime)
        if args.paper_replay_at and (args.action != 'paper-execute' or not args.dry_run):
            parser.error('--paper-replay-at requires paper-execute --dry-run')
        if args.action == 'paper-register-weekend':
            if not args.operator_decision or args.dry_run:
                parser.error('paper-register-weekend requires --operator-decision and forbids --dry-run')
            from .alpaca_paper import PaperAuthorization, PaperLedger
            paper_grant = PaperAuthorization.load(args.paper_authorization, grant)
            paper = PaperLedger(Path.home() / '.local/share/hyprl/trader-alpaca-paper', paper_grant)
            print(json.dumps({'state': 'REGISTERED', 'registration': paper.register_weekend(args.operator_decision)}, indent=2))
            return 0
        if args.action == 'paper-rebind':
            if not args.operator_decision:
                parser.error('--operator-decision is required for paper-rebind')
            from .alpaca_paper import PaperAuthorization, PaperLedger
            paper_grant = PaperAuthorization.load(args.paper_authorization, grant)
            rebound = PaperLedger.rebind(Path.home() / '.local/share/hyprl/trader-alpaca-paper', paper_grant,
                                        args.operator_decision)
            print(json.dumps({'state': 'REBOUND', 'binding': rebound.events(event='binding')[-1]}, indent=2))
            return 0
        if args.action == 'paper-report' and calendar_session(now().date().isoformat()) is None:
            print(json.dumps({'state': 'SKIPPED_CLOSED_EQUITY_MARKET'}))
            return 0
        executor = paper_executor(args, grant, runtime)
        if args.action == 'paper-report':
            from .paper_reporting import benchmarks
            result = executor.report(benchmarks(executor.ledger, runtime, grant))
            temp = runtime / 'paper-report.tmp'
            temp.write_text(json.dumps(result, indent=2) + '\n')
            temp.replace(runtime / 'paper-report.json')
        else:
            result = executor.run(args.action.removeprefix('paper-'), dry_run=args.dry_run,
                                  accounts=[args.paper_account] if args.paper_account else None, replay_at=args.paper_replay_at)
        print(json.dumps(result, indent=2))
        return 1 if result.get('state') == 'BLOCKED' else 0
    clock = Clock(instant(args.synthetic_time)) if args.dry_run else now
    runtime = private_root(args.runtime)
    if args.dry_run:
        runtime = private_root(runtime / "synthetic")
    # One production budget bank/owner across runtime paths: changing --runtime never refunds grants.
    bank = None if args.dry_run else Path.home() / '.local/share/hyprl/trader-agent-budget'
    ledger = Ledger(runtime, grant, clock=clock, budget_root=bank)
    if args.action == 'gpt-preflight':
        if args.dry_run:
            parser.error('gpt-preflight requires the real GPT/web boundary; use offline tests for a synthetic CLI')
        from .preflight import preflight
        result = preflight(ledger, ModelRunner(ledger))
    elif args.action in {"pause", "resume"}:
        marker = ledger.budget_root / "PAUSED"
        if args.action == "pause":
            marker.touch(mode=0o600)
        else:
            marker.unlink(missing_ok=True)
            (runtime / 'PAUSED').unlink(missing_ok=True)
        result = {"state": "PAUSED" if args.action == "pause" else "RESUMED"}
    elif args.action in {"health", "status"}:
        result = health(ledger)
    else:
        data = SyntheticData(ledger, clock) if args.dry_run else PublicData(ledger)
        runner = SyntheticRunner(ledger, clock=clock) if args.dry_run else ModelRunner(ledger)
        service = TraderService(ledger, data, runner, clock=clock,
                                fomc=None if args.dry_run else args.fomc_store,
                                edgar=None if args.dry_run else args.edgar_store)
        if args.action == 'recover':
            from .recovery import recover
            result = recover(service)
            # Close-entry recovery remains score-only. Only a fresh, successful
            # after-hours recovery may invoke the existing paper executor.
            if result.get('mode') == 'after_hours' and result.get('status') in {'COMPLETE', 'DEGRADED'} and args.paper_authorization and not args.dry_run:
                try:
                    result['paper_execution'] = paper_executor(args, grant, runtime).run('execute', accounts=['ia_actions'])
                    ledger.alert('RECOVERY_PAPER_' + result['paper_execution']['state'])
                except TraderError as error:
                    ledger.alert('RECOVERY_PAPER_' + error.code)
                    result['paper_execution'] = {'state': 'BLOCKED', 'reason': error.code}
                except Exception:
                    ledger.alert('RECOVERY_PAPER_UNEXPECTED_FAILURE')
                    raise
        elif args.action in {"catchup", 'catchup-after-hours'}:
            result = service.run(catchup=True, after_hours=args.action == 'catchup-after-hours')
        elif args.action == "run":
            result = service.run()
            if args.dry_run and result["status"] in {"COMPLETE", "DEGRADED"}:
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
    if result.get("status") == "FAILED" or result.get("state") == "BLOCKED" or result.get('paper_execution', {}).get('state') == 'BLOCKED':
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
