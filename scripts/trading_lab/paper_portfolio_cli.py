"""Local control for the shared shadow portfolio. The only way to start one.

Deliberately not an HTTP endpoint, for the same reason as Phase 5D: the
read-only API serves a cockpit, and a cockpit that can start a trading process
is one cross-site request away from doing it by accident.

Nothing here connects to a broker, holds a key, or moves money.

The Phase 5D per-product runtime is not started by this command any more. Its
database stays exactly where it is, readable and exportable, because two
independent accounts are not the history of a shared portfolio and merging
them would be a claim nobody can support.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import signal
import sys
from datetime import datetime, timedelta, timezone

DEFAULT_RUNTIME_DIR = pathlib.Path("var/trading_lab")
DEFAULT_MODEL_DIR = pathlib.Path("data/models/paper_v1")
DEFAULT_CORPUS = pathlib.Path("data/crypto")
SESSION_ID_FORMAT = "portfolio-%Y%m%dT%H%M%SZ"
DEFAULT_INSTRUMENTS = ("coinbase:BTC-USD", "coinbase:ETH-USD")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def _layout(runtime):
    from scripts.trading_lab.ops.runtime_paths import RuntimeLayout
    return RuntimeLayout(runtime).ensure()


def _load_models(model_dir: pathlib.Path, instruments):
    from scripts.trading_lab.identity import resolve_instrument
    from scripts.trading_lab.paper_model import load_paper_model, read_artifact

    models = {}
    for name in instruments:
        spec = resolve_instrument(name, context="portfolio instrument")
        path = model_dir / f"{spec.legacy_product_id}.json"
        if not path.is_file():
            raise SystemExit(
                f"no shadow model for {spec.canonical_id} at {path.name}; train "
                "and freeze the models before starting a session")
        # The artefact must agree with the slot it is loaded into: five hashes
        # verified and the product ignored is how an ETH model predicts BTC.
        models[spec.canonical_id] = load_paper_model(
            read_artifact(path), product=spec.legacy_product_id)
    return models


def _seed_history(corpus_root: pathlib.Path, instrument: str, bars: int):
    from scripts.trading_lab.capture_market_history import (
        CORPUS_ID, load_canonical_rows, load_manifest)
    from scripts.trading_lab.identity import resolve_instrument

    spec = resolve_instrument(instrument, context="portfolio instrument")
    manifest = load_manifest(corpus_root)
    entry = next(item for item in manifest["products"]
                 if item["product"] == spec.legacy_product_id)
    rows = load_canonical_rows(corpus_root / CORPUS_ID / entry["canonical_path"])
    return [dict(row) for row in rows[-bars:]]


def _store(layout):
    from scripts.trading_lab.paper_portfolio_store import PaperPortfolioStore
    return PaperPortfolioStore(layout.paper_portfolio_database)


def command_status(arguments) -> int:
    from scripts.trading_lab.portfolio import PORTFOLIO_SPEC_V1
    from scripts.trading_lab.protected_holdout import embargo_state

    layout = _layout(arguments.runtime)
    marker = layout.paper_portfolio_session_marker
    active = json.loads(marker.read_text()) if marker.is_file() else None
    payload = {
        "mode": "SHARED_PORTFOLIO",
        "active_session": active,
        "runtime": str(layout.root),
        "portfolio_spec_hash": PORTFOLIO_SPEC_V1.portfolio_spec_hash,
        "initial_equity": str(PORTFOLIO_SPEC_V1.initial_equity),
        "max_gross_exposure": str(PORTFOLIO_SPEC_V1.max_gross_exposure),
        "real_money": False,
        "broker_connected": False,
        "shadow_mode": True,
        "shared_capital": True,
        "legacy_individual_accounts_available":
            layout.paper_database.is_file(),
        "embargo": {name: embargo_state(name, now=_iso(_now()))
                    for name in arguments.instruments},
    }
    if layout.paper_portfolio_database.is_file():
        store = _store(layout)
        sessions = store.sessions()
        payload["sessions"] = len(sessions)
        if sessions:
            latest = sessions[-1]
            payload["latest_session"] = latest
            payload["events"] = store.count(session_id=latest)
            try:
                payload["chain"] = store.verify_chain(session_id=latest)
            except Exception as error:
                payload["chain"] = {"verified": False, "error": str(error)}
            snapshot = store.latest_snapshot(session_id=latest)
            payload["snapshot_verified"] = snapshot is not None
            if snapshot:
                payload["state"] = snapshot["state"].get("state")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def command_stop(arguments) -> int:
    layout = _layout(arguments.runtime)
    marker = layout.paper_portfolio_session_marker
    if not marker.is_file():
        print("no active shared portfolio session")
        return 0
    active = json.loads(marker.read_text())
    if layout.paper_portfolio_database.is_file():
        from scripts.trading_lab.paper_portfolio_store import (
            PaperPortfolioStoreError)
        store = _store(layout)
        try:
            store.append(session_id=active["session_id"],
                         event_type="PORTFOLIO_SESSION_STOPPED",
                         event_at=_iso(_now()), natural_key="session",
                         payload={"reason": "stopped from the command line"})
        except PaperPortfolioStoreError as error:
            print(f"note: {error}")
    marker.unlink()
    print(f"stopped shared portfolio session {active['session_id']}")
    return 0


def command_start(arguments) -> int:
    from scripts.trading_lab.live_market import (
        LiveMarketStatus, poll_closed_candles)
    from scripts.trading_lab.identity import resolve_instrument
    from scripts.trading_lab.paper_portfolio import (
        PaperPortfolioEngine, build_portfolio_session_spec)
    from scripts.trading_lab.protected_holdout import (
        ProtectedHoldoutError, embargo_state)

    layout = _layout(arguments.runtime)
    marker = layout.paper_portfolio_session_marker
    if marker.is_file():
        raise SystemExit(
            f"a shared portfolio session is already active ({marker.name}); "
            "stop it first")

    instruments = tuple(
        resolve_instrument(name, context="portfolio instrument").canonical_id
        for name in arguments.instruments)
    models = _load_models(pathlib.Path(arguments.models), instruments)
    store = _store(layout)
    session_spec = build_portfolio_session_spec(models, instruments=instruments)
    session_id = _now().strftime(SESSION_ID_FORMAT)
    engine = PaperPortfolioEngine(store=store, models=models,
                                  session_id=session_id,
                                  session_spec=session_spec)

    corpus = pathlib.Path(arguments.corpus)
    seeded = {}
    for instrument in instruments:
        rows = _seed_history(corpus, instrument, arguments.warmup)
        engine.seed_history(instrument, rows)
        seeded[instrument] = rows[-1]["bar_open_at"] if rows else None
    engine.start(now=_iso(_now()))
    marker.write_text(json.dumps({
        "session_id": session_id, "instruments": list(instruments),
        "session_spec_hash": session_spec.session_spec_hash,
        "portfolio_spec_hash": session_spec.portfolio_spec_hash,
        "mode": "SHARED_PORTFOLIO",
        "started_at": _iso(_now())}, indent=2))

    stopping = {"now": False}

    def _handle(signum, frame):
        stopping["now"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(json.dumps({"session_id": session_id, "mode": "SHARED_PORTFOLIO",
                      "session_spec_hash": session_spec.session_spec_hash,
                      "instruments": list(instruments), "shadow_mode": True,
                      "real_money": False, "shared_capital": True}, indent=2))
    try:
        for instrument in instruments:
            state = embargo_state(instrument, now=_iso(_now()))
            if state["embargoed"]:
                print(f"{instrument}: EMBARGOED — {state['reason']}")
                continue
            spec = resolve_instrument(instrument)
            try:
                rows = poll_closed_candles(
                    spec.legacy_product_id, now=_now(),
                    since=seeded.get(instrument),
                    max_candles=arguments.max_candles)
            except ProtectedHoldoutError as error:
                print(f"{instrument}: EMBARGOED — {error}")
                continue
            except Exception as error:
                print(f"{instrument}: {LiveMarketStatus.DEGRADED} — {error}")
                continue
            for row in rows:
                outcome = engine.ingest_candle(instrument, row,
                                               now=_iso(_now()),
                                               row_instrument=instrument)
                print(json.dumps(outcome, default=str))
    finally:
        engine.stop(now=_iso(_now()))
        if marker.is_file():
            marker.unlink()
    return 0


def command_restart(arguments) -> int:
    command_stop(arguments)
    return command_start(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HyprL shared shadow portfolio. No real money, no broker, "
                    "no exchange key.")
    parser.add_argument("command",
                        choices=("start", "status", "stop", "restart"))
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--models", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--instruments", nargs="+",
                        default=list(DEFAULT_INSTRUMENTS))
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--max-candles", dest="max_candles", type=int, default=24)
    parser.add_argument("--once", action="store_true",
                        help="poll once and exit instead of staying resident")
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    return {"start": command_start, "status": command_status,
            "stop": command_stop, "restart": command_restart}[arguments.command](
                arguments)


if __name__ == "__main__":                            # pragma: no cover
    sys.exit(main())
