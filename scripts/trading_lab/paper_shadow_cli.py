"""Local control for the shadow session. The only way to start or stop one.

Deliberately not an HTTP endpoint: the read-only API serves a cockpit, and a
cockpit that can start a trading process is one XSS away from doing it by
accident. Control stays on the command line, where it takes a deliberate act.

Nothing here connects to a broker, holds a key, or moves money.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import signal
import sys
from datetime import datetime, timedelta, timezone

DEFAULT_RUNTIME_DIR = pathlib.Path("var/trading_lab")
DEFAULT_DATABASE = "paper_v1.sqlite"
DEFAULT_MODEL_DIR = pathlib.Path("data/models/paper_v1")
DEFAULT_CORPUS = pathlib.Path("data/crypto")
SESSION_ID_FORMAT = "paper-%Y%m%dT%H%M%SZ"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def _load_models(model_dir: pathlib.Path, products):
    from scripts.trading_lab.paper_model import load_paper_model, read_artifact
    models = {}
    for product in products:
        path = model_dir / f"{product}.json"
        if not path.is_file():
            raise SystemExit(
                f"no shadow model for {product} at {path}; train and freeze the models "
                "before starting a session")
        models[product] = load_paper_model(read_artifact(path))
    return models


def _seed_history(corpus_root: pathlib.Path, product: str, bars: int):
    from scripts.trading_lab.capture_market_history import (
        CORPUS_ID, load_canonical_rows, load_manifest)
    manifest = load_manifest(corpus_root)
    entry = next(item for item in manifest["products"] if item["product"] == product)
    rows = load_canonical_rows(corpus_root / CORPUS_ID / entry["canonical_path"])
    return [dict(row) for row in rows[-bars:]]


def _state_path(runtime: pathlib.Path) -> pathlib.Path:
    return runtime / "paper_session.json"


def command_status(arguments) -> int:
    from scripts.trading_lab.paper_event_store import PaperEventStore
    from scripts.trading_lab.protected_holdout import embargo_state
    runtime = pathlib.Path(arguments.runtime)
    marker = _state_path(runtime)
    active = json.loads(marker.read_text()) if marker.is_file() else None
    payload = {
        "active_session": active,
        "runtime": str(runtime),
        "database": str(runtime / DEFAULT_DATABASE),
        "real_money": False,
        "broker_connected": False,
        "shadow_mode": True,
        "embargo": {product: embargo_state(product, now=_iso(_now()))
                    for product in arguments.products},
    }
    database = runtime / DEFAULT_DATABASE
    if database.is_file() and active:
        store = PaperEventStore(database)
        payload["events"] = store.count(session_id=active["session_id"])
        payload["chain"] = store.verify_chain(session_id=active["session_id"])
    print(json.dumps(payload, indent=2))
    return 0


def command_stop(arguments) -> int:
    from scripts.trading_lab.paper_event_store import PaperEventStore
    runtime = pathlib.Path(arguments.runtime)
    marker = _state_path(runtime)
    if not marker.is_file():
        print("no active shadow session")
        return 0
    active = json.loads(marker.read_text())
    store = PaperEventStore(runtime / DEFAULT_DATABASE)
    try:
        store.append(session_id=active["session_id"], event_type="SESSION_STOPPED",
                     event_at=_iso(_now()), natural_key="session",
                     payload={"reason": "stopped from the command line"})
    except Exception as error:            # already stopped is not a failure
        print(f"note: {error}")
    marker.unlink()
    print(f"stopped session {active['session_id']}")
    return 0


def command_start(arguments) -> int:
    from scripts.trading_lab.live_market import (
        LiveMarketStatus, poll_closed_candles)
    from scripts.trading_lab.paper_engine import PaperEngine, build_session_spec
    from scripts.trading_lab.paper_event_store import PaperEventStore
    from scripts.trading_lab.protected_holdout import (
        ProtectedHoldoutError, embargo_state)

    runtime = pathlib.Path(arguments.runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    marker = _state_path(runtime)
    if marker.is_file():
        raise SystemExit(f"a shadow session is already active ({marker}); stop it first")

    products = tuple(arguments.products)
    models = _load_models(pathlib.Path(arguments.models), products)
    store = PaperEventStore(runtime / DEFAULT_DATABASE)
    session_spec = build_session_spec(models, products=products)
    session_id = _now().strftime(SESSION_ID_FORMAT)
    engine = PaperEngine(store=store, models=models, session_id=session_id,
                         session_spec=session_spec)

    corpus = pathlib.Path(arguments.corpus)
    for product in products:
        engine.seed_history(product, _seed_history(corpus, product, arguments.warmup))
    engine.start(now=_iso(_now()))
    marker.write_text(json.dumps({
        "session_id": session_id, "products": list(products),
        "session_spec_hash": session_spec.session_spec_hash,
        "started_at": _iso(_now())}, indent=2))

    stopping = {"now": False}

    def _handle(signum, frame):
        stopping["now"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(json.dumps({"session_id": session_id,
                      "session_spec_hash": session_spec.session_spec_hash,
                      "products": list(products), "shadow_mode": True,
                      "real_money": False}, indent=2))
    try:
        for product in products:
            state = embargo_state(product, now=_iso(_now()))
            if state["embargoed"]:
                print(f"{product}: EMBARGOED — {state['reason']}")
                continue
            try:
                rows = poll_closed_candles(product, now=_now(),
                                           max_candles=arguments.max_candles)
            except ProtectedHoldoutError as error:
                print(f"{product}: EMBARGOED — {error}")
                continue
            except Exception as error:
                print(f"{product}: {LiveMarketStatus.DEGRADED} — {error}")
                continue
            for row in rows:
                outcome = engine.ingest_candle(product, row, now=_iso(_now()))
                print(json.dumps(outcome))
            if arguments.once:
                continue
    finally:
        engine.stop(now=_iso(_now()))
        if marker.is_file():
            marker.unlink()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HyprL shadow trading. No real money, no broker, no exchange key.")
    parser.add_argument("command", choices=("start", "status", "stop"))
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--models", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--products", nargs="+", default=["BTC-USD", "ETH-USD"])
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--max-candles", dest="max_candles", type=int, default=24)
    parser.add_argument("--once", action="store_true",
                        help="poll once and exit instead of staying resident")
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    return {"start": command_start, "status": command_status,
            "stop": command_stop}[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
