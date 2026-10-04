"""One entry point for running HyprL locally.

Before this, using the application meant knowing that the API is a Python
module, that the cockpit is a Vite project, that they must both be running,
and which ports each expects. That is a reasonable ask of the person who
wrote it and an unreasonable one of the person using it a month later.

``start`` builds the frontend if needed, launches one server that serves both
the app and its API from a single origin, and returns. ``stop`` ends it.
Nothing here requires the user to think about a PID.

Two things stay on the command line and never move to HTTP: starting the
application, and starting the shadow trading engine. A cockpit that could
start a trading process is one cross-site request away from doing it without
being asked.

Dev mode is untouched. ``scripts/dev_app.sh`` still runs Vite with hot reload
against the same API.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import webbrowser
from datetime import datetime, timezone

from scripts.trading_lab.ops import doctor as doctor_module
from scripts.trading_lab.ops import recovery as recovery_module
from scripts.trading_lab.ops import runtime_export, settings as settings_module
from scripts.trading_lab.ops import supervisor, support_bundle
from scripts.trading_lab.ops.health import (
    DEGRADED, ERROR, HEALTHY, HealthHistory, STOPPED)
from scripts.trading_lab.ops.runtime_paths import RuntimeLayout
from scripts.trading_lab.ops.static_assets import StaticSite
from scripts.trading_lab.ops.structured_log import StructuredLogger

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_DATA_ROOT = "data/crypto"
FRONTEND_DIR = pathlib.Path("apps/web")
DIST_DIR = FRONTEND_DIR / "dist"


def _root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _context(arguments):
    root = _root()
    layout = RuntimeLayout(root / arguments.runtime).ensure()
    stored = settings_module.load(layout.settings_file)
    logger = StructuredLogger(layout.application_log,
                              **{"max_files": settings_module.log_retention(stored)
                                 ["max_files"]})
    health = HealthHistory(layout.ops_database)
    return root, layout, stored, logger, health


def _emit(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


# --- application lifecycle -------------------------------------------------


def command_build(arguments) -> int:
    """Build the production frontend. The only command that needs npm."""
    root = _root()
    web = root / FRONTEND_DIR
    if not (web / "node_modules").is_dir():
        print("[hyprl] installing frontend packages from the lockfile…")
        result = subprocess.run(["npm", "ci", "--no-audit", "--no-fund"],  # noqa: S603
                                cwd=web)
        if result.returncode:
            return result.returncode
    print("[hyprl] building the production frontend…")
    result = subprocess.run(["npm", "run", "build"], cwd=web)  # noqa: S603
    if result.returncode:
        return result.returncode
    site = StaticSite(root / DIST_DIR)
    _emit({"built": site.available, **site.describe()})
    return 0


def command_release(arguments) -> int:
    """Assemble a local release bundle from the current build."""
    from scripts.trading_lab.ops import release as release_module

    root = _root()
    if not (root / DIST_DIR / "index.html").is_file():
        code = command_build(arguments)
        if code:
            return code
    try:
        report = release_module.build_release(
            root=root, output=arguments.output or (root / "dist/hyprl-local"),
            include_research_data=not arguments.without_research_data)
    except release_module.ReleaseError as error:
        print(f"[hyprl] {error}")
        return 2
    _emit(report)
    return 0


def command_verify_release(arguments) -> int:
    from scripts.trading_lab.ops import release as release_module

    try:
        report = release_module.verify_release(arguments.path)
    except release_module.ReleaseError as error:
        print(f"[hyprl] {error}")
        return 2
    _emit(report)
    return 0 if report["ok"] else 2


def command_start(arguments) -> int:
    root, layout, stored, logger, health = _context(arguments)
    site = StaticSite(root / DIST_DIR)
    if not site.available:
        if arguments.no_build:
            print("[hyprl] no frontend build and --no-build was given; "
                  "serving the API only")
        else:
            code = command_build(arguments)
            if code:
                return code
            site = StaticSite(root / DIST_DIR)

    command = [
        supervisor.python_executable(), "-m", "scripts.trading_lab.app_api.server",
        "--data-root", str(root / arguments.data_root),
        "--host", arguments.host, "--port", str(arguments.port),
        "--marker", supervisor.PROCESS_MARKER,
    ]
    if site.available:
        command += ["--dist-root", str(root / DIST_DIR)]
    # Official event-source stores, opened read-only by the API (an archive or a copy of a capture store).
    for flag, value in (("--fomc-store", arguments.fomc_store), ("--edgar-store", arguments.edgar_store)):
        if value:
            command += [flag, str(pathlib.Path(value).expanduser().resolve())]

    try:
        result = supervisor.start(
            pid_file=layout.pid_file, command=command, host=arguments.host,
            port=arguments.port, cwd=root, log_path=layout.logs / "server.log")
    except supervisor.PortInUseError as error:
        logger.error("app_api", "start_refused", message=str(error),
                     error_code="APP_PORT_IN_USE")
        health.observe("app_api", ERROR, error_code="APP_PORT_IN_USE")
        print(f"[hyprl] {error}")
        return 2

    if result["already_running"]:
        print(f"[hyprl] already running on http://{arguments.host}:{result['port']}/")
        _emit(supervisor.status(layout.pid_file))
        return 0

    serving = supervisor.wait_until_serving(arguments.host, arguments.port)
    lifecycle = recovery_module.mark_started(layout.lifecycle_file,
                                             port=arguments.port)
    if not lifecycle["previous_shutdown_clean"]:
        print("[hyprl] the previous run ended unexpectedly; verifying the "
              "runtime log…")
        report = recovery_module.verify_runtime(_store(layout))
        print(f"[hyprl] event chain: {report['status']}"
              f"{'' if report['event_chain_verified'] is not False else ' — INVALID'}")
        logger.warn("app_api", "unclean_restart",
                    context={"chain_status": report["status"]})

    logger.info("app_api", "started",
                context={"port": arguments.port, "single_origin": site.available})
    health.observe("app_api", HEALTHY if serving else DEGRADED,
                   error_code=None if serving else "APP_NOT_RESPONDING")
    url = f"http://{arguments.host}:{arguments.port}/"
    print(f"[hyprl] {'ready' if serving else 'starting'} — {url}")
    if not site.available:
        print("[hyprl] API only (no frontend build); run "
              "./scripts/hyprl.sh build")
    if stored.get("launch_browser") and not arguments.no_browser and serving:
        # A loopback URL only. Nothing here opens a remote origin.
        try:
            webbrowser.open(url)
        except Exception:                             # pragma: no cover
            pass
    return 0


def command_stop(arguments) -> int:
    root, layout, stored, logger, health = _context(arguments)
    result = supervisor.stop(pid_file=layout.pid_file)
    if result.get("refused"):
        logger.warn("app_api", "stop_refused", message=result["reason"],
                    error_code="APP_PID_FOREIGN")
        print(f"[hyprl] {result['reason']}")
        return 2
    if result["stopped"]:
        recovery_module.mark_stopped(layout.lifecycle_file)
        logger.info("app_api", "stopped",
                    context={"forced": result.get("forced", False)})
        health.observe("app_api", STOPPED)
        print(f"[hyprl] stopped (pid {result['pid']})"
              f"{' after SIGKILL' if result.get('forced') else ''}")
        return 0
    print(f"[hyprl] {result['reason']}")
    return 0


def command_restart(arguments) -> int:
    command_stop(arguments)
    return command_start(arguments)


def command_status(arguments) -> int:
    root, layout, stored, logger, health = _context(arguments)
    store = _store(layout)
    _emit({
        "app": supervisor.status(layout.pid_file),
        "paper": _paper_status(layout),
        "recovery": {
            "last_shutdown_clean": recovery_module.last_shutdown_clean(
                layout.lifecycle_file,
                running=supervisor.inspect(layout.pid_file)["state"]
                == supervisor.RUNNING),
        },
        "frontend_build": StaticSite(root / DIST_DIR).describe(),
        "snapshots": recovery_module.snapshot_pressure(store),
        "health": health.latest_per_component(),
        "real_money": False,
        "broker_connected": False,
    })
    return 0


def command_logs(arguments) -> int:
    root, layout, stored, logger, health = _context(arguments)
    for record in logger.tail(arguments.limit):
        print(json.dumps(record, sort_keys=True))
    return 0


def command_doctor(arguments) -> int:
    root, layout, stored, logger, health = _context(arguments)
    report = doctor_module.run(layout=layout, root=root,
                               dist_root=root / DIST_DIR, host=arguments.host,
                               port=arguments.port, include_git=arguments.git)
    print(doctor_module.render(report) if not arguments.json
          else json.dumps(report, indent=2, sort_keys=True))
    status = {doctor_module.PASS: HEALTHY, doctor_module.WARN: DEGRADED,
              doctor_module.FAIL: ERROR}[report["summary"]]
    health.observe("app_api", status,
                   details={"doctor": report["counts"]})
    return doctor_module.exit_code(report,
                                   ignore_warnings=arguments.ignore_warnings)


# --- paper passthrough -----------------------------------------------------


def command_paper(arguments) -> int:
    """Shadow trading control. Since Phase 6C this is the shared portfolio.

    Wrapping rather than reimplementing: the portfolio CLI carries the embargo
    behaviour and the batching rules, and a second copy of either is a second
    place for it to be wrong.

    The Phase 5D per-product runtime is no longer started from here. Its
    database stays readable and exportable -- two independent accounts are not
    the history of a shared portfolio, so they are kept apart rather than
    merged.
    """
    from scripts.trading_lab import paper_portfolio_cli

    forwarded = [arguments.paper_command, *arguments.rest]
    return paper_portfolio_cli.main(forwarded)


def _store(layout):
    from scripts.trading_lab.paper_event_store import PaperEventStore
    if not layout.paper_database.is_file():
        return None
    return PaperEventStore(layout.paper_database)


def _paper_status(layout) -> dict:
    """The shared portfolio, plus a pointer to the legacy per-product log."""
    from scripts.trading_lab.portfolio import PORTFOLIO_SPEC_V1

    marker = layout.paper_portfolio_session_marker
    active = None
    if marker.is_file():
        try:
            active = json.loads(marker.read_text())
        except (OSError, ValueError):
            active = None
    payload = {"mode": "SHARED_PORTFOLIO", "active_session": active,
               "portfolio_spec_hash": PORTFOLIO_SPEC_V1.portfolio_spec_hash,
               "shared_capital": True, "shadow_mode": True,
               "real_money": False, "broker_connected": False,
               "legacy_individual_accounts_available":
                   layout.paper_database.is_file()}
    if layout.paper_portfolio_database.is_file():
        from scripts.trading_lab.paper_portfolio_store import PaperPortfolioStore
        store = PaperPortfolioStore(layout.paper_portfolio_database)
        sessions = store.sessions()
        payload["sessions"] = len(sessions)
        payload["events"] = store.count(session_id=sessions[-1]) if sessions else 0
    return payload


def _legacy_paper_status(layout) -> dict:
    marker = layout.paper_session_marker
    active = None
    if marker.is_file():
        try:
            active = json.loads(marker.read_text())
        except (OSError, ValueError):
            active = None
    store = _store(layout)
    payload = {"active_session": active, "shadow_mode": True,
               "real_money": False, "broker_connected": False,
               "events": store.count() if store is not None else 0}
    return payload


# --- export / support ------------------------------------------------------


def command_export(arguments) -> int:
    root, layout, stored, logger, health = _context(arguments)
    destination = pathlib.Path(arguments.destination)
    if not destination.is_absolute():
        destination = pathlib.Path.cwd() / destination
    try:
        report = runtime_export.export_runtime(
            layout=layout, destination=destination,
            model_dir=root / "data/models/paper_v1",
            include_logs=arguments.include_logs)
    except runtime_export.ExportError as error:
        print(f"[hyprl] {error}")
        return 2
    logger.info("event_store", "exported",
                context={"bytes": report["bytes"], "sha256": report["sha256"]})
    _emit(report)
    return 0


def command_verify_export(arguments) -> int:
    try:
        report = runtime_export.verify_export(arguments.archive)
    except runtime_export.ExportError as error:
        print(f"[hyprl] {error}")
        return 2
    _emit(report)
    return 0 if report["ok"] else 2


def command_import(arguments) -> int:
    try:
        report = runtime_export.import_export(arguments.archive,
                                              destination=arguments.destination)
    except runtime_export.ExportError as error:
        print(f"[hyprl] {error}")
        return 2
    _emit(report)
    return 0


def command_support_bundle(arguments) -> int:
    root, layout, stored, logger, health = _context(arguments)
    bundle = support_bundle.build(layout=layout, root=root, health=health,
                                  static_site=StaticSite(root / DIST_DIR),
                                  store=_store(layout))
    destination = pathlib.Path(arguments.destination)
    if not destination.is_absolute():
        destination = pathlib.Path.cwd() / destination
    try:
        written = support_bundle.write(bundle, destination)
    except support_bundle.SupportBundleError as error:
        print(f"[hyprl] {error}")
        return 2
    print(f"[hyprl] support bundle written ({written.stat().st_size} bytes)")
    return 0


def command_settings(arguments) -> int:
    root, layout, stored, logger, health = _context(arguments)
    if not arguments.set:
        _emit({"current": stored, **settings_module.describe()})
        return 0
    payload = dict(stored)
    for pair in arguments.set:
        if "=" not in pair:
            print(f"[hyprl] expected field=value, got {pair!r}")
            return 2
        field, _, value = pair.partition("=")
        payload[field.strip()] = _coerce(value.strip())
    payload.pop("schema_version", None)
    try:
        saved = settings_module.save(layout.settings_file, payload)
    except settings_module.ForbiddenSettingError as error:
        print(f"[hyprl] {error}")
        return 2
    except settings_module.SettingsError as error:
        print(f"[hyprl] {error}")
        return 2
    _emit(saved)
    return 0


def _coerce(value: str):
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False
    return value


# --- parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyprl",
        description="HyprL local application. No real money, no broker, no "
                    "exchange key.")
    parser.add_argument("--runtime", default="var/trading_lab")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--fomc-store", default=None,
                        help="FOMC store directory for the Events page (opened read-only)")
    parser.add_argument("--edgar-store", default=None,
                        help="EDGAR store directory for the Events page (opened read-only)")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="build if needed and serve the app")
    start.add_argument("--no-browser", action="store_true")
    start.add_argument("--no-build", action="store_true")
    start.set_defaults(handler=command_start)

    sub.add_parser("stop", help="stop the app").set_defaults(handler=command_stop)
    restart = sub.add_parser("restart", help="stop then start")
    restart.add_argument("--no-browser", action="store_true")
    restart.add_argument("--no-build", action="store_true")
    restart.set_defaults(handler=command_restart)

    sub.add_parser("status", help="what is running").set_defaults(
        handler=command_status)

    build = sub.add_parser("build", help="build the production frontend")
    build.set_defaults(handler=command_build)

    release = sub.add_parser("release", help="assemble a local release bundle")
    release.add_argument("--output", default=None)
    release.add_argument("--without-research-data", action="store_true",
                         help="omit the research corpus and committed results")
    release.set_defaults(handler=command_release)

    verify_release = sub.add_parser("verify-release",
                                    help="re-check a release against its manifest")
    verify_release.add_argument("path")
    verify_release.set_defaults(handler=command_verify_release)

    logs = sub.add_parser("logs", help="recent structured log records")
    logs.add_argument("--limit", type=int, default=100)
    logs.set_defaults(handler=command_logs)

    doctor = sub.add_parser("doctor", help="offline diagnosis; changes nothing")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--git", action="store_true",
                        help="include a development-only git diagnostic")
    doctor.add_argument("--ignore-warnings", action="store_true",
                        help="exit 0 when only warnings were found")
    doctor.set_defaults(handler=command_doctor)

    paper = sub.add_parser("paper", help="shadow trading control")
    paper.add_argument("paper_command",
                       choices=("start", "stop", "restart", "status"))
    paper.add_argument("rest", nargs=argparse.REMAINDER)
    paper.set_defaults(handler=command_paper)

    export = sub.add_parser("export", help="auditable archive of the runtime")
    export.add_argument("destination")
    export.add_argument("--include-logs", action="store_true")
    export.set_defaults(handler=command_export)

    verify = sub.add_parser("verify-export", help="validate an archive")
    verify.add_argument("archive")
    verify.set_defaults(handler=command_verify_export)

    imported = sub.add_parser("import", help="extract an archive offline")
    imported.add_argument("archive")
    imported.add_argument("--destination", required=True)
    imported.set_defaults(handler=command_import)

    support = sub.add_parser("support-bundle", help="sanitized diagnostics")
    support.add_argument("destination")
    support.set_defaults(handler=command_support_bundle)

    settings = sub.add_parser("settings", help="local preferences")
    settings.add_argument("--set", action="append", default=[],
                          metavar="FIELD=VALUE")
    settings.set_defaults(handler=command_settings)
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":                            # pragma: no cover
    sys.exit(main())
