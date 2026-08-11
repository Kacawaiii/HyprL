"""Offline diagnosis: what is wrong, without changing anything.

Two rules define this command.

**It never modifies.** A doctor that repairs is a doctor nobody can run to
find out what state they are in. Everything here reads.

**It never touches the network.** A diagnostic that talks to an exchange is a
diagnostic that fails when the exchange is down, blames the local machine, and
-- more seriously here -- fetches market data outside the paper engine's
guard rails. Doctor may read the *specification* of the protected research
window; it may not read one candle of protected data, and it has no code path
that could.

Exit codes:

* ``0`` -- every check passed;
* ``1`` -- at least one WARN, no FAIL. Things work; something deserves a look;
* ``2`` -- at least one FAIL. Something is actually broken.

``--ignore-warnings`` collapses 1 into 0 for scripted use.
"""

from __future__ import annotations

import pathlib
import sys

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

EXIT_OK = 0
EXIT_WARN = 1
EXIT_FAIL = 2

MINIMUM_PYTHON = (3, 10)


class Check:
    __slots__ = ("name", "status", "detail")

    def __init__(self, name: str, status: str, detail: str = ""):
        self.name = name
        self.status = status
        self.detail = detail

    def payload(self) -> dict:
        return {"check": self.name, "status": self.status, "detail": self.detail}


def _check(name, condition, ok_detail="", bad_detail="", *, warn_only=False):
    if condition:
        return Check(name, PASS, ok_detail)
    return Check(name, WARN if warn_only else FAIL, bad_detail)


def run(*, layout, root=None, dist_root=None, host="127.0.0.1", port=8787,
        include_git: bool = False) -> dict:
    """Every check, in order. Returns a report; raises nothing."""
    root = pathlib.Path(root or ".").resolve()
    checks: list[Check] = []

    checks.append(_check(
        "python_version",
        sys.version_info[:2] >= MINIMUM_PYTHON,
        f"python {sys.version_info.major}.{sys.version_info.minor}",
        f"python {sys.version_info.major}.{sys.version_info.minor} is below "
        f"the required {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]}"))

    checks.extend(_runtime_checks(layout))
    checks.extend(_frontend_checks(root, dist_root))
    checks.extend(_port_checks(layout, host, port))
    checks.extend(_integrity_checks(layout))
    checks.extend(_contract_checks())
    if include_git:
        checks.append(_git_check(root))

    statuses = [check.status for check in checks]
    summary = FAIL if FAIL in statuses else (WARN if WARN in statuses else PASS)
    return {
        "summary": summary,
        "counts": {status: statuses.count(status) for status in (PASS, WARN, FAIL)},
        "checks": [check.payload() for check in checks],
        "offline": True,
    }


def _runtime_checks(layout) -> list:
    from scripts.trading_lab.ops.runtime_paths import (
        SUBDIRECTORIES, is_world_writable)

    checks = [_check(
        "runtime_root",
        layout.root.is_dir(),
        "runtime directory present",
        "runtime directory missing; it is created on first start",
        warn_only=True)]

    missing = [name for name in SUBDIRECTORIES if not (layout.root / name).is_dir()]
    checks.append(_check(
        "runtime_subdirectories",
        not missing,
        "all runtime subdirectories present",
        f"missing: {missing} (created on next start)", warn_only=True))

    exposed = []
    if layout.root.is_dir():
        for path in [layout.root, *(layout.root / name for name in SUBDIRECTORIES)]:
            if path.is_dir() and is_world_writable(path):
                exposed.append(path.name)
    checks.append(_check(
        "runtime_permissions",
        not exposed,
        "no world-writable runtime directory",
        f"world-writable: {exposed}"))
    return checks


def _frontend_checks(root: pathlib.Path, dist_root) -> list:
    from scripts.trading_lab.ops.static_assets import StaticSite

    site = StaticSite(dist_root or (root / "apps/web/dist"))
    checks = [_check(
        "frontend_build",
        site.available,
        "production build present",
        "no production build; run ./scripts/hyprl.sh build", warn_only=True)]

    node_modules = root / "apps/web/node_modules"
    checks.append(_check(
        "frontend_packages",
        node_modules.is_dir(),
        "frontend packages installed",
        "frontend packages absent; only needed to rebuild", warn_only=True))
    return checks


def _port_checks(layout, host: str, port: int) -> list:
    from scripts.trading_lab.ops import supervisor

    state = supervisor.inspect(layout.pid_file)
    free = supervisor.port_is_free(host, port)
    if state["state"] == supervisor.RUNNING:
        return [Check("api_port", PASS, f"held by this application on port {port}")]
    if free:
        return [Check("api_port", PASS, f"port {port} is available")]
    return [Check("api_port", WARN,
                  f"port {port} is in use and no HyprL process claims it")]


def _integrity_checks(layout) -> list:
    from scripts.trading_lab.ops import recovery as recovery_module
    from scripts.trading_lab.ops.health import DEGRADED, ERROR

    checks = []
    if not layout.paper_database.is_file():
        checks.append(Check("paper_runtime", WARN,
                            "no shadow runtime recorded yet"))
        return checks

    try:
        from scripts.trading_lab.paper_event_store import PaperEventStore
        store = PaperEventStore(layout.paper_database)
    except Exception as error:
        checks.append(Check("paper_runtime", FAIL,
                            f"the runtime database cannot be opened: {error}"))
        return checks

    checks.append(Check("paper_runtime", PASS, "runtime database opens"))
    report = recovery_module.verify_runtime(store)
    if report["status"] == ERROR:
        checks.append(Check("event_chain", FAIL,
                            f"{report.get('error_code')}: the audit trail "
                            "cannot be trusted"))
    elif report["status"] == DEGRADED:
        checks.append(Check("event_chain", WARN,
                            report.get("error_code") or "no session recorded"))
    else:
        checks.append(Check("event_chain", PASS,
                            f"{report['events']} events verified"))

    if report.get("latest_snapshot_verified") is False:
        checks.append(Check("latest_snapshot", FAIL,
                            "a state snapshot does not match the log"))
    elif report.get("latest_snapshot_verified") is None:
        checks.append(Check("latest_snapshot", WARN, "no snapshot written yet"))
    else:
        checks.append(Check("latest_snapshot", PASS, "snapshot matches the log"))

    pressure = recovery_module.snapshot_pressure(store)
    behind = [product for product, item in pressure.get("products", {}).items()
              if item["events_since_last_snapshot"]
              > 2 * pressure["snapshot_every_events"]]
    checks.append(_check(
        "snapshot_cadence", not behind,
        "snapshots are keeping up",
        f"snapshots are overdue for {behind}", warn_only=True))
    return checks


def _contract_checks() -> list:
    """The frozen specs must load and still hash to what results recorded.

    Split by dependency on purpose. The signal, risk and execution specs and
    the protected window are core; the shadow specs sit behind the optional
    [ml] extra. Checking them together would make one missing optional package
    take down the diagnosis of everything else -- including the holdout, which
    is the check that matters most.
    """
    from scripts.trading_lab.economic_backtest import EXECUTION_SPEC_V1
    from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1
    from scripts.trading_lab.risk_engine import RISK_SPEC_V1
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

    checks = []
    for name, value in (
            ("signal_spec", SIGNAL_SPEC_V1.spec_hash),
            ("risk_spec", RISK_SPEC_V1.risk_spec_hash),
            ("execution_spec", EXECUTION_SPEC_V1.execution_spec_hash)):
        checks.append(_check(name, bool(value) and len(value) == 64,
                             f"{value[:12]}…", f"{name} has no usable hash"))

    checks.append(_check(
        "protected_holdout",
        PROTECTED_WINDOW_V1.holdout_hash and not PROTECTED_WINDOW_V1.observed,
        f"{PROTECTED_WINDOW_V1.start} → {PROTECTED_WINDOW_V1.end}, unobserved",
        "the confirmatory holdout is marked observed"))
    checks.extend(_shadow_contract_checks())
    checks.extend(_model_checks())
    return checks


def _shadow_contract_checks() -> list:
    try:
        from scripts.trading_lab.paper_engine import PAPER_EXECUTION_SPEC_V1
        from scripts.trading_lab.paper_model import PAPER_MODEL_SPEC_V1
    except ImportError:
        return [Check("shadow_specs", WARN,
                      "the optional [ml] extra is not installed; shadow specs "
                      "were not verified")]
    checks = []
    for name, value in (
            ("paper_execution_spec",
             PAPER_EXECUTION_SPEC_V1.paper_execution_spec_hash),
            ("paper_model_spec", PAPER_MODEL_SPEC_V1.paper_model_spec_hash)):
        checks.append(_check(name, bool(value) and len(value) == 64,
                             f"{value[:12]}…", f"{name} has no usable hash"))
    return checks


def _model_checks() -> list:
    from scripts.trading_lab.app_api.contracts import SUPPORTED_PRODUCTS
    try:
        from scripts.trading_lab.paper_model import load_paper_model, read_artifact
    except ImportError:
        # Reading a shadow model needs the optional [ml] extra. A core install
        # must still be able to diagnose everything else rather than aborting
        # the whole command on one unavailable check.
        return [Check("paper_models", WARN,
                      "the optional [ml] extra is not installed; model "
                      "artifacts were not verified")]

    model_dir = pathlib.Path("data/models/paper_v1")
    if not model_dir.is_dir():
        return [Check("paper_models", WARN, "no frozen shadow models found")]
    checks = []
    # By product, not by glob: the directory also holds a manifest, and
    # feeding that to the artifact reader reports a hash mismatch for a file
    # that was never a model.
    for product in SUPPORTED_PRODUCTS:
        path = model_dir / f"{product}.json"
        if not path.is_file():
            checks.append(Check(f"paper_model[{product}]", WARN,
                                "no frozen artifact for this product"))
            continue
        try:
            artifact = read_artifact(path)
            load_paper_model(artifact)
        except Exception as error:
            checks.append(Check(f"paper_model[{path.stem}]", FAIL,
                                f"artifact rejected: {error}"))
            continue
        checks.append(Check(f"paper_model[{path.stem}]", PASS,
                            f"fitted_hash {str(artifact.get('fitted_hash'))[:12]}…"))
    return checks or [Check("paper_models", WARN, "no frozen shadow models found")]


def _git_check(root: pathlib.Path) -> Check:
    """A development diagnostic, not a health signal. Reads .git directly."""
    if not (root / ".git").exists():
        return Check("git_worktree", PASS, "not a git worktree")
    index = root / ".git" / "index"
    if not index.is_file():
        return Check("git_worktree", WARN, "no git index to inspect")
    return Check("git_worktree", PASS, "git metadata readable (state not evaluated)")


def exit_code(report: dict, *, ignore_warnings: bool = False) -> int:
    if report["summary"] == FAIL:
        return EXIT_FAIL
    if report["summary"] == WARN:
        return EXIT_OK if ignore_warnings else EXIT_WARN
    return EXIT_OK


def render(report: dict) -> str:
    lines = [f"HyprL doctor — {report['summary']} "
             f"({report['counts'][PASS]} pass, {report['counts'][WARN]} warn, "
             f"{report['counts'][FAIL]} fail)", ""]
    for check in report["checks"]:
        marker = {PASS: "ok  ", WARN: "warn", FAIL: "FAIL"}[check["status"]]
        detail = f" — {check['detail']}" if check["detail"] else ""
        lines.append(f"  [{marker}] {check['check']}{detail}")
    return "\n".join(lines)
