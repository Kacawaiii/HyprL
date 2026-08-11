"""Phase 5E: the operations layer.

Deliberately NOT marked `ml`. The whole point of these modules is that a
machine can run, diagnose and export the application without the optional
model stack, so they belong in the core suite that proves it.

The tests are written against the failure that would actually happen, not
against the implementation: a stop command that kills a stranger's process, a
static server that hands out /etc/passwd, a log that writes an Authorization
header, an archive that unpacks over ../../.ssh.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import zipfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture
def layout(tmp_path):
    from scripts.trading_lab.ops.runtime_paths import RuntimeLayout
    return RuntimeLayout(tmp_path / "var" / "trading_lab").ensure()


# --- runtime directories ---------------------------------------------------


def test_the_runtime_layout_creates_every_directory_it_promises(layout):
    from scripts.trading_lab.ops.runtime_paths import SUBDIRECTORIES

    for name in SUBDIRECTORIES:
        assert (layout.root / name).is_dir(), name
    assert layout.describe()["directories"] == {name: True for name in SUBDIRECTORIES}


def test_runtime_directories_are_not_world_writable(layout):
    from scripts.trading_lab.ops.runtime_paths import (
        SUBDIRECTORIES, is_world_writable)

    for path in [layout.root, *(layout.root / name for name in SUBDIRECTORIES)]:
        assert not is_world_writable(path), path.name


def test_a_runtime_file_is_written_private(layout):
    from scripts.trading_lab.ops.runtime_paths import write_private

    target = write_private(layout.runtime / "probe.json", "{}")
    assert target.stat().st_mode & 0o077 == 0, "runtime files must not be readable by others"


def test_ensuring_the_layout_twice_is_harmless(layout):
    before = sorted(path.name for path in layout.root.iterdir())
    layout.ensure()
    assert sorted(path.name for path in layout.root.iterdir()) == before


def test_the_layout_report_carries_no_absolute_path(layout):
    rendered = json.dumps(layout.describe())
    assert str(layout.root) not in rendered
    assert "/home/" not in rendered and "/tmp/" not in rendered


# --- structured logging ----------------------------------------------------


def test_a_log_record_has_the_agreed_shape(layout):
    from scripts.trading_lab.ops.structured_log import StructuredLogger

    logger = StructuredLogger(layout.application_log)
    record = logger.info("paper_engine", "candle_ingested", product="BTC-USD",
                         session_id="s1")
    for field in ("timestamp", "level", "component", "event", "schema_version"):
        assert field in record, field
    assert record["product"] == "BTC-USD"
    assert logger.tail(10)[-1]["event"] == "candle_ingested"


@pytest.mark.parametrize("field", [
    "Authorization", "authorization", "X-API-Key", "api_key", "cookie",
    "Cookie", "access_token", "client_secret", "passphrase", "password",
])
def test_a_secret_never_reaches_the_log_whatever_it_is_called(layout, field):
    """Structural redaction: not a habit of the caller, a property of the log."""
    from scripts.trading_lab.ops.structured_log import StructuredLogger

    logger = StructuredLogger(layout.application_log)
    logger.info("app_api", "request",
                context={field: "Bearer sk-live-000-SHOULD-NEVER-APPEAR"})
    written = layout.application_log.read_text()
    assert "SHOULD-NEVER-APPEAR" not in written
    assert "[redacted]" in written


def test_redaction_reaches_secrets_nested_deep_in_context(layout):
    from scripts.trading_lab.ops.structured_log import StructuredLogger

    logger = StructuredLogger(layout.application_log)
    logger.info("app_api", "request", context={
        "outer": {"inner": [{"headers": {"Authorization": "SECRET-VALUE"}}]}})
    assert "SECRET-VALUE" not in layout.application_log.read_text()


def test_a_home_directory_is_scrubbed_from_messages(layout):
    from scripts.trading_lab.ops.structured_log import StructuredLogger

    logger = StructuredLogger(layout.application_log)
    logger.warn("app_api", "failed",
                message="cannot read /home/someone/HyprL/var/x.sqlite")
    written = layout.application_log.read_text()
    assert "/home/someone" not in written
    assert "<home>" in written


def test_the_log_rotates_and_retention_caps_total_size(layout):
    from scripts.trading_lab.ops.structured_log import StructuredLogger

    logger = StructuredLogger(layout.application_log, max_bytes=2048, max_files=3)
    for index in range(400):
        logger.info("app_api", "tick", context={"index": index, "pad": "x" * 100})
    generations = logger.generations()
    assert len(generations) <= 3, "retention did not drop old generations"
    assert logger.total_bytes() <= logger.max_total_bytes()
    assert logger.total_bytes() < 400 * 100, "nothing was rotated away"


def test_rotation_survives_a_restart_of_the_logger(layout):
    from scripts.trading_lab.ops.structured_log import StructuredLogger

    for _ in range(6):
        logger = StructuredLogger(layout.application_log, max_bytes=1024, max_files=3)
        for index in range(60):
            logger.info("app_api", "tick", context={"pad": "y" * 80})
    assert len(StructuredLogger(layout.application_log, max_bytes=1024,
                                max_files=3).generations()) <= 3


def test_an_unbounded_log_cannot_be_configured(layout):
    from scripts.trading_lab.ops.structured_log import (
        StructuredLogError, StructuredLogger)

    with pytest.raises(StructuredLogError):
        StructuredLogger(layout.application_log, max_bytes=0, max_files=5)
    with pytest.raises(StructuredLogError):
        StructuredLogger(layout.application_log, max_bytes=1024, max_files=0)


def test_the_tail_is_bounded_even_when_asked_for_everything(layout):
    from scripts.trading_lab.ops.structured_log import StructuredLogger

    logger = StructuredLogger(layout.application_log)
    for index in range(50):
        logger.info("app_api", "tick", context={"index": index})
    assert len(logger.tail(10)) == 10


# --- health ----------------------------------------------------------------


def test_health_history_is_capped_and_keeps_the_newest(layout):
    from scripts.trading_lab.ops.health import HEALTHY, HealthHistory

    history = HealthHistory(layout.ops_database, max_records=25)
    for index in range(200):
        history.observe("app_api", HEALTHY, details={"index": index})
    assert history.count() <= 25
    newest = history.recent(limit=1)[0]
    assert newest["details"]["index"] == 199


def test_an_unknown_component_or_state_is_refused(layout):
    from scripts.trading_lab.ops.health import HealthError, HealthHistory

    history = HealthHistory(layout.ops_database)
    with pytest.raises(HealthError):
        history.observe("nonexistent_component", "HEALTHY")
    with pytest.raises(HealthError):
        history.observe("app_api", "GREENISH")


def test_the_headline_status_is_the_worst_one():
    from scripts.trading_lab.ops.health import (
        DEGRADED, EMBARGOED, ERROR, HEALTHY, STOPPED, worst)

    assert worst([HEALTHY, HEALTHY]) == HEALTHY
    assert worst([HEALTHY, EMBARGOED]) == EMBARGOED
    assert worst([HEALTHY, DEGRADED, EMBARGOED]) == DEGRADED
    # A healthy API in front of a corrupt log is not a healthy system.
    assert worst([HEALTHY, ERROR, DEGRADED]) == ERROR
    assert worst([STOPPED, HEALTHY]) == STOPPED


def test_health_history_reads_back_per_component(layout):
    from scripts.trading_lab.ops.health import DEGRADED, HEALTHY, HealthHistory

    history = HealthHistory(layout.ops_database)
    history.observe("app_api", HEALTHY)
    history.observe("paper_engine", DEGRADED, error_code="MARKET_NETWORK_UNAVAILABLE")
    latest = history.latest_per_component()
    assert latest["paper_engine"]["status"] == DEGRADED
    assert latest["paper_engine"]["error_code"] == "MARKET_NETWORK_UNAVAILABLE"
    assert len(history.recent(component="app_api", limit=10)) == 1


# --- static assets ---------------------------------------------------------


@pytest.fixture
def site(tmp_path):
    from scripts.trading_lab.ops.static_assets import StaticSite

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>HyprL</title>")
    (dist / "assets" / "index-abc123.js").write_text("console.log(1)")
    (dist / "assets" / "index-abc123.css").write_text("body{}")
    (tmp_path / "secret.txt").write_text("TOP-SECRET-OUTSIDE-DIST")
    return StaticSite(dist)


@pytest.mark.parametrize("attack", [
    "/../secret.txt",
    "/../../etc/passwd",
    "/assets/../../secret.txt",
    "/%2e%2e/secret.txt",
    "/%2e%2e%2fsecret.txt",
    "/..%2fsecret.txt",
    "/assets/%2e%2e/%2e%2e/secret.txt",
    "/./../secret.txt",
    "/\\..\\secret.txt",
    "/assets/..%5c..%5csecret.txt",
])
def test_no_encoding_of_dot_dot_escapes_the_site_root(site, attack):
    from scripts.trading_lab.ops.static_assets import ForbiddenPathError

    with pytest.raises((ForbiddenPathError, FileNotFoundError)):
        site.serve(attack)


def test_a_double_encoded_traversal_is_a_missing_file_not_an_escape(site):
    """%252e decodes once to the text %2e, which is a filename, not a parent."""
    with pytest.raises(FileNotFoundError):
        site.serve("/%252e%252e/secret.txt")


def test_a_symlink_out_of_the_site_root_is_refused(site, tmp_path):
    from scripts.trading_lab.ops.static_assets import ForbiddenPathError

    link = site.root / "escape.txt"
    try:
        link.symlink_to(tmp_path / "secret.txt")
    except OSError:                                   # pragma: no cover
        pytest.skip("symlinks unavailable on this filesystem")
    with pytest.raises(ForbiddenPathError):
        site.serve("/escape.txt")


def test_a_null_byte_in_the_path_is_refused(site):
    from scripts.trading_lab.ops.static_assets import ForbiddenPathError

    with pytest.raises(ForbiddenPathError):
        site.serve("/index.html\x00.png")


def test_ordinary_assets_are_served_with_their_own_content_type(site):
    assert site.serve("/assets/index-abc123.js")["content_type"].startswith(
        "text/javascript")
    assert site.serve("/assets/index-abc123.css")["content_type"].startswith("text/css")
    assert site.serve("/")["content_type"].startswith("text/html")


def test_hashed_assets_are_immutable_and_the_entry_document_is_not(site):
    """A build that cannot be picked up by a returning browser is not deployed."""
    assert "immutable" in site.serve("/assets/index-abc123.js")["cache_control"]
    assert "max-age=31536000" in site.serve("/assets/index-abc123.js")["cache_control"]
    assert site.serve("/")["cache_control"] == "no-cache"
    assert site.serve("/index.html")["cache_control"] == "no-cache"


def test_an_api_path_is_never_resolved_against_the_filesystem(site):
    from scripts.trading_lab.ops.static_assets import ForbiddenPathError, is_api_path

    assert is_api_path("/api/v1/health")
    with pytest.raises(ForbiddenPathError):
        site.serve("/api/v1/health")


def test_a_client_route_falls_back_to_the_document_but_a_missing_asset_does_not(site):
    assert site.spa_fallback("/paper")["relative"] == "index.html"
    assert site.looks_like_asset("/assets/missing-xyz.js")
    assert not site.looks_like_asset("/paper")
    assert not site.looks_like_asset("/backtests/v1/BTC-USD")


# --- supervisor ------------------------------------------------------------


def _sleeper(marker: str):
    """A real child process carrying the marker, used as a stand-in server."""
    return [sys.executable, "-c",
            f"import time,sys; sys.argv.append({marker!r}); time.sleep(120)",
            marker]


def test_a_second_start_does_not_create_a_second_process(layout):
    from scripts.trading_lab.ops import supervisor

    port = _free_port()
    first = supervisor.start(pid_file=layout.pid_file,
                             command=_sleeper(supervisor.PROCESS_MARKER),
                             host="127.0.0.1", port=port)
    try:
        second = supervisor.start(pid_file=layout.pid_file,
                                  command=_sleeper(supervisor.PROCESS_MARKER),
                                  host="127.0.0.1", port=port)
        assert first["started"] and not second["started"]
        assert second["already_running"] and second["pid"] == first["pid"]
    finally:
        supervisor.stop(pid_file=layout.pid_file, timeout=3)


def test_stopping_a_running_process_actually_stops_it(layout):
    from scripts.trading_lab.ops import supervisor

    started = supervisor.start(pid_file=layout.pid_file,
                               command=_sleeper(supervisor.PROCESS_MARKER),
                               host="127.0.0.1", port=_free_port())
    result = supervisor.stop(pid_file=layout.pid_file, timeout=5)
    assert result["stopped"]
    assert not supervisor.process_exists(started["pid"])
    assert not layout.pid_file.exists()


def test_a_stale_pid_file_is_cleaned_up_rather_than_signalled(layout):
    from scripts.trading_lab.ops import supervisor

    # A pid that has certainly exited: spawn and reap one.
    finished = subprocess.run([sys.executable, "-c", "pass"])
    supervisor.write_pid_file(layout.pid_file, {
        "pid": 999_999, "pgid": 999_999, "start_ticks": 1,
        "marker": supervisor.PROCESS_MARKER, "port": 1})
    assert supervisor.inspect(layout.pid_file)["state"] == supervisor.STALE
    result = supervisor.stop(pid_file=layout.pid_file)
    assert not result["stopped"]
    assert not layout.pid_file.exists()
    assert finished.returncode == 0


def test_a_pid_belonging_to_another_process_is_never_signalled(layout):
    """The failure this prevents is HyprL terminating an unrelated program."""
    from scripts.trading_lab.ops import supervisor

    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        supervisor.write_pid_file(layout.pid_file, {
            "pid": stranger.pid, "pgid": stranger.pid,
            "start_ticks": supervisor.process_start_ticks(stranger.pid),
            "marker": supervisor.PROCESS_MARKER, "port": 1})
        # It exists, and its start time matches -- only the marker says no.
        assert supervisor.inspect(layout.pid_file)["state"] == supervisor.FOREIGN
        result = supervisor.stop(pid_file=layout.pid_file)
        assert result.get("refused") is True
        assert not result["stopped"]
        assert stranger.poll() is None, "an unrelated process was signalled"
    finally:
        stranger.kill()
        stranger.wait(timeout=10)


def test_a_recycled_pid_is_detected_by_its_start_time(layout):
    from scripts.trading_lab.ops import supervisor

    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        # Same pid, same marker, but the recorded start time is older: this is
        # exactly what pid reuse looks like.
        supervisor.write_pid_file(layout.pid_file, {
            "pid": stranger.pid, "pgid": stranger.pid, "start_ticks": 1,
            "marker": supervisor.PROCESS_MARKER, "port": 1})
        state = supervisor.inspect(layout.pid_file)
        assert state["state"] == supervisor.FOREIGN
        assert "reused" in state["reason"]
        assert stranger.poll() is None
    finally:
        stranger.kill()
        stranger.wait(timeout=10)


def test_starting_without_the_marker_is_refused(layout):
    from scripts.trading_lab.ops import supervisor

    with pytest.raises(supervisor.SupervisorError):
        supervisor.start(pid_file=layout.pid_file,
                         command=[sys.executable, "-c", "pass"],
                         host="127.0.0.1", port=_free_port())


def test_a_held_port_refuses_the_start_instead_of_racing_it(layout):
    import socket

    from scripts.trading_lab.ops import supervisor

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        with pytest.raises(supervisor.PortInUseError):
            supervisor.start(pid_file=layout.pid_file,
                             command=_sleeper(supervisor.PROCESS_MARKER),
                             host="127.0.0.1", port=port)
    finally:
        holder.close()


def test_the_signal_helper_refuses_process_group_zero():
    """killpg(0) hits the caller's own group; kill(-1) hits everything."""
    import signal as signal_module

    from scripts.trading_lab.ops import supervisor

    for bad in (0, -1, 1):
        with pytest.raises(supervisor.SupervisorError):
            supervisor._signal_group(bad, bad, signal_module.SIGTERM)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# --- settings --------------------------------------------------------------


@pytest.mark.parametrize("field", [
    "signal_threshold", "long_threshold", "risk_cap", "max_long_exposure",
    "fee_rate", "slippage_rate", "model_alpha", "features",
    "holdout_start", "holdout_end", "protected_products", "embargo_enabled",
    "execution_policy", "terminal_liquidation",
])
def test_no_trading_contract_can_be_reached_through_settings(field):
    """The settings file is where a frozen protocol quietly becomes a knob."""
    from scripts.trading_lab.ops.settings import ForbiddenSettingError, validate

    with pytest.raises(ForbiddenSettingError) as error:
        validate({field: "0.01"})
    assert "frozen trading contracts" in str(error.value)


def test_an_unknown_settings_field_is_refused_rather_than_ignored():
    from scripts.trading_lab.ops.settings import SettingsError, validate

    with pytest.raises(SettingsError) as error:
        validate({"colour_scheme": "blue"})
    assert "unknown settings field" in str(error.value)


@pytest.mark.parametrize("payload", [
    {"theme": "chartreuse"},
    {"time_display": "swatch_beats"},
    {"default_product": "DOGE-USD"},
    {"default_chart_window": "forever"},
    {"log_retention_preset": "infinite"},
    {"launch_browser": "yes please"},
])
def test_a_settings_value_outside_its_options_is_refused(payload):
    from scripts.trading_lab.ops.settings import SettingsError, validate

    with pytest.raises(SettingsError):
        validate(payload)


def test_valid_settings_round_trip_through_disk(layout):
    from scripts.trading_lab.ops import settings as settings_module

    saved = settings_module.save(layout.settings_file,
                                 {"theme": "light", "sidebar_collapsed": True})
    assert saved["theme"] == "light"
    reloaded = settings_module.load(layout.settings_file)
    assert reloaded["theme"] == "light" and reloaded["sidebar_collapsed"] is True
    # untouched fields keep their defaults rather than disappearing
    assert reloaded["default_product"] == "BTC-USD"


def test_a_corrupt_settings_file_falls_back_without_touching_contracts(layout):
    from scripts.trading_lab.ops import settings as settings_module

    layout.settings_file.write_text("{not json at all")
    warnings = []
    loaded = settings_module.load(layout.settings_file, on_warning=warnings.append)
    assert loaded == settings_module.DEFAULT_SETTINGS
    assert warnings, "a corrupt settings file should say so"


def test_a_settings_file_from_an_unknown_schema_is_not_guessed_at(layout):
    from scripts.trading_lab.ops import settings as settings_module

    layout.settings_file.write_text(json.dumps(
        {"schema_version": "trading-lab.settings.v99", "theme": "light"}))
    assert settings_module.load(layout.settings_file) == \
        settings_module.DEFAULT_SETTINGS


def test_the_settings_description_names_what_it_refuses():
    from scripts.trading_lab.ops.settings import describe

    described = describe()
    assert described["trading_contracts_immutable"] is True
    assert "signal_threshold" in described["forbidden_trading_fields"]
    assert not set(described["allowed_fields"]) & set(
        described["forbidden_trading_fields"])


# --- recovery --------------------------------------------------------------


def test_a_clean_stop_is_recorded_as_clean(layout):
    from scripts.trading_lab.ops import recovery

    recovery.mark_started(layout.lifecycle_file)
    recovery.mark_stopped(layout.lifecycle_file)
    assert recovery.last_shutdown_clean(layout.lifecycle_file) is True


def test_a_process_that_never_wrote_an_exit_is_reported_unclean(layout):
    from scripts.trading_lab.ops import recovery

    recovery.mark_started(layout.lifecycle_file)          # and then killed
    assert recovery.last_shutdown_clean(layout.lifecycle_file) is False


def test_a_healthy_running_app_does_not_report_a_permanent_recovery(layout):
    """A live RUNNING marker means "this run"; the last shutdown was before it."""
    from scripts.trading_lab.ops import recovery

    recovery.mark_started(layout.lifecycle_file)
    recovery.mark_stopped(layout.lifecycle_file)
    recovery.mark_started(layout.lifecycle_file)
    assert recovery.last_shutdown_clean(layout.lifecycle_file, running=True) is True


def test_a_first_ever_run_has_no_shutdown_to_judge(layout):
    from scripts.trading_lab.ops import recovery

    assert recovery.last_shutdown_clean(layout.lifecycle_file) is None


def test_an_unclean_restart_is_carried_into_the_next_lifecycle_record(layout):
    from scripts.trading_lab.ops import recovery

    recovery.mark_started(layout.lifecycle_file)          # killed, no stop
    record = recovery.mark_started(layout.lifecycle_file)
    assert record["previous_shutdown_clean"] is False


# --- snapshot monitoring (the Phase 5D live smoke found a silent stall) -----


@pytest.fixture
def logged(layout):
    from scripts.trading_lab.paper_event_store import PaperEventStore
    return layout, PaperEventStore(layout.paper_database)


def _append(store, count, *, product="BTC-USD", start=0):
    for index in range(start, start + count):
        store.append(session_id="s1", event_type="CANDLE_INGESTED",
                     event_at=f"2026-08-01T00:00:{index % 60:02d}Z",
                     product=product, natural_key=f"{product}-{index}",
                     payload={"i": index})


def test_snapshot_pressure_reports_the_distance_and_whether_one_is_due(logged):
    from scripts.trading_lab.ops import recovery
    from scripts.trading_lab.paper_event_store import SNAPSHOT_EVERY_EVENTS

    layout, store = logged
    _append(store, 10)
    pressure = recovery.snapshot_pressure(store)
    btc = pressure["products"]["BTC-USD"]
    assert btc["events_since_last_snapshot"] == 10
    assert btc["snapshot_due"] is False
    assert btc["has_snapshot"] is False
    assert pressure["snapshot_every_events"] == SNAPSHOT_EVERY_EVENTS


def test_a_stalled_snapshot_trigger_is_degraded_never_corruption(logged):
    """The 5D failure mode: snapshots silently stop and nothing notices."""
    from scripts.trading_lab.ops import recovery
    from scripts.trading_lab.ops.health import DEGRADED
    from scripts.trading_lab.paper_event_store import SNAPSHOT_EVERY_EVENTS

    layout, store = logged
    _append(store, 3 * SNAPSHOT_EVERY_EVENTS)
    pressure = recovery.snapshot_pressure(store)
    assert pressure["status"] == DEGRADED
    assert pressure["error_code"] == "PAPER_SNAPSHOT_OVERDUE"
    # degraded, not error: a missing snapshot costs replay time, not integrity
    assert recovery.verify_runtime(store)["event_chain_verified"] is True


def test_a_product_that_stopped_receiving_candles_is_not_reported_overdue(logged):
    """Measured per product. A global event-id lag made a healthy product
    look thousands of events behind purely because the other one kept going."""
    from scripts.trading_lab.ops import recovery
    from scripts.trading_lab.ops.health import HEALTHY
    from scripts.trading_lab.paper_event_store import SNAPSHOT_EVERY_EVENTS

    layout, store = logged
    _append(store, 10, product="BTC-USD")
    head = store.latest_events(session_id="s1", limit=1)[-1]
    store.write_snapshot(session_id="s1", product="BTC-USD",
                         last_event_id=head.event_id,
                         last_event_hash=head.event_hash, state={"flat": True})
    # ETH now runs long past BTC's snapshot without BTC being at fault.
    _append(store, 3 * SNAPSHOT_EVERY_EVENTS, product="ETH-USD")
    head = store.latest_events(session_id="s1", limit=1)[-1]
    store.write_snapshot(session_id="s1", product="ETH-USD",
                         last_event_id=head.event_id,
                         last_event_hash=head.event_hash, state={"flat": True})
    pressure = recovery.snapshot_pressure(store)
    assert pressure["products"]["BTC-USD"]["events_since_last_snapshot"] == 0
    assert pressure["status"] == HEALTHY


def test_counting_after_an_event_is_scoped_to_one_product(logged):
    layout, store = logged
    _append(store, 5, product="BTC-USD")
    _append(store, 7, product="ETH-USD")
    assert store.count_after(session_id="s1", product="BTC-USD") == 5
    assert store.count_after(session_id="s1", product="ETH-USD") == 7
    assert store.count_after(session_id="s1") == 12


def test_the_critical_runtime_queries_use_an_index(logged):
    """A full scan here is invisible on a day of events and fatal on a year."""
    import sqlite3

    layout, store = logged
    _append(store, 20)
    connection = sqlite3.connect(layout.paper_database)
    try:
        for sql, params in (
            ("SELECT * FROM paper_events WHERE session_id = ? AND product = ? "
             "AND event_id > ? ORDER BY event_id LIMIT 100", ("s1", "BTC-USD", 0)),
            ("SELECT COUNT(*) FROM paper_events WHERE session_id = ? "
             "AND product = ? AND event_id > ?", ("s1", "BTC-USD", 0)),
            ("SELECT * FROM paper_state_snapshots WHERE session_id = ? "
             "AND product = ? ORDER BY last_event_id DESC LIMIT 1",
             ("s1", "BTC-USD")),
        ):
            plan = " ".join(str(row) for row in connection.execute(
                "EXPLAIN QUERY PLAN " + sql, params).fetchall())
            assert "SCAN" not in plan or "USING" in plan, f"{sql}\n{plan}"
    finally:
        connection.close()


# --- the unified CLI, end to end -------------------------------------------


def _cli(*argv, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "scripts.trading_lab.hyprl_cli", *argv],
        capture_output=True, text=True, cwd=str(cwd or REPO_ROOT), timeout=600)


def test_the_cli_exposes_every_documented_command():
    from scripts.trading_lab.hyprl_cli import build_parser

    parser = build_parser()
    actions = [action for action in parser._actions
               if hasattr(action, "choices") and isinstance(action.choices, dict)]
    commands = set(actions[0].choices)
    assert {"start", "stop", "restart", "status", "doctor", "logs", "build",
            "paper", "export", "verify-export", "import", "support-bundle",
            "settings"} <= commands


def test_the_paper_command_still_reaches_the_phase_5d_control():
    from scripts.trading_lab.hyprl_cli import build_parser

    parser = build_parser()
    arguments = parser.parse_args(["paper", "status"])
    assert arguments.paper_command == "status"


def test_doctor_runs_from_the_command_line_and_reports_an_exit_code(tmp_path):
    result = _cli("--runtime", str(tmp_path / "rt"), "doctor", "--json")
    assert result.returncode in (0, 1, 2), result.stderr
    report = json.loads(result.stdout)
    assert report["offline"] is True
    assert report["checks"]


def test_settings_refuse_a_trading_field_from_the_command_line(tmp_path):
    result = _cli("--runtime", str(tmp_path / "rt"),
                  "settings", "--set", "signal_threshold=0.0001")
    assert result.returncode == 2
    assert "frozen trading contracts" in result.stdout + result.stderr
    assert not (tmp_path / "rt" / "runtime" / "settings.json").exists()


def test_settings_accept_a_presentation_field_from_the_command_line(tmp_path):
    result = _cli("--runtime", str(tmp_path / "rt"), "settings",
                  "--set", "theme=light", "--set", "launch_browser=false")
    assert result.returncode == 0, result.stderr
    saved = json.loads(result.stdout)
    assert saved["theme"] == "light" and saved["launch_browser"] is False


def test_stopping_when_nothing_runs_is_not_an_error(tmp_path):
    result = _cli("--runtime", str(tmp_path / "rt"), "stop")
    assert result.returncode == 0
    assert "no session" in result.stdout


def test_status_answers_without_a_running_application(tmp_path):
    result = _cli("--runtime", str(tmp_path / "rt"), "status")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["app"]["state"] == "STOPPED"
    assert payload["real_money"] is False
    assert payload["broker_connected"] is False


def test_the_cli_never_reports_real_money_or_a_broker(tmp_path):
    for command in (["status"], ["doctor", "--json"]):
        result = _cli("--runtime", str(tmp_path / "rt"), *command)
        assert "real_money\": true" not in result.stdout.lower()
        assert "broker_connected\": true" not in result.stdout.lower()
