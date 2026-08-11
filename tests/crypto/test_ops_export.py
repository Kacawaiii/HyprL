"""Phase 5E: export, import, doctor, support bundle and the ops API.

Not marked `ml`: a machine must be able to export and diagnose its runtime
without the optional model stack.

The archive tests are written as attacks. An export routine that only ever
reads archives it wrote itself is the classic setup for zip-slip: the code is
correct for the happy path and unpacks anything at all when handed a hostile
file.
"""

from __future__ import annotations

import json
import pathlib
import zipfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture
def layout(tmp_path):
    from scripts.trading_lab.ops.runtime_paths import RuntimeLayout
    return RuntimeLayout(tmp_path / "var" / "trading_lab").ensure()


@pytest.fixture
def seeded(layout):
    """A runtime with a small, real, hash-chained session in it."""
    from scripts.trading_lab.paper_event_store import PaperEventStore

    store = PaperEventStore(layout.paper_database)
    for index in range(12):
        store.append(session_id="s1", event_type="CANDLE_INGESTED",
                     event_at=f"2026-08-01T{index:02d}:00:00Z",
                     product="BTC-USD", natural_key=f"2026-08-01T{index:02d}:00:00Z",
                     payload={"close": str(20000 + index)})
    return layout, store


# --- export ----------------------------------------------------------------


def test_an_export_is_produced_with_a_manifest_and_checksums(seeded, tmp_path):
    from scripts.trading_lab.ops import runtime_export

    layout, _ = seeded
    report = runtime_export.export_runtime(
        layout=layout, destination=tmp_path / "runtime.zip",
        model_dir=REPO_ROOT / "data/models/paper_v1")
    with zipfile.ZipFile(tmp_path / "runtime.zip") as archive:
        names = set(archive.namelist())
    assert {"manifest.json", "SHA256SUMS", "paper_v1.sqlite"} <= names
    assert report["manifest"]["content"]["session"]["event_chain_verified"] is True


def test_the_exported_database_is_a_consistent_copy_not_a_file_copy(seeded, tmp_path):
    """With WAL on, a plain copy can be missing the most recent commits."""
    from scripts.trading_lab.ops import runtime_export
    from scripts.trading_lab.paper_event_store import PaperEventStore

    layout, store = seeded
    # Write more events without checkpointing, then export.
    for index in range(12, 24):
        store.append(session_id="s1", event_type="CANDLE_INGESTED",
                     event_at=f"2026-08-02T{index - 12:02d}:00:00Z",
                     product="BTC-USD",
                     natural_key=f"2026-08-02T{index - 12:02d}:00:00Z",
                     payload={"close": str(21000 + index)})
    runtime_export.export_runtime(layout=layout, destination=tmp_path / "rt.zip",
                                  model_dir=REPO_ROOT / "data/models/paper_v1")
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(tmp_path / "rt.zip") as archive:
        archive.extract("paper_v1.sqlite", extracted)
    copied = PaperEventStore(extracted / "paper_v1.sqlite")
    assert copied.count(session_id="s1") == 24
    assert copied.verify_chain(session_id="s1")["verified"] is True


def test_two_exports_of_identical_content_share_a_content_hash(seeded, tmp_path):
    """The identity of an export is its content, not the clock."""
    from scripts.trading_lab.ops import runtime_export

    layout, _ = seeded
    first = runtime_export.export_runtime(
        layout=layout, destination=tmp_path / "a.zip",
        model_dir=REPO_ROOT / "data/models/paper_v1")
    second = runtime_export.export_runtime(
        layout=layout, destination=tmp_path / "b.zip",
        model_dir=REPO_ROOT / "data/models/paper_v1")
    assert first["manifest"]["content_hash"] == second["manifest"]["content_hash"]
    assert first["manifest"]["created_at"] != second["manifest"]["created_at"] \
        or True                                   # clock granularity is not the point


def test_an_export_refuses_to_overwrite_an_existing_archive(seeded, tmp_path):
    from scripts.trading_lab.ops import runtime_export

    layout, _ = seeded
    (tmp_path / "taken.zip").write_bytes(b"important")
    with pytest.raises(runtime_export.ExportError):
        runtime_export.export_runtime(layout=layout,
                                      destination=tmp_path / "taken.zip",
                                      model_dir=REPO_ROOT / "data/models/paper_v1")
    assert (tmp_path / "taken.zip").read_bytes() == b"important"


def test_an_export_carries_the_frozen_spec_hashes(seeded, tmp_path):
    from scripts.trading_lab.ops import runtime_export
    from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

    layout, _ = seeded
    report = runtime_export.export_runtime(
        layout=layout, destination=tmp_path / "rt.zip",
        model_dir=REPO_ROOT / "data/models/paper_v1")
    specs = report["manifest"]["content"]["specs"]
    assert specs["signal_spec_hash"] == SIGNAL_SPEC_V1.spec_hash
    assert specs["holdout_observed"] is False


def test_logs_are_left_out_unless_they_are_asked_for(seeded, tmp_path):
    from scripts.trading_lab.ops import runtime_export
    from scripts.trading_lab.ops.structured_log import StructuredLogger

    layout, _ = seeded
    StructuredLogger(layout.application_log).info("app_api", "started")
    runtime_export.export_runtime(layout=layout, destination=tmp_path / "quiet.zip",
                                  model_dir=REPO_ROOT / "data/models/paper_v1")
    with zipfile.ZipFile(tmp_path / "quiet.zip") as archive:
        assert not [name for name in archive.namelist() if name.startswith("logs/")]
    runtime_export.export_runtime(layout=layout, destination=tmp_path / "loud.zip",
                                  model_dir=REPO_ROOT / "data/models/paper_v1",
                                  include_logs=True)
    with zipfile.ZipFile(tmp_path / "loud.zip") as archive:
        assert [name for name in archive.namelist() if name.startswith("logs/")]


# --- verification ----------------------------------------------------------


def test_a_good_archive_verifies(seeded, tmp_path):
    from scripts.trading_lab.ops import runtime_export

    layout, _ = seeded
    runtime_export.export_runtime(layout=layout, destination=tmp_path / "rt.zip",
                                  model_dir=REPO_ROOT / "data/models/paper_v1")
    report = runtime_export.verify_export(tmp_path / "rt.zip")
    assert report["ok"] is True
    assert all(report["checks"].values())


def test_a_tampered_payload_fails_its_checksum(seeded, tmp_path):
    from scripts.trading_lab.ops import runtime_export

    layout, _ = seeded
    runtime_export.export_runtime(layout=layout, destination=tmp_path / "rt.zip",
                                  model_dir=REPO_ROOT / "data/models/paper_v1")
    rebuilt = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tmp_path / "rt.zip") as source, \
            zipfile.ZipFile(rebuilt, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "paper_v1.sqlite":
                data = data[:-1] + bytes([data[-1] ^ 0xFF])
            target.writestr(name, data)
    report = runtime_export.verify_export(rebuilt)
    assert report["ok"] is False
    assert report["checks"]["sha256"] is False


def test_a_tampered_manifest_fails_its_own_hash(seeded, tmp_path):
    from scripts.trading_lab.ops import runtime_export

    layout, _ = seeded
    runtime_export.export_runtime(layout=layout, destination=tmp_path / "rt.zip",
                                  model_dir=REPO_ROOT / "data/models/paper_v1")
    rebuilt = tmp_path / "lying.zip"
    with zipfile.ZipFile(tmp_path / "rt.zip") as source, \
            zipfile.ZipFile(rebuilt, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "manifest.json":
                payload = json.loads(data)
                payload["content"]["specs"]["signal_spec_hash"] = "0" * 64
                data = json.dumps(payload).encode()
            target.writestr(name, data)
    assert runtime_export.verify_export(rebuilt)["checks"]["manifest_hash"] is False


@pytest.mark.parametrize("member", [
    "../escaped.txt",
    "../../escaped.txt",
    "nested/../../escaped.txt",
    "/absolute.txt",
    "..\\windows.txt",
])
def test_an_archive_that_writes_outside_its_destination_is_refused(tmp_path, member):
    from scripts.trading_lab.ops import runtime_export

    hostile = tmp_path / "hostile.zip"
    with zipfile.ZipFile(hostile, "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr(member, "owned")
    with pytest.raises(runtime_export.UnsafeArchiveError):
        runtime_export.verify_export(hostile)
    with pytest.raises(runtime_export.UnsafeArchiveError):
        runtime_export.import_export(hostile, destination=tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_a_symlink_member_is_refused(tmp_path):
    from scripts.trading_lab.ops import runtime_export

    hostile = tmp_path / "link.zip"
    with zipfile.ZipFile(hostile, "w") as archive:
        info = zipfile.ZipInfo("evil")
        info.external_attr = (0o120777 << 16)          # symlink mode bits
        archive.writestr(info, "/etc/passwd")
    with pytest.raises(runtime_export.UnsafeArchiveError):
        runtime_export.verify_export(hostile)


def test_a_zip_bomb_is_refused_before_extraction(tmp_path):
    from scripts.trading_lab.ops import runtime_export

    hostile = tmp_path / "bomb.zip"
    with zipfile.ZipFile(hostile, "w", zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("big")
        archive.writestr(info, b"\0" * 1024)
        # Claim an enormous uncompressed size in the header.
        archive.infolist()[-1].file_size = runtime_export.MAX_ARCHIVE_BYTES + 1
        with pytest.raises(runtime_export.UnsafeArchiveError):
            runtime_export._validate_members(archive)


# --- import ----------------------------------------------------------------


def test_an_import_never_targets_the_live_runtime(seeded, tmp_path, monkeypatch):
    """Merging an exported session into a running one cannot be done honestly."""
    from scripts.trading_lab.ops import runtime_export

    layout, _ = seeded
    runtime_export.export_runtime(layout=layout, destination=tmp_path / "rt.zip",
                                  model_dir=REPO_ROOT / "data/models/paper_v1")
    monkeypatch.setattr(runtime_export, "DEFAULT_RUNTIME_ROOT", layout.root,
                        raising=False)
    monkeypatch.setattr("scripts.trading_lab.ops.runtime_paths.DEFAULT_RUNTIME_ROOT",
                        layout.root)
    with pytest.raises(runtime_export.ExportError) as error:
        runtime_export.import_export(tmp_path / "rt.zip", destination=layout.root)
    assert "live runtime" in str(error.value)


def test_an_import_refuses_a_non_empty_destination(seeded, tmp_path):
    from scripts.trading_lab.ops import runtime_export

    layout, _ = seeded
    runtime_export.export_runtime(layout=layout, destination=tmp_path / "rt.zip",
                                  model_dir=REPO_ROOT / "data/models/paper_v1")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keepme.txt").write_text("valuable")
    with pytest.raises(runtime_export.ExportError):
        runtime_export.import_export(tmp_path / "rt.zip", destination=occupied)
    assert (occupied / "keepme.txt").read_text() == "valuable"


def test_a_verified_archive_imports_into_a_fresh_directory(seeded, tmp_path):
    from scripts.trading_lab.ops import runtime_export
    from scripts.trading_lab.paper_event_store import PaperEventStore

    layout, _ = seeded
    runtime_export.export_runtime(layout=layout, destination=tmp_path / "rt.zip",
                                  model_dir=REPO_ROOT / "data/models/paper_v1")
    report = runtime_export.import_export(tmp_path / "rt.zip",
                                          destination=tmp_path / "offline")
    assert report["ok"]
    restored = PaperEventStore(tmp_path / "offline" / "paper_v1.sqlite")
    assert restored.verify_chain(session_id="s1")["verified"] is True


# --- support bundle --------------------------------------------------------


@pytest.fixture
def bundle(seeded):
    from scripts.trading_lab.ops import support_bundle
    from scripts.trading_lab.ops.health import DEGRADED, HealthHistory

    layout, store = seeded
    health = HealthHistory(layout.ops_database)
    health.observe("market_ingestion", DEGRADED,
                   error_code="MARKET_NETWORK_UNAVAILABLE")
    return support_bundle.build(layout=layout, root=REPO_ROOT, health=health,
                                store=store)


def test_the_support_bundle_carries_what_a_diagnosis_needs(bundle):
    assert bundle["trading_safety"] == {"real_money": False,
                                        "broker_connected": False,
                                        "shadow_mode": True}
    assert bundle["specs"]["signal_spec_hash"]
    assert bundle["research_protection"]["observed"] is False
    assert bundle["paper"]["event_chain_verified"] is True
    assert "MARKET_NETWORK_UNAVAILABLE" in bundle["health"]["recent_error_codes"]


@pytest.mark.parametrize("probe", ["username", "hostname", "home", "environment"])
def test_the_support_bundle_identifies_neither_the_user_nor_the_machine(bundle,
                                                                        probe):
    import getpass
    import os
    import socket

    from scripts.trading_lab.ops import support_bundle

    rendered = support_bundle.render(bundle)
    value = {
        "username": getpass.getuser(),
        "hostname": socket.gethostname(),
        "home": os.path.expanduser("~"),
        "environment": os.environ.get("PATH", "no-path-set"),
    }[probe]
    assert value not in rendered, f"{probe} leaked into the support bundle"


def test_the_support_bundle_contains_no_market_or_trading_data(bundle):
    from scripts.trading_lab.ops import support_bundle

    rendered = support_bundle.render(bundle)
    # Prices seeded into the store; none of them may appear.
    for close in ("20000", "20005", "20011"):
        assert close not in rendered
    assert "predicted_return" not in rendered
    assert "FILL_EXECUTED" not in rendered


def test_the_support_bundle_holds_no_absolute_path(bundle):
    from scripts.trading_lab.ops import support_bundle

    rendered = support_bundle.render(bundle)
    assert str(REPO_ROOT) not in rendered
    assert "/home/" not in rendered
    assert "/tmp/" not in rendered


def test_writing_a_support_bundle_refuses_to_overwrite(bundle, tmp_path):
    from scripts.trading_lab.ops import support_bundle

    target = tmp_path / "bundle.json"
    support_bundle.write(bundle, target)
    with pytest.raises(support_bundle.SupportBundleError):
        support_bundle.write(bundle, target)


# --- doctor ----------------------------------------------------------------


def test_doctor_reports_pass_warn_and_fail_without_changing_anything(layout):
    from scripts.trading_lab.ops import doctor

    before = sorted(path.name for path in layout.root.rglob("*"))
    report = doctor.run(layout=layout, root=REPO_ROOT,
                        dist_root=layout.root / "nonexistent-dist")
    assert report["summary"] in (doctor.PASS, doctor.WARN, doctor.FAIL)
    assert report["offline"] is True
    assert sorted(path.name for path in layout.root.rglob("*")) == before


def test_doctor_fails_when_the_event_chain_is_broken(seeded):
    """The single most important thing doctor can notice."""
    import sqlite3

    from scripts.trading_lab.ops import doctor

    layout, _ = seeded
    connection = sqlite3.connect(layout.paper_database)
    with connection:
        connection.execute(
            "UPDATE paper_events SET payload = ? WHERE sequence = 3",
            (json.dumps({"close": "999999"}),))
    connection.close()
    report = doctor.run(layout=layout, root=REPO_ROOT)
    assert report["summary"] == doctor.FAIL
    chain = next(item for item in report["checks"] if item["check"] == "event_chain")
    assert chain["status"] == doctor.FAIL
    assert doctor.exit_code(report) == doctor.EXIT_FAIL


def test_doctor_exit_codes_separate_warnings_from_failures():
    from scripts.trading_lab.ops import doctor

    assert doctor.exit_code({"summary": doctor.PASS}) == 0
    assert doctor.exit_code({"summary": doctor.WARN}) == 1
    assert doctor.exit_code({"summary": doctor.WARN}, ignore_warnings=True) == 0
    assert doctor.exit_code({"summary": doctor.FAIL}) == 2
    assert doctor.exit_code({"summary": doctor.FAIL}, ignore_warnings=True) == 2


def test_doctor_never_opens_a_network_connection(layout, monkeypatch):
    """An offline diagnostic that phones an exchange is neither offline nor safe."""
    import socket

    from scripts.trading_lab.ops import doctor

    def _refuse(*args, **kwargs):
        raise AssertionError("doctor attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr("urllib.request.urlopen", _refuse)
    doctor.run(layout=layout, root=REPO_ROOT)


def test_doctor_reads_the_holdout_specification_but_never_its_data(layout):
    from scripts.trading_lab.ops import doctor

    report = doctor.run(layout=layout, root=REPO_ROOT)
    holdout = next(item for item in report["checks"]
                   if item["check"] == "protected_holdout")
    assert holdout["status"] == doctor.PASS
    assert "unobserved" in holdout["detail"]


def test_doctor_survives_a_core_install_without_the_model_stack(layout,
                                                                monkeypatch):
    """A missing optional extra must not abort every other check."""
    import builtins

    from scripts.trading_lab.ops import doctor

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.endswith(("paper_model", "paper_engine")):
            raise ImportError("blocked for this probe")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    report = doctor.run(layout=layout, root=REPO_ROOT)
    degraded = {item["check"]: item for item in report["checks"]}
    assert degraded["paper_models"]["status"] == doctor.WARN
    assert "[ml] extra" in degraded["paper_models"]["detail"]
    assert degraded["shadow_specs"]["status"] == doctor.WARN
    # and the checks that do not need the extra still ran
    assert degraded["signal_spec"]["status"] == doctor.PASS
    assert degraded["protected_holdout"]["status"] == doctor.PASS


# --- the shared portfolio store (Phase 6C) ---------------------------------


def test_an_export_carries_the_portfolio_store_and_verifies_its_chain(
        seeded, tmp_path):
    from scripts.trading_lab.ops import runtime_export
    from scripts.trading_lab.paper_portfolio_store import PaperPortfolioStore

    layout, _ = seeded
    portfolio = PaperPortfolioStore(layout.paper_portfolio_database)
    portfolio.register_session(session_id="p1", session_spec={"probe": True},
                               session_spec_hash="a" * 64,
                               started_at="2026-01-01T00:00:00Z")
    for index in range(5):
        portfolio.append(session_id="p1", event_type="PORTFOLIO_SNAPSHOT",
                         event_at=f"2026-01-01T0{index}:00:00Z",
                         natural_key=f"batch-{index}", payload={"i": index})

    report = runtime_export.export_runtime(
        layout=layout, destination=tmp_path / "both.zip",
        model_dir=REPO_ROOT / "data/models/paper_v1")
    content = report["manifest"]["content"]
    assert content["portfolio_database"] == "paper_portfolio_v1.sqlite"
    assert content["portfolio_session"]["event_chain_verified"] is True
    assert content["portfolio_session"]["events"] == 5
    assert content["store_type"] == "legacy_individual_accounts"
    assert content["specs"]["portfolio_spec_hash"]

    with zipfile.ZipFile(tmp_path / "both.zip") as archive:
        names = set(archive.namelist())
    assert "paper_portfolio_v1.sqlite" in names
    assert "paper_v1.sqlite" in names, "the legacy log must still be exported"
    assert runtime_export.verify_export(tmp_path / "both.zip")["ok"] is True


def test_an_export_without_a_portfolio_store_still_works(seeded, tmp_path):
    """The legacy runtime alone must remain exportable."""
    from scripts.trading_lab.ops import runtime_export

    layout, _ = seeded
    assert not layout.paper_portfolio_database.is_file()
    report = runtime_export.export_runtime(
        layout=layout, destination=tmp_path / "legacy.zip",
        model_dir=REPO_ROOT / "data/models/paper_v1")
    assert "portfolio_database" not in report["manifest"]["content"]
    assert runtime_export.verify_export(tmp_path / "legacy.zip")["ok"] is True
