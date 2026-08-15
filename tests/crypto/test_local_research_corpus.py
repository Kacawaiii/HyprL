"""Discovery, binding and the read-only research API.

The corruption matrix is the point of this file. A discovery service that only
ever sees a healthy store is a service whose failure paths have never run, and
the failure paths are the entire reason it exists: every case below is a store
that a naive `directory.exists()` check would happily serve.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest

from scripts.trading_lab.app_api.contracts import (
    AppApiError, DEFAULT_RESEARCH_BAR_PAGE, MAX_RESEARCH_BAR_PAGE, NotFoundError)
from scripts.trading_lab.app_api.service import AppService
from scripts.trading_lab.equity_corpus import CORPUS_SPEC_V2
from scripts.trading_lab.local_research_corpus import (
    CorpusStatus, LocalCorpusError, LocalResearchCorpusRegistry)

REPO = pathlib.Path(__file__).resolve().parents[2]
REAL_STORE = REPO / "var/trading_lab/research/yahoo_us_equity_daily_v2"
FINGERPRINT = REPO / "docs/artifacts/us_equity_corpus_v2_fingerprint.json"
FAILED_ATTEMPT = (REPO /
                  "var/trading_lab/research/"
                  "yahoo_us_equity_daily_v2.failed-attempt-001")

INSTRUMENTS = ("xnas:AAPL", "xnas:MSFT", "xnas:NVDA", "xnas:QQQ")

# The corpus is local, gitignored and deliberately absent from CI. Every test
# that needs real bytes says so, rather than silently passing on a machine
# where there is nothing to check.
needs_corpus = pytest.mark.skipif(
    not (REAL_STORE / "manifest.local.json").is_file(),
    reason="the local research corpus is not installed on this machine")


def _slug(instrument_id: str) -> str:
    return instrument_id.replace(":", "_")


@pytest.fixture
def store(tmp_path):
    """A byte-for-byte copy of the frozen store, free to be corrupted."""
    target = tmp_path / "corpus"
    shutil.copytree(REAL_STORE, target)
    return target


def _registry(root, fingerprint=FINGERPRINT):
    return LocalResearchCorpusRegistry(corpus_root=root,
                                       fingerprint_path=fingerprint)


def _edit_manifest(root: pathlib.Path, mutate):
    path = root / "manifest.local.json"
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))


# --- §33 A: the healthy case ----------------------------------------------


@needs_corpus
def test_a_good_installation_is_available(store):
    report = _registry(store).report()
    assert report.status == CorpusStatus.AVAILABLE
    assert report.available is True
    assert report.reasons == ()
    assert [item.rows for item in report.instruments] == [501, 501, 501, 501]
    assert report.identity["rows_total"] == 2004


@needs_corpus
def test_the_real_installed_store_is_available():
    """The store this repository actually froze, at its real path."""
    assert _registry(REAL_STORE).report().available is True


def test_an_absent_store_is_not_installed(tmp_path):
    report = _registry(tmp_path / "nowhere").report()
    assert report.status == CorpusStatus.NOT_INSTALLED
    assert report.available is False
    # It still says WHICH corpus is missing: the UI has to name it.
    assert report.identity["corpus_id"] == CORPUS_SPEC_V2.corpus_id


# --- §33 B-H: the corruption matrix ---------------------------------------


@needs_corpus
def test_b_missing_manifest_is_not_installed(store):
    (store / "manifest.local.json").unlink()
    report = _registry(store).report()
    assert report.status == CorpusStatus.NOT_INSTALLED
    assert report.available is False


@needs_corpus
def test_c_modified_aggregate_hash_is_refused(store):
    _edit_manifest(store, lambda payload: payload["content"].update(
        {"corpus_content_hash": "0" * 64}))
    report = _registry(store).report()
    assert report.status == CorpusStatus.CORRUPT
    assert report.available is False


@needs_corpus
def test_d_one_modified_canonical_row_is_refused(store):
    """A single edited price. The file still parses and still has 501 rows."""
    path = store / "canonical" / f"{_slug('xnas:AAPL')}.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[10])
    row["close"] = "999.99"
    lines[10] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    report = _registry(store).report()
    assert report.status == CorpusStatus.CORRUPT
    assert report.available is False
    broken = [item for item in report.instruments if not item.ok]
    assert [item.instrument_id for item in broken] == ["xnas:AAPL"]
    # Still 501 rows: row count alone would have called this healthy.
    assert broken[0].rows == 501


@needs_corpus
def test_e_a_missing_instrument_file_invalidates_the_whole_corpus(store):
    """§8: three good instruments do not make a corpus."""
    (store / "canonical" / f"{_slug('xnas:QQQ')}.jsonl").unlink()
    report = _registry(store).report()
    assert report.available is False
    assert report.status == CorpusStatus.CORRUPT
    healthy = [item.instrument_id for item in report.instruments if item.ok]
    assert healthy == ["xnas:AAPL", "xnas:MSFT", "xnas:NVDA"]
    missing = [item for item in report.instruments
               if item.instrument_id == "xnas:QQQ"]
    assert missing[0].present is False


@needs_corpus
def test_f_swapped_instrument_files_are_refused(store):
    """AAPL's bars under MSFT's name. Every count stays perfect."""
    canonical = store / "canonical"
    aapl = canonical / f"{_slug('xnas:AAPL')}.jsonl"
    msft = canonical / f"{_slug('xnas:MSFT')}.jsonl"
    aapl_bytes, msft_bytes = aapl.read_bytes(), msft.read_bytes()
    aapl.write_bytes(msft_bytes)
    msft.write_bytes(aapl_bytes)

    report = _registry(store).report()
    assert report.status == CorpusStatus.CORRUPT
    assert report.available is False
    assert sorted(item.instrument_id for item in report.instruments
                  if not item.ok) == ["xnas:AAPL", "xnas:MSFT"]


@needs_corpus
def test_g_modified_calendar_hash_is_invalid(store):
    _edit_manifest(store, lambda payload: payload["content"]["corpus_spec"]
                   .update({"calendar_spec_hash": "f" * 64}))
    report = _registry(store).report()
    assert report.status == CorpusStatus.INVALID
    assert report.available is False
    assert any("calendar" in reason for reason in report.reasons)


@needs_corpus
def test_h_modified_spec_hash_is_invalid(store):
    _edit_manifest(store, lambda payload: payload["content"].update(
        {"corpus_spec_hash": "e" * 64}))
    report = _registry(store).report()
    assert report.status == CorpusStatus.INVALID
    assert report.available is False


@needs_corpus
@pytest.mark.parametrize("field", ["provider_id", "timeframe",
                                   "adjustment_policy", "session_type"])
def test_a_foreign_spec_field_is_invalid(store, field):
    _edit_manifest(store, lambda payload: payload["content"]["corpus_spec"]
                   .update({field: "SOMETHING-ELSE"}))
    report = _registry(store).report()
    assert report.status == CorpusStatus.INVALID
    assert any(field in reason for reason in report.reasons)


@needs_corpus
def test_a_shifted_requested_range_is_invalid(store):
    _edit_manifest(store, lambda payload: payload["content"]["corpus_spec"]
                   .update({"requested_range": {"start": "2020-01-01",
                                                "end": "2026-07-31"}}))
    report = _registry(store).report()
    assert report.status == CorpusStatus.INVALID
    assert any("range" in reason for reason in report.reasons)


@needs_corpus
def test_a_manifest_that_merely_claims_to_be_verified_is_not_enough(store):
    """§5: the flag is necessary and proves nothing on its own.

    The manifest keeps `verified: true` and every identity field it is
    supposed to have. Only the bytes moved.
    """
    path = store / "canonical" / f"{_slug('xnas:NVDA')}.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")

    manifest = json.loads((store / "manifest.local.json").read_text())
    assert manifest["content"]["verified"] is True
    assert _registry(store).report().status == CorpusStatus.CORRUPT


@needs_corpus
def test_a_self_consistent_manifest_that_contradicts_the_fingerprint_is_refused(
        store):
    """The committed fingerprint is the anchor, and only it catches this.

    Every hash inside the manifest is rewritten so the manifest agrees with
    itself perfectly: the per-instrument hash, the aggregate built from it,
    everything internally coherent. A check that validated the store against
    its own manifest would pass. Only the comparison against the committed
    fingerprint sees that it now describes different data.
    """
    from scripts.trading_lab.equity_corpus import sha256_canonical

    path = store / "manifest.local.json"
    payload = json.loads(path.read_text())
    entries = payload["content"]["instruments"]
    for entry in entries:
        if entry["instrument_id"] == "xnas:AAPL":
            entry["instrument_content_hash"] = "a" * 64
    payload["content"]["corpus_content_hash"] = sha256_canonical(sorted(
        [{"instrument_id": entry["instrument_id"],
          "instrument_content_hash": entry["instrument_content_hash"],
          "rows": entry["canonical_rows"]} for entry in entries],
        key=lambda item: item["instrument_id"]))
    path.write_text(json.dumps(payload))

    report = _registry(store).report()
    assert report.status == CorpusStatus.CORRUPT
    assert report.available is False


@needs_corpus
def test_a_re_captured_store_with_an_honest_manifest_is_still_refused(store):
    """The case only the committed fingerprint can catch.

    This is not a corrupted store. It is a *coherent* one: the bars were
    changed and the manifest was then regenerated to describe them truthfully
    -- every per-instrument content hash, every byte digest and the aggregate
    all recomputed, so the manifest is entirely honest about what is on disk.
    Nothing internal to the store is wrong.

    It is still not the corpus this repository froze, and the only thing that
    knows that is the fingerprint in Git. A discovery service that validated a
    store against its own manifest would call this AVAILABLE and chart data no
    review ever saw.
    """
    import hashlib

    from scripts.trading_lab.equity_corpus import sha256_canonical

    path = store / "canonical" / f"{_slug('xnas:NVDA')}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows[42]["close"] = "1234.56"
    path.write_text("\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows) + "\n")

    manifest_path = store / "manifest.local.json"
    payload = json.loads(manifest_path.read_text())
    entries = payload["content"]["instruments"]
    for entry in entries:
        if entry["instrument_id"] != "xnas:NVDA":
            continue
        entry["instrument_content_hash"] = sha256_canonical(rows)
        entry["canonical_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload["content"]["corpus_content_hash"] = sha256_canonical(sorted(
        [{"instrument_id": entry["instrument_id"],
          "instrument_content_hash": entry["instrument_content_hash"],
          "rows": entry["canonical_rows"]} for entry in entries],
        key=lambda item: item["instrument_id"]))
    manifest_path.write_text(json.dumps(payload))

    registry = _registry(store)
    report = registry.report()
    assert report.status == CorpusStatus.CORRUPT, (
        "a store validated only against its own manifest would pass here")
    assert report.available is False
    with pytest.raises(LocalCorpusError):
        registry.read_bars("xnas:NVDA")


@needs_corpus
def test_a_reserialised_canonical_file_is_refused(store):
    """Same rows, different bytes.

    Re-serialised with different spacing: every parsed value is identical, so
    a content hash over the rows still matches. The byte digest is what
    notices, and this is the case that proves the two checks are not
    redundant.
    """
    path = store / "canonical" / f"{_slug('xnas:MSFT')}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    path.write_text("\n".join(json.dumps(row, sort_keys=True, indent=None,
                                         separators=(", ", ": "))
                              for row in rows) + "\n")

    report = _registry(store).report()
    assert report.status == CorpusStatus.CORRUPT
    broken = [item for item in report.instruments if not item.ok]
    assert [item.instrument_id for item in broken] == ["xnas:MSFT"]
    # The rows still parse identically; only the bytes moved.
    assert broken[0].content_hash_matches is True
    assert broken[0].canonical_sha256_matches is False


@needs_corpus
def test_a_present_directory_without_binding_is_never_available(tmp_path):
    """§44 UI6E-1: existence is not availability.

    The directory is there, the subdirectories are there, and the store is
    empty. Anything that treated `is_dir()` as the answer would serve this.
    """
    root = tmp_path / "looks-real"
    (root / "canonical").mkdir(parents=True)
    (root / "raw").mkdir()
    report = _registry(root).report()
    assert report.available is False
    assert report.status == CorpusStatus.NOT_INSTALLED


@needs_corpus
def test_a_store_with_files_but_no_manifest_is_never_available(store):
    """The bars are all present and correct; the manifest is gone."""
    (store / "manifest.local.json").unlink()
    report = _registry(store).report()
    assert report.available is False
    assert all((store / "canonical" / f"{_slug(item)}.jsonl").is_file()
               for item in INSTRUMENTS)


@needs_corpus
def test_dropping_the_verified_flag_is_still_refused(store):
    _edit_manifest(store, lambda payload: payload["content"].update(
        {"verified": False}))
    assert _registry(store).report().status == CorpusStatus.INVALID


# --- §7 / §34: the failed attempt -----------------------------------------


@pytest.mark.skipif(not FAILED_ATTEMPT.is_dir(),
                    reason="no preserved failed attempt on this machine")
def test_the_failed_attempt_store_is_never_discovered_as_the_corpus():
    report = _registry(FAILED_ATTEMPT).report()
    assert report.available is False
    assert report.status != CorpusStatus.AVAILABLE


@needs_corpus
def test_the_failed_attempt_is_not_reachable_from_the_real_root():
    """Discovery looks at one path; a sibling directory is not a candidate."""
    registry = _registry(REAL_STORE)
    assert "failed-attempt" not in str(registry.corpus_root)
    assert registry.report().available is True


# --- §10: caching must not outlive the bytes ------------------------------


@needs_corpus
def test_a_cached_verdict_is_invalidated_when_a_file_changes(store):
    registry = _registry(store)
    assert registry.report().available is True

    path = store / "canonical" / f"{_slug('xnas:QQQ')}.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row["high"] = "12345.0"
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    # No refresh flag, no restart: the second call must notice on its own.
    assert registry.report().available is False


# --- §9: the reader ------------------------------------------------------


def _tamper_preserving_metadata(path: pathlib.Path, line: int = 5) -> str:
    """Rewrite one price without changing the file's size or mtime.

    The adversary this models is not exotic: anything that can write the file
    can also call ``os.utime``, and a same-length digit swap needs no effort
    at all. Returns the forged value so a caller can prove it was never
    served.
    """
    import os

    before = path.stat()
    rows = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[line])
    original = payload["close"]
    forged = ("9" + original[1:]) if original[0] != "9" else ("1" + original[1:])
    payload["close"] = forged
    rows[line] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    os.utime(path, ns=(before.st_mtime_ns, before.st_mtime_ns))

    after = path.stat()
    assert after.st_size == before.st_size, "the tamper changed the file size"
    assert after.st_mtime_ns == before.st_mtime_ns, "the tamper moved mtime"
    return forged


@needs_corpus
def test_same_size_same_mtime_tampering_is_detected(store):
    """UI6E-R13: metadata is not evidence.

    Path, size and mtime are all unchanged. Only the bytes moved, and that is
    the only thing that may decide. This is the exact case that used to be
    served as AVAILABLE from a warm cache.
    """
    registry = _registry(store)
    assert registry.report().status == CorpusStatus.AVAILABLE

    path = store / "canonical" / f"{_slug('xnas:AAPL')}.jsonl"
    forged = _tamper_preserving_metadata(path)

    # No refresh flag: the same registry instance must notice by itself.
    report = registry.report()
    assert report.status == CorpusStatus.CORRUPT
    assert report.available is False

    # Atomic: every instrument is refused, not just the tampered one, and the
    # forged price is never handed to a caller.
    for instrument in INSTRUMENTS:
        with pytest.raises(LocalCorpusError):
            registry.read_bars(instrument)
    assert forged in path.read_text(), "the tamper did not land on disk"


@needs_corpus
def test_read_bars_revalidates_even_against_a_stale_verdict(store):
    """The served bytes are checked by the read, not only by the verdict.

    A verdict is reached against a read that has already finished. This forces
    a stale AVAILABLE into the cache so the availability gate cannot be what
    refuses, and the refusal has to come from the read revalidating the bytes
    it just pulled off disk.
    """
    registry = _registry(store)
    good = registry.report()
    assert good.status == CorpusStatus.AVAILABLE

    path = store / "canonical" / f"{_slug('xnas:QQQ')}.jsonl"
    _tamper_preserving_metadata(path)

    # Pin the old verdict against the NEW inputs, defeating the cache on purpose.
    registry._cached = good
    registry._cache_key = registry._identity_key(registry._read_inputs())
    assert registry.report().available is True, "the stale verdict did not stick"

    with pytest.raises(LocalCorpusError, match="committed fingerprint"):
        registry.read_bars("xnas:QQQ")


@needs_corpus
def test_a_whole_file_swap_with_preserved_metadata_is_detected(store):
    """§15: replace the file entirely, keep path, size and mtime."""
    import os

    canonical = store / "canonical"
    target = canonical / f"{_slug('xnas:AAPL')}.jsonl"
    donor = canonical / f"{_slug('xnas:MSFT')}.jsonl"
    before = target.stat()
    replacement = donor.read_bytes()[:before.st_size].ljust(before.st_size, b" ")
    target.write_bytes(replacement)
    os.utime(target, ns=(before.st_mtime_ns, before.st_mtime_ns))

    assert target.stat().st_size == before.st_size
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    assert _registry(store).report().status == CorpusStatus.CORRUPT


@needs_corpus
def test_the_cache_key_is_built_from_content_not_metadata(store):
    """Stated directly, so a future refactor cannot quietly regress it."""
    registry = _registry(store)
    before = registry._identity_key(registry._read_inputs())
    _tamper_preserving_metadata(
        store / "canonical" / f"{_slug('xnas:NVDA')}.jsonl")
    after = registry._identity_key(registry._read_inputs())
    assert before != after, "identical metadata produced an identical key"
    # And the key carries digests, not sizes or timestamps.
    for _, value in after:
        assert value is None or (isinstance(value, str) and len(value) == 64)


@needs_corpus
def test_the_verdict_judges_the_bytes_it_hashed(store):
    """One read feeds both the cache key and the evaluation."""
    registry = _registry(store)
    reads = []
    original = registry._read_bytes

    def counting(path):
        reads.append(str(path))
        return original(path)

    registry._read_bytes = counting
    registry.report(refresh=True)
    # Six inputs, read once each: the fingerprint, the manifest and four
    # canonical files. A second read of any of them would be a second version.
    assert len(reads) == len(set(reads)) == 6


@needs_corpus
def test_read_bars_reads_the_canonical_file_exactly_once(store):
    """UI6E-R13-C: there is no window between hashing and parsing.

    Two reads would mean the digest describes one version of the file and the
    rows come from another, and a writer landing between them gets its bytes
    served under a passing check. One read makes that unexpressible rather
    than merely unlikely, so this counts the reads instead of trying to win a
    race in a test.
    """
    registry = _registry(store)
    registry.report()                       # warm, so the verdict is settled
    target = registry.canonical_path("xnas:AAPL")

    reads = []
    original = registry._read_bytes

    def counting(path):
        if path == target:
            reads.append(str(path))
        return original(path)

    registry._read_bytes = counting
    registry.read_bars("xnas:AAPL")
    # One for the availability verdict, one for the serve. Never a third,
    # which is what a hash-then-reopen implementation would need.
    assert len(reads) == 2


@needs_corpus
def test_the_rows_served_come_from_the_bytes_that_were_hashed(store):
    """A writer landing after the read still cannot change what is served."""
    registry = _registry(store)
    registry.report()
    target = registry.canonical_path("xnas:AAPL")
    good = target.read_bytes()

    original = registry._read_bytes
    seen = {"count": 0}

    def tamper_after_serving_read(path):
        data = original(path)
        if path == target:
            seen["count"] += 1
            if seen["count"] == 2:      # read_bars' own read has just returned
                rows = good.decode("utf-8").splitlines()
                payload = json.loads(rows[5])
                payload["close"] = "9" + payload["close"][1:]
                rows[5] = json.dumps(payload, sort_keys=True,
                                     separators=(",", ":"))
                target.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
        return data

    registry._read_bytes = tamper_after_serving_read
    bars = registry.read_bars("xnas:AAPL")
    assert not [bar for bar in bars if bar["close"].startswith("9")], (
        "a row written after the verified read was served")
    # The next request sees the new bytes and refuses them.
    assert _registry(store).report().status == CorpusStatus.CORRUPT


@needs_corpus
def test_read_bars_returns_canonical_rows_only(store):
    rows = _registry(store).read_bars("xnas:AAPL")
    assert len(rows) == 501
    assert set(rows[0]) >= {"instrument_id", "bar_open_at", "bar_close_at",
                            "open", "high", "low", "close", "volume",
                            "session_date"}
    # Never the provider payload: no adjclose, no chart envelope.
    assert "adjclose" not in rows[0]
    assert "chart" not in rows[0]


@needs_corpus
def test_read_bars_refuses_a_corrupt_corpus(store):
    (store / "canonical" / f"{_slug('xnas:MSFT')}.jsonl").unlink()
    with pytest.raises(LocalCorpusError):
        _registry(store).read_bars("xnas:AAPL")


@needs_corpus
def test_read_bars_refuses_an_instrument_outside_the_corpus(store):
    with pytest.raises(LocalCorpusError):
        _registry(store).read_bars("coinbase:BTC-USD")


# --- §2 / §35: the API ----------------------------------------------------


def _service(root=None, fingerprint=FINGERPRINT):
    return AppService(REPO / "data/crypto",
                      research_corpus_root=root or REAL_STORE,
                      research_fingerprint_path=fingerprint)


@needs_corpus
def test_status_endpoint_reports_available_and_its_limits():
    payload = _service().research_equity_corpus()
    assert payload["status"] == CorpusStatus.AVAILABLE
    assert payload["available"] is True
    assert payload["metadata"]["adjustment"] == "RAW"
    assert payload["metadata"]["source_timeframe"] == "1d"
    assert payload["metadata"]["official_contract"] is False
    assert payload["metadata"]["redistribution_permitted"] is False
    assert payload["capabilities"]["live"] is False
    assert payload["capabilities"]["download"] is False
    assert payload["capabilities"]["tradable"] is False


def test_status_endpoint_answers_when_the_corpus_is_absent(tmp_path):
    payload = _service(tmp_path / "nothing").research_equity_corpus()
    assert payload["status"] == CorpusStatus.NOT_INSTALLED
    assert payload["available"] is False
    assert payload["capabilities"]["local_history"] is False
    assert "metadata" not in payload


@needs_corpus
def test_status_endpoint_reports_corruption(store):
    (store / "canonical" / f"{_slug('xnas:QQQ')}.jsonl").unlink()
    payload = _service(store).research_equity_corpus()
    assert payload["status"] == CorpusStatus.CORRUPT
    assert payload["available"] is False


@needs_corpus
def test_the_service_stops_serving_after_metadata_preserving_tampering(store):
    """§10: through the object an HTTP request actually uses.

    One AppService per server process, so this is the long-lived instance
    whose warm cache used to keep answering AVAILABLE for the lifetime of the
    server.
    """
    service = _service(store)
    assert service.research_equity_corpus()["status"] == CorpusStatus.AVAILABLE
    assert service.research_equity_bars("xnas:AAPL", limit=5)["bars"]

    forged = _tamper_preserving_metadata(
        store / "canonical" / f"{_slug('xnas:AAPL')}.jsonl")

    payload = service.research_equity_corpus()
    assert payload["status"] == CorpusStatus.CORRUPT
    assert payload["available"] is False
    # The response must not keep asserting a verification that no longer holds.
    assert "metadata" not in payload
    assert payload["capabilities"]["local_history"] is False

    with pytest.raises(AppApiError):
        service.research_equity_bars("xnas:AAPL")
    assert forged  # the tamper was real


@needs_corpus
def test_no_route_serves_a_forged_bar(store):
    """§11: at the route table, the shape a browser would hit."""
    from scripts.trading_lab.app_api.server import build_routes

    service = _service(store)
    routes, *_ = build_routes(service)
    status_route = routes["/api/v1/research/equities/corpus"]
    assert status_route({})["status"] == CorpusStatus.AVAILABLE

    forged = _tamper_preserving_metadata(
        store / "canonical" / f"{_slug('xnas:MSFT')}.jsonl")

    assert status_route({})["status"] == CorpusStatus.CORRUPT
    for instrument in INSTRUMENTS:
        with pytest.raises(AppApiError):
            service.research_equity_bars(instrument, limit=5)
    assert forged


@needs_corpus
def test_instrument_metadata_stops_claiming_availability_after_tampering(store):
    service = _service(store)
    assert all(item["research"]["local_corpus_available"]
               for item in service.instruments()["instruments"]
               if item["instrument_id"] in INSTRUMENTS)

    _tamper_preserving_metadata(
        store / "canonical" / f"{_slug('xnas:NVDA')}.jsonl")

    for item in service.instruments()["instruments"]:
        if item["instrument_id"] in INSTRUMENTS:
            assert item["research"]["local_corpus_available"] is False
            assert item["research"]["local_corpus_status"] == CorpusStatus.CORRUPT


@needs_corpus
def test_tamper_detection_opens_no_socket(store, monkeypatch):
    """§18: the new path is as offline as the old one."""
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("revalidation attempted a network call")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)

    service = _service(store)
    assert service.research_equity_corpus()["status"] == CorpusStatus.AVAILABLE
    _tamper_preserving_metadata(
        store / "canonical" / f"{_slug('xnas:QQQ')}.jsonl")
    assert service.research_equity_corpus()["status"] == CorpusStatus.CORRUPT
    with pytest.raises(AppApiError):
        service.research_equity_bars("xnas:QQQ")


@needs_corpus
def test_the_status_payload_leaks_no_filesystem_path():
    payload = json.dumps(_service().research_equity_corpus())
    assert str(REPO) not in payload
    assert "/home/" not in payload
    assert "var/trading_lab" not in payload


@needs_corpus
@pytest.mark.parametrize("instrument", INSTRUMENTS)
def test_bars_endpoint_serves_each_corpus_instrument(instrument):
    payload = _service().research_equity_bars(instrument, limit=10)
    assert payload["instrument_id"] == instrument
    assert len(payload["bars"]) == 10
    assert all(bar["instrument_id"] == instrument for bar in payload["bars"])
    assert payload["metadata"]["adjustment"] == "RAW"
    assert payload["metadata"]["source_timeframe"] == "1d"
    assert payload["metadata"]["provider"] == "yahoo-chart-daily-v1"
    assert payload["metadata"]["official_contract"] is False
    assert payload["metadata"]["live"] is False


@needs_corpus
def test_every_bar_carries_the_fields_the_chart_needs():
    bar = _service().research_equity_bars("xnas:AAPL", limit=1)["bars"][0]
    assert set(bar) == {"instrument_id", "bar_open_at", "bar_close_at", "open",
                        "high", "low", "close", "volume", "session_date"}
    # A daily equity bar IS the session, never open + 24h.
    assert bar["bar_open_at"].endswith("Z")
    assert bar["bar_close_at"] != bar["bar_open_at"]


@needs_corpus
@pytest.mark.parametrize("bad", ["yahoo:AAPL", "AAPL", "xnas:ZZZZ",
                                 "coinbase:BTC-USD", "XNAS:AAPL",
                                 "../../etc/passwd"])
def test_bars_endpoint_fails_closed_on_a_wrong_identity(bad):
    with pytest.raises(NotFoundError):
        _service().research_equity_bars(bad)


@needs_corpus
def test_bars_endpoint_refuses_an_empty_instrument():
    with pytest.raises(AppApiError):
        _service().research_equity_bars("")


@needs_corpus
def test_bars_endpoint_is_bounded():
    service = _service()
    assert service.research_equity_bars("xnas:AAPL")["page"]["returned"] \
        == DEFAULT_RESEARCH_BAR_PAGE
    with pytest.raises(AppApiError):
        service.research_equity_bars("xnas:AAPL",
                                     limit=MAX_RESEARCH_BAR_PAGE + 1)
    # Refused, never silently clamped.
    with pytest.raises(AppApiError):
        service.research_equity_bars("xnas:AAPL", limit=100_000)


@needs_corpus
def test_the_whole_corpus_cannot_be_fetched_in_one_call():
    payload = _service().research_equity_bars("xnas:AAPL",
                                             limit=MAX_RESEARCH_BAR_PAGE)
    assert payload["page"]["returned"] == 501
    assert payload["page"]["has_more"] is False
    # 501 fits today; the ceiling is what stops a larger corpus later.
    assert MAX_RESEARCH_BAR_PAGE < 2004


@needs_corpus
def test_start_and_end_filter_the_window():
    payload = _service().research_equity_bars(
        "xnas:AAPL", start="2025-01-02T00:00:00Z", end="2025-01-31T23:59:59Z",
        limit=100)
    dates = [bar["session_date"] for bar in payload["bars"]]
    assert dates
    assert all(date.startswith("2025-01") for date in dates)


@needs_corpus
def test_the_cursor_pages_without_gaps_or_repeats():
    service = _service()
    first = service.research_equity_bars("xnas:MSFT", limit=200)
    assert first["page"]["has_more"] is True
    second = service.research_equity_bars(
        "xnas:MSFT", limit=200, cursor=first["page"]["next_cursor"])
    dates = [bar["session_date"] for bar in first["bars"] + second["bars"]]
    assert len(dates) == len(set(dates)) == 400
    assert dates == sorted(dates)


@needs_corpus
def test_a_cursor_does_not_work_on_another_instrument():
    service = _service()
    cursor = service.research_equity_bars("xnas:AAPL",
                                          limit=10)["page"]["next_cursor"]
    with pytest.raises(AppApiError):
        service.research_equity_bars("xnas:MSFT", limit=10, cursor=cursor)


@needs_corpus
def test_bars_are_refused_entirely_when_the_corpus_is_corrupt(store):
    (store / "canonical" / f"{_slug('xnas:QQQ')}.jsonl").unlink()
    service = _service(store)
    # Not just QQQ: the corpus is atomic, so AAPL is refused too.
    for instrument in INSTRUMENTS:
        with pytest.raises(AppApiError):
            service.research_equity_bars(instrument)


def test_bars_are_refused_when_the_corpus_is_absent(tmp_path):
    with pytest.raises(AppApiError) as excinfo:
        _service(tmp_path / "gone").research_equity_bars("xnas:AAPL")
    message = str(excinfo.value)
    assert "not available" in message
    # The message must not promise a download.
    assert "download" not in message.lower()
    assert "retry" not in message.lower()


# --- §12: instrument metadata ---------------------------------------------


@needs_corpus
def test_instruments_index_marks_equities_researchable_but_not_tradable():
    payload = _service().instruments()
    by_id = {item["instrument_id"]: item for item in payload["instruments"]}
    for instrument in INSTRUMENTS:
        entry = by_id[instrument]
        assert entry["tradable"] is False
        assert entry["research"]["local_corpus_available"] is True
        assert entry["research"]["source_timeframe"] == "1d"
        assert entry["research"]["adjustment"] == "RAW"
        assert entry["research"]["official_contract"] is False
    for crypto in ("coinbase:BTC-USD", "coinbase:ETH-USD"):
        assert by_id[crypto]["tradable"] is True
        assert by_id[crypto]["research"]["local_corpus_available"] is False


@needs_corpus
def test_an_atomically_broken_corpus_marks_every_equity_unavailable(store):
    payload = _service(store).instruments()
    assert all(item["research"]["local_corpus_available"]
               for item in payload["instruments"]
               if item["instrument_id"] in INSTRUMENTS)

    (store / "canonical" / f"{_slug('xnas:QQQ')}.jsonl").unlink()
    payload = _service(store).instruments()
    for item in payload["instruments"]:
        if item["instrument_id"] in INSTRUMENTS:
            assert item["research"]["local_corpus_available"] is False


# --- §2 / §16 / §17: no network, no writes, no raw ------------------------


def test_the_discovery_module_imports_nothing_that_can_reach_a_network():
    import ast

    source = (REPO / "scripts/trading_lab/local_research_corpus.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden = {"socket", "http", "urllib", "requests", "httpx", "ssl"}
    assert not any(any(part in name for part in forbidden)
                   for name in imported), imported
    for banned in ("yahoo_http_transport", "massive_http_transport",
                   "yahoo_chart_provider", "safe_http"):
        assert banned not in source


@needs_corpus
def test_no_socket_is_opened_while_serving_status_and_bars(monkeypatch):
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("the research API attempted a network call")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    service = _service()
    assert service.research_equity_corpus()["available"] is True
    assert service.research_equity_bars("xnas:NVDA", limit=5)["bars"]
    assert service.instruments()["count"] == 6


@needs_corpus
def test_an_absent_corpus_triggers_no_network_attempt(tmp_path, monkeypatch):
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("an absent corpus must never be fetched")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    payload = _service(tmp_path / "absent").research_equity_corpus()
    assert payload["status"] == CorpusStatus.NOT_INSTALLED


def test_the_route_table_exposes_no_raw_or_mutating_research_endpoint():
    from scripts.trading_lab.app_api.server import build_routes

    routes, *_ = build_routes(_service())
    research = [path for path in routes if "research/equities" in path]
    assert research == ["/api/v1/research/equities/corpus"]
    for path in routes:
        for banned in ("/raw", "source-response", "yahoo-payload",
                       "download", "capture", "refresh", "rebuild", "repair"):
            assert banned not in path


def test_the_api_refuses_every_mutating_verb():
    from scripts.trading_lab.app_api.server import AppApiHandler

    for verb in ("do_POST", "do_PUT", "do_PATCH", "do_DELETE"):
        assert getattr(AppApiHandler, verb) is AppApiHandler._refuse


@needs_corpus
def test_no_endpoint_returns_a_raw_provider_payload():
    payload = json.dumps(_service().research_equity_bars("xnas:AAPL", limit=5))
    # Structures that only exist in a Yahoo response envelope. "chart" alone
    # is not one of them -- it is a word inside the provider id, and asserting
    # on it would fail for the wrong reason.
    for marker in ("adjclose", '"chart"', '"result"', '"indicators"',
                   '"quote"', "exchangeTimezoneName", "query1.finance.yahoo.com"):
        assert marker not in payload


@needs_corpus
def test_the_reader_never_touches_the_raw_directory(store, monkeypatch):
    """Deleting every raw file must not change what the API serves."""
    shutil.rmtree(store / "raw")
    service = _service(store)
    assert service.research_equity_corpus()["available"] is True
    assert len(service.research_equity_bars("xnas:AAPL", limit=3)["bars"]) == 3


@needs_corpus
def test_serving_the_corpus_writes_nothing(store):
    before = {path: path.stat().st_mtime_ns
              for path in sorted(store.rglob("*")) if path.is_file()}
    service = _service(store)
    service.research_equity_corpus()
    service.research_equity_bars("xnas:AAPL", limit=50)
    service.research_equity_bars("xnas:QQQ", limit=50)
    after = {path: path.stat().st_mtime_ns
             for path in sorted(store.rglob("*")) if path.is_file()}
    assert before == after
