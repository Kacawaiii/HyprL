"""Phase 4A: the frozen real-market corpus, its identity, and its offline replay.

Not one test in this file touches the network. The capture path is exercised
through an injected fetch function serving bytes from local fixtures, and the
real corpus is only ever *verified* -- never re-downloaded.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import importlib
import json
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
HOUR = timedelta(hours=1)
GRID = datetime(2026, 3, 2, tzinfo=timezone.utc)


@pytest.fixture
def capture():
    return importlib.import_module("scripts.trading_lab.capture_market_history")


def _iso(moment): return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _candle(index, *, base=Decimal(100), volume="1.5"):
    """One Coinbase row: [time, low, high, open, close, volume]."""
    close = base + index
    return [int((GRID + HOUR * index).timestamp()), str(close - 2), str(close + 3),
            str(close - 1), str(close), volume]


def _page(indexes, **kwargs):
    return json.dumps([_candle(i, **kwargs) for i in reversed(list(indexes))],
                      separators=(",", ":")).encode("utf-8")


class Feed:
    """A deterministic offline stand-in for the Coinbase endpoint."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        return self.pages[len(self.urls) - 1]


def _capture(capture, tmp_path, pages, *, product="BTC-USD", start=None, end=None, count=25):
    start = start or _iso(GRID)
    end = end or _iso(GRID + HOUR * (count - 1))
    return capture.capture_corpus(tmp_path, products=(product,), range_start=start,
                                  range_end=end, fetch=Feed(pages), spacing_seconds=0)


# --- batch planning -------------------------------------------------------


def test_the_batch_plan_is_a_pure_function_of_the_request(capture):
    plan = capture.plan_batches("2025-08-01T00:00:00Z", "2026-07-31T23:00:00Z")
    assert len(plan) == 30
    assert plan == capture.plan_batches("2025-08-01T00:00:00Z", "2026-07-31T23:00:00Z")
    assert plan[0].requested_start == "2025-08-01T00:00:00Z"
    assert plan[-1].requested_end == "2026-07-31T23:00:00Z"
    assert sum(batch.expected_openings for batch in plan) == 8760
    assert [batch.index for batch in plan] == list(range(30))


def test_batch_windows_never_overlap_and_leave_no_hole(capture):
    plan = capture.plan_batches("2026-03-02T00:00:00Z", "2026-03-20T05:00:00Z", batch_size=50)
    for earlier, later in zip(plan, plan[1:]):
        gap = (datetime.fromisoformat(later.requested_start)
               - datetime.fromisoformat(earlier.requested_end))
        assert gap == HOUR, (earlier, later)
    covered = sum(batch.expected_openings for batch in plan)
    assert covered == len(capture.expected_openings("2026-03-02T00:00:00Z",
                                                    "2026-03-20T05:00:00Z"))


@pytest.mark.parametrize("start,end,fragment", [
    ("2026-03-02T00:00:00Z", "2026-03-01T00:00:00Z", "precedes"),
    ("2026-03-02T00:30:00Z", "2026-03-03T00:00:00Z", "aligned"),
    ("2026-03-02T00:00:00Z", "2026-03-03T00:30:00Z", "aligned"),
    ("not-a-date", "2026-03-03T00:00:00Z", "ISO-8601"),
])
def test_a_malformed_range_is_refused(capture, start, end, fragment):
    with pytest.raises(capture.MarketHistoryCaptureError, match=fragment):
        capture.plan_batches(start, end)


@pytest.mark.parametrize("size", [0, -1, 301, 3.0, True, "300"])
def test_a_batch_size_the_adapter_could_not_accept_is_refused(capture, size):
    with pytest.raises(capture.MarketHistoryCaptureError, match="batch_size"):
        capture.plan_batches("2026-03-02T00:00:00Z", "2026-03-03T00:00:00Z", batch_size=size)


def test_the_availability_marker_comes_from_the_plan_not_the_clock(capture):
    batch = capture.plan_batches("2026-03-02T00:00:00Z", "2026-03-02T05:00:00Z")[0]
    marker = capture.batch_marker(batch, timeframe="1h")
    assert marker == "2026-03-02T06:00:00Z"
    assert capture.batch_marker(batch, timeframe="1h") == marker


# --- canonicalisation -----------------------------------------------------


def test_canonicalisation_sorts_ascending_and_keeps_exact_decimals(capture):
    plan = capture.plan_batches(_iso(GRID), _iso(GRID + HOUR * 4))
    rows = capture.canonical_rows_from_payload(
        _page(range(5)), product="BTC-USD", timeframe="1h",
        marker=capture.batch_marker(plan[0], timeframe="1h"))
    openings = [row["bar_open_at"] for row in rows]
    assert openings == sorted(openings)          # the source page was descending
    assert len(rows) == 5
    # values survive as exact decimals, never through a binary float
    assert Decimal(rows[0]["close"]) == Decimal(100)
    assert Decimal(rows[4]["close"]) == Decimal(104)
    assert all(isinstance(value, str) for row in rows for value in row.values())


def test_a_page_reaching_beyond_the_requested_window_fails_closed(capture):
    """The marker makes an out-of-window candle a hard error, not a surprise."""
    plan = capture.plan_batches(_iso(GRID), _iso(GRID + HOUR * 2))
    with pytest.raises(Exception):
        capture.canonical_rows_from_payload(
            _page(range(10)), product="BTC-USD", timeframe="1h",
            marker=capture.batch_marker(plan[0], timeframe="1h"))


def test_identical_duplicate_openings_are_merged_and_counted(capture):
    plan = capture.plan_batches(_iso(GRID), _iso(GRID + HOUR * 9))
    marker = capture.batch_marker(plan[0], timeframe="1h")
    first = capture.canonical_rows_from_payload(_page(range(0, 6)), product="BTC-USD",
                                                timeframe="1h", marker=marker)
    second = capture.canonical_rows_from_payload(_page(range(4, 10)), product="BTC-USD",
                                                 timeframe="1h", marker=marker)
    merged, duplicates = capture.merge_canonical_rows([first, second])
    assert len(merged) == 10
    assert duplicates == 2                        # openings 4 and 5 arrived twice
    assert [row["bar_open_at"] for row in merged] == sorted(
        row["bar_open_at"] for row in merged)


def test_a_conflicting_duplicate_is_never_silently_resolved(capture):
    """Keeping "the last one" would bury a source contradicting itself."""
    plan = capture.plan_batches(_iso(GRID), _iso(GRID + HOUR * 9))
    marker = capture.batch_marker(plan[0], timeframe="1h")
    first = capture.canonical_rows_from_payload(_page(range(0, 6)), product="BTC-USD",
                                                timeframe="1h", marker=marker)
    other = capture.canonical_rows_from_payload(_page(range(4, 10), base=Decimal(500)),
                                                product="BTC-USD", timeframe="1h",
                                                marker=marker)
    with pytest.raises(capture.MarketHistoryCaptureError, match="conflicting payloads"):
        capture.merge_canonical_rows([first, other])


def test_canonical_bytes_are_line_delimited_and_round_trip(capture, tmp_path):
    plan = capture.plan_batches(_iso(GRID), _iso(GRID + HOUR * 4))
    rows = capture.canonical_rows_from_payload(
        _page(range(5)), product="BTC-USD", timeframe="1h",
        marker=capture.batch_marker(plan[0], timeframe="1h"))
    payload = capture.canonical_bytes(rows)
    assert payload.endswith(b"\n") and b"\r" not in payload
    assert payload.count(b"\n") == len(rows)
    path = tmp_path / "canonical.jsonl"
    path.write_bytes(payload)
    assert capture.load_canonical_rows(path) == rows
    assert capture.canonical_bytes(rows) == payload      # deterministic


# --- capture, offline ------------------------------------------------------


def test_a_capture_writes_raw_and_canonical_and_a_coherent_manifest(capture, tmp_path):
    manifest = _capture(capture, tmp_path, [_page(range(25))])
    entry = manifest["products"][0]
    base = tmp_path / capture.CORPUS_ID
    assert (base / "manifest.json").is_file()
    assert (base / entry["canonical_path"]).is_file()
    assert entry["canonical_rows"] == 25
    assert entry["missing_count"] == 0 and entry["duplicate_identical_count"] == 0
    for record in entry["batches"]:
        raw = (base / record["raw_path"]).read_bytes()
        # the raw response is kept byte-for-byte, not re-serialised
        assert hashlib.sha256(raw).hexdigest() == record["raw_sha256"]
        assert len(raw) == record["raw_bytes"]
    assert manifest["historical_candle_corpus"] is True
    assert manifest["point_in_time_exchange_revision_history"] is False
    assert capture.verify_corpus(tmp_path)["verified"] is True


def test_no_secret_or_machine_detail_reaches_the_artefacts(capture, tmp_path):
    _capture(capture, tmp_path, [_page(range(25))])
    text = (tmp_path / capture.CORPUS_ID / "manifest.json").read_text()
    for forbidden in ("Authorization", "api_key", "apikey", "token", "cookie",
                      "secret", "passphrase", "CB-ACCESS"):
        assert forbidden.lower() not in text.lower(), forbidden
    assert "/home/" not in text


def test_gaps_are_recorded_and_never_filled(capture, tmp_path):
    """A missing hour stays missing. Phase 2 already knows how to cope."""
    present = [index for index in range(25) if index not in {7, 8, 19}]
    manifest = _capture(capture, tmp_path, [_page(present)])
    entry = manifest["products"][0]
    assert entry["canonical_rows"] == 22
    assert entry["missing_count"] == 3
    assert entry["missing_openings"] == [
        (GRID + HOUR * index).isoformat() for index in (7, 8, 19)]
    rows = capture.load_canonical_rows(tmp_path / capture.CORPUS_ID / entry["canonical_path"])
    assert len(rows) == 22
    assert all(row["bar_open_at"] not in entry["missing_openings"] for row in rows)
    assert capture.verify_corpus(tmp_path)["verified"] is True


def test_the_capture_asks_for_exactly_the_planned_windows(capture, tmp_path):
    feed = Feed([_page(range(0, 10)), _page(range(10, 20))])
    capture.capture_corpus(tmp_path, products=("BTC-USD",), range_start=_iso(GRID),
                           range_end=_iso(GRID + HOUR * 19), batch_size=10,
                           fetch=feed, spacing_seconds=0)
    plan = capture.plan_batches(_iso(GRID), _iso(GRID + HOUR * 19), batch_size=10)
    assert len(feed.urls) == len(plan) == 2
    for url, batch in zip(feed.urls, plan):
        assert f"start={batch.requested_start}" in url
        assert f"end={batch.requested_end}" in url
        assert "granularity=3600" in url


def test_a_network_failure_stops_instead_of_inventing_candles(capture, tmp_path):
    def broken(url):
        raise OSError("connection reset")

    with pytest.raises(capture.MarketHistoryCaptureError, match="capture failed"):
        capture.capture_corpus(tmp_path, products=("BTC-USD",), range_start=_iso(GRID),
                               range_end=_iso(GRID + HOUR * 5), fetch=broken,
                               spacing_seconds=0)
    assert not (tmp_path / capture.CORPUS_ID / "manifest.json").exists()


def test_a_transient_failure_is_retried_a_bounded_number_of_times(capture, monkeypatch):
    monkeypatch.setattr(capture, "RETRY_BACKOFF_SECONDS", (0, 0, 0))
    attempts = []

    def flaky(url):
        attempts.append(url)
        if len(attempts) < 3:
            raise OSError("temporary")
        return _page(range(5))

    batch = capture.plan_batches(_iso(GRID), _iso(GRID + HOUR * 4))[0]
    assert capture._fetch_batch("BTC-USD", batch, timeframe="1h", fetch=flaky)
    assert len(attempts) == 3
    assert len(set(attempts)) == 1               # the window never moved

    always = []
    def dead(url):
        always.append(url)
        raise OSError("down")
    with pytest.raises(capture.MarketHistoryCaptureError):
        capture._fetch_batch("BTC-USD", batch, timeframe="1h", fetch=dead)
    assert len(always) == capture.MAX_ATTEMPTS   # bounded, never a loop


# --- identity --------------------------------------------------------------


def test_the_spec_hash_describes_the_request_and_the_content_hash_the_bytes(capture, tmp_path):
    first = _capture(capture, tmp_path, [_page(range(25))])
    other = _capture(capture, tmp_path / "other", [_page(range(25), base=Decimal(900))])
    assert first["corpus_spec_hash"] == other["corpus_spec_hash"]     # same request
    assert first["corpus_content_hash"] != other["corpus_content_hash"]  # different bytes


def test_the_spec_hash_moves_with_every_part_of_the_request(capture):
    baseline = capture._sha256_canonical(capture.corpus_spec())
    for kwargs in ({"products": ("BTC-USD",)}, {"timeframe": "1d"},
                   {"range_start": "2025-09-01T00:00:00Z"},
                   {"range_end": "2026-06-30T23:00:00Z"}, {"batch_size": 200}):
        assert capture._sha256_canonical(capture.corpus_spec(**kwargs)) != baseline


def test_the_manifest_hash_does_not_include_itself(capture, tmp_path):
    manifest = _capture(capture, tmp_path, [_page(range(25))])
    stored = manifest["manifest_content_sha256"]
    assert capture.manifest_content_sha256(manifest) == stored
    assert capture.manifest_content_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_content_sha256"}) == stored


def test_the_content_hash_covers_every_canonical_file(capture, tmp_path):
    manifest = _capture(capture, tmp_path, [_page(range(25))])
    entries = manifest["products"]
    baseline = capture.corpus_content_hash(entries)
    for index in range(len(entries)):
        mutated = [dict(entry) for entry in entries]
        mutated[index]["canonical_sha256"] = "0" * 64
        assert capture.corpus_content_hash(mutated) != baseline
    dropped = capture.corpus_content_hash(entries[:-1]) if len(entries) > 1 else None
    if dropped is not None:
        assert dropped != baseline


def test_two_captures_of_the_same_bytes_agree_on_the_content_hash(capture, tmp_path):
    pages = [_page(range(25))]
    first = _capture(capture, tmp_path / "a", pages)
    second = _capture(capture, tmp_path / "b", pages)
    assert first["corpus_content_hash"] == second["corpus_content_hash"]
    assert first["products"][0]["canonical_sha256"] == second["products"][0]["canonical_sha256"]


# --- verify ---------------------------------------------------------------


def test_verify_never_reaches_for_the_network(capture, tmp_path, monkeypatch):
    _capture(capture, tmp_path, [_page(range(25))])

    def explode(url):
        raise AssertionError("verify attempted a network call")

    monkeypatch.setattr(capture, "_http_get", explode)
    assert capture.verify_corpus(tmp_path)["verified"] is True


@pytest.mark.parametrize("target", ["raw", "canonical", "manifest"])
def test_a_single_flipped_byte_is_detected(capture, tmp_path, target):
    manifest = _capture(capture, tmp_path, [_page(range(25))])
    entry = manifest["products"][0]
    base = tmp_path / capture.CORPUS_ID
    path = {"raw": base / entry["batches"][0]["raw_path"],
            "canonical": base / entry["canonical_path"],
            "manifest": base / "manifest.json"}[target]
    data = bytearray(path.read_bytes())
    index = next(i for i, byte in enumerate(data) if chr(byte).isdigit())
    data[index] = ord("9") if chr(data[index]) != "9" else ord("8")
    path.write_bytes(bytes(data))
    with pytest.raises(capture.MarketHistoryCaptureError):
        capture.verify_corpus(tmp_path)


def test_a_byte_level_rewrite_that_parses_the_same_is_still_rejected(capture, tmp_path):
    """This is the check the canonical SHA-256 exists for.

    Flipping a digit is caught by re-deriving the canonical file from the raw
    responses -- the values stop matching. But a rewrite that reorders the JSON
    keys parses to exactly the same rows and has exactly the same length, so
    every semantic check passes. Only the byte hash notices, and without a test
    like this one that hash could be deleted and the suite would stay green.
    """
    manifest = _capture(capture, tmp_path, [_page(range(25))])
    entry = manifest["products"][0]
    path = tmp_path / capture.CORPUS_ID / entry["canonical_path"]
    before = path.read_bytes()

    lines = before.decode("utf-8").splitlines()
    first = json.loads(lines[0])
    reordered = json.dumps({key: first[key] for key in reversed(list(first))},
                           separators=(",", ":"))
    path.write_bytes(("\n".join([reordered, *lines[1:]]) + "\n").encode("utf-8"))
    after = path.read_bytes()

    assert after != before                                   # the bytes moved
    assert len(after) == len(before)                         # ... but not the length
    assert capture.load_canonical_rows(path)[0] == first     # ... and not the meaning
    with pytest.raises(capture.MarketHistoryCaptureError, match="canonical sha256"):
        capture.verify_corpus(tmp_path)


def test_verify_rejects_a_manifest_that_disagrees_with_its_own_files(capture, tmp_path):
    manifest = _capture(capture, tmp_path, [_page(range(25))])
    base = tmp_path / capture.CORPUS_ID
    manifest["products"][0]["canonical_rows"] = 24
    manifest["manifest_content_sha256"] = capture.manifest_content_sha256(manifest)
    (base / "manifest.json").write_bytes(
        (capture._canonical_json(manifest) + "\n").encode("utf-8"))
    with pytest.raises(capture.MarketHistoryCaptureError, match="row count mismatch"):
        capture.verify_corpus(tmp_path)


def test_verify_notices_a_deleted_artefact(capture, tmp_path):
    manifest = _capture(capture, tmp_path, [_page(range(25))])
    (tmp_path / capture.CORPUS_ID / manifest["products"][0]["batches"][0]["raw_path"]).unlink()
    with pytest.raises(capture.MarketHistoryCaptureError, match="missing raw batch"):
        capture.verify_corpus(tmp_path)


def test_verify_rejects_a_corpus_that_is_simply_absent(capture, tmp_path):
    with pytest.raises(capture.MarketHistoryCaptureError, match="no manifest"):
        capture.verify_corpus(tmp_path)


# --- replay ---------------------------------------------------------------


def test_replay_rebuilds_the_phase_one_chain_offline(capture, tmp_path, monkeypatch):
    manifest = _capture(capture, tmp_path, [_page(range(25))])

    def explode(url):
        raise AssertionError("replay attempted a network call")

    monkeypatch.setattr(capture, "_http_get", explode)
    result = capture.replay_corpus(tmp_path, product="BTC-USD",
                                   database_path=tmp_path / "replay.sqlite3")
    entry = manifest["products"][0]
    assert result["rows"] == result["points"] == entry["canonical_rows"]
    assert result["first_open"] == entry["first_open"]
    assert result["last_open"] == entry["last_open"]
    assert result["snapshot_id"].startswith("hyprl-market-snapshot-")


def test_replay_does_not_depend_on_the_wall_clock(capture, tmp_path):
    """Same frozen corpus, two runs, same snapshot identity."""
    _capture(capture, tmp_path, [_page(range(25))])
    first = capture.replay_corpus(tmp_path, product="BTC-USD",
                                  database_path=tmp_path / "one.sqlite3")
    second = capture.replay_corpus(tmp_path, product="BTC-USD",
                                   database_path=tmp_path / "two.sqlite3")
    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["entries_content_hash"] == second["entries_content_hash"]


def test_replay_preserves_the_gaps_instead_of_healing_them(capture, tmp_path):
    present = [index for index in range(25) if index not in {6, 13}]
    _capture(capture, tmp_path, [_page(present)])
    result = capture.replay_corpus(tmp_path, product="BTC-USD",
                                   database_path=tmp_path / "gapped.sqlite3")
    assert result["rows"] == result["points"] == 23
    assert result["missing_openings"] == 2


def test_replay_refuses_a_product_outside_the_corpus(capture, tmp_path):
    _capture(capture, tmp_path, [_page(range(25))])
    with pytest.raises(capture.MarketHistoryCaptureError, match="not part of this corpus"):
        capture.replay_corpus(tmp_path, product="ETH-USD",
                              database_path=tmp_path / "nope.sqlite3")


# --- the real corpus (offline, one integration check) ----------------------


REAL_CORPUS_ROOT = REPO_ROOT / "data" / "crypto"


@pytest.mark.skipif(not (REAL_CORPUS_ROOT / "coinbase_history_v1" / "manifest.json").is_file(),
                    reason="the real corpus has not been captured in this checkout")
def test_the_tracked_real_corpus_verifies_and_replays_offline(capture, tmp_path, monkeypatch):
    """One integration check over the real corpus -- not one per candle.

    Re-downloading is never part of a test run: the network function is
    replaced by something that raises, so a regression that reached for the
    exchange would fail here rather than silently pass on a good connection.
    """
    def explode(url):
        raise AssertionError("a test attempted to reach the exchange")

    monkeypatch.setattr(capture, "_http_get", explode)

    report = capture.verify_corpus(REAL_CORPUS_ROOT)
    assert report["verified"] is True
    manifest = capture.load_manifest(REAL_CORPUS_ROOT)

    # the frozen request, exactly as declared before any capture
    spec = manifest["spec"]
    assert spec["provider"] == "coinbase_exchange_rest"
    assert spec["products"] == ["BTC-USD", "ETH-USD"]
    assert spec["timeframe"] == "1h"
    assert spec["requested_range"] == {"start": "2025-08-01T00:00:00Z",
                                       "end": "2026-07-31T23:00:00Z"}
    assert manifest["point_in_time_exchange_revision_history"] is False
    assert manifest["historical_candle_corpus"] is True

    wanted = capture.expected_openings(spec["requested_range"]["start"],
                                       spec["requested_range"]["end"])
    assert len(wanted) == 8760
    for entry in manifest["products"]:
        assert entry["first_open"] == wanted[0]
        assert entry["last_open"] == wanted[-1]
        assert entry["canonical_rows"] + entry["missing_count"] == len(wanted)
        assert entry["canonical_rows"] > 8000          # a real year, not a stub
        rows = capture.load_canonical_rows(
            REAL_CORPUS_ROOT / capture.CORPUS_ID / entry["canonical_path"])
        assert len(rows) == entry["canonical_rows"]
        assert all(Decimal(row["close"]) > 0 for row in rows)

    # and it still rebuilds the Phase 1 chain from those files alone
    result = capture.replay_corpus(REAL_CORPUS_ROOT, product="BTC-USD",
                                   database_path=tmp_path / "real.sqlite3")
    btc = next(e for e in manifest["products"] if e["product"] == "BTC-USD")
    assert result["points"] == btc["canonical_rows"]
    assert result["missing_openings"] == btc["missing_count"]
