"""Phase 4A: capture a real Coinbase history once, then freeze it as input data.

Everything Phase 1-3 proved was proved against synthetic fixtures. This module
exists to replace that input with something real -- and, more importantly, to
make the real thing auditable: captured once, hashed, and replayable offline
forever after by anyone who has the repository.

Three commands, deliberately separated:

* `capture` is the only one allowed to touch the network. It writes the raw
  responses exactly as received, derives a canonical series from them, and
  records a manifest.
* `verify` recomputes every hash and invariant from the files alone. It never
  imports a socket, and a test proves it still passes when the HTTP function
  is replaced by something that explodes.
* `replay` rebuilds the Phase 1 chain from the frozen corpus into a temporary
  database. Also offline.

Two things are kept apart on purpose. The **raw** responses are never
discarded in favour of the tidy derived form -- a canonicalisation bug is
recoverable only if the source survives. And the corpus **identity** is split
in two: `corpus_spec_hash` describes what was asked for (provider, products,
range, protocol), while `corpus_content_hash` describes the bytes that came
back. Re-running the same request against a source that has changed its mind
produces the same spec hash and a different content hash, which is exactly the
signal an audit needs.

Honest limitation: this is a historical candle corpus, not a point-in-time
record of what Coinbase would have returned at each instant in the past. If
the exchange ever revised a candle, we captured the revised value. Causal
backtesting still holds -- a feature at T only ever reads openings <= T -- but
nothing here reconstructs a revision history, and nothing should pretend to.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request

from scripts.trading_lab.coinbase_candles import (
    MAX_CANDLES_PER_RESPONSE,
    TIMEFRAME_DURATIONS,
    adapt_coinbase_candles,
)

CORPUS_SCHEMA_VERSION = "trading-lab.market-corpus.v1"
CAPTURE_PROTOCOL_VERSION = "coinbase-exchange-rest-batched-v1"
CORPUS_ID = "coinbase_history_v1"
CORPUS_PROVIDER = "coinbase_exchange_rest"

# Frozen BEFORE any capture and before any model ever sees this data. Moving a
# range after looking at performance is how a backtest gets quietly curated.
CORPUS_PRODUCTS = ("BTC-USD", "ETH-USD")
CORPUS_TIMEFRAME = "1h"
CORPUS_RANGE_START = "2025-08-01T00:00:00Z"
CORPUS_RANGE_END = "2026-07-31T23:00:00Z"  # inclusive, i.e. the last bar OPENING

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
COINBASE_GRANULARITY = {"1h": 3600, "1d": 86400}
CAPTURE_USER_AGENT = "hyprl-trading-lab-capture/1.0"
BATCH_SIZE = MAX_CANDLES_PER_RESPONSE  # the adapter refuses more than this per page
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_SPACING_SECONDS = 0.35  # fixed pacing, not a random backoff
MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = (1.0, 3.0, 8.0)


class MarketHistoryCaptureError(RuntimeError):
    """Raised when a corpus cannot be captured, verified or replayed safely."""


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str, *, field: str) -> datetime:
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise MarketHistoryCaptureError(f"{field} is not ISO-8601: {value!r}") from error
    if moment.tzinfo is None:
        raise MarketHistoryCaptureError(f"{field} must carry a timezone: {value!r}")
    return moment.astimezone(timezone.utc)


# --- deterministic batch planning ----------------------------------------


@dataclass(frozen=True)
class CaptureBatch:
    index: int
    requested_start: str
    requested_end: str
    expected_openings: int


def plan_batches(range_start: str, range_end: str, *, timeframe: str = CORPUS_TIMEFRAME,
                 batch_size: int = BATCH_SIZE) -> tuple[CaptureBatch, ...]:
    """Split an inclusive opening range into non-overlapping windows.

    The plan is a pure function of the range, the timeframe and the batch
    size -- never of what the network happens to return. Two captures of the
    same corpus therefore ask for exactly the same windows, which is what
    makes the raw artefacts comparable at all.
    """
    if timeframe not in TIMEFRAME_DURATIONS:
        raise MarketHistoryCaptureError(f"unsupported timeframe {timeframe!r}")
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_CANDLES_PER_RESPONSE:
        raise MarketHistoryCaptureError(
            f"batch_size must be an int in 1..{MAX_CANDLES_PER_RESPONSE}, got {batch_size!r}")
    duration = TIMEFRAME_DURATIONS[timeframe]
    start = _parse_iso(range_start, field="range_start")
    end = _parse_iso(range_end, field="range_end")
    if end < start:
        raise MarketHistoryCaptureError("range_end precedes range_start")
    for label, moment in (("range_start", start), ("range_end", end)):
        if (moment - datetime(1970, 1, 1, tzinfo=timezone.utc)) % duration:
            raise MarketHistoryCaptureError(f"{label} is not aligned on the {timeframe} grid")

    batches: list[CaptureBatch] = []
    cursor = start
    while cursor <= end:
        last = min(cursor + duration * (batch_size - 1), end)
        batches.append(CaptureBatch(
            index=len(batches),
            requested_start=_iso(cursor),
            requested_end=_iso(last),
            expected_openings=int((last - cursor) / duration) + 1,
        ))
        cursor = last + duration
    return tuple(batches)


def _opening_text(moment: datetime) -> str:
    """The opening format Phase 1 already emits (`+00:00`), not a second one.

    `market_bar` writes `opened.isoformat()`. Inventing a `Z` variant here
    would make every comparison against a canonical row silently fail.
    """
    return moment.astimezone(timezone.utc).isoformat()


def expected_openings(range_start: str, range_end: str,
                      *, timeframe: str = CORPUS_TIMEFRAME) -> tuple[str, ...]:
    duration = TIMEFRAME_DURATIONS[timeframe]
    start = _parse_iso(range_start, field="range_start")
    end = _parse_iso(range_end, field="range_end")
    count = int((end - start) / duration) + 1
    return tuple(_opening_text(start + duration * index) for index in range(count))


# --- network (capture only) ----------------------------------------------


def _http_get(url: str) -> bytes:
    """The single network surface. `verify` and `replay` never call this."""
    request = urllib.request.Request(url, headers={"User-Agent": CAPTURE_USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read()


def _batch_url(product: str, batch: CaptureBatch, *, timeframe: str) -> str:
    base = COINBASE_CANDLES_URL.format(product=product)
    return (f"{base}?granularity={COINBASE_GRANULARITY[timeframe]}"
            f"&start={batch.requested_start}&end={batch.requested_end}")


def _fetch_batch(product: str, batch: CaptureBatch, *, timeframe: str,
                 fetch=_http_get) -> bytes:
    """Bounded retry for transient failures. The requested window never moves."""
    url = _batch_url(product, batch, timeframe=timeframe)
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return fetch(url)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as error:
            last_error = error
            if attempt == MAX_ATTEMPTS - 1:
                break
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])
    raise MarketHistoryCaptureError(
        f"capture failed for {product} batch {batch.index} "
        f"[{batch.requested_start}..{batch.requested_end}]: {last_error!r}")


# --- canonicalisation -----------------------------------------------------


def batch_marker(batch: "CaptureBatch", *, timeframe: str) -> str:
    """A SYNTHETIC deterministic replay marker derived from the plan.

    Three clocks must not be conflated, and this function returns the third:

    1. **bar time** -- `bar_open_at` / `bar_close_at`, when the market moved;
    2. **actual capture time** -- when these bytes were really obtained from
       the exchange, recorded separately in the manifest;
    3. **this marker** -- a synthetic value whose only job is to satisfy Phase
       1's `available_at >= bar_close_at` rule so a bar can be constructed at
       all.

    It is emphatically NOT a claim that the value was known to the world at
    this instant. Taking the close of the last opening the batch asked for
    keeps it a pure function of the plan -- capture and verify compute the
    identical value years apart -- and makes a response containing a bar
    beyond the requested window fail closed rather than pass unnoticed.
    """
    return _iso(_parse_iso(batch.requested_end, field="requested_end")
                + TIMEFRAME_DURATIONS[timeframe])


def canonical_rows_from_payload(raw_payload: bytes, *, product: str, timeframe: str,
                                marker: str) -> tuple[dict[str, str], ...]:
    """Normalise one raw response through the Phase 1 adapter, ascending.

    Numbers travel as their exact textual tokens: the adapter parses JSON
    numbers straight into `Decimal`, so no value ever passes through a binary
    float on the way in or out. `marker` only satisfies Phase 1's causality
    check; it never touches the OHLCV values.
    """
    bars = adapt_coinbase_candles(raw_payload, product_id=product, timeframe=timeframe,
                                  available_at=marker, ingested_at=marker)
    rows = tuple(
        {
            "bar_open_at": bar["bar_open_at"],
            "open": str(Decimal(bar["open"])),
            "high": str(Decimal(bar["high"])),
            "low": str(Decimal(bar["low"])),
            "close": str(Decimal(bar["close"])),
            "volume": str(Decimal(bar["volume"])),
        }
        for bar in bars
    )
    openings = [row["bar_open_at"] for row in rows]
    if openings != sorted(openings):
        raise MarketHistoryCaptureError("adapter returned rows out of order")
    return rows


def merge_canonical_rows(batches_rows) -> tuple[tuple[dict[str, str], ...], int]:
    """Merge batch results, ascending, with an explicit duplicate policy.

    An opening seen twice with a byte-identical payload is a harmless artefact
    of window boundaries and is counted. An opening seen twice with DIFFERENT
    values inside one capture means the source contradicted itself; picking
    "the last one" would bury that, so it fails closed.
    """
    merged: dict[str, dict[str, str]] = {}
    identical_duplicates = 0
    for rows in batches_rows:
        for row in rows:
            opening = row["bar_open_at"]
            existing = merged.get(opening)
            if existing is None:
                merged[opening] = row
                continue
            if existing == row:
                identical_duplicates += 1
                continue
            raise MarketHistoryCaptureError(
                f"conflicting payloads for opening {opening}: {existing} != {row}")
    ordered = tuple(merged[key] for key in sorted(merged))
    return ordered, identical_duplicates


def canonical_bytes(rows) -> bytes:
    """One canonical JSON object per line. No trailing whitespace, LF endings."""
    return "".join(f"{_canonical_json(row)}\n" for row in rows).encode("utf-8")


def load_canonical_rows(path: pathlib.Path) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    text = path.read_bytes().decode("utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise MarketHistoryCaptureError(f"{path.name}:{number} is empty")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise MarketHistoryCaptureError(f"{path.name}:{number} is not JSON") from error
        if not isinstance(row, dict):
            raise MarketHistoryCaptureError(f"{path.name}:{number} is not an object")
        rows.append(row)
    return tuple(rows)


# --- corpus identity ------------------------------------------------------


def corpus_spec(products=CORPUS_PRODUCTS, *, timeframe: str = CORPUS_TIMEFRAME,
                range_start: str = CORPUS_RANGE_START,
                range_end: str = CORPUS_RANGE_END,
                batch_size: int = BATCH_SIZE) -> dict[str, object]:
    """What was ASKED FOR. Never what came back."""
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "provider": CORPUS_PROVIDER,
        "products": list(products),
        "timeframe": timeframe,
        "requested_range": {"start": range_start, "end": range_end},
        "capture_protocol_version": CAPTURE_PROTOCOL_VERSION,
        "batch_size": batch_size,
    }


def corpus_content_hash(product_entries) -> str:
    """What CAME BACK: every canonical file and every raw batch, in order."""
    return _sha256_canonical([
        {
            "product": entry["product"],
            "canonical_sha256": entry["canonical_sha256"],
            "canonical_rows": entry["canonical_rows"],
            "raw_sha256": [batch["raw_sha256"] for batch in entry["batches"]],
        }
        for entry in sorted(product_entries, key=lambda entry: entry["product"])
    ])


def manifest_content_sha256(manifest: dict) -> str:
    """Hash the manifest WITHOUT its own hash field -- no self-reference."""
    return _sha256_canonical({k: v for k, v in manifest.items()
                              if k != "manifest_content_sha256"})


# --- capture (the only networked command) ---------------------------------


def _write(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def capture_corpus(root, *, products=CORPUS_PRODUCTS, timeframe: str = CORPUS_TIMEFRAME,
                   range_start: str = CORPUS_RANGE_START, range_end: str = CORPUS_RANGE_END,
                   batch_size: int = BATCH_SIZE, fetch=_http_get,
                   spacing_seconds: float = REQUEST_SPACING_SECONDS) -> dict:
    """Fetch the frozen range once and write raw + canonical + manifest."""
    base = pathlib.Path(root) / CORPUS_ID
    batches = plan_batches(range_start, range_end, timeframe=timeframe, batch_size=batch_size)
    started = _iso(datetime.now(timezone.utc))

    product_entries: list[dict] = []
    for product in products:
        batch_records: list[dict] = []
        batch_rows: list[tuple[dict[str, str], ...]] = []
        for batch in batches:
            raw = _fetch_batch(product, batch, timeframe=timeframe, fetch=fetch)
            if not isinstance(raw, bytes):
                raise MarketHistoryCaptureError("fetch must return bytes")
            rows = canonical_rows_from_payload(
                raw, product=product, timeframe=timeframe,
                marker=batch_marker(batch, timeframe=timeframe))
            relative = f"{product}/raw/batch_{batch.index:04d}.json"
            _write(base / relative, raw)
            batch_records.append({
                "index": batch.index,
                "requested_start": batch.requested_start,
                "requested_end": batch.requested_end,
                "expected_openings": batch.expected_openings,
                "raw_path": relative,
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_bytes": len(raw),
                "raw_candle_count": len(rows),
                "captured_at": _iso(datetime.now(timezone.utc)),
                "source_url": _batch_url(product, batch, timeframe=timeframe),
            })
            batch_rows.append(rows)
            if spacing_seconds:
                time.sleep(spacing_seconds)

        rows, identical_duplicates = merge_canonical_rows(batch_rows)
        payload = canonical_bytes(rows)
        canonical_relative = f"{product}/canonical.jsonl"
        _write(base / canonical_relative, payload)
        wanted = expected_openings(range_start, range_end, timeframe=timeframe)
        present = {row["bar_open_at"] for row in rows}
        missing = tuple(opening for opening in wanted if opening not in present)
        product_entries.append({
            "product": product,
            "batches": batch_records,
            "canonical_path": canonical_relative,
            "canonical_sha256": hashlib.sha256(payload).hexdigest(),
            "canonical_bytes": len(payload),
            "canonical_rows": len(rows),
            "first_open": rows[0]["bar_open_at"] if rows else None,
            "last_open": rows[-1]["bar_open_at"] if rows else None,
            "missing_count": len(missing),
            "missing_openings": list(missing),
            "duplicate_identical_count": identical_duplicates,
        })

    specification = corpus_spec(products, timeframe=timeframe, range_start=range_start,
                               range_end=range_end, batch_size=batch_size)
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "spec": specification,
        "corpus_spec_hash": _sha256_canonical(specification),
        "capture_started_at": started,
        "capture_completed_at": _iso(datetime.now(timezone.utc)),
        "products": product_entries,
        "corpus_content_hash": corpus_content_hash(product_entries),
        "point_in_time_exchange_revision_history": False,
        "historical_candle_corpus": True,
    }
    manifest["manifest_content_sha256"] = manifest_content_sha256(manifest)
    _write(base / "manifest.json", (_canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


# --- verify (offline) -----------------------------------------------------


def load_manifest(root) -> dict:
    path = pathlib.Path(root) / CORPUS_ID / "manifest.json"
    if not path.is_file():
        raise MarketHistoryCaptureError(f"no manifest at {path}")
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except json.JSONDecodeError as error:
        raise MarketHistoryCaptureError("manifest is not valid JSON") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MarketHistoryCaptureError(message)


def verify_corpus(root) -> dict:
    """Recompute every hash and invariant from the files. Never touches the network."""
    base = pathlib.Path(root) / CORPUS_ID
    manifest = load_manifest(root)

    _require(manifest.get("schema_version") == CORPUS_SCHEMA_VERSION,
             f"unexpected schema_version {manifest.get('schema_version')!r}")
    _require(manifest.get("corpus_id") == CORPUS_ID, "unexpected corpus_id")
    _require(manifest_content_sha256(manifest) == manifest.get("manifest_content_sha256"),
             "manifest_content_sha256 does not match the manifest body")

    specification = manifest["spec"]
    _require(_sha256_canonical(specification) == manifest.get("corpus_spec_hash"),
             "corpus_spec_hash does not match the specification")
    _require(specification["provider"] == CORPUS_PROVIDER, "unexpected provider")
    timeframe = specification["timeframe"]
    _require(timeframe in TIMEFRAME_DURATIONS, f"unsupported timeframe {timeframe!r}")
    range_start = specification["requested_range"]["start"]
    range_end = specification["requested_range"]["end"]
    wanted = expected_openings(range_start, range_end, timeframe=timeframe)
    planned = plan_batches(range_start, range_end, timeframe=timeframe,
                           batch_size=specification["batch_size"])
    _require([entry["product"] for entry in manifest["products"]] == specification["products"],
             "manifest products do not match the specification")

    duration = TIMEFRAME_DURATIONS[timeframe]
    report: dict[str, object] = {"products": {}, "timeframe": timeframe,
                                 "requested_range": {"start": range_start, "end": range_end}}
    for entry in manifest["products"]:
        product = entry["product"]
        _require(len(entry["batches"]) == len(planned),
                 f"{product}: manifest declares {len(entry['batches'])} batches, "
                 f"the plan has {len(planned)}")
        rebuilt: list[tuple[dict[str, str], ...]] = []
        for record, batch in zip(entry["batches"], planned):
            _require(record["requested_start"] == batch.requested_start
                     and record["requested_end"] == batch.requested_end,
                     f"{product}: batch {record['index']} window drifted from the plan")
            raw_path = base / record["raw_path"]
            _require(raw_path.is_file(), f"{product}: missing raw batch {record['raw_path']}")
            raw = raw_path.read_bytes()
            _require(hashlib.sha256(raw).hexdigest() == record["raw_sha256"],
                     f"{product}: raw sha256 mismatch for {record['raw_path']}")
            _require(len(raw) == record["raw_bytes"],
                     f"{product}: raw byte size mismatch for {record['raw_path']}")
            rows = canonical_rows_from_payload(
                raw, product=product, timeframe=timeframe,
                marker=batch_marker(batch, timeframe=timeframe))
            _require(len(rows) == record["raw_candle_count"],
                     f"{product}: candle count mismatch for {record['raw_path']}")
            rebuilt.append(rows)

        derived, identical_duplicates = merge_canonical_rows(rebuilt)
        _require(identical_duplicates == entry["duplicate_identical_count"],
                 f"{product}: duplicate_identical_count mismatch")

        canonical_path = base / entry["canonical_path"]
        _require(canonical_path.is_file(), f"{product}: missing {entry['canonical_path']}")
        payload = canonical_path.read_bytes()
        _require(hashlib.sha256(payload).hexdigest() == entry["canonical_sha256"],
                 f"{product}: canonical sha256 mismatch")
        _require(len(payload) == entry["canonical_bytes"],
                 f"{product}: canonical byte size mismatch")
        stored = load_canonical_rows(canonical_path)
        _require(stored == derived,
                 f"{product}: canonical file does not match what the raw responses derive")
        _require(len(stored) == entry["canonical_rows"],
                 f"{product}: canonical row count mismatch")

        openings = [row["bar_open_at"] for row in stored]
        _require(openings == sorted(openings), f"{product}: canonical rows are not ascending")
        _require(len(set(openings)) == len(openings), f"{product}: duplicate openings on disk")
        for opening in openings:
            _require(opening in set(wanted), f"{product}: opening {opening} outside the range")
            moment = _parse_iso(opening, field="bar_open_at")
            _require((moment - _parse_iso(range_start, field="range_start")) % duration
                     == timedelta(0), f"{product}: opening {opening} off the grid")
        for row in stored:
            low, high = Decimal(row["low"]), Decimal(row["high"])
            open_, close = Decimal(row["open"]), Decimal(row["close"])
            _require(low <= min(open_, close) and high >= max(open_, close) and low <= high,
                     f"{product}: OHLC invariant violated at {row['bar_open_at']}")
            _require(Decimal(row["volume"]) >= 0,
                     f"{product}: negative volume at {row['bar_open_at']}")

        present = set(openings)
        missing = tuple(opening for opening in wanted if opening not in present)
        _require(list(missing) == entry["missing_openings"],
                 f"{product}: missing_openings mismatch")
        _require(len(missing) == entry["missing_count"], f"{product}: missing_count mismatch")
        _require(entry["first_open"] == (openings[0] if openings else None),
                 f"{product}: first_open mismatch")
        _require(entry["last_open"] == (openings[-1] if openings else None),
                 f"{product}: last_open mismatch")
        report["products"][product] = {
            "rows": len(stored), "first_open": entry["first_open"],
            "last_open": entry["last_open"], "missing_count": entry["missing_count"],
            "duplicate_identical_count": entry["duplicate_identical_count"],
            "batches": len(entry["batches"]),
            "raw_bytes": sum(record["raw_bytes"] for record in entry["batches"]),
        }

    _require(corpus_content_hash(manifest["products"]) == manifest.get("corpus_content_hash"),
             "corpus_content_hash does not match the captured content")
    report["corpus_spec_hash"] = manifest["corpus_spec_hash"]
    report["corpus_content_hash"] = manifest["corpus_content_hash"]
    report["manifest_content_sha256"] = manifest["manifest_content_sha256"]
    report["verified"] = True
    return report


# --- replay (offline) -----------------------------------------------------


def _coinbase_page(rows) -> bytes:
    """Rebuild a Coinbase-shaped page so replay re-enters the Phase 1 adapter."""
    page = [
        [int(_parse_iso(row["bar_open_at"], field="bar_open_at").timestamp()),
         row["low"], row["high"], row["open"], row["close"], row["volume"]]
        for row in rows
    ]
    return json.dumps(page, separators=(",", ":")).encode("utf-8")


def replay_corpus(root, *, product: str, database_path) -> dict:
    """Rebuild the Phase 1 chain from the frozen corpus alone. No network.

    `ingested_at` and `available_at` are SYNTHETIC deterministic replay
    markers taken from the manifest's capture timestamps rather than from the
    clock. Contract A treats them as declared values, so replaying the same
    corpus twice yields the same snapshot identity instead of a fresh one per
    run -- which is the whole point. They say "this is the ordering the replay
    assumes", never "this is when the exchange first published this candle".
    """
    from scripts.trading_lab.market_data_store import MarketDataStore
    from scripts.trading_lab.market_series import load_market_series
    from scripts.trading_lab.market_snapshots import _materialize_snapshot

    base = pathlib.Path(root) / CORPUS_ID
    manifest = load_manifest(root)
    entry = next((item for item in manifest["products"] if item["product"] == product), None)
    _require(entry is not None, f"{product} is not part of this corpus")
    timeframe = manifest["spec"]["timeframe"]
    duration = TIMEFRAME_DURATIONS[timeframe]
    rows = load_canonical_rows(base / entry["canonical_path"])
    _require(bool(rows), f"{product}: canonical series is empty")

    declared = manifest["capture_completed_at"]
    ingested = _iso(_parse_iso(declared, field="capture_completed_at") + timedelta(seconds=1))
    store = MarketDataStore(database_path)
    for start in range(0, len(rows), MAX_CANDLES_PER_RESPONSE):
        chunk = rows[start:start + MAX_CANDLES_PER_RESPONSE]
        store.ingest_coinbase_response(_coinbase_page(chunk), product_id=product,
                                       timeframe=timeframe, available_at=declared,
                                       ingested_at=ingested)

    connection = store._connect()
    try:
        last = _parse_iso(rows[-1]["bar_open_at"], field="bar_open_at")
        snapshot = _materialize_snapshot(
            connection, provider=CORPUS_PROVIDER, product_id=product, timeframe=timeframe,
            range_start=rows[0]["bar_open_at"], range_end=_iso(last + duration),
            as_of=_iso(_parse_iso(ingested, field="ingested_at") + timedelta(seconds=1)),
        )
        series = load_market_series(connection, snapshot_id=snapshot.snapshot_id)
    finally:
        connection.close()

    _require(len(series.points) == len(rows),
             f"{product}: replay produced {len(series.points)} points for {len(rows)} rows")
    for row, point in zip(rows, series.points):
        _require(point.bar_open_at == row["bar_open_at"], f"{product}: opening drifted on replay")
        for field in ("open", "high", "low", "close", "volume"):
            _require(getattr(point, field) == Decimal(row[field]),
                     f"{product}: {field} drifted on replay at {row['bar_open_at']}")
    return {
        "product": product,
        "rows": len(rows),
        "points": len(series.points),
        "snapshot_id": snapshot.snapshot_id,
        "entries_content_hash": series.entries_content_hash,
        "missing_openings": len(series.missing_openings),
        "first_open": series.points[0].bar_open_at,
        "last_open": series.points[-1].bar_open_at,
    }
