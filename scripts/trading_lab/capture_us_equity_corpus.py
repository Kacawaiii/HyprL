"""Capture the first US equity corpus once, then freeze it.

Three commands, deliberately separated, the same split the Coinbase capture
uses and for the same reasons:

* ``capture`` is the only one allowed to touch the network. It writes each raw
  response byte-for-byte before anything parses it, derives canonical bars,
  audits them against the calendar, and writes a manifest.
* ``verify`` recomputes every hash and invariant from the files alone. It
  never opens a socket, and a test proves it still passes when the transport
  is replaced by something that explodes on use.
* ``rebuild`` reconstructs the canonical files from the raw ones and asserts
  they come out byte-identical.

Order of operations inside ``capture``, and none of it is negotiable:

1. The credential is checked **before** any request. No key means a clean stop
   with nothing written -- not a partial corpus, not an empty directory, not a
   manifest saying it tried.
2. Reference metadata is fetched **first** and each instrument's venue and
   asset class are checked against the registry. QQQ is not registered on
   `xnas` because that is where ETFs usually live; it is registered there
   because the metadata says so, and if the provider disagrees the capture
   stops before a single bar is stored under a name that means something else.
3. Bars are paged with a bound and a cycle check. A provider that returns the
   same page forever stops the run instead of filling a disk.
4. Every bar must sit on the calendar's expected grid. A provider returning
   pre-market, after-hours or holiday rows is normal; accepting them is not.
5. A failure anywhere leaves ``captured=false``. A half-corpus that looks
   whole is worse than no corpus.

**On the provider response shape.** The adapter below states exactly which
fields it reads and refuses anything it does not recognise, rather than
guessing at a shape. If the live API differs from this contract the capture
stops on the first response with a message naming the mismatch -- which is the
correct outcome, because silently coercing an unknown schema is how a corpus
ends up full of plausible nonsense.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from datetime import datetime, timezone

from scripts.trading_lab.credentials import (
    CredentialError, MASSIVE_API_KEY_ENV, massive_credentials)
from scripts.trading_lab.equity_corpus import (
    CORPUS_ROOT, CORPUS_SPEC_V1, EquityCorpusError, MAX_PAGES_PER_REQUEST,
    MAX_ROWS_TOTAL, RequestIdentity, SplitRecord, USEquityCorpusV1,
    accept_bar_opening, audit_gaps, build_canonical_bar, canonical_json,
    canonical_order, corporate_actions_hash, corpus_content_hash,
    instrument_content_hash, iso, overlapping_missing_intervals, parse_utc,
    serialise_rows, sha256_bytes, sha256_canonical)
from scripts.trading_lab.instrument_registry import CATALOGUE_V1
from scripts.trading_lab.instruments import InstrumentError, InstrumentId
from scripts.trading_lab.massive_http_transport import strip_credential_params
from scripts.trading_lab.massive_provider import (
    ADJUSTED_FLAG, AGGREGATE_FIELDS, MAX_ROWS_PER_PAGE, MASSIVE_STOCKS_HISTORICAL_V1,
    MassiveProviderError, SPLITS_ENDPOINT, adapt_aggregate_rows,
    adapt_split_rows, bars_path, parse_reference_ticker, reference_path,
    require_aggregate_window)

MANIFEST_SCHEMA_VERSION = "trading-lab.us-equity-corpus-manifest.v1"

# The fields the vendor's aggregate rows must carry. Named so a schema
# mismatch is reported against a list rather than discovered as a KeyError
# halfway through a two-year range.
REQUIRED_BAR_FIELDS = tuple(sorted(AGGREGATE_FIELDS))

SPLITS_PATH = SPLITS_ENDPOINT

# The timeframe recorded on a corporate-action request. Not a real timeframe:
# it marks the request identity so a verifier can tell a splits response from
# a bars response without sniffing the body, which both endpoints shape alike.
CORPORATE_ACTIONS_MARKER = "corporate-actions"


class CaptureError(RuntimeError):
    """Raised when a capture cannot proceed or cannot be trusted."""


class MissingCredentialStop(CaptureError):
    """A clean, expected stop: there is no key, so there is no capture."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --- layout ----------------------------------------------------------------


class CorpusLayout:
    """Where everything lives. One place, so verify and rebuild agree."""

    def __init__(self, root: pathlib.Path):
        self.root = pathlib.Path(root)

    @property
    def spec_path(self) -> pathlib.Path:
        return self.root / "spec.json"

    @property
    def manifest_path(self) -> pathlib.Path:
        return self.root / "manifest.json"

    def raw_dir(self, instrument_id: str) -> pathlib.Path:
        return self.root / "raw" / _slug(instrument_id)

    def canonical_path(self, instrument_id: str) -> pathlib.Path:
        return self.root / "canonical" / f"{_slug(instrument_id)}.jsonl"

    def corporate_actions_path(self, instrument_id: str) -> pathlib.Path:
        return self.root / "corporate_actions" / f"{_slug(instrument_id)}.json"

    def ensure(self) -> "CorpusLayout":
        for child in ("raw", "canonical", "corporate_actions"):
            (self.root / child).mkdir(parents=True, exist_ok=True)
        return self


def _slug(instrument_id: str) -> str:
    """A filesystem name that keeps the venue, because the venue is identity."""
    return instrument_id.replace(":", "_")


# --- the provider response adapter ----------------------------------------


def adapt_bar_rows(payload: dict, *, spec: USEquityCorpusV1,
                   instrument_id: str) -> list[dict]:
    """Read the rows out of an aggregates response, or refuse the response.

    Delegates to the provider's adapter so the capture runner and the provider
    can never disagree about what a row is. Everything it checks -- the echoed
    adjustment flag, the echoed ticker, the row keys, the millisecond
    timestamp -- happens before a single price is read.
    """
    return adapt_aggregate_rows(payload, instrument_id=instrument_id,
                                requested_policy=spec.adjustment_policy)


def next_cursor(payload: dict):
    """The provider's continuation URL, stripped of anything secret.

    The vendor paginates with a full ``next_url`` rather than an opaque token.
    That URL is data from the network, so it is never followed as given: the
    transport re-checks it against the host allowlist, and any credential-
    shaped query parameter is removed here before the value is used as a
    request identity or written into raw metadata. Some vendors embed the API
    key in next_url, and that value would otherwise be committed forever.
    """
    for name in ("next_url", "next_cursor", "next_page_token", "next"):
        value = payload.get(name)
        if value:
            return strip_credential_params(str(value))
    return None


# --- reference metadata gate ----------------------------------------------


def verify_instrument_identity(provider, instrument_id: str) -> dict:
    """Prove the venue and asset class before storing anything under them.

    The registry already asserts these against a recorded fixture. This checks
    the *live* provider agrees, because a fixture proves what the vendor said
    once and a capture stores what it says now. A disagreement means the
    ticker has been reassigned, the vendor has changed its coding, or we are
    about to file one company's prices under another company's name.
    """
    spec = CATALOGUE_V1.resolve(instrument_id)
    symbol = spec.instrument_id.symbol
    reference = provider.get_reference_ticker(symbol)

    # One canonical comparison rather than two raw string ones. The provider's
    # answer is turned into an instrument identity and compared as an identity,
    # so a venue that disagrees and a ticker that was never asked about are the
    # same failure -- which is what they are. Comparing the symbols as raw
    # strings would also miss a separator difference that canonicalisation
    # folds, and pass a pair that only looks equal.
    try:
        reported = InstrumentId(venue=reference.venue, symbol=reference.symbol)
    except InstrumentError as error:
        raise CaptureError(
            f"{symbol}: the provider's reference metadata does not describe a "
            f"usable market identity: {error}") from error
    if reported != spec.instrument_id:
        raise CaptureError(
            f"asked the provider about {spec.canonical_id} and it described "
            f"{reported.canonical}. Refusing to capture: these are a different "
            "instrument, and storing one under the other's name would be "
            "unrecoverable.")
    if reference.asset_class != spec.asset_class:
        raise CaptureError(
            f"{symbol}: the provider reports asset class "
            f"{reference.asset_class!r} but the registry has "
            f"{spec.asset_class!r}. An ETF and a common share are not "
            "interchangeable.")
    if not reference.active:
        raise CaptureError(f"{symbol}: the provider reports the ticker inactive")
    return {
        "instrument_id": spec.canonical_id,
        "symbol": symbol,
        "venue": reference.venue,
        "asset_class": reference.asset_class,
        "display_name": reference.display_name,
        "currency": reference.currency,
        "venue_verified_against_provider": True,
    }


# --- capture ---------------------------------------------------------------


class RawStore:
    """Raw responses, written once, never overwritten silently.

    A repeated request identity is either a bug in the pagination loop or a
    provider looping, and both are worth stopping for. When the bytes are
    identical the run continues and records the duplicate; when they differ,
    two different answers to the same question have been observed and the
    capture cannot claim either.
    """

    def __init__(self, layout: CorpusLayout):
        self.layout = layout
        self.records: list[dict] = []
        self._by_identity: dict[str, str] = {}
        self.duplicate_responses = 0

    def write(self, identity: RequestIdentity, raw: bytes, *,
              url: str, sequence: int) -> dict:
        digest = sha256_bytes(raw)
        previous = self._by_identity.get(identity.identity_hash)
        if previous is not None:
            if previous != digest:
                raise CaptureError(
                    f"the same request identity returned two different "
                    f"responses ({previous[:12]} then {digest[:12]}); the "
                    "capture cannot claim either as the answer")
            self.duplicate_responses += 1
            return next(record for record in self.records
                        if record["request_identity_hash"] == identity.identity_hash)

        directory = self.layout.raw_dir(identity.instrument_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{identity.slug}.json"
        if path.exists():
            raise CaptureError(f"refusing to overwrite existing raw file {path}")
        path.write_bytes(raw)

        record = {
            "instrument_id": identity.instrument_id,
            "request_identity": identity.canonical(),
            "request_identity_hash": identity.identity_hash,
            "raw_path": str(path.relative_to(self.layout.root)),
            "raw_sha256": digest,
            "raw_bytes": len(raw),
            "capture_sequence": sequence,
            "captured_at": _now(),
            # The URL never carries the credential -- it is header-only, and
            # the transport refuses a key-shaped query string.
            "source_url": url,
        }
        self.records.append(record)
        self._by_identity[identity.identity_hash] = digest
        return record


def capture_instrument(*, spec: USEquityCorpusV1, instrument_id: str,
                       provider, transport, store: RawStore,
                       session_index: dict, sequence_start: int) -> dict:
    """Page through one instrument's bars, accepting only legal ones."""
    identity_cursor, page, sequence = "", 0, sequence_start
    seen_cursors: set[str] = set()
    bars, raw_records = [], []

    while True:
        if page >= MAX_PAGES_PER_REQUEST:
            raise CaptureError(
                f"{instrument_id}: stopped after {page} pages, the per-request "
                f"ceiling. Either the range is larger than this corpus "
                "declares or the provider is not advancing.")
        identity = RequestIdentity(
            provider_id=spec.provider_id, instrument_id=instrument_id,
            timeframe=spec.timeframe, adjustment_policy=spec.adjustment_policy,
            start=spec.requested_start, end=spec.requested_end,
            page=page, cursor=identity_cursor)

        symbol = instrument_id.split(":")[-1]
        multiplier, timespan = require_aggregate_window(spec.timeframe)
        path = bars_path(symbol, multiplier=multiplier, timespan=timespan,
                         start=spec.requested_start, end=spec.requested_end)
        params = {
            "adjusted": ADJUSTED_FLAG[spec.adjustment_policy],
            "sort": "asc",
            "limit": MAX_ROWS_PER_PAGE,
        }

        if identity_cursor:
            # A continuation URL, already stripped of anything credential
            # shaped. The transport re-validates the host before following it.
            response = transport.fetch_absolute(identity_cursor,
                                                provider._headers())
        else:
            response = transport.fetch(path, params, provider._headers())
        record = store.write(identity, response.raw, url=response.url,
                             sequence=sequence)
        raw_records.append(record)
        sequence += 1

        rows = adapt_bar_rows(response.payload, spec=spec,
                              instrument_id=instrument_id)
        for row in rows:
            opening = parse_utc(row["bar_open_at"], field_name="bar_open_at")
            session = accept_bar_opening(spec, session_index, opening)
            bars.append(build_canonical_bar(
                spec=spec, instrument_id=instrument_id, session=session,
                bar_open_at=opening, row=row,
                source_raw_hash=record["raw_sha256"],
                source_record_identity=identity.identity_hash))
        if len(bars) > MAX_ROWS_TOTAL:
            raise CaptureError(
                f"{instrument_id}: exceeded {MAX_ROWS_TOTAL} rows; refusing to "
                "continue")

        cursor = next_cursor(response.payload)
        if not cursor:
            break
        if cursor in seen_cursors:
            raise CaptureError(
                f"{instrument_id}: the provider returned pagination cursor "
                f"{cursor!r} twice. Continuing would loop forever or duplicate "
                "rows; stopping instead of producing a corpus that looks whole.")
        seen_cursors.add(cursor)
        identity_cursor = cursor
        page += 1

    ordered = canonical_order(bars)
    keys = [bar.canonical_key for bar in ordered]
    if len(set(keys)) != len(keys):
        raise CaptureError(
            f"{instrument_id}: the same bar opening arrived more than once "
            "with different provenance; refusing to guess which is correct")
    return {"bars": ordered, "raw_records": raw_records, "pages": page + 1,
            "sequence": sequence}


def capture_splits(*, spec: USEquityCorpusV1, instrument_id: str, provider,
                   transport, store: RawStore, sequence: int) -> dict:
    """Split records over the corpus range, for provenance only.

    Never used to recompute a price. The provider already served an adjusted
    series; these records say which events that adjustment reflects.
    """
    identity = RequestIdentity(
        provider_id=spec.provider_id, instrument_id=instrument_id,
        timeframe=CORPORATE_ACTIONS_MARKER,
        adjustment_policy=spec.adjustment_policy,
        start=spec.requested_start, end=spec.requested_end, page=0)
    params = {
        "ticker": instrument_id.split(":")[-1],
        "execution_date.gte": spec.requested_start,
        "execution_date.lte": spec.requested_end,
    }
    response = transport.fetch(SPLITS_PATH, params, provider._headers())
    record = store.write(identity, response.raw, url=response.url,
                         sequence=sequence)
    rows = adapt_split_rows(response.payload, instrument_id=instrument_id)
    records = [
        SplitRecord(instrument_id=instrument_id,
                    effective_date=row["effective_date"],
                    ratio_numerator=row["ratio_numerator"],
                    ratio_denominator=row["ratio_denominator"],
                    provider_id=spec.provider_id,
                    source_raw_hash=record["raw_sha256"])
        for row in rows]
    return {"splits": records, "raw_record": record}


def run_capture(*, spec: USEquityCorpusV1 = CORPUS_SPEC_V1, root=None,
                transport=None, provider=None) -> dict:
    """The whole capture. Network-touching, and the only such function here."""
    from scripts.trading_lab.massive_http_transport import MassiveHTTPTransport
    from scripts.trading_lab.massive_provider import (
        MassiveStocksHistoricalProvider)

    # 1. The credential gate, before anything reaches the network.
    credentials = massive_credentials()
    if not credentials.available():
        raise MissingCredentialStop(
            "Massive market-data credential not configured. Set "
            f"{MASSIVE_API_KEY_ENV} and rerun. Nothing was written and no "
            "corpus was created; this is a clean stop, not a partial capture.")

    layout = CorpusLayout(pathlib.Path(root or CORPUS_ROOT))
    if layout.manifest_path.exists():
        raise CaptureError(
            f"{layout.manifest_path} already exists. A capture writes a corpus "
            "once; rerunning into the same directory would mix two attempts.")
    # Artifacts, not entries. The guard exists to stop two attempts being
    # blended, and an empty directory tree blends nothing -- it is what an
    # aborted run leaves behind, and refusing on it would mean the first
    # rejected key permanently wedges the corpus path.
    leftovers = [path for path in layout.root.rglob("*") if path.is_file()]
    if leftovers:
        raise CaptureError(
            f"{layout.root} already holds {len(leftovers)} file(s) from an "
            "earlier attempt. Start a capture run from a clean directory so "
            "two attempts can never be blended invisibly.")

    transport = transport or MassiveHTTPTransport()
    provider = provider or MassiveStocksHistoricalProvider(
        instruments=spec.instruments, transport=transport,
        credentials=credentials, adjustment_policy=spec.adjustment_policy)

    started = _now()
    print(json.dumps({
        "event": "FIRST_US_EQUITY_CORPUS_CAPTURE_START",
        "capture_commit": _git_commit(),
        "corpus_spec_hash": spec.corpus_spec_hash,
        "provider": spec.provider_id,
        "instruments": list(spec.instruments),
        "requested_range": {"start": spec.requested_start,
                            "end": spec.requested_end},
        "timeframe": spec.timeframe,
        "adjustment_policy": spec.adjustment_policy,
        "started_at": started,
    }, sort_keys=True))

    layout.ensure()
    try:
        return _capture_body(spec=spec, layout=layout, transport=transport,
                             provider=provider, started=started)
    except BaseException:
        # An aborted run must leave the path exactly as it found it. Only a
        # tree that holds no files is removed, so a partial capture is still
        # preserved for diagnosis rather than quietly deleted -- and a run
        # that never got past the first rejected request does not wedge the
        # next attempt.
        _discard_empty_tree(layout)
        raise


def _discard_empty_tree(layout: CorpusLayout) -> None:
    if not layout.root.exists():
        return
    if any(path.is_file() for path in layout.root.rglob("*")):
        return
    shutil.rmtree(layout.root, ignore_errors=True)


def _capture_body(*, spec: USEquityCorpusV1, layout: CorpusLayout, transport,
                  provider, started: str) -> dict:
    store = RawStore(layout)
    session_index = spec.session_index()
    expected_openings = tuple(session_index)

    # 2. Identity gate: prove every venue before storing a byte under it.
    reference = {}
    sequence = 0
    for instrument_id in spec.instruments:
        reference[instrument_id] = verify_instrument_identity(
            provider, instrument_id)

    by_instrument, audits, splits_by_instrument = {}, [], {}
    for instrument_id in spec.instruments:
        captured = capture_instrument(
            spec=spec, instrument_id=instrument_id, provider=provider,
            transport=transport, store=store, session_index=session_index,
            sequence_start=sequence)
        sequence = captured["sequence"]
        bars = captured["bars"]
        by_instrument[instrument_id] = bars

        split_result = capture_splits(
            spec=spec, instrument_id=instrument_id, provider=provider,
            transport=transport, store=store, sequence=sequence)
        sequence += 1
        splits_by_instrument[instrument_id] = split_result["splits"]

        audits.append(audit_gaps(spec, instrument_id, bars,
                                 expected_openings=expected_openings))

    write_corpus(layout=layout, spec=spec, by_instrument=by_instrument,
                 splits_by_instrument=splits_by_instrument)
    manifest = build_manifest(
        layout=layout, spec=spec, by_instrument=by_instrument,
        splits_by_instrument=splits_by_instrument, audits=audits,
        raw_records=store.records, reference=reference,
        transport_stats=getattr(transport, "stats", None),
        started_at=started, completed_at=_now(),
        duplicate_responses=store.duplicate_responses)
    layout.manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def _git_commit():
    import subprocess

    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=15)
        return result.stdout.strip() or None
    except Exception:                                # pragma: no cover
        return None


def write_corpus(*, layout: CorpusLayout, spec: USEquityCorpusV1,
                 by_instrument: dict, splits_by_instrument: dict) -> None:
    """The derived files. Deterministic: same bars in, same bytes out."""
    layout.ensure()
    layout.spec_path.write_text(
        json.dumps(spec.payload(), indent=1, sort_keys=True) + "\n")
    for instrument_id, bars in sorted(by_instrument.items()):
        layout.canonical_path(instrument_id).write_text(
            serialise_rows(canonical_order(bars)))
        records = splits_by_instrument.get(instrument_id, [])
        layout.corporate_actions_path(instrument_id).write_text(
            json.dumps({
                "instrument_id": instrument_id,
                "policy": spec.corporate_action_policy,
                "splits": [record.row() for record in
                           sorted(records, key=lambda item: item.effective_date)],
            }, indent=1, sort_keys=True) + "\n")


def build_manifest(*, layout: CorpusLayout, spec: USEquityCorpusV1,
                   by_instrument: dict, splits_by_instrument: dict, audits,
                   raw_records, reference: dict, transport_stats,
                   started_at: str, completed_at: str,
                   duplicate_responses: int = 0) -> dict:
    """Everything an auditor needs, and nothing that could identify a key."""
    instruments = []
    for instrument_id in sorted(by_instrument):
        bars = canonical_order(by_instrument[instrument_id])
        audit = next(item for item in audits
                     if item.instrument_id == instrument_id)
        canonical_path = layout.canonical_path(instrument_id)
        instruments.append({
            "instrument_id": instrument_id,
            "reference": reference.get(instrument_id, {}),
            "instrument_content_hash": instrument_content_hash(bars),
            "canonical_path": str(canonical_path.relative_to(layout.root)),
            "canonical_sha256": sha256_bytes(canonical_path.read_bytes()),
            "canonical_bytes": canonical_path.stat().st_size,
            "rows": len(bars),
            "first_bar_open_at": iso(bars[0].bar_open_at) if bars else None,
            "last_bar_open_at": iso(bars[-1].bar_open_at) if bars else None,
            "gap_audit": audit.payload(),
            "splits": [record.row()
                       for record in splits_by_instrument.get(instrument_id, [])],
            "corporate_actions_hash": corporate_actions_hash(
                splits_by_instrument.get(instrument_id, [])),
        })

    content = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "corpus_id": spec.corpus_id,
        "corpus_spec": spec.payload(),
        "corpus_spec_hash": spec.corpus_spec_hash,
        "corpus_content_hash": corpus_content_hash(by_instrument),
        "instruments": instruments,
        "expected_bars_per_instrument": len(spec.expected_bar_opens()),
        "sessions": len(spec.sessions()),
        "raw_requests": len(raw_records),
        "raw_files": [
            {key: record[key] for key in
             ("instrument_id", "request_identity", "request_identity_hash",
              "raw_path", "raw_sha256", "raw_bytes", "capture_sequence",
              "source_url")}
            for record in sorted(raw_records,
                                 key=lambda item: item["capture_sequence"])],
        "raw_bytes": sum(record["raw_bytes"] for record in raw_records),
        "duplicate_responses": duplicate_responses,
        "overlapping_missing_intervals": list(
            overlapping_missing_intervals(audits)),
        # State, never value. Whether a key existed is operationally useful;
        # which key it was is not, and has no business in a committed file.
        "credential_present": True,
        "captured": True,
        "verified": False,
        "reproducible": False,
        # The distinction this corpus must never lose. The provider serves
        # historical values, not necessarily the value it would have served on
        # each past date, and split adjustment restates history by definition.
        "point_in_time_exchange_revision_history": False,
        "historical_market_data_corpus": True,
    }
    return {
        "content": content,
        "manifest_content_sha256": sha256_canonical(content),
        "capture_started_at": started_at,
        "capture_completed_at": completed_at,
        "transport": transport_stats.payload() if transport_stats else None,
    }


# --- CLI -------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("capture", "plan"))
    parser.add_argument("--root", default=CORPUS_ROOT)
    arguments = parser.parse_args(argv)

    spec = CORPUS_SPEC_V1
    if arguments.command == "plan":
        # Offline. Says exactly what a capture would ask for, so the plan can
        # be reviewed before any request is made.
        print(json.dumps({
            "corpus_spec_hash": spec.corpus_spec_hash,
            "instruments": list(spec.instruments),
            "sessions": len(spec.sessions()),
            "expected_bars_per_instrument": len(spec.expected_bar_opens()),
            "expected_bars_total":
                len(spec.expected_bar_opens()) * len(spec.instruments),
            "credential_configured": massive_credentials().available(),
        }, indent=1, sort_keys=True))
        return 0

    try:
        manifest = run_capture(spec=spec, root=arguments.root)
    except MissingCredentialStop as error:
        print(json.dumps({
            "captured": False,
            "capture_blocked_missing_credential": True,
            "ready_for_capture_when_credential_available": True,
            "reason": str(error),
        }, indent=1, sort_keys=True), file=sys.stderr)
        return 2
    except (CaptureError, EquityCorpusError, MassiveProviderError,
            CredentialError) as error:
        print(json.dumps({
            "captured": False,
            "reason": f"{type(error).__name__}: {error}",
        }, indent=1, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({
        "captured": True,
        "corpus_spec_hash": manifest["content"]["corpus_spec_hash"],
        "corpus_content_hash": manifest["content"]["corpus_content_hash"],
        "rows": sum(item["rows"] for item in manifest["content"]["instruments"]),
    }, indent=1, sort_keys=True))
    return 0


__all__ = [
    "CaptureError", "CorpusLayout", "MANIFEST_SCHEMA_VERSION",
    "CORPORATE_ACTIONS_MARKER", "MissingCredentialStop",
    "REQUIRED_BAR_FIELDS", "SPLITS_PATH",
    "RawStore", "adapt_bar_rows", "build_manifest",
    "capture_instrument", "capture_splits", "main", "next_cursor",
    "run_capture", "verify_instrument_identity", "write_corpus",
]


if __name__ == "__main__":                           # pragma: no cover
    raise SystemExit(main())
