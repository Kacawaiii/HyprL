"""Capture the Yahoo daily equity corpus once, locally, then freeze it.

A separate runner from the Massive one on purpose. The two sources page
differently, shape their responses differently and carry different adjustment
semantics; teaching one runner both would produce a function whose every
branch is a guess about which vendor it is talking to. What they *do* share --
the corpus spec, the calendar, the canonical bar, the gap audit, the hashes --
is imported rather than re-implemented, because that is where a divergence
would actually change a number.

Three commands, deliberately separated:

* ``capture`` is the only one allowed to touch the network. It writes each raw
  response byte-for-byte before anything parses it, derives canonical bars,
  audits them against the frozen calendar, and writes a local manifest.
* ``verify`` recomputes every row and hash from the stored files alone.
* ``rebuild`` regenerates the canonical files from raw and asserts byte
  equality.

**The corpus does not live in the repository.** The source declares
``redistribution_permitted=false``, so committing its prices would publish a
dataset nobody granted us the right to publish. It is written under the
gitignored, release-excluded local research root instead, and what gets
committed is a fingerprint: hashes and counts that prove *which* corpus was
frozen and are useless for reconstructing it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from datetime import datetime, timezone

from scripts.trading_lab.equity_corpus import (
    CORPUS_SPEC_V2, EquityCorpusError, RequestIdentity, SplitRecord,
    USEquityCorpusV2, accept_bar_opening, audit_gaps, build_canonical_bar,
    canonical_order, corporate_actions_hash, corpus_content_hash,
    instrument_content_hash, iso, overlapping_missing_intervals,
    serialise_rows, sha256_bytes, sha256_canonical)
from scripts.trading_lab.instrument_registry import CATALOGUE_V1
from scripts.trading_lab.yahoo_chart_provider import (
    YAHOO_CHART_DAILY_V1, YahooChartDailyProvider, YahooProviderError,
    adapt_chart_rows, adapt_split_events, chart_path, parse_chart_meta)

MANIFEST_SCHEMA_VERSION = "trading-lab.yahoo-equity-corpus-manifest.v1"

# Gitignored (`var/` in .gitignore) and release-excluded (`var` in the release
# builder's EXCLUDED_DIRECTORIES). Both were already true before this runner
# existed, which is why this path was chosen rather than a new one.
LOCAL_CORPUS_ROOT = "var/trading_lab/research/yahoo_us_equity_daily_v2"

# What the tracked fingerprint is allowed to say. Everything here is a hash, a
# count or an identity -- nothing from which a price series can be rebuilt.
FINGERPRINT_SCHEMA_VERSION = "trading-lab.yahoo-equity-fingerprint.v1"


class CaptureError(RuntimeError):
    """Raised when a capture cannot proceed or cannot be trusted."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(instrument_id: str) -> str:
    """Keeps the venue, because the venue is part of the identity."""
    return instrument_id.replace(":", "_")


class LocalCorpusLayout:
    """Where the local corpus lives. One place, so verify and rebuild agree."""

    def __init__(self, root):
        self.root = pathlib.Path(root)

    @property
    def manifest_path(self) -> pathlib.Path:
        return self.root / "manifest.local.json"

    @property
    def spec_path(self) -> pathlib.Path:
        return self.root / "spec.json"

    def raw_path(self, instrument_id: str) -> pathlib.Path:
        return self.root / "raw" / f"{_slug(instrument_id)}.json"

    def canonical_path(self, instrument_id: str) -> pathlib.Path:
        return self.root / "canonical" / f"{_slug(instrument_id)}.jsonl"

    def corporate_actions_path(self, instrument_id: str) -> pathlib.Path:
        return self.root / "corporate_actions" / f"{_slug(instrument_id)}.json"

    def ensure(self) -> "LocalCorpusLayout":
        for child in ("raw", "canonical", "corporate_actions"):
            (self.root / child).mkdir(parents=True, exist_ok=True)
        return self


def request_identity(spec: USEquityCorpusV2, instrument_id: str) -> RequestIdentity:
    """What this request asks for, deterministically.

    Excludes the wall clock, the attempt id and the temporary directory. Two
    runs a week apart asking the same question produce the same identity,
    which is what makes a re-capture comparable and a duplicate detectable.
    """
    return RequestIdentity(
        provider_id=spec.provider_id, instrument_id=instrument_id,
        timeframe=spec.timeframe, adjustment_policy=spec.adjustment_policy,
        start=spec.requested_start, end=spec.requested_end, page=0)


def persist_raw(layout: LocalCorpusLayout, instrument_id: str,
                raw: bytes) -> dict:
    """Raw bytes to disk, durably, BEFORE anything parses them.

    The order is the whole invariant. A canonicalisation bug is recoverable
    only if the source survived it, and a source written after the fact is a
    source that never existed for the run that failed.
    """
    path = layout.raw_path(instrument_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CaptureError(
            f"refusing to overwrite an existing raw artifact at {path}")
    with open(path, "wb") as handle:
        handle.write(raw)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    return {"raw_path": str(path.relative_to(layout.root)),
            "raw_sha256": sha256_bytes(raw), "raw_bytes": len(raw)}


def build_instrument(*, spec: USEquityCorpusV2, instrument_id: str,
                     payload: dict, raw_digest: str,
                     identity_hash: str, session_index: dict) -> dict:
    """Everything derived from one stored response. No network, ever."""
    registered = CATALOGUE_V1.resolve(instrument_id)

    # Identity and timezone before a single price is read.
    meta = parse_chart_meta(payload)
    rows = adapt_chart_rows(payload, instrument_id=instrument_id)
    source_rows = len(payload["chart"]["result"][0].get("timestamp") or [])
    null_rows = source_rows - len(rows)

    bars, off_grid, outside = [], [], []
    for row in rows:
        opening = row["bar_open_at"]
        try:
            session = accept_bar_opening(spec, session_index, opening)
        except EquityCorpusError:
            # Recorded, never snapped to a nearby session and never dropped
            # silently: an unexplained row is a finding, not a rounding.
            off_grid.append(iso(opening))
            continue
        bars.append(build_canonical_bar(
            spec=spec, instrument_id=registered.canonical_id, session=session,
            bar_open_at=opening,
            row={name: row[name] for name in
                 ("open", "high", "low", "close", "volume")},
            source_raw_hash=raw_digest, source_record_identity=identity_hash))

    ordered = canonical_order(bars)

    # The explicit duplicate guard, before publication. Not delegated to the
    # gap audit: a duplicate must stop the capture, not merely be counted, and
    # "keep first" or "keep last" would both be a silent choice about which
    # price is real.
    keys = [bar.canonical_key for bar in ordered]
    if len(set(keys)) != len(keys):
        seen, dupes = set(), []
        for key in keys:
            if key in seen:
                dupes.append(key[1])
            seen.add(key)
        raise CaptureError(
            f"{instrument_id}: the same session arrived more than once "
            f"({sorted(set(dupes))[:5]}); refusing to guess which row is the "
            "real one")

    splits = [
        SplitRecord(instrument_id=registered.canonical_id,
                    effective_date=event["effective_date"],
                    ratio_numerator=event["ratio_numerator"],
                    ratio_denominator=event["ratio_denominator"],
                    provider_id=spec.provider_id, source_raw_hash=raw_digest)
        for event in adapt_split_events(payload, instrument_id=instrument_id)]
    split_keys = [(record.instrument_id, record.effective_date)
                  for record in splits]
    if len(set(split_keys)) != len(split_keys):
        raise CaptureError(
            f"{instrument_id}: two split events share an effective date; "
            "refusing to guess which is the real corporate action")

    return {"bars": ordered, "splits": splits, "meta": meta,
            "source_rows": source_rows, "null_rows": null_rows,
            "off_grid": off_grid, "outside_range": outside}


def capture(*, spec: USEquityCorpusV2 = CORPUS_SPEC_V2, root=None,
            transport=None, provider=None) -> dict:
    """The whole capture. The only network-touching function here."""
    from scripts.trading_lab.yahoo_http_transport import YahooHTTPTransport

    layout = LocalCorpusLayout(pathlib.Path(root or LOCAL_CORPUS_ROOT))
    if layout.manifest_path.exists():
        raise CaptureError(
            f"{layout.manifest_path} already exists. A capture writes a corpus "
            "once; rerunning into a frozen one would mix two attempts.")
    leftovers = [p for p in layout.root.rglob("*") if p.is_file()] \
        if layout.root.exists() else []
    if leftovers:
        raise CaptureError(
            f"{layout.root} already holds {len(leftovers)} file(s) from an "
            "earlier attempt; start from a clean directory")

    transport = transport or YahooHTTPTransport()
    provider = provider or YahooChartDailyProvider(
        instruments=spec.instruments, transport=transport)

    started = _now()
    print(json.dumps({
        "event": "YAHOO_V2_LOCAL_CORPUS_CAPTURE_START",
        "capture_commit": _git_commit(),
        "corpus_spec_hash": spec.corpus_spec_hash,
        "provider": spec.provider_id, "timeframe": spec.timeframe,
        "adjustment_policy": spec.adjustment_policy,
        "instruments": list(spec.instruments),
        "requested_range": {"start": spec.requested_start,
                            "end": spec.requested_end},
        "credential_required": False,
        "redistribution_permitted": False,
        "started_at": started,
    }, sort_keys=True))

    layout.ensure()
    try:
        return _capture_body(spec=spec, layout=layout, transport=transport,
                             provider=provider, started=started)
    except BaseException:
        _discard_empty_tree(layout)
        raise


def _discard_empty_tree(layout: LocalCorpusLayout) -> None:
    """An aborted run leaves the path as it found it -- unless it wrote bytes.

    Raw responses from a failed attempt are the only record of what the source
    actually said, so they are kept for diagnosis rather than deleted.
    """
    if not layout.root.exists():
        return
    if any(p.is_file() for p in layout.root.rglob("*")):
        return
    shutil.rmtree(layout.root, ignore_errors=True)


def _capture_body(*, spec, layout, transport, provider, started) -> dict:
    session_index = spec.session_index()
    expected_openings = tuple(session_index)
    raw_records, built = {}, {}

    for instrument_id in spec.instruments:
        symbol = instrument_id.split(":")[-1]
        params = provider.chart_params(timeframe=spec.timeframe,
                                       start=spec.requested_start,
                                       end=spec.requested_end)
        path = provider._require_allowed(chart_path(symbol))
        response = transport.fetch(path, params, provider._headers())

        # 1. raw first, durable, before anything parses it.
        identity = request_identity(spec, instrument_id)
        record = persist_raw(layout, instrument_id, response.raw)
        record.update({"instrument_id": instrument_id,
                       "request_identity": identity.canonical(),
                       "request_identity_hash": identity.identity_hash,
                       "source_url": response.url})
        raw_records[instrument_id] = record

        # 2. only now interpret it.
        built[instrument_id] = build_instrument(
            spec=spec, instrument_id=instrument_id, payload=response.payload,
            raw_digest=record["raw_sha256"],
            identity_hash=identity.identity_hash, session_index=session_index)

    return _finalise(spec=spec, layout=layout, built=built,
                     raw_records=raw_records, expected_openings=expected_openings,
                     transport=transport, started=started)


def _finalise(*, spec, layout, built, raw_records, expected_openings,
              transport, started) -> dict:
    audits = {instrument_id: audit_gaps(spec, instrument_id,
                                        data["bars"],
                                        expected_openings=expected_openings)
              for instrument_id, data in built.items()}

    # Atomic: one instrument failing its audit stops the whole corpus. Three
    # good instruments and one broken one is not three quarters of a corpus.
    for instrument_id, audit in audits.items():
        if audit.extra:
            raise CaptureError(
                f"{instrument_id}: {len(audit.extra)} bars sit outside the "
                "expected session grid")
        if audit.duplicates:
            raise CaptureError(
                f"{instrument_id}: {len(audit.duplicates)} duplicate sessions")

    write_corpus(layout=layout, spec=spec, built=built)
    manifest = build_manifest(
        layout=layout, spec=spec, built=built, audits=audits,
        raw_records=raw_records, transport=transport, started_at=started,
        completed_at=_now())
    layout.manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def write_corpus(*, layout, spec, built) -> None:
    """Deterministic: same bars in, same bytes out."""
    layout.ensure()
    layout.spec_path.write_text(
        json.dumps(spec.payload(), indent=1, sort_keys=True) + "\n")
    for instrument_id, data in sorted(built.items()):
        layout.canonical_path(instrument_id).write_text(
            serialise_rows(canonical_order(data["bars"])))
        layout.corporate_actions_path(instrument_id).write_text(
            json.dumps({
                "instrument_id": instrument_id,
                "policy": spec.corporate_action_policy,
                "splits": [record.row() for record in
                           sorted(data["splits"],
                                  key=lambda item: item.effective_date)],
            }, indent=1, sort_keys=True) + "\n")


def build_manifest(*, layout, spec, built, audits, raw_records, transport,
                   started_at, completed_at) -> dict:
    instruments = []
    for instrument_id in sorted(built):
        data = built[instrument_id]
        bars = canonical_order(data["bars"])
        audit = audits[instrument_id]
        canonical = layout.canonical_path(instrument_id)
        instruments.append({
            "instrument_id": instrument_id,
            "expected_sessions": audit.expected,
            "source_rows": data["source_rows"],
            "canonical_rows": len(bars),
            "null_rows": data["null_rows"],
            "missing_sessions": len(audit.missing),
            "duplicate_sessions": len(audit.duplicates),
            "off_grid_rows": len(data["off_grid"]),
            "outside_range_rows": len(data["outside_range"]),
            "timezone_verified": data["meta"].exchange_timezone,
            "source_symbol": data["meta"].symbol,
            "first_bar_open_at": iso(bars[0].bar_open_at) if bars else None,
            "last_bar_open_at": iso(bars[-1].bar_open_at) if bars else None,
            "split_count": len(data["splits"]),
            "instrument_content_hash": instrument_content_hash(bars),
            "canonical_sha256": sha256_bytes(canonical.read_bytes()),
            "corporate_actions_hash": corporate_actions_hash(data["splits"]),
            "raw_sha256": raw_records[instrument_id]["raw_sha256"],
            "raw_bytes": raw_records[instrument_id]["raw_bytes"],
            "missing": list(audit.missing),
        })

    content = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "corpus_id": spec.corpus_id,
        "corpus_spec": spec.payload(),
        "corpus_spec_hash": spec.corpus_spec_hash,
        "corpus_content_hash": corpus_content_hash(
            {k: v["bars"] for k, v in built.items()}),
        "instruments": instruments,
        "expected_sessions_per_instrument": len(spec.expected_bar_opens()),
        "raw_files": [
            {key: record[key] for key in
             ("instrument_id", "request_identity", "request_identity_hash",
              "raw_path", "raw_sha256", "raw_bytes", "source_url")}
            for record in sorted(raw_records.values(),
                                 key=lambda item: item["instrument_id"])],
        "overlapping_missing_sessions": list(
            overlapping_missing_intervals(list(audits.values()))),
        "capture_code_commit": _git_commit(),
        "credential_required": False,
        "captured": True,
        "verified": False,
        "reproducible": False,
        # The source's own terms. Recorded in the corpus so that anything
        # packaging it can refuse without having to know which provider it is.
        "redistribution_permitted": False,
        "official_contract": False,
        # RAW is not point-in-time. The source returns today's view of
        # history; a revision made since would be invisible here.
        "point_in_time_exchange_revision_history": False,
        "historical_market_data_corpus": True,
    }
    return {
        "content": content,
        "manifest_content_sha256": sha256_canonical(content),
        # Attempt metadata, deliberately outside the hashed content so a
        # re-capture of identical bytes produces an identical identity.
        "capture_started_at": started_at,
        "capture_completed_at": completed_at,
        "transport": transport.payload() if hasattr(transport, "payload") else None,
    }


def build_fingerprint(manifest: dict) -> dict:
    """The tracked proof of which corpus was frozen.

    Hashes, counts, dates and identities. Deliberately insufficient to
    reconstruct a single price: no OHLC, no timestamps beyond first/last, no
    raw payload, no event values. It says *which* corpus exists locally, not
    what is in it.
    """
    content = manifest["content"]
    return {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "corpus_id": content["corpus_id"],
        "provider_id": content["corpus_spec"]["provider_id"],
        "corpus_spec_hash": content["corpus_spec_hash"],
        "calendar_spec_hash": content["corpus_spec"]["calendar_spec_hash"],
        "calendar_dependency_version":
            content["corpus_spec"]["calendar_dependency_version"],
        "timeframe": content["corpus_spec"]["timeframe"],
        "session_type": content["corpus_spec"]["session_type"],
        "adjustment_policy": content["corpus_spec"]["adjustment_policy"],
        "requested_range": content["corpus_spec"]["requested_range"],
        "instruments": [
            {"instrument_id": item["instrument_id"],
             "expected_sessions": item["expected_sessions"],
             "canonical_rows": item["canonical_rows"],
             "missing_sessions": item["missing_sessions"],
             "duplicate_sessions": item["duplicate_sessions"],
             "off_grid_rows": item["off_grid_rows"],
             "null_rows": item["null_rows"],
             "split_count": item["split_count"],
             "first_bar_open_at": item["first_bar_open_at"],
             "last_bar_open_at": item["last_bar_open_at"],
             "instrument_content_hash": item["instrument_content_hash"],
             "canonical_sha256": item["canonical_sha256"],
             "corporate_actions_hash": item["corporate_actions_hash"],
             "raw_sha256": item["raw_sha256"]}
            for item in content["instruments"]],
        "corpus_content_hash": content["corpus_content_hash"],
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "capture_code_commit": content["capture_code_commit"],
        "local_storage_root": LOCAL_CORPUS_ROOT,
        "source_data_committed": False,
        "redistribution_permitted": False,
        "official_contract": False,
        "point_in_time_exchange_revision_history": False,
        "note": ("Fingerprint only. The Yahoo source data is not redistributed "
                 "and is not present in this repository; it lives in the "
                 "gitignored local research store named above."),
    }


def _git_commit():
    import subprocess
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=15)
        return result.stdout.strip() or None
    except Exception:                                # pragma: no cover
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("capture", "plan", "fingerprint"))
    parser.add_argument("--root", default=LOCAL_CORPUS_ROOT)
    arguments = parser.parse_args(argv)
    spec = CORPUS_SPEC_V2

    if arguments.command == "plan":
        print(json.dumps({
            "corpus_spec_hash": spec.corpus_spec_hash,
            "provider": spec.provider_id, "timeframe": spec.timeframe,
            "adjustment_policy": spec.adjustment_policy,
            "instruments": list(spec.instruments),
            "sessions": len(spec.sessions()),
            "expected_rows_per_instrument": len(spec.expected_bar_opens()),
            "expected_rows_total":
                len(spec.expected_bar_opens()) * len(spec.instruments),
            "local_root": arguments.root,
            "credential_required": False,
        }, indent=1, sort_keys=True))
        return 0

    if arguments.command == "fingerprint":
        layout = LocalCorpusLayout(pathlib.Path(arguments.root))
        manifest = json.loads(layout.manifest_path.read_text())
        print(json.dumps(build_fingerprint(manifest), indent=1, sort_keys=True))
        return 0

    try:
        manifest = capture(spec=spec, root=arguments.root)
    except (CaptureError, EquityCorpusError, YahooProviderError) as error:
        print(json.dumps({"captured": False,
                          "reason": f"{type(error).__name__}: {error}"},
                         indent=1, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({
        "captured": True,
        "corpus_spec_hash": manifest["content"]["corpus_spec_hash"],
        "corpus_content_hash": manifest["content"]["corpus_content_hash"],
        "rows": sum(item["canonical_rows"]
                    for item in manifest["content"]["instruments"]),
    }, indent=1, sort_keys=True))
    return 0


__all__ = [
    "CaptureError", "FINGERPRINT_SCHEMA_VERSION", "LOCAL_CORPUS_ROOT",
    "LocalCorpusLayout", "MANIFEST_SCHEMA_VERSION", "build_fingerprint",
    "build_instrument", "build_manifest", "capture", "main", "persist_raw",
    "request_identity", "write_corpus",
]


if __name__ == "__main__":                           # pragma: no cover
    raise SystemExit(main())
