"""Recompute the US equity corpus from its own files. Never from the network.

The capture is trusted once. After that, this module is what makes the corpus
worth anything: given the raw responses, the frozen spec and the pinned
calendar, it rebuilds every canonical row and every hash and checks they match
what was committed.

There is no import of a socket anywhere below, and a test replaces the
transport with an object that raises on use to prove the verification path
never reaches for one.

Two operations:

* ``verify`` -- recompute canonical rows from raw, re-audit sessions and gaps,
  recompute instrument and corpus content hashes, and check the manifest is
  internally consistent with the files beside it.
* ``rebuild`` -- delete the canonical outputs and regenerate them from raw
  alone, then assert byte-identity. A canonicalisation that is not
  reproducible is a canonicalisation nobody can audit.

Everything fails closed. A hash that does not match, a bar off the session
grid, a row count that drifted, a raw file whose digest has changed: each is a
failure, never a warning, because a corpus that reports a warning is a corpus
someone will use anyway.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from scripts.trading_lab.capture_us_equity_corpus import (
    CorpusLayout, adapt_bar_rows)
from scripts.trading_lab.equity_corpus import (
    CORPUS_ROOT, CORPUS_SPEC_V1, EquityCorpusError, SplitRecord,
    USEquityCorpusV1, accept_bar_opening, audit_gaps, build_canonical_bar,
    canonical_order, corporate_actions_hash, corpus_content_hash,
    instrument_content_hash, iso, overlapping_missing_intervals, parse_utc,
    serialise_rows, sha256_bytes, sha256_canonical)


class CorpusVerificationError(RuntimeError):
    """Raised when a corpus does not reproduce from its own bytes."""


def load_manifest(layout: CorpusLayout) -> dict:
    if not layout.manifest_path.is_file():
        raise CorpusVerificationError(
            f"no manifest at {layout.manifest_path}; there is no corpus here")
    return json.loads(layout.manifest_path.read_text())


def spec_from_manifest(manifest: dict) -> USEquityCorpusV1:
    """Rebuild the spec from what was committed, not from today's defaults.

    Reading the module constants instead would make this verifier agree with
    itself rather than with the corpus: a constant edited after the capture
    would silently redefine what was captured.
    """
    recorded = manifest["content"]["corpus_spec"]
    spec = USEquityCorpusV1(
        corpus_id=recorded["corpus_id"],
        instruments=tuple(recorded["instruments"]),
        provider_id=recorded["provider_id"],
        timeframe=recorded["timeframe"],
        session_type=recorded["session_type"],
        adjustment_policy=recorded["adjustment_policy"],
        calendar_id=recorded["calendar_id"],
        requested_start=recorded["requested_range"]["start"],
        requested_end=recorded["requested_range"]["end"],
        gap_policy=recorded["gap_policy"],
        corporate_action_policy=recorded["corporate_action_policy"],
        capture_protocol_version=recorded["capture_protocol_version"],
        canonical_schema_version=recorded["canonical_schema_version"],
        schema_version=recorded["schema_version"])
    if spec.corpus_spec_hash != manifest["content"]["corpus_spec_hash"]:
        raise CorpusVerificationError(
            "the recorded spec does not hash to the recorded spec hash; the "
            "manifest has been edited or the spec schema has changed")
    # The calendar is part of the spec's identity, so a dependency bump that
    # would move a holiday must be caught here rather than at read time.
    identity = spec.calendar_identity()
    for field in ("calendar_spec_hash", "calendar_dependency_version"):
        if identity[field] != recorded[field]:
            raise CorpusVerificationError(
                f"{field} is {identity[field]!r} in this environment but the "
                f"corpus was captured with {recorded[field]!r}; the expected "
                "bar grid may differ and the gap audit would be meaningless")
    return spec


def rebuild_bars_from_raw(layout: CorpusLayout, spec: USEquityCorpusV1,
                          manifest: dict) -> dict:
    """Canonical rows, recomputed from the raw responses and nothing else."""
    session_index = spec.session_index()
    by_instrument: dict[str, list] = {
        instrument_id: [] for instrument_id in spec.instruments}

    raw_files = sorted(manifest["content"]["raw_files"],
                       key=lambda item: item["capture_sequence"])
    for entry in raw_files:
        path = layout.root / entry["raw_path"]
        if not path.is_file():
            raise CorpusVerificationError(f"missing raw file {entry['raw_path']}")
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        if digest != entry["raw_sha256"]:
            raise CorpusVerificationError(
                f"{entry['raw_path']} has changed since capture: recorded "
                f"{entry['raw_sha256'][:12]}, found {digest[:12]}")
        payload = json.loads(raw.decode("utf-8"))
        instrument_id = entry["instrument_id"]
        if "bars" not in payload:
            continue                                 # a corporate-action file
        rows = adapt_bar_rows(payload, spec=spec, instrument_id=instrument_id)
        for row in rows:
            opening = parse_utc(row["bar_open_at"], field_name="bar_open_at")
            session = accept_bar_opening(spec, session_index, opening)
            by_instrument[instrument_id].append(build_canonical_bar(
                spec=spec, instrument_id=instrument_id, session=session,
                bar_open_at=opening, row=row, source_raw_hash=digest,
                source_record_identity=entry["request_identity_hash"]))
    return {instrument_id: canonical_order(bars)
            for instrument_id, bars in by_instrument.items()}


def verify(root=None) -> dict:
    """Recompute everything. Offline, fail-closed, no tolerance."""
    layout = CorpusLayout(pathlib.Path(root or CORPUS_ROOT))
    manifest = load_manifest(layout)
    content = manifest["content"]
    spec = spec_from_manifest(manifest)

    if sha256_canonical(content) != manifest["manifest_content_sha256"]:
        raise CorpusVerificationError(
            "the manifest content does not match its own recorded digest")

    rebuilt = rebuild_bars_from_raw(layout, spec, manifest)
    expected_openings = tuple(spec.session_index())
    audits, checks = [], {}

    for entry in content["instruments"]:
        instrument_id = entry["instrument_id"]
        bars = rebuilt.get(instrument_id, [])
        if len(bars) != entry["rows"]:
            raise CorpusVerificationError(
                f"{instrument_id}: manifest records {entry['rows']} rows, raw "
                f"rebuilds to {len(bars)}")
        digest = instrument_content_hash(bars)
        if digest != entry["instrument_content_hash"]:
            raise CorpusVerificationError(
                f"{instrument_id}: content hash differs; recorded "
                f"{entry['instrument_content_hash'][:12]}, rebuilt "
                f"{digest[:12]}")

        canonical_path = layout.root / entry["canonical_path"]
        if not canonical_path.is_file():
            raise CorpusVerificationError(
                f"missing canonical file {entry['canonical_path']}")
        on_disk = canonical_path.read_bytes()
        if sha256_bytes(on_disk) != entry["canonical_sha256"]:
            raise CorpusVerificationError(
                f"{instrument_id}: the canonical file has changed since capture")
        if on_disk.decode("utf-8") != serialise_rows(bars):
            raise CorpusVerificationError(
                f"{instrument_id}: the canonical file does not match what the "
                "raw responses rebuild to")

        audit = audit_gaps(spec, instrument_id, bars,
                           expected_openings=expected_openings)
        recorded_audit = entry["gap_audit"]
        for field, actual in (("expected_bars", audit.expected),
                              ("observed_bars", audit.observed),
                              ("missing_expected_bars", len(audit.missing)),
                              ("extra_bars", len(audit.extra)),
                              ("duplicate_bars", len(audit.duplicates))):
            if recorded_audit[field] != actual:
                raise CorpusVerificationError(
                    f"{instrument_id}: gap audit {field} recorded "
                    f"{recorded_audit[field]}, recomputed {actual}")
        if audit.extra:
            raise CorpusVerificationError(
                f"{instrument_id}: {len(audit.extra)} bars sit outside the "
                "expected session grid")
        audits.append(audit)

        splits = [SplitRecord(instrument_id=row["instrument_id"],
                              effective_date=row["effective_date"],
                              ratio_numerator=row["ratio_numerator"],
                              ratio_denominator=row["ratio_denominator"],
                              provider_id=row["provider_id"],
                              source_raw_hash=row["source_raw_hash"])
                  for row in entry["splits"]]
        if corporate_actions_hash(splits) != entry["corporate_actions_hash"]:
            raise CorpusVerificationError(
                f"{instrument_id}: corporate-action provenance hash differs")
        checks[instrument_id] = {"rows": len(bars),
                                 "missing": len(audit.missing),
                                 "splits": len(splits)}

    if corpus_content_hash(rebuilt) != content["corpus_content_hash"]:
        raise CorpusVerificationError(
            "the corpus content hash differs from the recorded one")
    recorded_overlap = list(content.get("overlapping_missing_intervals", []))
    if list(overlapping_missing_intervals(audits)) != recorded_overlap:
        raise CorpusVerificationError(
            "the overlapping missing intervals differ from the recorded set")

    _assert_no_credential(layout)
    _assert_equities_only(spec, content)

    return {
        "ok": True,
        "corpus_spec_hash": content["corpus_spec_hash"],
        "corpus_content_hash": content["corpus_content_hash"],
        "instruments": checks,
        "raw_files": len(content["raw_files"]),
        "offline": True,
    }


def _assert_no_credential(layout: CorpusLayout) -> None:
    """No committed file may contain anything key-shaped.

    A blunt scan over the corpus rather than a check of one field: the leak
    this guards against is a credential that reached a file nobody thought to
    inspect, so inspecting only the fields we remember would miss it.
    """
    markers = ("authorization", "bearer ", "api_key", "apikey",
               "hyprl_massive_api_key")
    for path in sorted(layout.root.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore").lower()
        for marker in markers:
            if marker in text:
                raise CorpusVerificationError(
                    f"{path.relative_to(layout.root)} contains {marker!r}; a "
                    "credential must never reach a committed corpus file")


def _assert_equities_only(spec: USEquityCorpusV1, content: dict) -> None:
    """This corpus is US equities. Nothing crypto may have leaked into it.

    The protected BTC/ETH window is enforced elsewhere; this is the cheap
    structural check that the equity transport never routed a crypto request
    into equity storage.
    """
    forbidden = ("BTC", "ETH", "coinbase")
    names = [spec.corpus_id, *spec.instruments,
             *(entry["instrument_id"] for entry in content["instruments"])]
    for name in names:
        for marker in forbidden:
            if marker.lower() in str(name).lower():
                raise CorpusVerificationError(
                    f"{name!r} looks like a crypto identity in a US equity "
                    "corpus; refusing to certify cross-routed data")


def rebuild(root=None) -> dict:
    """Delete the canonical files, regenerate from raw, assert byte-identity."""
    layout = CorpusLayout(pathlib.Path(root or CORPUS_ROOT))
    manifest = load_manifest(layout)
    spec = spec_from_manifest(manifest)

    before = {}
    for entry in manifest["content"]["instruments"]:
        path = layout.root / entry["canonical_path"]
        before[entry["instrument_id"]] = path.read_bytes()
        path.unlink()

    rebuilt = rebuild_bars_from_raw(layout, spec, manifest)
    for instrument_id, bars in rebuilt.items():
        layout.canonical_path(instrument_id).write_text(
            serialise_rows(canonical_order(bars)))

    differences = []
    for entry in manifest["content"]["instruments"]:
        instrument_id = entry["instrument_id"]
        after = (layout.root / entry["canonical_path"]).read_bytes()
        if after != before[instrument_id]:
            differences.append(instrument_id)
    if differences:
        raise CorpusVerificationError(
            f"rebuilding from raw produced different bytes for {differences}; "
            "the canonicalisation is not reproducible")
    return {"ok": True, "rebuilt": sorted(rebuilt), "byte_identical": True}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("verify", "rebuild"))
    parser.add_argument("--root", default=CORPUS_ROOT)
    arguments = parser.parse_args(argv)
    try:
        report = (verify(arguments.root) if arguments.command == "verify"
                  else rebuild(arguments.root))
    except (CorpusVerificationError, EquityCorpusError) as error:
        print(json.dumps({"ok": False, "reason": str(error)}, indent=1),
              file=sys.stderr)
        return 1
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


__all__ = [
    "CorpusVerificationError", "load_manifest", "main", "rebuild",
    "rebuild_bars_from_raw", "spec_from_manifest", "verify",
]


if __name__ == "__main__":                           # pragma: no cover
    raise SystemExit(main())
