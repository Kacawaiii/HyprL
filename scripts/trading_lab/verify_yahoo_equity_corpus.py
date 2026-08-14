"""Recompute the local Yahoo corpus from its own files. Never from the network.

The capture is trusted once. After that this is what makes the corpus worth
anything: given the stored raw responses, the frozen spec and the pinned
calendar, it rebuilds every canonical row and every hash and checks they match
what was frozen.

Nothing here imports a socket, and a test replaces ``urlopen`` with something
that raises to prove it never reaches for one.

Three levels, deliberately not the same code path three times:

* ``verify``    -- rebuild from raw and compare against the manifest.
* ``rebuild``   -- delete the canonical files, regenerate, assert byte equality.
* ``recount``   -- an independent recount of sessions, rows, events and hashes
                   that walks the stored files directly rather than calling the
                   verifier again. Running one function twice proves it is
                   deterministic, not that it is right.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from scripts.trading_lab.capture_yahoo_equity_corpus import (
    LOCAL_CORPUS_ROOT, LocalCorpusLayout, build_instrument, request_identity)
from scripts.trading_lab.equity_corpus import (
    EquityCorpusError, USEquityCorpusV2, audit_gaps, canonical_order,
    corporate_actions_hash, corpus_content_hash, instrument_content_hash,
    serialise_rows, sha256_bytes, sha256_canonical)


class CorpusVerificationError(RuntimeError):
    """Raised when the local corpus does not reproduce from its own bytes."""


def load_manifest(layout: LocalCorpusLayout) -> dict:
    if not layout.manifest_path.is_file():
        raise CorpusVerificationError(
            f"no manifest at {layout.manifest_path}; there is no corpus here")
    return json.loads(layout.manifest_path.read_text())


def spec_from_manifest(manifest: dict) -> USEquityCorpusV2:
    """Rebuild the spec from what was frozen, not from today's constants.

    Reading the module defaults would make this verifier agree with itself
    rather than with the corpus: a constant edited after the capture would
    silently redefine what was captured.
    """
    recorded = manifest["content"]["corpus_spec"]
    spec = USEquityCorpusV2(
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
            "the recorded spec does not hash to the recorded spec hash")
    identity = spec.calendar_identity()
    for field in ("calendar_spec_hash", "calendar_dependency_version"):
        if identity[field] != recorded[field]:
            raise CorpusVerificationError(
                f"{field} is {identity[field]!r} here but the corpus was "
                f"captured with {recorded[field]!r}; the expected session grid "
                "may differ and the gap audit would be meaningless")
    return spec


def rebuild_from_raw(layout: LocalCorpusLayout, spec: USEquityCorpusV2,
                     manifest: dict) -> dict:
    """Canonical rows, recomputed from the stored responses and nothing else."""
    session_index = spec.session_index()
    built = {}
    for entry in manifest["content"]["raw_files"]:
        instrument_id = entry["instrument_id"]
        path = layout.root / entry["raw_path"]
        if not path.is_file():
            raise CorpusVerificationError(f"missing raw file {entry['raw_path']}")
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        if digest != entry["raw_sha256"]:
            raise CorpusVerificationError(
                f"{entry['raw_path']} has changed since capture: recorded "
                f"{entry['raw_sha256'][:12]}, found {digest[:12]}")
        expected_identity = request_identity(spec, instrument_id)
        if expected_identity.identity_hash != entry["request_identity_hash"]:
            raise CorpusVerificationError(
                f"{instrument_id}: the recorded request identity does not match "
                "the one this spec would produce")
        built[instrument_id] = build_instrument(
            spec=spec, instrument_id=instrument_id,
            payload=json.loads(raw.decode("utf-8")), raw_digest=digest,
            identity_hash=entry["request_identity_hash"],
            session_index=session_index)
    return built


def verify(root=None) -> dict:
    """Recompute everything. Offline, fail-closed, no tolerance."""
    layout = LocalCorpusLayout(pathlib.Path(root or LOCAL_CORPUS_ROOT))
    manifest = load_manifest(layout)
    content = manifest["content"]
    spec = spec_from_manifest(manifest)

    if sha256_canonical(content) != manifest["manifest_content_sha256"]:
        raise CorpusVerificationError(
            "the manifest content does not match its own recorded digest")
    if content.get("redistribution_permitted") is not False:
        raise CorpusVerificationError(
            "this corpus must record redistribution_permitted=false")

    built = rebuild_from_raw(layout, spec, manifest)
    expected_openings = tuple(spec.session_index())
    checks = {}

    for entry in content["instruments"]:
        instrument_id = entry["instrument_id"]
        data = built.get(instrument_id)
        if data is None:
            raise CorpusVerificationError(f"{instrument_id}: no raw artifact")
        bars = canonical_order(data["bars"])
        if len(bars) != entry["canonical_rows"]:
            raise CorpusVerificationError(
                f"{instrument_id}: manifest records {entry['canonical_rows']} "
                f"rows, raw rebuilds to {len(bars)}")
        digest = instrument_content_hash(bars)
        if digest != entry["instrument_content_hash"]:
            raise CorpusVerificationError(
                f"{instrument_id}: content hash differs")

        path = layout.canonical_path(instrument_id)
        if not path.is_file():
            raise CorpusVerificationError(f"missing canonical file for {instrument_id}")
        on_disk = path.read_bytes()
        if sha256_bytes(on_disk) != entry["canonical_sha256"]:
            raise CorpusVerificationError(
                f"{instrument_id}: the canonical file has changed since capture")
        if on_disk.decode("utf-8") != serialise_rows(bars):
            raise CorpusVerificationError(
                f"{instrument_id}: the canonical file does not match what the "
                "raw response rebuilds to")

        audit = audit_gaps(spec, instrument_id, bars,
                           expected_openings=expected_openings)
        for field, actual in (("expected_sessions", audit.expected),
                              ("canonical_rows", audit.observed),
                              ("missing_sessions", len(audit.missing)),
                              ("duplicate_sessions", len(audit.duplicates))):
            if entry[field] != actual:
                raise CorpusVerificationError(
                    f"{instrument_id}: {field} recorded {entry[field]}, "
                    f"recomputed {actual}")
        if audit.extra or audit.duplicates:
            raise CorpusVerificationError(
                f"{instrument_id}: off-grid or duplicate sessions present")
        if corporate_actions_hash(data["splits"]) != entry["corporate_actions_hash"]:
            raise CorpusVerificationError(
                f"{instrument_id}: corporate-action provenance hash differs")
        checks[instrument_id] = {"rows": len(bars),
                                 "missing": len(audit.missing),
                                 "splits": len(data["splits"])}

    if corpus_content_hash({k: v["bars"] for k, v in built.items()}) \
            != content["corpus_content_hash"]:
        raise CorpusVerificationError("the corpus content hash differs")

    _assert_no_credential(layout)
    _assert_equities_only(spec)
    return {"ok": True, "offline": True,
            "corpus_spec_hash": content["corpus_spec_hash"],
            "corpus_content_hash": content["corpus_content_hash"],
            "instruments": checks}


def _assert_no_credential(layout: LocalCorpusLayout) -> None:
    """No stored file may contain anything key-shaped.

    A blunt scan rather than a check of named fields: the leak this guards
    against is a credential reaching a file nobody thought to inspect.
    """
    markers = ("authorization", "bearer ", "api_key", "apikey", "crumb",
               "hyprl_massive_api_key", "set-cookie")
    for path in sorted(layout.root.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore").lower()
        for marker in markers:
            if marker in text:
                raise CorpusVerificationError(
                    f"{path.relative_to(layout.root)} contains {marker!r}")


def _assert_equities_only(spec: USEquityCorpusV2) -> None:
    for name in (spec.corpus_id, *spec.instruments):
        for marker in ("BTC", "ETH", "coinbase"):
            if marker.lower() in str(name).lower():
                raise CorpusVerificationError(
                    f"{name!r} looks like a crypto identity in a US equity corpus")


def rebuild(root=None) -> dict:
    """Delete the canonical files, regenerate from raw, assert byte-identity."""
    layout = LocalCorpusLayout(pathlib.Path(root or LOCAL_CORPUS_ROOT))
    manifest = load_manifest(layout)
    spec = spec_from_manifest(manifest)

    before = {}
    for entry in manifest["content"]["instruments"]:
        path = layout.canonical_path(entry["instrument_id"])
        before[entry["instrument_id"]] = path.read_bytes()
        path.unlink()

    built = rebuild_from_raw(layout, spec, manifest)
    for instrument_id, data in built.items():
        layout.canonical_path(instrument_id).write_text(
            serialise_rows(canonical_order(data["bars"])))

    differences = [instrument_id for instrument_id, original in before.items()
                   if layout.canonical_path(instrument_id).read_bytes() != original]
    if differences:
        raise CorpusVerificationError(
            f"rebuilding from raw produced different bytes for {differences}")
    return {"ok": True, "byte_identical": True, "rebuilt": sorted(built)}


def recount(root=None) -> dict:
    """An independent recount that walks the stored files directly.

    Deliberately does NOT call verify(). Running the same function twice
    demonstrates determinism, not correctness -- a shared bug would agree with
    itself perfectly. This re-derives the session count from the calendar, the
    row counts by reading the canonical files line by line, and the hashes
    from those lines, then checks them against the manifest.
    """
    import hashlib

    layout = LocalCorpusLayout(pathlib.Path(root or LOCAL_CORPUS_ROOT))
    manifest = load_manifest(layout)
    spec = spec_from_manifest(manifest)

    calendar = spec.calendar()
    sessions = calendar.sessions_between(
        f"{spec.requested_start}T00:00:00Z", f"{spec.requested_end}T23:59:59Z")
    expected = len(sessions)

    totals = {"expected_sessions": expected, "instruments": {}}
    for entry in manifest["content"]["instruments"]:
        instrument_id = entry["instrument_id"]
        lines = layout.canonical_path(instrument_id).read_text().splitlines()
        rows = [json.loads(line) for line in lines]
        if entry["expected_sessions"] != expected:
            raise CorpusVerificationError(
                f"{instrument_id}: manifest expected {entry['expected_sessions']} "
                f"sessions, the calendar independently gives {expected}")
        if len(rows) != entry["canonical_rows"]:
            raise CorpusVerificationError(
                f"{instrument_id}: counted {len(rows)} rows, manifest says "
                f"{entry['canonical_rows']}")
        # Recompute the file digest from the bytes rather than trusting the
        # canonical serialiser used everywhere else.
        digest = hashlib.sha256(
            layout.canonical_path(instrument_id).read_bytes()).hexdigest()
        if digest != entry["canonical_sha256"]:
            raise CorpusVerificationError(
                f"{instrument_id}: independent file digest differs")
        # Every row must name this instrument -- the identity is in the hash
        # because it is in the row, and this checks the row rather than the hash.
        wrong = [row for row in rows if row["instrument_id"] != instrument_id]
        if wrong:
            raise CorpusVerificationError(
                f"{instrument_id}: {len(wrong)} rows name another instrument")
        openings = [row["bar_open_at"] for row in rows]
        if len(set(openings)) != len(openings):
            raise CorpusVerificationError(f"{instrument_id}: duplicate openings")
        if openings != sorted(openings):
            raise CorpusVerificationError(f"{instrument_id}: rows are not ordered")
        actions = json.loads(
            layout.corporate_actions_path(instrument_id).read_text())
        if len(actions["splits"]) != entry["split_count"]:
            raise CorpusVerificationError(
                f"{instrument_id}: split count differs")
        totals["instruments"][instrument_id] = {
            "rows": len(rows), "splits": len(actions["splits"])}

    totals["canonical_rows"] = sum(
        item["rows"] for item in totals["instruments"].values())
    totals["ok"] = True
    return totals


def promote(root=None) -> dict:
    """Mark the corpus verified and reproducible -- only after earning it.

    Runs all three passes first. The flags are claims, and a claim written by
    a function that did not check is worse than no claim: it would survive
    into the fingerprint and into every report downstream.

    The corpus content hash and every instrument hash are untouched by this;
    only the manifest's own digest moves, because the manifest changed.
    """
    layout = LocalCorpusLayout(pathlib.Path(root or LOCAL_CORPUS_ROOT))
    verify(layout.root)
    rebuild(layout.root)
    recount(layout.root)

    manifest = load_manifest(layout)
    before = manifest["content"]["corpus_content_hash"]
    manifest["content"]["verified"] = True
    manifest["content"]["reproducible"] = True
    manifest["manifest_content_sha256"] = sha256_canonical(manifest["content"])
    layout.manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    # The data identity must not have moved. Only the status did.
    assert manifest["content"]["corpus_content_hash"] == before
    verify(layout.root)
    return {"ok": True, "verified": True, "reproducible": True,
            "corpus_content_hash": before,
            "manifest_content_sha256": manifest["manifest_content_sha256"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command",
                        choices=("verify", "rebuild", "recount", "promote"))
    parser.add_argument("--root", default=LOCAL_CORPUS_ROOT)
    arguments = parser.parse_args(argv)
    runner = {"verify": verify, "rebuild": rebuild, "recount": recount,
              "promote": promote}
    try:
        report = runner[arguments.command](arguments.root)
    except (CorpusVerificationError, EquityCorpusError) as error:
        print(json.dumps({"ok": False, "reason": str(error)}, indent=1),
              file=sys.stderr)
        return 1
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


__all__ = ["CorpusVerificationError", "load_manifest", "main", "promote",
           "rebuild", "rebuild_from_raw", "recount", "spec_from_manifest",
           "verify"]


if __name__ == "__main__":                           # pragma: no cover
    raise SystemExit(main())
