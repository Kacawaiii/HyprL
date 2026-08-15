"""Is the local Yahoo research corpus actually usable, right now, on this box?

The question sounds like `directory.exists()` and is not. A directory can hold
a half-written capture, a failed attempt, a corpus from a different spec, or
four files whose bytes drifted after they were frozen. Every one of those
answers "yes" to existence and "no" to the only question that matters, which is
whether the bars about to be charted are the bars that were verified.

So availability here is a *binding*, not a lookup. The committed fingerprint is
the anchor -- it is in Git, it is reviewed, and it cannot be edited by anything
that touches `var/`. The local manifest is checked against it, the local files
are checked against the manifest, and only a corpus that survives all three is
AVAILABLE. A manifest saying `verified: true` proves nothing on its own: it is
a claim written by the process being audited, and this module exists precisely
because that claim can outlive the state that justified it.

**Atomic.** Three good instruments and one missing file is not a corpus with a
gap; it is an invalid corpus. The frozen artefact is the set of four, its
aggregate hash covers all of them, and serving three would mean serving
something no fingerprint describes.

**Offline by construction.** Nothing in this module imports a transport, opens
a socket or knows a URL. A missing corpus is a state to render, never a
download to start -- the capture path is a deliberate command-line act, and
putting it behind an HTTP GET would turn a page load into a network fetch from
an unofficial source.

**Says little.** The status it publishes carries hashes, counts and identities.
It never carries an absolute path: the frontend has no use for one, and a
filesystem layout is the kind of detail that leaks into a screenshot.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field

from scripts.trading_lab.equity_corpus import CORPUS_SPEC_V2, sha256_canonical

LOCAL_RESEARCH_CORPUS_SCHEMA_VERSION = "trading-lab.local-research-corpus.v1"

# Where a capture writes, and the only place discovery looks. A sibling
# directory holding a failed attempt is not a candidate: it is not this path.
DEFAULT_CORPUS_ROOT = "var/trading_lab/research/yahoo_us_equity_daily_v2"

# The committed anchor. In Git, reviewed, and outside the store it validates.
DEFAULT_FINGERPRINT_PATH = "docs/artifacts/us_equity_corpus_v2_fingerprint.json"


class CorpusStatus:
    """The four answers. Anything not AVAILABLE serves no bars."""

    AVAILABLE = "AVAILABLE"
    NOT_INSTALLED = "NOT_INSTALLED"
    INVALID = "INVALID"
    CORRUPT = "CORRUPT"


# INVALID vs CORRUPT is a real distinction and worth keeping: INVALID means the
# store describes a corpus this build did not freeze (wrong spec, wrong
# calendar, wrong instruments), CORRUPT means it describes the right one and
# the bytes no longer match. The first is usually a stale or foreign store; the
# second means something edited a frozen file. Neither serves bars.


class LocalCorpusError(RuntimeError):
    """Raised when a caller asks for bars a corpus cannot honestly provide."""


def _slug(instrument_id: str) -> str:
    return instrument_id.replace(":", "_")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class InstrumentDiagnostic:
    """Per-instrument detail. Published for diagnosis, never for eligibility.

    A caller reading this must not conclude "three are fine, chart those" --
    the corpus-level status is the only thing that decides, and §8 is why.
    """

    instrument_id: str
    present: bool
    rows: int = 0
    content_hash_matches: bool = False
    canonical_sha256_matches: bool = False
    problem: str | None = None

    @property
    def ok(self) -> bool:
        return (self.present and self.content_hash_matches
                and self.canonical_sha256_matches and self.problem is None)

    def payload(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "present": self.present,
            "rows": self.rows,
            "content_hash_matches": self.content_hash_matches,
            "canonical_sha256_matches": self.canonical_sha256_matches,
            "problem": self.problem,
        }


@dataclass(frozen=True)
class LocalCorpusReport:
    """What discovery concluded, and enough to explain why."""

    status: str
    reasons: tuple[str, ...] = ()
    instruments: tuple[InstrumentDiagnostic, ...] = ()
    identity: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == CorpusStatus.AVAILABLE

    def payload(self) -> dict:
        """Safe metadata only. No absolute path, no raw payload, no secret."""
        return {
            "schema_version": LOCAL_RESEARCH_CORPUS_SCHEMA_VERSION,
            "status": self.status,
            "available": self.available,
            "reasons": list(self.reasons),
            "instruments": [item.payload() for item in self.instruments],
            **self.identity,
        }


class LocalResearchCorpusRegistry:
    """Binds a committed fingerprint to a local store, or refuses.

    Built with explicit roots rather than reading a global: the corruption
    tests need to point it at a temporary store, and a registry that can only
    ever look at one hardcoded path is a registry that cannot be tested for
    the cases that matter.
    """

    def __init__(self, *, corpus_root=None, fingerprint_path=None, spec=None):
        self.corpus_root = pathlib.Path(corpus_root or DEFAULT_CORPUS_ROOT)
        self.fingerprint_path = pathlib.Path(
            fingerprint_path or DEFAULT_FINGERPRINT_PATH)
        self.spec = spec or CORPUS_SPEC_V2
        self._cached: LocalCorpusReport | None = None
        self._cache_key: tuple | None = None

    # --- paths ------------------------------------------------------------

    @property
    def manifest_path(self) -> pathlib.Path:
        return self.corpus_root / "manifest.local.json"

    def canonical_path(self, instrument_id: str) -> pathlib.Path:
        return self.corpus_root / "canonical" / f"{_slug(instrument_id)}.jsonl"

    # --- cache ------------------------------------------------------------

    def _identity_key(self) -> tuple:
        """File identity of everything the verdict depends on.

        Size and mtime of each input, so a corpus edited after it was cached
        is revalidated instead of being served from a verdict that describes
        the previous bytes. Deliberately not "cache forever": the whole point
        of this module is that a frozen file can stop being what it was.
        """
        paths = [self.fingerprint_path, self.manifest_path]
        paths.extend(self.canonical_path(item) for item in self.spec.instruments)
        key = []
        for path in paths:
            try:
                stat = path.stat()
                key.append((str(path), stat.st_size, stat.st_mtime_ns))
            except OSError:
                key.append((str(path), None, None))
        return tuple(key)

    def report(self, *, refresh: bool = False) -> LocalCorpusReport:
        """The verdict, cached against file identity."""
        key = self._identity_key()
        if not refresh and self._cached is not None and self._cache_key == key:
            return self._cached
        report = self._evaluate()
        self._cached = report
        self._cache_key = key
        return report

    # --- the binding ------------------------------------------------------

    def _read_json(self, path: pathlib.Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return None

    def _evaluate(self) -> LocalCorpusReport:
        spec = self.spec
        expected_instruments = tuple(spec.instruments)

        # 1. The committed anchor. Without it there is nothing to bind to, and
        #    a manifest validating only against itself is not a check.
        fingerprint = self._read_json(self.fingerprint_path)
        if fingerprint is None:
            return LocalCorpusReport(
                CorpusStatus.NOT_INSTALLED,
                ("the committed corpus fingerprint is missing or unreadable",))

        identity = {
            "corpus_id": fingerprint.get("corpus_id"),
            "provider_id": fingerprint.get("provider_id"),
            "timeframe": fingerprint.get("timeframe"),
            "adjustment_policy": fingerprint.get("adjustment_policy"),
            "session_type": fingerprint.get("session_type"),
            "requested_range": fingerprint.get("requested_range"),
            "corpus_spec_hash": fingerprint.get("corpus_spec_hash"),
            "calendar_spec_hash": fingerprint.get("calendar_spec_hash"),
            "calendar_dependency_version":
                fingerprint.get("calendar_dependency_version"),
            "corpus_content_hash": fingerprint.get("corpus_content_hash"),
            "official_contract": bool(
                fingerprint.get("source_official_contract", False)),
            "redistribution_permitted": bool(
                fingerprint.get("source_redistribution_permitted", False)),
            "instruments_expected": list(expected_instruments),
        }

        # 2/3. The store and its manifest.
        if not self.corpus_root.is_dir():
            return LocalCorpusReport(
                CorpusStatus.NOT_INSTALLED,
                ("no local research corpus store on this machine",),
                identity=identity)
        manifest = self._read_json(self.manifest_path)
        if manifest is None or not isinstance(manifest.get("content"), dict):
            return LocalCorpusReport(
                CorpusStatus.NOT_INSTALLED,
                ("the local store has no readable manifest",),
                identity=identity)
        content = manifest["content"]
        # The spec the capture recorded, as its own block. Read from where it
        # lives rather than from a flattened copy: a second copy of these
        # fields at the top level would be a second thing to keep in step.
        stored_spec = content.get("corpus_spec")
        if not isinstance(stored_spec, dict):
            return LocalCorpusReport(
                CorpusStatus.INVALID,
                ("the local manifest records no corpus spec",),
                identity=identity)

        reasons: list[str] = []

        # 4-10. Identity: the store must describe the corpus this build froze.
        #       Compared against the SPEC as well as the fingerprint, so a
        #       fingerprint and a manifest that were edited together still
        #       fail against the code.
        checks = (
            ("corpus_id", stored_spec.get("corpus_id"), spec.corpus_id),
            ("provider_id", stored_spec.get("provider_id"), spec.provider_id),
            ("corpus_spec_hash", content.get("corpus_spec_hash"),
             spec.corpus_spec_hash),
            ("timeframe", stored_spec.get("timeframe"), spec.timeframe),
            ("adjustment_policy", stored_spec.get("adjustment_policy"),
             spec.adjustment_policy),
            ("session_type", stored_spec.get("session_type"), spec.session_type),
        )
        for name, found, wanted in checks:
            if found != wanted:
                reasons.append(f"{name} does not match the frozen spec")

        calendar_identity = spec.calendar_identity()
        if stored_spec.get("calendar_spec_hash") != \
                calendar_identity["calendar_spec_hash"]:
            reasons.append("calendar_spec_hash does not match the frozen calendar")
        if stored_spec.get("calendar_dependency_version") != \
                calendar_identity["calendar_dependency_version"]:
            reasons.append("the calendar dependency version differs")

        requested = stored_spec.get("requested_range") or {}
        if (requested.get("start"), requested.get("end")) != \
                (spec.requested_start, spec.requested_end):
            reasons.append("the requested range does not match the frozen spec")

        entries = content.get("instruments")
        if not isinstance(entries, list):
            reasons.append("the manifest names no instruments")
            entries = []
        found_ids = tuple(entry.get("instrument_id") for entry in entries
                          if isinstance(entry, dict))
        if tuple(sorted(found_ids)) != tuple(sorted(expected_instruments)):
            reasons.append("the manifest instrument set is not the frozen set")

        # The fingerprint and the manifest must agree with each other too --
        # this is the join that makes a local edit detectable.
        if content.get("corpus_spec_hash") != fingerprint.get("corpus_spec_hash"):
            reasons.append("manifest and fingerprint disagree on the spec hash")
        if stored_spec.get("calendar_spec_hash") != \
                fingerprint.get("calendar_spec_hash"):
            reasons.append("manifest and fingerprint disagree on the calendar hash")

        if reasons:
            return LocalCorpusReport(CorpusStatus.INVALID, tuple(reasons),
                                     identity=identity)

        # 11/12. The manifest's own claims. Necessary, nowhere near sufficient.
        if not content.get("verified"):
            reasons.append("the local manifest does not claim to be verified")
        if not content.get("reproducible"):
            reasons.append("the local manifest does not claim to be reproducible")
        if reasons:
            return LocalCorpusReport(CorpusStatus.INVALID, tuple(reasons),
                                     identity=identity)

        # 13. Aggregate hash: fingerprint vs manifest, before touching a file.
        aggregate = fingerprint.get("corpus_content_hash")
        if content.get("corpus_content_hash") != aggregate:
            return LocalCorpusReport(
                CorpusStatus.CORRUPT,
                ("the local aggregate corpus hash does not match the "
                 "committed fingerprint",),
                identity=identity)

        # 14-16. Per instrument: the fingerprint's hash, the manifest's hash,
        #        and the bytes on disk all have to be the same story.
        fingerprint_by_id = {
            entry.get("instrument_id"): entry
            for entry in fingerprint.get("instruments", [])
            if isinstance(entry, dict)
        }
        manifest_by_id = {entry["instrument_id"]: entry for entry in entries
                          if isinstance(entry, dict)}

        diagnostics: list[InstrumentDiagnostic] = []
        corrupt = False
        for instrument_id in expected_instruments:
            expected = fingerprint_by_id.get(instrument_id)
            recorded = manifest_by_id.get(instrument_id)
            path = self.canonical_path(instrument_id)
            if expected is None or recorded is None:
                diagnostics.append(InstrumentDiagnostic(
                    instrument_id, present=False,
                    problem="not described by both fingerprint and manifest"))
                corrupt = True
                continue
            if not path.is_file():
                diagnostics.append(InstrumentDiagnostic(
                    instrument_id, present=False,
                    problem="the canonical file is missing"))
                corrupt = True
                continue

            file_digest = _sha256_file(path)
            file_ok = file_digest == expected.get("canonical_sha256")
            # The content hash is over the parsed rows, so it catches a
            # reordering or a re-serialisation that a byte digest would also
            # catch -- but it is what the aggregate is built from, so it is the
            # one that has to agree with the fingerprint.
            try:
                rows = [json.loads(line) for line in
                        path.read_text(encoding="utf-8").splitlines() if line]
            except (OSError, ValueError, UnicodeDecodeError):
                diagnostics.append(InstrumentDiagnostic(
                    instrument_id, present=True, canonical_sha256_matches=file_ok,
                    problem="the canonical file is not readable JSONL"))
                corrupt = True
                continue
            content_hash = sha256_canonical(rows)
            content_ok = content_hash == expected.get("instrument_content_hash")
            manifest_ok = (recorded.get("instrument_content_hash")
                           == expected.get("instrument_content_hash"))
            diagnostic = InstrumentDiagnostic(
                instrument_id, present=True, rows=len(rows),
                content_hash_matches=content_ok and manifest_ok,
                canonical_sha256_matches=file_ok,
                problem=None if (content_ok and manifest_ok and file_ok)
                else "the local bytes do not match the committed fingerprint")
            diagnostics.append(diagnostic)
            if not diagnostic.ok:
                corrupt = True

        if corrupt:
            # Atomic: one bad instrument invalidates the corpus, even though
            # the per-instrument detail above says which one.
            return LocalCorpusReport(
                CorpusStatus.CORRUPT,
                ("at least one instrument does not match the committed "
                 "fingerprint; the corpus is atomic and serves nothing",),
                tuple(diagnostics), identity=identity)

        # 13 again, computed rather than compared: the aggregate the local
        # files actually produce. If this differs, the manifest agreed with the
        # fingerprint about a number that no longer describes the data.
        rebuilt_aggregate = sha256_canonical(sorted(
            [{"instrument_id": entry["instrument_id"],
              "instrument_content_hash": entry["instrument_content_hash"],
              "rows": entry["canonical_rows"]}
             for entry in entries],
            key=lambda item: item["instrument_id"]))
        if rebuilt_aggregate != aggregate:
            return LocalCorpusReport(
                CorpusStatus.CORRUPT,
                ("the aggregate hash recomputed from the local manifest does "
                 "not match the committed fingerprint",),
                tuple(diagnostics), identity=identity)

        identity["rows_total"] = sum(item.rows for item in diagnostics)
        identity["expected_sessions"] = content.get(
            "expected_sessions_per_instrument")
        return LocalCorpusReport(CorpusStatus.AVAILABLE, (), tuple(diagnostics),
                                 identity=identity)

    # --- reading ----------------------------------------------------------

    def require_available(self) -> LocalCorpusReport:
        report = self.report()
        if not report.available:
            raise LocalCorpusError(
                f"the local research corpus is {report.status}")
        return report

    def instrument_ids(self) -> tuple[str, ...]:
        return tuple(self.spec.instruments)

    def read_bars(self, instrument_id: str) -> tuple[dict, ...]:
        """Canonical daily bars for one instrument, or refuse.

        Reads only the canonical artefact. Never the raw provider payload,
        never `adjclose`, never the network -- the canonical rows are the ones
        the fingerprint covers, and they are the only ones anything downstream
        is allowed to see.
        """
        report = self.require_available()
        if instrument_id not in self.spec.instruments:
            raise LocalCorpusError(
                f"{instrument_id!r} is not part of the local research corpus")
        path = self.canonical_path(instrument_id)
        try:
            rows = [json.loads(line) for line
                    in path.read_text(encoding="utf-8").splitlines() if line]
        except (OSError, ValueError, UnicodeDecodeError) as error:
            raise LocalCorpusError(
                "the canonical file could not be read") from error
        del report
        return tuple(rows)

    def metadata(self) -> dict:
        """Response metadata every bars payload carries. States its limits."""
        spec = self.spec
        return {
            "provider": spec.provider_id,
            "corpus_id": spec.corpus_id,
            "source_timeframe": spec.timeframe,
            "adjustment": spec.adjustment_policy,
            "session": spec.session_type,
            "official_contract": False,
            "redistribution_permitted": False,
            "local_verified": True,
            "source_kind": "LOCAL_RESEARCH_CORPUS",
            "live": False,
            "realtime": False,
        }


_DEFAULT_REGISTRY: LocalResearchCorpusRegistry | None = None


def default_registry() -> LocalResearchCorpusRegistry:
    """The process-wide registry over the conventional roots."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = LocalResearchCorpusRegistry()
    return _DEFAULT_REGISTRY


__all__ = [
    "DEFAULT_CORPUS_ROOT", "DEFAULT_FINGERPRINT_PATH",
    "LOCAL_RESEARCH_CORPUS_SCHEMA_VERSION", "CorpusStatus",
    "InstrumentDiagnostic", "LocalCorpusError", "LocalCorpusReport",
    "LocalResearchCorpusRegistry", "default_registry",
]
