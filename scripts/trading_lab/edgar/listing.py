"""The submissions listing: strict parsing, filing identity and normalization (spec: discovery, identity,
normalization, unverified UV1/UV2/UV6). Pure functions over bytes; nothing here touches a store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json

from scripts.trading_lab.edgar import spec
from scripts.trading_lab.sources.canonical import sha256_canonical


class ListingRejected(ValueError):
    """The response is not a listing this spec can read: nothing is derived from it (UV1)."""


@dataclass(frozen=True)
class Listing:
    cik10: str
    entity_name: str | None
    rows: int
    oldest_filing_date: str | None
    accessions: frozenset[str]
    in_scope: list[dict] = field(default_factory=list)  # normalized fields, in listing order
    out_of_scope_forms: dict[str, int] = field(default_factory=dict)
    has_older_pages: bool = False


def cik10(cik: str) -> str:
    if not isinstance(cik, str) or not spec.CIK.fullmatch(cik):
        raise ValueError(f"a CIK is 1 to 10 ASCII digits, got {cik!r}")
    return cik.zfill(10)


def submissions_url(cik: str) -> str:
    return spec.SUBMISSIONS_URL.format(cik10=cik10(cik))


def source_item_id(accession: str) -> str:
    return sha256_canonical(["EdgarFiling", spec.PROVIDER_ID, accession])


def filing_identity(fields: dict) -> str:
    """EDGAR_FILING_METADATA_V1: the identity of a filing's listed metadata (revision_policy.identity)."""
    return sha256_canonical({"identity": spec.METADATA_IDENTITY_ID, "fields": fields})


def derived_urls(cik: str, accession: str, primary_document: str) -> dict:
    """Provenance only: these documents are never fetched by this slice."""
    folder = f"{spec.ARCHIVE_BASE}/{int(cik)}/{accession.replace('-', '')}"
    return {"filing_index_url": f"{folder}/{accession}-index.htm", "primary_document_url": f"{folder}/{primary_document}"}


def content_type_ok(lines: list[str]) -> bool:
    if len(lines) != 1:
        return False
    media = lines[0].split(";", 1)[0].strip().lower()
    return media == spec.JSON_MEDIA


def _no_constant(name: str):
    raise ListingRejected(f"non-JSON number {name}")


def parse_listing(body: bytes, requested_cik: str) -> Listing:
    """Parse one listing of the requested CIK, or raise ListingRejected. The column names and types are
    the ones the spec freezes (UV1): any other shape is refused, never guessed."""
    try:
        doc = json.loads(body.decode("utf-8"), parse_constant=_no_constant)
    except ListingRejected:
        raise
    except (ValueError, RecursionError) as exc:
        raise ListingRejected(f"not a UTF-8 JSON document: {exc}") from exc
    if not isinstance(doc, dict):
        raise ListingRejected("the listing is not a JSON object")
    listed = doc.get("cik")
    if isinstance(listed, bool) or not isinstance(listed, int):
        raise ListingRejected(f"the listing has no integer cik: {listed!r}")
    requested = cik10(requested_cik)
    if listed != int(requested):
        raise ListingRejected(f"the listing is for cik {listed}, not {requested}")
    filings = doc.get("filings")
    recent = filings.get("recent") if isinstance(filings, dict) else None
    if not isinstance(recent, dict):
        raise ListingRejected("the listing has no filings.recent object")
    for name in spec.REQUIRED_COLUMNS:
        if name not in recent:
            raise ListingRejected(f"filings.recent has no {name} column")
    lengths = set()
    for name, column in recent.items():
        if not isinstance(column, list):
            raise ListingRejected(f"filings.recent.{name} is not an array")
        lengths.add(len(column))
    if len(lengths) != 1:
        raise ListingRejected(f"filings.recent columns have unequal lengths {sorted(lengths)}")
    count = lengths.pop()
    for name in spec.REQUIRED_COLUMNS + spec.TEXT_COLUMNS:
        if name in recent and not all(isinstance(v, str) for v in recent[name]):
            raise ListingRejected(f"filings.recent.{name} holds a value that is not text")
    for name in spec.INTEGER_COLUMNS:
        if name in recent and not all(isinstance(v, int) and not isinstance(v, bool) for v in recent[name]):
            raise ListingRejected(f"filings.recent.{name} holds a value that is not an integer")
    accessions, in_scope, other = [], [], {}
    for i in range(count):
        accession, form, filed = recent["accessionNumber"][i], recent["form"][i], recent["filingDate"][i]
        if not spec.ACCESSION.fullmatch(accession):
            raise ListingRejected(f"row {i} has an invalid accession number {accession!r}")
        if not spec.FILING_DATE.fullmatch(filed):
            raise ListingRejected(f"row {i} has an invalid filingDate {filed!r}")
        try:
            date.fromisoformat(filed)
        except ValueError as exc:
            raise ListingRejected(f"row {i} has an invalid filingDate {filed!r}") from exc
        accessions.append(accession)
        if form not in spec.FORMS_IN_SCOPE:
            other[form] = other.get(form, 0) + 1
            continue
        fields = {target: (recent[name][i] if name in recent else None) for name, target in spec.FIELD_OF.items()}
        fields["cik"] = requested
        in_scope.append(fields)
    if len(set(accessions)) != len(accessions):
        raise ListingRejected("an accession number is listed twice")
    name = doc.get("name")
    return Listing(
        cik10=requested, entity_name=name if isinstance(name, str) else None, rows=count,
        oldest_filing_date=min(recent["filingDate"]) if count else None, accessions=frozenset(accessions),
        in_scope=in_scope, out_of_scope_forms=dict(sorted(other.items())),
        has_older_pages=bool(filings.get("files")),
    )
