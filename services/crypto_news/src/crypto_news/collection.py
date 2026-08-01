"""Deterministic RSS/API ingestion with offline fixture support."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from crypto_news.egress import DEFAULT_EGRESS_POLICY, EgressPolicy
from crypto_news.evidence import EvidenceStore, RawArtifact
from crypto_news.jcs import canonicalize
from crypto_news.journal import Journal, JournalValidationError
from crypto_news.models import ModelValidationError, SourceReceipt
from crypto_news.sources import SourceDefinition, SourceParser, SourceRegistry


MAX_FIXTURE_BYTES = 2 * 1024 * 1024
MAX_FETCH_BYTES = 10 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
_TRACKING_PARAMETERS = frozenset(
    {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source"}
)
_PERCENT_ESCAPE = re.compile(r"%([0-9a-fA-F]{2})")
_URL_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)


class CollectionValidationError(ValueError):
    """Raised when collected input cannot be normalized without inventing facts."""


def _utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CollectionValidationError(f"{field_name} must be timezone-aware")
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise CollectionValidationError(
            f"{field_name} is outside the supported UTC range"
        ) from exc


def normalize_timestamp(value: str | datetime) -> datetime:
    """Normalize ISO-8601 or RFC-822 timestamps to timezone-aware UTC."""

    if isinstance(value, datetime):
        return _utc(value, "timestamp")
    if not isinstance(value, str) or not value.strip():
        raise CollectionValidationError("timestamp must be a non-empty string")
    text = value.strip()
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except (ValueError, OverflowError):
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CollectionValidationError(f"invalid timestamp: {text}") from exc
    if parsed.tzinfo is None:
        raise CollectionValidationError("timestamp must include a timezone")
    try:
        return _utc(parsed, "timestamp")
    except CollectionValidationError as exc:
        raise CollectionValidationError(f"invalid timestamp: {text}") from exc


def _canonical_url_path(path: str) -> str:
    def replace_escape(match: re.Match[str]) -> str:
        value = chr(int(match.group(1), 16))
        return value if value in _URL_UNRESERVED else f"%{match.group(1).upper()}"

    return _PERCENT_ESCAPE.sub(replace_escape, path)


def normalize_url(
    url: str,
    *,
    base_url: str | None = None,
    policy: EgressPolicy = DEFAULT_EGRESS_POLICY,
) -> str:
    """Resolve and canonicalize a source URL before applying the egress policy."""

    if not isinstance(url, str) or not url.strip():
        raise CollectionValidationError("source URL must be non-empty")
    resolved = urljoin(base_url, url.strip()) if base_url else url.strip()
    try:
        parsed = urlsplit(resolved)
    except ValueError as exc:
        raise CollectionValidationError("source URL is invalid") from exc
    retained_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lower_key = key.lower()
        if lower_key.startswith("utm_") or lower_key in _TRACKING_PARAMETERS:
            continue
        retained_query.append((key, value))
    retained_query.sort()
    candidate = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            _canonical_url_path(parsed.path or "/"),
            urlencode(retained_query, doseq=True),
            "",
        )
    )
    authorized = urlsplit(policy.authorize(candidate))
    path = authorized.path
    while "//" in path:
        path = path.replace("//", "/")
    path = posixpath.normpath(path)
    if not path.startswith("/"):
        path = f"/{path}"
    if authorized.path.endswith("/") and not path.endswith("/"):
        path = f"{path}/"
    return urlunsplit(
        (authorized.scheme, str(authorized.hostname), path, authorized.query, "")
    )


@dataclass(frozen=True, slots=True)
class FetchResult:
    source_id: str
    endpoint_url: str
    media_type: str
    first_seen_at: datetime
    retrieved_at: datetime
    body: bytes

    def __post_init__(self) -> None:
        if not self.source_id or not self.endpoint_url or not self.media_type:
            raise CollectionValidationError(
                "source_id, endpoint_url, and media_type are required"
            )
        first_seen = _utc(self.first_seen_at, "first_seen_at")
        retrieved = _utc(self.retrieved_at, "retrieved_at")
        if retrieved < first_seen:
            raise CollectionValidationError("retrieved_at must be >= first_seen_at")
        if not isinstance(self.body, bytes) or not self.body:
            raise CollectionValidationError("response body must be non-empty bytes")
        if len(self.body) > MAX_FETCH_BYTES:
            raise CollectionValidationError("response body exceeds maximum size")
        object.__setattr__(self, "first_seen_at", first_seen)
        object.__setattr__(self, "retrieved_at", retrieved)


@dataclass(frozen=True, slots=True)
class ParsedItem:
    external_id: str
    title: str
    summary: str
    url: str
    published_at: datetime
    author: str | None


@dataclass(frozen=True, slots=True)
class RejectedItem:
    external_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class CollectionBatch:
    parsed_count: int
    stored_count: int
    duplicate_count: int
    rejected_count: int
    receipts: tuple[SourceReceipt, ...]
    rejections: tuple[RejectedItem, ...]
    raw_artifact: RawArtifact


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child(element: ET.Element, *names: str) -> ET.Element | None:
    for name in names:
        match = next(
            (item for item in element if _local_name(item.tag) == name.lower()),
            None,
        )
        if match is not None:
            return match
    return None


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _rss_or_atom_items(body: bytes) -> tuple[ParsedItem, ...]:
    upper_body = body.replace(b"\x00", b"").upper()
    if b"<!DOCTYPE" in upper_body or b"<!ENTITY" in upper_body:
        raise CollectionValidationError("RSS/Atom payload must not declare entities")
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise CollectionValidationError("invalid RSS/Atom payload") from exc
    root_name = _local_name(root.tag)
    if root_name in {"rss", "rdf"}:
        entries = [item for item in root.iter() if _local_name(item.tag) == "item"]
    elif root_name == "feed":
        entries = [item for item in root if _local_name(item.tag) == "entry"]
    else:
        raise CollectionValidationError("invalid RSS/Atom root element")

    parsed: list[ParsedItem] = []
    for entry in entries:
        title = _text(_child(entry, "title"))
        summary = _text(_child(entry, "description", "summary", "content"))
        link_nodes = [item for item in entry if _local_name(item.tag) == "link"]
        link_node = next(
            (
                item
                for relation in ("alternate", "")
                for item in link_nodes
                if item.attrib.get("rel", "").lower() == relation
            ),
            link_nodes[0] if link_nodes else None,
        )
        link = ""
        if link_node is not None:
            link = link_node.attrib.get("href", "") or _text(link_node)
        published_text = _text(_child(entry, "pubdate", "published", "updated"))
        external_id = _text(_child(entry, "guid", "id")) or link or title
        author_node = _child(entry, "author", "creator")
        author = _text(_child(author_node, "name")) if author_node is not None else ""
        if author_node is not None and not author:
            author = _text(author_node)
        if not title or not link or not published_text:
            raise CollectionValidationError("RSS/Atom item lacks title, link, or timestamp")
        parsed.append(
            ParsedItem(
                external_id=external_id,
                title=title,
                summary=summary,
                url=link,
                published_at=normalize_timestamp(published_text),
                author=author or None,
            )
        )
    return tuple(parsed)


def _statuspage_items(body: bytes) -> tuple[ParsedItem, ...]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CollectionValidationError("invalid status API payload") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("incidents"), list):
        raise CollectionValidationError("status API payload must contain incidents")
    parsed: list[ParsedItem] = []
    for incident in payload["incidents"]:
        if not isinstance(incident, dict):
            raise CollectionValidationError("status API incident must be an object")
        updates = incident.get("incident_updates", [])
        if not isinstance(updates, list):
            raise CollectionValidationError("status API updates must be a list")
        summary = "\n".join(
            str(update.get("body", "")).strip()
            for update in updates
            if isinstance(update, dict) and update.get("body")
        )
        identifier = incident.get("id")
        title = incident.get("name")
        link = incident.get("shortlink")
        published = incident.get("created_at")
        if not all(isinstance(value, str) and value.strip() for value in (identifier, title, link, published)):
            raise CollectionValidationError("status API incident lacks required fields")
        parsed.append(
            ParsedItem(
                external_id=identifier.strip(),
                title=title.strip(),
                summary=summary,
                url=link.strip(),
                published_at=normalize_timestamp(published),
                author=None,
            )
        )
    return tuple(parsed)


def _parse(source: SourceDefinition, body: bytes) -> tuple[ParsedItem, ...]:
    if source.parser is SourceParser.RSS_ATOM:
        return _rss_or_atom_items(body)
    if source.parser is SourceParser.STATUSPAGE_INCIDENTS:
        return _statuspage_items(body)
    raise CollectionValidationError("source parser is not supported")


def _receipt(
    source: SourceDefinition,
    item: ParsedItem,
    fetch: FetchResult,
    raw_artifact: RawArtifact,
    evidence_store: EvidenceStore,
) -> SourceReceipt:
    canonical_url = normalize_url(item.url, base_url=fetch.endpoint_url)
    normalized_payload = {
        "author": item.author,
        "canonical_url": canonical_url,
        "published_at": item.published_at.isoformat().replace("+00:00", "Z"),
        "summary": item.summary,
        "title": item.title,
    }
    normalized_hash = hashlib.sha256(canonicalize(normalized_payload)).hexdigest()
    content_artifact = evidence_store.store(
        canonicalize(
            {
                **normalized_payload,
                "raw_artifact_sha256": raw_artifact.sha256,
            }
        )
    )
    identity = hashlib.sha256(
        canonicalize(
            {
                "canonical_url": canonical_url,
                "normalized_sha256": normalized_hash,
                "source_id": source.source_id,
            }
        )
    ).hexdigest()
    return SourceReceipt(
        receipt_id=f"receipt-{identity}",
        canonical_url=canonical_url,
        source_name=source.name,
        source_tier=int(source.tier),
        published_at=item.published_at,
        first_seen_at=fetch.first_seen_at,
        retrieved_at=fetch.retrieved_at,
        content_sha256=content_artifact.sha256,
        author=item.author,
    )


class CollectionService:
    def __init__(
        self,
        *,
        journal: Journal,
        evidence_store: EvidenceStore,
        registry: SourceRegistry,
    ):
        self.journal = journal
        self.evidence_store = evidence_store
        self.registry = registry

    def collect(self, fetch: FetchResult) -> CollectionBatch:
        source = self.registry.get(fetch.source_id)
        if not source.enabled or source.endpoint_url is None:
            raise CollectionValidationError("source is not enabled for collection")
        if normalize_url(fetch.endpoint_url) != source.endpoint_url:
            raise CollectionValidationError("fetch endpoint does not match source registry")
        raw_artifact = self.evidence_store.store(fetch.body)
        parsed = _parse(source, fetch.body)
        receipts: list[SourceReceipt] = []
        rejections: list[RejectedItem] = []
        stored_count = 0
        duplicate_count = 0
        seen_in_batch: set[str] = set()
        for item in parsed:
            try:
                receipt = _receipt(
                    source,
                    item,
                    fetch,
                    raw_artifact,
                    self.evidence_store,
                )
            except (CollectionValidationError, ModelValidationError, ValueError) as exc:
                rejections.append(RejectedItem(item.external_id, str(exc)))
                continue
            receipts.append(receipt)
            if receipt.receipt_id in seen_in_batch or self.journal.has_record(receipt.receipt_id):
                duplicate_count += 1
                continue
            try:
                self.journal.append(receipt, recorded_at=fetch.retrieved_at)
            except JournalValidationError:
                if self.journal.has_record(receipt.receipt_id):
                    duplicate_count += 1
                    continue
                raise
            seen_in_batch.add(receipt.receipt_id)
            stored_count += 1
        return CollectionBatch(
            parsed_count=len(parsed),
            stored_count=stored_count,
            duplicate_count=duplicate_count,
            rejected_count=len(rejections),
            receipts=tuple(receipts),
            rejections=tuple(rejections),
            raw_artifact=raw_artifact,
        )


def load_fixture_directory(directory: str | Path) -> tuple[FetchResult, ...]:
    """Load explicit fixture manifests without any network capability."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise CollectionValidationError("fixture directory does not exist")
    fetches: list[FetchResult] = []
    for manifest_path in sorted(root.glob("*.fixture.json")):
        try:
            with manifest_path.open("rb") as manifest_file:
                manifest_bytes = manifest_file.read(MAX_MANIFEST_BYTES + 1)
            if len(manifest_bytes) > MAX_MANIFEST_BYTES:
                raise CollectionValidationError("fixture manifest exceeds maximum size")
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except CollectionValidationError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise CollectionValidationError("invalid fixture manifest") from exc
        if not isinstance(manifest, dict):
            raise CollectionValidationError("fixture manifest must be an object")
        body_name = manifest.get("body_file")
        if not isinstance(body_name, str) or not body_name:
            raise CollectionValidationError("fixture body_file is required")
        try:
            body_path = (root / body_name).resolve()
        except (OSError, ValueError) as exc:
            raise CollectionValidationError("fixture body_file is invalid") from exc
        if not body_path.is_relative_to(root):
            raise CollectionValidationError("fixture body must remain inside fixture directory")
        try:
            with body_path.open("rb") as body_file:
                body = body_file.read(MAX_FIXTURE_BYTES + 1)
            if len(body) > MAX_FIXTURE_BYTES:
                raise CollectionValidationError("fixture body exceeds maximum size")
            fetches.append(
                FetchResult(
                    source_id=manifest["source_id"],
                    endpoint_url=manifest["endpoint_url"],
                    media_type=manifest["media_type"],
                    first_seen_at=normalize_timestamp(manifest["first_seen_at"]),
                    retrieved_at=normalize_timestamp(manifest["retrieved_at"]),
                    body=body,
                )
            )
        except KeyError as exc:
            raise CollectionValidationError(f"fixture field is missing: {exc.args[0]}") from exc
        except OSError as exc:
            raise CollectionValidationError("fixture body cannot be read") from exc
    if not fetches:
        raise CollectionValidationError("fixture directory contains no manifests")
    return tuple(sorted(fetches, key=lambda fetch: (fetch.retrieved_at, fetch.source_id)))
