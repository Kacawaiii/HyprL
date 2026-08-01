from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from crypto_news.collection import (
    CollectionService,
    CollectionValidationError,
    FetchResult,
    load_fixture_directory,
    normalize_timestamp,
    normalize_url,
)
from crypto_news.egress import DEFAULT_EGRESS_POLICY, EgressDeniedError
from crypto_news.evidence import EvidenceIntegrityError, EvidenceStore
from crypto_news.journal import Journal
from crypto_news.sources import (
    DEFAULT_SOURCE_REGISTRY,
    SourceDefinition,
    SourceRegistryError,
    SourceTier,
)


UTC = timezone.utc
BASE = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "collection"


def test_source_registry_covers_every_tier_and_authorizes_enabled_endpoints() -> None:
    assert {source.tier for source in DEFAULT_SOURCE_REGISTRY} == set(SourceTier)
    for source in DEFAULT_SOURCE_REGISTRY:
        if source.enabled:
            assert source.endpoint_url is not None
            assert source.parser is not None
            assert DEFAULT_EGRESS_POLICY.authorize(source.endpoint_url) == source.endpoint_url

    assert DEFAULT_SOURCE_REGISTRY.get("sec_press_releases").primary_authority is True
    assert DEFAULT_SOURCE_REGISTRY.get("social_unverified").enabled is False

    with pytest.raises(SourceRegistryError, match="tier"):
        SourceDefinition(
            source_id="invalid_bool_tier",
            name="Invalid",
            tier=True,  # type: ignore[arg-type]
            independence_group="invalid",
        )


def test_fetch_result_and_fixture_manifest_have_intrinsic_size_limits(
    tmp_path: Path,
) -> None:
    with pytest.raises(CollectionValidationError, match="maximum size"):
        FetchResult(
            source_id="coinbase_status",
            endpoint_url="https://status.coinbase.com/api/v2/incidents.json",
            media_type="application/json",
            first_seen_at=BASE,
            retrieved_at=BASE,
            body=b"x" * (10 * 1024 * 1024 + 1),
        )

    (tmp_path / "oversized.fixture.json").write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(CollectionValidationError, match="manifest exceeds"):
        load_fixture_directory(tmp_path)


def test_fixture_manifest_parser_failures_are_validation_errors(tmp_path: Path) -> None:
    deep = tmp_path / "deep"
    deep.mkdir()
    nested = b"[" * 10_000 + b"0" + b"]" * 10_000
    (deep / "deep.fixture.json").write_bytes(
        b'{"source_id":"coinbase_status","body_file":"body.json","nested":'
        + nested
        + b"}"
    )
    with pytest.raises(CollectionValidationError, match="invalid fixture manifest"):
        load_fixture_directory(deep)

    nul = tmp_path / "nul"
    nul.mkdir()
    (nul / "nul.fixture.json").write_text(
        json.dumps(
            {
                "source_id": "coinbase_status",
                "endpoint_url": "https://status.coinbase.com/api/v2/incidents.json",
                "media_type": "application/json",
                "first_seen_at": "2026-07-31T12:00:00Z",
                "retrieved_at": "2026-07-31T12:00:00Z",
                "body_file": "bad\u0000name.json",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CollectionValidationError, match="fixture body_file"):
        load_fixture_directory(nul)


def test_timestamp_and_url_normalization_is_deterministic() -> None:
    assert normalize_timestamp("Fri, 31 Jul 2026 11:59:00 GMT") == datetime(
        2026, 7, 31, 11, 59, tzinfo=UTC
    )
    assert normalize_timestamp("2026-07-31T13:59:00+02:00") == datetime(
        2026, 7, 31, 11, 59, tzinfo=UTC
    )
    assert normalize_url(
        "https://WWW.SEC.GOV/news/example?utm_source=rss&b=2&a=1#ignored"
    ) == "https://www.sec.gov/news/example?a=1&b=2"
    assert normalize_url("https://www.sec.gov:443/a/../news/example") == (
        "https://www.sec.gov/news/example"
    )
    assert normalize_url("https://www.sec.gov/%65xample/%2e%2e/news") == (
        "https://www.sec.gov/news"
    )

    with pytest.raises(EgressDeniedError):
        normalize_url("https://evil.example/news")

    with pytest.raises(CollectionValidationError, match="invalid timestamp"):
        normalize_timestamp("0001-01-01T00:00:00+23:59")
    with pytest.raises(CollectionValidationError, match="timestamp"):
        normalize_timestamp("not-a-date")


def test_deeply_nested_status_json_fails_cleanly_after_preserving_raw_evidence(
    tmp_path: Path,
) -> None:
    body = b'{"incidents":[],"nested":' + b"[" * 10000 + b"0" + b"]" * 10000 + b"}"
    fetch = FetchResult(
        source_id="coinbase_status",
        endpoint_url="https://status.coinbase.com/api/v2/incidents.json",
        media_type="application/json",
        first_seen_at=BASE,
        retrieved_at=BASE + timedelta(seconds=1),
        body=body,
    )
    journal = Journal(tmp_path / "journal.sqlite3")
    evidence = EvidenceStore(tmp_path / "evidence")
    try:
        with pytest.raises(CollectionValidationError, match="invalid status API"):
            CollectionService(
                journal=journal,
                evidence_store=evidence,
                registry=DEFAULT_SOURCE_REGISTRY,
            ).collect(fetch)
        assert evidence.read(sha256(body).hexdigest()) == body
    finally:
        journal.close()


def test_fixture_loader_is_offline_and_rejects_path_traversal(tmp_path: Path) -> None:
    fetches = load_fixture_directory(FIXTURE_DIR)
    assert {fetch.source_id for fetch in fetches} == {
        "coinbase_status",
        "sec_press_releases",
    }
    assert all(fetch.body for fetch in fetches)

    outside = tmp_path / "outside.xml"
    outside.write_text("<rss/>", encoding="utf-8")
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "bad.fixture.json").write_text(
        '{"source_id":"sec_press_releases",'
        '"endpoint_url":"https://www.sec.gov/news/pressreleases.rss",'
        '"media_type":"application/rss+xml",'
        '"first_seen_at":"2026-07-31T12:00:00Z",'
        '"retrieved_at":"2026-07-31T12:00:01Z",'
        '"body_file":"../outside.xml"}',
        encoding="utf-8",
    )
    with pytest.raises(CollectionValidationError, match="fixture directory"):
        load_fixture_directory(fixture_dir)


def test_raw_evidence_store_detects_tampering(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    artifact = store.store(b"original source bytes")
    artifact.path.write_bytes(b"tampered source bytes")
    with pytest.raises(EvidenceIntegrityError, match="hash mismatch"):
        store.read(artifact.sha256)


def test_rss_and_status_api_collection_preserve_raw_proof_and_deduplicate(
    tmp_path: Path,
) -> None:
    fetches = {fetch.source_id: fetch for fetch in load_fixture_directory(FIXTURE_DIR)}
    journal = Journal(tmp_path / "journal.sqlite3")
    evidence = EvidenceStore(tmp_path / "evidence")
    service = CollectionService(
        journal=journal,
        evidence_store=evidence,
        registry=DEFAULT_SOURCE_REGISTRY,
    )
    try:
        sec = service.collect(fetches["sec_press_releases"])
        assert (sec.parsed_count, sec.stored_count, sec.duplicate_count, sec.rejected_count) == (
            2,
            2,
            0,
            0,
        )
        assert sec.receipts[0].canonical_url.endswith("?a=1&b=2")
        normalized_content = json.loads(evidence.read(sec.receipts[0].content_sha256))
        assert normalized_content["raw_artifact_sha256"] == sec.raw_artifact.sha256
        assert evidence.read(sec.raw_artifact.sha256) == fetches["sec_press_releases"].body

        repeated = service.collect(fetches["sec_press_releases"])
        assert repeated.stored_count == 0
        assert repeated.duplicate_count == 2

        status = service.collect(fetches["coinbase_status"])
        assert status.stored_count == 1
        assert status.receipts[0].canonical_url == (
            "https://status.coinbase.com/incidents/abc123"
        )
        assert journal.count_records() == 3
        assert journal.verify_chain() is True
    finally:
        journal.close()


def test_invalid_future_item_is_rejected_but_raw_evidence_is_retained(
    tmp_path: Path,
) -> None:
    body = b"""<?xml version='1.0'?><rss><channel><item>
      <guid>future</guid><title>Future-dated claim</title>
      <link>https://www.sec.gov/news/future</link>
      <description>Untrusted clock.</description>
      <pubDate>2026-08-01T00:00:00Z</pubDate>
    </item></channel></rss>"""
    fetch = FetchResult(
        source_id="sec_press_releases",
        endpoint_url="https://www.sec.gov/news/pressreleases.rss",
        media_type="application/rss+xml",
        first_seen_at=BASE,
        retrieved_at=BASE + timedelta(seconds=1),
        body=body,
    )
    journal = Journal(tmp_path / "journal.sqlite3")
    evidence = EvidenceStore(tmp_path / "evidence")
    try:
        result = CollectionService(
            journal=journal,
            evidence_store=evidence,
            registry=DEFAULT_SOURCE_REGISTRY,
        ).collect(fetch)
        assert result.rejected_count == 1
        assert result.stored_count == 0
        assert journal.count_records() == 0
        assert evidence.read(sha256(body).hexdigest()) == body
    finally:
        journal.close()


def test_feed_revision_only_appends_genuinely_new_items(tmp_path: Path) -> None:
    original = {
        fetch.source_id: fetch for fetch in load_fixture_directory(FIXTURE_DIR)
    }["sec_press_releases"]
    extra_item = b"""<item>
      <guid>sec-new-2026</guid><title>SEC publishes a new crypto statement</title>
      <link>https://www.sec.gov/newsroom/press-releases/2026-102</link>
      <description>A genuinely new statement.</description>
      <pubDate>2026-07-31T12:00:04Z</pubDate>
    </item>"""
    revised = FetchResult(
        source_id=original.source_id,
        endpoint_url=original.endpoint_url,
        media_type=original.media_type,
        first_seen_at=BASE + timedelta(seconds=5),
        retrieved_at=BASE + timedelta(seconds=6),
        body=original.body.replace(b"</channel>", extra_item + b"</channel>"),
    )
    journal = Journal(tmp_path / "journal.sqlite3")
    service = CollectionService(
        journal=journal,
        evidence_store=EvidenceStore(tmp_path / "evidence"),
        registry=DEFAULT_SOURCE_REGISTRY,
    )
    try:
        assert service.collect(original).stored_count == 2
        result = service.collect(revised)
        assert result.parsed_count == 3
        assert result.stored_count == 1
        assert result.duplicate_count == 2
        assert journal.count_records() == 3
    finally:
        journal.close()


def test_xml_entity_declarations_are_rejected_after_arbitrary_padding(
    tmp_path: Path,
) -> None:
    body = b" " * 5000 + b"""<!DOCTYPE rss [<!ENTITY claim 'expanded'>]>
    <rss><channel><item><guid>entity</guid><title>&claim;</title>
    <link>https://www.sec.gov/news/entity</link><description>unsafe</description>
    <pubDate>2026-07-31T11:59:00Z</pubDate></item></channel></rss>"""
    fetch = FetchResult(
        source_id="sec_press_releases",
        endpoint_url="https://www.sec.gov/news/pressreleases.rss",
        media_type="application/rss+xml",
        first_seen_at=BASE,
        retrieved_at=BASE + timedelta(seconds=1),
        body=body,
    )
    journal = Journal(tmp_path / "journal.sqlite3")
    try:
        with pytest.raises(CollectionValidationError, match="declare entities"):
            CollectionService(
                journal=journal,
                evidence_store=EvidenceStore(tmp_path / "evidence"),
                registry=DEFAULT_SOURCE_REGISTRY,
            ).collect(fetch)
    finally:
        journal.close()


def test_utf16_xml_entities_are_rejected(tmp_path: Path) -> None:
    xml = """<!DOCTYPE rss [<!ENTITY claim 'expanded'>]>
    <rss><channel><item><guid>entity</guid><title>&claim;</title>
    <link>https://www.sec.gov/news/entity</link><description>unsafe</description>
    <pubDate>2026-07-31T11:59:00Z</pubDate></item></channel></rss>"""
    fetch = FetchResult(
        source_id="sec_press_releases",
        endpoint_url="https://www.sec.gov/news/pressreleases.rss",
        media_type="application/rss+xml",
        first_seen_at=BASE,
        retrieved_at=BASE + timedelta(seconds=1),
        body=xml.encode("utf-16"),
    )
    journal = Journal(tmp_path / "journal.sqlite3")
    try:
        with pytest.raises(CollectionValidationError, match="declare entities"):
            CollectionService(
                journal=journal,
                evidence_store=EvidenceStore(tmp_path / "evidence"),
                registry=DEFAULT_SOURCE_REGISTRY,
            ).collect(fetch)
    finally:
        journal.close()


def test_atom_prefers_published_timestamp_and_alternate_link(tmp_path: Path) -> None:
    body = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>atom-1</id><title>Official statement</title>
      <updated>2026-07-31T12:05:00Z</updated>
      <published>2026-07-31T12:00:00Z</published>
      <link rel="self" href="https://www.sec.gov/api/atom-1"/>
      <link rel="alternate" href="https://www.sec.gov/news/atom-1"/>
      <summary>Primary text.</summary></entry></feed>"""
    fetch = FetchResult(
        source_id="sec_press_releases",
        endpoint_url="https://www.sec.gov/news/pressreleases.rss",
        media_type="application/atom+xml",
        first_seen_at=BASE + timedelta(minutes=6),
        retrieved_at=BASE + timedelta(minutes=6, seconds=1),
        body=body,
    )
    journal = Journal(tmp_path / "journal.sqlite3")
    try:
        result = CollectionService(
            journal=journal,
            evidence_store=EvidenceStore(tmp_path / "evidence"),
            registry=DEFAULT_SOURCE_REGISTRY,
        ).collect(fetch)
        assert result.receipts[0].published_at == BASE
        assert result.receipts[0].canonical_url == "https://www.sec.gov/news/atom-1"
    finally:
        journal.close()


def test_malformed_payload_fails_cleanly_after_preserving_raw_evidence(
    tmp_path: Path,
) -> None:
    body = b"<rss><broken>"
    fetch = FetchResult(
        source_id="sec_press_releases",
        endpoint_url="https://www.sec.gov/news/pressreleases.rss",
        media_type="application/rss+xml",
        first_seen_at=BASE,
        retrieved_at=BASE + timedelta(seconds=1),
        body=body,
    )
    journal = Journal(tmp_path / "journal.sqlite3")
    evidence = EvidenceStore(tmp_path / "evidence")
    try:
        with pytest.raises(CollectionValidationError, match="RSS/Atom"):
            CollectionService(
                journal=journal,
                evidence_store=evidence,
                registry=DEFAULT_SOURCE_REGISTRY,
            ).collect(fetch)
        assert evidence.read(sha256(body).hexdigest()) == body
        assert journal.count_records() == 0
    finally:
        journal.close()


def test_fixture_collection_cli_is_persistent_and_deduplicates(tmp_path: Path) -> None:
    journal_path = tmp_path / "cli-journal.sqlite3"
    evidence_path = tmp_path / "cli-evidence"
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts" / "collect.py"),
        "--fixture",
        str(FIXTURE_DIR),
        "--journal",
        str(journal_path),
        "--evidence-dir",
        str(evidence_path),
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(first.stdout) == {
        "duplicate_count": 0,
        "ephemeral": False,
        "fetch_count": 2,
        "network_access": False,
        "parsed_count": 3,
        "rejected_count": 0,
        "stored_count": 3,
    }
    assert json.loads(second.stdout)["duplicate_count"] == 3

    journal = Journal(journal_path)
    try:
        assert journal.count_records() == 3
        assert journal.verify_chain() is True
    finally:
        journal.close()
