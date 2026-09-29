"""Constants of FOMC capture spec V1, revision 22, and the binding to its canonical hash.

Every number here is copied from the authoritative JSON; `verify_spec_binding` fails if the
JSON on disk is not the revision this code was written against.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs" / "artifacts" / "fomc_capture_spec_v1.json"
SPEC_HASH = "ba6a01e5f12e810ecde89711304c278862d147de298c63e7602a8359fa18e678"
SPEC_REVISION = 22

PROVIDER_ID = "federal_reserve_fomc_statements_v1"
EVENT_FAMILY = "FOMC_MONETARY_POLICY_STATEMENT"
ALLOWED_HOST = "www.federalreserve.gov"
FEED_URL = "https://www.federalreserve.gov/feeds/press_monetary.xml"
FEED_TITLE_EXACT = "Federal Reserve issues FOMC statement"
PRIMARY_TITLE_EXACT = "Federal Reserve issues FOMC statement"
USER_AGENT = "hyprl-trading-lab-event-capture/1.0"
CAPTURE_SPEC_ID = "federal_reserve_fomc_capture_v1"
CAPTURE_SCOPE_ID = "standard_fomc_statement_release_pattern_v1"  # scope.capture_scope_id
PROVIDER_CLASS = "CENTRAL_BANK_GOVERNMENT"
SOURCE_TIER = "TIER_1_OFFICIAL"
TAXONOMY_TYPE = "CENTRAL_BANK"
TAXONOMY_VERSION = "trading-lab.event-taxonomy.v1"  # binds.taxonomy_version

# normalized_minimum_fields (27): each spec name, the key it is stored under, and where it lives.
# A revision is keyed by (source item, content hash) and immutable; what differs per observation lives
# on its observation-to-revision link (revision_mode_neutrality). observation_mode and ingested_at are
# on both: on the revision they are creation provenance (the creating observation's mode, the avail
# of the creating transaction); on a link they are that observation's own.
NORMALIZED_KEYS = {
    "content_source_available_at_always_null_in_v1": "content_source_available_at",
    "declared_release_at_nullable": "declared_release_at",
    "source_updated_at_always_null_in_v1": "source_updated_at",
}
REVISION_FIELDS = (
    "canonical_source_url", "capture_scope_id", "capture_spec_hash", "capture_spec_id", "classification_state",
    "content_source_available_at", "declared_release_at", "declared_release_text", "declared_release_trust_verdict",
    "event_family", "ingested_at", "observation_mode", "official_statement_date", "provider_class", "provider_id",
    "revision_id", "source_item_id", "source_tier", "source_updated_at", "taxonomy_type", "taxonomy_version",
    "timestamp_semantics", "timestamp_trust_verdict",
)
OBSERVATION_FIELDS = (
    "ingested_at", "observation_mode", "observed_at", "raw_artifact_identities_and_hashes", "rss_guid_if_available",
    "source_observation_id",
)

# causal_predicates.constants
CLOCK_CHECK_TOLERANCE_S = 90
CLOCK_ERROR_BOUND_S = 92
AGE_CAP_S = 86400
AGE_MAX = 2147483647

# revision_policy.reobservation_* and retry_policy
OFFSETS_S = (300, 3600, 86400, 604800)
ATTEMPTS_PER_EPISODE = 6
BACKOFF_S = (60, 300, 900, 3600, 14400)
FIRST_ATTEMPT_GAP_S = 60

# http_policy / request_accounting / rate_limiter
MAX_REDIRECTS = 3
MAX_PHYSICAL_PER_ATTEMPT = 4
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
FEED_BODY_CAP = 2097152
STATEMENT_BODY_CAP = 5242880
CONNECT_TIMEOUT_S = 10
READ_TIMEOUT_S = 30
ATTEMPT_DEADLINE_S = 60
WINDOW_S = 60
WINDOW_MAX_STARTS = 6
SPACING_S = 10
EMBARGO_S = 60
GRANT_TO_TRANSPORT_S = 1
FEED_CADENCE_S = 60

# post_durable_processing.reconciliation and retry_policy.interrupted / local_failure
RUN_DEADLINE_S = 600
ATTEMPT_ABSOLUTE_DEADLINE_S = 600
SAVE_DEADLINE_S = 120
POISON_DEAD_RUNS = 2

MANIFEST_MAX_RAW_ENTRIES = 1000
FEED_MEDIA = frozenset({"application/rss+xml", "application/xml", "text/xml"})
PRIMARY_MEDIA = frozenset({"text/html"})
ADMITTED_CONTENT_CODINGS = frozenset({"identity", "gzip"})


def canonical_bytes(payload: object) -> bytes:
    """canonical_serialization.rule: the repo's sha256_canonical byte form."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_canonical(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_spec_binding(path: Path = SPEC_PATH) -> str:
    """Return the canonical hash of the spec on disk; raise if it is not revision 22."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    digest = sha256_canonical(spec)
    if digest != SPEC_HASH or spec.get("spec_revision") != SPEC_REVISION:
        raise RuntimeError(f"FOMC spec drift: {digest} rev {spec.get('spec_revision')}, expected {SPEC_HASH}")
    return digest
