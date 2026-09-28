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
