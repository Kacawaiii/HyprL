"""Constants of the SEC EDGAR capture spec V1, revision 1, and the binding to its canonical hash."""

from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import re

from scripts.trading_lab.sources.canonical import sha256_canonical

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs" / "artifacts" / "edgar_capture_spec_v1.json"
SPEC_HASH = "98828c552bd2ca50550c07d542d28382e1138eae6a32493ee247466b5ffee5ce"
SPEC_REVISION = 1
SCHEMA_VERSION = "edgar-store-v1"

PROVIDER_ID = "sec_edgar_submissions_v1"
PROVIDER_CLASS = "REGULATOR_GOVERNMENT"
SOURCE_TIER = "TIER_1_OFFICIAL"
EVENT_FAMILY = "SEC_CURRENT_REPORT_FILING"
SUBMISSIONS_HOST = "data.sec.gov"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"

FORMS_IN_SCOPE = frozenset({"8-K", "8-K/A"})
WATCHLIST_MAX = 10
CIK = re.compile(r"^[0-9]{1,10}$")
ACCESSION = re.compile(r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$")
FILING_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
REQUIRED_COLUMNS = ("accessionNumber", "form", "filingDate", "acceptanceDateTime", "primaryDocument")
TEXT_COLUMNS = ("reportDate", "act", "fileNumber", "filmNumber", "items", "primaryDocDescription")
INTEGER_COLUMNS = ("size", "isXBRL", "isInlineXBRL")
FIELD_OF = {  # listing column -> normalized field
    "accessionNumber": "accession_number", "form": "form", "filingDate": "filing_date", "reportDate": "report_date",
    "acceptanceDateTime": "acceptance_datetime_text", "act": "act", "fileNumber": "file_number",
    "filmNumber": "film_number", "items": "items", "size": "size", "isXBRL": "is_xbrl",
    "isInlineXBRL": "is_inline_xbrl", "primaryDocument": "primary_document",
    "primaryDocDescription": "primary_doc_description",
}
METADATA_IDENTITY_ID = "EDGAR_FILING_METADATA_V1"

CLOCK_CHECK_TOLERANCE_S = 90
CLOCK_ERROR_BOUND = timedelta(seconds=92)
AGE_CAP_S = 86400
AGE_MAX = 2147483647
SPACING_S, WINDOW_S, WINDOW_MAX_STARTS, EMBARGO_S = 10, 60, 6, 60
POLL_INTERVAL_S = 600
THROTTLE_PAUSE_S = 600
DEADLINE_S = 30
BODY_CAP = 16 * 1024 * 1024
JSON_MEDIA = "application/json"
CONTENT_CODINGS = frozenset({"identity", "gzip"})
USER_AGENT = re.compile(r"^\S.*\s[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def verify_spec_binding(path: Path = SPEC_PATH) -> str:
    """Return the canonical hash of the spec on disk; raise if it is not revision 1."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    digest = sha256_canonical(spec)
    if digest != SPEC_HASH or spec.get("spec_revision") != SPEC_REVISION:
        raise RuntimeError(f"EDGAR spec drift: {digest} rev {spec.get('spec_revision')}, expected {SPEC_HASH}")
    return digest
