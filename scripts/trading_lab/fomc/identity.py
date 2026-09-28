"""URL admission, canonical identity URL and durable identity keys (identity, url_canonicalization,
surfaces.statement_family_path, FIX4/FIX5/FIX6/FIX7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re

from scripts.trading_lab import safe_http
from scripts.trading_lab.fomc import spec

_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):(.*)$", re.S)
_HEX = frozenset("0123456789ABCDEFabcdef")
_FAMILY = re.compile(
    r"^/newsevents/pressreleases/monetary(?P<date>[0-9]{8})(?P<letter>[a-z])(?P<n>[0-9]*)\.htm$"
)


class UrlRejected(ValueError):
    """The URL is not admissible under V1; no identity is derived and no request is made."""


@dataclass(frozen=True)
class AdmittedUrl:
    raw: str
    canonical: str
    path: str


def _check_percent(text: str) -> None:
    i = 0
    while i < len(text):
        if text[i] == "%":
            pair = text[i + 1 : i + 3]
            if len(pair) != 2 or pair[0] not in _HEX or pair[1] not in _HEX:
                raise UrlRejected("malformed percent escape")
            i += 3
        else:
            i += 1


def admit_url(text: str) -> AdmittedUrl:
    """Validate one absolute URL and return its canonical identity form (never repaired)."""
    if not isinstance(text, str) or not text:
        raise UrlRejected("empty URL")
    if any(ord(ch) <= 0x20 or ord(ch) == 0x7F or ord(ch) > 0x7E for ch in text):
        raise UrlRejected("whitespace, control or non-ASCII character in URL")
    _check_percent(text)
    match = _SCHEME.match(text)
    if not match or match.group(1).lower() != "https":
        raise UrlRejected("scheme is not https")
    rest = match.group(2)
    if not rest.startswith("//"):
        raise UrlRejected("missing authority")
    rest = rest[2:]
    cut = len(rest)
    for sep in "/?#":
        pos = rest.find(sep)
        if pos != -1:
            cut = min(cut, pos)
    authority, tail = rest[:cut], rest[cut:]
    if "@" in authority:
        raise UrlRejected("userinfo is forbidden")
    host, port = authority, None
    if ":" in authority:
        host, port = authority.split(":", 1)
        if not port or not port.isdigit() or not port.isascii() or int(port, 10) > 65535:
            raise UrlRejected("malformed explicit port")
        if int(port, 10) != 443:
            raise UrlRejected("port other than 443")
    if host.lower() != spec.ALLOWED_HOST:
        raise UrlRejected("host not allowlisted")
    fragment_at = tail.find("#")
    before_fragment = tail if fragment_at == -1 else tail[:fragment_at]
    if "?" in before_fragment:
        raise UrlRejected("query component is forbidden")
    path = before_fragment
    if not path.startswith("/"):
        raise UrlRejected("empty or relative path")
    canonical = "https://" + spec.ALLOWED_HOST + path
    try:  # binds.shared_http_primitive: necessary, not sufficient
        safe_http.require_https_host(canonical, {spec.ALLOWED_HOST})
    except Exception as exc:  # any exception is a rejection
        raise UrlRejected(f"safe_http rejected: {exc}") from exc
    return AdmittedUrl(raw=text, canonical=canonical, path=path)


def is_statement_family_path(path: str) -> bool:
    match = _FAMILY.match(path)
    if not match:
        return False
    raw = match.group("date")
    try:
        date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))
    except ValueError:
        return False
    return True


def source_item_id(canonical_url: str) -> str:
    """identity.source_item_id: SHA-256 of [provider_id, event_family, canonical URL]."""
    return spec.sha256_canonical([spec.PROVIDER_ID, spec.EVENT_FAMILY, canonical_url])


def unidentifiable_key(raw_link: str | None, raw_title: str | None, raw_guid: str | None) -> str:
    return spec.sha256_canonical([raw_link, raw_title, raw_guid])


def reobservation_episode_key(sid: str, anchor_iso: str, offset: int) -> str:
    return spec.sha256_canonical(["REOBSERVATION", sid, anchor_iso, offset])


def acquisition_episode_key(sid: str, acquisition_seq: int) -> str:
    return spec.sha256_canonical(["LIVE_ACQUISITION", sid, acquisition_seq])


def backfill_episode_key(manifest_seq: int, sid: str) -> str:
    return spec.sha256_canonical(["HISTORICAL_BACKFILL", manifest_seq, sid])


def manual_episode_key(operator_seq: int) -> str:
    return spec.sha256_canonical(["MANUAL_RETRY", operator_seq])
