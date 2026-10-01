"""FOMC_CANON_V1 (spec revision 24, revision_policy.content_identity): the content identity of a
primary statement record, separate from its raw integrity.

The raw bytes keep their own SHA-256 (raw_sha256: integrity, storage, replay). The content identity is
the SHA-256 of canonical bytes in which only three Cloudflare spans, proved volatile and without effect
on the statement, are neutralized:
- CF_EMAIL_LINK  `href="/cdn-cgi/l/email-protection#H"` inside an `<a ` start tag,
- CF_EMAIL_SPAN  `<span class="__cf_email__" data-cfemail="H">[email&#160;protected]</span>`,
  both re-keyed: H (key byte k, then bytes XOR k) becomes "00" + hex(decoded) - the effective address
  stays in the identity, only the per-response key goes;
- CF_CHALLENGE_PARAMS  `window.__CF$cv$params={r:'R',t:'T'}` inside the exact Cloudflare challenge
  script (checked by digest) right before `</body>`: R and T are emptied.
Every other byte is kept. Anything unrecognized fails closed: the record is REFUSED and its canonical
bytes are its raw bytes, so it never merges with other content.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
import hashlib
import re

CANONICALIZER_ID = "FOMC_CANON_V1"
MAX_EMAIL_LINKS = 16
MAX_EMAIL_SPANS = 16
MAX_CHALLENGE_PARAMS = 1
MARKERS = (b"email-protection", b"__cf_email__", b"data-cfemail", b"__CF$cv$params")

_HEX = rb"([0-9a-f]{4,2048})"
EMAIL_LINK = re.compile(rb'href="/cdn-cgi/l/email-protection#' + _HEX + rb'"')
EMAIL_SPAN = re.compile(rb'<span class="__cf_email__" data-cfemail="' + _HEX + rb'">\[email&#160;protected\]</span>')
CHALLENGE_PARAMS = re.compile(rb"window\.__CF\$cv\$params=\{r:'([0-9a-f]{16})',t:'([A-Za-z0-9+/]{4,64}={0,2})'\}")
# SHA-256 of the whole Cloudflare challenge <script>...</script> element with R and T emptied, as served
# with every statement of the 2026-10-01 pilot (32 responses, 17 URLs); any other script is refused.
CHALLENGE_SCRIPT_SHA256 = "f69b046ab2c86664e4f0d0ab8179163a3d0333c65183fc32b5976fbf9159a7b2"


@dataclass(frozen=True)
class Canonical:
    status: str  # CANONICAL or REFUSED
    content_sha256: str
    canonical: bytes = field(repr=False)
    reason: str | None = None
    neutralized: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"canonicalizer": CANONICALIZER_ID, "status": self.status, "reason": self.reason,
                "content_sha256": self.content_sha256, "neutralized": dict(self.neutralized)}


class _Refused(Exception):
    pass


def _rekey(hex_text: bytes) -> bytes:
    """Decode a Cloudflare-obfuscated value and re-encode it with key 0."""
    if len(hex_text) % 2:
        raise _Refused("obfuscated value of odd length")
    data = binascii.unhexlify(hex_text)
    key, decoded = data[0], bytes(b ^ data[0] for b in data[1:])
    if not 1 <= len(decoded) <= 1023 or any(not 0x20 <= b <= 0x7E for b in decoded):
        raise _Refused("obfuscated value does not decode to printable ASCII")
    del key
    return b"00" + binascii.hexlify(decoded)


def _inside_a_tag(raw: bytes, start: int) -> bool:
    opening = raw.rfind(b"<", 0, start)
    return opening >= 0 and raw.startswith(b"<a ", opening) and b">" not in raw[opening:start]


def _challenge_script_ok(raw: bytes, match: re.Match) -> bool:
    opening = raw.rfind(b"<script>", 0, match.start())
    closing = raw.find(b"</script>", match.end())
    if opening < 0 or closing < 0 or not raw.startswith(b"</body>", closing + len(b"</script>")):
        return False
    script = raw[opening:match.start(1)] + raw[match.end(1):match.start(2)] + raw[match.end(2):closing + len(b"</script>")]
    return hashlib.sha256(script).hexdigest() == CHALLENGE_SCRIPT_SHA256


def _timestamp_ok(token: bytes) -> bool:
    try:
        decoded = base64.b64decode(token, validate=True)
    except binascii.Error:
        return False
    return 1 <= len(decoded) <= 20 and decoded.isdigit()


def canonicalize(raw: bytes) -> Canonical:
    """FOMC_CANON_V1 over the complete raw bytes of a primary statement record."""
    try:
        raw.decode("utf-8", errors="strict")
        replacements = []  # (start, end, new bytes), non-overlapping
        counts = {"CF_EMAIL_LINK": 0, "CF_EMAIL_SPAN": 0, "CF_CHALLENGE_PARAMS": 0}
        covered = {marker: 0 for marker in MARKERS}
        for m in EMAIL_LINK.finditer(raw):
            if not _inside_a_tag(raw, m.start()):
                raise _Refused("email-protection link outside an <a> start tag")
            replacements.append((m.start(1), m.end(1), _rekey(m.group(1))))
            counts["CF_EMAIL_LINK"] += 1
            covered[b"email-protection"] += 1
        for m in EMAIL_SPAN.finditer(raw):
            replacements.append((m.start(1), m.end(1), _rekey(m.group(1))))
            counts["CF_EMAIL_SPAN"] += 1
            covered[b"__cf_email__"] += 1
            covered[b"data-cfemail"] += 1
        for m in CHALLENGE_PARAMS.finditer(raw):
            if not _challenge_script_ok(raw, m):
                raise _Refused("challenge parameters outside the known Cloudflare script")
            if not _timestamp_ok(m.group(2)):
                raise _Refused("challenge t is not a base64 timestamp")
            replacements += [(m.start(1), m.end(1), b""), (m.start(2), m.end(2), b"")]
            counts["CF_CHALLENGE_PARAMS"] += 1
            covered[b"__CF$cv$params"] += 1
        for marker in MARKERS:
            if raw.count(marker) != covered[marker]:
                raise _Refused(f"unrecognized occurrence of {marker.decode()}")
        if (counts["CF_EMAIL_LINK"] > MAX_EMAIL_LINKS or counts["CF_EMAIL_SPAN"] > MAX_EMAIL_SPANS
                or counts["CF_CHALLENGE_PARAMS"] > MAX_CHALLENGE_PARAMS):
            raise _Refused("bound exceeded")
    except UnicodeDecodeError:
        return _refused(raw, "not strict UTF-8")
    except _Refused as exc:
        return _refused(raw, str(exc))
    out, position = [], 0
    for start, end, new in sorted(replacements):
        if start < position:
            return _refused(raw, "overlapping spans")
        out += [raw[position:start], new]
        position = end
    out.append(raw[position:])
    canonical = b"".join(out)
    return Canonical("CANONICAL", hashlib.sha256(canonical).hexdigest(), canonical, None, counts)


def _refused(raw: bytes, reason: str) -> Canonical:
    return Canonical("REFUSED", hashlib.sha256(raw).hexdigest(), raw, reason, {})
