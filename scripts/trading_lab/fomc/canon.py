"""FOMC_CANON_V2 and FOMC_CONTENT_IDENTITY_V2 (spec revision 25, revision_policy.content_identity): the
content identity of a primary statement record, separate from its raw integrity.

The raw bytes keep their own SHA-256 (raw_sha256: integrity, storage, replay). Content identity is the
SHA-256 of a canonical JSON object naming its identity version, its canonicalizer, its domain and the
SHA-256 of its bytes. In the CANONICAL domain the bytes are the raw bytes in which only three Cloudflare
spans, proved volatile and without effect on the statement, are neutralized, each recognized as a real
HTML token at its position in the original bytes:
- CF_EMAIL_LINK  the value of the only `href` attribute of an `<a>` start tag, exactly
  `/cdn-cgi/l/email-protection#H` (double-quoted);
- CF_EMAIL_SPAN  a `<span>` start tag with exactly `class="__cf_email__" data-cfemail="H"`, whose content
  is exactly `[email&#160;protected]` followed by `</span>`;
  both re-keyed: H (key byte k, then bytes XOR k) becomes "00" + hex(decoded) - the effective address
  stays in the identity, only the per-response key goes;
- CF_CHALLENGE_PARAMS  `window.__CF$cv$params={r:'R',t:'T'}` inside the content of an attribute-less
  `<script>` element whose bytes, R and T removed, have the frozen digest, immediately followed by
  `</body>`: R and T are emptied.
Every other byte is kept and nothing is reserialized. Comments, other attributes' values and raw-text
elements (script, style, textarea, title, ...) are not contexts for the e-mail spans. A marker anywhere
else, or a document the strict tokenizer cannot read unambiguously, fails closed: the RAW_FALLBACK
domain over the raw bytes, which never equals a CANONICAL identity.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
import hashlib
import re

from scripts.trading_lab.fomc import spec

IDENTITY_ID = "FOMC_CONTENT_IDENTITY_V2"
CANONICALIZER_ID = "FOMC_CANON_V2"
MAX_EMAIL_LINKS = 16
MAX_EMAIL_SPANS = 16
MAX_CHALLENGE_PARAMS = 1
MARKERS = (b"email-protection", b"__cf_email__", b"data-cfemail", b"__CF$cv$params")

EMAIL_LINK_VALUE = re.compile(rb"/cdn-cgi/l/email-protection#([0-9a-f]{4,2048})")
CHALLENGE_PARAMS = re.compile(rb"window\.__CF\$cv\$params=\{r:'([0-9a-f]{16})',t:'([A-Za-z0-9+/]{4,64}={0,2})'\}")
EMAIL_SPAN_TEXT = b"[email&#160;protected]"
# SHA-256 of the whole Cloudflare challenge <script>...</script> element with R and T emptied, as served
# with every statement of the 2026-10-01 pilot (33 responses, 17 URLs); any other script is refused.
CHALLENGE_SCRIPT_SHA256 = "f69b046ab2c86664e4f0d0ab8179163a3d0333c65183fc32b5976fbf9159a7b2"

RAW_TEXT_ELEMENTS = frozenset({b"script", b"style", b"textarea", b"title", b"xmp", b"iframe", b"noembed",
                               b"noframes", b"noscript", b"plaintext"})
_NAME = re.compile(rb"[A-Za-z][A-Za-z0-9:-]*")
_SPACE = b" \t\n\r\f"


@dataclass(frozen=True)
class Canonical:
    domain: str  # CANONICAL or RAW_FALLBACK
    content_sha256: str  # the content identity (domain-separated)
    bytes_sha256: str  # SHA-256 of the canonical bytes (CANONICAL) or of the raw bytes (RAW_FALLBACK)
    canonical: bytes = field(repr=False)
    reason: str | None = None
    neutralized: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.domain

    def summary(self) -> dict:
        return {"identity": IDENTITY_ID, "canonicalizer": CANONICALIZER_ID, "domain": self.domain,
                "bytes_sha256": self.bytes_sha256, "content_sha256": self.content_sha256, "reason": self.reason,
                "neutralized": dict(self.neutralized)}


def identity(domain: str, bytes_sha256: str) -> str:
    """The content identity: SHA-256 of the canonical serialization of its four components."""
    return spec.sha256_canonical({"identity": IDENTITY_ID, "canonicalizer": CANONICALIZER_ID, "domain": domain,
                                  "bytes_sha256": bytes_sha256})


class _Refused(Exception):
    pass


# ------------------------------------------------------------------ strict HTML tokenizer ----------
@dataclass
class _Tag:
    name: bytes
    start: int
    end: int  # exclusive, after '>'
    attrs: list  # (name, value_start, value_end, quote) with positions in the original bytes


def _tokens(raw: bytes):
    """Yield ("tag", _Tag), ("end", name, start, end) and ("raw", name, content_start, content_end, end)
    for every real token, skipping comments and declarations. Raises _Refused on anything that cannot be
    read unambiguously (unterminated comment, tag, attribute value or raw-text element)."""
    i, n = 0, len(raw)
    while True:
        lt = raw.find(b"<", i)
        if lt < 0:
            return
        if raw.startswith(b"<!--", lt):
            close = raw.find(b"-->", lt + 4)
            if close < 0:
                raise _Refused("unterminated comment")
            i = close + 3
            continue
        nxt = raw[lt + 1:lt + 2]
        if nxt in (b"!", b"?"):
            close = raw.find(b">", lt)
            if close < 0:
                raise _Refused("unterminated declaration")
            i = close + 1
            continue
        if nxt == b"/":
            m = _NAME.match(raw, lt + 2)
            close = raw.find(b">", lt)
            if close < 0:
                raise _Refused("unterminated end tag")
            if m:
                yield ("end", m.group(0).lower(), lt, close + 1)
            i = close + 1
            continue
        m = _NAME.match(raw, lt + 1)
        if not m:
            i = lt + 1  # a '<' in text
            continue
        tag = _start_tag(raw, lt, m)
        yield ("tag", tag)
        i = tag.end
        if tag.name in RAW_TEXT_ELEMENTS:
            if tag.name == b"plaintext":
                yield ("raw", tag.name, i, n, n)
                return
            close = re.compile(rb"</" + tag.name + rb"[\s/>]", re.IGNORECASE).search(raw, i)
            if close is None:
                raise _Refused(f"unterminated {tag.name.decode()} element")
            content_end = close.start()
            if tag.name == b"script" and b"<!--" in raw[i:content_end]:
                raise _Refused("script content with '<!--' (escaped script data is ambiguous)")
            yield ("raw", tag.name, i, content_end, content_end)
            i = content_end


def _start_tag(raw: bytes, lt: int, name: re.Match) -> _Tag:
    i, n, attrs = name.end(), len(raw), []
    while True:
        while i < n and raw[i] in _SPACE:
            i += 1
        if i >= n:
            raise _Refused("unterminated start tag")
        if raw[i:i + 1] == b">":
            return _Tag(name.group(0).lower(), lt, i + 1, attrs)
        if raw[i:i + 2] == b"/>":
            return _Tag(name.group(0).lower(), lt, i + 2, attrs)
        a = re.compile(rb"[^\s\"'>/=]+").match(raw, i)
        if not a:
            raise _Refused("malformed attribute")
        attr_name, i = a.group(0).lower(), a.end()
        j = i
        while j < n and raw[j] in _SPACE:
            j += 1
        if raw[j:j + 1] != b"=":
            attrs.append((attr_name, i, i, None))
            continue
        j += 1
        while j < n and raw[j] in _SPACE:
            j += 1
        quote = raw[j:j + 1]
        if quote in (b'"', b"'"):
            close = raw.find(quote, j + 1)
            if close < 0:
                raise _Refused("unterminated attribute value")
            attrs.append((attr_name, j + 1, close, quote))
            i = close + 1
        else:
            v = re.compile(rb"[^\s\"'=<>`]+").match(raw, j)
            if not v:
                raise _Refused("malformed attribute value")
            attrs.append((attr_name, j, v.end(), b""))
            i = v.end()


# ------------------------------------------------------------------ the canonicalizer -------------
def _rekey(hex_text: bytes) -> bytes:
    """Decode a Cloudflare-obfuscated value and re-encode it with key 0."""
    if len(hex_text) % 2:
        raise _Refused("obfuscated value of odd length")
    data = binascii.unhexlify(hex_text)
    decoded = bytes(b ^ data[0] for b in data[1:])
    if not 1 <= len(decoded) <= 1023 or any(not 0x20 <= b <= 0x7E for b in decoded):
        raise _Refused("obfuscated value does not decode to printable ASCII")
    return b"00" + binascii.hexlify(decoded)


def _timestamp_ok(token: bytes) -> bool:
    try:
        decoded = base64.b64decode(token, validate=True)
    except binascii.Error:
        return False
    return 1 <= len(decoded) <= 20 and decoded.isdigit()


def _spans(raw: bytes):
    """The allowed spans: (kind, covered region, replacements)."""
    tokens = list(_tokens(raw))
    found = []
    for index, token in enumerate(tokens):
        if token[0] == "tag" and token[1].name == b"a":
            tag = token[1]
            hrefs = [a for a in tag.attrs if a[0] == b"href"]
            if len(hrefs) > 1:
                raise _Refused("an <a> start tag with two href attributes")
            if hrefs and hrefs[0][3] == b'"':
                _name, vs, ve, _q = hrefs[0]
                m = EMAIL_LINK_VALUE.fullmatch(raw, vs, ve)
                if m:
                    found.append(("CF_EMAIL_LINK", (vs, ve), [(m.start(1), m.end(1), _rekey(m.group(1)))]))
        elif token[0] == "tag" and token[1].name == b"span":
            tag = token[1]
            names = [a[0] for a in tag.attrs]
            if b"data-cfemail" in names or any(raw[vs:ve] == b"__cf_email__" for _a, vs, ve, _q in tag.attrs):
                if names != [b"class", b"data-cfemail"] or any(a[3] != b'"' for a in tag.attrs):
                    raise _Refused("a Cloudflare e-mail span with other attributes or quoting")
                (_c, cs, ce, _q1), (_d, ds, de, _q2) = tag.attrs
                following = tokens[index + 1] if index + 1 < len(tokens) else None
                if (raw[cs:ce] != b"__cf_email__" or following is None or following[0] != "end"
                        or following[1] != b"span" or raw[tag.end:following[2]] != EMAIL_SPAN_TEXT):
                    raise _Refused("a Cloudflare e-mail span with other content")
                if not re.fullmatch(rb"[0-9a-f]{4,2048}", raw[ds:de]):
                    raise _Refused("obfuscated value of odd length" if re.fullmatch(rb"[0-9a-f]+", raw[ds:de])
                                   else "a Cloudflare e-mail span with a malformed value")
                found.append(("CF_EMAIL_SPAN", (tag.start, tag.end), [(ds, de, _rekey(raw[ds:de]))]))
        elif token[0] == "raw" and token[1] == b"script":
            _kind, _name, cs, ce, _end = token
            params = list(CHALLENGE_PARAMS.finditer(raw, cs, ce))
            if not params:
                continue
            opening = tokens[index - 1][1] if index and tokens[index - 1][0] == "tag" else None
            if len(params) != 1 or opening is None or opening.attrs:
                raise _Refused("challenge parameters outside the known Cloudflare script")
            m = params[0]
            end_tag = tokens[index + 1] if index + 1 < len(tokens) else None
            after = tokens[index + 2] if index + 2 < len(tokens) else None
            if (end_tag is None or end_tag[0] != "end" or end_tag[1] != b"script" or end_tag[2] != ce
                    or after is None or after[0] != "end" or after[1] != b"body" or after[2] != end_tag[3]):
                raise _Refused("challenge parameters outside the known Cloudflare script")
            script = raw[opening.start:m.start(1)] + raw[m.end(1):m.start(2)] + raw[m.end(2):end_tag[3]]
            if hashlib.sha256(script).hexdigest() != CHALLENGE_SCRIPT_SHA256:
                raise _Refused("challenge parameters outside the known Cloudflare script")
            if not _timestamp_ok(m.group(2)):
                raise _Refused("challenge t is not a base64 timestamp")
            found.append(("CF_CHALLENGE_PARAMS", (cs, ce), [(m.start(1), m.end(1), b""), (m.start(2), m.end(2), b"")]))
    return found


def canonicalize(raw: bytes) -> Canonical:
    """FOMC_CANON_V2 over the complete raw bytes of a primary statement record."""
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return _fallback(raw, "not strict UTF-8")
    occurrences = {marker: [m.start() for m in re.finditer(re.escape(marker), raw)] for marker in MARKERS}
    if not any(occurrences.values()):
        return _canonical(raw, {"CF_EMAIL_LINK": 0, "CF_EMAIL_SPAN": 0, "CF_CHALLENGE_PARAMS": 0}, raw)
    try:
        spans = _spans(raw)
        allowed = {b"email-protection": "CF_EMAIL_LINK", b"__cf_email__": "CF_EMAIL_SPAN",
                   b"data-cfemail": "CF_EMAIL_SPAN", b"__CF$cv$params": "CF_CHALLENGE_PARAMS"}
        for marker, positions in occurrences.items():
            for p in positions:
                if not any(kind == allowed[marker] and start <= p and p + len(marker) <= end
                           for kind, (start, end), _r in spans):
                    raise _Refused(f"{marker.decode()} outside an allowed HTML context")
        counts = {"CF_EMAIL_LINK": 0, "CF_EMAIL_SPAN": 0, "CF_CHALLENGE_PARAMS": 0}
        for kind, _region, _r in spans:
            counts[kind] += 1
        if (counts["CF_EMAIL_LINK"] > MAX_EMAIL_LINKS or counts["CF_EMAIL_SPAN"] > MAX_EMAIL_SPANS
                or counts["CF_CHALLENGE_PARAMS"] > MAX_CHALLENGE_PARAMS):
            raise _Refused("bound exceeded")
        replacements = sorted(r for _k, _region, rs in spans for r in rs)
    except _Refused as exc:
        return _fallback(raw, str(exc))
    out, position = [], 0
    for start, end, new in replacements:
        if start < position:
            return _fallback(raw, "overlapping spans")
        out += [raw[position:start], new]
        position = end
    out.append(raw[position:])
    return _canonical(b"".join(out), counts, raw)


def _canonical(canonical: bytes, counts: dict, raw: bytes) -> Canonical:
    digest = hashlib.sha256(canonical).hexdigest()
    return Canonical("CANONICAL", identity("CANONICAL", digest), digest, canonical, None, counts)


def _fallback(raw: bytes, reason: str) -> Canonical:
    digest = hashlib.sha256(raw).hexdigest()
    return Canonical("RAW_FALLBACK", identity("RAW_FALLBACK", digest), digest, raw, reason, {})
