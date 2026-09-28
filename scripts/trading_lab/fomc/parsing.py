"""Offline parsing of durable raw bodies: Content-Type gate, strict UTF-8 text decoding, the secure
RSS extraction and the token-bounded HTML semantic anchors (parser_policy, http_policy.content_type_gate,
item_predicate.feed_text_normalization, timestamps.release_time / source_date).

Every function is a pure function of bytes and persisted metadata; failures are values, never repairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import html
from html.parser import HTMLParser
import re
import unicodedata
from xml.parsers import expat

from scripts.trading_lab.fomc import spec


class ParseFailed(Exception):
    """PARSER_FAILED with a reason; channel-level for a feed record."""


# ---------------------------------------------------------------- Content-Type gate --------------
_TOKEN = r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+"
_PARAM = re.compile(rf'[ \t]*;[ \t]*(?:({_TOKEN})=({_TOKEN}|"(?:[^"\\]|\\.)*"))?')


def _label_ok(value: str) -> bool:
    return value.strip(" \t\n\f\r").translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")) in ("utf-8", "utf8")


def content_type_gate(lines: list[str], surface: str) -> None:
    """CONTENT_TYPE_GATE_V1; raises ParseFailed. Every value must parse, share one media type and
    carry only UTF-8-compatible charset parameters."""
    if not lines:
        raise ParseFailed("Content-Type missing")
    accepted = spec.FEED_MEDIA if surface == "feed" else spec.PRIMARY_MEDIA
    media_types = set()
    for value in lines:
        match = re.match(rf"^[ \t]*({_TOKEN})/({_TOKEN})", value)
        if not match:
            raise ParseFailed("Content-Type unparseable")
        pos, names = match.end(), set()
        while pos < len(value.rstrip(" \t")):
            param = _PARAM.match(value, pos)
            if not param or param.end() == pos:
                raise ParseFailed("Content-Type parameter unparseable")
            if param.group(1):
                name = param.group(1).lower()
                if name in names:
                    raise ParseFailed("duplicate Content-Type parameter")
                names.add(name)
                raw = param.group(2)
                if raw.startswith('"'):
                    raw = re.sub(r"\\(.)", r"\1", raw[1:-1])
                if name == "charset" and not _label_ok(raw):
                    raise ParseFailed("non-UTF-8 charset")
            pos = param.end()
        media_types.add(f"{match.group(1)}/{match.group(2)}".lower())
    if len(media_types) != 1 or next(iter(media_types)) not in accepted:
        raise ParseFailed("unaccepted media type")


# ---------------------------------------------------------------- text decoding v1 --------------
_BAD_BOMS = (b"\x00\x00\xfe\xff", b"\xff\xfe\x00\x00", b"\xfe\xff", b"\xff\xfe")
_XML_DECL = re.compile(r'^<\?xml[ \t\r\n]+version[ \t\r\n]*=[ \t\r\n]*["\'][^"\']*["\']'
                       r'(?:[ \t\r\n]+encoding[ \t\r\n]*=[ \t\r\n]*(["\'])(.*?)\1)?[^>]*\?>')


def decode_text(body: bytes) -> str:
    """Strict UTF-8 over the text view (one leading EF BB BF omitted); no fallback."""
    for bom in _BAD_BOMS:
        if body.startswith(bom):
            raise ParseFailed("UTF-16/UTF-32 BOM")
    view = body[3:] if body.startswith(b"\xef\xbb\xbf") else body
    try:
        return view.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ParseFailed("invalid UTF-8") from exc


def check_xml_declaration(text: str) -> None:
    match = _XML_DECL.match(text)
    if match and match.group(2) is not None and not _label_ok(match.group(2)):
        raise ParseFailed("XML declaration encoding is not UTF-8")


# ---------------------------------------------------------------- normalization -----------------
def normalize_feed_text(value: str) -> str:
    """FEED_TEXT_NORMALIZATION_V1: NFC, collapse runs of U+0020/0009/000D/000A, trim U+0020."""
    value = unicodedata.normalize("NFC", value)
    value = re.sub("[ \u0009\u000d\u000a]+", " ", value)
    return value.strip(" ")


def normalize_semantic(value: str) -> str:
    """parser_policy.semantic_text_normalization steps 2-4 (references already resolved)."""
    value = unicodedata.normalize("NFC", value)
    value = re.sub("[\u0009\u000a\u000c\u000d ]+", " ", value)
    return value.strip(" ")


# ---------------------------------------------------------------- RSS ----------------------------
XINCLUDE_NS = "http://www.w3.org/2001/XInclude"


@dataclass
class FeedItem:
    fields: dict = field(default_factory=lambda: {"title": [], "link": [], "guid": []})
    bad: set = field(default_factory=set)  # fields that contained a child element

    def single(self, name: str) -> str | None:
        values = self.fields[name]
        if len(values) != 1 or name in self.bad:
            return None
        return values[0]

    def raw(self, name: str) -> str | None:
        values = self.fields[name]
        return values[0] if len(values) == 1 else None


class _SecurityReject(Exception):
    pass


def parse_feed(body: bytes) -> list[FeedItem]:
    """Secure RSS 2.0 item extraction. Raises ParseFailed (channel level) on any security construct,
    malformed XML or missing channel structure."""
    text = decode_text(body)
    check_xml_declaration(text)
    parser = expat.ParserCreate(namespace_separator=" ")
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)

    def reject(*_args):
        raise _SecurityReject()

    parser.StartDoctypeDeclHandler = reject
    parser.EntityDeclHandler = reject
    parser.ExternalEntityRefHandler = reject
    parser.ElementDeclHandler = reject
    parser.AttlistDeclHandler = reject
    parser.NotationDeclHandler = reject
    parser.UnparsedEntityDeclHandler = reject
    stack: list[str] = []
    items: list[FeedItem] = []
    state = {"channel": False, "item": None, "field": None, "buf": []}

    def start(name, _attrs):
        if name.startswith(XINCLUDE_NS + " "):
            raise _SecurityReject()
        local = name.split(" ")[-1] if " " in name else name
        depth = len(stack)
        if depth == 0 and name != "rss":
            raise ParseFailed("root is not rss")
        if depth == 1 and name == "channel":
            state["channel"] = True
        if depth == 2 and stack[1] == "channel" and name == "item":
            state["item"] = FeedItem()
        elif depth == 3 and state["item"] is not None and name in ("title", "link", "guid"):
            state["field"], state["buf"] = name, []
        elif depth >= 4 and state["field"] is not None:
            state["item"].bad.add(state["field"])
        stack.append(name if " " not in name else "{ns}" + local)

    def end(name):
        stack.pop()
        depth = len(stack)
        if depth == 3 and state["field"] == name and state["item"] is not None:
            state["item"].fields[name].append("".join(state["buf"]))
            state["field"] = None
        elif depth == 2 and name == "item" and state["item"] is not None:
            items.append(state["item"])
            state["item"] = None

    def chars(data):
        if state["field"] is not None and len(stack) == 4:
            state["buf"].append(data)

    parser.StartElementHandler, parser.EndElementHandler, parser.CharacterDataHandler = start, end, chars
    try:
        parser.Parse(text.encode("utf-8"), True)
    except _SecurityReject as exc:
        raise ParseFailed("XML security construct (DOCTYPE, entity or XInclude)") from exc
    except expat.ExpatError as exc:
        raise ParseFailed(f"malformed XML: {exc}") from exc
    if not state["channel"]:
        raise ParseFailed("channel structure missing")
    return items


# ---------------------------------------------------------------- HTML anchors -------------------
_VOID = frozenset("area base br col embed hr img input link meta source track wbr".split())
_CLOSES_P = frozenset("address article aside blockquote details div dl fieldset figcaption figure footer form "
                      "h1 h2 h3 h4 h5 h6 header hgroup hr main menu nav ol p pre section table ul".split())


@dataclass
class PrimaryFields:
    titles: list[str]
    dates: list[str]
    release_segments: list[str | None]  # None: empty leading segment
    meta_charsets: list[str]


class _AnchorParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.stack: list[tuple[str, dict]] = []
        self.titles, self.dates, self.releases, self.metas = [], [], [], []
        self._capture = None  # (kind, depth, parts)
        self._release = None

    def _classes(self, attrs):
        return set((dict(attrs).get("class") or "").split())

    def _in_heading(self) -> bool:
        return (len(self.stack) >= 2 and self.stack[-1][0] == "div" and "heading" in self.stack[-1][1]["classes"]
                and self.stack[-2][0] == "div" and self.stack[-2][1]["id"] == "article")

    def _markup(self):
        if self._release is not None:
            text = normalize_semantic("".join(self._release))
            self.releases.append(text or None)
            self._release = None

    def handle_starttag(self, tag, attrs):
        self._markup()
        if tag == "meta":
            a = {k.lower(): (v or "") for k, v in attrs}
            if "charset" in a:
                self.metas.append(a["charset"])
            if a.get("http-equiv", "").lower() == "content-type":
                m = re.search(r"charset[ \t]*=[ \t]*([^;]*)", a.get("content", ""), re.I)
                if m:
                    self.metas.append(m.group(1).strip().strip("\"'"))
        if tag in _CLOSES_P and self.stack and self.stack[-1][0] == "p":
            self._close_to("p")
        if tag in _VOID:
            return
        classes = self._classes(attrs)
        if self._in_heading() and self._capture is None:
            if tag == "h3" and "title" in classes:
                self._capture = ("title", len(self.stack) + 1, [])
            elif tag == "p" and "article__time" in classes:
                self._capture = ("date", len(self.stack) + 1, [])
            elif tag == "p" and "releaseTime" in classes:
                self._release = []
        self.stack.append((tag, {"classes": classes, "id": dict(attrs).get("id")}))

    def handle_startendtag(self, tag, attrs):
        self._markup()

    def _close_to(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                while len(self.stack) > i:
                    self.stack.pop()
                    if self._capture is not None and len(self.stack) < self._capture[1]:
                        kind, _depth, parts = self._capture
                        (self.titles if kind == "title" else self.dates).append(normalize_semantic("".join(parts)))
                        self._capture = None
                return

    def handle_endtag(self, tag):
        self._markup()
        self._close_to(tag)

    def _text(self, text):
        if self._capture is not None:
            self._capture[2].append(text)
        if self._release is not None:
            self._release.append(text)

    def handle_data(self, data):
        self._text(data)

    def handle_entityref(self, name):
        self._text(html.unescape(f"&{name};"))

    def handle_charref(self, name):
        self._text(html.unescape(f"&#{name};"))

    def handle_comment(self, data):
        self._markup()

    def handle_decl(self, decl):
        self._markup()

    def handle_pi(self, data):
        self._markup()


def parse_primary(body: bytes, http_charsets_ok: bool = True) -> PrimaryFields:
    text = decode_text(body)
    parser = _AnchorParser()
    parser.feed(text)
    parser.close()
    parser._markup()
    for label in parser.metas:
        if not _label_ok(label):
            raise ParseFailed("HTML meta charset is not UTF-8")
    return PrimaryFields(parser.titles, parser.dates, parser.releases, parser.metas)


# ---------------------------------------------------------------- grammars ----------------------
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December")
_DATE = re.compile(r"^(" + "|".join(_MONTHS) + r") ([1-9][0-9]?), ([0-9]{4})$")
_RELEASE = re.compile(r"^For release at ([1-9]|1[0-2]):([0-5][0-9]) (a\.m\.|p\.m\.) (EST|EDT)$")
_OFFSETS = {"EST": -5, "EDT": -4}


def parse_statement_date(text: str) -> date | None:
    match = _DATE.match(text)
    if not match:
        return None
    try:
        return date(int(match.group(3)), _MONTHS.index(match.group(1)) + 1, int(match.group(2)))
    except ValueError:
        return None


def parse_release(segment: str | None, statement_date: date) -> tuple[str, datetime | None]:
    """Returns (semantics, declared_release_at): EXACT with an instant, IMMEDIATE, or UNPARSED."""
    if segment == "For immediate release":
        return "IMMEDIATE", None
    match = _RELEASE.match(segment or "")
    if not match:
        return "UNPARSED", None
    hour, minute = int(match.group(1)), int(match.group(2))
    if match.group(3) == "a.m.":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    local = datetime(statement_date.year, statement_date.month, statement_date.day, hour, minute,
                     tzinfo=timezone(timedelta(hours=_OFFSETS[match.group(4)])))
    return "EXACT", local.astimezone(timezone.utc)
