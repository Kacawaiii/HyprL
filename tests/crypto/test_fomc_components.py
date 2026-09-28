"""Component boundaries of the FOMC slice: text decoding, HTML anchors, XML security, limiter bounds,
body cap, healthy negatives, GUID diagnostics and unidentifiable items."""

from __future__ import annotations

from datetime import date

import pytest

from scripts.trading_lab.fomc import identity, parsing, spec, state
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.limiter import Limiter

from tests.crypto.fomc_support import EXACT, P1, SID1, Env, statement_item


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.provider.close()


@pytest.mark.parametrize("body", [
    b"\xff\xfe<\x00r\x00",  # UTF-16 BOM
    b"<rss>\xe9 </rss>",  # invalid UTF-8 (FOMC46)
])
def test_strict_utf8_rejects(body):
    with pytest.raises(parsing.ParseFailed):
        parsing.decode_text(body)


def test_bom_is_omitted_from_text_only_fomc43():
    assert parsing.decode_text(b"\xef\xbb\xbfabc") == "abc"


def test_charset_signals_must_all_be_utf8():
    with pytest.raises(parsing.ParseFailed):  # FOMC44
        parsing.parse_feed(b'<?xml version="1.0" encoding="ISO-8859-1"?><rss><channel/></rss>')
    with pytest.raises(parsing.ParseFailed):  # FOMC41 (meta conflicts)
        parsing.parse_primary(b'<html><head><meta charset="windows-1252"></head></html>')
    with pytest.raises(parsing.ParseFailed):
        parsing.content_type_gate(["text/html; charset=x-unknown"], "primary")  # FOMC45


def test_anchor_cardinality_and_token_bound_release_fomc49_fomc54():
    page = syn.statement_html()
    fields = parsing.parse_primary(page)
    assert fields.release_segments == ["For release at 2:00 p.m. EDT"]  # the <ul> and body are excluded
    duplicate = page.replace(b'<h3 class="title">', b'<h3 class="title">x</h3><h3 class="title">', 1)
    assert len(parsing.parse_primary(duplicate).titles) == 2
    outside = page.replace(b'<div id="article">', b'<div id="other">')
    assert parsing.parse_primary(outside).titles == []  # no search outside div#article


def test_release_and_date_grammars_fomc01_fomc04_fomc132():
    assert parsing.parse_release("For release at 2:00 p.m. EST", date(2026, 7, 15))[1].hour == 19
    assert parsing.parse_release("For immediate release", date(2015, 3, 18)) == ("IMMEDIATE", None)
    assert parsing.parse_release("For release at 2:00 p.m. ET", date(2026, 6, 17)) == ("UNPARSED", None)
    assert parsing.parse_release("For release at 02:00 p.m. EDT", date(2026, 6, 17)) == ("UNPARSED", None)
    assert parsing.parse_statement_date("June 17, 2026") == date(2026, 6, 17)
    assert parsing.parse_statement_date("June 31, 2026") is None


@pytest.mark.parametrize("doc", [
    b'<!DOCTYPE rss [<!ENTITY a "lol">]><rss><channel/></rss>',  # FOMC57
    b'<!DOCTYPE rss [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><rss><channel>&xxe;</channel></rss>',  # FOMC58
    b'<rss xmlns:xi="http://www.w3.org/2001/XInclude"><channel><xi:include href="https://example.invalid/x"/></channel></rss>',
])
def test_xml_security_constructs_are_rejected(doc):
    with pytest.raises(parsing.ParseFailed):
        parsing.parse_feed(doc)


def test_xml_builtins_and_commented_doctype_are_accepted_fomc61_fomc62():
    items = parsing.parse_feed(b'<!-- <!DOCTYPE rss> --><rss><channel><item><title>a &amp; b &#66;</title></item></channel></rss>')
    assert items[0].single("title") == "a & b B"


def test_limiter_window_spacing_and_embargo_fomc90_fomc91_fomc97():
    clock = syn.SimClock(syn.START)
    limiter = Limiter(clock.mono, clock.sleep)
    first = limiter.grant()
    assert first - limiter.epoch_begin == spec.EMBARGO_S
    grants = [first] + [limiter.grant() for _ in range(6)]
    assert [round(g - first) for g in grants] == [0, 10, 20, 30, 40, 50, 60]
    for i, t in enumerate(grants):
        assert sum(1 for s in grants if 0 <= t - s < 60) <= 6


def test_oversized_statement_is_parser_failed_without_a_record_fomc39(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.SyntheticResponse(body=b"x" * (spec.STATEMENT_BODY_CAP + 1), headers=syn.HTML_HEADERS)
    env.drive(80)
    assert [r for r in env.store.rows("RESPONSE") if r.body["surface"] == "primary"] == []
    assert any(o.body["outcome"] == "PARSER_FAILED" and "size bound" in o.body["reason"]
               for o in env.store.rows("ATTEMPT_OUTCOME"))


def test_healthy_negative_concludes_and_zero_resumes_fomc137(env):
    env.feed([statement_item(title="Federal Reserve issues something else")])
    env.provider.routes[P1] = syn.page_response(title="Federal Reserve announces a routine action")
    env.drive(240)
    assert "DEFINITELY_OUT_OF_SCOPE" in env.outcomes()
    assert env.cycles()[0] == "NOT_ZERO" and env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"


def test_guid_conflict_collision_and_missing_are_diagnostics_fomc181_fomc182(env):
    p2 = syn.statement_path("20260618")
    env.provider.routes[P1] = syn.page_response()
    env.provider.routes[p2] = syn.page_response(date_text="June 18, 2026")
    env.feed([statement_item(guid="g1"), statement_item(path=p2, guid=None)])
    env.drive(240)
    kinds = {d.body["type"] for d in env.store.rows("DIAGNOSTIC_ONCE")}
    assert "GUID_MISSING" in kinds and "GUID_CONFLICT" not in kinds
    assert env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"  # a missing GUID never blocks zero
    before = len(env.cycles())
    env.feed([statement_item(guid="g2"), statement_item(path=p2, guid="g2")])
    env.drive(150)
    changed = env.store.rows("CYCLE_CONCLUSION")[before].body
    assert changed["result"] == "NOT_ZERO" and {"GUID_CONFLICT", "GUID_COLLISION"} <= set(changed["reasons"])
    assert len(env.store.rows("CANDIDATE")) == 2  # never merged
    assert env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"  # raised once per value


def test_unidentifiable_item_blocks_zero_until_resolved_fomc205(env):
    env.feed([{"title": EXACT, "link": "https://federalreserve.gov/x.htm", "guid": "gx"}])  # apex host
    env.drive(200)
    unid = env.store.rows("UNIDENTIFIABLE")
    assert len(unid) == 1 and set(env.cycles()) == {"NOT_ZERO"}
    env.feed([])  # rotated out
    env.drive(70)
    assert env.cycles()[-1] == "NOT_ZERO"
    assert env.collector.resolve(f"UNIDENTIFIABLE_ITEM:{unid[0].key}")["valid"]
    env.drive(70)
    assert env.cycles()[-1] == "EVENTS_OBSERVED_ZERO"
