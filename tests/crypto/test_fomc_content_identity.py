"""FOMC_CONTENT_IDENTITY_V2 / FOMC_CANON_V2 (spec revision 25, F82-F83, FOMC251-FOMC258): raw integrity
and content identity are distinct. Synthetic pages carry representative Cloudflare spans; the real
Cloudflare bytes stay local (redistribution.raw_storage) and are checked by the private proof at the end,
which runs only where the local fixtures exist."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

from scripts.trading_lab.fomc import canon, snapshot, spec, state
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.store import FomcStore

from tests.crypto.fomc_support import P1, SID1, Env, statement_item

URL1 = syn.url(P1)


@pytest.fixture(autouse=True)
def synthetic_challenge_script(monkeypatch):
    monkeypatch.setattr(canon, "CHALLENGE_SCRIPT_SHA256", syn.CF_SCRIPT_SHA256)


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.provider.close()


def page(key=7, ray="0123456789abcdef", stamp=1790879001, **kw):
    kw.setdefault("page_url", URL1)
    return syn.cloudflare_html(key=key, ray=ray, stamp=stamp, **kw)


# ------------------------------------------------------------------ the canonicalizer ---------------
def test_only_the_three_spans_change_and_the_effective_address_stays():
    a, b = page(key=7), page(key=201, ray="fedcba9876543210", stamp=1790879999)
    ca, cb = canon.canonicalize(a), canon.canonicalize(b)
    assert a != b and ca.status == cb.status == "CANONICAL"
    assert ca.content_sha256 == cb.content_sha256 and ca.canonical == cb.canonical
    assert ca.neutralized == {"CF_EMAIL_LINK": 2, "CF_EMAIL_SPAN": 1, "CF_CHALLENGE_PARAMS": 1}
    assert syn.obfuscate("media@frb.gov", 0).encode() in ca.canonical  # re-keyed with key 0, not removed
    assert b"r:'',t:''" in ca.canonical and b"0123456789abcdef" not in ca.canonical
    # every other byte is kept: removing the three replaced values from both gives the same bytes
    assert len(ca.canonical) == len(a) - len("0123456789abcdef") - len("MTc5MDg3OTAwMQ==")


@pytest.mark.parametrize("change", [
    dict(body="The Committee decided to lower the target range."),  # the text
    dict(body="The Committee decided to maintain the target range at 4 to 4-1/4 percent."),  # a rate
    dict(date_text="June 18, 2026"),  # the date
    dict(title="Federal Reserve issues FOMC statement on longer-run goals"),  # the title
    dict(release="For release at 2:30 p.m. EDT"),  # the release line
    dict(email="press@frb.gov"),  # the effective address behind the obfuscation
    dict(page_url=syn.url(syn.statement_path("20260618"))),  # the share link's effective body
])
def test_any_real_change_next_to_cloudflare_bytes_is_detected(change):
    base = dict(body="The Committee decided to maintain the target range at 4-1/4 to 4-1/2 percent.")
    first = canon.canonicalize(page(key=11, **base))
    changed = canon.canonicalize(page(key=99, ray="aaaaaaaaaaaaaaaa", **{**base, **change}))
    assert first.status == changed.status == "CANONICAL"
    assert first.content_sha256 != changed.content_sha256


@pytest.mark.parametrize("mutate, reason", [
    (lambda d: d.replace(b"The Committee decided", b"See /cdn-cgi/l/email-protection#0a0b0c. The Committee decided"),
     "email-protection outside an allowed HTML context"),  # a false marker in the text
    (lambda d: d.replace(b"The Committee decided", b"window.__CF$cv$params={r:'x'} The Committee decided"),
     "__CF$cv$params outside an allowed HTML context"),
    (lambda d: d.replace(b'<a class="shareDL__link" href', b'<link class="shareDL__link" href'),
     "email-protection outside an allowed HTML context"),  # not an <a> start tag
    (lambda d: d.replace(b'data-cfemail="', b'data-cfemail="0', 1), "obfuscated value of odd length"),
    (lambda d: d.replace(b"document.cf=p;", b"document.cf=p;fetch(p);"), "challenge parameters outside the known Cloudflare script"),
    (lambda d: d.replace(b"</script></body>", b"</script>\n<p>after</p></body>"), "challenge parameters outside the known Cloudflare script"),
    (lambda d: d + b"\xff\xfe", "not strict UTF-8"),
])
def test_unknown_or_ambiguous_structure_is_refused_and_never_merges(mutate, reason):
    one, other = mutate(page(key=7)), mutate(page(key=8, ray="1111111111111111"))
    first, second = canon.canonicalize(one), canon.canonicalize(other)
    assert first.domain == "RAW_FALLBACK" and first.reason == reason
    assert first.bytes_sha256 == spec.sha256_bytes(one) and first.canonical == one  # raw bytes, fallback domain
    assert first.content_sha256 == canon.identity("RAW_FALLBACK", spec.sha256_bytes(one))
    assert first.content_sha256 != second.content_sha256  # the same content with other keys does not merge


def test_odd_values_non_printable_addresses_and_bounds_are_refused():
    bad_hex = syn.cloudflare_html(key=7, ray="0123456789abcdef", stamp=1, page_url=URL1).replace(
        syn.obfuscate("media@frb.gov", 7 ^ 0x33).encode(), syn.obfuscate("media\x01frb.gov", 7 ^ 0x33).encode())
    assert canon.canonicalize(bad_hex).reason == "obfuscated value does not decode to printable ASCII"
    extra = ('<a href="/cdn-cgi/l/email-protection#' + syn.obfuscate("x@y.z", 9) + '">x</a>').encode()
    many = page().replace(b'<div id="lastUpdate">', extra * 15 + b'<div id="lastUpdate">')
    assert canon.canonicalize(many).reason == "bound exceeded"
    assert canon.canonicalize(page()).status == "CANONICAL"  # the unchanged page: control
    odd = page().replace(b'data-cfemail="', b'data-cfemail="f', 1)
    assert canon.canonicalize(odd).reason == "obfuscated value of odd length"
    no_spans = syn.statement_html()
    plain = canon.canonicalize(no_spans)
    assert plain.domain == "CANONICAL" and plain.canonical == no_spans and plain.bytes_sha256 == spec.sha256_bytes(no_spans)
    assert plain.content_sha256 == canon.identity("CANONICAL", spec.sha256_bytes(no_spans)) != canon.identity(
        "RAW_FALLBACK", spec.sha256_bytes(no_spans))  # the domain is part of the identity


# ------------------------------------------------------------------ revisions end to end ------------
def _revisions(env):
    return env.store.rows("REVISION")


def _links(env):
    return env.store.rows("LINK")


def test_new_cloudflare_bytes_make_one_revision_with_one_observation_each_fomc251(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.cloudflare_route(page_url=URL1)
    env.drive(900, idle=60)  # acquisition and the O300 recheck
    records = state.primary_responses(env.store, SID1)
    assert len(records) >= 2 and len({r.body["raw_sha"] for r in records}) == len(records)  # distinct raws
    assert len(_revisions(env)) == 1
    links = _links(env)
    assert len(links) == len(records) and len({l.body["content_sha256"] for l in links}) == 1
    assert [l.body["raw_artifact_identities_and_hashes"][0]["raw_sha256"] for l in links] == [r.body["raw_sha"] for r in records]
    assert all(l.body["canonicalization"]["domain"] == "CANONICAL" for l in links)
    primary = [state.processing_outcome(env.store, r.seq).body["outcome"] for r in records]
    assert primary[0] == "NORMALIZED_REVISION_COMMITTED" and set(primary[1:]) == {"NORMALIZED_SAME_CONTENT_NO_NEW_REVISION"}
    revision = _revisions(env)[0].body
    assert revision["first_raw_sha256"] == records[0].body["raw_sha"] and revision["content_hash"] == links[0].body["content_sha256"]


def test_aba_reuses_a_across_cloudflare_bytes_fomc254(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.cloudflare_route(page_url=URL1, body="A")
    env.drive(240)
    env.provider.routes[P1] = syn.cloudflare_route(page_url=URL1, body="B")
    env.drive(900, idle=60)
    env.provider.routes[P1] = syn.cloudflare_route(page_url=URL1, body="A")
    env.drive(3600, idle=300)
    env.drive(130)
    item = next(i for i in snapshot.events_as_of(env.store, env.clock.true)["items"] if i["sid"] == SID1)
    revisions = {r.key: r.body for r in _revisions(env)}
    assert len(revisions) == 2 and item["step"] == 4 and item["state"] == "CURRENT_REVISION"
    first_a = _links(env)[0].body["revision"]
    assert item["revision"] == first_a and len({l.body["raw_artifact_identities_and_hashes"][0]["raw_sha256"] for l in _links(env)}) == 3


def test_backfill_then_live_one_revision_live_only_through_the_live_link_fomc255(env):
    env.feed([])
    env.provider.routes[P1] = syn.cloudflare_route(page_url=URL1)
    env.collector.submit_manifest(json.dumps({"version": 1, "urls": [URL1]}).encode(), "operator")
    env.drive(240)
    env.drive(130)
    backfill = next(i for i in snapshot.events_as_of(env.store, env.clock.true)["items"] if i["sid"] == SID1)
    assert backfill["state"] == "CURRENT_REVISION" and backfill["live_available"] is False
    env.feed([statement_item()])
    env.drive(300)
    live = next(i for i in snapshot.events_as_of(env.store, env.clock.true)["items"] if i["sid"] == SID1)
    assert len(_revisions(env)) == 1 and live["revision"] == backfill["revision"] and live["live_available"] is True
    modes = [l["observation_mode"] for l in live["links"]]
    assert modes[0] == "HISTORICAL_BACKFILL" and "LIVE" in modes
    raws = {l["raw_artifact_identities_and_hashes"][0]["raw_sha256"] for l in live["links"]}
    assert len(raws) == len(live["links"])  # each observation keeps its own raw


def test_a_page_rejected_on_its_original_document_stays_rejected(env):
    env.feed([statement_item()])
    duplicated = page().replace(b'<h3 class="title">', b'<h3 class="title">Federal Reserve issues FOMC statement</h3><h3 class="title">', 1)
    env.provider.routes[P1] = syn.SyntheticResponse(body=duplicated, headers=list(syn.HTML_HEADERS))
    env.drive(200)
    assert "PARSER_FAILED" in env.outcomes() and _revisions(env) == []


def test_a_false_marker_page_keeps_raw_identity_end_to_end(env):
    env.feed([statement_item()])
    def route(count):
        body = page(key=count + 3).replace(b"The Committee decided", b"Mail /cdn-cgi/l/email-protection#0a0b0c. The Committee decided")
        return syn.SyntheticResponse(body=body, headers=list(syn.HTML_HEADERS))
    env.provider.routes[P1] = route
    env.drive(900, idle=60)
    links = _links(env)
    assert len(links) >= 2 and all(l.body["canonicalization"]["domain"] == "RAW_FALLBACK" for l in links)
    assert len(_revisions(env)) == len(links)  # never merged silently
    assert all(l.body["canonicalization"]["bytes_sha256"] == l.body["raw_artifact_identities_and_hashes"][0]["raw_sha256"]
               for l in links)


def test_a_corrupt_raw_fails_closed_despite_an_equal_content_identity_fomc256(env):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.cloudflare_route(page_url=URL1)
    env.drive(900, idle=60)
    env.drive(130)
    T, H = env.clock.true, env.store.horizon()
    snap = snapshot.events_as_of(env.store, T, H)
    newest = state.primary_responses(env.store, SID1)[-1]
    link = env.store.rows("LINK", key=str(newest.seq))[0].body
    assert link["content_sha256"] == next(i for i in snap["items"] if i["sid"] == SID1)["content_hash"]
    raw = env.root / "raw" / newest.body["raw_sha"][:2] / newest.body["raw_sha"]
    raw.write_bytes(page(key=250))  # other bytes, same content identity
    assert canon.canonicalize(raw.read_bytes()).content_sha256 == link["content_sha256"]
    with pytest.raises(snapshot.SnapshotFailed):
        snapshot.events_as_of(env.store, T, H)
    with pytest.raises(snapshot.ReplayFailed):
        snapshot.replay(env.store, T, H)


def test_the_same_t_h_after_reopening_and_replay_has_the_same_identity(env, tmp_path):
    env.feed([statement_item()])
    env.provider.routes[P1] = syn.cloudflare_route(page_url=URL1)
    env.drive(900, idle=60)
    env.drive(130)
    T, H = env.clock.true, env.store.horizon()
    snap = snapshot.events_as_of(env.store, T, H)
    env.collector.close()
    env.store.close()
    reopened = FomcStore(env.root, wall_clock=env.clock.wall)
    try:
        assert snapshot.events_as_of(reopened, T, H) == snap and snapshot.replay(reopened, T, H) == snap
    finally:
        reopened.close()
    # a copy whose stored identity was altered no longer re-derives: replay fails closed
    copy = tmp_path / "copy"
    shutil.copytree(env.root, copy)
    import sqlite3
    with sqlite3.connect(copy / "fomc.sqlite3") as conn:
        conn.execute("UPDATE rec SET body = json_set(body, '$.content_sha256', '0') WHERE kind = 'LINK' "
                     "AND commit_seq = (SELECT MAX(commit_seq) FROM rec WHERE kind = 'LINK')")
    tampered = FomcStore(copy, wall_clock=env.clock.wall)
    try:
        with pytest.raises(snapshot.ReplayFailed):
            snapshot.replay(tampered, T, H)
    finally:
        tampered.close()


def test_the_frozen_challenge_digest_is_the_one_in_the_spec(monkeypatch):
    monkeypatch.undo()
    content = json.loads(spec.SPEC_PATH.read_text(encoding="utf-8"))["revision_policy"]["content_identity"]
    assert canon.CHALLENGE_SCRIPT_SHA256 in content["spans"]["CF_CHALLENGE_PARAMS"]
    assert content["canonicalizer_id"] == canon.CANONICALIZER_ID and content["id"] == spec.CONTENT_IDENTITY_ID


# ------------------------------------------------------------------ private proof on official bytes ---
FIXTURES = Path(os.environ.get("FOMC_PRIVATE_FIXTURES", "/home/kyo/fomc-pilot/fixtures-v1"))


@pytest.mark.skipif(not FIXTURES.exists(), reason="official fixtures are local only (redistribution.raw_storage)")
def test_private_official_fixtures_canonicalize_with_the_frozen_digest(monkeypatch):
    monkeypatch.undo()
    for name, spans in [("summer_edt", {"CF_EMAIL_LINK": 2, "CF_EMAIL_SPAN": 1, "CF_CHALLENGE_PARAMS": 1}),
                        ("winter_est", {"CF_EMAIL_LINK": 2, "CF_EMAIL_SPAN": 1, "CF_CHALLENGE_PARAMS": 1}),
                        ("immediate_release", {"CF_EMAIL_LINK": 1, "CF_EMAIL_SPAN": 0, "CF_CHALLENGE_PARAMS": 1})]:
        body = (FIXTURES / name / "body.html").read_bytes()
        result = canon.canonicalize(body)
        assert result.domain == "CANONICAL" and result.neutralized == spans, (name, result.reason)
        from scripts.trading_lab.fomc import parsing
        original, canonical = parsing.parse_primary(body), parsing.parse_primary(result.canonical)
        assert (original.titles, original.dates, original.release_segments) == \
            (canonical.titles, canonical.dates, canonical.release_segments)
        tmp = shutil.copy  # noqa: F841 - nothing is written back; the fixtures stay as acquired


# ------------------------------------------------------------------ the two counterexamples (rev 25) ---
def _title_attr(key):  # a marker inside another attribute's value: not an href
    value = syn.obfuscate("media@frb.gov", key)
    return page(key=key).replace(b'<div id="lastUpdate">',
                                 b"<a title='href=\"/cdn-cgi/l/email-protection#" + value.encode() + b"\"'>texte</a>"
                                 + b'<div id="lastUpdate">')


def _textarea(key):  # a span inside a raw-text element: not markup
    value = syn.obfuscate("media@frb.gov", key)
    return page(key=key).replace(b'<div id="lastUpdate">',
                                 b'<textarea><span class="__cf_email__" data-cfemail="' + value.encode()
                                 + b'">[email&#160;protected]</span></textarea><div id="lastUpdate">')


def _comment(key):  # a link inside a comment
    value = syn.obfuscate("media@frb.gov", key)
    return page(key=key).replace(b'<div id="lastUpdate">',
                                 b'<!-- <a href="/cdn-cgi/l/email-protection#' + value.encode() + b'">x</a> --><div id="lastUpdate">')


@pytest.mark.parametrize("build, reason", [
    (_title_attr, "email-protection outside an allowed HTML context"),
    (_textarea, "__cf_email__ outside an allowed HTML context"),
    (_comment, "email-protection outside an allowed HTML context"),
])
def test_cloudflare_markers_in_false_html_contexts_never_merge(build, reason):
    one, other = build(7), build(201)  # two XOR keys encoding the same address in a false context
    first, second = canon.canonicalize(one), canon.canonicalize(other)
    assert first.domain == second.domain == "RAW_FALLBACK" and first.reason == reason
    assert first.content_sha256 != second.content_sha256
    assert canon.canonicalize(build(7)).content_sha256 == first.content_sha256  # identical raws meet again


def test_recanonicalizing_canonical_bytes_never_shares_the_identity():
    result = canon.canonicalize(page(key=7))
    again = canon.canonicalize(result.canonical)  # r and t empty: no longer the known challenge parameters
    assert result.domain == "CANONICAL" and again.domain == "RAW_FALLBACK"
    assert again.bytes_sha256 == result.bytes_sha256 and again.content_sha256 != result.content_sha256


@pytest.mark.parametrize("build", [_title_attr, _textarea])
def test_false_contexts_end_to_end_keep_separate_revisions_and_the_read_follows(env, build):
    env.feed([statement_item()])
    env.provider.routes[P1] = lambda count: syn.SyntheticResponse(body=build(count + 3), headers=list(syn.HTML_HEADERS))
    env.drive(900, idle=60)
    env.drive(130)
    links = _links(env)
    assert len(links) >= 2 and len(_revisions(env)) == len(links)  # classified, never merged
    assert all(l.body["canonicalization"]["domain"] == "RAW_FALLBACK" for l in links)
    item = next(i for i in snapshot.events_as_of(env.store, env.clock.true)["items"] if i["sid"] == SID1)
    newest = _revisions(env)[-1]
    assert item["state"] == "CURRENT_REVISION" and item["revision"] == newest.key  # the newest observation's own revision
