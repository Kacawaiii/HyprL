"""Synthetic wire responses only; official dispatch requires a separate grant."""
from datetime import timedelta
import json

import pytest

from scripts.radar.core import Client, RadarError
from tests.radar.test_radar import state

SEC_FEED = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom&count=100'
TICKERS = 'https://www.sec.gov/files/company_tickers.json'
PRESS = 'https://www.federalreserve.gov/feeds/press_all.xml'
SPEECHES = 'https://www.federalreserve.gov/feeds/speeches.xml'


@pytest.fixture
def official(state, tmp_path):
    from scripts.radar.official import OfficialAuthorization
    payload = {'authorization': 'news-radar-sec-fed-telegram-v1', 'extends': 'news-radar-v1',
               'operator_signed': True, 'granted_at': '2026-10-09T00:00:00Z',
               'not_after': '2026-10-12T00:00:00Z', 'scope': {
        'https://www.sec.gov': {'methods': ['GET'], 'paths': [
            '/cgi-bin/browse-edgar (action=getcurrent, type=8-K, output=atom)',
            '/files/company_tickers.json'], 'max_requests_per_day': 300,
            'max_requests_per_second': 2, 'user_agent': 'Synthetic research contact'},
        'https://www.federalreserve.gov': {'methods': ['GET'], 'paths': [
            '/feeds/press_all.xml', '/feeds/speeches.xml'], 'max_requests_per_day': 48},
        'https://api.telegram.org': {'methods': ['POST'], 'paths': ['/bot<token>/sendMessage'],
            'recipient': "the operator's own chat (ADMIN_TELEGRAM_ID) only",
            'max_messages_per_day': 12}}}
    path = tmp_path / 'official.json'
    path.write_text(json.dumps(payload))
    return OfficialAuthorization(path), path


def test_granted_sources_have_separate_durable_limits_and_contact_header(state, official):
    store, _, clock, _ = state
    grant, _ = official
    calls = []
    def send(url, headers):
        calls.append((url, headers))
        return 200, {}, b'synthetic'
    client = Client(store, grant, send=send)
    client.get(SEC_FEED, 'sec', 'sec:8k')
    assert calls[0][1]['User-Agent'] == 'Synthetic research contact'
    with pytest.raises(RadarError, match='POLL_SPACING'):
        client.get(TICKERS, 'sec', 'sec:tickers')
    clock.at += timedelta(seconds=.5)
    client.get(TICKERS, 'sec', 'sec:tickers')
    clock.at += timedelta(seconds=1)
    with pytest.raises(RadarError, match='POLL_SPACING'):
        client.get(TICKERS, 'sec', 'sec:tickers')
    assert store.counts() == {'sec': 2}
    with pytest.raises(RadarError, match='SOURCE_OUTSIDE_GRANT'):
        client.get(PRESS, 'sec', 'wrong-group')


@pytest.mark.parametrize('url', [
    SEC_FEED.replace('type=8-K', 'type=10-K'), SEC_FEED + '&CIK=123',
    SEC_FEED + '&type=10-K', SEC_FEED + '&utm_bypass=value', SEC_FEED.replace('count=100', 'count=10000'),
    'https://data.sec.gov/submissions/CIK0000320193.json',
    'https://www.federalreserve.gov/newsevents/pressreleases/monetary.htm',
    'https://www.sec.gov/Archives/edgar/data/1/filing.htm'])
def test_supplement_never_grants_crawling_or_other_query_scope(state, official, url):
    store, _, _, _ = state
    grant, _ = official
    calls = []
    with pytest.raises(RadarError):
        Client(store, grant, send=lambda *args: calls.append(args)).get(url, 'sec', 'sec:8k')
    assert calls == [] and store.counts() == {}


# The observed current Atom feed has summary Filed/AccNo/Size, updated (no
# published), category term and a filing-index link. Items may be unavailable.
ATOM = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<title>8-K - SYNTHETIC CORP (0000000123) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/123/000000012326000001/0000000123-26-000001-index.htm"/>
<summary type="html">&lt;b&gt;Filed:&lt;/b&gt; 2026-10-10 &lt;b&gt;AccNo:&lt;/b&gt; 0000000123-26-000001 &lt;b&gt;Size:&lt;/b&gt; 12 KB</summary>
<updated>2026-10-10T10:30:00-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0000000123-26-000001</id>
</entry></feed>'''


def test_current_atom_preserves_filing_metadata_and_unknown_items(state, official):
    from scripts.radar.official import OfficialCollector
    from scripts.radar.analysis import cluster
    store, _, clock, _ = state
    clock.at += timedelta(hours=4)
    grant, _ = official
    calls = []
    def send(url, headers):
        calls.append(url)
        if url == TICKERS:
            return 200, {'ETag': '"mapping"'}, b'{"0":{"cik_str":123,"ticker":"SYN","title":"SYNTHETIC CORP"}}'
        return 200, {'ETag': '"filings"'}, ATOM
    collector = OfficialCollector(Client(store, grant, send=send), sleep=lambda seconds: setattr(clock, 'at', clock.at + timedelta(seconds=seconds)))
    collector.collect(sec=True, fed=False)
    stories = store.rows('news')
    assert len(stories) == 1 and stories[0]['symbols'] == ['SYN']
    assert stories[0]['filing']['cik'] == '0000000123'
    assert stories[0]['filing']['items_status'] == 'UNKNOWN_FEED_METADATA'
    assert stories[0]['published_at'] is None  # Atom updated is not original publication
    assert stories[0]['filing']['updated_at'] == '2026-10-10T14:30:00Z'
    event = cluster(stories, at=clock.at)[0]
    assert event['evidence']['score'] == 90
    assert calls == [TICKERS, SEC_FEED]  # no filing page crawling


def test_item_classification_uses_explicit_item_metadata_only(state):
    from scripts.radar.official import parse_sec
    _, _, clock, _ = state
    raw = ATOM.replace(b'12 KB', b'12 KB &lt;br&gt;Item 2.02: Results of Operations and Financial Condition &lt;br&gt;Item 1.01: Entry into a Material Definitive Agreement &lt;br&gt;Item 5.02: Departure of Directors or Certain Officers &lt;br&gt;Item 8.01: Other Events &lt;br&gt;Item 9.01: Financial Statements and Exhibits')
    row = parse_sec(raw, clock.at + timedelta(hours=4), {})[0]
    assert row['filing']['classifications'] == ['material_agreement', 'results', 'executive_change', 'other']
    assert row['filing']['items'] == ['1.01', '2.02', '5.02', '8.01', '9.01']
    assert 'earnings' in row['themes']


def test_conditional_feeds_and_daily_mapping_survive_restart(state, official):
    from scripts.radar.official import OfficialCollector
    store, _, clock, _ = state
    grant, _ = official
    calls = []
    def send(url, headers):
        calls.append((url, headers))
        if len(calls) == 1:
            return 200, {'etag': '"tickers"'}, b'{"0":{"cik_str":123,"ticker":"SYN","title":"SYNTHETIC CORP"}}'
        if len(calls) == 2:
            return 200, {'etag': '"8k"'}, ATOM
        assert headers['If-None-Match'] == '"8k"'
        return 304, {}, b''
    def sleep(seconds):
        clock.at += timedelta(seconds=seconds)
    OfficialCollector(Client(store, grant, send=send), sleep=sleep).collect(sec=True, fed=False)
    clock.at += timedelta(hours=1)
    from scripts.radar.core import Store
    health = OfficialCollector(Client(Store(store.root, clock=clock), grant, send=send), sleep=sleep).collect(sec=True, fed=False)
    assert len(calls) == 3 and calls[-1][0] == SEC_FEED
    assert len(store.rows('news')) == 1
    assert all(h['status'] == 'LIVE' for h in health)


def test_fed_feeds_are_primary_macro_events_and_polled_once_an_hour(state, official):
    from scripts.radar.official import OfficialCollector
    from scripts.radar.analysis import cluster
    store, _, clock, _ = state
    grant, _ = official
    calls = []
    raw = b'''<rss version="2.0"><channel><item><title>Synthetic speech on payment systems</title>
    <link>https://www.federalreserve.gov/newsevents/speech/synthetic.htm</link>
    <description>Payment system update</description><pubDate>Sat, 10 Oct 2026 10:00:00 GMT</pubDate></item></channel></rss>'''
    def send(url, headers):
        calls.append(url)
        return 200, {}, raw
    collector = OfficialCollector(Client(store, grant, send=send))
    collector.collect(sec=False)
    collector.collect(sec=False)
    assert len(calls) == 2
    event = cluster(store.rows('news'), at=clock.at)[0]
    assert event['evidence']['score'] == 90 and 'macro' in event['entities']['themes']
    assert event['importance'] >= 80 and event['transmission_hypotheses']


def test_two_accessions_for_same_issuer_remain_two_filing_events(state):
    from scripts.radar.official import parse_sec
    from scripts.radar.analysis import cluster
    _, _, clock, _ = state
    first = parse_sec(ATOM, clock.at, {})[0]
    second = parse_sec(ATOM.replace(b'000000012326000001', b'000000012326000002').replace(b'0000000123-26-000001', b'0000000123-26-000002'), clock.at, {})[0]
    assert len(cluster([first, second], at=clock.at)) == 2


def test_missing_ticker_scope_is_never_dispatched_and_later_mapping_is_causal(state, official):
    from scripts.radar.official import OfficialAuthorization, OfficialCollector
    store, _, clock, _ = state
    grant, path = official
    payload = json.loads(path.read_text())
    payload['scope']['https://www.sec.gov']['paths'].remove('/files/company_tickers.json')
    path.write_text(json.dumps(payload))
    calls = []
    def send(url, headers):
        calls.append(url)
        if url == TICKERS:
            return 200, {}, b'{"0":{"cik_str":123,"ticker":"ETH","title":"SYNTHETIC CORP"}}'
        if len(calls) == 1:
            return 200, {'etag': '"filing"'}, ATOM
        return 304, {}, b''
    def sleep(seconds):
        clock.at += timedelta(seconds=seconds)
    health = OfficialCollector(Client(store, OfficialAuthorization(path), send=send), sleep=sleep).collect(fed=False)
    assert calls == [SEC_FEED] and health[0]['reason'] == 'PATH_OUTSIDE_GRANT'
    original = store.rows('news')[0]
    assert original['symbols'] == []
    payload['scope']['https://www.sec.gov']['paths'].append('/files/company_tickers.json')
    path.write_text(json.dumps(payload))
    clock.at += timedelta(hours=1)
    OfficialCollector(Client(store, OfficialAuthorization(path), send=send), sleep=sleep).collect(fed=False)
    enriched = store.rows('news')[-1]
    assert enriched['symbols'] == ['ETH'] and enriched['received_at'] > original['received_at']
    from scripts.radar.entities import Dictionary
    assert Dictionary().map(enriched)['symbols'] == ['ETH']  # equity ticker, not ETH/USD


def test_expired_supplement_and_lower_provider_budget_stop_before_dispatch(state, official):
    store, _, clock, _ = state
    grant, _ = official
    calls = []
    client = Client(store, grant, send=lambda *args: (calls.append(args) or 200, {}, b'synthetic'))
    grant.payload['budgets']['fed'] = 1
    client.get(PRESS, 'fed', 'fed:press')
    with pytest.raises(RadarError, match='DAILY_BUDGET'):
        client.get(SPEECHES, 'fed', 'fed:speeches')
    clock.at += timedelta(days=3)
    with pytest.raises(RadarError, match='EXPIRED'):
        client.get(SEC_FEED, 'sec', 'sec:8k')
    assert len(calls) == 1
