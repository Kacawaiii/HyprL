"""All provider responses are synthetic. These tests never access the network."""
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit
import json
import sqlite3

import pytest

from scripts.radar.analysis import (anomalies, bar_map, cluster, daily_observations, paper_followup, priced_in,
                                    regime, valid_bars)
from scripts.radar.core import Authorization, Client, RadarError, Store, iso
from scripts.radar.deploy import unit_texts
from scripts.radar.digest import command, render, summarize, validate
from scripts.radar.entities import Dictionary
from scripts.radar.registry import FEEDS
from scripts.radar.service import radar
from scripts.radar.sources import Collector, item, parse_feed
from scripts.radar.synthetic import seed

UTC = timezone.utc
AT = datetime(2026, 10, 10, 11, 30, tzinfo=UTC)


class Clock:
    def __init__(self, at=AT):
        self.at = at

    def __call__(self):
        return self.at


@pytest.fixture
def state(tmp_path):
    clock = Clock()
    store = Store(tmp_path / 'private', clock=clock)
    payload = {'authorization': 'news-radar-v1', 'operator_signed': 'synthetic operator',
               'granted_at': '2026-10-09T00:00:00Z', 'not_after': '2026-10-12T00:00:00Z',
               'scope': {origin: {'methods': ['GET'], 'path_prefixes': ['/']} for origin in [
                   'https://example.invalid', 'https://data.alpaca.markets',
                   'https://api.gdeltproject.org', 'https://www.youtube.com']},
               'budgets': {'alpaca': 400, 'gdelt': 60, 'rss': 300, 'youtube': 400, 'market': 24, 'llm': 2},
               'llm': {'cli': 'claude', 'model': 'sonnet', 'tools': [], 'max_calls_per_day': 2}}
    path = tmp_path / 'authorization.json'
    path.write_text(json.dumps(payload))
    grant = Authorization(path)
    return store, grant, clock, path


@pytest.fixture
def synthetic_credentials(tmp_path):
    path = tmp_path / 'synthetic.env'
    path.write_text('APCA_API_KEY_ID=synthetic-id\nAPCA_API_SECRET_KEY=synthetic-secret\n')
    return path


def story(slug='one', headline='Micron raises memory guidance after earnings', publisher='wire',
          symbols=(), retail=False, received=AT, published=AT-timedelta(hours=1), primary=False):
    return item('synthetic', publisher, 'https://example.invalid/' + slug, headline, '',
                iso(published) if published else None, received, symbols=symbols, retail=retail, primary=primary)


def bar(at, close=100, volume=1000):
    return {'t': iso(at), 'o': close, 'h': close + 2, 'l': close - 2, 'c': close, 'v': volume, 'n': 100, 'vw': close}


def daily_history():
    return [bar((AT - timedelta(days=i)).replace(hour=0, minute=0), close=100) for i in range(40, 0, -1)]


def test_missing_authorization_is_fail_closed(tmp_path):
    with pytest.raises(RadarError, match='AUTHORIZATION_MISSING'):
        Authorization(tmp_path / 'absent')


def test_unsigned_grant_rejected(state):
    _, _, _, path = state
    payload = json.loads(path.read_text())
    payload['operator_signed'] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(RadarError):
        Authorization(path)


@pytest.mark.parametrize('url', ['https://sec.gov/feed', 'https://data.sec.gov/submissions', 'https://www.federalreserve.gov/feed'])
def test_disabled_official_sources_even_if_scope_added(state, url):
    _, grant, _, _ = state
    grant.payload['scope']['https://' + urlsplit(url).netloc] = {'methods': ['GET'], 'path_prefixes': ['/']}
    with pytest.raises(RadarError, match='OFFICIAL_SOURCE_DISABLED'):
        grant.permit(url)


@pytest.mark.parametrize('url', ['http://example.invalid/feed', 'https://user:secret@example.invalid/feed', 'https://example.invalid:8443/feed', 'https://elsewhere.invalid/feed'])
def test_non_public_or_out_of_scope_urls_denied_before_send(state, url):
    store, grant, _, _ = state
    calls = []
    client = Client(store, grant, send=lambda *a: calls.append(a))
    with pytest.raises(RadarError):
        client.get(url, 'rss', 'feed')
    assert not calls and not store.counts()


def test_expiry_rechecked_for_every_dispatch(state):
    store, grant, clock, _ = state
    calls = []
    clock.at = datetime(2026, 10, 12, tzinfo=UTC)
    with pytest.raises(RadarError, match='EXPIRED'):
        Client(store, grant, send=lambda *a: calls.append(a)).get('https://example.invalid/feed', 'rss', 'feed')
    assert not calls


def test_append_only_store_and_dispatch(state):
    store, grant, _, _ = state
    store.append('news', 'a', {'synthetic': True})
    store.reserve(grant, 'rss', 'feed', 300, 900)
    with store.connect() as db:
        for sql in ('DELETE FROM evidence', 'UPDATE evidence SET key="b"', 'DELETE FROM dispatch', 'UPDATE dispatch SET day="tomorrow"'):
            with pytest.raises(sqlite3.IntegrityError, match='append only'):
                db.execute(sql)
    assert store.root.stat().st_mode & 0o777 == 0o700
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_restart_preserves_budget_and_per_feed_spacing(state):
    store, grant, clock, _ = state
    grant.payload['budgets']['rss'] = 2
    store.reserve(grant, 'rss', 'first', 300, 900)
    restarted = Store(store.root, clock=clock)
    with pytest.raises(RadarError, match='POLL_SPACING'):
        restarted.reserve(grant, 'rss', 'first', 300, 900)
    restarted.reserve(grant, 'rss', 'second', 300, 900)
    clock.at += timedelta(minutes=16)
    with pytest.raises(RadarError, match='DAILY_BUDGET'):
        restarted.reserve(grant, 'rss', 'first', 300, 900)


def test_failed_transport_consumes_reservation(state):
    store, grant, _, _ = state
    def send(*args):
        raise RadarError('HTTP_UNAVAILABLE')
    with pytest.raises(RadarError):
        Client(store, grant, send=send).get('https://example.invalid/feed', 'rss', 'feed')
    assert store.counts()['rss'] == 1


@pytest.mark.parametrize('hour,minute,allowed', [(11, 44, True), (11, 45, False), (12, 20, False), (12, 29, False), (12, 30, True)])
def test_gdelt_trader_window(state, hour, minute, allowed):
    store, grant, clock, _ = state
    clock.at = AT.replace(hour=hour, minute=minute)
    if allowed:
        store.reserve(grant, 'gdelt', 'energy', 60, 7)
    else:
        with pytest.raises(RadarError, match='GDELT_TRADER_WINDOW'):
            store.reserve(grant, 'gdelt', 'energy', 60, 7)
        assert not store.counts()


def test_gdelt_spacing_shared_by_all_themes(state):
    store, grant, clock, _ = state
    store.reserve(grant, 'gdelt', 'energy', 60, 7)
    clock.at += timedelta(seconds=6.9)
    with pytest.raises(RadarError, match='POLL_SPACING'):
        store.reserve(grant, 'gdelt', 'crypto', 60, 7)
    clock.at += timedelta(seconds=.1)
    store.reserve(grant, 'gdelt', 'crypto', 60, 7)


def test_429_stops_group_across_restarts(state):
    store, grant, _, _ = state
    calls = []
    def send(*args):
        calls.append(args)
        return 429, {'Retry-After': '7200'}, b'secret body is never stored'
    client = Client(store, grant, send=send)
    with pytest.raises(RadarError, match='429'):
        client.get('https://example.invalid/feed', 'rss', 'first')
    with pytest.raises(RadarError, match='COOLDOWN'):
        Client(Store(store.root, clock=store.clock), grant, send=send).get('https://example.invalid/other', 'rss', 'other')
    assert len(calls) == 1
    assert 'secret body' not in store.path.read_bytes().decode(errors='ignore')


RSS = b'''<rss version="2.0"><channel><item><title>Micron guidance</title><link>https://example.invalid/story?utm_source=x</link><description>Short summary</description><pubDate>Sat, 10 Oct 2026 10:00:00 GMT</pubDate><content:encoded xmlns:content="urn:content">PRIVATE FULL ARTICLE</content:encoded></item></channel></rss>'''


def test_rss_uses_publication_and_receipt_without_full_content():
    rows = parse_feed(RSS, 'rss', 'wire', AT)
    assert len(rows) == 1
    assert rows[0]['published_at'] == '2026-10-10T10:00:00Z'
    assert rows[0]['received_at'] == iso(AT)
    assert rows[0]['summary'] == 'Short summary'
    assert rows[0]['url'] == 'https://example.invalid/story'
    assert 'FULL ARTICLE' not in json.dumps(rows)


def test_atom_video_only_titles_descriptions_and_published_time():
    raw = b'''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:media="http://search.yahoo.com/mrss/"><entry><title>Bitcoin narrative</title><link rel="alternate" href="https://www.youtube.com/watch?v=synthetic"/><published>2026-10-10T10:00:00Z</published><updated>2026-10-10T11:00:00Z</updated><media:group><media:description>Retail summary</media:description></media:group></entry></feed>'''
    row = parse_feed(raw, 'youtube', 'channel', AT, retail=True)[0]
    assert row['retail'] and row['summary'] == 'Retail summary'
    assert row['published_at'] == '2026-10-10T10:00:00Z'


def test_channel_metadata_id_wins_over_related_video_ids(state):
    store, grant, _, _ = state
    channel_id = 'UC' + 'a' * 22
    related = 'UC' + 'b' * 22
    calls = []
    def send(url, headers):
        calls.append(url)
        if '/@' in url:
            body = json.dumps({'channelMetadataRenderer': {'externalId': channel_id},
                               'relatedVideoRenderer': {'channelId': related}}).encode()
            return 200, {}, body
        return 200, {}, RSS
    collector = Collector(Client(store, grant, send=send), channels=[])
    assert collector.channel({'handle': '@Synthetic'}) == 1
    assert parse_qs(urlsplit(calls[-1]).query)['channel_id'] == [channel_id]
    assert store.latest('channel', '@Synthetic')['channel_id'] == channel_id


def test_ambiguous_channel_discovery_never_guesses_an_id(state):
    store, grant, _, _ = state
    body = json.dumps({'first': {'channelId': 'UC' + 'a' * 22}, 'second': {'channelId': 'UC' + 'b' * 22}}).encode()
    collector = Collector(Client(store, grant, send=lambda *a: (200, {}, body)), channels=[])
    with pytest.raises(RadarError, match='CHANNEL_ID_UNVERIFIED'):
        collector.channel({'handle': '@Synthetic'})
    assert store.latest('channel', '@Synthetic') is None


@pytest.mark.parametrize('published', [None, '2026-10-11T00:00:00Z', '2026-10-10T10:00:00', 'nonsense'])
def test_unknown_or_future_publication_not_backfilled(published):
    row = item('synthetic', 'wire', 'https://example.invalid/story', 'headline', '', published, AT)
    assert row['published_at'] is None


@pytest.mark.parametrize('raw', [b'<!DOCTYPE rss><rss/>', b'<!ENTITY x "y"><rss/>', b'<html>error</html>', b'<rss>'])
def test_feed_parser_rejects_unsafe_xml_and_html(raw):
    with pytest.raises(RadarError):
        parse_feed(raw, 'rss', 'wire', AT)


def test_conditional_get_and_verified_304(state):
    store, grant, clock, _ = state
    sent = []
    def send(url, headers):
        sent.append(headers)
        return (200, {'ETag': 'synthetic-etag', 'Last-Modified': 'synthetic-date'}, RSS) if len(sent) == 1 else (304, {}, b'')
    collector = Collector(Client(store, grant, send=send), channels=[])
    assert collector.feed('rss:first', 'https://example.invalid/feed', 'wire') == 1
    clock.at += timedelta(minutes=15)
    assert collector.feed('rss:first', 'https://example.invalid/feed', 'wire') == 0
    assert sent[1]['If-None-Match'] == 'synthetic-etag'
    assert sent[1]['If-Modified-Since'] == 'synthetic-date'
    assert len(store.rows('news')) == 1


def test_unverified_304_not_reported_live(state):
    store, grant, _, _ = state
    collector = Collector(Client(store, grant, send=lambda *a: (304, {}, b'')), channels=[])
    collector.attempt('rss:x', lambda: collector.feed('rss:x', 'https://example.invalid/feed', 'wire'))
    assert collector.health[0]['status'] == 'BLOCKED'


def test_dead_feeds_are_not_polled_again(state):
    store, grant, _, _ = state
    store.append('health', 'rss:' + FEEDS[0][0], {'source': 'rss:' + FEEDS[0][0], 'status': 'DEAD', 'reason': 'HTTP_404', 'items': 0})
    seen = []
    collector = Collector(Client(store, grant, send=lambda *a: seen.append(a)), channels=[])
    collector.collect(markets=False, gdelt=False)
    assert not seen  # all other hosts are outside this synthetic grant
    assert any(h['status'] == 'DEAD' for h in collector.health)


@pytest.mark.parametrize('code', ['REDIRECT_REQUIRES_EXPLICIT_URL', 'HTTP_301', 'HTTP_302', 'HTTP_303', 'HTTP_307', 'HTTP_308'])
def test_redirected_feeds_are_dropped_without_a_second_dispatch(state, code):
    store, grant, clock, _ = state
    sent = []
    def send(url, headers):
        sent.append(url)
        raise RadarError(code)
    collector = Collector(Client(store, grant, send=send), channels=[])
    source = 'rss:' + FEEDS[0][0]
    collector.attempt(source, lambda: collector.feed(source, 'https://example.invalid/feed', 'wire'))
    assert collector.health[0]['status'] == 'DEAD'
    assert collector.health[0]['reason'] == code
    clock.at += timedelta(hours=1)
    collector.collect(markets=False, gdelt=False)
    assert sent == ['https://example.invalid/feed']
    assert store.counts()['rss'] == 1


def test_gdelt_seen_time_never_becomes_publication(state):
    store, grant, _, _ = state
    payload = {'articles': [{'url': 'https://example.invalid/event', 'title': 'Brazil election',
                             'seendate': '20261010T100000Z', 'domain': 'example.invalid'}]}
    collector = Collector(Client(store, grant, send=lambda *a: (200, {}, json.dumps(payload).encode())), channels=[])
    collector.gdelt('elections', 'Brazil')
    row = store.rows('news')[0]
    assert row['published_at'] is None and row['discovery_at'] == '20261010T100000Z'


def test_alpaca_class_symbols_map_at_the_provider_boundary(state, synthetic_credentials):
    store, grant, _, _ = state
    requests = []
    def send(url, headers):
        requests.append(parse_qs(urlsplit(url).query)['symbols'][0])
        return 200, {}, json.dumps({'bars': {'BRK.B': daily_history(), 'BF.B': daily_history()}, 'next_page_token': None}).encode()
    collector = Collector(Client(store, grant, send=send, credentials=synthetic_credentials), channels=[])
    assert collector.bars(['BRK-B', 'BF-B']) == 80
    assert requests == ['BRK.B,BF.B']
    assert {r['symbol'] for r in store.rows('daily_bars')} == {'BRK-B', 'BF-B'}


def test_empty_alpaca_bars_do_not_verify_price_coverage(state, synthetic_credentials):
    store, grant, _, _ = state
    collector = Collector(Client(store, grant, credentials=synthetic_credentials, send=lambda *a: (200, {}, b'{"bars":{},"next_page_token":null}')), channels=[])
    collector.attempt('alpaca:crypto_daily:SOL', lambda: collector.bars(['SOL/USD'], crypto=True))
    assert collector.health[0]['status'] == 'BLOCKED'
    assert collector.health[0]['reason'] == 'EMPTY_BARS'


def test_alpaca_unrequested_bar_identity_is_rejected(state, synthetic_credentials):
    store, grant, _, _ = state
    payload = {'bars': {'QQQ': daily_history()}, 'next_page_token': None}
    collector = Collector(Client(store, grant, credentials=synthetic_credentials, send=lambda *a: (200, {}, json.dumps(payload).encode())), channels=[])
    with pytest.raises(RadarError, match='BARS_SYMBOL_MISMATCH'):
        collector.bars(['SPY'])
    assert not store.rows('daily_bars')


def test_alpaca_all_symbols_pagination_and_source_shape(state, tmp_path):
    store, grant, _, _ = state
    keys = tmp_path / 'synthetic.env'
    keys.write_text('APCA_API_KEY_ID=synthetic-key\nAPCA_API_SECRET_KEY=synthetic-secret\n')
    calls = []
    def send(url, headers):
        calls.append((url, headers))
        article = {'id': len(calls), 'headline': 'Micron guidance', 'summary': 'Synthetic.', 'author': 'Benzinga',
                   'url': 'https://example.invalid/article' + str(len(calls)), 'created_at': '2026-10-10T10:00:00Z', 'symbols': ['MU'], 'content': 'FULL BODY NOT TO STORE'}
        return 200, {}, json.dumps({'news': [article], 'next_page_token': 'second' if len(calls) == 1 else None}).encode()
    collector = Collector(Client(store, grant, send=send, credentials=keys), channels=[])
    assert collector.news() == 2
    assert 'symbols' not in parse_qs(urlsplit(calls[0][0]).query)
    assert parse_qs(urlsplit(calls[1][0]).query)['page_token'] == ['second']
    assert parse_qs(urlsplit(calls[0][0]).query)['include_content'] == ['false']
    assert 'FULL BODY' not in json.dumps(store.rows('news'))
    assert 'synthetic-secret' not in store.path.read_bytes().decode(errors='ignore')


def test_news_cursor_resumes_after_crash_without_losing_page(state, tmp_path):
    store, grant, _, _ = state
    keys = tmp_path / 'synthetic.env'
    keys.write_text('APCA_API_KEY_ID=x\nAPCA_API_SECRET_KEY=y\n')
    store.append('news_cursor', 'alpaca', {'start': '2026-10-09T00:00:00Z', 'end': iso(AT), 'token': 'resume', 'backlog': True})
    calls = []
    def send(url, headers):
        calls.append(url)
        return 200, {}, b'{"news":[],"next_page_token":null}'
    Collector(Client(store, grant, send=send, credentials=keys), channels=[]).news()
    assert parse_qs(urlsplit(calls[0]).query)['page_token'] == ['resume']
    assert store.latest('news_cursor', 'alpaca')['backlog'] is False


def test_entity_dictionary_and_transmission_cover_operator_examples():
    dictionary = Dictionary()
    event = cluster([story(symbols=['MU'])])[0]
    assert {'MU'} <= set(event['entities']['symbols'])
    assert {'AMAT', 'LRCX', 'KLAC', 'SMH'} <= {r['symbol'] for r in event['transmission_hypotheses']}
    brazil = cluster([story(headline='Brazil election rally')])[0]
    assert 'Brazil' in brazil['entities']['countries']
    assert 'EWZ' in {r['symbol'] for r in brazil['transmission_hypotheses']}
    euro = dictionary.map(story(headline='BCE euro numérique réglementation'))
    assert 'regulation' in euro['themes'] and 'Europe' in euro['countries']
    assert dictionary.coverage()['company_names'] >= 490


def test_ambiguous_short_tickers_not_matched_as_english_words():
    dictionary = Dictionary()
    mapped = dictionary.map(story(headline='All stocks are on sale now with a key change'))
    assert not {'ALL', 'ARE', 'ON', 'NOW', 'KEY', 'A', 'T'} & set(mapped['symbols'])


def test_new_receipt_of_old_feed_article_does_not_rank_as_new_event():
    old = story(slug='old', headline='Bitcoin regulation', published=AT-timedelta(days=10))
    fresh = story(slug='fresh', headline='Ethereum guidance', published=AT-timedelta(hours=1))
    unknown = story(slug='unknown', headline='Brazil election', published=None)
    events = cluster([old, fresh, unknown], at=AT)
    assert {s['url'] for e in events for s in e['stories']} == {fresh['url'], unknown['url']}


def test_rolling_daily_crypto_index_is_not_an_event_time_catalyst(state):
    store, _, _, _ = state
    roundup = story(slug='daily-index', headline='Here’s what happened in crypto today')
    event = story(slug='specific', headline='Ethereum network upgrade')
    collector = Collector(type('Client', (), {'store': store})(), channels=[])
    collector.add([roundup, event])
    ranked = cluster(store.rows('news'), at=AT)
    assert len(ranked) == 1 and ranked[0]['headline'] == event['headline']
    assert len(store.rows('news')) == 2


def test_retired_observation_watch_does_not_generate_future_labels(state):
    store, _, _, _ = state
    store.append('paper_watch', 'synthetic-watch', {'key': 'synthetic-watch', 'symbol': 'BTC/USD',
                 'recorded_at': iso(AT-timedelta(days=2)), 'baseline': 100, 'horizons_calendar_days': [1, 5]})
    store.append('paper_watch_retired', 'synthetic-watch', {'watch_key': 'synthetic-watch', 'reason': 'non_event_roundup'})
    assert paper_followup(store, [], {'BTC/USD': valid_bars(daily_history(), AT)}, {}, AT) == []
    assert len(store.rows('paper_watch')) == 1
    assert not store.rows('paper_label')


def test_unavailable_youtube_is_unknown_hype_rather_than_zero(state):
    store, _, _, _ = state
    seed(store)
    report = radar(store)
    report['live'] = True
    report['sources'] = [{'source': 'youtube:@synthetic', 'status': 'DEAD'}]
    markdown = render(report)
    assert 'hype inconnu (collecte YouTube indisponible)' in markdown
    assert 'hype 0.' not in markdown


def test_generic_financial_words_do_not_create_crypto_entities():
    mapped = Dictionary().map(story(headline='Chip maker sees optimism as the yield curve and compound costs ripple through margins'))
    assert not {'MKR/USD', 'OP/USD', 'CRV/USD', 'COMP/USD', 'XRP/USD'} & set(mapped['symbols'])
    assert 'crypto' not in mapped['themes']
    explicit = Dictionary().map(story(headline='Crypto lending at Compound and Curve'))
    assert {'COMP/USD', 'CRV/USD'} <= set(explicit['symbols'])


def test_weekend_prioritizes_crypto_text_over_provider_basket_tags():
    macro = story(slug='macro', headline='Brazil election and inflation guidance', symbols=['BTC'])
    crypto = story(slug='crypto', headline='Solana network update')
    events = cluster([macro, crypto], at=AT, crypto_first=True)
    assert events[0]['headline'] == crypto['headline']
    assert 'BTC/USD' in events[1]['entities']['symbols']


def test_dedupe_syndication_does_not_create_confirmation():
    stories = [story(slug='a', publisher='yahoo', headline='Reuters: Micron raises memory guidance after earnings'),
               story(slug='b', publisher='benzinga', headline='Reuters: Micron raises memory guidance after earnings')]
    events = cluster(stories)
    assert len(events) == 1
    assert events[0]['evidence']['publishers'] == ['reuters']
    assert events[0]['evidence']['status'] == 'single_source'


def test_two_editorial_groups_corroborate_but_retail_never_confirms():
    events = cluster([story(slug='a', publisher='wire'), story(slug='b', publisher='editor'),
                      story(slug='c', publisher='retail', retail=True)])
    assert len(events) == 1 and events[0]['evidence']['status'] == 'corroborated_reporting'
    assert events[0]['retail_hype']['channels'] == 1
    assert events[0]['conviction'] is None


def test_repeated_rumour_stays_rumour_and_distinct_company_events_stay_separate():
    headline = 'Micron could raise guidance on memory rumour'
    events = cluster([story(slug='a', headline=headline, publisher='wire'), story(slug='b', headline=headline, publisher='editor'),
                      story(slug='c', headline='Micron factory fire disrupts output', publisher='wire')])
    assert len(events) == 2
    assert next(e for e in events if e['headline'] == headline)['evidence']['status'] == 'rumour'


def test_novelty_is_durable_and_repeats_do_not_gain_importance():
    first = cluster([story()])
    second = cluster([story()], previous=first)
    assert first[0]['novelty'] == 'new' and second[0]['novelty'] == 'repeat'
    assert second[0]['importance'] < first[0]['importance']


def test_fractional_timestamps_keep_true_publication_and_receipt_order():
    later = AT + timedelta(microseconds=500000)
    event = cluster([story(slug='later', received=later, published=later),
                     story(slug='first', received=AT, published=AT)])[0]
    assert event['first_received_at'] == iso(AT)
    assert event['published_at'] == iso(AT)


def test_store_receipt_range_respects_fractional_boundary(state):
    store, _, clock, _ = state
    store.append('synthetic', 'first', {'id': 'first'})
    clock.at = AT + timedelta(microseconds=300000)
    store.append('synthetic', 'second', {'id': 'second'})
    assert store.rows('synthetic', since=clock.at) == [{'id': 'second'}]


def test_future_receipt_and_incomplete_or_invalid_bars_are_excluded():
    assert cluster([story(received=AT + timedelta(seconds=1))], at=AT) == []
    assert valid_bars([bar(AT), {**bar(AT-timedelta(days=2)), 'c': float('nan')},
                       {**bar(AT-timedelta(days=2)), 'h': 1}], AT) == []


def test_price_anomaly_uses_prior_twenty_days():
    history = valid_bars(daily_history(), AT)
    last = {**history[-1], 'o': 120, 'h': 122, 'l': 118, 'c': 120, 'v': 3000}
    result = anomalies({'MU': history[:-1] + [last]})
    assert result[0]['volume_vs_20d'] == 3
    assert result[0]['move_atr'] == 5


def test_priced_in_uses_pre_publication_atr_and_closed_minute_baseline():
    event = cluster([story()])[0]
    daily = valid_bars(daily_history(), AT)
    publication = AT - timedelta(hours=1)
    minutes = valid_bars([bar(publication-timedelta(minutes=1)), bar(publication+timedelta(minutes=1), 108)], AT, daily=False)
    result = priced_in(event, {'MU': daily}, {'MU': minutes}, AT)
    assert result['status'] == 'MEASURED'
    assert result['measurements'][0]['move_atr'] == 2
    event['published_at'] = None
    assert priced_in(event, {'MU': daily}, {'MU': minutes}, AT)['status'] == 'UNKNOWN_PUBLICATION_TIME'


def test_stale_price_baseline_cannot_measure_priced_in():
    event = cluster([story()])[0]
    publication = AT-timedelta(hours=1)
    minute = valid_bars([bar(publication-timedelta(hours=2)), bar(publication+timedelta(minutes=1), 108)], AT, daily=False)
    assert priced_in(event, {'MU': valid_bars(daily_history(), AT)}, {'MU': minute}, AT)['measurements'] == []


def test_regime_missing_exact_instruments_never_uses_proxies():
    panel = regime({'UUP': valid_bars(daily_history(), AT), 'GLD': valid_bars(daily_history(), AT)}, AT)
    assert panel['dollar_index']['last'] is None and panel['gold']['last'] is None
    assert set(panel) == {'SPY', 'QQQ', '10y_yield', 'dollar_index', 'EURUSD', 'gold', 'Brent', 'BTC', 'ETH', 'VIX'}


def scenario(event):
    exposure = event['transmission_hypotheses'][0]
    return {'id': event['id'], 'summary': 'Information synthétique.', 'changed_expectations': 'Consensus inconnu.',
            'impact': 'Effet possible à confirmer.', 'horizon': 'Quelques séances.',
            'priced_in': 'Mouvement observé sans causalité établie.', 'invalidation': 'Démenti ou révision contraire.',
            'conviction': 'faible', 'exposures': [{'symbol': exposure['symbol'], 'role': exposure['role'],
                                                'direction': 'beneficiary', 'mechanism': 'Si la demande progresse, les marges peuvent augmenter.'}]}


def test_llm_command_has_no_tools_settings_hooks_or_sessions():
    args = command()
    assert args[args.index('--tools')+1] == ''
    assert args[args.index('--allowedTools')+1] == ''
    assert args[args.index('--model')+1] == 'sonnet'
    assert '--safe-mode' in args and '--no-session-persistence' in args
    assert '--disable-slash-commands' in args and '--strict-mcp-config' in args
    assert not any('dangerously' in a for a in args)


@pytest.mark.parametrize('mutation', ['numbers', 'symbol', 'role', 'event', 'duplicate'])
def test_llm_rejects_invented_numbers_entities_and_event_ids(mutation):
    event = cluster([story()])[0]
    output = {'events': [scenario(event)]}
    if mutation == 'numbers':
        output['events'][0]['impact'] = 'Le prix est 467 dollars.'
    elif mutation == 'symbol':
        output['events'][0]['exposures'][0]['symbol'] = 'INVENTED'
    elif mutation == 'role':
        output['events'][0]['exposures'][0]['role'] = 'invented'
    elif mutation == 'event':
        output['events'][0]['id'] = 'unknown'
    else:
        output['events'] *= 2
    with pytest.raises(RadarError):
        validate(output, [event])


@pytest.mark.parametrize('malformed', [[], None, {'events': [None]}, {'events': [{'id': []}]}])
def test_malformed_llm_schema_yields_fixed_error_code(malformed):
    with pytest.raises(RadarError, match='LLM_SCHEMA_INVALID'):
        validate(malformed, cluster([story()]))


def test_llm_failed_calls_count_and_weekend_has_same_two_call_limit(state):
    store, grant, _, _ = state
    event = cluster([story()])[0]
    def fail(*a):
        raise RadarError('LLM_FAILED')
    for _ in range(2):
        with pytest.raises(RadarError, match='LLM_FAILED'):
            summarize(store, grant, [event], {}, runner=fail)
    with pytest.raises(RadarError, match='DAILY_BUDGET'):
        summarize(store, grant, [event], {}, runner=fail)
    assert store.counts()['llm'] == 2


def test_tool_free_llm_attaches_separate_scenario_and_conviction(state):
    store, grant, _, _ = state
    event = cluster([story()])[0]
    result = summarize(store, grant, [event], {}, runner=lambda *a: {'events': [scenario(event)]})
    assert result['calls'] == 1
    assert event['conviction'] == 'faible'
    assert event['evidence']['status'] == 'single_source'


def test_paper_watch_is_prospective_durable_and_requires_future_observation(state):
    store, _, clock, _ = state
    event = cluster([story()])[0]
    event['scenario'] = scenario(event)
    daily = {'MU': valid_bars(daily_history(), AT)}
    paper_followup(store, [event], daily, {}, AT)
    paper_followup(store, [event], daily, {}, AT)
    assert len(store.rows('paper_watch')) == 1
    clock.at += timedelta(days=2)
    assert paper_followup(store, [], daily, {}, clock.at) == []
    rows = valid_bars([bar(AT+timedelta(days=1), 110)], clock.at, daily=False)
    labels = paper_followup(store, [], daily, {'MU': rows}, clock.at)
    assert len(labels) == 1 and labels[0]['return_pct'] == 10
    assert labels[0]['observed_at'] > labels[0]['target_at']
    assert paper_followup(store, [], daily, {'MU': rows}, clock.at) == []


def test_offline_end_to_end_french_report_bounded_and_no_book_write(state, tmp_path):
    store, _, _, _ = state
    seed(store)
    report = radar(store, slot='morning')
    assert report['llm']['status'] == 'OFFLINE_NOT_CALLED'
    assert report['sources'] == [] and not report['live']
    assert len(report['markdown'].splitlines()) <= 60
    assert 'HORS LIGNE' in report['markdown']
    assert report['paper_watch_count'] == 0
    assert len(report['events']) == 4
    assert store.counts() == {}
    assert radar(store, slot='morning') == report
    assert len(store.rows('report')) == 1


def test_timer_schedules_are_utc_and_only_radar_units():
    texts = unit_texts('/synthetic/bin/python', '/synthetic/worktree', claude='/synthetic/bin/claude')
    assert len(texts) == 6
    assert all(name.startswith('hyprl-radar-') for name in texts)
    assert '12:20:00 UTC' in texts['hyprl-radar-morning.timer']
    assert '11:30:00 UTC' in texts['hyprl-radar-morning.timer']
    assert '21:20:00 UTC' in texts['hyprl-radar-evening.timer']
    assert '19:30:00 UTC' in texts['hyprl-radar-evening.timer']
    assert all('KillMode=process' in value for name, value in texts.items() if name.endswith('.service'))
    assert all('Persistent=false' in value for name, value in texts.items() if name.endswith('.timer'))


def test_partial_capture_does_not_become_final_as_clock_advances(state):
    store, _, clock, _ = state
    opened = AT.replace(hour=0, minute=0)
    store.append('daily_bars', 'MU', {'symbol': 'MU', 'bars': [bar(opened, 110)], 'received_at': iso(AT)})
    clock.at += timedelta(days=2)
    assert bar_map(store, 'daily_bars', clock.at)['MU'] == []
    partial = daily_observations(store, clock.at)['MU']
    assert not partial['complete_at_receipt']
    store.append('daily_bars', 'MU', {'symbol': 'MU', 'bars': [bar(opened, 120)], 'received_at': iso(clock.at)})
    assert bar_map(store, 'daily_bars', clock.at)['MU'][0]['c'] == 120


def test_broad_daily_scan_keeps_partial_day_volume_explicit(state):
    store, _, _, _ = state
    rows = daily_history() + [bar(AT.replace(hour=0, minute=0), close=112, volume=2500)]
    store.append('daily_bars', 'MU', {'symbol': 'MU', 'bars': rows, 'received_at': iso(AT)})
    daily = bar_map(store, 'daily_bars', AT)
    result = anomalies(daily, daily_observations(store, AT))[0]
    assert result['partial_day'] and result['closed_at'] is None
    assert result['move_atr'] == 3 and result['volume_vs_20d'] == 2.5
    assert result['received_at'] == iso(AT)


def test_new_independent_reporting_updates_evidence_and_novelty():
    first = cluster([story(publisher='first')])
    second = cluster([story(publisher='first'), story(slug='another', publisher='second')], previous=first)
    assert second[0]['novelty'] == 'updated'
    assert second[0]['evidence']['status'] == 'corroborated_reporting'


def test_original_http_links_are_preserved_but_never_fetched():
    row = item('synthetic', 'wire', 'http://example.invalid/story?utm_source=synthetic', 'headline', '', None, AT)
    assert row['source_url'].startswith('http://') and row['url'] == 'http://example.invalid/story'


def test_observed_regime_quote_is_timed_and_future_quote_ignored():
    daily = {'^VIX': valid_bars(daily_history(), AT)}
    observed = {'^VIX': {'price': 104, 'at': iso(AT)}}
    panel = regime(daily, AT, observed)
    assert panel['VIX']['last'] == 104 and panel['VIX']['returns_pct']['1d'] == 4
    assert panel['VIX']['return_anchors']['1d'] < iso(AT)
    observed['^VIX']['at'] = iso(AT + timedelta(minutes=1))
    assert regime(daily, AT, observed)['VIX']['last'] == 100


def test_unsigned_template_cannot_authorize_requests(state, tmp_path):
    from scripts.radar.grant_template import draft
    path = tmp_path / 'draft.json'
    path.write_text(json.dumps(draft(AT)))
    with pytest.raises(RadarError, match='AUTHORIZATION_MISSING_OR_INVALID'):
        Authorization(path)


def test_keys_are_never_forwarded_to_rss_host(state, tmp_path):
    store, grant, _, _ = state
    path = tmp_path / 'synthetic.env'
    path.write_text('APCA_API_KEY_ID=synthetic-id\nAPCA_API_SECRET_KEY=synthetic-secret\n')
    headers = []
    def send(url, request_headers):
        headers.append(request_headers)
        return 200, {}, RSS
    Client(store, grant, send=send, credentials=path).get('https://example.invalid/feed', 'rss', 'feed')
    assert all('APCA' not in key for key in headers[0])


def test_publication_preserves_existing_book_directory_permissions(tmp_path):
    from scripts.radar.core import write_private
    folder = tmp_path / 'synthetic-book'
    folder.mkdir(mode=0o750)
    # mkdir's mode is filtered by the host umask; establish the intended
    # pre-existing publication-directory permissions explicitly.
    folder.chmod(0o750)
    write_private(folder / 'radar-latest.md', 'Synthetic radar')
    assert folder.stat().st_mode & 0o777 == 0o750
    assert (folder / 'radar-latest.md').stat().st_mode & 0o777 == 0o600


def test_full_live_pipeline_with_synthetic_transports_and_llm(state, tmp_path, monkeypatch):
    from scripts.radar.core import instant
    from scripts.radar.registry import REGIME
    from scripts.radar.grant_template import draft
    from urllib.parse import unquote
    store, _, clock, path = state
    payload = draft(AT)
    payload['operator_signed'] = 'synthetic operator'
    path.write_text(json.dumps(payload))
    grant = Authorization(path)
    keys = tmp_path / 'synthetic.env'
    keys.write_text('APCA_API_KEY_ID=synthetic-id\nAPCA_API_SECRET_KEY=synthetic-secret\n')
    sent = []
    def send(url, headers):
        sent.append(url)
        parsed = urlsplit(url)
        params = parse_qs(parsed.query)
        if parsed.hostname == 'data.alpaca.markets':
            if parsed.path.endswith('/news'):
                value = {'news': [{'headline': 'Micron raises memory guidance after earnings', 'summary': 'Synthetic context.',
                                   'url': 'https://example.invalid/micron', 'created_at': iso(AT-timedelta(hours=1)), 'symbols': ['MU']}], 'next_page_token': None}
            elif '/screener/' in parsed.path:
                value = {'most_actives': [{'symbol': 'MU', 'volume': 1000}]} if 'most-actives' in parsed.path else {'gainers': [{'symbol': 'MU', 'percent_change': 5}], 'losers': []}
            else:
                symbols = params['symbols'][0].split(',')
                rows = daily_history() if params['timeframe'] == ['1Day'] else [bar(AT-timedelta(hours=1, minutes=1)), bar(AT-timedelta(minutes=2), 108)]
                value = {'bars': {s: rows for s in symbols}, 'next_page_token': None}
            return 200, {}, json.dumps(value).encode()
        if parsed.hostname == 'query1.finance.yahoo.com':
            symbol = unquote(parsed.path.rsplit('/', 1)[-1])
            history = daily_history()
            value = {'chart': {'result': [{'meta': {'symbol': symbol, 'regularMarketPrice': 100, 'regularMarketTime': int(AT.timestamp())},
                      'timestamp': [int(instant(b['t']).timestamp()) for b in history],
                      'indicators': {'quote': [{k: [b[v] for b in history] for k, v in [('open', 'o'), ('high', 'h'), ('low', 'l'), ('close', 'c'), ('volume', 'v')]}]}}], 'error': None}}
            return 200, {}, json.dumps(value).encode()
        if parsed.hostname == 'api.gdeltproject.org':
            return 200, {}, b'{"articles":[]}'
        return 200, {}, RSS
    def sleep(seconds):
        clock.at += timedelta(seconds=seconds)
    monkeypatch.setattr('scripts.radar.service.time.sleep', sleep)
    def llm(prompt, root):
        data = json.loads(prompt.split('\n', 1)[1])
        return {'events': [scenario(e) for e in data['events']]}
    book = tmp_path / 'book/radar-latest.md'
    report = radar(store, grant, live=True, client=Client(store, grant, send=send, credentials=keys),
                   channels=[{'channel_id': 'UC' + 'a' * 22}], llm_runner=llm, publish_book=book)
    assert report['status'] == 'READY'
    assert len(report['regime']) == len(REGIME)
    assert all(row['status'] == 'OBSERVED' for row in report['regime'].values())
    assert report['llm']['calls'] == 1 and store.counts()['llm'] == 1
    assert book.read_text() == report['markdown'] and len(report['markdown'].splitlines()) <= 60
    assert len(store.rows('paper_watch')) >= 1
    assert all(urlsplit(url).hostname not in ('sec.gov', 'federalreserve.gov') for url in sent)
    assert store.counts()['gdelt'] == 3
    again = radar(store, grant, live=True, client=Client(store, grant, send=send, credentials=keys), llm_runner=llm, publish_book=book)
    assert again == report and store.counts()['llm'] == 1


def test_owner_lock_waits_for_a_concurrent_unit_then_fails_closed(state):
    store = state[0] if isinstance(state, tuple) else state
    waits = []
    with store.owner():
        other = Store(store.root, clock=store.clock)

        def release(seconds):
            waits.append(seconds)

        with pytest.raises(RadarError, match='OWNER_BUSY'):
            with other.owner(timeout=3, poll=1, sleep=release, monotonic=iter([0, 0, 1, 2, 3, 4]).__next__):
                pass
    assert waits == [1, 1, 1]


def test_owner_lock_is_acquired_once_the_holder_releases(state):
    store = state[0] if isinstance(state, tuple) else state
    holder = store.owner()
    holder.__enter__()
    other = Store(store.root, clock=store.clock)

    def release(_seconds):
        holder.__exit__(None, None, None)

    with other.owner(timeout=5, poll=1, sleep=release):
        pass
