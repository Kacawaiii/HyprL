"""Bounded metadata resolution and provider-wide throttle behavior."""
from datetime import timedelta
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts.radar.core import Client, RadarError, Store
from scripts.radar.sources import Collector
from tests.radar.test_radar import state

YOUTUBE_ATOM = b'''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:yt="http://www.youtube.com/xml/schemas/2015" xmlns:media="http://search.yahoo.com/mrss/">
<id>yt:channel:UCaaaaaaaaaaaaaaaaaaaaaa</id><yt:channelId>UCaaaaaaaaaaaaaaaaaaaaaa</yt:channelId>
<entry><id>yt:video:synthetic01</id><yt:videoId>synthetic01</yt:videoId><yt:channelId>UCaaaaaaaaaaaaaaaaaaaaaa</yt:channelId>
<title>Synthetic market update</title><link rel="alternate" href="https://www.youtube.com/watch?v=synthetic01"/>
<published>2026-10-10T10:00:00Z</published><updated>2026-10-10T11:00:00Z</updated>
<media:group><media:description>Synthetic retail narrative</media:description></media:group></entry></feed>'''


def test_channel_redirect_is_grant_checked_and_identity_cached_once(state):
    store, grant, clock, _ = state
    calls = []
    channel_id = 'UC' + 'a' * 22
    def send(url, headers):
        calls.append(url)
        if urlsplit(url).path == '/@Synthetic/about':
            return 302, {'Location': '/@Synthetic/featured'}, b''
        if urlsplit(url).path == '/@Synthetic/featured':
            return 200, {}, json.dumps({'externalId': channel_id}).encode()
        assert parse_qs(urlsplit(url).query)['channel_id'] == [channel_id]
        return 200, {}, YOUTUBE_ATOM
    collector = Collector(Client(store, grant, send=send), channels=[])
    assert collector.channel({'handle': '@Synthetic'}) == 1
    clock.at += timedelta(hours=1)
    collector.channel({'handle': '@Synthetic'})
    assert sum('/@Synthetic' in url for url in calls) == 2
    assert store.latest('channel', '@Synthetic')['channel_id'] == channel_id


@pytest.mark.parametrize('location', ['https://consent.youtube.com/', 'https://elsewhere.invalid/@Synthetic', '/@Other/featured', '/watch?v=synthetic'])
def test_channel_resolution_cannot_leave_approved_handle(state, location):
    store, grant, _, _ = state
    calls = []
    def send(url, headers):
        calls.append(url)
        return 302, {'Location': location}, b''
    with pytest.raises(RadarError, match='CHANNEL_REDIRECT_OUTSIDE_SCOPE'):
        Collector(Client(store, grant, send=send), channels=[]).channel({'handle': '@Synthetic'})
    assert len(calls) == 1


def test_gdelt_throttle_backoff_is_exponential_across_restarts(state):
    store, grant, clock, _ = state
    url = 'https://api.gdeltproject.org/api/v2/doc/doc?query=synthetic'
    send = lambda *args: (429, {}, b'')
    for hours in (1, 2, 4):
        client = Client(Store(store.root, clock=clock), grant, send=send)
        with pytest.raises(RadarError, match='HTTP_429'):
            client.get(url, 'gdelt', 'gdelt:synthetic')
        assert store.latest('cooldown', 'gdelt')['until'] == (clock.at + timedelta(hours=hours)).isoformat().replace('+00:00', 'Z')
        clock.at += timedelta(hours=hours)


def test_collect_uses_three_gdelt_themes_with_minimum_fifteen_second_spacing(state):
    store, grant, clock, _ = state
    times = []
    def send(url, headers):
        if urlsplit(url).hostname == 'api.gdeltproject.org':
            times.append(clock.at)
            return 200, {}, b'{"articles":[]}'
        return 200, {}, b'{"news":[],"next_page_token":null}'
    def sleep(seconds):
        clock.at += timedelta(seconds=seconds)
    Collector(Client(store, grant, send=send), channels=[]).collect(markets=False, sleep=sleep)
    assert len(times) == 3
    assert all((b-a).total_seconds() >= 15 for a, b in zip(times, times[1:]))


def channel_page(channel_id, *, video_id='synthetic01', title='Synthetic market update', channel_title='Synthetic'):
    def lockup(video_id, title, content_type):
        return {'richItemRenderer': {'content': {'lockupViewModel': {
            'contentId': video_id, 'contentType': content_type,
            'metadata': {'lockupMetadataViewModel': {'title': {'content': title}}}}}}}
    entries = [lockup(video_id, title, 'LOCKUP_CONTENT_TYPE_VIDEO'),
               lockup('playlist001', 'Not a video', 'LOCKUP_CONTENT_TYPE_PLAYLIST')]
    tabs = [{'tabRenderer': {'selected': True, 'content': {'richGridRenderer': {'contents': entries}}}},
            {'tabRenderer': {'content': {'videoRenderer': {'videoId': 'related0001',
             'title': {'simpleText': 'Not in the selected channel tab'}}}}}]
    data = {'metadata': {'channelMetadataRenderer': {'externalId': channel_id, 'title': channel_title}},
            'contents': {'twoColumnBrowseResultsRenderer': {'tabs': tabs}}}
    return ('<html><script>var ytInitialData = ' + json.dumps(data) + ';</script></html>').encode()


def test_dead_channel_feed_recovers_titles_from_verified_handle_metadata(state):
    store, grant, clock, _ = state
    channel_id = 'UC' + 'a' * 22
    store.append('channel', '@Synthetic', {'channel_id': channel_id})
    store.append('health', 'youtube:@Synthetic', {'source': 'youtube:@Synthetic',
        'status': 'DEAD', 'reason': 'HTTP_404', 'items': 0})
    calls = []
    def send(url, headers):
        calls.append(url)
        if urlsplit(url).path == '/@Synthetic/videos':
            return 200, {}, channel_page(channel_id)
        return 200, {}, b'{"news":[],"next_page_token":null}'
    collector = Collector(Client(store, grant, send=send), channels=[{'handle': '@Synthetic'}])
    collector.collect(markets=False, gdelt=False)
    health = next(h for h in collector.health if h['source'] == 'youtube:@Synthetic')
    assert health['status'] == 'LIVE' and health['items'] == 1
    news = store.rows('news')
    assert len(news) == 1 and news[0]['url'] == 'https://www.youtube.com/watch?v=synthetic01'
    assert news[0]['published_at'] is None and news[0]['summary'] == '' and news[0]['retail']
    assert news[0]['source'] == 'youtube:' + channel_id
    clock.at += timedelta(hours=1)
    Collector(Client(Store(store.root, clock=clock), grant, send=send), channels=[{'handle': '@Synthetic'}]).collect(markets=False, gdelt=False)
    assert calls.count('https://www.youtube.com/@Synthetic/videos') == 2
    assert not any('/feeds/' in url for url in calls)
    assert store.counts()['youtube'] == 2


def test_bloomberg_alias_replaces_stale_channel_identity_only_after_metadata_verification(state):
    store, grant, _, _ = state
    old_id, new_id = 'UC' + 'a' * 22, 'UC' + 'b' * 22
    store.append('channel', '@BloombergTelevision', {'channel_id': old_id,
        'metadata_url': 'https://www.youtube.com/@BloombergTelevision/about'})
    store.append('channel_resolution', '@BloombergTelevision', {'status': 'STARTED'})
    store.append('health', 'youtube:@BloombergTelevision', {'source': 'youtube:@BloombergTelevision',
        'status': 'DEAD', 'reason': 'HTTP_404', 'items': 0})
    calls = []
    def send(url, headers):
        calls.append(url)
        if urlsplit(url).path == '/@markets/about':
            return 200, {}, json.dumps({'channelMetadataRenderer': {
                'externalId': new_id, 'title': 'Bloomberg Television'}}).encode()
        if urlsplit(url).path == '/@markets/videos':
            return 200, {}, channel_page(new_id, channel_title='Bloomberg Television')
        raise AssertionError('Unexpected endpoint')
    collector = Collector(Client(store, grant, send=send), channels=[])
    assert collector.channel({'handle': '@BloombergTelevision'}) == 1
    assert store.latest('channel', '@BloombergTelevision')['channel_id'] == new_id
    assert store.rows('news')[0]['publisher'] == 'bloomberg'
    assert calls == ['https://www.youtube.com/@markets/about', 'https://www.youtube.com/@markets/videos']


@pytest.mark.parametrize('response', [
    channel_page('UC' + 'b' * 22),
    b'<html><script>var ytInitialData = {"metadata":{}};</script></html>',
    channel_page('UC' + 'a' * 22, video_id='invalid'),
])
def test_channel_page_failure_never_claims_live_or_fetches_video_pages(state, response):
    store, grant, _, _ = state
    channel_id = 'UC' + 'a' * 22
    store.append('channel', '@Synthetic', {'channel_id': channel_id})
    store.append('health', 'youtube:@Synthetic', {'source': 'youtube:@Synthetic',
        'status': 'DEAD', 'reason': 'HTTP_404', 'items': 0})
    sent = []
    def send(url, headers):
        sent.append(url)
        return 200, {}, response
    collector = Collector(Client(store, grant, send=send), channels=[{'handle': '@Synthetic'}])
    collector.attempt('youtube:@Synthetic', lambda: collector.channel({'handle': '@Synthetic'}))
    assert collector.health[0]['status'] == 'BLOCKED'
    assert not store.rows('news')
    assert sent == ['https://www.youtube.com/@Synthetic/videos']


def test_channel_page_respects_scope_spacing_and_lowered_daily_budget(state):
    store, grant, clock, _ = state
    channel_id = 'UC' + 'a' * 22
    store.append('channel', '@Synthetic', {'channel_id': channel_id})
    store.append('health', 'youtube:@Synthetic', {'source': 'youtube:@Synthetic',
        'status': 'DEAD', 'reason': 'HTTP_404', 'items': 0})
    grant.payload['scope']['https://www.youtube.com'] = {'methods': ['GET'], 'paths': ['/feeds/videos.xml']}
    calls = []
    def send(url, headers):
        calls.append(url)
        return 200, {}, channel_page(channel_id)
    collector = Collector(Client(store, grant, send=send), channels=[])
    with pytest.raises(RadarError, match='PATH_OUTSIDE_GRANT'):
        collector.channel({'handle': '@Synthetic'})
    assert not calls and not store.counts()
    grant.payload['scope']['https://www.youtube.com']['path_prefixes'] = ['/@']
    grant.payload['budgets']['youtube'] = 1
    assert collector.channel({'handle': '@Synthetic'}) == 1
    # A second request on the same day is refused even after spacing elapsed.
    clock.at += timedelta(minutes=15)
    with pytest.raises(RadarError, match='DAILY_BUDGET'):
        collector.channel({'handle': '@Synthetic'})
    assert len(calls) == 1
    clock.at += timedelta(days=1)
    assert collector.channel({'handle': '@Synthetic'}) == 0
    grant.payload['budgets']['youtube'] = 2
    with pytest.raises(RadarError, match='POLL_SPACING'):
        Collector(Client(Store(store.root, clock=clock), grant, send=send), channels=[]).channel({'handle': '@Synthetic'})
    assert len(calls) == 2


def test_fresh_channel_feed_404_switches_once_then_reuses_metadata_route(state):
    store, grant, clock, _ = state
    channel_id = 'UC' + 'a' * 22
    store.append('channel', '@Synthetic', {'channel_id': channel_id})
    calls = []
    def send(url, headers):
        calls.append(url)
        return (404, {}, b'') if '/feeds/' in url else (200, {}, channel_page(channel_id))
    collector = Collector(Client(store, grant, send=send), channels=[])
    assert collector.channel({'handle': '@Synthetic'}) == 1
    clock.at += timedelta(hours=1)
    assert Collector(Client(Store(store.root, clock=clock), grant, send=send), channels=[]).channel({'handle': '@Synthetic'}) == 0
    assert sum('/feeds/' in u for u in calls) == 1 and sum('/@Synthetic/videos' in u for u in calls) == 2
    assert store.counts()['youtube'] == 3


def test_alias_identity_cannot_come_from_an_unrelated_video_title(state):
    store, grant, _, _ = state
    body = json.dumps({'channelMetadataRenderer': {'externalId': 'UC' + 'a' * 22,
                      'title': 'Different creator'}, 'relatedTitle': 'Bloomberg Television'}).encode()
    calls = []
    def send(url, headers):
        calls.append(url)
        return 200, {}, body
    with pytest.raises(RadarError, match='CHANNEL_ID_UNVERIFIED'):
        Collector(Client(store, grant, send=send), channels=[]).channel({'handle': '@BloombergTelevision'})
    assert calls == ['https://www.youtube.com/@markets/about']
    assert store.latest('channel', '@BloombergTelevision') is None


def test_repaired_coindesk_path_is_checked_again_without_implicitly_extending_grant(state):
    store, grant, _, _ = state
    old_url = 'https://www.coindesk.com/arc/outboundfeeds/rss/'
    grant.payload['scope']['https://www.coindesk.com'] = {'methods': ['GET'], 'paths': ['/arc/outboundfeeds/rss/']}
    store.append('http', 'rss:coindesk', {'url': old_url, 'status': 308})
    store.append('health', 'rss:coindesk', {'source': 'rss:coindesk', 'status': 'DEAD', 'reason': 'HTTP_308', 'items': 0})
    calls = []
    def send(url, headers):
        calls.append(url)
        return 200, {}, b'{"news":[],"next_page_token":null}'
    collector = Collector(Client(store, grant, send=send), channels=[])
    collector.collect(markets=False, gdelt=False)
    health = next(h for h in collector.health if h['source'] == 'rss:coindesk')
    assert health['status'] == 'BLOCKED' and health['reason'] == 'PATH_OUTSIDE_GRANT'
    assert not any('coindesk.com' in url for url in calls)
    assert 'rss' not in store.counts()
