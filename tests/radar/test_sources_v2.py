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
