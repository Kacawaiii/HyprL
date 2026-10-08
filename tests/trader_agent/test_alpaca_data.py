from copy import deepcopy
import json
from urllib.parse import parse_qs, urlparse

import pytest

from scripts.trading_lab.trader_agent import alpaca_data as data
from scripts.trading_lab.trader_agent.config import TraderError, instant, iso
from tests.trader_agent.test_alpaca_paper import environment, issue


@pytest.fixture
def feed(environment, tmp_path):
    _, _, paper_grant, clock = environment
    clock.at = instant('2026-10-08T20:10:00Z')
    payload = {'authorization': 'trader-alpaca-data-v1', 'extends': 'trader-alpaca-paper-v1',
               'granted_at': '2026-01-01T00:00:00Z', 'not_after': '2027-01-01T00:00:00Z',
               'data_endpoint': {'origin': data.DATA_URL, 'methods': ['GET'], 'credentials': 'ia_actions keys'}}
    path = tmp_path / 'data-grant.json'
    path.write_text(json.dumps(payload))
    grant = data.DataAuthorization.load(path, paper_grant)
    calls = []
    def http(method, url, headers, body):
        calls.append((method, url, body))
        assert headers['APCA-API-KEY-ID'] == 'synthetic-key'
        u = urlparse(url)
        params = parse_qs(u.query)
        assert u.scheme + '://' + u.netloc == data.DATA_URL
        assert params['feed'] == ['iex']
        assets = params['symbols'][0].split(',')
        if '/quotes/' in u.path:
            return {'quotes': {a: {'bp': 99.95, 'ap': 100.05, 't': iso(clock()),
                                   'bs': 100, 'as': 100, 'bx': 'V', 'ax': 'V'} for a in assets}}
        return {'trades': {a: {'p': 100, 's': 10, 't': iso(clock()), 'x': 'V'} for a in assets}}
    return data.DataClient(grant, clock=clock, transport=http), calls, clock, path


def test_latest_universe_quotes_and_trades_use_get_iex_and_real_wire_shape(feed):
    client, calls, clock, _ = feed
    quotes = client.latest(['AAPL', 'MSFT'])
    assert quotes['AAPL'] == {'bid': '99.95', 'ask': '100.05', 'at': iso(clock()),
                               'trade': {'price': '100', 'at': iso(clock())}}
    assert len(calls) == 2 and all(c[0] == 'GET' and c[2] is None for c in calls)


@pytest.mark.parametrize('origin', ['https://data.alpaca.markets/', 'http://data.alpaca.markets',
                                  'https://data.alpaca.markets.evil.invalid', 'https://api.alpaca.markets'])
def test_non_exact_origin_refused_before_transport(feed, origin):
    client, calls, _, _ = feed
    client.base_url = origin
    with pytest.raises(TraderError, match='DATA_URL_REFUSED'):
        client.latest(['AAPL'])
    assert not calls


@pytest.mark.parametrize('method', ['POST', 'PUT', 'PATCH', 'DELETE'])
def test_data_mutations_refused_before_transport(feed, method):
    client, calls, _, _ = feed
    with pytest.raises(TraderError, match='DATA_GET_ONLY'):
        client.request(method, '/v2/stocks/quotes/latest', params={'symbols': 'AAPL', 'feed': 'iex'})
    assert not calls


@pytest.mark.parametrize('asset', ['SECRET', 'BTC-USD'])
def test_unknown_and_protected_products_never_contact_data_host(feed, asset):
    client, calls, _, _ = feed
    with pytest.raises(TraderError, match='DATA_(UNAUTHORIZED_SYMBOL|PROTECTED)'):
        client.latest([asset])
    assert not calls


@pytest.mark.parametrize('timestamp', ['2026-10-08T20:08:59Z', '2026-10-08T20:10:01Z'])
def test_stale_and_future_quotes_refuse_execution(feed, timestamp):
    client, _, _, _ = feed
    original = client.transport
    def http(*args):
        value = original(*args)
        for q in value.get('quotes', {}).values():
            q['t'] = timestamp
        return value
    client.transport = http
    with pytest.raises(TraderError, match='PAPER_QUOTE_STALE'):
        client.latest(['AAPL'])


def test_expired_data_grant_refuses_before_http(feed):
    client, calls, clock, _ = feed
    client.grant.payload['not_after'] = '2026-10-08T20:11:00Z'
    clock.at = instant('2026-10-08T20:11:00Z')
    with pytest.raises(TraderError, match='DATA_AUTHORIZATION_EXPIRED_OR_NOT_STARTED'):
        client.latest(['AAPL'])
    assert not calls


def test_data_feed_drives_after_hours_limit_orders(environment, feed):
    executor, broker, paper_grant, clock = environment
    client, calls, _, path = feed
    issue(executor, variant='alpaca_after_hours_v1')
    executor.quotes = data.QuoteFeed(paper_grant, path, clock=clock, transport=client.transport)
    assert executor.run('execute', accounts=['ia_actions'])['state'] == 'COMPLETE'
    order = broker.posts()[0][3]
    assert order['extended_hours'] is True and order['limit_price'] == '100.05'
    assert order['qty'] == '43' and order['type'] == 'limit'
    assert calls and all(c[0] == 'GET' for c in calls)


def test_data_unavailable_uses_only_a_fresh_private_file_fallback(feed, tmp_path):
    client, _, clock, path = feed
    fallback = tmp_path / 'quotes.json'
    fallback.write_text(json.dumps({'quotes': {'AAPL': {'bid': 99.9, 'ask': 100.1, 'at': iso(clock())}}}))
    def unavailable(*args):
        raise TraderError('DATA_HTTP_503')
    quotes = data.QuoteFeed(client.grant.parent, path, fallback, clock=clock, transport=unavailable)
    assert quotes.get('AAPL', clock())['bid'] == 99.9
    clock.at = instant('2026-10-08T20:11:01Z')
    with pytest.raises(TraderError, match='PAPER_QUOTE_STALE'):
        quotes.get('AAPL', clock())


def test_invalid_data_grant_is_never_masked_by_file_fallback(feed, tmp_path):
    client, calls, clock, path = feed
    payload = deepcopy(client.grant.payload)
    payload['data_endpoint']['origin'] += '.evil.invalid'
    path.write_text(json.dumps(payload))
    fallback = tmp_path / 'quotes.json'
    fallback.write_text(json.dumps({'quotes': {'AAPL': {'bid': 99.9, 'ask': 100.1, 'at': iso(clock())}}}))
    quotes = data.QuoteFeed(client.grant.parent, path, fallback, clock=clock, transport=client.transport)
    with pytest.raises(TraderError, match='DATA_INVALID_GRANT'):
        quotes.get('AAPL', clock())
    assert not calls


def test_redirect_refused_without_following_location():
    with pytest.raises(TraderError, match='DATA_REDIRECT_REFUSED'):
        data.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.invalid')


def test_wide_spread_cannot_submit_a_limit_order(environment, feed):
    executor, broker, paper_grant, clock = environment
    client, _, _, path = feed
    issue(executor, variant='alpaca_after_hours_v1')
    def wide(*args):
        result = client.transport(*args)
        for q in result.get('quotes', {}).values():
            q.update(bp=99, ap=101)
        return result
    executor.quotes = data.QuoteFeed(paper_grant, path, clock=clock, transport=wide)
    assert executor.run('execute', accounts=['ia_actions'])['accounts'][0]['reason'] == 'PAPER_SPREAD_CAP'
    assert not broker.posts()


def test_live_credential_origin_refused_before_data_request(feed):
    client, calls, clock, _ = feed
    from pathlib import Path
    path = Path(client.grant.parent.payload['accounts']['ia_actions']['credentials'])
    path.write_text('APCA_API_KEY_ID=synthetic\nAPCA_API_SECRET_KEY=synthetic\nAPCA_API_BASE_URL=https://api.alpaca.markets\n')
    with pytest.raises(TraderError, match='PAPER_URL_REFUSED'):
        data.DataClient(client.grant, clock=clock, transport=client.transport)
    assert not calls


def test_synthetic_unprotected_crypto_uses_the_latest_crypto_wire_shape(feed, monkeypatch):
    client, calls, clock, _ = feed
    # No protected price is acquired; these literal values are synthetic.
    monkeypatch.setattr(data, 'protected', lambda *args: None)
    def http(method, url, headers, body):
        calls.append((method, url, body))
        parsed = urlparse(url)
        assert parse_qs(parsed.query) == {'symbols': ['BTC/USD']}
        if parsed.path.endswith('/quotes'):
            return {'quotes': {'BTC/USD': {'bp': 100, 'ap': 100.1, 't': iso(clock())}}}
        assert parsed.path == '/v1beta3/crypto/us/latest/trades'
        return {'trades': {'BTC/USD': {'p': 100, 't': iso(clock()), 's': 1}}}
    client.transport = http
    assert client.latest(['BTC-USD'])['BTC-USD']['trade']['price'] == '100'
    assert len(calls) == 2
