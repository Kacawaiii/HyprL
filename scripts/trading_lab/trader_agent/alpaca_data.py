"""GET-only, grant-bound latest market data. No bodies or credentials are stored."""
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
import json

from scripts.trading_lab.sources.canonical import sha256_canonical

from .alpaca_paper import PaperClient, PrivateQuotes, decimal, symbol
from .config import TraderError, instant, iso, now, strict_json
from .data import protected
from .paper_spec import execution_spec

DATA_URL = 'https://data.alpaca.markets'
PATHS = {'/v2/stocks/quotes/latest': ('stocks', 'quotes'),
         '/v2/stocks/trades/latest': ('stocks', 'trades'),
         '/v1beta3/crypto/us/latest/quotes': ('crypto', 'quotes'),
         '/v1beta3/crypto/us/latest/trades': ('crypto', 'trades')}


@dataclass(frozen=True)
class DataAuthorization:
    payload: dict
    identity: str
    parent: object

    @classmethod
    def load(cls, path, parent):
        payload = strict_json(Path(path).read_text())
        grant = cls(payload, sha256_canonical(payload), parent)
        endpoint = payload.get('data_endpoint', {})
        if (payload.get('authorization') != 'trader-alpaca-data-v1'
                or payload.get('extends') != 'trader-alpaca-paper-v1'
                or endpoint.get('origin') != DATA_URL or endpoint.get('methods') != ['GET']
                or endpoint.get('credentials') != 'ia_actions keys'
                or instant(payload['not_after']) <= instant(payload['granted_at'])):
            raise TraderError('DATA_INVALID_GRANT')
        return grant

    def check(self, at):
        self.parent.check(at)
        if not instant(self.payload['granted_at']) <= at < instant(self.payload['not_after']):
            raise TraderError('DATA_AUTHORIZATION_EXPIRED_OR_NOT_STARTED')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TraderError('DATA_REDIRECT_REFUSED')


def http_transport(method, url, headers, body):
    if method != 'GET' or body is not None:
        raise TraderError('DATA_GET_ONLY')
    try:
        with build_opener(NoRedirect()).open(Request(url, headers=headers, method='GET'), timeout=20) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise TraderError('DATA_RESPONSE_TOO_LARGE')
            return strict_json(raw.decode())
    except HTTPError as error:
        raise TraderError('DATA_HTTP_' + str(error.code)) from None
    except (URLError, OSError, UnicodeError, json.JSONDecodeError):
        raise TraderError('DATA_HTTP_UNAVAILABLE') from None


class DataClient:
    def __init__(self, grant, *, clock=now, transport=http_transport):
        grant.check(clock())
        self.grant, self.clock, self.transport = grant, clock, transport
        self.base_url = grant.payload['data_endpoint']['origin']
        # Reuse the paper client's strict credential parser and paper-origin check;
        # construction sends no broker request and uses only ia_actions credentials.
        self._headers = PaperClient(grant.parent, 'ia_actions', clock=clock)._headers
        self.assets = {symbol(a): a for a in grant.parent.parent.universe}

    def request(self, method, path, *, params=None, body=None):
        if self.base_url != DATA_URL:
            raise TraderError('DATA_URL_REFUSED')
        self.grant.check(self.clock())
        if method != 'GET' or body is not None:
            raise TraderError('DATA_GET_ONLY')
        if path not in PATHS:
            raise TraderError('DATA_PATH_REFUSED')
        group, _ = PATHS[path]
        expected = {'symbols', 'feed'} if group == 'stocks' else {'symbols'}
        if (not isinstance(params, dict) or set(params) != expected
                or (group == 'stocks' and params['feed'] != 'iex')
                or not isinstance(params['symbols'], str)):
            raise TraderError('DATA_PARAMS_REFUSED')
        assets = params['symbols'].split(',')
        if not assets or len(assets) != len(set(assets)):
            raise TraderError('DATA_UNAUTHORIZED_SYMBOL')
        at = self.clock()
        for asset in assets:
            if asset not in self.assets or ('/' in asset) != (group == 'crypto'):
                raise TraderError('DATA_UNAUTHORIZED_SYMBOL')
            if protected(self.assets[asset], at - timedelta(seconds=60), at):
                raise TraderError('DATA_PROTECTED')
        return self.transport('GET', DATA_URL + path + '?' + urlencode(params), dict(self._headers), None)

    def latest(self, assets):
        if not assets or len(assets) != len(set(assets)) or any(symbol(a) not in self.assets for a in assets):
            raise TraderError('DATA_UNAUTHORIZED_SYMBOL')
        result = {}
        for crypto in (False, True):
            selected = [a for a in assets if ('/' in symbol(a)) == crypto]
            if not selected:
                continue
            prefix = '/v1beta3/crypto/us/latest/' if crypto else '/v2/stocks/'
            params = {'symbols': ','.join(symbol(a) for a in selected)}
            if not crypto:
                params['feed'] = 'iex'
            quotes = self.request('GET', prefix + ('quotes' if crypto else 'quotes/latest'), params=params)
            trades = self.request('GET', prefix + ('trades' if crypto else 'trades/latest'), params=params)
            try:
                for asset in selected:
                    q, t = quotes['quotes'][symbol(asset)], trades['trades'].get(symbol(asset))
                    bid, ask = decimal(q['bp']), decimal(q['ap'])
                    age = (self.clock() - instant(q['t'])).total_seconds()
                    if not 0 <= age <= execution_spec()['after_hours']['max_quote_age_seconds']:
                        raise TraderError('PAPER_QUOTE_STALE')
                    if bid <= 0 or ask < bid:
                        raise TraderError('PAPER_QUOTE_INVALID')
                    trade = None
                    if t:
                        price = decimal(t['p'])
                        if price <= 0 or instant(t['t']) > self.clock():
                            raise TraderError('DATA_TRADE_INVALID')
                        trade = {'price': str(price), 'at': iso(instant(t['t']))}
                    result[asset] = {'bid': str(bid), 'ask': str(ask), 'at': iso(instant(q['t'])), 'trade': trade}
            except (KeyError, TypeError, AttributeError, ValueError) as error:
                if isinstance(error, TraderError):
                    raise
                raise TraderError('PAPER_QUOTE_INVALID') from None
        return result


class QuoteFeed:
    """Lazy acquisition only when execution needs a quote; private file fallback."""
    def __init__(self, paper_grant, authorization=None, fallback=None, *, clock=now, transport=http_transport):
        self.paper_grant, self.clock, self.transport = paper_grant, clock, transport
        self.authorization = Path(authorization) if authorization else Path.home() / 'authorizations/trader-alpaca-data-v1.json'
        self.fallback = PrivateQuotes(fallback)

    def get(self, asset, at):
        if not self.authorization.is_file():
            return self.fallback.get(asset, at)
        grant = DataAuthorization.load(self.authorization, self.paper_grant)
        try:
            return DataClient(grant, clock=self.clock, transport=self.transport).latest([asset])[asset]
        except TraderError as error:
            if self.fallback.path and (error.code.startswith('DATA_HTTP_') or
                                      error.code in {'PAPER_QUOTE_INVALID', 'PAPER_QUOTE_STALE'}):
                return self.fallback.get(asset, self.clock())
            raise
