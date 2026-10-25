"""Grant-bound paper execution. Broker bodies and credentials never enter evidence.

The private, append-only intent journal precedes dispatch. Cumulative broker fills
project into lots; restarts query the deterministic client id before any POST.
An account with a position discrepancy is halted, without corrective orders.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, ROUND_DOWN
import fcntl
import json
from pathlib import Path
import shlex
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4
from zoneinfo import ZoneInfo

from scripts.trading_lab.research.store import ResearchStore
from scripts.trading_lab.sources.canonical import sha256_canonical

from .config import Authorization, TraderError, instant, iso, now, private_root, strict_json
from .data import calendar_session, label_window, protected
from . import slots
from .paper_spec import (SPEC_HASH, PREVIOUS_SPEC_HASH, V3_SPEC_HASH, V3_DECISION_HASH,
                         OPERATOR_DECISION_HASH, execution_spec, execution_preregistration)
from .execution_tiers import HALF_DOWNGRADE, selections, execution_scores
from . import retirement_spec
from .scoring import rows

PAPER_URL = 'https://paper-api.alpaca.markets'
FIRST_EXECUTION = instant('2026-10-08T12:00:00Z')
NY = ZoneInfo('America/New_York')
TERMINAL = {'filled', 'canceled', 'expired', 'rejected', 'replaced', 'done_for_day', 'internal', 'not_submitted'}
LIMITS = {'ia_actions': {'max_weight_per_name': .08, 'max_gross': .80, 'max_short_gross': .30},
          'ia_crypto': {'max_weight_per_asset': .25, 'max_gross': .50, 'short': False}}
D = Decimal
OWNER_WAIT_SECONDS = 120
INCIDENT_DECISION_HASH = '435841de1bf86af67b80a4ad908002959b0b839714f3f390d0e4f63436462753'


def decimal(value):
    try:
        result = D(str(value))
        if not result.is_finite():
            raise ValueError()
        return result
    except Exception:
        raise TraderError('PAPER_INVALID_NUMBER') from None


def symbol(asset):
    return asset.replace('-USD', '/USD')


def broker_symbol(value):
    if isinstance(value, str) and '/' not in value and value.endswith('USD'):
        return value[:-3] + '/USD'
    return value


def allocated_quantities(intent, observation):
    """Cumulative fills, in broker quantity units; largest remainders keep lots exact."""
    total = decimal(intent['order']['qty'])
    filled = decimal(observation.get('filled_qty', 0))
    fraction = filled / total if total else D(1) if observation.get('status') == 'internal' else D(0)
    values = [decimal(lot['qty']) * fraction for lot in intent['lots']]
    crypto_unit = D('.000000001') if intent.get('spec_hash') in {SPEC_HASH, PREVIOUS_SPEC_HASH, V3_SPEC_HASH, slots.SPEC_HASH, retirement_spec.SPEC_HASH} else D('.00000001')
    unit = crypto_unit if '/USD' in intent['order']['symbol'] else D(1)
    rounded = [v.quantize(unit, rounding=ROUND_DOWN) for v in values]
    signed_total = sum((decimal(l['qty']) for l in intent['lots']), D(0))
    if abs(signed_total) != total:
        raise TraderError('PAPER_LOT_ORDER_DELTA_MISMATCH')
    target = filled * (1 if signed_total >= 0 else -1) if total else signed_total
    # Alpaca whole-share auction fills cannot be fractional. Refuse corrupt observations.
    if target != target.quantize(unit):
        raise TraderError('PAPER_FILL_QUANTITY_PRECISION')
    residual = target - sum(rounded, D(0))
    step = unit if residual > 0 else -unit
    ranked = sorted(range(len(values)), key=lambda i: (-(values[i] - rounded[i]) * (1 if step > 0 else -1), intent['lots'][i]['lot_id']))
    for i in ranked[:int(abs(residual / unit))]:
        rounded[i] += step
    return rounded


@dataclass(frozen=True)
class PaperAuthorization:
    payload: dict
    identity: str
    parent: Authorization

    @classmethod
    def load(cls, path, parent):
        payload = strict_json(Path(path).read_text())
        grant = cls(payload, sha256_canonical(payload), parent)
        grant.validate()
        return grant

    def validate(self):
        p = self.payload
        if (p.get('authorization') != 'trader-alpaca-paper-v1' or p.get('extends') != 'trader-agent-v1'
                or p.get('mode') != 'ALPACA_PAPER_ONLY' or p.get('endpoint', '').split(' (')[0] != PAPER_URL):
            raise TraderError('PAPER_INVALID_GRANT')
        if set(p['accounts']) != {'ia_actions', 'ia_crypto', 'momentum'}:
            raise TraderError('PAPER_INVALID_ACCOUNTS')
        for name, account in p['accounts'].items():
            if account.get('orders') is not (name != 'momentum'):
                raise TraderError('PAPER_INVALID_ACCOUNT_PERMISSIONS')
            if not isinstance(account.get('account_suffix'), str) or len(account['account_suffix']) != 4:
                raise TraderError('PAPER_INVALID_SUFFIX')
            if not Path(account['credentials']).is_absolute():
                raise TraderError('PAPER_INVALID_CREDENTIAL_PATH')
        if len({a['account_suffix'] for a in p['accounts'].values()}) != 3:
            raise TraderError('PAPER_DUPLICATE_ACCOUNT')
        r = p['risk_limits']
        if (any(r.get(k) != v for k, v in LIMITS.items()) or r.get('max_orders_per_account_per_day') != 40
                or r.get('kill_switch') != "equity below 90% of the account's peak: flatten that account, halt its executor, raise an alert; resuming needs the operator"
                or r.get('trade_only') != 'consensus views of a COMPLETE or DEGRADED run that the reviewer did not reject; ABSTAIN, MISSING, FAILED or tainted runs never trade'):
            raise TraderError('PAPER_INVALID_RISK_GRANT')
        if instant(p['not_after']) <= instant(p['granted_at']):
            raise TraderError('PAPER_INVALID_GRANT_WINDOW')

    def check(self, at):
        self.parent.check(at)
        if not instant(self.payload['granted_at']) <= at < instant(self.payload['not_after']):
            raise TraderError('PAPER_AUTHORIZATION_EXPIRED_OR_NOT_STARTED')

    def assets(self, name):
        u = self.parent.payload['universe']
        return u['stocks'] + u['sector_etfs'] if name == 'ia_actions' else u['crypto'] if name == 'ia_crypto' else []


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TraderError('PAPER_REDIRECT_REFUSED')


def http_transport(method, url, headers, body):
    """No redirects, SDK, retries, or raw response/error logging."""
    request = Request(url, data=None if body is None else json.dumps(body).encode(), headers=headers, method=method)
    try:
        with build_opener(NoRedirect()).open(request, timeout=20) as response:
            content = response.read(2 * 1024 * 1024 + 1)
            if len(content) > 2 * 1024 * 1024:
                raise TraderError('PAPER_RESPONSE_TOO_LARGE')
            return strict_json(content.decode()) if content else None
    except HTTPError as error:
        raise TraderError('PAPER_HTTP_' + str(error.code)) from None
    except (URLError, TimeoutError, UnicodeError, json.JSONDecodeError):
        raise TraderError('PAPER_HTTP_UNAVAILABLE') from None


class PaperClient:
    def __init__(self, grant, account, *, clock=now, transport=http_transport):
        if account not in grant.payload['accounts']:
            raise TraderError('PAPER_UNKNOWN_ACCOUNT')
        grant.check(clock())
        self.grant, self.account, self.clock, self.transport = grant, account, clock, transport
        self.config = grant.payload['accounts'][account]
        values = {}
        for line in Path(self.config['credentials']).read_text().splitlines():
            line = line.strip().removeprefix('export ')
            if not line or line.startswith('#'):
                continue
            key, sep, value = line.partition('=')
            if not sep or key.strip() in values:
                raise TraderError('PAPER_INVALID_ENV')
            tokens = shlex.split(value, comments=True)
            if len(tokens) != 1:
                raise TraderError('PAPER_INVALID_ENV')
            values[key.strip()] = tokens[0]
        self.base_url = values.get('APCA_API_BASE_URL')
        if self.base_url != PAPER_URL:
            raise TraderError('PAPER_URL_REFUSED')
        if not values.get('APCA_API_KEY_ID') or not values.get('APCA_API_SECRET_KEY'):
            raise TraderError('PAPER_MISSING_CREDENTIALS')
        self._headers = {'APCA-API-KEY-ID': values['APCA_API_KEY_ID'],
                         'APCA-API-SECRET-KEY': values['APCA_API_SECRET_KEY'], 'Content-Type': 'application/json'}
        self.verified = False

    def request(self, method, path, *, params=None, body=None):
        if self.base_url != PAPER_URL:
            raise TraderError('PAPER_URL_REFUSED')
        self.grant.check(self.clock())
        if not path.startswith('/v2/') or any(c in path for c in ('?', '#', '..', '\\')):
            raise TraderError('PAPER_PATH_REFUSED')
        if method != 'GET':
            if self.account == 'momentum' or not self.config['orders']:
                raise TraderError('PAPER_MOMENTUM_READ_ONLY')
            if not self.verified:
                raise TraderError('PAPER_ACCOUNT_NOT_VERIFIED')
            if self.clock() < FIRST_EXECUTION:
                raise TraderError('PAPER_ORDERS_NOT_STARTED')
            if not ((method == 'POST' and path == '/v2/orders') or (method == 'DELETE' and path.startswith('/v2/orders/'))):
                raise TraderError('PAPER_MUTATION_REFUSED')
        url = PAPER_URL + path + ('?' + urlencode(params) if params else '')
        return self.transport(method, url, dict(self._headers), body)

    def start(self):
        self.verified = False
        account = self.request('GET', '/v2/account')
        if not isinstance(account, dict):
            raise TraderError('PAPER_ACCOUNT_SHAPE_INVALID')
        if not str(account.get('account_number', '')).endswith(self.config['account_suffix']):
            raise TraderError('PAPER_ACCOUNT_SUFFIX_MISMATCH')
        self.verified = True
        return account

    def lookup(self, client_id):
        try:
            return self.request('GET', '/v2/orders:by_client_order_id', params={'client_order_id': client_id})
        except TraderError as error:
            if error.code == 'PAPER_HTTP_404':
                return None
            raise


class PaperLedger:
    """Private event journal; the execution spec cannot be silently rebound."""
    def __init__(self, root, grant, *, clock=now):
        self.root, self.grant, self.clock = private_root(root), grant, clock
        self.store = ResearchStore(self.root / 'paper-evidence')
        bindings = self.events(event='binding')
        execution_spec()
        self.store.verify()
        self.validate_bindings(bindings)
        target = self.binding_target()
        if bindings and any(bindings[-1].get(k) != v for k, v in target.items()):
            raise TraderError('PAPER_LEDGER_BINDING_MISMATCH')
        if not bindings:
            self.append('binding', None, **target)

    def binding_target(self):
        from .service import preregistration
        from .weekend import PREREG_HASH as weekend_prereg, SPEC_HASH as weekend_spec
        parent = self.grant.parent
        return {'grant_hash': self.grant.identity, 'spec_hash': SPEC_HASH,
                'base_grant_hash': parent.base_identity or parent.identity,
                'effective_grant_hash': parent.identity, 'crypto_amendment_hash': parent.amendment_identity,
                'preregistration_hash': execution_preregistration(), 'research_preregistration_hash': preregistration(),
                'weekend_preregistration_hash': weekend_prereg, 'weekend_spec_hash': weekend_spec,
                'operator_decision_hash': OPERATOR_DECISION_HASH}

    def register_pm(self, decision_path):
        decision = strict_json(Path(decision_path).read_text())
        if sha256_canonical(decision) != slots.DECISION_HASH or instant(decision['decided_at']) > self.clock():
            raise TraderError('PM_OPERATOR_DECISION_REFUSED')
        self.grant.parent.check_slot('pm', self.clock(), preflight=True)
        with self.owner():
            target = slots.registration_target(self.grant)
            previous = self.events(event='pm_registration')
            if previous:
                if any(previous[-1].get(k) != v for k, v in target.items()):
                    raise TraderError('PM_PAPER_REGISTRATION_MISMATCH')
                return previous[-1]
            return self.append('pm_registration', None, **target,
                               reason='OPERATOR_APPROVED_ADDITIVE_PM_SLOT')

    def check_pm_registration(self):
        target = slots.registration_target(self.grant)
        previous = self.events(event='pm_registration')
        if not previous or any(previous[-1].get(k) != v for k, v in target.items()):
            raise TraderError('PM_PAPER_REGISTRATION_REQUIRED')

    def register_gpt_retirement(self, decision_path):
        decision = strict_json(Path(decision_path).read_text())
        if (sha256_canonical(decision) != retirement_spec.DECISION_HASH
                or instant(decision['decided_at']) > self.clock()):
            raise TraderError('GPT_RETIREMENT_OPERATOR_DECISION_REFUSED')
        self.grant.check(self.clock())
        with self.owner():
            target = retirement_spec.registration_target(self)
            previous = self.events(event='gpt_retirement_registration')
            if previous:
                if any(previous[-1].get(k) != v for k, v in target.items()):
                    raise TraderError('PAPER_GPT_RETIREMENT_REGISTRATION_MISMATCH')
                return previous[-1]
            return self.append('gpt_retirement_registration', None, **target,
                               reason='OPERATOR_APPROVED_FORWARD_ONLY_GPT_RETIREMENT')

    def check_gpt_retirement_registration(self):
        target = retirement_spec.registration_target(self)
        previous = self.events(event='gpt_retirement_registration')
        if not previous or any(previous[-1].get(k) != v for k, v in target.items()):
            raise TraderError('PAPER_GPT_RETIREMENT_REGISTRATION_REQUIRED')

    def register_weekend(self, decision_path):
        """Explicit additive registration; no grant/spec migration or broker request."""
        from .config import WEEKEND_DECISION_HASH
        from .weekend import registration_target
        decision = strict_json(Path(decision_path).read_text())
        if sha256_canonical(decision) != WEEKEND_DECISION_HASH:
            raise TraderError('WEEKEND_OPERATOR_DECISION_HASH_MISMATCH')
        self.grant.check(self.clock())
        self.grant.parent.check_weekend(self.clock())
        with self.owner():
            target = registration_target(self.grant)
            previous = self.events(event='weekend_registration')
            if previous:
                if any(previous[-1].get(k) != v for k, v in target.items()):
                    raise TraderError('WEEKEND_PAPER_REGISTRATION_MISMATCH')
                return previous[-1]
            return self.append('weekend_registration', None, **target,
                               reason='OPERATOR_APPROVED_SEPARATE_WEEKEND_POPULATION')

    def check_weekend_registration(self):
        from .weekend import registration_target
        target = registration_target(self.grant)
        registrations = self.events(event='weekend_registration')
        if not registrations or any(registrations[-1].get(k) != v for k, v in target.items()):
            raise TraderError('WEEKEND_PAPER_REGISTRATION_REQUIRED')

    def register_news_guard(self, decision_path):
        from .news_spec import DECISION_HASH, registration_target
        if sha256_canonical(strict_json(Path(decision_path).read_text())) != DECISION_HASH:
            raise TraderError('NEWS_OPERATOR_DECISION_HASH_MISMATCH')
        self.grant.parent.check_news(self.clock())
        with self.owner():
            target = registration_target(self.grant)
            previous = self.events(event='news_guard_registration')
            if previous:
                if any(previous[-1].get(k) != v for k, v in target.items()):
                    raise TraderError('PAPER_NEWS_GUARD_REGISTRATION_MISMATCH')
                return previous[-1]
            return self.append('news_guard_registration', None, **target,
                               reason='OPERATOR_APPROVED_ADDITIVE_SAFETY_CONTRACT_NO_REBIND')

    @staticmethod
    def validate_bindings(bindings):
        for previous, current in zip(bindings, bindings[1:]):
            if ((previous.get('spec_hash'), current.get('spec_hash')) not in {
                    ('c07ae3243d0b130c4ba9972158a2081ac3ff7fa5f280879a003915b98af5b386', PREVIOUS_SPEC_HASH),
                    (PREVIOUS_SPEC_HASH, V3_SPEC_HASH), (V3_SPEC_HASH, SPEC_HASH)}
                    or current.get('previous_binding_hash') != sha256_canonical(previous)
                    or current.get('previous_spec_hash') != previous['spec_hash']
                    or current.get('previous_grant_hash') != previous['grant_hash']
                    or current.get('operator_decision_hash') != (
                        '0caee3e1145b4cc401cccaface8ca7ecc928c6337c335ab23cc1695a3677e7a9'
                        if current.get('spec_hash') == PREVIOUS_SPEC_HASH else
                        V3_DECISION_HASH if current.get('spec_hash') == V3_SPEC_HASH else OPERATOR_DECISION_HASH)):
                raise TraderError('PAPER_LEDGER_BINDING_MISMATCH')

    @classmethod
    def rebind(cls, root, grant, decision_path, *, clock=now, client_factory=PaperClient):
        decision = strict_json(Path(decision_path).read_text())
        if sha256_canonical(decision) != OPERATOR_DECISION_HASH or instant(decision['decided_at']) > clock():
            raise TraderError('PAPER_REBIND_OPERATOR_DECISION_REFUSED')
        grant.check(clock())
        execution_spec()
        ledger = cls.__new__(cls)
        ledger.root, ledger.grant, ledger.clock = private_root(root), grant, clock
        ledger.store = ResearchStore(ledger.root / 'paper-evidence')
        with ledger.owner():
            ledger.store.verify()
            bindings = ledger.events(event='binding')
            ledger.validate_bindings(bindings)
            if not bindings or bindings[-1]['grant_hash'] != grant.identity or bindings[-1]['spec_hash'] not in {V3_SPEC_HASH, SPEC_HASH}:
                raise TraderError('PAPER_REBIND_PRIOR_BINDING_REFUSED')
            target = ledger.binding_target()
            if all(bindings[-1].get(k) == v for k, v in target.items()):
                return ledger
            if bindings[-1]['spec_hash'] != V3_SPEC_HASH:
                raise TraderError('PAPER_REBIND_PRIOR_BINDING_REFUSED')
            # This transition covers an empty ledger only; historical lots are never rebound.
            for name in ('ia_actions', 'ia_crypto'):
                intents, observed, lots = ledger.projection(name)
                if any(decimal(l['qty']) for l in lots.values()):
                    raise TraderError('PAPER_REBIND_OPEN_LOTS')
                if any(observed.get(k, {}).get('status') not in TERMINAL for k in intents):
                    raise TraderError('PAPER_REBIND_PENDING_INTENTS')
                client = client_factory(grant, name, clock=clock)
                client.start()  # verify the account suffix before relying on emptiness
                positions = client.request('GET', '/v2/positions')
                orders = client.request('GET', '/v2/orders', params={'status': 'open', 'limit': 500})
                if not isinstance(positions, list) or not isinstance(orders, list) or positions or orders:
                    raise TraderError('PAPER_REBIND_BROKER_NOT_EMPTY')
            previous = bindings[-1]
            ledger.append('binding', None, **target, previous_binding_hash=sha256_canonical(previous),
                          previous_spec_hash=previous['spec_hash'], previous_grant_hash=previous['grant_hash'],
                          reason='OPERATOR_APPROVED_EMPTY_LEDGER_REBIND')
        return ledger

    @contextmanager
    def owner(self, *, timeout=OWNER_WAIT_SECONDS, poll=1.0, sleep=time.sleep, monotonic=time.monotonic):
        """Wait for a concurrent paper job; fail closed only after the bounded wait."""
        deadline = monotonic() + timeout
        with (self.root / 'paper-owner.lock').open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TraderError('PAPER_OWNER_BUSY') from None
                    sleep(min(poll, remaining))
            yield

    def append(self, event, account, **data):
        identifier = 'paper:' + uuid4().hex
        payload = {'schema': 'alpaca-paper-event-v1', 'event_id': identifier, 'event': event, 'at': iso(self.clock()),
                   'account': account, 'account_suffix': self.grant.payload['accounts'][account]['account_suffix'] if account else None,
                   **data}
        self.store.append('replay-summary', payload, object_id=identifier, recorded_at=payload['at'])
        return payload

    def events(self, account=None, event=None):
        return [r['payload'] for r in rows(self.store, 'replay-summary')
                if (account is None or r['payload']['account'] == account) and (event is None or r['payload']['event'] == event)]

    def active_halts(self, account):
        halts = []
        for event in self.events(account):
            if event['event'] == 'resume':
                halts = []
            elif event['event'] == 'halt':
                halts.append(event)
        return halts

    def halted(self, account):
        return bool(self.active_halts(account))

    def halt(self, account, reason):
        self.append('halt', account, reason=reason)
        self.append('alert', account, code=reason)
        # Also expose the current alert alongside the append-only history.
        target = self.root / ('paper-alert-' + account + '.json')
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps({'at': iso(self.clock()), 'account': account,
                                  'account_suffix': self.grant.payload['accounts'][account]['account_suffix'], 'code': reason}))
        temp.replace(target)

    def projection(self, account):
        events = self.events(account)
        intents = {e['client_id']: e for e in events if e['event'] == 'intent'}
        observations = {e['client_id']: e for e in events if e['event'] == 'order'}
        lots = {}
        for key, intent in intents.items():
            observation = observations.get(key, {})
            allocations = allocated_quantities(intent, observation)
            if intent['purpose'] == 'entry':
                for lot, quantity in zip(intent['lots'], allocations):
                    lots[lot['lot_id']] = {**lot, 'qty': str(quantity),
                                          'entry_price': observation.get('filled_avg_price'),
                                          'entry_filled_at': observation.get('filled_at'), 'client_id': key}
            else:
                for lot, quantity in zip(intent['lots'], allocations):
                    if lot['lot_id'] in lots:
                        lots[lot['lot_id']]['qty'] = str(decimal(lots[lot['lot_id']]['qty']) - quantity)
        return intents, observations, lots

    def reserve(self, account, plan):
        day = self.clock().date().isoformat()
        count = sum(e['at'][:10] == day and decimal(e['order']['qty']) != 0 for e in self.events(account, 'intent'))
        if count >= self.grant.payload['risk_limits']['max_orders_per_account_per_day']:
            raise TraderError('PAPER_ORDER_CAP')
        return self.append('intent', account, **plan)


def after_hours_cutoff(session):
    return session.close_at.astimezone(NY).replace(hour=19, minute=30, second=0, microsecond=0).astimezone(session.close_at.tzinfo)


def order_terms(account, purpose, at, session, *, after_hours=False, pm=False, bid=None, ask=None, side='buy'):
    if account == 'ia_crypto':
        return {'type': 'market', 'time_in_force': 'gtc'}
    if after_hours:
        if not session or not session.close_at <= at < after_hours_cutoff(session):
            raise TraderError('PAPER_AFTER_HOURS_CUTOFF')
        bid, ask = decimal(bid), decimal(ask)
        mid = (bid + ask) / 2
        spec = execution_spec()['after_hours']
        if bid <= 0 or ask < bid or (ask - bid) / mid > decimal(spec['max_spread_bps']) / 10000:
            raise TraderError('PAPER_SPREAD_CAP')
        slip = decimal(spec['max_slippage_bps']) / 10000
        limit = min(ask, mid * (1 + slip)) if side == 'buy' else max(bid, mid * (1 - slip))
        rounding = 'ROUND_FLOOR' if side == 'buy' else 'ROUND_CEILING'
        return {'type': 'limit', 'time_in_force': 'day', 'extended_hours': True,
                'limit_price': str(limit.quantize(D('.01'), rounding=rounding))}
    if purpose == 'entry' and pm:
        if not session or not session.open_at <= at < session.close_at - timedelta(minutes=15):
            raise TraderError('PAPER_PM_ENTRY_WINDOW')
        return {'type': 'market', 'time_in_force': 'day'}
    if purpose == 'entry':
        if not session or not session.open_at + timedelta(seconds=30) <= at < session.open_at + timedelta(minutes=5):
            raise TraderError('PAPER_ENTRY_WINDOW')
        return {'type': 'market', 'time_in_force': 'day'}
    if purpose == 'exit':
        if not session or not (session.close_at - timedelta(minutes=5) <= at < session.close_at
                or session.open_at + timedelta(seconds=30) <= at < session.open_at + timedelta(minutes=5)):
            raise TraderError('PAPER_EXIT_WINDOW')
        return {'type': 'market', 'time_in_force': 'day'}
    return {'type': 'market', 'time_in_force': 'day'}


class PrivateQuotes:
    """Operator-fed quotes; no ungranted market-data HTTP host is contacted.

    File shape: {"quotes": {"AAPL": {"bid":..., "ask":..., "at":...}}}.
    Refuse missing/stale/future data. A broker data-host grant is a separate scope.
    """
    def __init__(self, path=None):
        self.path = Path(path).expanduser() if path else None

    def get(self, asset, at):
        if not self.path or not self.path.is_file():
            raise TraderError('PAPER_QUOTE_UNAVAILABLE')
        try:
            value = strict_json(self.path.read_text())['quotes'][asset]
            age = (at - instant(value['at'])).total_seconds()
            if not 0 <= age <= execution_spec()['after_hours']['max_quote_age_seconds']:
                raise TraderError('PAPER_QUOTE_STALE')
            return value
        except (KeyError, TypeError, json.JSONDecodeError):
            raise TraderError('PAPER_QUOTE_INVALID') from None


class PaperExecutor:
    def __init__(self, ledger, research, *, client_factory=PaperClient, quotes=None, clock=now, pause_paths=(), slot=None):
        self.ledger, self.grant, self.research = ledger, ledger.grant, research
        self.slot = slot
        self.clock, self.dispatch_clock, self.client_factory = clock, clock, client_factory
        from .alpaca_data import QuoteFeed
        self.quotes = quotes or QuoteFeed(self.grant, clock=clock)
        self.replay_prior_registration = False
        self.pause_paths = [self.ledger.root / 'PAPER_PAUSED', self.ledger.root / 'PAUSED', *map(Path, pause_paths)]
        execution_spec()

    def resume_after_reconciliation(self, decision_path, name):
        """The incident's explicit operator resume, with a fresh GET-only proof."""
        decision = strict_json(Path(decision_path).read_text())
        if (name != 'ia_actions' or sha256_canonical(decision) != INCIDENT_DECISION_HASH
                or instant(decision['decided_at']) > self.clock()):
            raise TraderError('PAPER_RESUME_OPERATOR_DECISION_REFUSED')
        with self.ledger.owner():
            self.grant.check(self.dispatch_clock())
            self.ledger.store.verify()
            self.research.verify()
            halts = self.ledger.events(name, 'halt')
            if not self.ledger.halted(name):
                return {'state': 'ALREADY_RESUMED', 'account': name}
            if any(e['reason'] != 'PAPER_RECONCILIATION_MISMATCH' or e['at'][:10] != '2026-10-16' for e in halts):
                raise TraderError('PAPER_RESUME_HALT_NOT_AUTHORIZED')
            client = self.client_factory(self.grant, name, clock=self.dispatch_clock)
            client.start()
            orders = client.request('GET', '/v2/orders', params={'status': 'all', 'limit': 500})
            if not isinstance(orders, list) or any(not isinstance(o, dict) for o in orders) or len(orders) >= 500:
                raise TraderError('PAPER_RESUME_ORDER_HISTORY_INCOMPLETE')
            intents, observed, _ = self.ledger.projection(name)
            keys = [o.get('client_order_id') for o in orders]
            if len(set(keys)) != len(keys) or any(k not in intents for k in keys):
                raise TraderError('PAPER_RESUME_UNKNOWN_ORDER')
            required = {k for k, i in intents.items() if decimal(i['order']['qty'])
                        and observed.get(k, {}).get('status') != 'not_submitted'}
            if required - set(keys):
                raise TraderError('PAPER_RESUME_ORDER_HISTORY_INCOMPLETE')
            for order in orders:
                self.observe(name, intents[order['client_order_id']], order)
            _, account, positions = self.start(name)
            if account['_paper_risk_breach'] or any(e['reason'] == 'PAPER_KILL_SWITCH'
                                                   for e in self.ledger.events(name, 'halt')):
                raise TraderError('PAPER_RESUME_RISK_BREACH')
            proof = self.ledger.events(name, 'reconciliation')[-1]
            event = self.ledger.append('resume', name, operator_decision_hash=INCIDENT_DECISION_HASH,
                halt_hashes=[sha256_canonical(e) for e in halts], reconciliation_hash=sha256_canonical(proof),
                broker_orders=len(orders), orders_digest=sha256_canonical(orders),
                positions_digest=sha256_canonical(positions), broker_methods=['GET'],
                reason='OPERATOR_APPROVED_MATCHED_INCIDENT_RECONCILIATION')
            return {'state': 'RESUMED', 'account': name, 'proof': event}

    def observe(self, name, intent, order):
        if (order.get('client_order_id') != intent['client_id'] or broker_symbol(order.get('symbol')) != intent['order']['symbol']
                or order.get('side') != intent['order']['side'] or decimal(order.get('qty')) != decimal(intent['order']['qty'])):
            raise TraderError('PAPER_ORDER_BINDING_MISMATCH')
        filled = decimal(order.get('filled_qty', 0))
        if not 0 <= filled <= decimal(intent['order']['qty']):
            raise TraderError('PAPER_FILL_INVALID')
        if filled and (order.get('filled_avg_price') is None or decimal(order['filled_avg_price']) <= 0 or not order.get('filled_at')):
            raise TraderError('PAPER_FILL_PRICE_MISSING')
        event = {'client_id': intent['client_id'], 'status': order['status'], 'filled_qty': str(filled),
                 'filled_avg_price': str(decimal(order['filled_avg_price'])) if order.get('filled_avg_price') is not None else None,
                 'filled_at': order.get('filled_at'), 'broker_order_digest': sha256_canonical(str(order.get('id')))}
        allocated_quantities(intent, event)
        old = self.ledger.projection(name)[1].get(intent['client_id'])
        if old and decimal(old['filled_qty']) > filled:
            raise TraderError('PAPER_FILL_REGRESSION')
        if not old or any(old[k] != v for k, v in event.items()):
            self.ledger.append('order', name, **event)
            self.outcomes(name, intent, event)
        if order['status'] in {'expired', 'rejected'}:
            self.order_alert(name, intent['client_id'], 'PAPER_ORDER_' + order['status'].upper())

    def order_alert(self, name, key, code):
        prior = [e for e in self.ledger.events(name, 'alert') if e.get('client_id') == key and e['code'] == code]
        if not prior:
            event = self.ledger.append('alert', name, client_id=key, code=code)
        else:
            event = prior[-1]
        target = self.ledger.root / ('paper-alert-' + name + '.json')
        if not prior or not target.exists():
            temp = target.with_suffix('.tmp')
            temp.write_text(json.dumps({k: event[k] for k in ('at', 'account', 'account_suffix', 'client_id', 'code')}))
            temp.replace(target)

    def audit(self, name):
        """No mutations: account reconciliation plus a disposition for every intent."""
        intents, observed, lots = self.ledger.projection(name)
        cancellations = {e['client_id'] for e in self.ledger.events(name, 'cancel') if e.get('reason')}
        orders = []
        for key, intent in intents.items():
            o = observed.get(key, {})
            state = o.get('status')
            if state == 'filled' or state == 'internal':
                disposition = 'FILLED' if state == 'filled' else 'SETTLED_INTERNAL'
            elif state in {'canceled', 'not_submitted'} and (key in cancellations or o.get('reason')):
                disposition = 'CANCELED_BY_RULE'
            else:
                code = 'PAPER_ORDER_' + state.upper() if state in {'expired', 'rejected'} else 'PAPER_ORDER_UNRESOLVED'
                self.order_alert(name, key, code)
                disposition = 'ALERTED'
            orders.append({'client_id': key, 'status': state, 'disposition': disposition})
        if name == 'ia_actions':
            seen = {l['prediction_hash'] for i in intents.values() if i['purpose'] == 'entry' for l in i['lots']}
            for row, _, _, _ in self.eligible_predictions(name):
                p = row['payload']
                definition = p['signal']['label_definition']
                session = calendar_session(p['signal']['session'])
                pm = p['signal'].get('slot') == 'pm'
                intent_assets = {(l['run_id'], l['asset']) for i in intents.values() if i['purpose'] == 'entry' for l in i['lots']}
                occupied = {l['asset'] for l in lots.values() if decimal(l['qty'])}
                collision = slots.collision_policy(self.grant.parent, self.clock()) and (
                    (p['signal']['run_id'], p['product']) in intent_assets or p['product'] in occupied)
                due = instant(p['decision_at']) + timedelta(minutes=10) if pm else session.open_at + timedelta(minutes=5) if session else None
                if (not collision and definition.get('variant') != 'alpaca_after_hours_v1' and due
                        and self.clock() >= due and row['identity'] not in seen):
                    self.order_alert(name, row['identity'], 'PAPER_ELIGIBLE_ENTRY_NOT_DISPATCHED')
            for lot in lots.values():
                if decimal(lot['qty']) and instant(lot['exit_at']) <= self.clock():
                    self.order_alert(name, lot['lot_id'], 'PAPER_LOT_PAST_HORIZON')
        return self.ledger.append('order_audit', name, orders=orders, checked=len(orders))

    def outcomes(self, name, intent, observation):
        allocations = allocated_quantities(intent, observation)
        intents, observed, _ = self.ledger.projection(name)
        entries = {l['lot_id']: (i, l) for i in intents.values() if i['purpose'] == 'entry' for l in i['lots']}
        for lot, quantity in zip(intent['lots'], allocations):
            entry_intent, original = entries[lot['lot_id']]
            entry = observed.get(entry_intent['client_id'], {})
            data = {'lot_id': lot['lot_id'], 'prediction_hash': lot['prediction_hash'], 'horizon': lot['horizon'],
                    'variant': lot['variant'], 'symbol': lot['symbol'], 'client_id': intent['client_id'],
                    'slot': lot.get('slot', 'am'), 'purpose': intent['purpose'], 'cumulative_lot_qty': str(quantity), 'broker_status': observation['status']}
            data['broker_fill'] = observation['status'] != 'internal' and bool(quantity)
            if intent['purpose'] == 'entry':
                data['state'] = 'FILLED' if quantity else 'NOT_TRADED' if observation['status'] in TERMINAL else 'PENDING'
                data['decision_reference'] = original['reference_price']
                data['entry_price'] = observation.get('filled_avg_price')
                data['entry_at'] = observation.get('filled_at')
                if quantity:
                    data['entry_slippage_bps_vs_decision_reference'] = str((decimal(data['entry_price']) / decimal(original['reference_price']) - 1) *
                                                                         (1 if quantity > 0 else -1) * 10000)
            elif quantity:
                entry_price, exit_price = decimal(entry['filled_avg_price']), decimal(observation['filled_avg_price'])
                data.update(state='INTERNAL_SETTLEMENT' if observation['status'] == 'internal' else 'CLOSED' if observation['status'] in TERMINAL else 'PARTIAL_EXIT', entry_price=str(entry_price),
                            exit_price=str(exit_price), raw_return=str(exit_price / entry_price - 1),
                            lot_pnl_before_broker_fees=str(quantity * (exit_price - entry_price)), exit_at=observation.get('filled_at'))
            else:
                data['state'] = 'EXIT_PENDING'
            self.ledger.append('trade_outcome', name, **data)

    def reconcile(self, client, name):
        # Orders and positions are independent broker snapshots. A partial fill
        # can advance between their GETs even under the local ownership lock.
        _, initial_observed, _ = self.ledger.projection(name)
        for _ in range(3):
            intents, observed, _ = self.ledger.projection(name)
            for key, intent in intents.items():
                if key in observed and observed[key]['status'] in TERMINAL:
                    continue
                order = client.lookup(key)
                if order:
                    self.observe(name, intent, order)
            open_orders = client.request('GET', '/v2/orders', params={'status': 'open', 'limit': 500})
            if not isinstance(open_orders, list) or any(not isinstance(o, dict) for o in open_orders):
                raise TraderError('PAPER_ORDER_SHAPE_INVALID')
            if len(open_orders) >= 500 or any(o.get('client_order_id') not in intents for o in open_orders):
                raise TraderError('PAPER_UNKNOWN_OPEN_ORDER')
            positions = client.request('GET', '/v2/positions')
            if not isinstance(positions, list) or any(not isinstance(p, dict) or
                    not {'symbol', 'qty', 'current_price', 'market_value'} <= p.keys() for p in positions):
                raise TraderError('PAPER_POSITION_SHAPE_INVALID')
            _, _, lots = self.ledger.projection(name)
            expected = {}
            for lot in lots.values():
                expected[lot['symbol']] = expected.get(lot['symbol'], D(0)) + decimal(lot['qty'])
            actual = {}
            for position in positions:
                asset = broker_symbol(position['symbol'])
                if asset in actual:
                    raise TraderError('PAPER_POSITION_SHAPE_INVALID')
                actual[asset] = decimal(position['qty'])
            mismatches = [s for s in set(expected) | set(actual) if abs(expected.get(s, D(0)) - actual.get(s, D(0))) > D('.00000001')]
            self.ledger.append('reconciliation', name, matched=not mismatches,
                               expected={k: str(v) for k, v in expected.items()}, actual={k: str(v) for k, v in actual.items()})
            if not mismatches:
                return positions
        # A moving, known fill defers this pass without enabling dispatch or
        # misclassifying it as an unknown position. Unexplained deltas still halt.
        intents, observed, _ = self.ledger.projection(name)
        lower, upper = {}, {}
        for key, intent in intents.items():
            observation = observed.get(key, {})
            initial = initial_observed.get(key, {})
            if initial.get('status') in TERMINAL and observation.get('status') in TERMINAL:
                continue
            before = decimal(initial.get('filled_qty', 0))
            filled = decimal(observation.get('filled_qty', 0))
            maximum = filled if observation.get('status') in TERMINAL else decimal(intent['order']['qty'])
            asset = intent['order']['symbol']
            sign = 1 if intent['order']['side'] == 'buy' else -1
            # Position GETs may lag a newly observed terminal fill as well.
            deltas = ((before - filled) * sign, (maximum - filled) * sign)
            lower[asset] = lower.get(asset, D(0)) + min(deltas)
            upper[asset] = upper.get(asset, D(0)) + max(deltas)
        if all(lower.get(s, D(0)) <= actual.get(s, D(0)) - expected.get(s, D(0)) <= upper.get(s, D(0))
               for s in mismatches):
            raise TraderError('PAPER_RECONCILIATION_PENDING')
        raise TraderError('PAPER_RECONCILIATION_MISMATCH')

    def start(self, name):
        client = self.client_factory(self.grant, name, clock=self.dispatch_clock)
        try:
            account = client.start()
            equity = decimal(account['equity'])
            if equity <= 0:
                raise TraderError('PAPER_INVALID_EQUITY')
            if name == 'momentum':
                positions = client.request('GET', '/v2/positions')
                return client, account, positions
            positions = self.reconcile(client, name)
            previous = self.ledger.events(name, 'account')
            peak = max([equity, D(100000)] + [decimal(e['equity']) for e in previous])
            self.ledger.append('account', name, equity=str(equity), peak=str(peak),
                               last_equity=str(decimal(account['last_equity'])), position_count=len(positions))
            marked = {broker_symbol(p['symbol']): abs(decimal(p['market_value'])) / equity for p in positions}
            limits = LIMITS[name]
            account['_paper_risk_breach'] = (sum(marked.values(), D(0)) > decimal(limits['max_gross'])
                or any(w > decimal(limits.get('max_weight_per_name', limits.get('max_weight_per_asset'))) for w in marked.values())
                or sum((abs(decimal(p['market_value'])) / equity for p in positions if decimal(p['qty']) < 0), D(0)) > decimal(limits.get('max_short_gross', 0)))
            if equity < peak * D('.90') and not any(e.get('reason') == 'PAPER_KILL_SWITCH' for e in self.ledger.active_halts(name)):
                self.ledger.halt(name, 'PAPER_KILL_SWITCH')
            return client, account, positions
        except TraderError as error:
            if error.code in {'PAPER_ACCOUNT_SUFFIX_MISMATCH', 'PAPER_ORDER_BINDING_MISMATCH', 'PAPER_FILL_INVALID',
                              'PAPER_FILL_REGRESSION', 'PAPER_FILL_QUANTITY_PRECISION', 'PAPER_FILL_PRICE_MISSING',
                              'PAPER_UNKNOWN_OPEN_ORDER', 'PAPER_RECONCILIATION_MISMATCH', 'PAPER_POSITION_SHAPE_INVALID'}:
                self.ledger.halt(name, error.code)
            raise

    def eligible_predictions(self, name):
        from .pm_retirement_spec import active
        day = self.clock().date().isoformat()
        pm_retired = active(self.clock())
        summaries = [r['payload'] for r in rows(self.research, 'replay-summary')
                     if r['payload'].get('schema') == 'trader-run-v1' and not r['payload'].get('synthetic', True)
                     and r['payload'].get('status') in {'COMPLETE', 'DEGRADED'}
                     and (not pm_retired or slots.slot_of(r['payload']) != 'pm')
                     and (self.slot is None or slots.slot_of(r['payload']) == self.slot)
                     and (r['payload']['at'][:10] == day or
                          r['payload'].get('decision', {}).get('session') == self.clock().astimezone(NY).date().isoformat())]
        if not summaries:
            return []
        run = summaries[-1]
        pm = slots.slot_of(run) == 'pm'
        if pm:
            self.grant.parent.check_slot('pm', self.clock())
            self.ledger.check_pm_registration()
        decision = run.get('decision', {})
        reviewer = decision.get('models', {}).get('reviewer', {})
        if (run.get('tainted') or decision.get('tainted') or 'TAINTED_RUN' in run.get('degraded', {}).values()
                or not reviewer or reviewer.get('cli_version') == 'NOT_RUN'):
            return []
        retired = decision.get('preregistration_hash') == retirement_spec.PREREG_HASH
        if retired:
            self.ledger.check_gpt_retirement_registration()
        if retirement_spec.active(self.clock()) and not self.replay_prior_registration and not retired:
            return []
        views = selections(decision.get('views', []), name, retired=retired)
        expected_execution = retirement_spec.preregistration() if retired else slots.preregistration() if pm else execution_preregistration()
        if not self.replay_prior_registration and decision.get('execution_preregistration_hash') != expected_execution:
            return []
        predictions = []
        for row in rows(self.research, 'prediction'):
            p = row['payload']
            view, definition = p['signal']['view'], p['signal']['label_definition']
            if p['synthetic'] or p['signal']['run_id'] != run['run_id'] or view['analyst'] not in {'analyst_claude', 'analyst_gpt', 'consensus'}:
                continue
            from .service import preregistration
            from . import weekend
            is_weekend = definition.get('variant') in {weekend.VARIANT, 'pm_weekend_crypto_v1', 'pm_weekend_crypto_recovery_v1'}
            if pm and (p['signal'].get('slot') != 'pm' or definition.get('variant') !=
                       slots.variant(is_weekend, 'pm_label_anchor' in decision) or
                       decision.get('slot') != 'pm' or decision.get('pm_amendment_hash') != self.grant.parent.pm_identity):
                raise TraderError('PAPER_PM_PREDICTION_BINDING_MISMATCH')
            expected_registration = retirement_spec.preregistration() if retired else slots.preregistration() if pm else weekend.preregistration() if is_weekend else preregistration()
            if is_weekend:
                if name != 'ia_crypto' or calendar_session(p['signal']['session']) is not None:
                    continue
                weekend.check(self.grant.parent, instant(p['decision_at']))
                self.ledger.check_weekend_registration()
            elif calendar_session(p['signal']['session']) is None:
                continue
            if not self.replay_prior_registration and (p['artifact_hash'] != expected_registration or
                    instant(p['decision_at']) < instant(retirement_spec.EFFECTIVE_FROM if retired else execution_spec()['effective_from'])):
                continue
            selection = views.get((p['product'], definition['horizon']))
            if selection is None or selection[0]['analyst'] != view['analyst']:
                continue
            selected, multiplier, tier = selection
            if selected != view:
                raise TraderError('PAPER_ANALYST_BINDING_MISMATCH')
            if (view['view'] not in {'UP', 'DOWN'} or view['verdict'] not in ({'KEEP', 'DOWNGRADE'} if name == 'ia_actions' else {'KEEP'})
                    or p['risk'].get('tainted') or definition.get('variant') == 'catchup_close_entry_v1'):
                continue
            if definition.get('variant') not in {None, 'alpaca_after_hours_v1', weekend.VARIANT, 'pm_slot_v1', 'pm_weekend_crypto_v1', 'pm_slot_recovery_v1', 'pm_weekend_crypto_recovery_v1'}:
                continue
            if definition['horizon'] not in {'1d', '5d'}:
                raise TraderError('PAPER_LABEL_BINDING_MISMATCH')
            session_day = p['signal'].get('session', instant(p['decision_at']).astimezone(NY).date().isoformat())
            anchor = None
            if pm and 'pm_label_anchor' in decision:
                started = instant(decision['pm_recovery_started_at'])
                anchor = slots.recovery_anchor(session_day, started, is_weekend)
                if (instant(decision['pm_label_anchor']) != anchor or
                        not slots.start_at(session_day, 'pm') <= started <= instant(p['decision_at']) < anchor):
                    raise TraderError('PAPER_PM_RECOVERY_ANCHOR_MISMATCH')
            window = slots.label_window if pm else label_window
            expected_entry, expected_exit = window(p['product'], session_day, definition['horizon'],
                self.grant.parent.payload['universe']['crypto'], *([] if pm else ['close' if definition.get('variant') == 'alpaca_after_hours_v1' else 'open']), **({'anchor': anchor} if pm else {}))
            if (instant(definition['exit_at']) != expected_exit or
                    ((pm or not definition.get('variant') or is_weekend) and instant(definition['entry_at']) != expected_entry)):
                raise TraderError('PAPER_LABEL_BINDING_MISMATCH')
            if (name == 'ia_actions' and not definition.get('variant')
                    and (instant(p['decision_at']) >= expected_entry or instant(row['recorded_at']) >= expected_entry)):
                continue
            if is_weekend and (instant(p['decision_at']) >= expected_entry or instant(row['recorded_at']) >= expected_entry):
                raise TraderError('PAPER_LABEL_BINDING_MISMATCH')
            if instant(p['decision_at']) > self.clock() or instant(row['recorded_at']) > self.clock():
                continue
            if instant(definition['exit_at']) >= min(instant(self.grant.payload['not_after']),
                                                     instant(self.grant.parent.payload['not_after'])):
                continue   # Never open a lot whose scheduled exit is outside the authorization.
            evidence = self.research.records('inputs', parent_id=row['identity'])
            if len(evidence) != 1 or evidence[0]['payload']['input_quality'].get('state') != 'AVAILABLE':
                continue
            if protected(p['product'], instant(p['decision_at']), instant(definition['exit_at'])):
                continue
            predictions.append((row, evidence[0]['payload'], multiplier, tier))
        return predictions

    def entries(self, name, equity, positions):
        from .downgrade_guard import evaluate
        guard = evaluate(self.ledger, persist=False) if name == 'ia_actions' else {}
        intents, observations, live = self.ledger.projection(name)
        seen = {l['lot_id'] for i in intents.values() if i['purpose'] == 'entry' for l in i['lots']}
        prices = {broker_symbol(p['symbol']): decimal(p['current_price']) for p in positions}
        reserved = dict(live)
        for key, intent in intents.items():
            if intent['purpose'] == 'entry' and observations.get(key, {}).get('status') not in TERMINAL:
                for lot in intent['lots']:
                    reserved[lot['lot_id']] = lot
        gross, short, weights = D(0), D(0), {}
        for lot in reserved.values():
            weight = decimal(lot['qty']) * prices.get(lot['symbol'], decimal(lot['reference_price'])) / equity
            gross += abs(weight)
            short += max(-weight, D(0))
            weights[lot['symbol']] = weights.get(lot['symbol'], D(0)) + abs(weight)
        candidates = []
        occupied = {l['symbol'] for l in reserved.values() if decimal(l['qty'])}
        collisions = slots.collision_policy(self.grant.parent, self.clock())
        for row, evidence, multiplier, tier in self.eligible_predictions(name):
            if tier == HALF_DOWNGRADE and guard.get('state') == 'HALTED':
                continue
            p = row['payload']
            if p['product'] not in self.grant.assets(name):
                continue
            if name == 'ia_crypto' and (self.clock() - instant(p['decision_at'])).total_seconds() > 300:
                continue
            pm = p['signal'].get('slot') == 'pm' or ':pm' in p['signal']['run_id']
            if collisions and symbol(p['product']) in occupied:
                continue
            if name == 'ia_actions' and pm:
                session = calendar_session(p['signal']['session'])
                if not session or not session.open_at <= self.clock() < session.close_at - timedelta(minutes=15):
                    continue
            if name == 'ia_actions' and not pm and p['signal']['label_definition'].get('variant') != 'alpaca_after_hours_v1':
                session = calendar_session(p['signal']['session'])
                if not session or not session.open_at + timedelta(seconds=30) <= self.clock() < session.open_at + timedelta(minutes=5):
                    continue
            v, definition = p['signal']['view'], p['signal']['label_definition']
            probability = decimal(v['p_outperform'])
            direction = 1 if v['view'] == 'UP' else -1
            directional = probability if direction > 0 else 1 - probability
            if (directional <= D('.5') or directional > D('.80')
                    or (name == 'ia_crypto' and (directional < D('.55') or direction < 0))):
                continue
            lot_id = sha256_canonical({'run_id': p['signal']['run_id'], 'asset': p['product'], 'horizon': definition['horizon']})
            if lot_id in seen:
                continue
            price = decimal(evidence['snapshot']['price']['recent_closes'][-1])
            if definition.get('variant') == 'alpaca_after_hours_v1':
                q = self.quotes.get(p['product'], self.clock())
                price = (decimal(q['bid']) + decimal(q['ask'])) / 2
            if price <= 0:
                raise TraderError('PAPER_INVALID_REFERENCE_PRICE')
            horizon_fraction = D('.5') if definition['horizon'] == '1d' else D('.1')
            name_cap = decimal(LIMITS[name].get('max_weight_per_name', LIMITS[name].get('max_weight_per_asset')))
            weight = name_cap * horizon_fraction * abs(probability - D('.5')) / D('.30')
            weight = min(weight, max(D(0), name_cap - weights.get(symbol(p['product']), D(0))))
            candidates.append({'lot_id': lot_id, 'run_id': p['signal']['run_id'], 'asset': p['product'], 'symbol': symbol(p['product']),
                               'session': p['signal'].get('session', instant(p['decision_at']).astimezone(NY).date().isoformat()),
                               'horizon': definition['horizon'], 'exit_at': definition['exit_at'], 'entry_at': definition['entry_at'],
                               'prediction_hash': row['identity'], 'reference_price': str(price), 'weight': weight,
                               **({'execution_spec_hash': retirement_spec.SPEC_HASH}
                                  if p['artifact_hash'] == retirement_spec.PREREG_HASH else {}),
                               'direction': direction, 'size_multiplier': str(multiplier), 'slot': 'pm' if pm else 'am',
                               'population': 'weekend_crypto_v1' if definition.get('variant') in {'weekend_crypto_v1', 'pm_weekend_crypto_v1', 'pm_weekend_crypto_recovery_v1'}
                               else 'after_hours' if definition.get('variant') == 'alpaca_after_hours_v1' else 'weekday',
                               'variant': tier or ((definition['variant'] if pm else 'alpaca_after_hours_v2' if definition.get('variant') else 'alpaca_open_entry_v2')
                               if name == 'ia_actions' else definition.get('variant', 'alpaca_open_entry_v1'))})
        candidates.sort(key=lambda l: (l['asset'], l['horizon']))
        total = sum((l['weight'] for l in candidates), D(0))
        short_total = sum((l['weight'] for l in candidates if l['direction'] < 0), D(0))
        scale = min(D(1), max(D(0), decimal(LIMITS[name]['max_gross']) - gross) / total) if total else D(0)
        if short_total:
            scale = min(scale, max(D(0), decimal(LIMITS[name]['max_short_gross']) - short) / short_total)
        result = []
        for lot in candidates:
            if collisions and lot['symbol'] in occupied:
                continue
            cap = decimal(LIMITS[name].get('max_weight_per_name', LIMITS[name].get('max_weight_per_asset')))
            remaining = max(D(0), cap - weights.get(lot['symbol'], D(0)))
            weight = min(lot.pop('weight') * scale, remaining)
            # Fixed 10% price reserve; marked risk remains checked on every start.
            qty = (weight * equity / (decimal(lot['reference_price']) * D('1.10'))).quantize(D(1) if name == 'ia_actions' else D('.00000001'), rounding=ROUND_DOWN)
            qty = (qty * decimal(lot['size_multiplier'])).quantize(D(1) if name == 'ia_actions' else D('.00000001'), rounding=ROUND_DOWN)
            weights[lot['symbol']] = weights.get(lot['symbol'], D(0)) + qty * decimal(lot['reference_price']) / equity
            lot['qty'] = str(qty * lot.pop('direction'))
            if qty:
                result.append(lot)
                occupied.add(lot['symbol'])
        return result

    def plans(self, name, lots, purpose):
        groups = {}
        for lot in lots:
            groups.setdefault(lot['symbol'], []).append(lot)
        plans = []
        at = self.clock()
        for asset, group in sorted(groups.items()):
            net = sum((decimal(l['qty']) for l in group), D(0)) * (1 if purpose == 'entry' else -1)
            run = group[0]['run_id'] if purpose == 'entry' else ':'.join(sorted(l['lot_id'] for l in group))
            identity = {'run_id': run, 'symbol': asset, 'purpose': purpose}
            if name == 'ia_actions' and purpose == 'exit':
                intents, observed, _ = self.ledger.projection(name)
                lot_ids = {l['lot_id'] for l in group}
                previous = sorted(k for k, i in intents.items() if i['purpose'] != 'entry'
                                  and observed.get(k, {}).get('status') in TERMINAL
                                  and lot_ids.intersection(l['lot_id'] for l in i['lots']))
                if previous:
                    identity['previous_exits'] = previous
            key = sha256_canonical(identity)[:40]
            side = 'buy' if net >= 0 else 'sell'
            after = purpose == 'entry' and any(l.get('population') == 'after_hours' or l['variant'] in {'alpaca_after_hours_v1', 'alpaca_after_hours_v2'} for l in group)
            session = calendar_session(group[0]['session'] if after else at.astimezone(NY).date().isoformat())
            q = self.quotes.get(asset, at) if after else None
            terms = order_terms(name, purpose, at, session, after_hours=after, pm=any(l.get('slot') == 'pm' for l in group),
                                bid=q['bid'] if q else None, ask=q['ask'] if q else None, side=side)
            if after:
                for lot in group:
                    lot['reference_price'] = str((decimal(q['bid']) + decimal(q['ask'])) / 2)
            plans.append({'client_id': key, 'run_id': run, 'purpose': purpose, 'lots': group,
                          'spec_hash': retirement_spec.SPEC_HASH if any(l.get('execution_spec_hash') == retirement_spec.SPEC_HASH for l in group)
                          else slots.SPEC_HASH if any(l.get('slot') == 'pm' for l in group) else SPEC_HASH, 'order': {'symbol': asset, 'qty': str(abs(net)), 'side': side,
                                                         **terms, 'client_order_id': key}})
        return plans

    def submit(self, client, name, intent, *, dry_run):
        order = client.lookup(intent['client_id']) if decimal(intent['order']['qty']) else None
        if order:
            if not dry_run:
                self.observe(name, intent, order)
            return
        if name == 'ia_actions' and intent['purpose'] == 'entry' and any(l['variant'] == HALF_DOWNGRADE for l in intent['lots']):
            from .downgrade_guard import evaluate
            if evaluate(self.ledger, persist=not dry_run).get('state') == 'HALTED':
                return
        if dry_run:
            return
        at = self.clock()
        pm = any(l.get('slot') == 'pm' for l in intent['lots'])
        if pm:
            # Retirement closes entry permission; existing lots still exit
            # against their original, explicitly registered PM binding.
            if intent['purpose'] == 'entry':
                self.grant.parent.check_slot('pm', at)
            self.ledger.check_pm_registration()
        if intent['purpose'] in {'entry', 'exit'}:
            session_day = intent['lots'][0]['session'] if intent['purpose'] == 'entry' else instant(intent['at']).astimezone(NY).date().isoformat()
            session = calendar_session(session_day)
            after = bool(intent['order'].get('extended_hours'))
            if name == 'ia_actions':
                try:
                    if at.astimezone(NY).date().isoformat() != session_day:
                        raise TraderError('PAPER_STALE_INTENT')
                    q = self.quotes.get(intent['order']['symbol'], at) if after else None
                    terms = order_terms(name, intent['purpose'], at, session, after_hours=after, pm=pm,
                        bid=q['bid'] if q else None, ask=q['ask'] if q else None, side=intent['order']['side'])
                    if intent['purpose'] == 'exit' and any(instant(l['exit_at']) > at + timedelta(minutes=5) for l in intent['lots']):
                        raise TraderError('PAPER_EXIT_BEFORE_HORIZON')
                    if not after and intent['order']['time_in_force'] != 'day':
                        raise TraderError('PAPER_LEGACY_AUCTION_INTENT')
                    if after and terms != {k: intent['order'][k] for k in terms}:
                        raise TraderError('PAPER_QUOTE_MOVED_REPLAN_REQUIRED')
                except TraderError as error:
                    self.ledger.append('order', name, client_id=intent['client_id'], status='not_submitted', filled_qty='0',
                                       filled_avg_price=None, filled_at=None, broker_order_digest=None, reason=error.code)
                    self.outcomes(name, intent, self.ledger.projection(name)[1][intent['client_id']])
                    raise error
            elif intent['purpose'] == 'entry' and (at - instant(intent['at'])).total_seconds() > 300:
                self.ledger.append('order', name, client_id=intent['client_id'], status='not_submitted', filled_qty='0',
                                   filled_avg_price=None, filled_at=None, broker_order_digest=None)
                self.outcomes(name, intent, self.ledger.projection(name)[1][intent['client_id']])
                raise TraderError('PAPER_STALE_INTENT')
        if not decimal(intent['order']['qty']):
            observation = {'client_id': intent['client_id'], 'status': 'not_submitted' if intent['purpose'] == 'entry' else 'internal',
                           'filled_qty': '0', 'filled_avg_price': intent['lots'][0]['reference_price'],
                           'filled_at': iso(self.clock()), 'broker_order_digest': None}
            self.ledger.append('order', name, **observation)
            self.outcomes(name, intent, observation)
            return
        if any(p.exists() for p in self.pause_paths):
            raise TraderError('PAPER_PAUSED')
        try:
            result = client.request('POST', '/v2/orders', body=intent['order'])
            self.observe(name, intent, result)
        except TraderError as error:
            self.ledger.append('dispatch_error', name, client_id=intent['client_id'], code=error.code)
            self.ledger.append('alert', name, code=error.code)
            # Unknown outcomes are left reserved; next start looks up the client id.
            raise

    def flatten(self, client, name, positions, *, dry_run):
        intents, observations, lots = self.ledger.projection(name)
        canceled = False
        for key, intent in intents.items():
            if observations.get(key, {}).get('status') in TERMINAL:
                continue
            order = client.lookup(key)
            if order and order['status'] not in TERMINAL:
                if not dry_run:
                    client.request('DELETE', '/v2/orders/' + quote(str(order['id']), safe=''))
                    self.ledger.append('cancel', name, client_id=key)
                    final = client.lookup(key)
                    if not final or final['status'] not in TERMINAL:
                        self.ledger.append('alert', name, code='PAPER_KILL_CANCEL_PENDING')
                        return []
                    self.observe(name, intent, final)
                    canceled = True
                else:
                    return []
        if canceled:
            client, _, positions = self.start(name)
            _, _, lots = self.ledger.projection(name)
        live = [l for l in lots.values() if decimal(l['qty'])]
        plans = self.plans(name, live, 'kill')
        return plans

    def expire_after_hours(self, client, name, *, dry_run):
        intents, observations, _ = self.ledger.projection(name)
        changed = False
        for key, intent in intents.items():
            if (not intent['order'].get('extended_hours') or observations.get(key, {}).get('status') in TERMINAL
                    or self.clock() < after_hours_cutoff(calendar_session(intent['lots'][0]['session']))):
                continue
            order = client.lookup(key)
            if order and order['status'] not in TERMINAL:
                if dry_run:
                    continue
                client.request('DELETE', '/v2/orders/' + quote(str(order['id']), safe=''))
                self.ledger.append('cancel', name, client_id=key, reason='AFTER_HOURS_CUTOFF')
                order = client.lookup(key)
                if not order or order['status'] not in TERMINAL:
                    raise TraderError('PAPER_AFTER_HOURS_CANCEL_PENDING')
            if order and not dry_run:
                self.observe(name, intent, order)
                changed = True
        return changed

    def cancel_stale_regular(self, client, name, *, dry_run):
        """Finish old reservations before any exit or replacement can be planned."""
        if name != 'ia_actions' or dry_run:
            return False
        intents, observed, _ = self.ledger.projection(name)
        changed = False
        for key, intent in intents.items():
            if intent['order'].get('extended_hours') or observed.get(key, {}).get('status') in TERMINAL:
                continue
            session = calendar_session(intent['lots'][0]['session'] if intent['purpose'] == 'entry'
                                       else instant(intent['at']).astimezone(NY).date().isoformat())
            deadline = (min(instant(intent['at']) + timedelta(minutes=5), session.close_at - timedelta(minutes=15))
                        if intent['purpose'] == 'entry' and any(l.get('slot') == 'pm' for l in intent['lots']) else
                        session.open_at + timedelta(minutes=5) if intent['purpose'] == 'entry' else session.close_at)
            if self.clock() < deadline:
                continue
            order = client.lookup(key)
            if order and order['status'] not in TERMINAL:
                client.request('DELETE', '/v2/orders/' + quote(str(order['id']), safe=''))
                self.ledger.append('cancel', name, client_id=key, reason='REGULAR_ORDER_WINDOW_ENDED')
                order = client.lookup(key)
                if not order or order['status'] not in TERMINAL:
                    raise TraderError('PAPER_STALE_CANCEL_PENDING')
            if order:
                self.observe(name, intent, order)
            else:
                self.ledger.append('order', name, client_id=key, status='not_submitted', filled_qty='0',
                                   filled_avg_price=None, filled_at=None, broker_order_digest=None,
                                   reason='REGULAR_ORDER_WINDOW_ENDED')
                self.outcomes(name, intent, self.ledger.projection(name)[1][key])
            changed = True
        return changed

    def run_account(self, name, action, *, dry_run=False):
        client, account, positions = self.start(name)
        from .downgrade_guard import evaluate
        guard = evaluate(self.ledger, persist=not dry_run) if name == 'ia_actions' else {}
        if guard.get('state') == 'HALTED' and action not in {'status', 'audit'}:
            # Cancel already accepted, still-open entry orders containing this
            # tier. A mixed broker order cannot be edited without changing its
            # registered intent, so cancellation covers that entire order.
            intents, observed, _ = self.ledger.projection(name)
            for key, intent in intents.items():
                if (intent['purpose'] != 'entry' or observed.get(key, {}).get('status') in TERMINAL
                        or not any(l['variant'] == HALF_DOWNGRADE for l in intent['lots'])):
                    continue
                order = client.lookup(key)
                if order and order['status'] not in TERMINAL and not dry_run:
                    client.request('DELETE', '/v2/orders/' + quote(str(order['id']), safe=''))
                    self.ledger.append('cancel', name, client_id=key, reason='PAPER_DOWNGRADE_TIER_HALTED')
                    order = client.lookup(key)
                    if not order or order['status'] not in TERMINAL:
                        raise TraderError('PAPER_DOWNGRADE_CANCEL_PENDING')
                if order and not dry_run:
                    self.observe(name, intent, order)
                elif not order and not dry_run:
                    self.ledger.append('order', name, client_id=key, status='not_submitted', filled_qty='0',
                                       filled_avg_price=None, filled_at=None, broker_order_digest=None)
                    self.outcomes(name, intent, self.ledger.projection(name)[1][key])
            # Cancellations may change filled quantities and broker positions.
            if not dry_run:
                client, account, positions = self.start(name)
        if action not in {'status', 'audit'} and self.expire_after_hours(client, name, dry_run=dry_run):
            client, account, positions = self.start(name)
        if action not in {'status', 'audit'} and self.cancel_stale_regular(client, name, dry_run=dry_run):
            client, account, positions = self.start(name)
        equity = decimal(account['equity'])
        intents, observations, lots = self.ledger.projection(name)
        peak = max([equity, D(100000)] + [decimal(e['equity']) for e in self.ledger.events(name, 'account')])
        status = {'account': name, 'account_suffix': client.config['account_suffix'], 'equity': str(equity),
                  'pnl': str(equity - D(100000)), 'day_pnl': str(equity - decimal(account['last_equity'])),
                  'peak': str(peak), 'open_lots': sum(decimal(l['qty']) != 0 for l in lots.values()),
                  'halted': self.ledger.halted(name), 'planned_orders': []}
        status['downgrade_guard'] = guard
        if action == 'audit':
            if not dry_run:
                status['audit'] = self.audit(name)
            return status
        if action == 'status':
            return status
        if self.ledger.halted(name):
            reasons = {e['reason'] for e in self.ledger.active_halts(name)}
            if reasons == {'PAPER_KILL_SWITCH'}:
                plans = self.flatten(client, name, positions, dry_run=dry_run)
            else:
                return status
        elif action == 'execute':
            if account['_paper_risk_breach']:
                raise TraderError('PAPER_MARKED_RISK_CAP')
            if account.get('trading_blocked') or account.get('status') != 'ACTIVE':
                raise TraderError('PAPER_BROKER_TRADING_BLOCKED')
            plans = self.plans(name, self.entries(name, equity, positions), 'entry')
            # Reserve today's close budget before admitting entries. A 1d lot needs
            # an entry and an exit, within the same 40-order account allowance.
            today = self.clock().date().isoformat()
            used = sum(i['at'][:10] == today and decimal(i['order']['qty']) != 0 for i in intents.values())
            prospective = dict(lots)
            for key, intent in intents.items():
                if intent['purpose'] == 'entry' and observations.get(key, {}).get('status') not in TERMINAL:
                    prospective.update({l['lot_id']: l for l in intent['lots']})
            due_symbols = {l['symbol'] for l in prospective.values() if decimal(l['qty']) and
                           (l['exit_at'][:10] <= today if name == 'ia_actions' else l['exit_at'][:10] == today)}
            already_reserved = {i['order']['symbol'] for key, i in intents.items() if i['purpose'] != 'entry'
                                and i['at'][:10] == today and observations.get(key, {}).get('status') not in TERMINAL}
            due_symbols -= already_reserved
            selected, skipped = [], []
            for plan in plans:
                extra_due = {l['symbol'] for l in plan['lots'] if l['exit_at'][:10] == today} if decimal(plan['order']['qty']) else set()
                cost = int(decimal(plan['order']['qty']) != 0)
                if used + cost + len(due_symbols | extra_due) <= 40:
                    selected.append(plan)
                    used += cost
                    due_symbols |= extra_due
                else:
                    skipped.append(plan['order']['symbol'])
                    if not dry_run:
                        self.order_alert(name, plan['client_id'], 'PAPER_ENTRY_ORDER_BUDGET_SKIPPED')
            plans = selected
            status['budget_skipped_symbols'] = skipped
            status['orders_reserved_for_close'] = len(due_symbols)
            for plan in plans:
                if name == 'ia_actions' and plan['order']['side'] == 'sell':
                    asset = client.request('GET', '/v2/assets/' + quote(plan['order']['symbol'], safe=''))
                    if not account.get('shorting_enabled') or not asset.get('shortable') or not asset.get('tradable'):
                        raise TraderError('PAPER_SHORT_NOT_ALLOWED')
        else:
            due = [l for l in lots.values() if decimal(l['qty']) and instant(l['exit_at']) -
                   (timedelta(minutes=5) if name == 'ia_actions' else timedelta()) <= self.clock()]
            reserved_exits = {l['lot_id'] for k, i in intents.items() if i['purpose'] != 'entry'
                              and observations.get(k, {}).get('status') not in TERMINAL for l in i['lots']}
            due = [l for l in due if l['lot_id'] not in reserved_exits]
            if name == 'ia_actions':
                session = calendar_session(self.clock().astimezone(NY).date().isoformat())
                in_window = session and (session.close_at - timedelta(minutes=5) <= self.clock() < session.close_at
                    or session.open_at + timedelta(seconds=30) <= self.clock() < session.open_at + timedelta(minutes=5))
                if not in_window:
                    for lot in due:
                        if instant(lot['exit_at']) <= self.clock():
                            self.order_alert(name, lot['lot_id'], 'PAPER_LOT_PAST_HORIZON')
                    due = []
            plans = self.plans(name, due, 'exit')
        # Resume durable dispatches within their registered window; stale equity exits get explicit retry identities.
        recover_purpose = 'kill' if self.ledger.halted(name) else 'entry' if action == 'execute' else 'exit'
        pending = [i for k, i in intents.items() if k not in observations and i['purpose'] == recover_purpose]
        plans = pending + [p for p in plans if p['client_id'] not in intents]
        day_count = sum(e['at'][:10] == self.clock().date().isoformat() and decimal(e['order']['qty']) != 0
                        for e in self.ledger.events(name, 'intent'))
        if day_count + sum(p['client_id'] not in intents and decimal(p['order']['qty']) != 0 for p in plans) > 40:
            raise TraderError('PAPER_ORDER_CAP')
        for plan in plans:
            if (plan['purpose'] == 'entry' and guard.get('state') == 'HALTED'
                    and any(l['variant'] == HALF_DOWNGRADE for l in plan['lots'])):
                # Includes durable intents that crashed before submission. Exits
                # and full KEEP entries remain enabled, with original identities.
                continue
            if plan['client_id'] not in intents and not dry_run:
                self.ledger.reserve(name, plan)
            status['planned_orders'].append({k: v for k, v in plan['order'].items() if k != 'client_order_id'})
            if not dry_run:
                # Newly reserved plans carry their durable dispatch time as well.
                plan = self.ledger.projection(name)[0][plan['client_id']]
            self.submit(client, name, plan, dry_run=dry_run)
        return status

    def run(self, action, *, dry_run=False, accounts=None, replay_at=None):
        if replay_at is not None and (not dry_run or action != 'execute'):
            raise TraderError('PAPER_REPLAY_REQUIRES_DRY_RUN')
        original_clock = self.clock
        try:
            if replay_at is not None:
                self.clock = lambda: instant(replay_at)
                self.replay_prior_registration = True
            result = self._run(action, dry_run=dry_run, accounts=accounts)
            if replay_at is not None:
                result.update(replay_at=replay_at, observed_at=iso(original_clock()), no_order=True)
            return result
        finally:
            self.clock = original_clock
            self.replay_prior_registration = False

    def _run(self, action, *, dry_run=False, accounts=None):
        if action not in {'status', 'execute', 'exit', 'audit'}:
            raise TraderError('PAPER_UNKNOWN_ACTION')
        with self.ledger.owner():
            self.grant.check(self.dispatch_clock())
            self.research.verify()
            self.ledger.store.verify()
            result = {'schema': 'alpaca-paper-' + action + '-v1', 'at': iso(self.clock()), 'dry_run': dry_run, 'accounts': [], 'state': 'COMPLETE'}
            if action != 'status' and not dry_run and self.clock() < FIRST_EXECUTION:
                return {**result, 'state': 'NOT_STARTED', 'first_execution': iso(FIRST_EXECUTION)}
            if action != 'status' and any(p.exists() for p in self.pause_paths):
                return {**result, 'state': 'PAUSED'}
            if action in {'execute', 'exit'} and calendar_session(self.clock().date().isoformat()) is None:
                accounts = [a for a in (accounts or ('ia_actions', 'ia_crypto')) if a == 'ia_crypto']
                if not accounts:
                    return {**result, 'state': 'SKIPPED_CLOSED_EQUITY_MARKET'}
            for name in accounts or ('ia_actions', 'ia_crypto'):
                if name not in {'ia_actions', 'ia_crypto'}:
                    raise TraderError('PAPER_MOMENTUM_READ_ONLY')
                try:
                    result['accounts'].append(self.run_account(name, action, dry_run=dry_run))
                except TraderError as error:
                    self.ledger.append('alert', name, code=error.code)
                    result['state'] = 'BLOCKED'
                    result['accounts'].append({'account': name, 'account_suffix': self.grant.payload['accounts'][name]['account_suffix'],
                                               'state': 'BLOCKED', 'reason': error.code, 'planned_orders': []})
            if action == 'status':
                try:
                    _, momentum, positions = self.start('momentum')
                    result['claude_book'] = {'account_suffix': self.grant.payload['accounts']['momentum']['account_suffix'],
                                          'equity': str(decimal(momentum['equity'])), 'last_equity': str(decimal(momentum['last_equity'])),
                                          'position_count': len(positions), 'mode': 'GET_ONLY', 'included_in_ai': False, 'baseline': False}
                except TraderError as error:
                    self.ledger.append('alert', 'momentum', code=error.code, display_account='claude_book')
                    result['claude_book'] = {'state': 'BLOCKED', 'reason': error.code}
                    result['state'] = 'BLOCKED'
            return result

    def report(self, benchmarks=None):
        result = self.run('status')
        accounts = [a for a in result['accounts'] if 'equity' in a]
        result['complet'] = {'base': '200000', 'equity': str(sum((decimal(a['equity']) for a in accounts), D(0))) if len(accounts) == 2 else None,
                             'pnl': str(sum((decimal(a['pnl']) for a in accounts), D(0))) if len(accounts) == 2 else None,
                             'open_lots': sum(a['open_lots'] for a in accounts)}
        for account in accounts:
            account['return_since_paper_start'] = str(decimal(account['pnl']) / 100000)
        result['complet']['return_since_paper_start'] = str(decimal(result['complet']['pnl']) / 200000) if len(accounts) == 2 else None
        result['benchmarks'] = benchmarks or {s: {'state': 'PENDING'} for s in ('SPY', 'QQQ', 'BTC')}
        outcomes = self.ledger.events(event='trade_outcome')
        latest = {(e['client_id'], e['lot_id']): e for e in outcomes}
        result['execution_outcomes'] = list(latest.values())
        result['execution_scores'] = execution_scores(self.ledger)
        result['execution_scores_by_slot'] = {slot: execution_scores(self.ledger, slot=slot) for slot in ('am', 'pm')}
        result['execution_scores_combined'] = result['execution_scores']
        from .execution_gaps import execution_gaps
        result['execution_gap_scores'] = execution_gaps(self.ledger, self.research, self.clock())
        result['downgrade_guard'] = next((a.get('downgrade_guard') for a in accounts if a['account'] == 'ia_actions'), None)
        result['limitations'] = ['After-hours paper fills in thin liquidity are optimistic.',
                                'Crypto market entry follows the decision; its label uses the registered am/pm anchor.',
                                'DAY fills can differ from label open/close; marked risk caps remain enforced before entries.']
        returns = {s: b.get('return_since_paper_start') for s, b in result['benchmarks'].items()}
        result['versus'] = {a['account']: {s: str(decimal(a['return_since_paper_start']) - decimal(value)) if value is not None else None
                           for s, value in returns.items()} for a in accounts}
        result['versus']['complet'] = {s: str(decimal(result['complet']['return_since_paper_start']) - decimal(value))
                                      if len(accounts) == 2 and value is not None else None for s, value in returns.items()}
        self.ledger.append('report', None, report=result)
        return result
