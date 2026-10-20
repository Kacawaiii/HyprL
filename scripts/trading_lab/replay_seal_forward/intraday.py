"""One prospective recovery session: seal inputs, decide, publish, then enter after T+30s."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
from pathlib import Path
import re
import time

from scripts.trading_lab.replay_bench import marketdata as md, snapshot as sn, run as parent
from scripts.trading_lab.replay_bench.grant import ReplayError, instant, iso, now
from scripts.trading_lab.replay_bench.storage import atomic_json
from scripts.trading_lab.sources.canonical import sha256_canonical
from . import collect_plan, commitment, experiment as ex, forward, schemas, spec


def clock_record(root, day):
    value = json.loads((forward.directory(root, day) / 'decision-clock.json').read_text())
    check = dict(value)
    if check.pop('clock_hash') != sha256_canonical(check):
        raise ReplayError('INTRADAY_CLOCK_CHANGED')
    return value


def deadline(root, day):
    return instant(clock_record(root, day)['entry_after'])


def run(root, client, *, expected_commit, grants, alpaca_env, clock=now, ledger=None, sleep=time.sleep):
    at = clock()
    day = at.astimezone(md.NEW_YORK).date().isoformat()
    rule = spec.spec()['night_fixes_amendment']['intraday']
    sessions = md.sessions(day, day)
    if day != rule['session'] or not sessions or not sessions[0].open_at <= at < sessions[0].close_at:
        raise ReplayError('NOT_INTRADAY_SESSION')
    ex.committed()
    if ex.head() != expected_commit:
        raise ReplayError('INSTALLED_SHA_MISMATCH')
    if json.loads(ex.FREEZE.read_text()) != ex.frozen_payload(root):
        raise ReplayError('TEST_CONFIG_NOT_FROZEN')
    base = forward.directory(root, day)
    if (base / 'snapshot.json.gz').exists() or (base / 'dispatch.json').exists():
        raise ReplayError('INTRADAY_ALREADY_ATTEMPTED')
    if ledger is None:
        ledger = commitment.from_environment(root)
    cutoff = at + timedelta(seconds=120)
    atomic_json(base / 'dispatch.json', {'state': 'INTENT', 'at': iso(at), 'input_cutoff': iso(cutoff),
                                       'spec_hash': spec.SPEC_HASH, 'commit': expected_commit})
    collect_plan.live(root, grants, alpaca_env, None, clock=clock, decision_at=cutoff)
    sealed = forward.snapshot(root, clock(), cutoff=cutoff)
    while clock() < cutoff:
        sleep(min(1, (cutoff - clock()).total_seconds()))
    text = forward.prompt(sealed)
    def decide(family):
        try:
            output, call = client.call(family, text, forward.SCHEMA,
                tag='live-forward-intraday:' + day + ':' + sealed['snapshot_hash'], timeout=900)
            schemas.analyst(output['analyst'], sealed['contexts']['ab'])
            schemas.news(output['news'], sealed['contexts']['cd'])
            return family, {'output': output, 'call': call, 'received_at': iso(clock())}
        except ReplayError as error:
            return family, {'error': error.code, 'stored_at': iso(clock())}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='forward') as pool:
        outcomes = dict(pool.map(decide, ex.FAMILIES))
    T = clock()
    entry_after = T + timedelta(seconds=30)
    if entry_after >= sessions[0].close_at:
        raise ReplayError('INTRADAY_ENTRY_AFTER_CLOSE')
    timing = {'mode': 'INTRADAY', 'decision_time': iso(T), 'input_cutoff': iso(cutoff),
              'entry_after': iso(entry_after), 'snapshot_hash': sealed['snapshot_hash'],
              'spec_hash': spec.SPEC_HASH, 'code_hash': spec.code_hash()}
    timing['clock_hash'] = sha256_canonical(timing)
    atomic_json(base / 'decision-clock.json', timing)
    statuses = {}
    for family, outcome in outcomes.items():
        if 'error' in outcome:
            statuses[family] = outcome['error']
            atomic_json(base / (family + '-failure.json'), dict(outcome, snapshot_hash=sealed['snapshot_hash']))
            continue
        value = dict(outcome, schema='live-forward-decision-v3', provenance=forward.PROVENANCE,
                     stored_at=iso(clock()),
                     family=family, model=spec.spec()['models']['identities'][family],
                     configuration_hash=forward.forward_config_hash(family), snapshot_hash=sealed['snapshot_hash'],
                     decision_time=iso(T), input_cutoff=iso(cutoff), entry_after=iso(entry_after),
                     clock_hash=timing['clock_hash'], mode='INTRADAY')
        value['decision_hash'] = sha256_canonical(value)
        atomic_json(base / (family + '.json'), value)
        statuses[family] = 'RECEIVED'
    try:
        receipt = forward.publish(root, day, ledger, sealed['snapshot_hash'])
        publication = {'ledger_commit': receipt['ledger_commit'], 'published_at': receipt['published_at']}
    except ReplayError as error:
        publication = {'error': error.code}
    ok = all(s == 'RECEIVED' for s in statuses.values()) and 'error' not in publication
    result = dict(timing, status='DONE' if ok else 'BLOCKED', provenance=forward.PROVENANCE,
                  session=day, models=statuses, publication=publication)
    atomic_json(base / 'status.json', result)
    return result


def check_record(root, day, family, record, sealed, *, expected_spec=None, expected_code=None):
    timing = clock_record(root, day)
    T, cutoff = instant(timing['decision_time']), instant(timing['input_cutoff'])
    if (day != spec.spec()['night_fixes_amendment']['intraday']['session'] or
            timing['snapshot_hash'] != sealed['snapshot_hash'] or
            timing['spec_hash'] != (expected_spec or spec.SPEC_HASH) or
            timing['code_hash'] != (expected_code or spec.code_hash()) or
            sealed['decision_time'] != iso(cutoff) or sealed.get('input_cutoff') != iso(cutoff) or
            any(record.get(k) != timing[k] for k in ('clock_hash', 'decision_time', 'input_cutoff', 'entry_after')) or
            record.get('family') != family or record.get('model') != spec.spec()['models']['identities'][family] or
            not cutoff <= instant(record['call']['called_at']) <= instant(record['received_at']) <= T <=
                instant(record['stored_at']) < instant(timing['entry_after']) or
            instant(timing['entry_after']) != T + timedelta(seconds=30)):
        raise ReplayError('FORWARD_DECISION_BINDING_MISMATCH')
    commitment.require(root, 'forward', day + '-intraday', {'snapshot': sealed['snapshot_hash'], family: record['decision_hash']},
                       instant(timing['entry_after']), protocol_hash=expected_spec)


def trade_stamp(text):
    # Keep the provider's nanoseconds when checking a strictly-after boundary.
    found = re.search(r'\.(\d+)', text)
    fraction = (found.group(1) if found else '')[:9].ljust(9, '0')
    return instant(text), int(fraction[-3:] or '0')


def first_trades(transport, symbols, after):
    rows = {}
    params = {'symbols': ','.join(sorted(symbols)), 'start': iso(after),
              'end': iso(after + timedelta(seconds=1)), 'feed': 'sip', 'limit': 10000}
    for body, _ in md.paginate(transport, '/v2/stocks/trades', params, 'trades'):
        for symbol, trades in (body.get('trades') or {}).items():
            for trade in trades:
                stamp = trade_stamp(trade['t'])
                if symbol in symbols and (after, 0) < stamp < (after + timedelta(seconds=1), 0):
                    if symbol not in rows or stamp < trade_stamp(rows[symbol]['t']):
                        rows[symbol] = trade
    return rows


from .score import Outcomes as RegularOutcomes
from scripts.trading_lab.replay_bench import labels as lb
from . import quality


class Outcomes(RegularOutcomes):
    def get(self, group, day, asset, h):
        key = (group, day, asset, h)
        if asset in sn.CRYPTO:
            # The grant has crypto bars, not crypto trades. Never manufacture the
            # first actual trade from a minute open for this recovery session.
            self.exclusion_reasons[key] = 'INTRADAY_CRYPTO_TRADE_NOT_GRANTED'
            return None
        if key in self.cache:
            return self.cache[key]
        T = self.decision_time(day)
        following = md.sessions(day, (T + timedelta(days=21)).date().isoformat())
        if len(following) < h or following[h-1].session_date > self.last:
            return None
        try:
            after = T + timedelta(seconds=30)
            if group == 'ab':
                bars = self.universe.bars
            else:
                if asset not in self.news_bars:
                    self.news_bars[asset], _ = md.stock_bars(self.transport, sorted([asset, 'SPY']),
                        parent.DATA_START, self.daily_end)
                bars = self.news_bars[asset]
            rows, spy = md.session_rows(bars.get(asset, [])), md.session_rows(bars.get('SPY', []))
            for values in (bars.get(asset, []), bars.get('SPY', [])):
                for row in values:
                    quality.validate_bar(row)
            entries = first_trades(self.transport, [asset, 'SPY'], after)
            if any(s not in entries for s in (asset, 'SPY')):
                raise ReplayError('INCOMPLETE_LABEL_COVERAGE')
            prices = {s: entries[s]['p'] for s in entries}
            stats = lb.daily_stats(rows, T)
            if stats is None:
                raise ReplayError('INCOMPLETE_LABEL_COVERAGE')
            end, path = lb.marked_path(day, asset, rows, spy, h, 1, 0, entry_prices=prices)
            if group == 'ab' and h == 1:
                exit_at = following[0].close_at - timedelta(minutes=5)
                exits = self.anchors.stock(day, exit_at.strftime('%H:%M:%S'), list(sn.STOCKS + sn.ETFS + ('SPY',)))
                if any(s not in exits for s in (asset, 'SPY')):
                    raise ReplayError('INCOMPLETE_LABEL_COVERAGE')
                path = {day: exits[asset]['p'] / prices[asset] - exits['SPY']['p'] / prices['SPY']}
            cost = 2 * (self.costs['fee_per_side'] + lb.tier(stats['dollar_volume'], self.costs))
            result = {'rel': sum(path.values()), 'gross_path': path, 'cost': cost, 'vol': stats['vol_20d'],
                      'cluster': sn.CLUSTERS.get(asset, 'smallmid'), 'entry_day': day, 'exit_day': end,
                      'entry_at': entries[asset]['t'], 'spy_entry_at': entries['SPY']['t'],
                      'available_at': iso(following[h-1].close_at),
                      'corporate_action': self.corporate_action([asset, 'SPY'], bars, day, end)}
            self.cache[key] = result
            return result
        except ReplayError as error:
            self.exclusion_reasons[key] = quality.require_data_error(error)
            self.cache[key] = None
            return None
