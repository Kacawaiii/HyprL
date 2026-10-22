"""Premarket plans, independent judgments, later regular-session simulated fills."""
from datetime import timedelta
from hashlib import sha256
import math

from scripts.trading_lab.agent_v2.core import candidates, plan_id
from scripts.trading_lab.agent_v2.inputs import (event_block, fresh, holding_days, session, validate_quotes,
                                                validate_snapshot)
from scripts.trading_lab.agent_v2.risk import equity, size
from .contracts import (ARMS, Blocked, check_facts, check_judgment, digest, factors, instant, iso, utc)
from .producer import calendar, evaluation_end, group_for, registry
from .scoring import disable_underperformers
from .spec import spec


def book(journal):
    state = {'cash': spec()['risk']['initial_equity'], 'initial_equity': spec()['risk']['initial_equity'],
             'positions': {}, 'pending': {}, 'marks': []}
    for row in journal.records():
        body = row['body']
        if row['kind'] == 'decision':
            for plan in body['plans']:
                state['pending'][plan['plan_id']] = plan
        elif row['kind'] in ('evaluation_expired', 'evaluation_rejected'):
            state['pending'].pop(body['plan_id'], None)
        elif row['kind'] == 'evaluation_open':
            state['pending'].pop(body['plan_id'], None)
            state['positions'][body['symbol']] = body
            state['cash'] -= body['qty'] * (body['entry'] + body['unit_entry_fee'])
        elif row['kind'] == 'evaluation_close':
            position = state['positions'].pop(body['symbol'])
            state['cash'] += position['qty'] * (body['exit'] - body['unit_exit_fee'])
        elif row['kind'] == 'evaluation_mark':
            state['marks'].append({'at': row['at'], 'equity': body['equity']})
    return state


def code_plans(bundle, state, at):
    snapshot = bundle['snapshot']
    instruments = validate_snapshot(snapshot, at, snapshot['mode'])
    reg = registry(bundle['registry'], at)
    if {r['symbol'] for r in reg['instruments']} != set(instruments):
        raise Blocked('REGISTRY_SNAPSHOT_MISMATCH')
    calendar(bundle['calendar'], list(instruments), at)
    if snapshot['events'] != bundle['calendar']['events']:
        raise Blocked('CALENDAR_SNAPSHOT_MISMATCH')
    settings = spec()['risk']
    raw, _ = candidates(snapshot, instruments, [], at, {**settings, 'etf_universe': []})
    # Premarket close-based sizing is explicitly provisional, never a quote or fill.
    planning = {s: {'bid': r['bars'][-1]['c'], 'ask': r['bars'][-1]['c'],
                    'ask_size': 10**9} for s, r in instruments.items()}
    out = []
    for candidate in raw:
        edge = reg.get('edges', {}).get(candidate['playbook'], {})
        if (not isinstance(edge.get('net_expectancy_after_costs'), (int, float))
                or not math.isfinite(edge['net_expectancy_after_costs']) or edge['net_expectancy_after_costs'] <= 0
                or type(edge.get('decisions')) is not int or edge['decisions'] <= 0
                or not isinstance(edge.get('evidence_sha256'), str)
                or len(edge['evidence_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in edge['evidence_sha256'])):
            raise Blocked('MEASURED_EDGE_EVIDENCE_REQUIRED')
        candidate['holding_days'] = 5  # common, frozen five-session evaluation horizon
        horizon_days = (evaluation_end(at).date() - at.date()).days + 1
        if event_block(snapshot, candidate['symbol'], at, horizon_days):
            continue
        plan, reason = size(candidate, state, planning, snapshot, [], settings, at)
        if reason:
            continue
        plan.update({'planned_at': iso(at), 'edge_evidence': edge['evidence_sha256'],
                     'sizing_basis': 'PREVIOUS_COMPLETED_CLOSE_PROVISIONAL'})
        plan['plan_id'] = plan_id(plan)
        out.append(plan)
        state['pending'][plan['plan_id']] = plan
        if len(out) == spec()['max_candidates_per_day']:
            break
    return out


class Service:
    def __init__(self, journal, models, *, clock=utc):
        self.journal, self.models, self.clock = journal, models, clock
        if journal.binding['mode'] == 'SHADOW_FORWARD':
            if getattr(models, 'synthetic', False):
                raise Blocked('SYNTHETIC_RUNNER_FORBIDDEN')
            if not journal.records('activation'):
                raise Blocked('RUNTIME_NOT_ACTIVATED')

    def decide(self, bundle, load_primary):
        at = self.clock()
        day, opened, closed = session(at)
        slot = at.replace(hour=12, minute=45, second=0, microsecond=0)
        if not 0 <= (at - slot).total_seconds() <= 180:
            raise Blocked('OUTSIDE_DECISION_SLOT')
        if bundle['snapshot']['mode'] != self.journal.binding['mode']:
            raise Blocked('MODE_MISMATCH')
        with self.journal.owned():
            if self.journal.binding['mode'] == 'SHADOW_FORWARD' and not any(
                    r['body']['hash'] == digest(bundle) for r in self.journal.records('producer_snapshot')):
                raise Blocked('VERIFIED_PRODUCER_REQUIRED')
            if any(r['body']['day'] == day for r in self.journal.records('cycle_start')):
                raise Blocked('SLOT_ALREADY_ATTEMPTED')
            disabled = disable_underperformers(self.journal, at)
            plans = code_plans(bundle, book(self.journal), at)
            self.journal.append([('cycle_start', {'day': day, 'bundle_hash': digest(bundle),
                                                 'code_plans': plans})], at)
        sources, facts, valid, error = [], {'facts': []}, True, None
        decisions = []
        active = set(('extraction', 'veto', 'sizing')) - set(disabled)
        try:
            registrations = {r['symbol']: r for r in bundle['registry']['instruments']}
            for plan in plans if active else []:
                source = load_primary(registrations[plan['symbol']])
                if (source['symbol'] != plan['symbol'] or source['sha256'] != sha256(source['text'].encode()).hexdigest()
                        or source['id'] != digest([source['symbol'], source['url'], source['raw_sha256']])
                        or group_for(source['url']) != 'sec'
                        or not source['url'].startswith('https://www.sec.gov/Archives/edgar/data/' +
                                                        str(int(registrations[plan['symbol']]['cik'])) + '/')
                        or not fresh(source['published_at'], self.clock(), 120 * 86400)
                        or not instant(source['published_at']) <= instant(source['received_at']) <= self.clock()):
                    raise Blocked('PRIMARY_EVIDENCE_INVALID')
                if self.journal.binding['mode'] == 'SHADOW_FORWARD' and not any(
                        r['body']['url'] == source['url'] and r['body']['sha256'] == source['raw_sha256']
                        and r['body']['received_at'] == source['received_at']
                        for r in self.journal.records('http_receipt')):
                    raise Blocked('PRIMARY_RECEIPT_REQUIRED')
                sources.append(source)
            if plans and active:
                context = {'plans': plans, 'sources': sources, 'radar': bundle['radar'], 'at': iso(self.clock())}
                if 'extraction' in active:
                    facts = check_facts(self.models.call('extraction', context), sources)
                context = {**context, 'facts': facts}
                if active & {'veto', 'sizing'}:
                    analyst = check_judgment(self.models.call('analyst', context), plans, sources)
                    # Same evidence, no analyst response: independent judgment.
                    contradictor = check_judgment(self.models.call('contradictor', context), plans, sources)
                else:
                    analyst = contradictor = {p['plan_id']: {'verdict': 'KEEP', 'state': 'DISABLED'} for p in plans}
                for plan in plans:
                    decisions.append({**plan, 'analyst': analyst[plan['plan_id']],
                                      'contradictor': contradictor[plan['plan_id']],
                                      'factors': factors(plan, facts, analyst[plan['plan_id']],
                                                         contradictor[plan['plan_id']], disabled)})
            elif plans:
                valid = False  # no new efficacy sample when all AI layers have been removed
                decisions = [{**p, 'factors': dict.fromkeys(ARMS, 1)} for p in plans]
        except Blocked as failure:
            valid, error = False, str(failure)
            decisions = [{**p, 'factors': dict.fromkeys(ARMS, 1) | {'combined': 0}} for p in plans]
        approved_at = self.clock()
        if approved_at >= opened or (approved_at - at).total_seconds() > 900:
            raise Blocked('DECISION_EXPIRED')
        decisions = [{**p, 'layer_valid': valid,
                      'scored_layers': [layer for layer in ARMS[1:] if layer not in disabled],
                      'approved_at': iso(approved_at),
                      'expires_at': iso(opened + timedelta(minutes=5)), 'snapshot': bundle['snapshot']}
                     for p in decisions]
        with self.journal.owned():
            self.journal.append([('decision', {'day': day, 'plans': decisions, 'facts': facts,
                                  'primary': sources, 'error': error, 'input_hashes': bundle['input_hashes']}),
                                 ('cycle_complete', {'day': day, 'plans': len(plans), 'layer_valid': valid,
                                                     'broker_orders': 0})], approved_at)
        return {'status': 'SHADOW_DECIDED' if valid else 'BLOCKED', 'day': day,
                'code_plans': len(plans), 'reason': error, 'broker_orders': 0}

    def watch(self, value):
        at = self.clock()
        _, opened, closed = session(at)
        if not opened <= at <= closed:
            raise Blocked('MARKET_CLOSED')
        quotes = validate_quotes(value, at, self.journal.binding['mode'])
        with self.journal.owned():
            if self.journal.binding['mode'] == 'SHADOW_FORWARD' and not any(
                    r['body']['hash'] == digest(value) for r in self.journal.records('producer_quotes')):
                raise Blocked('VERIFIED_QUOTE_PRODUCER_REQUIRED')
            state = book(self.journal)
            if any(symbol not in quotes for symbol in state['positions']):
                raise Blocked('HELD_QUOTE_MISSING')
            events = []
            for symbol, position in list(state['positions'].items()):
                q = quotes[symbol]
                if instant(q['at']) <= instant(position['quote_at']):
                    continue
                reason = 'STOP' if q['bid'] <= position['stop'] else 'TIME' if holding_days(
                    position['filled_at'], at) >= position['holding_days'] else None
                if reason:
                    price = q['bid'] * (1 - spec()['risk']['slippage_per_side'])
                    fee = price * spec()['risk']['fee_per_side']
                    unit_pnl = price - fee - position['entry'] - position['unit_entry_fee']
                    body = {'plan_id': position['plan_id'], 'symbol': symbol, 'exit': price,
                            'unit_exit_fee': fee, 'unit_pnl': unit_pnl, 'closed_at': iso(at),
                            'quote_at': q['at'], 'quote_hash': digest(q), 'reason': reason}
                    events.append(('evaluation_close', body))
                    state['cash'] += position['qty'] * (price - fee)
                    del state['positions'][symbol]
            for identity, plan in list(state['pending'].items()):
                q = quotes.get(plan['symbol'])
                if at >= instant(plan['expires_at']):
                    events.append(('evaluation_expired', {'plan_id': identity}))
                    del state['pending'][identity]
                    continue
                # Strictly later than approval, and at/after the actual session open.
                if q is None or instant(q['at']) <= instant(plan['approved_at']) or instant(q['at']) < opened:
                    continue
                del state['pending'][identity]
                horizon_days = (evaluation_end(at).date() - at.date()).days + 1
                if event_block(plan['snapshot'], plan['symbol'], at, horizon_days):
                    events.append(('evaluation_rejected', {'plan_id': identity, 'reason': 'EVENT_RISK'}))
                    continue
                sized, reason = size(plan, state, quotes, plan['snapshot'], [], spec()['risk'], at)
                if reason:
                    events.append(('evaluation_rejected', {'plan_id': identity, 'reason': reason}))
                    continue
                qty = min(plan['qty'], sized['qty'])
                price = sized['entry']
                body = {**plan, 'qty': qty, 'entry': price, 'unit_entry_fee': price * spec()['risk']['fee_per_side'],
                        'entry_fee': qty * price * spec()['risk']['fee_per_side'],
                        'initial_risk': sized['initial_risk'] * qty / sized['qty'],
                        'filled_at': iso(at), 'quote_at': q['at'], 'quote_hash': digest(q)}
                body.pop('snapshot')
                events.append(('evaluation_open', body))
                state['cash'] -= qty * (price + body['unit_entry_fee'])
                state['positions'][plan['symbol']] = body
            nav = equity(state, quotes)
            events.append(('evaluation_mark', {'equity': nav, 'quote_hash': digest(value)}))
            self.journal.append(events, at)
            disable_underperformers(self.journal, at)
        return {'status': 'SHADOW_QUOTES', 'events': len(events), 'positions': len(state['positions']),
                'broker_orders': 0}
